# coding=utf-8
"""端到端推理入口：inference / inference_stream。

本文件是 ``MossTTSNanoForCausalLM`` 的 mixin：方法通过 ``self`` 访问模型其他部分，
类本身不单独实例化。
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Iterator, Optional, Union

import torch
import torchaudio

from .constants import (
    DEFAULT_VOICE_CLONE_MAX_TEXT_TOKENS,
    DEFAULT_VOICE_CLONE_MAX_MEMORY_PER_SAMPLE_GB,
)



class MossTTSNanoInferenceMixin:
    # ------------------------------------------------------------------
    # 端到端推理入口
    # ------------------------------------------------------------------

    @torch.no_grad()
    def inference_stream(
        self,
        text: str,
        output_audio_path: Union[str, Path],
        mode: str = "continuation",
        prompt_text: Optional[str] = None,
        prompt_audio_path: Optional[Union[str, Path]] = None,
        reference_audio_path: Optional[Union[str, Path]] = None,
        text_tokenizer=None,
        text_tokenizer_path: Optional[str] = None,
        audio_tokenizer=None,
        audio_tokenizer_type: Optional[str] = None,
        audio_tokenizer_pretrained_name_or_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        nq: Optional[int] = None,
        max_new_frames: int = 300,
        do_sample: bool = False,
        text_temperature: float = 1.5,
        text_top_p: float = 1.0,
        text_top_k: int = 50,
        audio_temperature: float = 1.7,
        audio_top_p: float = 0.8,
        audio_top_k: int = 25,
        audio_repetition_penalty: float = 1.0,
        use_kv_cache: bool = True,
        voice_clone_max_text_tokens: int = DEFAULT_VOICE_CLONE_MAX_TEXT_TOKENS,
        voice_clone_max_memory_per_sample_gb: float = DEFAULT_VOICE_CLONE_MAX_MEMORY_PER_SAMPLE_GB,
        tts_max_batch_size: int = 0,
        codec_max_batch_size: int = 0,
    ) -> Iterator[dict[str, Any]]:
        """流式端到端推理：文本 -> 音频 token -> 边生成边解码出波形片段。

        产出两类事件：
        - ``{"type": "audio", ...}``：一段解码好的波形（含分片间静音停顿）；
        - ``{"type": "result", ...}``：最终结果汇总，并把完整音频写入 output_audio_path。
        """
        resolved_device = self._resolve_device(device)
        effective_nq = self._resolve_inference_nq(nq)
        if next(self.parameters()).device != resolved_device:
            self.to(resolved_device)

        was_training = self.training
        self.eval()

        text_tokenizer = self._load_text_tokenizer(
            text_tokenizer=text_tokenizer,
            text_tokenizer_path=text_tokenizer_path,
        )
        audio_tokenizer = self._load_audio_tokenizer(
            audio_tokenizer=audio_tokenizer,
            audio_tokenizer_type=audio_tokenizer_type,
            audio_tokenizer_pretrained_name_or_path=audio_tokenizer_pretrained_name_or_path,
            device=resolved_device,
        )

        target_sample_rate = self._resolve_audio_tokenizer_sample_rate(audio_tokenizer)
        target_channels = self._resolve_audio_tokenizer_channels(audio_tokenizer)
        effective_prompt_audio_path = prompt_audio_path or reference_audio_path
        resolved_mode = self._resolve_inference_mode(
            mode=mode,
            has_prompt_text=prompt_text is not None,
            has_prompt_audio=effective_prompt_audio_path is not None,
        )
        if reference_audio_path is not None and prompt_audio_path is None:
            logging.warning(
                "reference_audio_path=%s is treated as prompt_audio_path for backward compatibility.",
                reference_audio_path,
            )

        # 参考音频：加载 -> 编码为码本 token -> 屏蔽未使用通道
        prompt_audio_codes = None
        if effective_prompt_audio_path is not None:
            waveform, sample_rate = self._load_reference_audio(
                effective_prompt_audio_path,
                target_sample_rate,
                target_channels,
            )
            encoded = self._call_audio_encode(
                audio_tokenizer=audio_tokenizer,
                waveform=waveform.to(resolved_device),
                sample_rate=sample_rate,
            )
            prompt_audio_codes = self._mask_unused_audio_channels(
                self._normalize_audio_codes(encoded),
                nq=effective_nq,
            ).to(resolved_device)

        # 语音克隆 + 长文本：先按句子分片；只有单片时保留原文本
        if resolved_mode == "voice_clone":
            split_voice_clone_text_chunks = self._split_text_into_best_sentences(
                text_tokenizer=text_tokenizer,
                text=text,
                max_tokens=voice_clone_max_text_tokens,
            )
            voice_clone_text_chunks = split_voice_clone_text_chunks if len(split_voice_clone_text_chunks) > 1 else [text]
        else:
            voice_clone_text_chunks = [text]

        if resolved_mode == "voice_clone" and len(voice_clone_text_chunks) > 1:
            voice_clone_chunk_batch_size, voice_clone_codec_batch_size = self._resolve_effective_voice_clone_batch_sizes(
                resolved_device=resolved_device,
                chunk_count=len(voice_clone_text_chunks),
                max_memory_per_sample_gb=float(voice_clone_max_memory_per_sample_gb),
                requested_tts_max_batch_size=tts_max_batch_size,
                requested_codec_max_batch_size=codec_max_batch_size,
                realtime_streaming=True,
            )
        else:
            voice_clone_chunk_batch_size = 1
            voice_clone_codec_batch_size = 1

        generated_audio_token_chunks: list[torch.LongTensor] = []
        emitted_waveform_segments: list[torch.FloatTensor] = []
        decoded_sample_rate: Optional[int] = None
        emitted_samples_total = 0
        first_audio_emitted_at: Optional[float] = None
        streaming_reset_fn = getattr(audio_tokenizer, "_reset_batch_decode_streaming_state", None)

        try:
            for batch_start in range(0, len(voice_clone_text_chunks), voice_clone_chunk_batch_size):
                batch_chunks = voice_clone_text_chunks[batch_start : batch_start + voice_clone_chunk_batch_size]
                batch_prompt_input_ids: list[torch.LongTensor] = []
                batch_attention_masks: list[torch.BoolTensor] = []
                for text_chunk in batch_chunks:
                    prompt_input_ids, attention_mask = self.build_inference_input_ids(
                        text=text_chunk,
                        text_tokenizer=text_tokenizer,
                        mode=resolved_mode,
                        prompt_text=prompt_text,
                        prompt_audio_codes=prompt_audio_codes,
                        device=resolved_device,
                    )
                    batch_prompt_input_ids.append(prompt_input_ids)
                    batch_attention_masks.append(attention_mask)

                batched_prompt_input_ids, batched_attention_mask = self._left_pad_inference_batch(
                    input_id_batches=batch_prompt_input_ids,
                    attention_mask_batches=batch_attention_masks,
                    device=resolved_device,
                )

                # 每条样本的流式状态：待解码帧队列、已解码波形队列、生成/停顿标记
                row_states = [
                    {
                        "pending_decode_frames": [],
                        "decoded_audio_segments": [],
                        "generation_complete": False,
                        "pause_emitted": False,
                    }
                    for _ in batch_chunks
                ]
                batch_emit_index = 0
                codec_stream_started = False

                # continuation 模式下先把参考音频送入流式解码器预热，
                # 让 codec 的流式状态与 prompt 对齐
                if resolved_mode == "continuation" and prompt_audio_codes is not None:
                    prompt_decode_codes = self._prepare_audio_codes_for_decode(prompt_audio_codes, nq=effective_nq)
                    _ = audio_tokenizer.batch_decode(
                        [prompt_decode_codes],
                        num_quantizers=effective_nq,
                        streaming=True,
                        max_batch_size=1,
                        reset_stream=True,
                    )
                    codec_stream_started = True

                def _emit_ready_segments() -> Iterator[dict[str, Any]]:
                    """把已解码、且排在队首的波形片段按顺序产出（含分片间静音）。"""
                    nonlocal batch_emit_index, emitted_samples_total, first_audio_emitted_at
                    active_sample_rate = decoded_sample_rate or target_sample_rate
                    while batch_emit_index < len(batch_chunks):
                        state = row_states[batch_emit_index]
                        decoded_segments = state["decoded_audio_segments"]
                        if decoded_segments:
                            next_segment = decoded_segments.pop(0)
                            if next_segment.numel() == 0 or int(next_segment.shape[-1]) <= 0:
                                continue
                            emitted_waveform_segments.append(next_segment)
                            if first_audio_emitted_at is None:
                                first_audio_emitted_at = time.monotonic()
                            emitted_samples_total += int(next_segment.shape[-1])
                            yield {
                                "type": "audio",
                                "waveform": next_segment,
                                "sample_rate": active_sample_rate,
                                "chunk_index": batch_start + batch_emit_index,
                                "is_pause": False,
                                "emitted_audio_seconds": float(emitted_samples_total) / float(active_sample_rate),
                                "lead_seconds": self._compute_stream_lead_seconds(
                                    emitted_samples_total=emitted_samples_total,
                                    sample_rate=active_sample_rate,
                                    first_audio_emitted_at=first_audio_emitted_at,
                                ),
                            }
                            continue

                        # 队首样本还没生成完：必须等它，保证输出顺序
                        if not state["generation_complete"]:
                            break

                        # 当前样本已生成完：在 voice_clone 多分片场景下，先补一段静音停顿
                        if (
                            resolved_mode == "voice_clone"
                            and len(voice_clone_text_chunks) > 1
                            and not state["pause_emitted"]
                            and (batch_start + batch_emit_index) < len(voice_clone_text_chunks) - 1
                        ):
                            state["pause_emitted"] = True
                            pause_seconds = self._estimate_voice_clone_inter_chunk_pause_seconds(
                                voice_clone_text_chunks[batch_start + batch_emit_index]
                            )
                            pause_samples = max(0, int(round(float(active_sample_rate) * pause_seconds)))
                            if pause_samples > 0:
                                silence = torch.zeros((target_channels, pause_samples), dtype=torch.float32)
                                emitted_waveform_segments.append(silence)
                                if first_audio_emitted_at is None:
                                    first_audio_emitted_at = time.monotonic()
                                emitted_samples_total += int(silence.shape[-1])
                                yield {
                                    "type": "audio",
                                    "waveform": silence,
                                    "sample_rate": active_sample_rate,
                                    "chunk_index": batch_start + batch_emit_index,
                                    "is_pause": True,
                                    "emitted_audio_seconds": float(emitted_samples_total) / float(active_sample_rate),
                                    "lead_seconds": self._compute_stream_lead_seconds(
                                        emitted_samples_total=emitted_samples_total,
                                        sample_rate=active_sample_rate,
                                        first_audio_emitted_at=first_audio_emitted_at,
                                    ),
                                }
                            batch_emit_index += 1
                            continue

                        batch_emit_index += 1

                def _maybe_decode_pending(force: bool) -> Iterator[dict[str, Any]]:
                    """把积压的生成帧送去流式解码。

                    非强制模式下根据领先优势决定要不要解码、解多少帧，
                    避免解码太快（浪费算力）或太慢（播放断流）。
                    """
                    nonlocal codec_stream_started, decoded_sample_rate
                    pending_counts = [len(state["pending_decode_frames"]) for state in row_states]
                    total_pending = sum(pending_counts)
                    if total_pending <= 0:
                        return

                    active_sample_rate = decoded_sample_rate or target_sample_rate
                    head_pending = pending_counts[batch_emit_index] if batch_emit_index < len(batch_chunks) else 0
                    lead_seconds = self._compute_stream_lead_seconds(
                        emitted_samples_total=emitted_samples_total,
                        sample_rate=active_sample_rate,
                        first_audio_emitted_at=first_audio_emitted_at,
                    )
                    decode_budget = self._resolve_stream_decode_frame_budget(
                        emitted_samples_total=emitted_samples_total,
                        sample_rate=active_sample_rate,
                        first_audio_emitted_at=first_audio_emitted_at,
                    )

                    if not force:
                        should_decode = False
                        if first_audio_emitted_at is None and head_pending > 0:
                            # 首包音频优先，尽快产出
                            should_decode = True
                        elif head_pending > 0 and lead_seconds < 0.45:
                            # 领先优势太小：队首有待解码帧就立刻解
                            should_decode = True
                        elif max(pending_counts) >= decode_budget:
                            # 积压帧数达到预算：凑够一批再解
                            should_decode = True
                        elif lead_seconds < 0.0 and total_pending > 0:
                            # 已经落后于播放进度：有什么解什么
                            should_decode = True
                        if not should_decode:
                            return

                    decode_window = max(pending_counts) if force else max(1, decode_budget)
                    empty_codes = torch.empty((effective_nq, 0), dtype=torch.long, device=resolved_device)
                    codes_list: list[torch.Tensor] = []
                    for state in row_states:
                        take_count = min(len(state["pending_decode_frames"]), decode_window)
                        if take_count <= 0:
                            codes_list.append(empty_codes)
                            continue
                        frame_rows = state["pending_decode_frames"][:take_count]
                        del state["pending_decode_frames"][:take_count]
                        frame_tensor = torch.cat(frame_rows, dim=0).to(device=resolved_device, dtype=torch.long)
                        codes_list.append(frame_tensor[:, :effective_nq].transpose(0, 1).contiguous())

                    decode_output = audio_tokenizer.batch_decode(
                        codes_list,
                        num_quantizers=effective_nq,
                        streaming=True,
                        max_batch_size=(voice_clone_codec_batch_size if not codec_stream_started else None),
                        reset_stream=not codec_stream_started,
                    )
                    codec_stream_started = True
                    waveform_rows, current_sample_rate = self._extract_batch_waveforms_and_sample_rate(
                        decode_output,
                        fallback_sample_rate=target_sample_rate,
                        batch_size=len(batch_chunks),
                    )
                    if decoded_sample_rate is None:
                        decoded_sample_rate = current_sample_rate
                    elif decoded_sample_rate != current_sample_rate:
                        raise ValueError(
                            f"Decoded sample rates differ across streaming decode calls: {decoded_sample_rate} vs {current_sample_rate}"
                        )

                    for row_index, waveform_row in enumerate(waveform_rows):
                        if waveform_row.numel() == 0 or int(waveform_row.shape[-1]) <= 0:
                            continue
                        row_states[row_index]["decoded_audio_segments"].append(waveform_row)

                    yield from _emit_ready_segments()

                try:
                    final_generation = None
                    for event in self.generate_stream(
                        input_ids=batched_prompt_input_ids,
                        attention_mask=batched_attention_mask,
                        nq=effective_nq,
                        max_new_frames=max_new_frames,
                        do_sample=do_sample,
                        text_temperature=text_temperature,
                        text_top_p=text_top_p,
                        text_top_k=text_top_k,
                        audio_temperature=audio_temperature,
                        audio_top_p=audio_top_p,
                        audio_top_k=audio_top_k,
                        audio_repetition_penalty=audio_repetition_penalty,
                        use_kv_cache=use_kv_cache,
                        return_dict_in_generate=True,
                    ):
                        if event["type"] == "frame":
                            frame_audio_token_ids = event["audio_token_ids"]
                            active_mask = event["active_mask"]
                            finished_mask = event["finished_mask"]
                            for row_index in range(len(batch_chunks)):
                                if bool(active_mask[row_index].item()):
                                    row_states[row_index]["pending_decode_frames"].append(
                                        frame_audio_token_ids[row_index : row_index + 1].detach().clone()
                                    )
                                if bool(finished_mask[row_index].item()):
                                    row_states[row_index]["generation_complete"] = True
                            yield from _maybe_decode_pending(force=False)
                            continue
                        final_generation = event.get("generation")

                    if final_generation is None:
                        raise RuntimeError("Streaming generation finished without a final output.")

                    for state in row_states:
                        state["generation_complete"] = True

                    # 收尾：强制解完剩余帧，并把队列里的波形全部产出
                    yield from _maybe_decode_pending(force=True)
                    yield from _emit_ready_segments()

                    batched_audio_token_ids = self._mask_unused_audio_channels(final_generation.audio_token_ids, nq=effective_nq)
                    for sample_index in range(len(batch_chunks)):
                        generated_audio_token_chunks.append(
                            self._trim_generated_audio_token_ids(
                                batched_audio_token_ids[sample_index],
                                effective_nq=effective_nq,
                            )
                        )
                finally:
                    if codec_stream_started and callable(streaming_reset_fn):
                        streaming_reset_fn()

            # 汇总全部音频 token 与波形
            if generated_audio_token_chunks:
                audio_token_ids = torch.cat(generated_audio_token_chunks, dim=0)
            else:
                audio_token_ids = torch.empty((0, self.config.n_vq), dtype=torch.long, device=resolved_device)

            if emitted_waveform_segments:
                waveform = torch.cat(emitted_waveform_segments, dim=-1)
            else:
                waveform = torch.zeros((target_channels, 0), dtype=torch.float32)

            decoded_sample_rate = decoded_sample_rate or target_sample_rate
            output_path = Path(output_audio_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(str(output_path), waveform, decoded_sample_rate)

            yield {
                "type": "result",
                "audio_path": str(output_path),
                "sample_rate": decoded_sample_rate,
                "audio_token_ids": audio_token_ids.detach().cpu(),
                "waveform": waveform,
                "reference_audio_token_ids": None if prompt_audio_codes is None else prompt_audio_codes.detach().cpu(),
                "voice_clone_text_chunks": voice_clone_text_chunks,
                "voice_clone_chunk_batch_size": int(voice_clone_chunk_batch_size),
                "voice_clone_codec_batch_size": int(voice_clone_codec_batch_size),
            }
        finally:
            if was_training:
                self.train()

    @torch.no_grad()
    def inference(
        self,
        text: str,
        output_audio_path: Union[str, Path],
        mode: str = "continuation",
        prompt_text: Optional[str] = None,
        prompt_audio_path: Optional[Union[str, Path]] = None,
        reference_audio_path: Optional[Union[str, Path]] = None,
        text_tokenizer=None,
        text_tokenizer_path: Optional[str] = None,
        audio_tokenizer=None,
        audio_tokenizer_type: Optional[str] = None,
        audio_tokenizer_pretrained_name_or_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        nq: Optional[int] = None,
        max_new_frames: int = 300,
        do_sample: bool = False,
        text_temperature: float = 1.5,
        text_top_p: float = 1.0,
        text_top_k: int = 50,
        audio_temperature: float = 1.7,
        audio_top_p: float = 0.8,
        audio_top_k: int = 25,
        audio_repetition_penalty: float = 1.0,
        use_kv_cache: bool = True,
        voice_clone_max_text_tokens: int = DEFAULT_VOICE_CLONE_MAX_TEXT_TOKENS,
        voice_clone_max_memory_per_sample_gb: float = DEFAULT_VOICE_CLONE_MAX_MEMORY_PER_SAMPLE_GB,
        tts_max_batch_size: int = 0,
        codec_max_batch_size: int = 0,
    ) -> dict[str, Any]:
        """非流式端到端推理：生成全部音频 token 后一次性解码、写盘。

        流程：加载分词器/编解码器 -> 编码参考音频 -> （语音克隆时）文本分片
        -> 分批生成音频 token -> 解码为波形 -> 拼接分片并写入 wav 文件。
        """
        resolved_device = self._resolve_device(device)
        effective_nq = self._resolve_inference_nq(nq)
        if next(self.parameters()).device != resolved_device:
            self.to(resolved_device)

        was_training = self.training
        self.eval()

        text_tokenizer = self._load_text_tokenizer(
            text_tokenizer=text_tokenizer,
            text_tokenizer_path=text_tokenizer_path,
        )
        audio_tokenizer = self._load_audio_tokenizer(
            audio_tokenizer=audio_tokenizer,
            audio_tokenizer_type=audio_tokenizer_type,
            audio_tokenizer_pretrained_name_or_path=audio_tokenizer_pretrained_name_or_path,
            device=resolved_device,
        )

        target_sample_rate = self._resolve_audio_tokenizer_sample_rate(audio_tokenizer)
        target_channels = self._resolve_audio_tokenizer_channels(audio_tokenizer)
        effective_prompt_audio_path = prompt_audio_path or reference_audio_path
        resolved_mode = self._resolve_inference_mode(
            mode=mode,
            has_prompt_text=prompt_text is not None,
            has_prompt_audio=effective_prompt_audio_path is not None,
        )
        if reference_audio_path is not None and prompt_audio_path is None:
            logging.warning(
                "reference_audio_path=%s is treated as prompt_audio_path for backward compatibility.",
                reference_audio_path,
            )

        # 参考音频：加载 -> 编码为码本 token -> 屏蔽未使用通道
        prompt_audio_codes = None
        if effective_prompt_audio_path is not None:
            waveform, sample_rate = self._load_reference_audio(
                effective_prompt_audio_path,
                target_sample_rate,
                target_channels,
            )
            encoded = self._call_audio_encode(
                audio_tokenizer=audio_tokenizer,
                waveform=waveform.to(resolved_device),
                sample_rate=sample_rate,
            )
            prompt_audio_codes = self._mask_unused_audio_channels(
                self._normalize_audio_codes(encoded),
                nq=effective_nq,
            ).to(resolved_device)

        # 语音克隆 + 长文本：先按句子分片；只有单片时保留原文本
        if resolved_mode == "voice_clone":
            split_voice_clone_text_chunks = self._split_text_into_best_sentences(
                text_tokenizer=text_tokenizer,
                text=text,
                max_tokens=voice_clone_max_text_tokens,
            )
            voice_clone_text_chunks = split_voice_clone_text_chunks if len(split_voice_clone_text_chunks) > 1 else [text]
        else:
            voice_clone_text_chunks = [text]

        generated_audio_token_chunks: list[torch.LongTensor] = []
        decoded_waveform_chunks: list[torch.FloatTensor] = []
        decoded_sample_rate: Optional[int] = None

        if resolved_mode == "voice_clone" and len(voice_clone_text_chunks) > 1:
            # 多分片语音克隆：按显存估算 + 用户上限决定每批的分片数
            #   voice_clone_chunk_batch_size -> TTS 自回归生成音频 token 时的批大小
            #   voice_clone_codec_batch_size -> codec 解码成波形时的批大小（更吃显存，故单独限制）
            voice_clone_chunk_batch_size, voice_clone_codec_batch_size = self._resolve_effective_voice_clone_batch_sizes(
                resolved_device=resolved_device,
                chunk_count=len(voice_clone_text_chunks),
                max_memory_per_sample_gb=float(voice_clone_max_memory_per_sample_gb),
                requested_tts_max_batch_size=tts_max_batch_size,
                requested_codec_max_batch_size=codec_max_batch_size,
                realtime_streaming=False,
            )
        else:
            voice_clone_chunk_batch_size = 1
            voice_clone_codec_batch_size = 1

        # 主驱动循环：每次取 voice_clone_chunk_batch_size 个分片组成一批，
        # 走「构输入 -> 生成音频 token -> 裁剪 -> 解码波形」的流水线。
        # 单分片 / continuation 时 batch_size = 1，循环只跑一次，等价于普通推理路径。
        for batch_start in range(0, len(voice_clone_text_chunks), voice_clone_chunk_batch_size):
            # 取出本批要处理的文本分片
            batch_chunks = voice_clone_text_chunks[batch_start : batch_start + voice_clone_chunk_batch_size]
            batch_prompt_input_ids: list[torch.LongTensor] = []
            batch_attention_masks: list[torch.BoolTensor] = []
            # 逐分片构造推理输入：prompt 文本 + 参考音频码 + 待合成文本
            for text_chunk in batch_chunks:
                prompt_input_ids, attention_mask = self.build_inference_input_ids(
                    text=text_chunk,
                    text_tokenizer=text_tokenizer,
                    mode=resolved_mode,
                    prompt_text=prompt_text,
                    prompt_audio_codes=prompt_audio_codes,
                    device=resolved_device,
                )
                batch_prompt_input_ids.append(prompt_input_ids)
                batch_attention_masks.append(attention_mask)

            # 左 padding 成一个 batch（padding 侧不参与 attention）
            batched_prompt_input_ids, batched_attention_mask = self._left_pad_inference_batch(
                input_id_batches=batch_prompt_input_ids,
                attention_mask_batches=batch_attention_masks,
                device=resolved_device,
            )
            # 批量自回归生成音频 token（带数值稳定性兜底）
            batched_audio_token_ids = self._generate_audio_token_ids_with_fallback(
                prompt_input_ids=batched_prompt_input_ids,
                attention_mask=batched_attention_mask,
                effective_nq=effective_nq,
                max_new_frames=max_new_frames,
                do_sample=do_sample,
                text_temperature=text_temperature,
                text_top_p=text_top_p,
                text_top_k=text_top_k,
                audio_temperature=audio_temperature,
                audio_top_p=audio_top_p,
                audio_top_k=audio_top_k,
                audio_repetition_penalty=audio_repetition_penalty,
                use_kv_cache=use_kv_cache,
                resolved_device=resolved_device,
            )

            # 逐样本裁剪掉末尾的无效帧
            batch_audio_token_chunks: list[torch.LongTensor] = []
            for sample_index in range(len(batch_chunks)):
                audio_token_ids = self._trim_generated_audio_token_ids(
                    batched_audio_token_ids[sample_index],
                    effective_nq=effective_nq,
                )
                generated_audio_token_chunks.append(audio_token_ids)
                batch_audio_token_chunks.append(audio_token_ids)

            # 多分片的语音克隆：按 codec 批大小分批解码为波形
            # 解码显存峰值高于文本生成，故用更小的 voice_clone_codec_batch_size 再切一次
            if resolved_mode == "voice_clone" and len(voice_clone_text_chunks) > 1:
                for codec_batch_start in range(0, len(batch_audio_token_chunks), voice_clone_codec_batch_size):
                    codec_audio_token_batches = batch_audio_token_chunks[
                        codec_batch_start : codec_batch_start + voice_clone_codec_batch_size
                    ]
                    decoded_waveforms, current_sample_rate = self._decode_audio_token_id_batch_to_waveforms(
                        audio_tokenizer=audio_tokenizer,
                        audio_token_id_batches=codec_audio_token_batches,
                        target_sample_rate=target_sample_rate,
                        effective_nq=effective_nq,
                        resolved_device=resolved_device,
                    )
                    if decoded_sample_rate is None:
                        decoded_sample_rate = current_sample_rate
                    elif decoded_sample_rate != current_sample_rate:
                        raise ValueError(
                            f"Decoded sample rates differ across voice_clone chunks: {decoded_sample_rate} vs {current_sample_rate}"
                        )
                    decoded_waveform_chunks.extend(decoded_waveforms)

        if generated_audio_token_chunks:
            audio_token_ids = torch.cat(generated_audio_token_chunks, dim=0)
        else:
            audio_token_ids = torch.empty((0, self.config.n_vq), dtype=torch.long, device=resolved_device)

        if resolved_mode == "voice_clone" and len(voice_clone_text_chunks) > 1:
            # 多分片：拼接各分片波形，分片间插入静音停顿
            waveform = (
                self._concat_voice_clone_waveform_chunks(
                    waveform_chunks=decoded_waveform_chunks,
                    text_chunks=voice_clone_text_chunks,
                    sample_rate=decoded_sample_rate,
                )
                if decoded_waveform_chunks
                else torch.zeros((target_channels, 0), dtype=torch.float32)
            )
        else:
            # 单分片 / continuation：一次性解码
            decode_audio_token_ids = audio_token_ids
            prompt_waveform_prefix_samples = 0
            if resolved_mode == "continuation" and prompt_audio_codes is not None:
                # continuation 模式下把参考音频的 token 拼在前面一起解码
                # （保证声学连贯），解码后再把参考音频对应的波形前缀裁掉
                decode_audio_token_ids = torch.cat([prompt_audio_codes, audio_token_ids], dim=0)
                prompt_waveform_prefix_samples = (
                    int(prompt_audio_codes.shape[0]) * self._resolve_audio_tokenizer_downsample_rate(audio_tokenizer)
                )

            waveform, decoded_sample_rate = self._decode_audio_token_ids_to_waveform(
                audio_tokenizer=audio_tokenizer,
                audio_token_ids=decode_audio_token_ids,
                target_sample_rate=target_sample_rate,
                effective_nq=effective_nq,
                resolved_device=resolved_device,
            )
            if prompt_waveform_prefix_samples > 0:
                if decoded_sample_rate != target_sample_rate:
                    prompt_waveform_prefix_samples = int(
                        round(prompt_waveform_prefix_samples * float(decoded_sample_rate) / float(target_sample_rate))
                    )
                prompt_waveform_prefix_samples = min(prompt_waveform_prefix_samples, int(waveform.shape[-1]))
                waveform = waveform[:, prompt_waveform_prefix_samples:]

        assert decoded_sample_rate is not None

        output_path = Path(output_audio_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(output_path), waveform, decoded_sample_rate)

        if was_training:
            self.train()

        return {
            "audio_path": str(output_path),
            "sample_rate": decoded_sample_rate,
            "audio_token_ids": audio_token_ids.detach().cpu(),
            "waveform": waveform,
            "reference_audio_token_ids": None if prompt_audio_codes is None else prompt_audio_codes.detach().cpu(),
            "voice_clone_text_chunks": voice_clone_text_chunks,
            "voice_clone_chunk_batch_size": int(voice_clone_chunk_batch_size),
            "voice_clone_codec_batch_size": int(voice_clone_codec_batch_size),
        }
