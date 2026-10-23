# coding=utf-8
"""推理输入构造与语音克隆批处理大小推断。

本文件是 ``MossTTSNanoForCausalLM`` 的 mixin：方法通过 ``self`` 访问模型其他部分，
类本身不单独实例化。
"""
from __future__ import annotations

import logging
from typing import Optional, Union

import torch

from .prompting import (
    build_assistant_prompt_prefix,
    build_prompt_token_ids,
    build_user_prompt_after_reference,
    build_user_prompt_prefix,
)



class MossTTSNanoInferenceInputsMixin:
    # ------------------------------------------------------------------
    # 输入构造：把文本 token / 音频 token 排成 [seq, n_vq+1] 的行
    # ------------------------------------------------------------------

    def _build_text_rows(
        self,
        token_ids: list[int],
        device: torch.device,
    ) -> torch.LongTensor:
        """构造纯文本行：第 0 通道为文本 token，其余音频通道全部填 pad。"""
        rows = torch.full(
            (len(token_ids), self.config.n_vq + 1),
            self.config.audio_pad_token_id,
            dtype=torch.long,
            device=device,
        )
        if token_ids:
            rows[:, 0] = torch.tensor(token_ids, dtype=torch.long, device=device)
        return rows


    # ------------------------------------------------------------------
    # 推理参数解析
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_inference_mode(
        mode: str,
        has_prompt_text: bool,
        has_prompt_audio: bool,
    ) -> str:
        """校验并规范化推理模式。

        - ``continuation``（续写）：只给目标文本，或 prompt_text + prompt_audio 同时给；
        - ``voice_clone``（语音克隆）：只给 prompt_audio，不接受 prompt_text。
        """
        normalized_mode = str(mode or "continuation").strip().lower() or "continuation"
        if normalized_mode not in {"continuation", "voice_clone"}:
            raise ValueError(f"Unsupported inference mode {mode!r}.")
        if normalized_mode == "voice_clone":
            if not has_prompt_audio:
                raise ValueError("voice_clone mode requires prompt_audio_path.")
            if has_prompt_text:
                raise ValueError("voice_clone mode does not accept prompt_text.")
        elif has_prompt_text != has_prompt_audio:
            raise ValueError(
                "continuation mode accepts either target text only, or prompt_text and prompt_audio_path together."
            )
        return normalized_mode

    def _resolve_inference_nq(self, nq: Optional[int] = None) -> int:
        """解析实际使用的码本通道数，默认用全部 n_vq 通道。"""
        if nq is None:
            return int(self.config.n_vq)
        resolved_nq = int(nq)
        if resolved_nq < 1 or resolved_nq > int(self.config.n_vq):
            raise ValueError(f"nq must be in [1, {self.config.n_vq}], got {resolved_nq}.")
        return resolved_nq

    def _mask_unused_audio_channels(
        self,
        audio_token_ids: torch.LongTensor,
        nq: int,
    ) -> torch.LongTensor:
        """把超出前 nq 个通道的音频 token 置为 pad（推理只使用前 nq 层码本）。"""
        tensor = torch.as_tensor(audio_token_ids, dtype=torch.long)
        if tensor.shape[-1] != self.config.n_vq:
            raise ValueError(
                f"Expected audio token ids with trailing dim {self.config.n_vq}, got {tuple(tensor.shape)}"
            )
        if nq < self.config.n_vq:
            tensor = tensor.clone()
            tensor[..., nq:] = self.config.audio_pad_token_id
        return tensor

    def _build_audio_prefix_rows(
        self,
        prompt_audio_codes: torch.LongTensor,
        slot_token_id: int,
        device: torch.device,
    ) -> torch.LongTensor:
        """构造音频行：第 0 通道为槽位 token，其余通道为参考音频的码本 token。"""
        rows = torch.full(
            (int(prompt_audio_codes.shape[0]), self.config.n_vq + 1),
            self.config.audio_pad_token_id,
            dtype=torch.long,
            device=device,
        )
        if rows.shape[0] > 0:
            rows[:, 0] = int(slot_token_id)
            rows[:, 1:] = prompt_audio_codes
        return rows

    def build_inference_input_ids(
        self,
        text: str,
        text_tokenizer,
        mode: str = "continuation",
        prompt_text: Optional[str] = None,
        prompt_audio_codes: Optional[torch.LongTensor] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> tuple[torch.LongTensor, torch.BoolTensor]:
        """构造推理输入：把 prompt 文本与参考音频 token 排成 [1, seq, n_vq+1]。

        voice_clone 布局：
            [文本行: user 前缀 + audio_start]
            [音频行: 用户槽位 token + 参考音频 codes]
            [文本行: audio_end + 固定字段 + 目标文本 + assistant 前缀 + audio_start]
        continuation 布局：
            [文本行: 完整 prompt（含参考文本转写 + 目标文本）]
            [文本行: audio_start]
            [音频行: 助手槽位 token + 参考音频 codes]（可选）
        """
        resolved_device = self._resolve_device(device)
        resolved_mode = self._resolve_inference_mode(
            mode=mode,
            has_prompt_text=prompt_text is not None,
            has_prompt_audio=prompt_audio_codes is not None,
        )

        if resolved_mode == "voice_clone":
            assert prompt_audio_codes is not None
            text_token_ids = self._encode_text(text_tokenizer, text)
            prompt_token_ids = build_user_prompt_prefix(text_tokenizer, self.config) + [self.config.audio_start_token_id]
            suffix_token_ids = (
                [self.config.audio_end_token_id]
                + build_user_prompt_after_reference(text_tokenizer)
                + text_token_ids
                + build_assistant_prompt_prefix(text_tokenizer, self.config)
                + [self.config.audio_start_token_id]
            )
            sections = [
                self._build_text_rows(prompt_token_ids, device=resolved_device),
                self._build_audio_prefix_rows(
                    prompt_audio_codes=prompt_audio_codes.to(resolved_device),
                    slot_token_id=self.config.audio_user_slot_token_id,
                    device=resolved_device,
                ),
                self._build_text_rows(suffix_token_ids, device=resolved_device),
            ]
            input_ids = torch.cat(sections, dim=0).unsqueeze(0)
            attention_mask = torch.ones(input_ids.shape[:2], dtype=torch.bool, device=resolved_device)
            return input_ids, attention_mask

        # continuation：参考文本转写与目标文本拼接后一起进 prompt
        effective_text = text if prompt_text is None else prompt_text + text
        prompt_token_ids = build_prompt_token_ids(
            tokenizer=text_tokenizer,
            config=self.config,
            text_token_ids=self._encode_text(text_tokenizer, effective_text),
        )
        sections = [
            self._build_text_rows(prompt_token_ids, device=resolved_device),
            self._build_text_rows([self.config.audio_start_token_id], device=resolved_device),
        ]
        if prompt_audio_codes is not None:
            sections.append(
                self._build_audio_prefix_rows(
                    prompt_audio_codes=prompt_audio_codes.to(resolved_device),
                    slot_token_id=self.config.audio_assistant_slot_token_id,
                    device=resolved_device,
                )
            )
        input_ids = torch.cat(sections, dim=0).unsqueeze(0)
        attention_mask = torch.ones(input_ids.shape[:2], dtype=torch.bool, device=resolved_device)
        return input_ids, attention_mask

    def _left_pad_inference_batch(
        self,
        input_id_batches: list[torch.LongTensor],
        attention_mask_batches: list[torch.BoolTensor],
        device: torch.device,
    ) -> tuple[torch.LongTensor, torch.BoolTensor]:
        """把多条长度不一的输入左 padding 对齐，拼成一个 batch。

        左 padding 保证所有样本的生成起点（序列末尾）对齐。
        """
        if not input_id_batches:
            raise ValueError("input_id_batches must not be empty.")
        if len(input_id_batches) != len(attention_mask_batches):
            raise ValueError("input_id_batches and attention_mask_batches must have the same length.")

        batch_size = len(input_id_batches)
        max_seq_len = max(int(batch.shape[1]) for batch in input_id_batches)
        row_width = self.config.n_vq + 1

        # padding 位置的文本通道填文本 pad，音频通道填音频 pad
        padded_input_ids = torch.full(
            (batch_size, max_seq_len, row_width),
            self.config.audio_pad_token_id,
            dtype=torch.long,
            device=device,
        )
        padded_input_ids[:, :, 0] = self.config.pad_token_id
        padded_attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool, device=device)

        for batch_index, (input_ids, attention_mask) in enumerate(zip(input_id_batches, attention_mask_batches)):
            normalized_input_ids = input_ids.squeeze(0).to(device=device, dtype=torch.long)
            normalized_attention_mask = attention_mask.squeeze(0).to(device=device, dtype=torch.bool)
            seq_len = int(normalized_input_ids.shape[0])
            padded_input_ids[batch_index, -seq_len:, :] = normalized_input_ids
            padded_attention_mask[batch_index, -seq_len:] = normalized_attention_mask

        return padded_input_ids, padded_attention_mask

    def _trim_generated_audio_token_ids(
        self,
        audio_token_ids: torch.LongTensor,
        effective_nq: int,
    ) -> torch.LongTensor:
        """裁掉生成结果末尾的无效帧（整行都是 pad 的帧）。"""
        tensor = self._mask_unused_audio_channels(audio_token_ids, nq=effective_nq)
        if tensor.ndim != 2:
            raise ValueError(f"Expected a 2D audio token tensor, got {tuple(tensor.shape)}")
        valid_rows = tensor[:, :effective_nq].ne(self.config.audio_pad_token_id).any(dim=-1)
        if not bool(valid_rows.any()):
            return tensor[:0]
        last_valid_index = int(torch.nonzero(valid_rows, as_tuple=False)[-1].item()) + 1
        return tensor[:last_valid_index]

    # ------------------------------------------------------------------
    # 语音克隆批处理大小推断（按显存自动估算）
    # ------------------------------------------------------------------

    def _resolve_voice_clone_chunk_batch_size(
        self,
        *,
        resolved_device: torch.device,
        chunk_count: int,
        max_memory_per_sample_gb: float,
    ) -> int:
        """按剩余显存估算一次能并行生成多少个文本分片（仅 CUDA 生效）。"""
        if chunk_count <= 1 or max_memory_per_sample_gb <= 0 or resolved_device.type != "cuda":
            return 1
        if not hasattr(torch.cuda, "mem_get_info"):
            return 1
        try:
            free_bytes, _ = torch.cuda.mem_get_info(resolved_device)
        except Exception:
            return 1
        bytes_per_sample = int(float(max_memory_per_sample_gb) * (1024**3))
        if bytes_per_sample <= 0:
            return 1
        # 留 10% 显存余量，避免 OOM
        usable_free_bytes = max(0, int(free_bytes * 0.9))
        batch_size = max(1, usable_free_bytes // bytes_per_sample)
        resolved_batch_size = max(1, min(int(chunk_count), int(batch_size)))
        logging.info(
            "voice_clone chunk batching device=%s free_gb=%.2f max_memory_per_sample_gb=%.2f resolved_batch_size=%d chunk_count=%d",
            resolved_device,
            float(free_bytes) / float(1024**3),
            float(max_memory_per_sample_gb),
            resolved_batch_size,
            int(chunk_count),
        )
        return resolved_batch_size

    @staticmethod
    def _resolve_requested_batch_size_limit(requested_batch_size: Optional[int]) -> Optional[int]:
        """规范化用户指定的批大小上限：None/<=0 表示不限制。"""
        if requested_batch_size is None:
            return None
        resolved_batch_size = int(requested_batch_size)
        if resolved_batch_size <= 0:
            return None
        return max(1, resolved_batch_size)

    def _resolve_effective_voice_clone_batch_sizes(
        self,
        *,
        resolved_device: torch.device,
        chunk_count: int,
        max_memory_per_sample_gb: float,
        requested_tts_max_batch_size: Optional[int] = None,
        requested_codec_max_batch_size: Optional[int] = None,
        realtime_streaming: bool = False,
    ) -> tuple[int, int]:
        """综合显存估算与用户上限，得出 (TTS 生成批大小, codec 解码批大小)。"""
        effective_tts_batch_size = self._resolve_voice_clone_chunk_batch_size(
            resolved_device=resolved_device,
            chunk_count=chunk_count,
            max_memory_per_sample_gb=max_memory_per_sample_gb,
        )
        requested_tts_limit = self._resolve_requested_batch_size_limit(requested_tts_max_batch_size)
        requested_codec_limit = self._resolve_requested_batch_size_limit(requested_codec_max_batch_size)

        if requested_tts_limit is not None:
            effective_tts_batch_size = min(effective_tts_batch_size, requested_tts_limit)
        if realtime_streaming and requested_codec_limit is not None:
            effective_tts_batch_size = min(effective_tts_batch_size, requested_codec_limit)

        effective_tts_batch_size = max(1, min(int(chunk_count), int(effective_tts_batch_size)))

        # 流式场景下解码与生成同批推进；非流式默认逐条解码
        if realtime_streaming:
            effective_codec_batch_size = effective_tts_batch_size
        elif requested_codec_limit is None:
            effective_codec_batch_size = 1
        else:
            effective_codec_batch_size = max(1, min(int(requested_codec_limit), int(effective_tts_batch_size)))

        return int(effective_tts_batch_size), int(effective_codec_batch_size)
