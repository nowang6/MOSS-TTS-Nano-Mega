# coding=utf-8
"""采样、自回归生成循环与生成->解码封装。

本文件是 ``MossTTSNanoForCausalLM`` 的 mixin：方法通过 ``self`` 访问模型其他部分，
类本身不单独实例化。
"""
from __future__ import annotations

import logging
import time
from typing import Any, Iterator, Optional, Sequence

import torch



class MossTTSNanoGenerationMixin:
    # ------------------------------------------------------------------
    # 生成 -> 解码 的封装（含数值稳定性兜底）
    # ------------------------------------------------------------------

    def _generate_audio_token_ids_with_fallback(
        self,
        *,
        prompt_input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        effective_nq: int,
        max_new_frames: int,
        do_sample: bool,
        text_temperature: float,
        text_top_p: float,
        text_top_k: int,
        audio_temperature: float,
        audio_top_p: float,
        audio_top_k: int,
        audio_repetition_penalty: float,
        use_kv_cache: bool,
        resolved_device: torch.device,
    ) -> torch.LongTensor:
        """调用 generate 生成音频 token；遇到数值不稳定错误时降级重试一次。

        降级策略：切换到 fp32 + sdpa/eager 注意力（见 _apply_inference_stability_fallback）。
        """
        try:
            generation = self.generate(
                input_ids=prompt_input_ids,
                attention_mask=attention_mask,
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
            )
        except (RuntimeError, ValueError) as exc:
            if not self._is_generation_stability_error(exc):
                raise
            self._apply_inference_stability_fallback(resolved_device)
            generation = self.generate(
                input_ids=prompt_input_ids,
                attention_mask=attention_mask,
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
            )
        return self._mask_unused_audio_channels(generation.audio_token_ids, nq=effective_nq)

    def _decode_audio_token_ids_to_waveform(
        self,
        *,
        audio_tokenizer,
        audio_token_ids: torch.LongTensor,
        target_sample_rate: int,
        effective_nq: int,
        resolved_device: torch.device,
    ) -> tuple[torch.FloatTensor, int]:
        """把音频 token 解码成波形，返回 (波形 [C, T], 采样率)。"""
        decoded = self._call_audio_decode(
            audio_tokenizer=audio_tokenizer,
            audio_token_ids=audio_token_ids.to(resolved_device),
            sample_rate=target_sample_rate,
            nq=effective_nq,
        )
        return self._extract_waveform_and_sample_rate(decoded, fallback_sample_rate=target_sample_rate)

    def _decode_audio_token_id_batch_to_waveforms(
        self,
        *,
        audio_tokenizer,
        audio_token_id_batches: Sequence[torch.LongTensor],
        target_sample_rate: int,
        effective_nq: int,
        resolved_device: torch.device,
    ) -> tuple[list[torch.FloatTensor], int]:
        """批量解码多条音频 token 序列；批量解码失败时逐条重试。"""
        if not audio_token_id_batches:
            return [], target_sample_rate

        if len(audio_token_id_batches) == 1:
            waveform, sample_rate = self._decode_audio_token_ids_to_waveform(
                audio_tokenizer=audio_tokenizer,
                audio_token_ids=audio_token_id_batches[0],
                target_sample_rate=target_sample_rate,
                effective_nq=effective_nq,
                resolved_device=resolved_device,
            )
            return [waveform], sample_rate

        decode_codes = [
            self._prepare_audio_codes_for_decode(audio_token_ids.to(resolved_device), nq=effective_nq)
            for audio_token_ids in audio_token_id_batches
        ]
        batch_decode_fn = getattr(audio_tokenizer, "batch_decode", None)
        if batch_decode_fn is None:
            raise AttributeError("audio_tokenizer must provide a batch_decode method.")

        try:
            with self._audio_tokenizer_inference_context(audio_tokenizer, resolved_device):
                decode_output = batch_decode_fn(
                    decode_codes,
                    num_quantizers=effective_nq,
                    chunk_duration=None,
                )
            return self._extract_batch_waveforms_and_sample_rate(
                decode_output,
                fallback_sample_rate=target_sample_rate,
                batch_size=len(audio_token_id_batches),
            )
        except Exception:
            logging.warning(
                "batched audio decode failed; falling back to per-chunk decode for batch_size=%d",
                len(audio_token_id_batches),
                exc_info=True,
            )
            waveform_rows: list[torch.FloatTensor] = []
            sample_rate = target_sample_rate
            for audio_token_ids in audio_token_id_batches:
                waveform_row, sample_rate = self._decode_audio_token_ids_to_waveform(
                    audio_tokenizer=audio_tokenizer,
                    audio_token_ids=audio_token_ids,
                    target_sample_rate=target_sample_rate,
                    effective_nq=effective_nq,
                    resolved_device=resolved_device,
                )
                waveform_rows.append(waveform_row)
            return waveform_rows, sample_rate

    def _build_generation_row(
        self,
        batch_size: int,
        device: torch.device,
        audio_token_ids: torch.LongTensor,
    ) -> torch.LongTensor:
        """把一个生成帧包装成 [batch, 1, n_vq+1] 的输入行（下一帧的模型输入）。"""
        row = torch.full(
            (batch_size, 1, self.config.n_vq + 1),
            self.config.audio_pad_token_id,
            dtype=torch.long,
            device=device,
        )
        row[:, :, 0] = self.config.audio_assistant_slot_token_id
        row[:, :, 1:] = audio_token_ids.unsqueeze(1)
        return row

    # ------------------------------------------------------------------
    # 流式解码的节拍控制：根据音频领先优势动态调整每次解码的帧数
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_stream_lead_seconds(
        emitted_samples_total: int,
        sample_rate: int,
        first_audio_emitted_at: Optional[float],
    ) -> float:
        """计算「已产出音频时长 - 实际经过时长」，即流式产出的领先秒数。"""
        if first_audio_emitted_at is None or sample_rate <= 0:
            return 0.0
        elapsed_seconds = max(0.0, time.monotonic() - first_audio_emitted_at)
        emitted_seconds = float(emitted_samples_total) / float(sample_rate)
        return emitted_seconds - elapsed_seconds

    @staticmethod
    def _resolve_stream_decode_frame_budget(
        *,
        emitted_samples_total: int,
        sample_rate: int,
        first_audio_emitted_at: Optional[float],
    ) -> int:
        """根据领先优势决定单次流式解码的帧数预算：领先越多，单次解码越多帧。"""
        lead_seconds = MossTTSNanoGenerationMixin._compute_stream_lead_seconds(
            emitted_samples_total=emitted_samples_total,
            sample_rate=sample_rate,
            first_audio_emitted_at=first_audio_emitted_at,
        )
        if first_audio_emitted_at is None or lead_seconds < 0.20:
            return 1
        if lead_seconds < 0.55:
            return 2
        if lead_seconds < 1.10:
            return 4
        return 8

    # ------------------------------------------------------------------
    # 采样：温度 / top-k / top-p / 重复惩罚
    # ------------------------------------------------------------------

    def _sample_next_token(
        self,
        logits: torch.FloatTensor,
        do_sample: bool,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        previous_token_ids: Optional[torch.LongTensor] = None,
        repetition_penalty: float = 1.0,
    ) -> torch.LongTensor:
        """从 logits 采样下一个 token；do_sample=False 时贪心取 argmax。"""
        scores = self._apply_repetition_penalty(
            logits=logits,
            previous_token_ids=previous_token_ids,
            repetition_penalty=repetition_penalty,
        )
        if not do_sample:
            return scores.argmax(dim=-1)
        if temperature <= 0:
            raise ValueError("temperature must be positive when do_sample=True")

        scores = scores / temperature
        if top_k is not None and top_k > 0:
            # top-k：只保留分数最高的 k 个候选
            top_k = min(top_k, scores.shape[-1])
            threshold = torch.topk(scores, top_k, dim=-1).values[..., -1, None]
            scores = scores.masked_fill(scores < threshold, float("-inf"))

        if top_p is not None and 0.0 < top_p < 1.0:
            # top-p（核采样）：按累计概率截断候选集
            sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_scores, dim=-1)
            sorted_cumsum = torch.cumsum(sorted_probs, dim=-1)
            sorted_remove = sorted_cumsum > top_p
            # 整体右移一位：保证累计概率首次超过 top_p 的那个 token 仍被保留
            sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
            sorted_remove[..., 0] = False
            sorted_scores = sorted_scores.masked_fill(sorted_remove, float("-inf"))
            scores = torch.full_like(scores, float("-inf"))
            scores.scatter_(dim=-1, index=sorted_indices, src=sorted_scores)

        probs = torch.softmax(scores, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @staticmethod
    def _ensure_finite_generation_logits(logits: torch.FloatTensor, name: str) -> None:
        """检查 logits 是否含有 NaN/Inf；有则抛出带定位信息的错误。"""
        if torch.isfinite(logits).all():
            return
        finite_mask = torch.isfinite(logits)
        finite_logits = logits[finite_mask]
        min_value = float(finite_logits.min().item()) if finite_logits.numel() > 0 else float("nan")
        max_value = float(finite_logits.max().item()) if finite_logits.numel() > 0 else float("nan")
        raise RuntimeError(
            f"Non-finite {name} during generation: dtype={logits.dtype} "
            f"shape={tuple(logits.shape)} finite={int(finite_mask.sum().item())}/{int(logits.numel())} "
            f"min={min_value} max={max_value}"
        )

    def _apply_repetition_penalty(
        self,
        logits: torch.FloatTensor,
        previous_token_ids: Optional[torch.LongTensor],
        repetition_penalty: float,
    ) -> torch.FloatTensor:
        """重复惩罚：已出现过的 token，正分除以 penalty、负分乘以 penalty。"""
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if repetition_penalty == 1.0 or previous_token_ids is None:
            return logits

        token_ids = torch.as_tensor(previous_token_ids, device=logits.device, dtype=torch.long)
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)
        elif token_ids.ndim > 2:
            token_ids = token_ids.reshape(token_ids.shape[0], -1)

        scores = logits.clone()
        vocab_size = scores.shape[-1]
        for batch_index in range(scores.shape[0]):
            valid_token_ids = token_ids[batch_index]
            valid_token_ids = valid_token_ids[(valid_token_ids >= 0) & (valid_token_ids < vocab_size)]
            if valid_token_ids.numel() == 0:
                continue
            unique_token_ids = torch.unique(valid_token_ids)
            token_scores = scores[batch_index].index_select(0, unique_token_ids)
            token_scores = torch.where(
                token_scores < 0,
                token_scores * repetition_penalty,
                token_scores / repetition_penalty,
            )
            scores[batch_index].scatter_(0, unique_token_ids, token_scores)
        return scores

    def _sample_next_assistant_text_token(
        self,
        logits: torch.FloatTensor,
        do_sample: bool,
        temperature: float,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.LongTensor:
        """预测文本侧控制 token：只在「继续生成音频」与「结束」两个候选中采样。"""
        candidate_ids = torch.tensor(
            [
                self.config.audio_assistant_slot_token_id,
                self.config.audio_end_token_id,
            ],
            dtype=torch.long,
            device=logits.device,
        )
        candidate_logits = logits.index_select(dim=-1, index=candidate_ids)
        sampled_indices = self._sample_next_token(
            logits=candidate_logits,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        return candidate_ids[sampled_indices]


    # ------------------------------------------------------------------
    # 局部 Transformer 前向
    # ------------------------------------------------------------------

    def _decode_local_last_hidden_state(
        self,
        local_inputs_embeds: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """跑一次局部 Transformer，取最后一个位置的隐藏状态。

        局部序列固定为「全局隐藏状态, 已预测的各通道 embedding...」，
        长度 <= n_vq + 1，因此不需要 KV cache。
        """
        local_attention_mask = torch.ones(
            local_inputs_embeds.shape[:2],
            dtype=torch.bool,
            device=local_inputs_embeds.device,
        )
        local_outputs = self.local_transformer(
            input_ids=None,
            attention_mask=local_attention_mask,
            position_ids=None,
            inputs_embeds=local_inputs_embeds,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            cu_seqlens=None,
            num_sequences=None,
        )
        return local_outputs.last_hidden_state[:, -1, :]

    # ------------------------------------------------------------------
    # 自回归生成
    # ------------------------------------------------------------------

    def _iter_generation_events(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
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
        return_dict_in_generate: bool = True,
    ) -> Iterator[dict[str, Any]]:
        """逐帧生成音频 token 的事件流。

        每帧的流程：
        1. 全局 Transformer 前向（带 KV cache），取最后位置隐藏状态；
        2. 局部 Transformer 先预测文本侧控制 token：
           - audio_assistant_slot_token_id：继续生成本帧音频；
           - audio_end_token_id：该样本生成结束；
        3. 局部 Transformer 逐码本通道自回归预测本帧的音频 token；
        4. 产出 ``frame`` 事件，并把本帧拼回输入，进入下一帧。

        全部样本结束或达到 max_new_frames 后，产出 ``final`` 事件。
        """
        if input_ids.ndim == 2:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.ndim != 3:
            raise ValueError(f"Expected input_ids with 3 dims, got shape {tuple(input_ids.shape)}")
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape[:2], dtype=torch.bool, device=input_ids.device)
        elif attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)

        effective_nq = self._resolve_inference_nq(nq) # 本次推理实际使用的残差量化码本（quantizer）通道数
        batch_size = input_ids.shape[0]
        current_input_ids = input_ids
        current_attention_mask = attention_mask.to(device=input_ids.device)
        # KV cache 模式下只喂最新一帧；否则每次喂完整序列
        current_model_input_ids = current_input_ids
        generated_frames = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        past_key_values = None
        local_dtype = self.local_transformer.ln_f.weight.dtype

        for step_index in range(max_new_frames):
            # 历史音频帧，用于重复惩罚
            generated_audio_history = torch.stack(generated_frames, dim=1) if generated_frames else None

            # ---- 第 1 步：全局 Transformer ----
            global_inputs_embeds = self._build_inputs_embeds(current_model_input_ids)
            global_outputs = self.transformer(
                input_ids=None,
                past_key_values=past_key_values,
                attention_mask=current_attention_mask,
                position_ids=None,
                inputs_embeds=global_inputs_embeds,
                use_cache=use_kv_cache,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                cu_seqlens=None,
                num_sequences=None,
            )
            global_hidden_states = global_outputs.last_hidden_state[:, -1, :].to(dtype=local_dtype)

            # ---- 第 2 步：局部 Transformer 预测文本侧控制 token ----
            local_inputs_embeds = global_hidden_states.unsqueeze(1)
            local_hidden_states = self._decode_local_last_hidden_state(local_inputs_embeds)
            text_logits = self.text_lm_head(local_hidden_states)
            self._ensure_finite_generation_logits(text_logits, "text logits")
            next_text_tokens = self._sample_next_assistant_text_token(
                logits=text_logits,
                do_sample=do_sample,
                temperature=text_temperature,
                top_k=text_top_k,
                top_p=text_top_p,
            )
            # 已结束的样本不再继续；预测到结束 token 的样本标记为 finished
            should_continue = next_text_tokens.eq(self.config.audio_assistant_slot_token_id) & ~finished
            finished = finished | next_text_tokens.eq(self.config.audio_end_token_id)
            if not should_continue.any():
                break

            # ---- 第 3 步：逐码本通道自回归预测本帧音频 token ----
            next_frame_tokens = []
            current_local_input = self.transformer.wte(next_text_tokens).to(dtype=local_dtype)
            for channel_index in range(effective_nq):
                local_inputs_embeds = torch.cat([local_inputs_embeds, current_local_input.unsqueeze(1)], dim=1)
                local_hidden_states = self._decode_local_last_hidden_state(local_inputs_embeds)
                channel_logits = self.audio_lm_heads[channel_index](local_hidden_states)
                self._ensure_finite_generation_logits(channel_logits, f"audio logits[{channel_index}]")
                channel_token = self._sample_next_token(
                    logits=channel_logits,
                    do_sample=do_sample,
                    temperature=audio_temperature,
                    top_k=audio_top_k,
                    top_p=audio_top_p,
                    previous_token_ids=(
                        None if generated_audio_history is None else generated_audio_history[:, :, channel_index]
                    ),
                    repetition_penalty=audio_repetition_penalty,
                )
                next_frame_tokens.append(channel_token)
                current_local_input = self.audio_embeddings[channel_index](channel_token).to(dtype=local_dtype)

            # 未使用的通道（nq 之后的）填 pad
            next_frame_prefix = torch.stack(next_frame_tokens, dim=-1)
            if effective_nq < self.config.n_vq:
                next_frame = torch.full(
                    (batch_size, self.config.n_vq),
                    self.config.audio_pad_token_id,
                    dtype=next_frame_prefix.dtype,
                    device=next_frame_prefix.device,
                )
                next_frame[:, :effective_nq] = next_frame_prefix
            else:
                next_frame = next_frame_prefix
            # 已结束样本的本帧全部填 pad
            padded_next_frame = next_frame.masked_fill(~should_continue.unsqueeze(-1), self.config.audio_pad_token_id)
            generated_frames.append(padded_next_frame)

            # ---- 第 4 步：把本帧拼回输入，准备下一帧 ----
            next_row = self._build_generation_row(
                batch_size=batch_size,
                device=input_ids.device,
                audio_token_ids=padded_next_frame,
            )
            if (~should_continue).any():
                # 已结束样本：文本通道填 pad，不再占语义槽位
                next_row[~should_continue, 0, 0] = self.config.pad_token_id
                next_row[~should_continue, 0, 1:] = self.config.audio_pad_token_id

            current_input_ids = torch.cat([current_input_ids, next_row], dim=1)
            current_attention_mask = torch.cat([current_attention_mask, should_continue.unsqueeze(1)], dim=1)
            if use_kv_cache:
                current_model_input_ids = next_row
                past_key_values = global_outputs.past_key_values
            else:
                current_model_input_ids = current_input_ids

            yield {
                "type": "frame",
                "step_index": int(step_index),
                "audio_token_ids": padded_next_frame.detach().clone(),
                "active_mask": should_continue.detach().clone(),
                "finished_mask": finished.detach().clone(),
            }

        if generated_frames:
            audio_token_ids = torch.stack(generated_frames, dim=1)
        else:
            audio_token_ids = torch.empty((batch_size, 0, self.config.n_vq), dtype=torch.long, device=input_ids.device)

        if not return_dict_in_generate:
            yield {"type": "final", "audio_token_ids": audio_token_ids}
            return
        yield {
            "type": "final",
            "generation": _generation_output_type()(audio_token_ids=audio_token_ids, prompt_input_ids=input_ids),
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
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
        return_dict_in_generate: bool = True,
    ):
        """一次性生成全部音频 token，返回最终结果（非流式）。"""
        final_output: Any = None
        for event in self._iter_generation_events(
            input_ids=input_ids,
            attention_mask=attention_mask,
            nq=nq,
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
            return_dict_in_generate=return_dict_in_generate,
        ):
            if event["type"] != "final":
                continue
            final_output = event.get("generation", event.get("audio_token_ids"))
        if final_output is None:
            raise RuntimeError("Generation finished without producing a final output.")
        return final_output

    @torch.no_grad()
    def generate_stream(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
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
        return_dict_in_generate: bool = True,
    ) -> Iterator[dict[str, Any]]:
        """流式生成：每生成一帧就产出一个 ``frame`` 事件，最后产出 ``final``。"""
        yield from self._iter_generation_events(
            input_ids=input_ids,
            attention_mask=attention_mask,
            nq=nq,
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
            return_dict_in_generate=return_dict_in_generate,
        )


def _generation_output_type():
    """惰性导入生成输出 dataclass（定义在 modeling 模块，避免循环导入）。"""
    from .modeling import MossTTSNanoGenerationOutput

    return MossTTSNanoGenerationOutput

