# coding=utf-8
"""音频编解码器（audio tokenizer）输出解析与调用封装。

本文件是 ``MossTTSNanoForCausalLM`` 的 mixin：方法通过 ``self`` 访问模型其他部分，
类本身不单独实例化。
"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torchaudio



class MossTTSNanoAudioCodecMixin:
    # ------------------------------------------------------------------
    # 音频编解码器输出的解析（兼容多种返回格式）
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_tensor_candidate(output: Any) -> Any:
        """从编解码器的各种可能返回类型中取出码本张量。"""
        if torch.is_tensor(output) or isinstance(output, np.ndarray):
            return output
        for attr_name in ("audio_codes", "audio_token_ids", "codes", "tokens", "input_ids"):
            value = getattr(output, attr_name, None)
            if value is not None:
                return value
        if isinstance(output, dict):
            for key in ("audio_codes", "audio_token_ids", "codes", "tokens", "input_ids"):
                if key in output:
                    return output[key]
            if len(output) == 1:
                return next(iter(output.values()))
        if isinstance(output, (list, tuple)) and output:
            # (张量, 长度) 形式的二元组，取第一个元素
            if len(output) == 2 and isinstance(output[1], (int, float)):
                return output[0]
            return MossTTSNanoAudioCodecMixin._extract_tensor_candidate(output[0])
        raise TypeError(f"Unsupported audio tokenizer output type: {type(output)!r}")

    @staticmethod
    def _extract_audio_code_length(output: Any) -> Optional[int]:
        """从编解码器输出中提取有效码本长度（去除 padding 用）。"""
        for attr_name in ("audio_codes_lengths", "audio_token_ids_lengths", "codes_lengths", "lengths"):
            candidate = getattr(output, attr_name, None)
            if candidate is not None:
                lengths = torch.as_tensor(candidate).reshape(-1)
                if lengths.numel() > 0:
                    return int(lengths[0].item())
        if isinstance(output, dict):
            for key in ("audio_codes_lengths", "audio_token_ids_lengths", "codes_lengths", "lengths"):
                if key in output:
                    lengths = torch.as_tensor(output[key]).reshape(-1)
                    if lengths.numel() > 0:
                        return int(lengths[0].item())
        if isinstance(output, (list, tuple)) and len(output) >= 2:
            candidate = output[1]
            if torch.is_tensor(candidate) or isinstance(candidate, np.ndarray):
                lengths = torch.as_tensor(candidate).reshape(-1)
                if lengths.numel() > 0:
                    return int(lengths[0].item())
            if isinstance(candidate, (int, float)):
                return int(candidate)
        return None

    def _normalize_audio_codes(self, audio_codes: Any) -> torch.LongTensor:
        """把编解码器的输出统一成 [frames, n_vq] 的二维 long 张量。

        兼容 (n_vq, T)、(1, n_vq, T)、(B, n_vq, T) 等多种布局，
        多余通道截断、按有效长度裁剪。
        """
        code_length = self._extract_audio_code_length(audio_codes)
        tensor = torch.as_tensor(self._extract_tensor_candidate(audio_codes))
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(-1)
        if tensor.ndim == 3:
            if tensor.shape[1] == 1 and tensor.shape[0] >= self.config.n_vq:
                # (n_vq, 1, T) -> (T, n_vq)
                tensor = tensor[: self.config.n_vq, 0, :].transpose(0, 1)
            elif tensor.shape[0] == 1:
                tensor = tensor[0]
            elif tensor.shape[1] == self.config.n_vq:
                # (B, n_vq, T) -> 取第 0 条后转置
                tensor = tensor.transpose(1, 2)[0]
            elif tensor.shape[-1] == self.config.n_vq:
                tensor = tensor[0]
            else:
                raise ValueError(f"Unable to normalize audio codes with shape {tuple(tensor.shape)}")

        if tensor.ndim != 2:
            raise ValueError(f"Expected audio codes with 2 dims after normalization, got {tuple(tensor.shape)}")
        if tensor.shape[-1] != self.config.n_vq and tensor.shape[0] == self.config.n_vq:
            # (n_vq, T) -> (T, n_vq)
            tensor = tensor.transpose(0, 1)
        elif tensor.shape[-1] != self.config.n_vq and tensor.shape[0] > self.config.n_vq:
            tensor = tensor[: self.config.n_vq].transpose(0, 1)
        elif tensor.shape[-1] > self.config.n_vq:
            tensor = tensor[:, : self.config.n_vq]
        if tensor.shape[-1] != self.config.n_vq:
            raise ValueError(
                f"Expected normalized audio codes with trailing dim {self.config.n_vq}, got {tuple(tensor.shape)}"
            )
        if code_length is not None:
            tensor = tensor[:code_length]
        return tensor.to(dtype=torch.long)

    def _extract_waveform_and_sample_rate(
        self,
        decode_output: Any,
        fallback_sample_rate: int,
    ) -> tuple[torch.FloatTensor, int]:
        """从解码输出中提取 (波形 [C, T], 采样率)，兼容多种返回格式。"""
        sample_rate = fallback_sample_rate
        waveform = decode_output
        waveform_length = None

        # 先按对象属性查找
        for key in ("sample_rate", "sampling_rate"):
            value = getattr(decode_output, key, None)
            if value is not None:
                sample_rate = int(value)
                break
        for key in ("waveform", "audio", "wav", "samples"):
            value = getattr(decode_output, key, None)
            if value is not None:
                waveform = value
                break
        for key in ("audio_lengths", "waveform_lengths", "lengths"):
            value = getattr(decode_output, key, None)
            if value is not None:
                lengths = torch.as_tensor(value).reshape(-1)
                if lengths.numel() > 0:
                    waveform_length = int(lengths[0].item())
                break

        # 再按 dict / 元组形式查找
        if isinstance(decode_output, dict):
            for key in ("sample_rate", "sampling_rate"):
                if key in decode_output:
                    sample_rate = int(decode_output[key])
                    break
            for key in ("waveform", "audio", "wav", "samples"):
                if key in decode_output:
                    waveform = decode_output[key]
                    break
            for key in ("audio_lengths", "waveform_lengths", "lengths"):
                if key in decode_output:
                    lengths = torch.as_tensor(decode_output[key]).reshape(-1)
                    if lengths.numel() > 0:
                        waveform_length = int(lengths[0].item())
                        break
        elif isinstance(decode_output, (list, tuple)) and decode_output:
            if len(decode_output) == 2 and isinstance(decode_output[1], (int, float)):
                waveform = decode_output[0]
                sample_rate = int(decode_output[1])
            else:
                waveform = decode_output[0]

        # 统一形状为 [channels, samples]
        waveform_tensor = torch.as_tensor(waveform, dtype=torch.float32)
        if waveform_tensor.ndim == 3 and waveform_tensor.shape[0] == 1:
            waveform_tensor = waveform_tensor[0]
        if waveform_tensor.ndim == 2 and waveform_tensor.shape[0] > waveform_tensor.shape[1]:
            # (T, C) -> (C, T)
            waveform_tensor = waveform_tensor.transpose(0, 1)
        if waveform_tensor.ndim == 1:
            waveform_tensor = waveform_tensor.unsqueeze(0)
        if waveform_tensor.ndim != 2:
            raise ValueError(f"Expected decoded waveform with 2 dims, got {tuple(waveform_tensor.shape)}")
        if waveform_length is not None:
            waveform_tensor = waveform_tensor[..., : max(0, waveform_length)]
        return waveform_tensor.cpu(), sample_rate

    def _call_audio_encode(
        self,
        audio_tokenizer,
        waveform: torch.FloatTensor,
        sample_rate: int,
    ) -> Any:
        """调用音频编解码器的 batch_encode，把波形编码为码本 token。"""
        del sample_rate
        batch_encode_fn = getattr(audio_tokenizer, "batch_encode", None)
        if batch_encode_fn is None:
            raise AttributeError("audio_tokenizer must provide a batch_encode method.")

        waveform_tensor = torch.as_tensor(waveform, dtype=torch.float32, device=self._resolve_device(waveform.device))
        if waveform_tensor.ndim == 1:
            waveform_tensor = waveform_tensor.unsqueeze(0)
        if waveform_tensor.ndim != 2:
            raise ValueError(
                f"MOSS audio tokenizer encode expects waveform shaped like (C, T), got {tuple(waveform_tensor.shape)}"
            )

        with self._audio_tokenizer_inference_context(audio_tokenizer, waveform_tensor.device):
            return batch_encode_fn([waveform_tensor], chunk_duration=None)

    def _call_audio_decode(
        self,
        audio_tokenizer,
        audio_token_ids: torch.LongTensor,
        sample_rate: int,
        nq: Optional[int] = None,
    ) -> Any:
        """调用音频编解码器的 batch_decode，把码本 token 解码为波形。"""
        del sample_rate
        batch_decode_fn = getattr(audio_tokenizer, "batch_decode", None)
        if batch_decode_fn is None:
            raise AttributeError("audio_tokenizer must provide a batch_decode method.")

        effective_nq = self._resolve_inference_nq(nq)
        decode_codes = self._prepare_audio_codes_for_decode(audio_token_ids, nq=effective_nq)
        with self._audio_tokenizer_inference_context(audio_tokenizer, decode_codes.device):
            return batch_decode_fn([decode_codes], num_quantizers=effective_nq, chunk_duration=None)

    def _extract_batch_waveforms_and_sample_rate(
        self,
        decode_output: Any,
        fallback_sample_rate: int,
        batch_size: int,
    ) -> tuple[list[torch.FloatTensor], int]:
        """从批量解码输出中提取每条样本的波形列表与采样率。"""
        sample_rate = fallback_sample_rate
        audio = decode_output
        audio_lengths = None

        for key in ("sample_rate", "sampling_rate"):
            value = getattr(decode_output, key, None)
            if value is not None:
                sample_rate = int(value)
                break
        for key in ("waveform", "audio", "wav", "samples"):
            value = getattr(decode_output, key, None)
            if value is not None:
                audio = value
                break
        for key in ("audio_lengths", "waveform_lengths", "lengths"):
            value = getattr(decode_output, key, None)
            if value is not None:
                audio_lengths = value
                break

        if isinstance(decode_output, dict):
            for key in ("sample_rate", "sampling_rate"):
                if key in decode_output:
                    sample_rate = int(decode_output[key])
                    break
            for key in ("waveform", "audio", "wav", "samples"):
                if key in decode_output:
                    audio = decode_output[key]
                    break
            for key in ("audio_lengths", "waveform_lengths", "lengths"):
                if key in decode_output:
                    audio_lengths = decode_output[key]
                    break

        # 统一形状为 [batch, channels, samples]
        audio_tensor = torch.as_tensor(audio, dtype=torch.float32)
        if audio_tensor.ndim == 2:
            audio_tensor = audio_tensor.unsqueeze(0)
        if audio_tensor.ndim != 3:
            raise ValueError(f"Expected batched decoded audio with 3 dims, got {tuple(audio_tensor.shape)}")
        if audio_tensor.shape[0] != int(batch_size):
            raise ValueError(
                f"Expected decoded batch size {batch_size}, got audio tensor shape {tuple(audio_tensor.shape)}"
            )

        # 每条样本的有效长度；缺失时视为整段有效
        if audio_lengths is None:
            lengths_tensor = torch.full(
                (batch_size,),
                int(audio_tensor.shape[-1]),
                device=audio_tensor.device,
                dtype=torch.long,
            )
        else:
            lengths_tensor = torch.as_tensor(audio_lengths, dtype=torch.long, device=audio_tensor.device).reshape(-1)
            if lengths_tensor.numel() != int(batch_size):
                raise ValueError(f"Expected {batch_size} decoded audio lengths, got {int(lengths_tensor.numel())}")

        waveform_rows: list[torch.FloatTensor] = []
        for row_index in range(batch_size):
            row_length = max(0, int(lengths_tensor[row_index].item()))
            waveform_rows.append(audio_tensor[row_index, :, :row_length].detach().cpu())
        return waveform_rows, sample_rate

    # ------------------------------------------------------------------
    # 音频编解码器的元信息探测
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_audio_tokenizer_downsample_rate(audio_tokenizer) -> int:
        """探测编解码器的下采样率（波形采样点数 / 码本帧数）。"""
        for holder in (audio_tokenizer, getattr(audio_tokenizer, "config", None)):
            if holder is None:
                continue
            for attr_name in ("downsample_rate", "hop_length", "frame_size"):
                value = getattr(holder, attr_name, None)
                if value is not None:
                    return int(value)
            sampling_rate = getattr(holder, "sampling_rate", None)
            frame_rate = getattr(holder, "frame_rate", None)
            if sampling_rate is not None and frame_rate not in (None, 0):
                return int(round(float(sampling_rate) / float(frame_rate)))
        raise ValueError("audio_tokenizer.downsample_rate is required for prompt-audio decoding.")

    def _resolve_audio_tokenizer_sample_rate(self, audio_tokenizer) -> int:
        """探测编解码器的工作采样率。"""
        for holder in (audio_tokenizer, getattr(audio_tokenizer, "config", None)):
            if holder is None:
                continue
            for attr_name in ("sampling_rate", "sample_rate"):
                value = getattr(holder, attr_name, None)
                if value is not None:
                    return int(value)
        return int(self.config.audio_tokenizer_sample_rate)

    @staticmethod
    def _resolve_audio_tokenizer_channels(audio_tokenizer) -> int:
        """探测编解码器输出的声道数。"""
        for holder in (audio_tokenizer, getattr(audio_tokenizer, "config", None)):
            if holder is None:
                continue
            for attr_name in ("number_channels", "channels_numbers", "audio_channels", "channels", "num_channels"):
                value = getattr(holder, attr_name, None)
                if value is not None:
                    return int(value)
        return 1

    @staticmethod
    def _audio_tokenizer_inference_context(audio_tokenizer, device: Union[str, torch.device]):
        """编解码器推理上下文；当前为空操作，预留钩子。"""
        del audio_tokenizer, device
        return nullcontext()

    def _prepare_audio_codes_for_decode(
        self,
        audio_token_ids: torch.LongTensor,
        nq: Optional[int] = None,
    ) -> torch.LongTensor:
        """把生成的音频 token 整理成编解码器要求的 [nq, T] 布局。"""
        effective_nq = self._resolve_inference_nq(nq)
        tensor = torch.as_tensor(audio_token_ids, dtype=torch.long)
        if tensor.ndim == 2:
            if tensor.shape[-1] == self.config.n_vq and tensor.shape[0] != self.config.n_vq:
                # (T, n_vq) -> 截取前 nq 通道后转置为 (nq, T)
                return tensor[:, :effective_nq].transpose(0, 1).contiguous()
            if tensor.shape[0] == self.config.n_vq:
                return tensor[:effective_nq].contiguous()
        elif tensor.ndim == 3:
            if tensor.shape[-1] == self.config.n_vq:
                # (B, T, n_vq) -> (nq, B, T)
                return tensor[..., :effective_nq].permute(2, 0, 1).contiguous()
            if tensor.shape[0] == self.config.n_vq:
                return tensor[:effective_nq].contiguous()
        raise ValueError(
            f"Expected generated audio token ids shaped like (T, {self.config.n_vq}) or ({self.config.n_vq}, T); got {tuple(tensor.shape)}"
        )

    def _load_reference_audio(
        self,
        reference_audio_path: Union[str, Path],
        target_sample_rate: int,
        target_channels: int,
    ) -> tuple[torch.FloatTensor, int]:
        """加载参考音频，并重采样 / 转换声道到目标格式。"""
        waveform, sample_rate = torchaudio.load(str(reference_audio_path))
        waveform = waveform.to(torch.float32)
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
            sample_rate = target_sample_rate
        current_channels = int(waveform.shape[0])
        if current_channels == target_channels:
            return waveform, sample_rate
        if current_channels == 1 and target_channels > 1:
            # 单声道 -> 多声道：复制
            return waveform.repeat(target_channels, 1), sample_rate
        if current_channels > 1 and target_channels == 1:
            # 多声道 -> 单声道：取平均
            return waveform.mean(dim=0, keepdim=True), sample_rate
        raise ValueError(f"Unsupported reference audio channel conversion: {current_channels} -> {target_channels}")
