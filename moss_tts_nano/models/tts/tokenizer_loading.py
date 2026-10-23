# coding=utf-8
"""文本/音频分词器的定位与加载、注意力后端切换与数值稳定性兜底。

本文件是 ``MossTTSNanoForCausalLM`` 的 mixin：方法通过 ``self`` 访问模型其他部分，
类本身不单独实例化。
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional, Union

import torch
from transformers import AutoModel

from .tokenization import MossTTSNanoSentencePieceTokenizer
from .constants import (
    MOSS_AUDIO_TOKENIZER_NANO_TYPE,
    DEFAULT_MOSS_AUDIO_TOKENIZER_PRETRAINED_NAME_OR_PATH,
)

try:  # 优先：已安装 moss_tts_nano 包，模型代码与权重分离
    from moss_tts_nano.models.audio_tokenizer import MossAudioTokenizerModel
except ImportError:  # 通过 trust_remote_code 加载 checkpoint、包未安装时的兜底
    MossAudioTokenizerModel = None




class MossTTSNanoTokenizerLoadingMixin:
    def _resolve_device(self, device: Optional[Union[str, torch.device]] = None) -> torch.device:
        """解析运行设备；未指定时跟随模型参数所在设备。"""
        return torch.device(device) if device is not None else next(self.parameters()).device

    # ------------------------------------------------------------------
    # 文本分词器的定位与加载
    # ------------------------------------------------------------------

    @staticmethod
    def _looks_like_hf_tokenizer_dir(candidate_path: Path) -> bool:
        """判断目录是否像 HF 分词器目录（含 tokenizer.model / tokenizer.json 等）。"""
        if not candidate_path.is_dir():
            return False
        if (candidate_path / "tokenizer.model").is_file():
            return True
        if (candidate_path / "tokenizer.json").is_file():
            return True
        if (candidate_path / "tokenizer_config.json").is_file() and (
            (candidate_path / "vocab.json").is_file()
            or (candidate_path / "merges.txt").is_file()
            or (candidate_path / "special_tokens_map.json").is_file()
        ):
            return True
        return False

    @staticmethod
    def _looks_like_hf_repo_id(candidate: str) -> bool:
        """判断字符串是否像 HF 仓库 id（形如 "org/name"），而非本地路径。"""
        stripped = candidate.strip()
        if not stripped:
            return False
        if stripped.startswith((os.sep, ".", "~")):
            return False
        if "\\" in stripped:
            return False
        parts = stripped.split("/")
        return len(parts) == 2 and all(part.strip() for part in parts)

    @staticmethod
    def _existing_local_path(raw_path: Union[str, Path]) -> Optional[Path]:
        """展开 ~ 并解析为绝对路径；不存在时返回 None。"""
        candidate_path = Path(raw_path).expanduser()
        if not candidate_path.exists():
            return None
        return candidate_path.resolve()

    def _resolve_text_tokenizer_path(self, raw_path: Union[str, Path]) -> str:
        """把用户给出的分词器路径解析成可加载的来源。

        支持：本地 .model 文件、HF 分词器目录、实验目录（内含 hf_tokenizer/
        sentencepiece/ 或 final_summary.json）、HF 仓库 id。
        """
        local_candidate_path = self._existing_local_path(raw_path)
        if local_candidate_path is None:
            raw_source = str(raw_path).strip()
            if self._looks_like_hf_repo_id(raw_source):
                return raw_source
            raise FileNotFoundError(f"Tokenizer path does not exist: {raw_source}")

        candidate_path = local_candidate_path
        if candidate_path.is_file() and candidate_path.suffix == ".model":
            return str(candidate_path)
        if candidate_path.is_dir():
            if (candidate_path / "tokenizer.model").is_file():
                return str(candidate_path)
            if self._looks_like_hf_tokenizer_dir(candidate_path):
                return str(candidate_path)
            hf_dir = candidate_path / "hf_tokenizer"
            if self._looks_like_hf_tokenizer_dir(hf_dir):
                return str(hf_dir)
            sentencepiece_model = candidate_path / "sentencepiece" / "mossttsnano_spm_bpe.model"
            if sentencepiece_model.is_file():
                return str(sentencepiece_model)
            # 训练实验目录：从 final_summary.json 里读最近一次导出的分词器目录
            final_summary_path = candidate_path / "final_summary.json"
            if final_summary_path.is_file():
                final_summary = json.loads(final_summary_path.read_text(encoding="utf-8"))
                latest_hf_dir = final_summary.get("latest_hf_tokenizer_dir")
                if latest_hf_dir:
                    latest_hf_path = Path(str(latest_hf_dir))
                    if self._looks_like_hf_tokenizer_dir(latest_hf_path):
                        return str(latest_hf_path.resolve())
        raise ValueError(
            "Could not resolve a tokenizer from the provided path. Expected a tokenizer dir, experiment dir, or SentencePiece .model file."
        )

    def _load_resolved_text_tokenizer(self, resolved_path: str):
        """按解析后的来源实际加载分词器（本地文件优先离线加载）。"""
        local_path = self._existing_local_path(resolved_path)
        load_source = str(local_path) if local_path is not None else str(resolved_path)
        if local_path is not None and local_path.is_file() and local_path.suffix == ".model":
            return MossTTSNanoSentencePieceTokenizer(vocab_file=str(local_path))
        try:
            load_kwargs: dict[str, object] = {}
            if local_path is not None:
                load_kwargs["local_files_only"] = True
            return MossTTSNanoSentencePieceTokenizer.from_pretrained(
                load_source,
                **load_kwargs,
            )
        except Exception:
            # 兜底：目录下直接找 tokenizer.model
            if local_path is not None:
                model_path = local_path / "tokenizer.model"
                if model_path.is_file():
                    return MossTTSNanoSentencePieceTokenizer(vocab_file=str(model_path))
            raise

    def _resolve_default_text_tokenizer_path(self) -> str:
        """未显式指定分词器路径时，从 checkpoint 所在位置推断。"""
        candidates: list[Union[str, Path]] = []

        raw_name_or_path = getattr(self.config, "_name_or_path", None)
        if raw_name_or_path:
            candidates.append(str(raw_name_or_path).strip())

        raw_model_name_or_path = getattr(self, "name_or_path", None)
        if raw_model_name_or_path:
            candidates.append(str(raw_model_name_or_path).strip())

        checked: set[str] = set()
        for candidate in candidates:
            raw_candidate = str(candidate).strip()
            if not raw_candidate:
                continue
            try:
                resolved_candidate = self._resolve_text_tokenizer_path(raw_candidate)
            except (FileNotFoundError, ValueError):
                continue
            if resolved_candidate in checked:
                continue
            checked.add(resolved_candidate)
            return resolved_candidate

        # 都解析失败时，原样返回第一个候选，让后续加载报错信息更直观
        for candidate in candidates:
            raw_candidate = str(candidate).strip()
            if raw_candidate:
                return raw_candidate

        raise FileNotFoundError(
            "Could not locate the text tokenizer next to the checkpoint; pass text_tokenizer_path explicitly."
        )

    def _load_text_tokenizer(self, text_tokenizer=None, text_tokenizer_path: Optional[str] = None):
        """加载文本分词器（带缓存）；已传入实例时直接返回。"""
        if text_tokenizer is not None:
            return text_tokenizer

        resolved_path = (
            self._resolve_text_tokenizer_path(text_tokenizer_path)
            if text_tokenizer_path is not None
            else self._resolve_default_text_tokenizer_path()
        )
        normalized_path = str(resolved_path)
        cached = getattr(self, "_cached_text_tokenizer", None)
        cached_path = getattr(self, "_cached_text_tokenizer_path", None)
        if cached is not None and cached_path == normalized_path:
            return cached

        tokenizer = self._load_resolved_text_tokenizer(resolved_path=resolved_path)
        if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        self._cached_text_tokenizer = tokenizer
        self._cached_text_tokenizer_path = normalized_path
        return tokenizer

    # ------------------------------------------------------------------
    # 音频编解码器（audio tokenizer）的类型解析与加载
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_audio_tokenizer_type(audio_tokenizer_type: Optional[str]) -> Optional[str]:
        """规范化音频编解码器类型名；开源包只支持 moss-audio-tokenizer-nano。"""
        if audio_tokenizer_type is None:
            return None
        normalized = str(audio_tokenizer_type).strip().lower()
        if not normalized:
            return None
        if normalized == MOSS_AUDIO_TOKENIZER_NANO_TYPE:
            return MOSS_AUDIO_TOKENIZER_NANO_TYPE
        raise ValueError(
            "Unsupported audio tokenizer type. "
            f"The open-source package only supports '{MOSS_AUDIO_TOKENIZER_NANO_TYPE}'."
        )

    def _resolve_audio_tokenizer_type(self, audio_tokenizer_type: Optional[str]) -> str:
        """解析音频编解码器类型：显式参数 > 配置 > 默认值。"""
        explicit_type = self._normalize_audio_tokenizer_type(audio_tokenizer_type)
        if explicit_type is not None:
            return explicit_type
        config_type = self._normalize_audio_tokenizer_type(getattr(self.config, "audio_tokenizer_type", None))
        return MOSS_AUDIO_TOKENIZER_NANO_TYPE if config_type is None else config_type

    @staticmethod
    def _set_decoder_attention_implementation(decoder, attn_implementation: str) -> None:
        """把一个 GPT-2 主干（含每一层 block）的注意力实现统一切换。"""
        decoder.attn_implementation = str(attn_implementation)
        if getattr(decoder, "config", None) is not None:
            decoder.config._attn_implementation = str(attn_implementation)
        for block in getattr(decoder, "h", []):
            block.attn.attn_implementation = str(attn_implementation)

    def _set_attention_implementation(
        self,
        attn_implementation: str,
        local_attn_implementation: Optional[str] = None,
    ) -> None:
        """同时切换全局 / 局部 Transformer 的注意力实现。"""
        resolved_global = str(attn_implementation)
        resolved_local = resolved_global if local_attn_implementation is None else str(local_attn_implementation)
        self.config.attn_implementation = resolved_global
        self.config.gpt2_config._attn_implementation = resolved_global
        self._set_decoder_attention_implementation(self.transformer, resolved_global)
        self.config.local_transformer_attn_implementation = resolved_local
        self._set_decoder_attention_implementation(self.local_transformer, resolved_local)

    @staticmethod
    def _select_fallback_attention_implementation(device: torch.device) -> str:
        """数值不稳定时的降级注意力实现：CUDA 用 sdpa，其余用 eager。"""
        return "sdpa" if device.type == "cuda" else "eager"

    @staticmethod
    def _is_generation_stability_error(exc: Exception) -> bool:
        """判断异常是否属于生成过程中的数值不稳定（可降级重试）。"""
        message = str(exc)
        return any(
            marker in message
            for marker in (
                "Non-finite",
                "device-side assert triggered",
                "probability tensor contains either",
                "flash_attention_2 requires fp16/bf16 tensors",
            )
        )

    def _apply_inference_stability_fallback(self, device: torch.device) -> None:
        """应用稳定性降级：模型转 fp32，并切换到更稳的注意力实现。"""
        fallback_attn = self._select_fallback_attention_implementation(device)
        if next(self.parameters()).dtype != torch.float32:
            self.to(device=device, dtype=torch.float32)
        self._set_attention_implementation(fallback_attn)
        logging.warning(
            "retrying inference with dtype=float32 attn_implementation=%s due to numerical instability",
            fallback_attn,
        )

    def _load_audio_tokenizer(
        self,
        audio_tokenizer=None,
        audio_tokenizer_type: Optional[str] = None,
        audio_tokenizer_pretrained_name_or_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """加载音频编解码器（带缓存）；已传入实例时直接返回。"""
        if audio_tokenizer is not None:
            return audio_tokenizer

        resolved_type = self._resolve_audio_tokenizer_type(audio_tokenizer_type=audio_tokenizer_type)
        if resolved_type != MOSS_AUDIO_TOKENIZER_NANO_TYPE:
            raise ValueError(
                f"Unsupported audio tokenizer type {resolved_type!r}; expected '{MOSS_AUDIO_TOKENIZER_NANO_TYPE}'."
            )

        resolved_pretrained_name_or_path = (
            audio_tokenizer_pretrained_name_or_path
            or getattr(self.config, "audio_tokenizer_pretrained_name_or_path", None)
            or DEFAULT_MOSS_AUDIO_TOKENIZER_PRETRAINED_NAME_OR_PATH
        )
        # 本地路径存在则离线加载，否则按 HF 仓库 id 处理
        candidate_path = Path(str(resolved_pretrained_name_or_path)).expanduser()
        if candidate_path.exists():
            load_source = str(candidate_path.resolve())
            load_kwargs: dict[str, object] = {
                "local_files_only": True,
            }
            cache_key = f"{resolved_type}|{load_source}"
        else:
            load_source = str(resolved_pretrained_name_or_path)
            load_kwargs = {}
            cache_key = f"{resolved_type}|{load_source}"

        cached = getattr(self, "_cached_audio_tokenizer", None)
        cached_path = getattr(self, "_cached_audio_tokenizer_path", None)
        if cached is not None and cached_path == cache_key:
            tokenizer = cached
        else:
            if MossAudioTokenizerModel is not None:
                tokenizer = MossAudioTokenizerModel.from_pretrained(load_source, **load_kwargs)
            else:  # 动态模块加载的 checkpoint：音频编解码器代码来自其自身仓库
                tokenizer = AutoModel.from_pretrained(load_source, trust_remote_code=True, **load_kwargs)
            if hasattr(tokenizer, "eval"):
                tokenizer.eval()
            self._cached_audio_tokenizer = tokenizer
            self._cached_audio_tokenizer_path = cache_key

        resolved_device = self._resolve_device(device)
        return tokenizer.to(resolved_device) if hasattr(tokenizer, "to") else tokenizer
