# coding=utf-8
"""MOSS-TTS-Nano 主模型：文本 -> 离散音频 token 的因果语言模型。

模型结构（global + local 两级 Transformer）：
- 全局 Transformer（GPT-2 架构）：沿时间轴逐帧建模。每一帧的输入 embedding
  = 文本 token embedding + 各音频码本通道 embedding 之和；
- 局部 Transformer（层数更少的小 GPT-2）：在每一帧内部，先预测一个文本侧
  控制 token（继续生成 / 结束），再逐码本通道自回归地预测音频 token。

输入张量 ``input_ids`` 的形状为 ``[batch, seq, n_vq + 1]``：
- 第 0 通道：文本 token id；
- 第 1 ~ n_vq 通道：各音频码本的 token id（无音频处填 audio_pad_token_id）。
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from transformers import AutoModel
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

from .configuration_moss_tts_nano import MossTTSNanoConfig
from .gpt2_decoder import MossTTSNanoGPT2Block, MossTTSNanoGPT2Model
from .prompting import (
    build_assistant_prompt_prefix,
    build_prompt_token_ids,
    build_user_prompt_after_reference,
    build_user_prompt_prefix,
)
from .tokenization_moss_tts_nano import MossTTSNanoSentencePieceTokenizer

try:  # 优先：已安装 moss_tts_nano 包，模型代码与权重分离
    from moss_tts_nano.models.audio_tokenizer import MossAudioTokenizerModel
except ImportError:  # 通过 trust_remote_code 加载 checkpoint、包未安装时的兜底
    MossAudioTokenizerModel = None


@dataclass
class MossTTSNanoOutput(ModelOutput):
    """前向输出：全局 Transformer 最后一层的隐藏状态及可选的缓存/中间结果。"""

    global_hidden_states: Optional[torch.FloatTensor] = None
    past_key_values: Optional[tuple[tuple[torch.Tensor, torch.Tensor], ...]] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


@dataclass
class MossTTSNanoGenerationOutput(ModelOutput):
    """生成输出：音频 token 序列 [batch, frames, n_vq] 及对应的 prompt。"""

    audio_token_ids: torch.LongTensor
    prompt_input_ids: Optional[torch.LongTensor] = None


MOSS_AUDIO_TOKENIZER_NANO_TYPE = "moss-audio-tokenizer-nano"
DEFAULT_MOSS_AUDIO_TOKENIZER_PRETRAINED_NAME_OR_PATH = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
# 语音克隆时，单个文本分片允许的最大 token 数
DEFAULT_VOICE_CLONE_MAX_TEXT_TOKENS = 50
# 语音克隆分片批处理时，每条样本预估的显存占用上限（GB）
DEFAULT_VOICE_CLONE_MAX_MEMORY_PER_SAMPLE_GB = 1.0
# 分片拼接时，短句/长句之后插入的停顿时长（秒）
DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_SHORT_SECONDS = 0.40
DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_LONG_SECONDS = 0.24
# 句子结束标点 / 子句切分标点 / 右闭合标点（中英文）
_SENTENCE_END_PUNCTUATION = frozenset(".!?。！？；;")
_CLAUSE_SPLIT_PUNCTUATION = frozenset(",，、；;：:")
_CLOSING_PUNCTUATION = frozenset("\"'”’)]}）】》」』")


class MossTTSNanoPreTrainedModel(PreTrainedModel):
    """HF 集成基类：声明配置类与权重切分/注意力后端支持。"""

    config_class = MossTTSNanoConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = False
    _no_split_modules = ["MossTTSNanoGPT2Block"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True


class MossTTSNanoForCausalLM(MossTTSNanoPreTrainedModel):
    """MOSS-TTS-Nano 因果语言模型（仅推理，不支持训练前向）。"""

    # 局部 Transformer 不自带词 embedding（输入是全局隐藏状态），加载时忽略该键
    _keys_to_ignore_on_load_unexpected = [r"local_transformer\.wte\.weight"]

    def __init__(self, config: MossTTSNanoConfig) -> None:
        super().__init__(config)
        config.gpt2_config.pad_token_id = config.pad_token_id
        config.gpt2_config._attn_implementation = config.attn_implementation

        # 全局 Transformer：沿时间轴建模
        self.transformer = MossTTSNanoGPT2Model(
            config.gpt2_config,
            attn_implementation=config.attn_implementation,
        )
        hidden_size = config.gpt2_config.hidden_size
        init_std = config.gpt2_config.initializer_range

        # 每个音频码本通道一个 embedding 表
        self.audio_embeddings = nn.ModuleList(
            [
                nn.Embedding(int(config.audio_codebook_sizes[index]), hidden_size)
                for index in range(config.n_vq)
            ]
        )
        # 文本输出头（与词 embedding 权重共享）
        self.text_lm_head = nn.Linear(hidden_size, config.gpt2_config.vocab_size, bias=False)
        # 每个音频码本通道一个输出头（与对应 embedding 权重共享）
        self.audio_lm_heads = nn.ModuleList(
            [
                nn.Linear(hidden_size, int(config.audio_codebook_sizes[index]), bias=False)
                for index in range(config.n_vq)
            ]
        )

        # 局部 Transformer：序列长度固定为 n_vq + 1（1 个文本控制 token + n_vq 个音频通道）
        local_gpt2_config = config.gpt2_config.to_dict()
        local_gpt2_config["n_layer"] = int(config.local_transformer_layers)
        local_gpt2_config["n_positions"] = config.n_vq + 1
        local_gpt2_config["n_ctx"] = config.n_vq + 1
        self.local_transformer = MossTTSNanoGPT2Model(
            GPT2Config(**local_gpt2_config),
            attn_implementation=str(config.local_transformer_attn_implementation),
        )
        # 局部 Transformer 的输入是 embedding 向量而非 token id，wte 用 Identity 占位
        self.local_transformer.wte = nn.Identity()

        for module in list(self.audio_embeddings) + [self.text_lm_head] + list(self.audio_lm_heads):
            if hasattr(module, "weight") and module.weight is not None:
                nn.init.normal_(module.weight, mean=0.0, std=init_std)

        self._tied_weights_keys = tuple(self.all_tied_weights_keys.keys())
        self.tie_weights()

    @property
    def all_tied_weights_keys(self) -> dict[str, str]:
        """权重共享表：输出头 -> 对应输入 embedding。"""
        tied_weights = {"text_lm_head.weight": "transformer.wte.weight"}
        tied_weights.update(
            {
                f"audio_lm_heads.{index}.weight": f"audio_embeddings.{index}.weight"
                for index in range(self.config.n_vq)
            }
        )
        return tied_weights

    def tie_weights(self, *args, **kwargs) -> None:
        """执行权重共享：文本/音频输出头分别复用词 embedding 与音频 embedding。"""
        del args, kwargs
        self.text_lm_head.weight = self.transformer.wte.weight
        for embedding, lm_head in zip(self.audio_embeddings, self.audio_lm_heads):
            lm_head.weight = embedding.weight

    def get_input_embeddings(self) -> nn.Embedding:
        return self.transformer.wte

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.transformer.wte = value
        self.tie_weights()

    def _build_inputs_embeds(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        """把 [batch, seq, n_vq+1] 的多通道 token id 融合成 embedding。

        每一帧的 embedding = 文本 embedding + 各音频通道 embedding 之和；
        音频通道为 pad 的位置不计入。
        """
        if input_ids.ndim != 3 or input_ids.shape[-1] != self.config.n_vq + 1:
            raise ValueError(
                f"Expected input_ids shape [batch, seq, {self.config.n_vq + 1}], got {tuple(input_ids.shape)}"
            )

        text_ids = input_ids[..., 0]
        inputs_embeds = self.transformer.wte(text_ids)

        for channel_index, embedding in enumerate(self.audio_embeddings):
            channel_ids = input_ids[..., channel_index + 1]
            valid_mask = channel_ids.ne(self.config.audio_pad_token_id)
            # 非 pad 但超出码本词表范围的 id 属于数据错误，直接报错
            invalid_mask = valid_mask & ((channel_ids < 0) | (channel_ids >= embedding.num_embeddings))
            if invalid_mask.any():
                invalid_token_ids = channel_ids[invalid_mask]
                raise ValueError(
                    "Found out-of-range audio token ids for channel "
                    f"{channel_index}: min={int(invalid_token_ids.min().item())} "
                    f"max={int(invalid_token_ids.max().item())} "
                    f"codebook_size={embedding.num_embeddings} "
                    f"audio_pad_token_id={self.config.audio_pad_token_id}"
                )
            # pad 位置先替换成 0 查表，再用掩码清零，避免越界
            safe_ids = channel_ids.masked_fill(~valid_mask, 0)
            audio_embeds = embedding(safe_ids)
            audio_embeds = audio_embeds * valid_mask.unsqueeze(-1)
            inputs_embeds = inputs_embeds + audio_embeds

        return inputs_embeds

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[tuple[tuple[torch.Tensor, torch.Tensor], ...]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        labels = kwargs.pop("labels", None)
        if labels is not None:
            raise NotImplementedError("This open-source package is inference-only and does not support training forward.")
        if kwargs:
            ignored = ", ".join(sorted(kwargs.keys()))
            logging.debug("ignoring unsupported forward kwargs: %s", ignored)

        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided.")
            inputs_embeds = self._build_inputs_embeds(input_ids)

        outputs = self.transformer(
            input_ids=None,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=None,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cu_seqlens=None,
            num_sequences=None,
        )

        if not return_dict:
            return (
                outputs.last_hidden_state,
                outputs.past_key_values,
                outputs.hidden_states,
                outputs.attentions,
            )

        return MossTTSNanoOutput(
            global_hidden_states=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

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

    def _encode_text(self, tokenizer, text: str) -> list[int]:
        """文本编码（不添加特殊 token），兼容不支持该参数的 tokenizer。"""
        try:
            return list(tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return list(tokenizer.encode(text))

    # ------------------------------------------------------------------
    # 文本预处理与分片（语音克隆长文本时使用）
    # ------------------------------------------------------------------

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        """判断文本是否包含中日韩字符。"""
        return any(
            "一" <= ch <= "鿿"
            or "㐀" <= ch <= "䶿"
            or "぀" <= ch <= "ヿ"
            or "가" <= ch <= "힯"
            for ch in str(text)
        )

    @staticmethod
    def _prepare_text_for_sentence_chunking(text: str) -> str:
        """分片前的文本规范化：去空白、补结尾标点、英文首字母大写等。"""
        normalized_text = str(text).strip()
        if normalized_text == "":
            raise ValueError("Text prompt cannot be empty.")

        # 换行折叠为空格，压缩连续空格
        normalized_text = normalized_text.replace("\n", " ").replace("\r", " ")
        while "  " in normalized_text:
            normalized_text = normalized_text.replace("  ", " ")

        contains_cjk = MossTTSNanoForCausalLM._contains_cjk(normalized_text)
        if contains_cjk:
            # 中文文本保证以句末标点结尾
            if normalized_text[-1] not in _SENTENCE_END_PUNCTUATION:
                normalized_text = normalized_text + "。"
            return normalized_text

        if not normalized_text[0].isupper():
            normalized_text = normalized_text[0].upper() + normalized_text[1:]
        if normalized_text[-1].isalnum():
            normalized_text = normalized_text + "."
        # 英文短文本前补空格，缓解模型对超短输入的不稳定
        if len(normalized_text.split()) < 5:
            normalized_text = " " * 8 + normalized_text
        return normalized_text

    @staticmethod
    def _split_text_by_punctuation(text: str, punctuation: set[str] | frozenset[str]) -> list[str]:
        """按标点切分文本，标点保留在句尾，右闭合标点（引号/括号）跟随标点。"""
        sentences: list[str] = []
        current_chars: list[str] = []
        text = str(text)
        index = 0
        while index < len(text):
            char = text[index]
            current_chars.append(char)
            if char in punctuation:
                # 把紧跟在标点后的右闭合标点（如 ”）】）也并入当前句
                lookahead = index + 1
                while lookahead < len(text) and text[lookahead] in _CLOSING_PUNCTUATION:
                    current_chars.append(text[lookahead])
                    lookahead += 1
                sentence = "".join(current_chars).strip()
                if sentence:
                    sentences.append(sentence)
                current_chars = []
                # 跳过句间空白
                while lookahead < len(text) and text[lookahead].isspace():
                    lookahead += 1
                index = lookahead
                continue
            index += 1

        # 收尾：最后一段没有标点结尾的文本
        tail = "".join(current_chars).strip()
        if tail:
            sentences.append(tail)
        return sentences

    def _count_text_tokens(self, text_tokenizer, text: str) -> int:
        return len(self._encode_text(text_tokenizer, text))

    def _split_text_by_token_budget(
        self,
        text_tokenizer,
        text: str,
        max_tokens: int,
    ) -> list[str]:
        """把超长文本按 token 预算硬切分，优先在标点/空格处断开。

        先二分找到不超过预算的最长前缀，再在前缀尾部 ~25 字符内
        寻找最近的标点/空格作为切分点。
        """
        remaining_text = str(text).strip()
        if remaining_text == "":
            return []

        pieces: list[str] = []
        preferred_boundary_chars = _CLAUSE_SPLIT_PUNCTUATION | _SENTENCE_END_PUNCTUATION | frozenset({" "})
        while remaining_text:
            if self._count_text_tokens(text_tokenizer, remaining_text) <= int(max_tokens):
                pieces.append(remaining_text)
                break

            # 二分查找：满足 token 预算的最长前缀长度
            low = 1
            high = len(remaining_text)
            best_prefix_length = 1
            while low <= high:
                middle = (low + high) // 2
                candidate = remaining_text[:middle].strip()
                if not candidate:
                    low = middle + 1
                    continue
                if self._count_text_tokens(text_tokenizer, candidate) <= int(max_tokens):
                    best_prefix_length = middle
                    low = middle + 1
                else:
                    high = middle - 1

            # 在前缀末尾附近找标点/空格，尽量不在词中间切开
            cut_index = best_prefix_length
            prefix = remaining_text[:best_prefix_length]
            preferred_index = -1
            for scan_index in range(len(prefix) - 1, max(-1, len(prefix) - 25), -1):
                if prefix[scan_index] in preferred_boundary_chars:
                    preferred_index = scan_index + 1
                    break
            if preferred_index > 0:
                cut_index = preferred_index

            piece = remaining_text[:cut_index].strip()
            if not piece:
                piece = remaining_text[:best_prefix_length].strip()
                cut_index = best_prefix_length
            pieces.append(piece)
            remaining_text = remaining_text[cut_index:].strip()
        return pieces

    @staticmethod
    def _join_sentence_parts(left: str, right: str) -> str:
        """拼接两个句子片段：含 CJK 时直接相连，否则用空格分隔。"""
        if not left:
            return right
        if not right:
            return left
        if MossTTSNanoForCausalLM._contains_cjk(left) or MossTTSNanoForCausalLM._contains_cjk(right):
            return left + right
        return left + " " + right

    def _split_text_into_best_sentences(
        self,
        text_tokenizer,
        text: str,
        max_tokens: int,
    ) -> list[str]:
        """把长文本切成若干不超过 max_tokens 的分片。

        切分优先级：句子 > 子句 > 按 token 预算硬切；
        之后再贪心地把相邻小片段合并，减少分片数量。
        """
        if int(max_tokens) <= 0:
            return [str(text)]

        prepared_text = self._prepare_text_for_sentence_chunking(text)
        sentence_candidates = self._split_text_by_punctuation(prepared_text, punctuation=_SENTENCE_END_PUNCTUATION)
        if not sentence_candidates:
            sentence_candidates = [prepared_text.strip()]

        # 第一遍：把每个句子拆成满足 token 预算的 (token 数, 文本) 片段
        sentence_slices: list[tuple[int, str]] = []
        for sentence_text in sentence_candidates:
            normalized_sentence = sentence_text.strip()
            if not normalized_sentence:
                continue
            sentence_token_count = self._count_text_tokens(text_tokenizer, normalized_sentence)
            if sentence_token_count <= int(max_tokens):
                sentence_slices.append((sentence_token_count, normalized_sentence))
                continue

            # 句子超长：降级按子句（逗号等）切
            clause_candidates = self._split_text_by_punctuation(
                normalized_sentence,
                punctuation=_CLAUSE_SPLIT_PUNCTUATION,
            )
            if len(clause_candidates) <= 1:
                clause_candidates = [normalized_sentence]

            for clause_text in clause_candidates:
                normalized_clause = clause_text.strip()
                if not normalized_clause:
                    continue
                clause_token_count = self._count_text_tokens(text_tokenizer, normalized_clause)
                if clause_token_count <= int(max_tokens):
                    sentence_slices.append((clause_token_count, normalized_clause))
                    continue
                # 子句仍超长：按 token 预算硬切
                for piece in self._split_text_by_token_budget(
                    text_tokenizer=text_tokenizer,
                    text=normalized_clause,
                    max_tokens=max_tokens,
                ):
                    normalized_piece = piece.strip()
                    if normalized_piece:
                        sentence_slices.append(
                            (self._count_text_tokens(text_tokenizer, normalized_piece), normalized_piece)
                        )

        # 第二遍：贪心合并相邻片段，直到再加一个就会超预算
        chunks: list[str] = []
        current_chunk = ""
        current_chunk_token_count = 0
        for sentence_token_count, sentence_text in sentence_slices:
            if current_chunk == "":
                current_chunk = sentence_text
                current_chunk_token_count = sentence_token_count
                continue
            if current_chunk_token_count + sentence_token_count > int(max_tokens):
                chunks.append(current_chunk.strip())
                current_chunk = sentence_text
                current_chunk_token_count = sentence_token_count
            else:
                current_chunk = self._join_sentence_parts(current_chunk, sentence_text)
                current_chunk_token_count = self._count_text_tokens(text_tokenizer, current_chunk)

        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks or [prepared_text.strip()]

    # ------------------------------------------------------------------
    # 语音克隆的分片拼接与停顿估计
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_voice_clone_inter_chunk_pause_seconds(text_chunk: str) -> float:
        """根据分片长短估计其后的停顿时长：短句停顿更长，长句停顿更短。"""
        return (
            DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_SHORT_SECONDS
            if len(str(text_chunk).strip().split()) <= 4
            else DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_LONG_SECONDS
        )

    def _concat_voice_clone_waveform_chunks(
        self,
        waveform_chunks: list[torch.FloatTensor],
        text_chunks: list[str],
        sample_rate: int,
    ) -> torch.FloatTensor:
        """把多个分片波形拼接起来，相邻分片之间插入估计的静音停顿。"""
        if not waveform_chunks:
            return torch.zeros((1, 0), dtype=torch.float32)
        if len(waveform_chunks) != len(text_chunks):
            raise ValueError("waveform_chunks and text_chunks must have the same length.")
        if len(waveform_chunks) == 1:
            return waveform_chunks[0]

        segments: list[torch.FloatTensor] = []
        for chunk_index, waveform_chunk in enumerate(waveform_chunks):
            segments.append(waveform_chunk)
            if chunk_index >= len(waveform_chunks) - 1:
                continue
            pause_seconds = self._estimate_voice_clone_inter_chunk_pause_seconds(text_chunks[chunk_index])
            pause_samples = max(0, int(round(float(sample_rate) * pause_seconds)))
            if pause_samples > 0:
                silence = torch.zeros((waveform_chunk.shape[0], pause_samples), dtype=waveform_chunk.dtype)
                segments.append(silence)
        return torch.cat(segments, dim=-1)

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
        lead_seconds = MossTTSNanoForCausalLM._compute_stream_lead_seconds(
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
            return MossTTSNanoForCausalLM._extract_tensor_candidate(output[0])
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

        effective_nq = self._resolve_inference_nq(nq)
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
            "generation": MossTTSNanoGenerationOutput(audio_token_ids=audio_token_ids, prompt_input_ids=input_ids),
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
