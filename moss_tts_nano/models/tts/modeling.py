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

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

from .audio_codec import MossTTSNanoAudioCodecMixin
from .configuration import MossTTSNanoConfig
from .constants import (
    DEFAULT_MOSS_AUDIO_TOKENIZER_PRETRAINED_NAME_OR_PATH,
    DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_LONG_SECONDS,
    DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_SHORT_SECONDS,
    DEFAULT_VOICE_CLONE_MAX_MEMORY_PER_SAMPLE_GB,
    DEFAULT_VOICE_CLONE_MAX_TEXT_TOKENS,
    MOSS_AUDIO_TOKENIZER_NANO_TYPE,
)
from .generation import MossTTSNanoGenerationMixin
from .gpt2_decoder import MossTTSNanoGPT2Model
from .inference_api import MossTTSNanoInferenceMixin
from .inference_inputs import MossTTSNanoInferenceInputsMixin
from .text_chunking import MossTTSNanoTextChunkingMixin
from .tokenizer_loading import MossTTSNanoTokenizerLoadingMixin

__all__ = [
    "MossTTSNanoForCausalLM",
    "MossTTSNanoGenerationOutput",
    "MossTTSNanoOutput",
    "MossTTSNanoPreTrainedModel",
    "MOSS_AUDIO_TOKENIZER_NANO_TYPE",
    "DEFAULT_MOSS_AUDIO_TOKENIZER_PRETRAINED_NAME_OR_PATH",
    "DEFAULT_VOICE_CLONE_MAX_TEXT_TOKENS",
    "DEFAULT_VOICE_CLONE_MAX_MEMORY_PER_SAMPLE_GB",
    "DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_SHORT_SECONDS",
    "DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_LONG_SECONDS",
]



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


class MossTTSNanoPreTrainedModel(PreTrainedModel):
    """HF 集成基类：声明配置类与权重切分/注意力后端支持。"""

    config_class = MossTTSNanoConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = False
    _no_split_modules = ["MossTTSNanoGPT2Block"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True


class MossTTSNanoForCausalLM(
    MossTTSNanoPreTrainedModel,
    MossTTSNanoTextChunkingMixin,
    MossTTSNanoInferenceInputsMixin,
    MossTTSNanoTokenizerLoadingMixin,
    MossTTSNanoAudioCodecMixin,
    MossTTSNanoGenerationMixin,
    MossTTSNanoInferenceMixin,
):
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
