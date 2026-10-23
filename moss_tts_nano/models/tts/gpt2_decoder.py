# coding=utf-8
"""MOSS-TTS-Nano 的 GPT-2 解码器实现。

在 HF GPT-2 的基础上做了两点扩展：
1. 支持 RoPE 旋转位置编码（``position_embedding_type="rope"``），
   替代 GPT-2 原本的绝对位置 embedding；
2. 支持 flash_attention_2 / sdpa / eager 三种注意力后端，
   并支持把多条样本「打包」（packing）成变长序列做 flash attention。

注意：属性名（wte/wpe/c_attn/c_proj/ln_1/ln_2/h/ln_f 等）与 GPT-2
checkpoint 的权重键一一对应，不能随意改名。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input

    _FLASH_ATTN_AVAILABLE = True
except Exception:
    flash_attn_func = None
    flash_attn_varlen_func = None
    pad_input = None
    unpad_input = None
    _FLASH_ATTN_AVAILABLE = False


@dataclass
class PackedSequenceMetadata:
    """变长打包序列的元信息（供 flash_attn_varlen_func 使用）。

    - ``cu_seqlens``：各子序列在打包后序列中的累计边界，形如 [0, l1, l1+l2, ...]；
    - ``max_seqlen``：最长子序列的长度；
    - ``indices``：打包后的 token 在原 [batch, seq] 张量中的扁平下标；
    - ``batch_size`` / ``seq_len``：打包前的原始形状，用于解包（pad_input）。
    """

    cu_seqlens: torch.Tensor
    max_seqlen: int
    indices: Optional[torch.Tensor] = None
    batch_size: Optional[int] = None
    seq_len: Optional[int] = None


class MossTTSNanoGPT2RotaryEmbedding(nn.Module):
    """RoPE 旋转位置编码，预计算各频率的 cos/sin 表。"""

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE head_dim must be even, got {dim}")
        # inv_freq[i] = base^(-2i/dim)，即每对相邻维度的旋转角频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        position_ids: torch.LongTensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """根据位置 id 计算 cos/sin，返回形状均为 [batch, seq, 1, dim]。"""
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        # freqs[b, s, i] = position[b, s] * inv_freq[i]
        freqs = torch.einsum(
            "bs,d->bsd",
            position_ids.to(device=device, dtype=self.inv_freq.dtype),
            self.inv_freq,
        )
        # repeat_interleave(2)：把 dim/2 个频率复制成 dim 份，与 (偶数位, 奇数位) 一一对应
        cos = freqs.cos().repeat_interleave(2, dim=-1).unsqueeze(2).to(dtype=dtype)
        sin = freqs.sin().repeat_interleave(2, dim=-1).unsqueeze(2).to(dtype=dtype)
        return cos, sin


def rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    """把相邻的 (偶数位, 奇数位) 对旋转为 (-奇数位, 偶数位)，即复数乘法中的乘以 i。"""
    even = hidden_states[..., ::2]
    odd = hidden_states[..., 1::2]
    return torch.stack((-odd, even), dim=-1).reshape_as(hidden_states)


def apply_rotary_pos_emb(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """对 q/k 应用旋转位置编码：x*cos + rotate_half(x)*sin。"""
    return (hidden_states * cos) + (rotate_half(hidden_states) * sin)


class MossTTSNanoGPT2MLP(nn.Module):
    """GPT-2 的前馈网络：Linear -> 激活 -> Linear -> Dropout。"""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        # GPT-2 默认中间层宽度为 4 倍隐藏层
        inner_size = int(config.n_inner or 4 * hidden_size)
        self.fc_in = nn.Linear(hidden_size, inner_size)
        self.fc_out = nn.Linear(inner_size, hidden_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        return self.dropout(hidden_states)


class MossTTSNanoGPT2Attention(nn.Module):
    """GPT-2 因果自注意力，支持 RoPE 与多种注意力后端。"""

    def __init__(self, config: GPT2Config, layer_idx: int, attn_implementation: str) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        num_heads = int(config.num_attention_heads)
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_attention_heads={num_heads}")

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.embed_dim = hidden_size
        self.layer_idx = layer_idx
        self.attn_implementation = attn_implementation
        self.attn_dropout = float(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.scale_attn_weights = bool(getattr(config, "scale_attn_weights", True))
        self.scale_attn_by_inverse_layer_idx = bool(getattr(config, "scale_attn_by_inverse_layer_idx", False))
        self.position_embedding_type = str(getattr(config, "position_embedding_type", "absolute")).lower()
        if self.position_embedding_type not in {"absolute", "rope"}:
            raise ValueError(f"Unsupported position_embedding_type={self.position_embedding_type!r}")

        # QKV 合并投影与输出投影（名称与 GPT-2 checkpoint 对齐）
        self.c_attn = nn.Linear(hidden_size, 3 * hidden_size)
        self.c_proj = nn.Linear(hidden_size, hidden_size)
        self.rotary_emb = None
        if self.position_embedding_type == "rope":
            self.rotary_emb = MossTTSNanoGPT2RotaryEmbedding(
                self.head_dim,
                base=float(getattr(config, "rope_base", 10000.0)),
            )

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """把最后一维 hidden_size 拆成 (num_heads, head_dim)。

        3D 输入 [batch, seq, hidden] -> [batch, seq, heads, head_dim]；
        2D 输入 [total_tokens, hidden] -> [total_tokens, heads, head_dim]（打包序列）。
        """
        if tensor.ndim == 3:
            batch_size, seq_len, _ = tensor.shape
            return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
        if tensor.ndim == 2:
            total_tokens, _ = tensor.shape
            return tensor.view(total_tokens, self.num_heads, self.head_dim)
        raise ValueError(f"Unsupported tensor rank for attention split: {tensor.ndim}")

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """_split_heads 的逆操作：把 (num_heads, head_dim) 合并回 hidden_size。"""
        if tensor.ndim == 4:
            batch_size, seq_len, _, _ = tensor.shape
            return tensor.reshape(batch_size, seq_len, self.embed_dim)
        if tensor.ndim == 3:
            total_tokens, _, _ = tensor.shape
            return tensor.reshape(total_tokens, self.embed_dim)
        raise ValueError(f"Unsupported tensor rank for attention merge: {tensor.ndim}")

    def _causal_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        query_length: int,
        key_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """构造布尔因果掩码 [1, 1, query_length, key_length]，并与 padding 掩码取交集。

        解码（带 KV cache）时 query_length < key_length，此时 query 的绝对位置
        从 key_length - query_length 开始，保证最后一个 query 能看到全部 key。
        """
        query_positions = torch.arange(query_length, device=device, dtype=torch.long)
        query_positions = query_positions + max(key_length - query_length, 0)
        key_positions = torch.arange(key_length, device=device, dtype=torch.long)
        causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        if attention_mask is None:
            return causal_mask
        key_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
        return causal_mask & key_mask

    def _eager_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """标准（逐算子）注意力实现，便于调试与 CPU 推理。"""
        # [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        scale = 1.0
        if self.scale_attn_weights:
            scale /= self.head_dim ** 0.5
        if self.scale_attn_by_inverse_layer_idx:
            scale /= float(self.layer_idx + 1)

        scores = torch.matmul(query, key.transpose(-1, -2)) * scale
        causal_mask = self._causal_attention_mask(
            attention_mask=attention_mask,
            query_length=query.shape[-2],
            key_length=key.shape[-2],
            device=query.device,
        )
        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        if self.training and self.attn_dropout > 0:
            probs = torch.dropout(probs, self.attn_dropout, train=True)
        output = torch.matmul(probs, value)
        return output.transpose(1, 2).contiguous()

    def _sdpa_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """PyTorch SDPA（scaled_dot_product_attention）实现。"""
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        mask = None
        query_attention_mask = None
        if attention_mask is not None:
            query_length = query.shape[-2]
            key_length = key.shape[-2]
            mask = self._causal_attention_mask(
                attention_mask=attention_mask,
                query_length=query_length,
                key_length=key_length,
                device=query.device,
            )
            query_attention_mask = attention_mask[:, -query_length:].to(dtype=torch.bool, device=query.device)
            if not bool(query_attention_mask.all()):
                # SDPA 在「某一行 query 全部被 mask」时会产生 NaN。
                # 对被 padding 的 query 行，保留一个与其对齐的 key 可见，
                # 注意力算完后再把该行输出清零。
                mask = mask.expand(query.shape[0], -1, -1, -1).clone()
                invalid_batch, invalid_query = torch.nonzero(~query_attention_mask, as_tuple=True)
                aligned_key = invalid_query + max(key_length - query_length, 0)
                mask[invalid_batch, :, invalid_query, aligned_key] = True
        output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=mask is None,
        )
        if query_attention_mask is not None and not bool(query_attention_mask.all()):
            output = output.masked_fill(~query_attention_mask[:, None, :, None], 0.0)
        return output.transpose(1, 2).contiguous()

    def _flash_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        packed_metadata: Optional[PackedSequenceMetadata],
    ) -> torch.Tensor:
        """FlashAttention-2 实现（仅 CUDA + fp16/bf16），支持打包变长序列。"""
        if not _FLASH_ATTN_AVAILABLE:
            raise ImportError("flash_attn is not installed, but attn_implementation='flash_attention_2' was requested.")
        if query.device.type != "cuda":
            raise ValueError("flash_attention_2 requires CUDA tensors.")
        if query.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                f"flash_attention_2 requires fp16/bf16 tensors, but received dtype={query.dtype}."
            )

        dropout_p = self.attn_dropout if self.training else 0.0
        if packed_metadata is not None:
            # 打包模式：q/k/v 是 [total_tokens, heads, head_dim] 的紧凑布局
            if packed_metadata.indices is not None:
                query = query.reshape(-1, self.num_heads, self.head_dim).index_select(0, packed_metadata.indices)
                key = key.reshape(-1, self.num_heads, self.head_dim).index_select(0, packed_metadata.indices)
                value = value.reshape(-1, self.num_heads, self.head_dim).index_select(0, packed_metadata.indices)
            output = flash_attn_varlen_func(
                query,
                key,
                value,
                packed_metadata.cu_seqlens,
                packed_metadata.cu_seqlens,
                packed_metadata.max_seqlen,
                packed_metadata.max_seqlen,
                dropout_p=dropout_p,
                causal=True,
            )
            if packed_metadata.indices is None:
                return output
            # 解包回 [batch, seq, heads, head_dim] 的带 padding 布局
            return pad_input(
                output,
                packed_metadata.indices,
                packed_metadata.batch_size,
                packed_metadata.seq_len,
            )

        # 无 padding 的整齐 batch，直接调用定长接口
        if attention_mask is None or bool(attention_mask.all()):
            return flash_attn_func(
                query,
                key,
                value,
                dropout_p=dropout_p,
                causal=True,
            )

        # 有 padding：先去掉 padding 走变长接口，再补回来
        unpadded_query, indices, cu_seqlens, max_seqlen, _ = unpad_input(query, attention_mask)
        unpadded_key, _, _, _, _ = unpad_input(key, attention_mask)
        unpadded_value, _, _, _, _ = unpad_input(value, attention_mask)
        output = flash_attn_varlen_func(
            unpadded_query,
            unpadded_key,
            unpadded_value,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            dropout_p=dropout_p,
            causal=True,
        )
        return pad_input(output, indices, query.shape[0], query.shape[1])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        packed_metadata: Optional[PackedSequenceMetadata] = None,
        layer_past: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        # 一次性算出 Q/K/V 再切分
        qkv = self.c_attn(hidden_states)
        query, key, value = qkv.split(self.embed_dim, dim=-1)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        if self.rotary_emb is not None:
            if position_ids is None:
                raise ValueError("position_ids must be provided when position_embedding_type='rope'.")
            cos, sin = self.rotary_emb(
                position_ids.to(device=query.device),
                device=query.device,
                dtype=query.dtype,
            )
            query = apply_rotary_pos_emb(query, cos, sin)
            key = apply_rotary_pos_emb(key, cos, sin)

        # 拼接 KV cache（增量解码时 layer_past 非空）
        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat([past_key.to(device=key.device, dtype=key.dtype), key], dim=1)
            value = torch.cat([past_value.to(device=value.device, dtype=value.dtype), value], dim=1)

        present = (key, value) if use_cache else None

        # KV cache 模式下逐 token 解码，flash_attention_2 反而没有优势，走 sdpa/eager
        if self.attn_implementation == "flash_attention_2" and layer_past is None:
            attn_output = self._flash_attention(
                query=query,
                key=key,
                value=value,
                attention_mask=attention_mask,
                packed_metadata=packed_metadata,
            )
        elif self.attn_implementation == "sdpa":
            attn_output = self._sdpa_attention(
                query=query,
                key=key,
                value=value,
                attention_mask=attention_mask,
            )
        else:
            attn_output = self._eager_attention(
                query=query,
                key=key,
                value=value,
                attention_mask=attention_mask,
            )

        attn_output = self._merge_heads(attn_output)
        attn_output = self.c_proj(attn_output)
        return self.resid_dropout(attn_output), present


class MossTTSNanoGPT2Block(nn.Module):
    """单层 GPT-2 block：LayerNorm -> 注意力 -> 残差 -> LayerNorm -> MLP -> 残差。"""

    def __init__(self, config: GPT2Config, layer_idx: int, attn_implementation: str) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = MossTTSNanoGPT2Attention(config, layer_idx=layer_idx, attn_implementation=attn_implementation)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = MossTTSNanoGPT2MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        packed_metadata: Optional[PackedSequenceMetadata] = None,
        layer_past: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        attn_output, present = self.attn(
            self.ln_1(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            packed_metadata=packed_metadata,
            layer_past=layer_past,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))
        return hidden_states, present


class MossTTSNanoGPT2Model(nn.Module):
    """GPT-2 主干网络（embedding + N 层 block + 最终 LayerNorm）。

    本模型只接受 ``inputs_embeds``（由外层 TTS 模型把文本 embedding 与
    多通道音频 embedding 相加后传入），不直接使用 ``input_ids``。
    """

    def __init__(self, config: GPT2Config, attn_implementation: str = "eager") -> None:
        super().__init__()
        self.config = config
        self.attn_implementation = attn_implementation
        self.position_embedding_type = str(getattr(config, "position_embedding_type", "absolute")).lower()
        if self.position_embedding_type not in {"absolute", "rope"}:
            raise ValueError(f"Unsupported position_embedding_type={self.position_embedding_type!r}")
        hidden_size = int(config.hidden_size)
        # wte：词 embedding；wpe：绝对位置 embedding（RoPE 模式下为 Identity）
        self.wte = nn.Embedding(config.vocab_size, hidden_size)
        self.wpe = nn.Embedding(config.n_positions, hidden_size) if self.position_embedding_type == "absolute" else nn.Identity()
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [MossTTSNanoGPT2Block(config, layer_idx=index, attn_implementation=attn_implementation) for index in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.gradient_checkpointing = False
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """按 GPT-2 惯例初始化权重：Linear/Embedding ~ N(0, initializer_range)。"""
        init_std = float(self.config.initializer_range)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _normalize_num_sequences(
        cu_seqlens: torch.Tensor,
        num_sequences: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """把 num_sequences 统一成 [batch] 的张量。

        未提供时从 cu_seqlens 推导：每个 batch 内长度大于 0 的子序列个数。
        """
        if cu_seqlens.ndim == 1:
            cu_seqlens = cu_seqlens.unsqueeze(0)
        if num_sequences is None:
            counts = []
            for boundary in cu_seqlens:
                segment_lengths = boundary[1:] - boundary[:-1]
                counts.append(int((segment_lengths > 0).sum().item()))
            return torch.tensor(counts, dtype=torch.int32, device=device)
        if num_sequences.ndim == 0:
            return num_sequences.unsqueeze(0)
        return num_sequences

    @staticmethod
    def build_packed_position_ids(
        attention_mask: Optional[torch.Tensor],
        cu_seqlens: torch.Tensor,
        num_sequences: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """为打包序列构造 position_ids：每条子序列内部从 0 重新计数。"""
        if cu_seqlens.ndim == 1:
            cu_seqlens = cu_seqlens.unsqueeze(0)
        batch_size, seq_len = cu_seqlens.shape[0], cu_seqlens.shape[1] - 1
        device = cu_seqlens.device
        position_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        counts = MossTTSNanoGPT2Model._normalize_num_sequences(cu_seqlens, num_sequences, device=device)
        for batch_index in range(batch_size):
            sequence_count = int(counts[batch_index].item())
            boundaries = cu_seqlens[batch_index, : sequence_count + 1].tolist()
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                start = int(start)
                end = int(end)
                if end > start:
                    position_ids[batch_index, start:end] = torch.arange(end - start, device=device)
        if attention_mask is not None:
            position_ids = position_ids * attention_mask.to(dtype=position_ids.dtype)
        return position_ids

    @staticmethod
    def build_packed_metadata(
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        num_sequences: Optional[torch.Tensor],
    ) -> PackedSequenceMetadata:
        """把 [batch, seq] 布局按 cu_seqlens 打包成变长序列的元信息。

        遍历每个 batch 中的每条非空子序列，记录其 token 在扁平化后的下标，
        并计算打包后的 cu_seqlens 与 max_seqlen。
        """
        if cu_seqlens.ndim == 1:
            cu_seqlens = cu_seqlens.unsqueeze(0)
        device = hidden_states.device
        counts = MossTTSNanoGPT2Model._normalize_num_sequences(cu_seqlens, num_sequences, device=device)
        flat_indices = []
        cumulative = [0]
        max_seqlen = 0
        seq_len = hidden_states.shape[1]

        for batch_index in range(hidden_states.shape[0]):
            sequence_count = int(counts[batch_index].item())
            boundaries = cu_seqlens[batch_index, : sequence_count + 1].tolist()
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                start = int(start)
                end = int(end)
                if end <= start:
                    continue
                segment_indices = batch_index * seq_len + torch.arange(start, end, device=device)
                flat_indices.append(segment_indices)
                cumulative.append(cumulative[-1] + (end - start))
                max_seqlen = max(max_seqlen, end - start)

        if not flat_indices:
            raise ValueError("cu_seqlens did not describe any non-empty packed sequences.")

        indices = torch.cat(flat_indices, dim=0)
        return PackedSequenceMetadata(
            cu_seqlens=torch.tensor(cumulative, dtype=torch.int32, device=device),
            max_seqlen=max_seqlen,
            indices=indices,
            batch_size=hidden_states.shape[0],
            seq_len=hidden_states.shape[1],
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple[tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: bool = True,
        cu_seqlens: Optional[torch.Tensor] = None,
        num_sequences: Optional[torch.Tensor] = None,
    ) -> BaseModelOutputWithPast:
        del input_ids, output_attentions
        if inputs_embeds is None:
            raise ValueError("inputs_embeds must be provided.")

        use_cache = bool(use_cache)
        if use_cache and cu_seqlens is not None:
            raise ValueError("use_cache=True is not supported together with cu_seqlens packing.")

        hidden_states = inputs_embeds
        if attention_mask is None:
            attention_mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        else:
            attention_mask = attention_mask.to(dtype=torch.bool, device=hidden_states.device)
        # 增量解码时 attention_mask 比当前输入长，只取末尾与当前输入对齐的部分
        query_attention_mask = attention_mask[:, -hidden_states.shape[1] :]

        packed_metadata = None
        if position_ids is None:
            if cu_seqlens is not None:
                # 打包模式：每条子序列内部从 0 开始计数
                position_ids = self.build_packed_position_ids(
                    attention_mask=attention_mask,
                    cu_seqlens=cu_seqlens.to(device=hidden_states.device),
                    num_sequences=num_sequences.to(device=hidden_states.device) if num_sequences is not None else None,
                )
            elif attention_mask is not None:
                # 常规模式：用 cumsum 让 padding 位置的 position 保持为 0
                position_ids = attention_mask.long().cumsum(dim=-1) - 1
                position_ids = position_ids.masked_fill(~attention_mask, 0)
                position_ids = position_ids[:, -hidden_states.shape[1] :]
            else:
                # 无掩码：位置 = 已有缓存长度 + 当前偏移
                past_length = 0
                if past_key_values is not None and len(past_key_values) > 0:
                    past_length = past_key_values[0][0].shape[1]
                position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device, dtype=torch.long)
                position_ids = position_ids + past_length
                position_ids = position_ids.unsqueeze(0).expand(hidden_states.shape[0], -1)

        if cu_seqlens is not None and self.attn_implementation == "flash_attention_2":
            packed_metadata = self.build_packed_metadata(
                hidden_states=hidden_states,
                cu_seqlens=cu_seqlens.to(device=hidden_states.device),
                num_sequences=num_sequences.to(device=hidden_states.device) if num_sequences is not None else None,
            )

        if self.position_embedding_type == "absolute":
            hidden_states = hidden_states + self.wpe(position_ids)
        hidden_states = self.drop(hidden_states)
        # 把 padding 位置的 embedding 清零，避免污染后续层
        hidden_states = hidden_states * query_attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)

        all_hidden_states = () if output_hidden_states else None
        presents = [] if use_cache else None
        for layer_index, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    raise ValueError("use_cache=True is not supported when gradient checkpointing is enabled during training.")

                def custom_forward(*inputs):
                    output, _ = block(
                        inputs[0],
                        attention_mask=inputs[1],
                        position_ids=inputs[2],
                        packed_metadata=packed_metadata,
                        layer_past=None,
                        use_cache=False,
                    )
                    return output

                hidden_states = torch.utils.checkpoint.checkpoint(
                    custom_forward,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    use_reentrant=False,
                )
                present = None
            else:
                hidden_states, present = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    packed_metadata=packed_metadata,
                    layer_past=None if past_key_values is None else past_key_values[layer_index],
                    use_cache=use_cache,
                )
            # 每层之后都把 padding 位置重新清零
            hidden_states = hidden_states * query_attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
            if presents is not None:
                presents.append(present)

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states * query_attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return (hidden_states, tuple(presents) if presents is not None else None, all_hidden_states, None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=tuple(presents) if presents is not None else None,
            hidden_states=all_hidden_states,
            attentions=None,
        )
