from __future__ import annotations

from typing import List, Sequence

from .configuration import MossTTSNanoConfig


# ---------------------------------------------------------------------------
# Prompt 模板常量
#
# MOSS-TTS-Nano 的输入是一段类似多轮对话的文本：
#
#   <|im_start|>user
#   <user_inst>
#   - Reference(s):
#   None
#   - Instruction:
#   None
#   ...（若干固定字段）...
#   - Text:
#   {要合成的文本}<|im_end|>
#   <|im_start|>assistant
#
# 下面的常量把这段模板拆成若干片段，方便在「参考音频」前后分别插入内容。
# ---------------------------------------------------------------------------

# user 角色行的开头（紧跟在 <|im_start|> 之后）
USER_ROLE_PREFIX = "user\n"

# <user_inst> 块中「参考音频占位」之前的部分
USER_TEMPLATE_REFERENCE_PREFIX = (
    "<user_inst>\n"
    "- Reference(s):\n"
)

# 参考音频之后、待合成文本之前的固定字段（全部填 None，仅推理使用）
USER_TEMPLATE_AFTER_REFERENCE = (
    "\n- Instruction:\nNone\n"
    "- Tokens:\nNone\n"
    "- Quality:\nNone\n"
    "- Sound Event:\nNone\n"
    "- Ambient Sound:\nNone\n"
    "- Language:\nNone\n"
    "- Text:\n"
)

# 没有参考音频时，Reference(s) 字段直接填 "None"
USER_TEMPLATE_PREFIX = USER_TEMPLATE_REFERENCE_PREFIX + "None" + USER_TEMPLATE_AFTER_REFERENCE

# <user_inst> 块的结束标记
USER_TEMPLATE_SUFFIX = "\n</user_inst>"

# user 回合结束（<|im_end|>）与 assistant 回合开始之间的换行
ASSISTANT_TURN_PREFIX = "\n"

# assistant 角色行的开头
ASSISTANT_ROLE_PREFIX = "assistant\n"


def encode_text(tokenizer, text: str) -> List[int]:
    """把文本编码为 token id 列表（不添加 BOS/EOS 等特殊 token）。

    部分 tokenizer 不支持 ``add_special_tokens`` 参数，此时退化为普通 encode。
    """
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


def decode_text(tokenizer, token_ids: Sequence[int]) -> str:
    """把 token id 列表解码回文本（保留特殊 token，不做空格清理）。

    逐层退化以兼容不同 tokenizer 实现的参数签名。
    """
    try:
        return str(
            tokenizer.decode(
                list(token_ids),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
    except TypeError:
        try:
            return str(tokenizer.decode(list(token_ids), skip_special_tokens=False))
        except TypeError:
            return str(tokenizer.decode(list(token_ids)))


def build_user_prompt_prefix(tokenizer, config: MossTTSNanoConfig) -> List[int]:
    """构造 prompt 开头：``<|im_start|>user\n<user_inst>\n- Reference(s):\n``。

    参考音频（audio token）会紧跟在这段 token 之后插入。
    """
    im_start = [config.im_start_token_id]
    role_tokens = encode_text(tokenizer, USER_ROLE_PREFIX)
    reference_header_tokens = encode_text(tokenizer, USER_TEMPLATE_REFERENCE_PREFIX)
    return im_start + role_tokens + reference_header_tokens


def build_user_prompt_after_reference(tokenizer) -> List[int]:
    """构造参考音频之后的固定字段部分（Instruction/Tokens/.../Text:）。"""
    return encode_text(tokenizer, USER_TEMPLATE_AFTER_REFERENCE)


def build_assistant_prompt_prefix(tokenizer, config: MossTTSNanoConfig) -> List[int]:
    """构造 user 回合收尾 + assistant 回合开头：

    ``\n</user_inst><|im_end|>\n<|im_start|>assistant\n``
    """
    user_suffix_tokens = encode_text(tokenizer, USER_TEMPLATE_SUFFIX)
    im_end = [config.im_end_token_id]
    turn_break_tokens = encode_text(tokenizer, ASSISTANT_TURN_PREFIX)
    im_start = [config.im_start_token_id]
    assistant_role_tokens = encode_text(tokenizer, ASSISTANT_ROLE_PREFIX)
    return user_suffix_tokens + im_end + turn_break_tokens + im_start + assistant_role_tokens


def build_prompt_prefix(tokenizer, config: MossTTSNanoConfig) -> List[int]:
    """构造完整 user prompt 的前半段（无参考音频，Reference(s) 填 None）。

    返回内容截止到 ``- Text:\n``，待合成文本的 token 紧随其后。
    """
    return (
        build_user_prompt_prefix(tokenizer, config)
        + encode_text(tokenizer, "None")
        + build_user_prompt_after_reference(tokenizer)
    )


def build_prompt_suffix(tokenizer, config: MossTTSNanoConfig) -> List[int]:
    """构造 prompt 的后半段：user 回合收尾 + assistant 回合开头。"""
    return build_assistant_prompt_prefix(tokenizer, config)


def build_prompt_token_ids(
    tokenizer,
    config: MossTTSNanoConfig,
    text_token_ids: Sequence[int],
) -> List[int]:
    """拼装完整 prompt：前缀 + 待合成文本 token + 后缀。"""
    prefix_ids = build_prompt_prefix(tokenizer, config)
    body_ids = [int(token_id) for token_id in text_token_ids]
    suffix_ids = build_prompt_suffix(tokenizer, config)
    return prefix_ids + body_ids + suffix_ids
