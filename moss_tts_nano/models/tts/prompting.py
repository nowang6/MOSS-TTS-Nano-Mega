from __future__ import annotations

from typing import List, Sequence

from .configuration_moss_tts_nano import MossTTSNanoConfig


USER_ROLE_PREFIX = "user\n"
USER_TEMPLATE_REFERENCE_PREFIX = (
    "<user_inst>\n"
    "- Reference(s):\n"
)
USER_TEMPLATE_AFTER_REFERENCE = (
    "\n- Instruction:\nNone\n"
    "- Tokens:\nNone\n"
    "- Quality:\nNone\n"
    "- Sound Event:\nNone\n"
    "- Ambient Sound:\nNone\n"
    "- Language:\nNone\n"
    "- Text:\n"
)
USER_TEMPLATE_PREFIX = USER_TEMPLATE_REFERENCE_PREFIX + "None" + USER_TEMPLATE_AFTER_REFERENCE
USER_TEMPLATE_SUFFIX = "\n</user_inst>"
ASSISTANT_TURN_PREFIX = "\n"
ASSISTANT_ROLE_PREFIX = "assistant\n"


def encode_text(tokenizer, text: str) -> List[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


def decode_text(tokenizer, token_ids: Sequence[int]) -> str:
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
    return [config.im_start_token_id] + encode_text(tokenizer, USER_ROLE_PREFIX) + encode_text(
        tokenizer,
        USER_TEMPLATE_REFERENCE_PREFIX,
    )


def build_user_prompt_after_reference(tokenizer) -> List[int]:
    return encode_text(tokenizer, USER_TEMPLATE_AFTER_REFERENCE)


def build_assistant_prompt_prefix(tokenizer, config: MossTTSNanoConfig) -> List[int]:
    return encode_text(tokenizer, USER_TEMPLATE_SUFFIX) + [config.im_end_token_id] + encode_text(
        tokenizer,
        ASSISTANT_TURN_PREFIX,
    ) + [config.im_start_token_id] + encode_text(
        tokenizer,
        ASSISTANT_ROLE_PREFIX,
    )


def build_prompt_prefix(tokenizer, config: MossTTSNanoConfig) -> List[int]:
    return (
        build_user_prompt_prefix(tokenizer, config)
        + encode_text(tokenizer, "None")
        + build_user_prompt_after_reference(tokenizer)
    )


def build_prompt_suffix(tokenizer, config: MossTTSNanoConfig) -> List[int]:
    return build_assistant_prompt_prefix(tokenizer, config)


def build_prompt_token_ids(
    tokenizer,
    config: MossTTSNanoConfig,
    text_token_ids: Sequence[int],
) -> List[int]:
    return build_prompt_prefix(tokenizer, config) + [int(token_id) for token_id in text_token_ids] + build_prompt_suffix(
        tokenizer,
        config,
    )
