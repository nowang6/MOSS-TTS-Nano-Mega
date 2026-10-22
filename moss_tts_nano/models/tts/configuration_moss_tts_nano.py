# coding=utf-8
from typing import Any, Dict, Optional, Union

from transformers.configuration_utils import PretrainedConfig
from transformers.models.gpt2.configuration_gpt2 import GPT2Config


class MossTTSNanoConfig(PretrainedConfig):
    model_type = "moss_tts_nano"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        gpt2_config: Optional[Union[GPT2Config, Dict[str, Any]]] = None,
        n_vq: int = 8,
        audio_vocab_size: Optional[int] = 1024,
        audio_codebook_sizes: Optional[list[int]] = None,
        audio_pad_token_id: int = 1024,
        pad_token_id: int = 151643,
        im_start_token_id: int = 151644,
        im_end_token_id: int = 151645,
        audio_start_token_id: int = 151652,
        audio_end_token_id: int = 151653,
        audio_user_slot_token_id: int = 151654,
        audio_assistant_slot_token_id: int = 151656,
        tokenizer_use_fast: bool = False,
        audio_tokenizer_type: str = "moss-audio-tokenizer-nano",
        audio_tokenizer_pretrained_name_or_path: Optional[str] = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano",
        audio_tokenizer_sample_rate: int = 48000,
        attn_implementation: str = "flash_attention_2",
        initializer_range: float = 0.02,
        model_architecture: str = "global_local_transformer",
        local_transformer_layers: int = 4,
        local_transformer_attn_implementation: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(gpt2_config, dict):
            self.gpt2_config = GPT2Config(**gpt2_config)
        elif gpt2_config is None:
            self.gpt2_config = GPT2Config()
        else:
            self.gpt2_config = gpt2_config

        self.n_vq = int(n_vq)
        if audio_codebook_sizes is None:
            if audio_vocab_size is None:
                raise ValueError("audio_vocab_size must be set when audio_codebook_sizes is not provided.")
            resolved_audio_codebook_sizes = [int(audio_vocab_size)] * self.n_vq
        else:
            resolved_audio_codebook_sizes = [int(codebook_size) for codebook_size in audio_codebook_sizes]
        if len(resolved_audio_codebook_sizes) != self.n_vq:
            raise ValueError(
                "audio_codebook_sizes must have length n_vq "
                f"(expected {self.n_vq}, got {len(resolved_audio_codebook_sizes)})."
            )
        if any(codebook_size <= 0 for codebook_size in resolved_audio_codebook_sizes):
            raise ValueError("audio_codebook_sizes must contain positive integers.")

        max_audio_codebook_size = max(resolved_audio_codebook_sizes)
        if audio_vocab_size is not None and int(audio_vocab_size) < max_audio_codebook_size:
            raise ValueError(
                "audio_vocab_size must be >= max(audio_codebook_sizes) "
                f"(got {audio_vocab_size}, expected at least {max_audio_codebook_size})."
            )

        self.audio_codebook_sizes = resolved_audio_codebook_sizes
        self.audio_vocab_size = (
            max_audio_codebook_size if audio_vocab_size is None else int(audio_vocab_size)
        )
        self.audio_pad_token_id = int(audio_pad_token_id)
        if self.audio_pad_token_id < max_audio_codebook_size:
            raise ValueError(
                "audio_pad_token_id must be >= max(audio_codebook_sizes) so pad stays outside every codebook "
                f"(got {self.audio_pad_token_id}, max codebook size {max_audio_codebook_size})."
            )
        self.pad_token_id = pad_token_id
        self.im_start_token_id = im_start_token_id
        self.im_end_token_id = im_end_token_id
        self.audio_start_token_id = audio_start_token_id
        self.audio_end_token_id = audio_end_token_id
        self.audio_user_slot_token_id = audio_user_slot_token_id
        self.audio_assistant_slot_token_id = audio_assistant_slot_token_id
        self.tokenizer_use_fast = tokenizer_use_fast
        self.audio_tokenizer_type = audio_tokenizer_type
        self.audio_tokenizer_pretrained_name_or_path = audio_tokenizer_pretrained_name_or_path
        self.audio_tokenizer_sample_rate = audio_tokenizer_sample_rate
        self.attn_implementation = attn_implementation
        self.initializer_range = initializer_range
        self.model_architecture = model_architecture
        self.local_transformer_layers = local_transformer_layers
        self.local_transformer_attn_implementation = (
            attn_implementation
            if local_transformer_attn_implementation is None
            else local_transformer_attn_implementation
        )
        self.vocab_size = self.gpt2_config.vocab_size
        self.hidden_size = self.gpt2_config.hidden_size
        self.max_position_embeddings = self.gpt2_config.n_positions

        super().__init__(pad_token_id=pad_token_id, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        output = super().to_dict()
        output["gpt2_config"] = self.gpt2_config.to_dict()
        return output


__all__ = ["MossTTSNanoConfig"]
