from .configuration_moss_tts_nano import MossTTSNanoConfig
from .modeling_moss_tts_nano import (
    MossTTSNanoForCausalLM,
    MossTTSNanoGenerationOutput,
    MossTTSNanoOutput,
)
from .tokenization_moss_tts_nano import MossTTSNanoSentencePieceTokenizer

try:
    MossTTSNanoConfig.register_for_auto_class()
except Exception:
    pass

for auto_class_name in ("AutoModel", "AutoModelForCausalLM"):
    try:
        MossTTSNanoForCausalLM.register_for_auto_class(auto_class_name)
    except Exception:
        pass

try:
    MossTTSNanoSentencePieceTokenizer.register_for_auto_class("AutoTokenizer")
except Exception:
    pass

__all__ = [
    "MossTTSNanoConfig",
    "MossTTSNanoForCausalLM",
    "MossTTSNanoSentencePieceTokenizer",
    "MossTTSNanoGenerationOutput",
    "MossTTSNanoOutput",
]
