from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PROMPT_AUDIO_DIR = REPO_ROOT / "assets" / "audio"
DEFAULT_PROMPT_AUDIO_PATH = DEFAULT_PROMPT_AUDIO_DIR / "zh_1.wav"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "generated_audio"
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "weights" / "MOSS-TTS-Nano-100M"
DEFAULT_AUDIO_TOKENIZER_PATH = REPO_ROOT / "weights" / "MOSS-Audio-Tokenizer-Nano"
