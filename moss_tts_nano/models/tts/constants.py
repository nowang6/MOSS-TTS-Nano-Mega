# coding=utf-8
"""MOSS-TTS-Nano 模型共享常量（被 modeling 与各 mixin 模块引用，避免循环导入）。"""


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
