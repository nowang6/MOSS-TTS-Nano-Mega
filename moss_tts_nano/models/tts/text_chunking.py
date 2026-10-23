# coding=utf-8
"""文本预处理与分片、语音克隆的分片拼接与停顿估计。

本文件是 ``MossTTSNanoForCausalLM`` 的 mixin：方法通过 ``self`` 访问模型其他部分，
类本身不单独实例化。
"""
from __future__ import annotations


import torch

from .constants import (
    DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_SHORT_SECONDS,
    DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_LONG_SECONDS,
    _SENTENCE_END_PUNCTUATION,
    _CLAUSE_SPLIT_PUNCTUATION,
    _CLOSING_PUNCTUATION,
)



class MossTTSNanoTextChunkingMixin:
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

        contains_cjk = MossTTSNanoTextChunkingMixin._contains_cjk(normalized_text)
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
        if MossTTSNanoTextChunkingMixin._contains_cjk(left) or MossTTSNanoTextChunkingMixin._contains_cjk(right):
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
