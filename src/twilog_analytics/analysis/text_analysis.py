from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Optional

import polars as pl
from sudachipy import Dictionary, Tokenizer

__all__ = ["TextAnalyzer"]

JP_NOUN = "\u540d\u8a5e"
JP_VERB = "\u52d5\u8a5e"
JP_ADJ = "\u5f62\u5bb9\u8a5e"

_DEFAULT_POS = {JP_NOUN, JP_VERB, JP_ADJ}
_POS_ALIASES = {"noun": JP_NOUN, "verb": JP_VERB, "adj": JP_ADJ, "adjective": JP_ADJ}


def _normalize_pos_filter(pos_filter: Optional[str]) -> set[str]:
    if not pos_filter:
        return set(_DEFAULT_POS)
    parts = re.split(r"[,\s/]+", pos_filter.strip())
    allowed: set[str] = set()
    for part in parts:
        if not part:
            continue
        key = part.lower()
        allowed.add(_POS_ALIASES.get(key, part))
    return allowed


class TextAnalyzer:
    """SudachiPy based tokenizer and text analysis utilities."""

    def __init__(
        self,
        mode: str = "C",
        pos_filter: Optional[str] = None,
        stopwords: Optional[Iterable[str]] = None,
    ) -> None:
        self.tokenizer_obj = Dictionary(dict="full").create()
        self.set_mode(mode)
        self.allowed_pos = _normalize_pos_filter(pos_filter)
        self.stopwords = {w.strip() for w in (stopwords or []) if w and w.strip()}

    def set_mode(self, mode: str) -> None:
        if mode == "A":
            self.mode = Tokenizer.SplitMode.A
        elif mode == "B":
            self.mode = Tokenizer.SplitMode.B
        else:
            self.mode = Tokenizer.SplitMode.C

    def _is_valid_word(self, word: str, pos: str) -> bool:
        if len(word) <= 1:
            return False
        if self.allowed_pos and pos not in self.allowed_pos:
            return False
        if self.stopwords and word in self.stopwords:
            return False
        return True

    def extract_words_from_text(self, text: str) -> List[str]:
        if not text or not isinstance(text, str):
            return []
        words: List[str] = []
        tokens = self.tokenizer_obj.tokenize(text, self.mode)
        for token in tokens:
            word = token.surface()
            pos = token.part_of_speech()[0]
            if self._is_valid_word(word, pos):
                words.append(word)
        return words

    def extract_words(self, texts: Iterable[str]) -> List[str]:
        words: List[str] = []
        for text in texts:
            words.extend(self.extract_words_from_text(text))
        return words

    def get_word_frequency(self, df: pl.DataFrame, text_column: str = "text") -> Dict[str, int]:
        if text_column not in df.columns:
            return {}
        texts: List[str] = df[text_column].cast(pl.Utf8).fill_null("").to_list()
        words = self.extract_words(texts)
        word_freq: Dict[str, int] = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        return word_freq

    @staticmethod
    def get_top_words(word_freq: Dict[str, int], top_n: int = 50) -> pl.DataFrame:
        sorted_words: List[Tuple[str, int]] = sorted(
            word_freq.items(), key=lambda x: x[1], reverse=True
        )
        top_words = sorted_words[:top_n]
        if not top_words:
            return pl.DataFrame({"word": [], "count": []})
        words, counts = zip(*top_words)
        return pl.DataFrame({"word": list(words), "count": list(counts)})

    def get_tfidf_ranking(
        self, df: pl.DataFrame, text_column: str = "text", top_n: int = 50
    ) -> pl.DataFrame:
        if text_column not in df.columns:
            return pl.DataFrame({"word": [], "tf": [], "df": [], "score": []})

        texts: List[str] = df[text_column].cast(pl.Utf8).fill_null("").to_list()
        doc_count = max(len(texts), 1)
        tf: Dict[str, int] = {}
        df_counts: Dict[str, int] = {}

        for text in texts:
            terms = self.extract_words_from_text(text)
            if not terms:
                continue
            for term in terms:
                tf[term] = tf.get(term, 0) + 1
            for term in set(terms):
                df_counts[term] = df_counts.get(term, 0) + 1

        if not tf:
            return pl.DataFrame({"word": [], "tf": [], "df": [], "score": []})

        rows = []
        for term, tf_count in tf.items():
            df_count = df_counts.get(term, 0)
            idf = math.log((1 + doc_count) / (1 + df_count)) + 1
            score = round(tf_count * idf, 4)
            rows.append({"word": term, "tf": tf_count, "df": df_count, "score": score})

        rows.sort(key=lambda r: r["score"], reverse=True)
        return pl.DataFrame(rows[:top_n])
