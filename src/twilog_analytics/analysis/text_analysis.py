from __future__ import annotations

import math
import re
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Tuple

import polars as pl
from sudachipy import Dictionary, Tokenizer

__all__ = [
    "TextAnalyzer",
    "hashtag_ranking",
    "hashtag_cooccurrence",
    "parse_keyword_dictionary",
    "keyword_category_counts",
    "word_monthly_counts",
]

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
    def get_top_words(word_freq: Dict[str, int], top_n: int | None = 50) -> pl.DataFrame:
        sorted_words: List[Tuple[str, int]] = sorted(
            word_freq.items(), key=lambda x: x[1], reverse=True
        )
        top_words = sorted_words if top_n is None else sorted_words[:top_n]
        if not top_words:
            return pl.DataFrame({"word": [], "count": []})
        words, counts = zip(*top_words)
        return pl.DataFrame({"word": list(words), "count": list(counts)})

    def get_tfidf_ranking(
        self, df: pl.DataFrame, text_column: str = "text", top_n: int | None = 50
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
        if top_n is not None:
            rows = rows[:top_n]
        return pl.DataFrame(rows)


def hashtag_ranking(frame: pl.DataFrame, top_n: int | None = 20) -> pl.DataFrame:
    """ハッシュタグ出現数ランキングを返す。"""

    if "hashtag_list" not in frame.columns:
        return pl.DataFrame()
    exploded = frame.explode("hashtag_list").drop_nulls("hashtag_list")
    if exploded.is_empty():
        return pl.DataFrame()
    cleaned = exploded.with_columns(
        pl.col("hashtag_list").cast(pl.Utf8).str.strip_chars().alias("hashtag")
    ).filter(pl.col("hashtag") != "")
    if cleaned.is_empty():
        return pl.DataFrame()
    result = (
        cleaned.group_by("hashtag")
        .count()
        .rename({"count": "occurrences"})
        .sort("occurrences", descending=True)
    )
    return result if top_n is None else result.head(top_n)


def hashtag_cooccurrence(
    frame: pl.DataFrame, top_n: int = 50, min_count: int = 2
) -> pl.DataFrame:
    """同一投稿内のハッシュタグ共起ペアを返す。"""

    if "hashtag_list" not in frame.columns:
        return pl.DataFrame()
    pairs: Dict[Tuple[str, str], int] = {}
    for tags in frame.select("hashtag_list").to_series():
        if not tags:
            continue
        unique = sorted({str(t).strip() for t in tags if str(t).strip()})
        if len(unique) < 2:
            continue
        for tag_a, tag_b in combinations(unique, 2):
            key = (tag_a, tag_b)
            pairs[key] = pairs.get(key, 0) + 1
    if not pairs:
        return pl.DataFrame()
    rows = [
        {"tag_a": tag_a, "tag_b": tag_b, "count": count}
        for (tag_a, tag_b), count in pairs.items()
        if count >= min_count
    ]
    if not rows:
        return pl.DataFrame()
    rows.sort(key=lambda r: r["count"], reverse=True)
    if top_n:
        rows = rows[:top_n]
    return pl.DataFrame(rows)


def parse_keyword_dictionary(raw: Optional[str]) -> Dict[str, List[str]]:
    """カテゴリ: キーワード1,キーワード2 の形式を辞書に変換する。"""

    if not raw:
        return {}
    mapping: Dict[str, List[str]] = {}
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if ":" in stripped:
            label, keywords = stripped.split(":", 1)
        elif "\uFF1A" in stripped:
            label, keywords = stripped.split("\uFF1A", 1)
        else:
            continue
        label = label.strip()
        if not label:
            continue
        terms = re.split(r"[,\u3001]+", keywords)
        cleaned_terms = [term.strip() for term in terms if term.strip()]
        if not cleaned_terms:
            continue
        mapping[label] = cleaned_terms
    return mapping


def keyword_category_counts(frame: pl.DataFrame, keyword_dict: Dict[str, List[str]]) -> pl.DataFrame:
    """カテゴリ別投稿数を返す。"""

    if not keyword_dict or "text" not in frame.columns:
        return pl.DataFrame()
    texts = frame.select(pl.col("text").cast(pl.Utf8).fill_null("")).to_series().to_list()
    counts = {category: 0 for category in keyword_dict}
    for text in texts:
        lowered = text.lower()
        for category, terms in keyword_dict.items():
            if any(term.lower() in lowered for term in terms):
                counts[category] += 1
    rows = [{"category": category, "posts": posts} for category, posts in counts.items()]
    rows.sort(key=lambda r: r["posts"], reverse=True)
    return pl.DataFrame(rows)


def word_monthly_counts(
    frame: pl.DataFrame, term: str, analyzer: "TextAnalyzer"
) -> pl.DataFrame:
    """指定語の月次出現数を返す。"""

    if not term or "text" not in frame.columns:
        return pl.DataFrame()
    if "year" in frame.columns and "month" in frame.columns:
        rows = frame.select(
            pl.col("text").cast(pl.Utf8).fill_null("").alias("text"),
            pl.col("year").cast(pl.Int32).alias("year"),
            pl.col("month").cast(pl.Int8).alias("month"),
        ).to_dicts()
    elif "created_at" in frame.columns:
        rows = frame.select(
            pl.col("text").cast(pl.Utf8).fill_null("").alias("text"),
            pl.col("created_at").dt.year().alias("year"),
            pl.col("created_at").dt.month().alias("month"),
        ).to_dicts()
    else:
        return pl.DataFrame()

    counts: Dict[Tuple[int, int], int] = {}
    for row in rows:
        tokens = analyzer.extract_words_from_text(row["text"])
        if term in tokens:
            key = (int(row["year"]), int(row["month"]))
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return pl.DataFrame()
    data = [
        {"year": year, "month": month, "posts": count}
        for (year, month), count in counts.items()
    ]
    data.sort(key=lambda r: (r["year"], r["month"]))
    return pl.DataFrame(data).with_columns(
        (
            pl.col("year").cast(pl.Utf8) + "-" + pl.col("month").cast(pl.Utf8).str.zfill(2)
        ).alias("year_month")
    )
