from __future__ import annotations

from typing import Sequence

import polars as pl

__all__ = ["add_derived_columns", "filter_by_years"]


# 簡易なパターン類（正規表現）
CODE_PATTERN = r"(?i)\b(import|from|def|class|const|let|var|function|async|await)\b"
URL_PATTERN = r"https?://[^\s]+"
HASHTAG_PATTERN = r"#(\w+)"
MENTION_PATTERN = r"@(\w+)"


def _safe_ratio(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    """0除算を避けつつ安全に比率を計算する。"""

    return numerator / pl.when(denominator == 0).then(1).otherwise(denominator)


def add_derived_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """分析でよく使う派生カラムを追加する。"""

    # テキストを欠損なしの文字列に統一
    df = frame.with_columns(pl.col("text").cast(pl.Utf8).fill_null(""))
    if "created_at" in df.columns:
        df = df.with_columns(
            pl.col("created_at").dt.date().alias("date"),
            pl.col("created_at").dt.year().alias("year"),
            pl.col("created_at").dt.month().alias("month"),
            pl.col("created_at").dt.weekday().alias("weekday"),
            pl.col("created_at").dt.hour().alias("hour"),
        )

    df = df.with_columns(
        pl.col("text").str.len_chars().alias("text_length"),
        pl.col("text").str.extract_all(URL_PATTERN).alias("urls"),
    )

    df = df.with_columns(
        pl.col("urls").list.len().alias("url_count"),
        pl.col("text").str.extract_all(HASHTAG_PATTERN).alias("hashtag_list"),
        pl.col("text").str.extract_all(MENTION_PATTERN).alias("mention_list"),
        pl.col("text").str.starts_with("@").alias("is_reply"),
        pl.col("text").str.contains(CODE_PATTERN).alias("contains_code"),
    )

    df = df.with_columns(
        pl.col("hashtag_list").list.len().alias("hashtag_count"),
        pl.col("mention_list").list.len().alias("mention_count"),
        pl.col("urls")
        .list.eval(
            pl.element()
            .str.replace(r"^https?://", "")
            .str.split("/")
            .list.get(0)
        )
        .alias("domains"),
    )

    # 抽出結果がnullの場合に備えて空リストで補完
    df = df.with_columns(
        pl.col("urls").fill_null(pl.lit([])),
        pl.col("hashtag_list").fill_null(pl.lit([])),
        pl.col("mention_list").fill_null(pl.lit([])),
        pl.col("domains").fill_null(pl.lit([])),
    )

    # ASCII割合の簡易推定（非ASCIIを除去して文字数比を算出）
    ascii_chars = pl.col("text").str.replace_all(r"[^\x00-\x7F]", "")
    ascii_ratio = _safe_ratio(ascii_chars.str.len_chars(), pl.col("text").str.len_chars())
    df = df.with_columns(
        pl.when(ascii_ratio > 0.6)
        .then(pl.lit("en-biased"))
        .otherwise(pl.lit("ja-biased"))
        .alias("language_hint")
    )

    print(
        "[preprocessor] derived dtypes:",
        dict(zip(df.columns, [str(dt) for dt in df.dtypes])),
    )
    print(
        "[preprocessor] date/year/month/hour sample:",
        df.select(["created_at", "date", "year", "month", "hour"])
        .head(5)
        .to_dicts(),
    )
    print(
        "[preprocessor] domains sample:",
        df.select(["domains", "url_count"]).head(5).to_dicts(),
    )

    return df


def filter_by_years(frame: pl.DataFrame, years: Sequence[int] | None) -> pl.DataFrame:
    """特定年のみ抽出するシンプルなフィルタ。"""

    if not years or "year" not in frame.columns:
        return frame
    return frame.filter(pl.col("year").is_in(list(years)))
