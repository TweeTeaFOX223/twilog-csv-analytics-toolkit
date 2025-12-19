from __future__ import annotations

import polars as pl

__all__ = ["domain_ranking", "tld_distribution"]


def domain_ranking(frame: pl.DataFrame, top_n: int | None = 20) -> pl.DataFrame:
    """本文中URLのドメイン出現数ランキングを返す。"""
    if "domains" not in frame.columns:
        return pl.DataFrame()
    exploded = frame.explode("domains").drop_nulls("domains")
    result = (
        exploded.group_by("domains")
        .count()
        .rename({"count": "occurrences", "domains": "domain"})
        .sort("occurrences", descending=True)
    )
    return result if top_n is None else result.head(top_n)


def tld_distribution(frame: pl.DataFrame, top_n: int | None = 20) -> pl.DataFrame:
    """TLD別の出現数ランキングを返す。"""
    if "domains" not in frame.columns:
        return pl.DataFrame()
    exploded = frame.explode("domains").drop_nulls("domains")
    if exploded.is_empty():
        return pl.DataFrame()
    tlds = exploded.with_columns(
        pl.col("domains")
        .cast(pl.Utf8)
        .str.to_lowercase()
        .str.extract(r"\.([A-Za-z0-9-]+)$", 1)
        .alias("tld")
    ).drop_nulls("tld")
    if tlds.is_empty():
        return pl.DataFrame()
    result = (
        tlds.group_by("tld")
        .count()
        .rename({"count": "occurrences"})
        .sort("occurrences", descending=True)
    )
    return result if top_n is None else result.head(top_n)
