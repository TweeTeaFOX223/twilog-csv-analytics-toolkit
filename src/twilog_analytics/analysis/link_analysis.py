from __future__ import annotations

import polars as pl

__all__ = [
    "domain_ranking",
    "tld_distribution",
    "domain_year_trend",
    "self_reference_stats",
]


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


def domain_year_trend(frame: pl.DataFrame, top_n: int = 5) -> pl.DataFrame:
    """ドメイン×年の投稿数推移を返す。"""

    if "domains" not in frame.columns or "year" not in frame.columns:
        return pl.DataFrame()
    exploded = frame.explode("domains").drop_nulls("domains")
    if exploded.is_empty():
        return pl.DataFrame()
    top_domains = (
        exploded.group_by("domains")
        .count()
        .rename({"count": "total"})
        .sort("total", descending=True)
        .head(top_n)
        .select("domains")
    )
    if top_domains.is_empty():
        return pl.DataFrame()
    filtered = exploded.join(top_domains, on="domains", how="inner")
    return (
        filtered.group_by(["year", "domains"])
        .count()
        .rename({"count": "occurrences", "domains": "domain"})
        .sort(["year", "occurrences"], descending=[False, True])
    )


def self_reference_stats(frame: pl.DataFrame) -> dict[str, int | float]:
    """twitter.com/x.com 参照を含む投稿の比率を返す。"""

    if "domains" not in frame.columns:
        return {"self_ref": 0, "external": 0, "rate": 0.0}
    total = frame.height
    flags = (
        frame.select(pl.col("domains"))
        .with_columns(
            pl.col("domains")
            .list.eval(pl.element().cast(pl.Utf8).str.to_lowercase())
            .alias("domains_lower")
        )
        .with_columns(
            pl.col("domains_lower")
            .list.eval(
                pl.element().str.ends_with("twitter.com") | pl.element().str.ends_with("x.com")
            )
            .list.any()
            .alias("self_ref")
        )
        .select("self_ref")
        .to_series()
        .to_list()
    )
    self_ref = sum(1 for flag in flags if flag)
    external = max(total - self_ref, 0)
    rate = round(self_ref / total, 4) if total else 0.0
    return {"self_ref": self_ref, "external": external, "rate": rate}
