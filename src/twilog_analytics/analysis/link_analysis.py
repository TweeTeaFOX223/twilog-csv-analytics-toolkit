from __future__ import annotations

import polars as pl

__all__ = ["domain_ranking"]


def domain_ranking(frame: pl.DataFrame, top_n: int = 20) -> pl.DataFrame:
    """本文中URLのドメイン出現数ランキングを返す。"""
    if "domains" not in frame.columns:
        return pl.DataFrame()
    exploded = frame.explode("domains").drop_nulls("domains")
    return (
        exploded.group_by("domains")
        .count()
        .rename({"count": "occurrences", "domains": "domain"})
        .sort("occurrences", descending=True)
        .head(top_n)
    )
