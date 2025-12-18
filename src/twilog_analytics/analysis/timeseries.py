from __future__ import annotations

import polars as pl

__all__ = ["daily_counts"]


def daily_counts(frame: pl.DataFrame) -> pl.DataFrame:
    """日付単位で投稿数をカウントする。"""
    if "date" not in frame.columns:
        return pl.DataFrame()
    return frame.group_by("date").count().rename({"count": "posts"}).sort("date")
