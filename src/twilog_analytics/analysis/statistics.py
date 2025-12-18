from __future__ import annotations

import polars as pl

__all__ = ["basic_counts", "hourly_counts", "weekday_counts", "monthly_counts", "yearly_counts"]


def basic_counts(frame: pl.DataFrame) -> dict[str, int | float]:
    """総投稿数や1日平均などの概要値を返す。"""
    if frame.is_empty():
        return {"total": 0, "avg_per_day": 0.0}
    total = frame.height
    min_date = frame.select(pl.col("created_at").min()).item()
    max_date = frame.select(pl.col("created_at").max()).item()
    days = max((max_date - min_date).days + 1, 1) if min_date and max_date else 1
    return {"total": total, "avg_per_day": round(total / days, 2)}


def yearly_counts(frame: pl.DataFrame) -> pl.DataFrame:
    """年単位の投稿数を集計する。"""
    if "year" not in frame.columns:
        return pl.DataFrame()
    return frame.group_by("year").count().rename({"count": "posts"}).sort("year")


def monthly_counts(frame: pl.DataFrame) -> pl.DataFrame:
    """年×月単位の投稿数を集計する。"""
    if "year" not in frame.columns or "month" not in frame.columns:
        return pl.DataFrame()
    return (
        frame.group_by(["year", "month"]).count().rename({"count": "posts"}).sort(["year", "month"])
    )


def weekday_counts(frame: pl.DataFrame) -> pl.DataFrame:
    """曜日ごとの投稿数を返す（0=月）。"""
    if "weekday" not in frame.columns:
        return pl.DataFrame()
    return frame.group_by("weekday").count().rename({"count": "posts"}).sort("weekday")


def hourly_counts(frame: pl.DataFrame) -> pl.DataFrame:
    """時刻ごとの投稿数を返す。"""
    if "hour" not in frame.columns:
        return pl.DataFrame()
    return frame.group_by("hour").count().rename({"count": "posts"}).sort("hour")
