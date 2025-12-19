from __future__ import annotations

import polars as pl

__all__ = [
    "basic_counts",
    "hourly_counts",
    "weekday_counts",
    "monthly_counts",
    "yearly_counts",
    "average_counts",
    "average_yearly_counts",
    "average_monthly_counts",
    "average_weekday_counts",
]


def basic_counts(frame: pl.DataFrame) -> dict[str, int | float]:
    """総投稿数や1日平均などの概要値を返す。"""
    if frame.is_empty():
        return {"total": 0, "avg_per_day": 0.0}
    total = frame.height
    min_date = frame.select(pl.col("created_at").min()).item()
    max_date = frame.select(pl.col("created_at").max()).item()
    days = max((max_date - min_date).days + 1, 1) if min_date and max_date else 1
    return {"total": total, "avg_per_day": round(total / days, 2)}


def average_counts(frame: pl.DataFrame) -> dict[str, float]:
    """1日/1週/1ヶ月/1年の平均投稿数を返す。"""

    if frame.is_empty():
        return {"per_day": 0.0, "per_week": 0.0, "per_month": 0.0, "per_year": 0.0}

    total = frame.height
    min_date = frame.select(pl.col("created_at").min()).item()
    max_date = frame.select(pl.col("created_at").max()).item()
    days = max((max_date - min_date).days + 1, 1) if min_date and max_date else 1
    weeks = max(days / 7, 1)

    # 月数は year*12+month を使って差分＋1。ただし欠損がある場合は日数から概算。
    ym_df = frame.select((pl.col("created_at").dt.year() * 12 + pl.col("created_at").dt.month()).alias("ym"))
    if ym_df.is_empty():
        months_span = max(days / 30.4, 1)
    else:
        ym_min = ym_df.select(pl.col("ym").min()).item()
        ym_max = ym_df.select(pl.col("ym").max()).item()
        months_span = max((ym_max - ym_min) + 1, 1)

    years_span = max(days / 365, 1)

    return {
        "per_day": round(total / days, 2),
        "per_week": round(total / weeks, 2),
        "per_month": round(total / months_span, 2),
        "per_year": round(total / years_span, 2),
    }


def _daily_counts_with_zeros(frame: pl.DataFrame) -> pl.DataFrame:
    """日次の投稿数を欠損日=0で補完して返す。"""

    if frame.is_empty() or "date" not in frame.columns:
        return pl.DataFrame()

    min_date = frame.select(pl.col("date").min()).item()
    max_date = frame.select(pl.col("date").max()).item()
    if min_date is None or max_date is None:
        return pl.DataFrame()

    date_range = pl.date_range(min_date, max_date, "1d", eager=True)
    base = pl.DataFrame({"date": date_range})
    counts = frame.group_by("date").count().rename({"count": "posts"})
    merged = base.join(counts, on="date", how="left").fill_null(0)
    weekday_expr = pl.col("date").dt.strftime("%u").cast(pl.Int8) - 1
    return merged.with_columns(
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.month().alias("month"),
        weekday_expr.alias("weekday"),
    )


def average_yearly_counts(frame: pl.DataFrame) -> pl.DataFrame:
    """年ごとの平均投稿数（1日平均）を返す。"""

    daily = _daily_counts_with_zeros(frame)
    if daily.is_empty():
        return pl.DataFrame()
    return (
        daily.group_by("year")
        .agg(pl.col("posts").mean().round(2).alias("avg_posts"))
        .sort("year")
    )


def average_monthly_counts(frame: pl.DataFrame) -> pl.DataFrame:
    """年×月ごとの平均投稿数（1日平均）を返す。"""

    daily = _daily_counts_with_zeros(frame)
    if daily.is_empty():
        return pl.DataFrame()
    return (
        daily.group_by(["year", "month"])
        .agg(pl.col("posts").mean().round(2).alias("avg_posts"))
        .sort(["year", "month"])
    )


def average_weekday_counts(frame: pl.DataFrame) -> pl.DataFrame:
    """曜日ごとの平均投稿数（1日平均）を返す。"""

    daily = _daily_counts_with_zeros(frame)
    if daily.is_empty():
        return pl.DataFrame()
    return daily.group_by("weekday").agg(pl.col("posts").mean().round(2).alias("avg_posts")).sort(
        "weekday"
    )


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
