from __future__ import annotations

import polars as pl

__all__ = ["daily_counts", "weekday_hour_counts", "weekday_hour_matrix"]


def daily_counts(frame: pl.DataFrame) -> pl.DataFrame:
    """Count posts per date (requires a date column)."""

    if "date" not in frame.columns:
        return pl.DataFrame()
    return frame.group_by("date").count().rename({"count": "posts"}).sort("date")


def weekday_hour_counts(frame: pl.DataFrame) -> pl.DataFrame:
    """曜日×時間帯の投稿数を長い形式で返す。"""

    if "weekday" not in frame.columns or "hour" not in frame.columns:
        return pl.DataFrame()
    return (
        frame.group_by(["weekday", "hour"])
        .count()
        .rename({"count": "posts"})
        .sort(["weekday", "hour"])
    )


def weekday_hour_matrix(frame: pl.DataFrame) -> tuple[list[str], list[str], list[list[int]]]:
    """Return a weekday x hour post count matrix for heatmap plotting."""

    if "weekday" not in frame.columns or "hour" not in frame.columns:
        return [], [], []

    counts = (
        frame.group_by(["weekday", "hour"])
        .count()
        .rename({"count": "posts"})
        .with_columns(pl.col("weekday").cast(pl.Int8), pl.col("hour").cast(pl.Int8))
    )

    weekdays = list(range(7))
    hours = list(range(24))
    hours_labels = [str(h) for h in hours]
    # Japanese weekday labels (Mon-Sun) expressed via escapes to keep the file ASCII
    weekday_labels = ["\u6708", "\u706b", "\u6c34", "\u6728", "\u91d1", "\u571f", "\u65e5"]

    pivoted = counts.pivot(index="weekday", columns="hour", values="posts", aggregate_function="sum").sort(
        "weekday"
    )
    rename_map = {col: str(col) for col in pivoted.columns if col != "weekday"}
    pivoted = pivoted.rename(rename_map)

    for h in hours:
        if str(h) not in pivoted.columns:
            pivoted = pivoted.with_columns(pl.lit(0).alias(str(h)))

    z_matrix: list[list[int]] = []
    for wd in weekdays:
        row = pivoted.filter(pl.col("weekday") == wd)
        if row.is_empty():
            z_matrix.append([0 for _ in hours])
        else:
            z_matrix.append([int(row.select(str(h)).item() if str(h) in row.columns else 0) for h in hours])

    return hours_labels, [weekday_labels[wd] for wd in weekdays], z_matrix
