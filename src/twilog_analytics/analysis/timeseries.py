from __future__ import annotations

import polars as pl

__all__ = [
    "daily_counts",
    "daily_counts_full",
    "weekly_counts",
    "monthday_counts",
    "weekday_monthday_matrix",
    "weekday_hour_counts",
    "weekday_hour_matrix",
    "daily_calendar_matrix",
]


def daily_counts(frame: pl.DataFrame) -> pl.DataFrame:
    """Count posts per date (requires a date column)."""

    if "date" not in frame.columns:
        return pl.DataFrame()
    return frame.group_by("date").count().rename({"count": "posts"}).sort("date")


def daily_counts_full(frame: pl.DataFrame) -> pl.DataFrame:
    """日次の投稿数を欠損日=0で補完して返す。"""

    if "date" not in frame.columns:
        return pl.DataFrame()
    min_date = frame.select(pl.col("date").min()).item()
    max_date = frame.select(pl.col("date").max()).item()
    if min_date is None or max_date is None:
        return pl.DataFrame()
    date_range = pl.date_range(min_date, max_date, "1d", eager=True)
    base = pl.DataFrame({"date": date_range})
    counts = frame.group_by("date").count().rename({"count": "posts"})
    return base.join(counts, on="date", how="left").fill_null(0).sort("date")


def weekly_counts(frame: pl.DataFrame) -> pl.DataFrame:
    """ISO週単位の投稿数を返す。"""

    if "date" not in frame.columns:
        return pl.DataFrame()
    return (
        frame.with_columns(pl.col("date").dt.strftime("%G-W%V").alias("iso_week"))
        .group_by("iso_week")
        .count()
        .rename({"count": "posts"})
        .sort("iso_week")
    )


def monthday_counts(frame: pl.DataFrame) -> pl.DataFrame:
    """月内日（1-31）の投稿数を返す。"""

    if "date" not in frame.columns:
        return pl.DataFrame()
    df = frame.with_columns(pl.col("date").dt.day().alias("monthday"))
    base = pl.DataFrame({"monthday": list(range(1, 32))})
    counts = df.group_by("monthday").count().rename({"count": "posts"})
    return base.join(counts, on="monthday", how="left").fill_null(0).sort("monthday")


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


def weekday_monthday_matrix(frame: pl.DataFrame) -> tuple[list[str], list[str], list[list[int]]]:
    """曜日×月内日の投稿数行列を返す。"""

    if "date" not in frame.columns:
        return [], [], []
    df = frame.with_columns(
        (pl.col("date").dt.strftime("%u").cast(pl.Int8) - 1).alias("weekday"),
        pl.col("date").dt.day().alias("monthday"),
    )
    counts = (
        df.group_by(["weekday", "monthday"])
        .count()
        .rename({"count": "posts"})
        .sort(["weekday", "monthday"])
    )
    weekdays = list(range(7))
    monthdays = list(range(1, 32))
    weekday_labels = ["\u6708", "\u706b", "\u6c34", "\u6728", "\u91d1", "\u571f", "\u65e5"]

    pivoted = counts.pivot(
        index="weekday", columns="monthday", values="posts", aggregate_function="sum"
    ).sort("weekday")
    rename_map = {col: str(col) for col in pivoted.columns if col != "weekday"}
    pivoted = pivoted.rename(rename_map)

    for day in monthdays:
        if str(day) not in pivoted.columns:
            pivoted = pivoted.with_columns(pl.lit(0).alias(str(day)))

    z_matrix: list[list[int]] = []
    for wd in weekdays:
        row = pivoted.filter(pl.col("weekday") == wd)
        if row.is_empty():
            z_matrix.append([0 for _ in monthdays])
        else:
            z_matrix.append(
                [int(row.select(str(day)).item() if str(day) in row.columns else 0) for day in monthdays]
            )
    return [str(day) for day in monthdays], weekday_labels, z_matrix


def daily_calendar_matrix(frame: pl.DataFrame) -> tuple[list[str], list[str], list[list[int]]]:
    """日別投稿数を週×曜日の行列で返す。"""

    daily = daily_counts_full(frame)
    if daily.is_empty():
        return [], [], []
    weekday_expr = pl.col("date").dt.strftime("%u").cast(pl.Int8) - 1
    daily = daily.with_columns(
        weekday_expr.alias("weekday"),
        pl.col("date").dt.strftime("%G-W%V").alias("iso_week"),
    ).with_columns(
        (pl.col("date") - (pl.col("weekday") * pl.duration(days=1))).alias("week_start"),
    )
    week_order = (
        daily.group_by("iso_week")
        .agg(pl.col("week_start").min().alias("week_start"))
        .sort("week_start")
        .select("iso_week")
        .to_series()
        .to_list()
    )
    values = {
        (row["iso_week"], row["weekday"]): row["posts"]
        for row in daily.select(["iso_week", "weekday", "posts"]).to_dicts()
    }
    weekday_labels = ["\u6708", "\u706b", "\u6c34", "\u6728", "\u91d1", "\u571f", "\u65e5"]
    z_matrix: list[list[int]] = []
    for iso_week in week_order:
        z_matrix.append([int(values.get((iso_week, wd), 0)) for wd in range(7)])
    return weekday_labels, [str(w) for w in week_order], z_matrix
