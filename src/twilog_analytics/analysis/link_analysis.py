from __future__ import annotations

import polars as pl

__all__ = [
    "domain_ranking",
    "tld_distribution",
    "domain_year_trend",
    "domain_month_trend",
    "path_depth_distribution",
    "self_reference_stats",
    "self_post_url_stats",
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


def domain_month_trend(frame: pl.DataFrame, domain: str) -> pl.DataFrame:
    """指定ドメインの月次推移を返す。"""

    if not domain or "domains" not in frame.columns:
        return pl.DataFrame()
    if "year" in frame.columns and "month" in frame.columns:
        df = frame
    elif "created_at" in frame.columns:
        df = frame.with_columns(
            pl.col("created_at").dt.year().alias("year"),
            pl.col("created_at").dt.month().alias("month"),
        )
    else:
        return pl.DataFrame()
    exploded = df.explode("domains").drop_nulls("domains")
    if exploded.is_empty():
        return pl.DataFrame()
    filtered = exploded.filter(pl.col("domains").cast(pl.Utf8) == domain)
    if filtered.is_empty():
        return pl.DataFrame()
    return (
        filtered.group_by(["year", "month"])
        .count()
        .rename({"count": "occurrences"})
        .sort(["year", "month"])
        .with_columns(
            (
                pl.col("year").cast(pl.Utf8)
                + "-"
                + pl.col("month").cast(pl.Utf8).str.zfill(2)
            ).alias("year_month")
        )
    )


def path_depth_distribution(frame: pl.DataFrame, exclude_self_ref: bool = True) -> pl.DataFrame:
    """URLのパス深さ分布を返す。"""

    if "urls" not in frame.columns:
        return pl.DataFrame()
    exploded = frame.explode("urls").drop_nulls("urls")
    if exploded.is_empty():
        return pl.DataFrame()
    cleaned = exploded.with_columns(
        pl.col("urls")
        .cast(pl.Utf8)
        .str.replace(r"^https?://", "")
        .str.split("#")
        .list.get(0)
        .str.split("?")
        .list.get(0)
        .alias("url_trimmed")
    ).with_columns(
        pl.col("url_trimmed").str.split("/").alias("url_parts")
    )
    if exclude_self_ref:
        cleaned = cleaned.with_columns(
            pl.col("url_parts").list.get(0).alias("domain")
        ).filter(
            ~pl.col("domain").cast(pl.Utf8).str.to_lowercase().str.ends_with("twitter.com")
        ).filter(
            ~pl.col("domain").cast(pl.Utf8).str.to_lowercase().str.ends_with("x.com")
        )
    cleaned = cleaned.with_columns(
        (pl.col("url_parts").list.len() - 1).alias("depth")
    ).filter(pl.col("depth") >= 0)
    if cleaned.is_empty():
        return pl.DataFrame()
    return (
        cleaned.group_by("depth")
        .count()
        .rename({"count": "occurrences"})
        .sort("depth")
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


def self_post_url_stats(frame: pl.DataFrame) -> dict[str, int | float]:
    """自分のツイートURL参照の件数と比率を返す。"""

    if "urls" not in frame.columns or "status_url" not in frame.columns:
        return {"self_post": 0, "other_url": 0, "no_url": 0, "rate": 0.0}
    url_rows = frame.select(
        pl.col("urls"),
        pl.col("status_url").cast(pl.Utf8).fill_null("").alias("status_url"),
    ).to_dicts()
    self_post = 0
    other_url = 0
    no_url = 0
    for row in url_rows:
        urls = row["urls"] or []
        status_url = str(row["status_url"]).split("?")[0].split("#")[0]
        if not urls:
            no_url += 1
            continue
        if status_url and any(str(url).split("?")[0].split("#")[0] == status_url for url in urls):
            self_post += 1
        else:
            other_url += 1
    total = len(url_rows)
    rate = round(self_post / total, 4) if total else 0.0
    return {"self_post": self_post, "other_url": other_url, "no_url": no_url, "rate": rate}
