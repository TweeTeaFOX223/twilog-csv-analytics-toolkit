from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import polars as pl

__all__ = ["CSVLoadError", "load_twilog_csv", "ColumnMapping"]


ENCODING_CANDIDATES: tuple[str, ...] = (
    "utf8",
    "utf8-lossy",
    "cp932",
    "shift_jis",
    "euc_jp",
)
DEFAULT_COLUMN_ORDER: tuple[str, ...] = (
    "tweet_id",
    "created_at",
    "text",
    "status_url",
)


class CSVLoadError(Exception):
    """CSVの読み込みや正規化に失敗したときの例外。"""


@dataclass
class ColumnMapping:
    tweet_id: str
    created_at: str
    text: str
    status_url: Optional[str] = None


def _try_read_csv(path: Path, has_header: bool) -> Tuple[str, pl.DataFrame]:
    """エンコーディング候補を総当たりでCSVを試し読みする。"""

    last_error: Exception | None = None
    for encoding in ENCODING_CANDIDATES:
        try:
            frame = pl.read_csv(
                path,
                has_header=has_header,
                encoding=encoding,
                infer_schema_length=0,
            )
            print(
                f"[loader] read_csv success enc={encoding} header={has_header} "
                f"rows={frame.height} cols={frame.columns}"
            )
            return encoding, frame
        except Exception as exc:  # pragma: no cover - errorは後でまとめて表示
            last_error = exc
            print(f"[loader] read_csv failed enc={encoding} header={has_header} error={exc}")
            continue
    raise CSVLoadError(
        f"Failed to read CSV with encodings {ENCODING_CANDIDATES}: {last_error}"
    )


def _samples(frame: pl.DataFrame, col: str, limit: int = 50) -> List[str]:
    """指定列の先頭サンプルを文字列として取得する。"""

    return [
        str(x)
        for x in frame.select(pl.col(col).head(limit)).to_series().to_list()
        if x is not None
    ]


def _guess_mapping(frame: pl.DataFrame) -> ColumnMapping:
    """列名と中身をスコアリングしてTwilog風のカラム割当を推定する。"""

    columns = frame.columns
    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?$")

    date_scores: Dict[str, int] = {}
    url_scores: Dict[str, int] = {}
    text_lengths: Dict[str, float] = {}
    id_scores: Dict[str, int] = {}

    for col in columns:
        values = _samples(frame, col)
        date_scores[col] = sum(1 for v in values if date_pattern.match(v))
        url_scores[col] = sum(1 for v in values if "http" in v.lower())
        text_lengths[col] = sum(len(v) for v in values) / max(len(values), 1)
        id_scores[col] = sum(1 for v in values if v.isdigit() and len(v) >= 10)

    created_at_col = max(columns, key=lambda c: date_scores.get(c, 0))
    status_url_col = max(columns, key=lambda c: url_scores.get(c, 0))

    text_candidates = [c for c in columns if c not in {created_at_col, status_url_col}]
    text_col = (
        max(text_candidates, key=lambda c: text_lengths.get(c, 0))
        if text_candidates
        else max(columns, key=lambda c: text_lengths.get(c, 0))
    )

    id_candidates = [c for c in columns if c not in {created_at_col, status_url_col, text_col}]
    tweet_id_col = (
        max(id_candidates, key=lambda c: id_scores.get(c, 0))
        if id_candidates
        else max(columns, key=lambda c: id_scores.get(c, 0))
    )

    guessed = ColumnMapping(
        tweet_id=tweet_id_col,
        created_at=created_at_col,
        text=text_col,
        status_url=status_url_col if status_url_col else None,
    )
    print(
        "[loader] guessed mapping by content:",
        guessed,
        "date_scores",
        date_scores,
        "url_scores",
        url_scores,
    )
    return guessed


def _normalize_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """列名の正規化・型揃え・日付パースを行う。"""

    mapping = _guess_mapping(frame)
    rename_map = {
        mapping.tweet_id: "tweet_id",
        mapping.created_at: "created_at",
        mapping.text: "text",
    }
    if mapping.status_url:
        rename_map[mapping.status_url] = "status_url"

    normalized = frame.rename(rename_map)
    if "status_url" not in normalized.columns:
        normalized = normalized.with_columns(pl.lit(None).alias("status_url"))

    normalized = normalized.select(["tweet_id", "created_at", "text", "status_url"])
    normalized = normalized.with_columns(
        pl.col("tweet_id").cast(pl.Utf8),
        pl.col("text").cast(pl.Utf8),
        pl.col("status_url").cast(pl.Utf8),
    )

    normalized = normalized.with_columns(
        pl.coalesce(
            [
                pl.col("created_at"),
                pl.col("created_at").cast(pl.Utf8),
            ]
        ).alias("created_at")
    )

    normalized = normalized.with_columns(
        pl.coalesce(
            [
                pl.col("created_at")
                .cast(pl.Utf8)
                .str.strip_chars()
                .str.to_datetime(format="%Y-%m-%d %H:%M:%S", strict=False),
                pl.col("created_at")
                .cast(pl.Utf8)
                .str.strip_chars()
                .str.to_datetime(format="%Y-%m-%d %H:%M", strict=False),
                pl.col("created_at")
                .cast(pl.Utf8)
                .str.strip_chars()
                .str.to_date(format="%Y-%m-%d", strict=False)
                .cast(pl.Datetime("ms")),
            ]
        ).alias("created_at")
    )

    print(
        "[loader] normalized dtypes:",
        dict(zip(normalized.columns, [str(dt) for dt in normalized.dtypes])),
    )
    print("[loader] normalized created_at samples:", normalized.select("created_at").head(5).to_dicts())
    return normalized


def _score_created_at(frame: pl.DataFrame) -> int:
    """created_at が非NULLの行数をスコアとして返す。"""

    if "created_at" not in frame.columns:
        return 0
    return frame.select(pl.col("created_at").drop_nulls().count()).item()


def load_twilog_csv(path: Path, has_header: bool | None = None) -> pl.DataFrame:
    """Twilog風CSVをエンコーディング・ヘッダ有無を推測しながら読み込む。"""

    tried_frames: list[tuple[bool, pl.DataFrame]] = []
    errors: list[str] = []

    header_options = [True, False] if has_header is None else [has_header]
    for header_flag in header_options:
        try:
            encoding, frame = _try_read_csv(path, header_flag)
            tried_frames.append((header_flag, frame))
            print(f"[loader] candidate frame header={header_flag} enc={encoding} shape={frame.shape}")
        except CSVLoadError as exc:
            errors.append(str(exc))
            continue

    if not tried_frames:
        raise CSVLoadError("; ".join(errors))

    best_frame: pl.DataFrame | None = None
    best_score = -1

    for header_flag, frame in tried_frames:
        try:
            normalized = _normalize_frame(frame)
            score = _score_created_at(normalized)
            print(f"[loader] candidate header={header_flag} created_at non-null score={score}")
            if score > best_score:
                best_score = score
                best_frame = normalized
        except Exception as exc:
            print(f"[loader] normalize failed header={header_flag} error={exc}")
            continue

    if best_frame is None:
        raise CSVLoadError("CSV normalization failed for all header options")

    if best_frame.is_empty():
        raise CSVLoadError("CSV was read but contains no rows")

    print("[loader] normalized head:", best_frame.head(3).to_dicts())
    return best_frame