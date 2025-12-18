from __future__ import annotations

from pathlib import Path
from typing import Iterable

import polars as pl

__all__ = ["build_wordcloud_placeholder"]


def build_wordcloud_placeholder(output_path: Path) -> Path:
    """実装前のプレースホルダ：空ファイルを生成するだけ。"""
    output_path.write_bytes(b"")
    return output_path
