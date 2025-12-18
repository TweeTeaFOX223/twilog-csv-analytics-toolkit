from __future__ import annotations

from pathlib import Path

import polars as pl

__all__ = ["save_matplotlib_placeholder"]


def save_matplotlib_placeholder(output_path: Path) -> Path:
    """実装前のプレースホルダ：空ファイルを生成するだけ。"""
    output_path.write_bytes(b"")
    return output_path
