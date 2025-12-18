from __future__ import annotations

import polars as pl

__all__ = ["word_ranking"]


def word_ranking(frame: pl.DataFrame, top_n: int = 30) -> pl.DataFrame:
    """SudachiPyベースの単語ランキング（未実装のプレースホルダ）。"""
    return pl.DataFrame({"token": [], "count": []})
