from __future__ import annotations

import json
from typing import Any, Dict

import polars as pl

__all__ = ["plotly_bar"]


def plotly_bar(frame: pl.DataFrame, x: str, y: str, title: str = "") -> Dict[str, Any]:
    """Plotlyに渡すシンプルな棒グラフ仕様を組み立てる。"""
    if frame.is_empty():
        return {"data": [], "layout": {"title": title}}
    return {
        "data": [
            {
                "type": "bar",
                "x": frame[x].to_list(),
                "y": frame[y].to_list(),
            }
        ],
        "layout": {"title": title},
    }
