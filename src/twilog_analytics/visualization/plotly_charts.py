from __future__ import annotations

import json
from typing import Any, Dict

import polars as pl

__all__ = ["plotly_bar", "plotly_line", "plotly_heatmap"]


def plotly_bar(
    frame: pl.DataFrame,
    x: str,
    y: str,
    title: str = "",
    x_title: str | None = None,
    y_title: str | None = None,
    orientation: str = "v",
    category_order: str | None = None,
    category_array: list[str] | None = None,
    margin_left: int | None = None,
    reverse_category: bool = False,
    show_legend: bool = False,
    legend_bottom: bool = False,
    trace_name: str | None = None,
    margin_bottom: int | None = None,
    xaxis_title_standoff: int | None = None,
    height: int | None = None,
) -> Dict[str, Any]:
    """Build a simple bar chart spec for Plotly embedding."""
    if frame.is_empty():
        return {"data": [], "layout": {"title": title}}
    if orientation == "h":
        x_values = frame[x].to_list()  # keep numeric ordering
        y_values = frame[y].cast(pl.Utf8).to_list()
        xaxis_cfg = {"title": x_title or x, "type": "linear"}
        yaxis_cfg = {"title": y_title or y, "type": "category"}
        if category_order:
            yaxis_cfg["categoryorder"] = category_order
        if category_array:
            yaxis_cfg["categoryarray"] = category_array
        if reverse_category:
            yaxis_cfg["autorange"] = "reversed"
    else:
        x_values = frame[x].cast(pl.Utf8).to_list()
        y_values = frame[y].to_list()
        xaxis_cfg = {"title": x_title or x, "type": "category"}
        yaxis_cfg = {"title": y_title or y}
        if category_order:
            xaxis_cfg["categoryorder"] = category_order
        if category_array:
            xaxis_cfg["categoryarray"] = category_array
    if xaxis_title_standoff is not None:
        xaxis_cfg["title"] = {"text": xaxis_cfg.get("title"), "standoff": xaxis_title_standoff}
    layout: Dict[str, Any] = {
        "title": title,
        "xaxis": xaxis_cfg,
        "yaxis": yaxis_cfg,
        "showlegend": show_legend or legend_bottom,
    }
    margin: Dict[str, Any] = {}
    if margin_left is not None:
        margin["l"] = margin_left
    if margin_bottom is not None:
        margin["b"] = margin_bottom
    if margin:
        layout["margin"] = margin
    if legend_bottom:
        layout["legend"] = {
            "orientation": "h",
            "y": -0.35,
            "x": 0.5,
            "xanchor": "center",
        }
    if height is not None:
        layout["height"] = height
    return {
        "data": [
            {
                "type": "bar",
                "x": x_values,
                "y": y_values,
                "orientation": orientation,
                "name": trace_name or y_title or y,
            }
        ],
        "layout": layout,
    }


def plotly_line(
    frame: pl.DataFrame,
    x: str,
    y: str,
    title: str = "",
    x_title: str | None = None,
    y_title: str | None = None,
    show_legend: bool = False,
    legend_bottom: bool = False,
    trace_name: str | None = None,
    margin_bottom: int | None = None,
    xaxis_title_standoff: int | None = None,
) -> Dict[str, Any]:
    """折れ線図の仕様を返す。"""

    if frame.is_empty():
        return {"data": [], "layout": {"title": title}}
    x_values = frame[x].cast(pl.Utf8).to_list()
    xaxis_cfg: Dict[str, Any] = {"title": x_title or x, "type": "category"}
    if xaxis_title_standoff is not None:
        xaxis_cfg["title"] = {"text": xaxis_cfg.get("title"), "standoff": xaxis_title_standoff}
    layout: Dict[str, Any] = {
        "title": title,
        "xaxis": xaxis_cfg,
        "yaxis": {"title": y_title or y},
        "showlegend": show_legend or legend_bottom,
    }
    if legend_bottom:
        layout["legend"] = {
            "orientation": "h",
            "y": -0.35,
            "x": 0.5,
            "xanchor": "center",
        }
    if margin_bottom is not None:
        layout["margin"] = {"b": margin_bottom}
    return {
        "data": [
            {
                "type": "scatter",
                "mode": "lines+markers",
                "x": x_values,
                "y": frame[y].to_list(),
                "name": trace_name or y_title or y,
            }
        ],
        "layout": layout,
    }


def plotly_heatmap(
    x: list[Any],
    y: list[Any],
    z: list[list[Any]],
    title: str = "",
    x_title: str | None = None,
    y_title: str | None = None,
) -> Dict[str, Any]:
    """Build a heatmap spec with a white→orange→red scale."""

    max_value = 0
    if z:
        max_value = max(max(row) if row else 0 for row in z)
    zmax = max(max_value, 1)
    colorscale = [
        [0.0, "rgb(255,255,255)"],   # white at 0
        [0.5, "rgb(255,165,0)"],     # orange at 50%
        [1.0, "rgb(200,0,0)"],       # deep red at max
    ]
    return {
        "data": [
            {
                "type": "heatmap",
                "x": [str(v) for v in x],
                "y": [str(v) for v in y],
                "z": z,
                "colorscale": colorscale,
                "zmin": 0,
                "zmax": zmax,
                "zauto": False,
            }
        ],
        "layout": {
            "title": title,
            "xaxis": {"title": x_title or "x"},
            "yaxis": {"title": y_title or "y"},
        },
    }
