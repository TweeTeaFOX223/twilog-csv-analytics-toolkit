from __future__ import annotations

import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List
from uuid import uuid4

import polars as pl
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from twilog_analytics.data import loader, preprocessor
from twilog_analytics.analysis import statistics, timeseries, link_analysis
from twilog_analytics.visualization import plotly_charts

# テンプレートやルーターを初期化（HTMX用の部分テンプレートを返却する）
router = APIRouter()
TEMPLATES = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))


@dataclass
class UploadSession:
    """アップロード済みファイルのメタ情報と解析済みDataFrameを保持する。"""

    file_id: str
    path: Path
    frame: pl.DataFrame
    options: dict


# メモリ上の簡易セッションストア（本番では永続化を検討）
UPLOAD_STORE: Dict[str, UploadSession] = {}


def _safe_filename(original: str) -> str:
    """アップロードファイル名を簡易サニタイズして安全な名前を返す。"""

    stem = Path(original or "upload").stem
    suffix = Path(original or "upload.csv").suffix or ".csv"
    sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", stem) or "upload"
    return f"{sanitized}{suffix}"


def _render_error(request: Request, message: str, status_code: int = 400) -> HTMLResponse:
    """HTMX部品としてエラーメッセージを返す。"""

    return TEMPLATES.TemplateResponse(
        "partials/error_fragment.html",
        {"request": request, "message": message},
        status_code=status_code,
    )


def _get_session(file_id: str) -> UploadSession:
    """file_idに紐づくアップロードセッションを取得する。"""

    session = UPLOAD_STORE.get(file_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


def _filtered_frame(session: UploadSession) -> pl.DataFrame:
    """オプションに応じてフィルタを適用したDataFrameを返す。"""

    frame = session.frame
    opts = session.options or {}
    if opts.get("years_mode") == "selected" and opts.get("years"):
        frame = preprocessor.filter_by_years(frame, opts.get("years"))
    return frame


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """アップロードフォームを表示するトップページ。"""

    return TEMPLATES.TemplateResponse("index.html", {"request": request})


@router.post("/upload")
async def upload_csv(
    request: Request,
    file: UploadFile = File(...),
):
    """CSVを安全に受け取り、一時ディレクトリに保存して前処理する。"""

    if not file.filename or not file.filename.lower().endswith(".csv"):
        return _render_error(request, "CSVファイルを選択してください。", status_code=400)

    temp_dir = Path(tempfile.mkdtemp(prefix="twilog_"))
    dest = temp_dir / _safe_filename(file.filename)

    # アップロードストリームを安全なパスに書き出す
    with dest.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        frame = loader.load_twilog_csv(dest)
        frame = preprocessor.add_derived_columns(frame)
    except Exception as exc:  # pragma: no cover - UIで表示
        return _render_error(request, f"CSVの読み込みに失敗しました: {exc}", status_code=400)

    file_id = uuid4().hex
    # 初期オプション（ダッシュボード側で設定可能）
    options = {
        "years": [],
        "years_mode": "all",
        "sudachi_mode": "C",
        "pos_filter": None,
        "stopwords": None,
    }
    UPLOAD_STORE[file_id] = UploadSession(file_id=file_id, path=dest, frame=frame, options=options)

    return RedirectResponse(url=f"/dashboard?file_id={file_id}", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, file_id: str) -> HTMLResponse:
    """ダッシュボード骨格（HTMXでパネル差し替え）を返す。"""

    try:
        session = _get_session(file_id)
    except HTTPException:
        return _render_error(request, "アップロードが見つかりません。再度アップロードしてください。", 404)

    available_years = (
        session.frame.select(pl.col("year").drop_nulls().unique().sort()).to_series().to_list()
        if "year" in session.frame.columns
        else []
    )

    return TEMPLATES.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "file_id": file_id,
            "options": session.options,
            "available_years": available_years,
        },
    )


@router.get("/partials/summary", response_class=HTMLResponse)
async def summary_partial(request: Request, file_id: str) -> HTMLResponse:
    """基本統計（総投稿数、年次・月次集計）を返す。"""

    try:
        session = _get_session(file_id)
    except HTTPException:
        return _render_error(request, "セッションが見つかりません", 404)

    frame = _filtered_frame(session)
    counts = statistics.basic_counts(frame)
    yearly = statistics.yearly_counts(frame)
    monthly = statistics.monthly_counts(frame)

    yearly_rows = yearly.to_dicts() if not yearly.is_empty() else []
    monthly_rows = monthly.to_dicts() if not monthly.is_empty() else []

    return TEMPLATES.TemplateResponse(
        "partials/summary.html",
        {
            "request": request,
            "file_id": file_id,
            "counts": counts,
            "yearly": yearly_rows,
            "monthly": monthly_rows,
        },
    )


@router.get("/partials/time", response_class=HTMLResponse)
async def time_partial(request: Request, file_id: str) -> HTMLResponse:
    """時間帯・曜日・日次の集計を返す。"""

    try:
        session = _get_session(file_id)
    except HTTPException:
        return _render_error(request, "セッションが見つかりません", 404)

    frame = _filtered_frame(session)
    hourly = statistics.hourly_counts(frame)
    weekday = statistics.weekday_counts(frame)
    daily = timeseries.daily_counts(frame)

    hourly_rows = hourly.to_dicts() if not hourly.is_empty() else []
    weekday_rows = weekday.to_dicts() if not weekday.is_empty() else []
    daily_rows = daily.to_dicts() if not daily.is_empty() else []

    return TEMPLATES.TemplateResponse(
        "partials/time.html",
        {
            "request": request,
            "file_id": file_id,
            "hourly": hourly_rows,
            "weekday": weekday_rows,
            "daily": daily_rows,
        },
    )


@router.get("/partials/domains", response_class=HTMLResponse)
async def domains_partial(request: Request, file_id: str) -> HTMLResponse:
    """URLドメインのランキングと棒グラフを返す。"""

    try:
        session = _get_session(file_id)
    except HTTPException:
        return _render_error(request, "セッションが見つかりません", 404)

    frame = _filtered_frame(session)
    domains = link_analysis.domain_ranking(frame, top_n=20)
    plot_spec = plotly_charts.plotly_bar(domains, x="domain", y="occurrences", title="Top Domains")

    domain_rows = domains.to_dicts() if not domains.is_empty() else []
    plot_json = json.dumps(plot_spec)

    return TEMPLATES.TemplateResponse(
        "partials/domains.html",
        {
            "request": request,
            "file_id": file_id,
            "domains": domain_rows,
            "plot_json": plot_json,
        },
    )


@router.post("/options", response_class=HTMLResponse)
async def update_options(
    request: Request,
    file_id: str = Form(...),
    years: List[str] = Form([]),
    years_mode: str = Form("all"),
    sudachi_mode: str = Form("C"),
    pos_filter: Optional[str] = Form(None),
    stopwords: Optional[str] = Form(None),
) -> HTMLResponse:
    """ダッシュボード上の設定を更新する。"""

    try:
        session = _get_session(file_id)
    except HTTPException:
        return _render_error(request, "セッションが見つかりません", 404)

    session.options = {
        "years": [int(y) for y in years if str(y).strip().isdigit()] if years else [],
        "years_mode": years_mode,
        "sudachi_mode": sudachi_mode,
        "pos_filter": pos_filter,
        "stopwords": stopwords,
    }
    UPLOAD_STORE[file_id] = session

    return TEMPLATES.TemplateResponse(
        "partials/options_status.html",
        {"request": request, "message": "設定を保存しました。必要に応じてパネルを再読み込みしてください。"},
    )
