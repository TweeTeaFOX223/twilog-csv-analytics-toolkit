from __future__ import annotations

import io
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
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from twilog_analytics.data import loader, preprocessor
from twilog_analytics.analysis import (
    statistics,
    timeseries,
    link_analysis,
    text_analysis,
    topic_analysis,
)
from twilog_analytics.visualization import plotly_charts, wordcloud_viz

WEEKDAY_LABELS = {0: "月", 1: "火", 2: "水", 3: "木", 4: "金", 5: "土", 6: "日"}
WEEKDAY_LABEL_EXPR = pl.col("weekday").cast(pl.Int64, strict=False).map_elements(
    lambda v: WEEKDAY_LABELS.get(int(v), "") if v is not None else None, return_dtype=pl.Utf8
).alias("weekday_label")
WEEKDAY_ORDER_LABELS = ["月", "火", "水", "木", "金", "土", "日"]
WEEKDAY_BASE = pl.DataFrame({"weekday": list(range(7))})

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


def _parse_stopwords(raw: Optional[str]) -> set[str]:
    """改行/カンマ区切りのストップワードをセットに変換する。"""

    if not raw:
        return set()
    parts = re.split(r"[,\n\r]+", raw)
    return {p.strip() for p in parts if p.strip()}


def _build_text_analyzer(
    session: UploadSession, pos_filter_override: Optional[str] = None
) -> text_analysis.TextAnalyzer:
    """Sudachi設定とストップワードを反映したTextAnalyzerを作成する。"""

    opts = session.options or {}
    mode = opts.get("sudachi_mode", "C")
    pos_filter = pos_filter_override if pos_filter_override is not None else opts.get("pos_filter")
    stopwords = _parse_stopwords(opts.get("stopwords"))
    return text_analysis.TextAnalyzer(mode=mode, pos_filter=pos_filter, stopwords=stopwords)


def _weekday_full_counts(counts_df: pl.DataFrame, value_col: str = "posts") -> pl.DataFrame:
    """曜日が欠けても順序付きで7件そろえる（空きは0で埋める）。"""

    if counts_df.is_empty():
        filled = WEEKDAY_BASE.with_columns(pl.lit(0).alias(value_col))
    else:
        filled = WEEKDAY_BASE.join(counts_df, on="weekday", how="left").fill_null(0)
    return filled.with_columns(WEEKDAY_LABEL_EXPR)


def _clamp_max_words(value: int, min_value: int = 10, max_value: int = 400) -> int:
    """ワードクラウドの最大単語数を安全な範囲に丸める。"""

    if value < min_value:
        return min_value
    if value > max_value:
        return max_value
    return value


def _clamp_max_minutes(value: int, min_value: int = 30, max_value: int = 1440) -> int:
    """投稿間隔の最大分数を安全な範囲に丸める。"""

    if value < min_value:
        return min_value
    if value > max_value:
        return max_value
    return value


def _exclude_retweets(frame: pl.DataFrame, exclude_retweets: bool) -> pl.DataFrame:
    """RT投稿を除外したDataFrameを返す。"""

    if not exclude_retweets:
        return frame
    if "text" not in frame.columns:
        return frame
    return frame.filter(
        ~pl.col("text").cast(pl.Utf8).fill_null("").str.contains(r"^RT\s*@?")
    )


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
        "keyword_dict": None,
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
    counts = statistics.average_counts(frame)
    yearly = statistics.average_yearly_counts(frame).with_columns(
        pl.col("year").cast(pl.Utf8).alias("year_label")
    )
    monthly = statistics.average_monthly_counts(frame).with_columns(
        (
            pl.col("year").cast(pl.Utf8) + "年" + pl.col("month").cast(pl.Utf8) + "月"
        ).alias("year_month_label")
    )
    weekday_counts_df = _weekday_full_counts(statistics.average_weekday_counts(frame), value_col="avg_posts")

    yearly_plot = plotly_charts.plotly_bar(
        yearly,
        x="year_label",
        y="avg_posts",
        title="年別平均投稿数",
        x_title="年",
        y_title="平均投稿数",
    )
    monthly_plot = plotly_charts.plotly_bar(
        monthly,
        x="year_month_label",
        y="avg_posts",
        title="月別平均投稿数",
        x_title="年月",
        y_title="平均投稿数",
        margin_bottom=110,
        xaxis_title_standoff=30,
    )
    weekday_plot = plotly_charts.plotly_bar(
        weekday_counts_df,
        x="weekday_label",
        y="avg_posts",
        title="曜日別平均投稿数",
        x_title="曜日",
        y_title="平均投稿数",
        category_order="array",
        category_array=WEEKDAY_ORDER_LABELS,
    )

    yearly_rows = yearly.to_dicts() if not yearly.is_empty() else []
    monthly_rows = monthly.to_dicts() if not monthly.is_empty() else []
    weekday_rows = weekday_counts_df.to_dicts() if not weekday_counts_df.is_empty() else []

    return TEMPLATES.TemplateResponse(
        "partials/summary.html",
        {
            "request": request,
            "file_id": file_id,
            "counts": counts,
            "yearly": yearly_rows,
            "monthly": monthly_rows,
            "weekday": weekday_rows,
            "yearly_plot": json.dumps(yearly_plot, default=str),
            "monthly_plot": json.dumps(monthly_plot, default=str),
            "weekday_plot": json.dumps(weekday_plot, default=str),
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
    weekday = _weekday_full_counts(statistics.weekday_counts(frame))
    daily = timeseries.daily_counts(frame).with_columns(pl.col("date").cast(pl.Utf8))
    hours, weekdays, z_matrix = timeseries.weekday_hour_matrix(frame)
    daily_plot = plotly_charts.plotly_line(
        daily,
        x="date",
        y="posts",
        title="日別投稿数",
        x_title="日付",
        y_title="投稿数",
        legend_bottom=True,
        trace_name="投稿数",
        margin_bottom=90,
        xaxis_title_standoff=30,
    )
    hourly_plot = plotly_charts.plotly_bar(
        hourly,
        x="hour",
        y="posts",
        title="時間帯別投稿数",
        x_title="時刻",
        y_title="投稿数",
        legend_bottom=True,
        trace_name="投稿数",
        margin_bottom=80,
    )
    heatmap_plot = plotly_charts.plotly_heatmap(
        hours, weekdays, z_matrix, title="曜日×時間ヒートマップ", x_title="時刻", y_title="曜日"
    )
    weekday_plot = plotly_charts.plotly_bar(
        weekday,
        x="weekday_label",
        y="posts",
        title="曜日別投稿数",
        x_title="曜日",
        y_title="投稿数",
        category_order="array",
        category_array=WEEKDAY_ORDER_LABELS,
        legend_bottom=True,
        trace_name="投稿数",
        margin_bottom=80,
    )

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
            "daily_plot": json.dumps(daily_plot, default=str),
            "hourly_plot": json.dumps(hourly_plot, default=str),
            "heatmap_plot": json.dumps(heatmap_plot, default=str),
            "weekday_plot": json.dumps(weekday_plot, default=str),
        },
    )


@router.get("/partials/weekly", response_class=HTMLResponse)
async def weekly_partial(request: Request, file_id: str) -> HTMLResponse:
    """週別投稿数と日別カレンダーを返す。"""

    try:
        session = _get_session(file_id)
    except HTTPException:
        return _render_error(request, "セッションが見つかりません", 404)

    frame = _filtered_frame(session)
    weekly = timeseries.weekly_counts(frame)
    daily_full = timeseries.daily_counts_full(frame).with_columns(pl.col("date").cast(pl.Utf8))
    week_labels, weekday_labels, calendar_matrix = timeseries.daily_calendar_matrix(frame)

    weekly_plot = plotly_charts.plotly_bar(
        weekly,
        x="iso_week",
        y="posts",
        title="週別投稿数（ISO週）",
        x_title="週",
        y_title="投稿数",
        margin_bottom=90,
        xaxis_title_standoff=30,
    )
    calendar_plot = plotly_charts.plotly_heatmap(
        weekday_labels,
        week_labels,
        calendar_matrix,
        title="日別投稿数カレンダー",
        x_title="曜日",
        y_title="週",
    )

    weekly_rows = weekly.to_dicts() if not weekly.is_empty() else []
    daily_rows = daily_full.to_dicts() if not daily_full.is_empty() else []

    return TEMPLATES.TemplateResponse(
        "partials/weekly.html",
        {
            "request": request,
            "file_id": file_id,
            "weekly": weekly_rows,
            "daily": daily_rows,
            "weekly_plot": json.dumps(weekly_plot, default=str),
            "calendar_plot": json.dumps(calendar_plot, default=str),
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
    domain_rows = domains.to_dicts() if not domains.is_empty() else []
    chart_height = max(400, 24 * len(domain_rows) + 120) if domain_rows else 400
    plot_spec = plotly_charts.plotly_bar(
        domains,
        x="occurrences",
        y="domain",
        title="ドメイン上位",
        x_title="回数",
        y_title="ドメイン",
        orientation="h",
        category_order="array",
        category_array=domains["domain"].cast(pl.Utf8).to_list() if not domains.is_empty() else [],
        reverse_category=True,
        margin_left=200,
        height=chart_height,
    )

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


@router.get("/partials/tlds", response_class=HTMLResponse)
async def tlds_partial(request: Request, file_id: str) -> HTMLResponse:
    """TLD別の分布を返す。"""

    try:
        session = _get_session(file_id)
    except HTTPException:
        return _render_error(request, "セッションが見つかりません", 404)

    frame = _filtered_frame(session)
    tlds = link_analysis.tld_distribution(frame, top_n=20)
    tld_rows = tlds.to_dicts() if not tlds.is_empty() else []
    chart_height = max(400, 24 * len(tld_rows) + 120) if tld_rows else 400
    plot_spec = plotly_charts.plotly_bar(
        tlds,
        x="occurrences",
        y="tld",
        title="TLD分布",
        x_title="回数",
        y_title="TLD",
        orientation="h",
        category_order="array",
        category_array=tlds["tld"].cast(pl.Utf8).to_list() if not tlds.is_empty() else [],
        reverse_category=True,
        margin_left=140,
        height=chart_height,
    )
    plot_json = json.dumps(plot_spec)

    return TEMPLATES.TemplateResponse(
        "partials/tlds.html",
        {
            "request": request,
            "file_id": file_id,
            "tlds": tld_rows,
            "plot_json": plot_json,
        },
    )


@router.get("/partials/word_ranking", response_class=HTMLResponse)
async def word_ranking_partial(request: Request, file_id: str) -> HTMLResponse:
    """単語ランキングを返す。"""

    try:
        session = _get_session(file_id)
    except HTTPException:
        return _render_error(request, "セッションが見つかりません", 404)

    frame = _filtered_frame(session)
    analyzer = _build_text_analyzer(session)
    word_freq = analyzer.get_word_frequency(frame, text_column="text")
    top_words = analyzer.get_top_words(word_freq, top_n=50)
    rows = top_words.to_dicts() if not top_words.is_empty() else []

    return TEMPLATES.TemplateResponse(
        "partials/word_ranking.html",
        {"request": request, "file_id": file_id, "rows": rows},
    )


@router.get("/partials/hashtags", response_class=HTMLResponse)
async def hashtags_partial(request: Request, file_id: str) -> HTMLResponse:
    """ハッシュタグランキングを返す。"""

    try:
        session = _get_session(file_id)
    except HTTPException:
        return _render_error(request, "セッションが見つかりません", 404)

    frame = _filtered_frame(session)
    hashtags = text_analysis.hashtag_ranking(frame, top_n=30)
    hashtag_rows = hashtags.to_dicts() if not hashtags.is_empty() else []
    chart_height = max(400, 24 * len(hashtag_rows) + 120) if hashtag_rows else 400
    plot_frame = hashtags.with_columns(
        pl.concat_str([pl.lit("#"), pl.col("hashtag")]).alias("label")
    ) if not hashtags.is_empty() else hashtags
    plot_spec = plotly_charts.plotly_bar(
        plot_frame,
        x="occurrences",
        y="label",
        title="ハッシュタグ上位",
        x_title="回数",
        y_title="ハッシュタグ",
        orientation="h",
        category_order="array",
        category_array=plot_frame["label"].cast(pl.Utf8).to_list() if not plot_frame.is_empty() else [],
        reverse_category=True,
        margin_left=200,
        height=chart_height,
    )
    plot_json = json.dumps(plot_spec)

    return TEMPLATES.TemplateResponse(
        "partials/hashtags.html",
        {
            "request": request,
            "file_id": file_id,
            "hashtags": hashtag_rows,
            "plot_json": plot_json,
        },
    )


@router.get("/partials/wordcloud", response_class=HTMLResponse)
async def wordcloud_partial(request: Request, file_id: str, max_words: int = 150) -> HTMLResponse:
    """名詞ワードクラウドを返す。"""

    try:
        _get_session(file_id)
    except HTTPException:
        return _render_error(request, "セッションが見つかりません", 404)

    max_words = _clamp_max_words(max_words)
    return TEMPLATES.TemplateResponse(
        "partials/wordcloud.html",
        {"request": request, "file_id": file_id, "max_words": max_words},
    )


@router.get("/partials/wordcloud_nv", response_class=HTMLResponse)
async def wordcloud_nv_partial(
    request: Request, file_id: str, max_words: int = 150
) -> HTMLResponse:
    """名詞+動詞ワードクラウドを返す。"""

    try:
        _get_session(file_id)
    except HTTPException:
        return _render_error(request, "セッションが見つかりません", 404)

    max_words = _clamp_max_words(max_words)
    return TEMPLATES.TemplateResponse(
        "partials/wordcloud_nv.html",
        {"request": request, "file_id": file_id, "max_words": max_words},
    )


@router.get("/partials/tfidf", response_class=HTMLResponse)
async def tfidf_partial(request: Request, file_id: str) -> HTMLResponse:
    """TF-IDFランキングを返す。"""

    try:
        session = _get_session(file_id)
    except HTTPException:
        return _render_error(request, "セッションが見つかりません", 404)

    frame = _filtered_frame(session)
    analyzer = _build_text_analyzer(session)
    tfidf_df = analyzer.get_tfidf_ranking(frame, text_column="text", top_n=50)
    rows = tfidf_df.to_dicts() if not tfidf_df.is_empty() else []

    return TEMPLATES.TemplateResponse(
        "partials/tfidf.html",
        {"request": request, "file_id": file_id, "rows": rows},
    )


@router.get("/partials/intervals", response_class=HTMLResponse)
async def intervals_partial(
    request: Request, file_id: str, max_minutes: int = 720
) -> HTMLResponse:
    """投稿間隔とセッション長分布を返す。"""

    try:
        session = _get_session(file_id)
    except HTTPException:
        return _render_error(request, "セッションが見つかりません", 404)

    frame = _filtered_frame(session)
    max_minutes = _clamp_max_minutes(max_minutes)
    intervals = statistics.posting_interval_distribution(
        frame, bin_size=30, max_minutes=max_minutes
    )
    sessions = statistics.session_length_distribution(frame, gap_minutes=30)

    interval_plot = plotly_charts.plotly_bar(
        intervals,
        x="bin_label",
        y="posts",
        title="投稿間隔分布（分）",
        x_title="間隔（分）",
        y_title="投稿数",
        margin_bottom=90,
        xaxis_title_standoff=20,
    )
    session_plot = plotly_charts.plotly_bar(
        sessions,
        x="session_size",
        y="sessions",
        title="連投セッション長分布",
        x_title="セッション内投稿数",
        y_title="セッション数",
        category_order="array",
        category_array=sessions["session_size"].cast(pl.Utf8).to_list()
        if not sessions.is_empty()
        else [],
        margin_bottom=80,
    )

    interval_rows = (
        intervals.select(["bin_label", "posts"]).to_dicts() if not intervals.is_empty() else []
    )
    session_rows = (
        sessions.select(["session_size", "sessions"]).to_dicts()
        if not sessions.is_empty()
        else []
    )

    return TEMPLATES.TemplateResponse(
        "partials/intervals.html",
        {
            "request": request,
            "file_id": file_id,
            "intervals": interval_rows,
            "sessions": session_rows,
            "interval_plot": json.dumps(interval_plot, default=str),
            "session_plot": json.dumps(session_plot, default=str),
            "max_minutes": max_minutes,
        },
    )


@router.get("/partials/urls", response_class=HTMLResponse)
async def urls_partial(request: Request, file_id: str) -> HTMLResponse:
    """URL含有率とURL数分布を返す。"""

    try:
        session = _get_session(file_id)
    except HTTPException:
        return _render_error(request, "セッションが見つかりません", 404)

    frame = _filtered_frame(session)
    url_stats = statistics.url_presence_stats(frame)
    url_counts = statistics.url_count_distribution(frame)
    url_counts_plot = plotly_charts.plotly_bar(
        url_counts.with_columns(pl.col("url_count").cast(pl.Utf8).alias("url_count_label"))
        if not url_counts.is_empty()
        else url_counts,
        x="url_count_label",
        y="posts",
        title="URL数分布",
        x_title="URL数",
        y_title="投稿数",
        category_order="array",
        category_array=url_counts["url_count"].cast(pl.Utf8).to_list()
        if not url_counts.is_empty()
        else [],
        margin_bottom=80,
    )

    url_rows = url_counts.to_dicts() if not url_counts.is_empty() else []

    return TEMPLATES.TemplateResponse(
        "partials/urls.html",
        {
            "request": request,
            "file_id": file_id,
            "url_stats": url_stats,
            "url_counts": url_rows,
            "url_counts_plot": json.dumps(url_counts_plot, default=str),
        },
    )


@router.get("/partials/long_texts", response_class=HTMLResponse)
async def long_texts_partial(request: Request, file_id: str) -> HTMLResponse:
    """長文ランキングを返す。"""

    try:
        session = _get_session(file_id)
    except HTTPException:
        return _render_error(request, "セッションが見つかりません", 404)

    frame = _filtered_frame(session)
    ranking = statistics.long_text_ranking(frame, top_n=30)
    rows = ranking.to_dicts() if not ranking.is_empty() else []

    return TEMPLATES.TemplateResponse(
        "partials/long_texts.html",
        {
            "request": request,
            "file_id": file_id,
            "rows": rows,
        },
    )


@router.get("/partials/lengths", response_class=HTMLResponse)
async def lengths_partial(request: Request, file_id: str) -> HTMLResponse:
    """文字数分布を返す。"""

    try:
        session = _get_session(file_id)
    except HTTPException:
        return _render_error(request, "セッションが見つかりません", 404)

    frame = _filtered_frame(session)
    lengths = statistics.text_length_distribution(frame, bin_size=20)
    length_rows = (
        lengths.select(["bin_label", "posts"]).to_dicts() if not lengths.is_empty() else []
    )
    plot_spec = plotly_charts.plotly_bar(
        lengths,
        x="bin_label",
        y="posts",
        title="文字数分布",
        x_title="文字数帯",
        y_title="投稿数",
        category_order="array",
        category_array=lengths["bin_label"].cast(pl.Utf8).to_list() if not lengths.is_empty() else [],
        margin_bottom=90,
        xaxis_title_standoff=20,
    )
    plot_json = json.dumps(plot_spec)

    return TEMPLATES.TemplateResponse(
        "partials/lengths.html",
        {
            "request": request,
            "file_id": file_id,
            "lengths": length_rows,
            "plot_json": plot_json,
        },
    )


@router.get("/partials/path_depth", response_class=HTMLResponse)
async def path_depth_partial(request: Request, file_id: str) -> HTMLResponse:
    """URLパス深さの分布を返す。"""

    try:
        session = _get_session(file_id)
    except HTTPException:
        return _render_error(request, "セッションが見つかりません", 404)

    frame = _filtered_frame(session)
    depth_df = link_analysis.path_depth_distribution(frame)
    rows = depth_df.to_dicts() if not depth_df.is_empty() else []
    plot_spec = plotly_charts.plotly_bar(
        depth_df.with_columns(pl.col("depth").cast(pl.Utf8).alias("depth_label"))
        if not depth_df.is_empty()
        else depth_df,
        x="depth_label",
        y="occurrences",
        title="URLパス深さ分布",
        x_title="パス深さ",
        y_title="件数",
        category_order="array",
        category_array=depth_df["depth"].cast(pl.Utf8).to_list() if not depth_df.is_empty() else [],
        margin_bottom=80,
    )

    return TEMPLATES.TemplateResponse(
        "partials/path_depth.html",
        {
            "request": request,
            "file_id": file_id,
            "rows": rows,
            "plot_json": json.dumps(plot_spec, default=str),
        },
    )


@router.get("/partials/domain_month", response_class=HTMLResponse)
async def domain_month_partial(
    request: Request, file_id: str, domain: Optional[str] = None
) -> HTMLResponse:
    """ドメイン×月の推移を返す。"""

    try:
        session = _get_session(file_id)
    except HTTPException:
        return _render_error(request, "セッションが見つかりません", 404)

    frame = _filtered_frame(session)
    top_domains = link_analysis.domain_ranking(frame, top_n=10)
    domain_choices = top_domains["domain"].cast(pl.Utf8).to_list() if not top_domains.is_empty() else []
    selected = domain or (domain_choices[0] if domain_choices else "")
    trend = link_analysis.domain_month_trend(frame, selected) if selected else pl.DataFrame()
    rows = trend.to_dicts() if not trend.is_empty() else []
    plot_spec = plotly_charts.plotly_line(
        trend,
        x="year_month",
        y="occurrences",
        title="ドメイン月次推移",
        x_title="年月",
        y_title="投稿数",
        legend_bottom=True,
        trace_name=selected or "投稿数",
        margin_bottom=90,
        xaxis_title_standoff=30,
    )

    return TEMPLATES.TemplateResponse(
        "partials/domain_month.html",
        {
            "request": request,
            "file_id": file_id,
            "domains": domain_choices,
            "selected": selected,
            "rows": rows,
            "plot_json": json.dumps(plot_spec, default=str),
        },
    )


@router.get("/partials/mentions", response_class=HTMLResponse)
async def mentions_partial(
    request: Request, file_id: str, exclude_retweets: bool = False
) -> HTMLResponse:
    """メンションランキングを返す。"""

    try:
        session = _get_session(file_id)
    except HTTPException:
        return _render_error(request, "セッションが見つかりません", 404)

    frame = _exclude_retweets(_filtered_frame(session), exclude_retweets)
    mentions = text_analysis.mention_ranking(frame, top_n=30)
    rows = mentions.to_dicts() if not mentions.is_empty() else []
    plot_frame = (
        mentions.with_columns(pl.concat_str([pl.lit("@"), pl.col("mention")]).alias("label"))
        if not mentions.is_empty()
        else mentions
    )
    plot_spec = plotly_charts.plotly_bar(
        plot_frame,
        x="occurrences",
        y="label",
        title="メンション上位",
        x_title="回数",
        y_title="メンション",
        orientation="h",
        category_order="array",
        category_array=plot_frame["label"].cast(pl.Utf8).to_list()
        if not plot_frame.is_empty()
        else [],
        reverse_category=True,
        margin_left=200,
    )

    return TEMPLATES.TemplateResponse(
        "partials/mentions.html",
        {
            "request": request,
            "file_id": file_id,
            "rows": rows,
            "plot_json": json.dumps(plot_spec, default=str),
            "exclude_retweets": exclude_retweets,
        },
    )


@router.get("/partials/mention_weekday", response_class=HTMLResponse)
async def mention_weekday_partial(request: Request, file_id: str) -> HTMLResponse:
    """メンション×曜日の分布を返す。"""

    try:
        session = _get_session(file_id)
    except HTTPException:
        return _render_error(request, "セッションが見つかりません", 404)

    frame = _filtered_frame(session)
    mention_df = text_analysis.mention_weekday_counts(frame)
    totals = (
        mention_df.group_by("weekday").agg(pl.col("occurrences").sum().alias("occurrences"))
        if not mention_df.is_empty()
        else mention_df
    )
    totals = totals.with_columns(WEEKDAY_LABEL_EXPR) if not totals.is_empty() else totals
    rows = totals.to_dicts() if not totals.is_empty() else []
    plot_spec = plotly_charts.plotly_bar(
        totals,
        x="weekday_label",
        y="occurrences",
        title="曜日別メンション数",
        x_title="曜日",
        y_title="回数",
        category_order="array",
        category_array=WEEKDAY_ORDER_LABELS,
        margin_bottom=80,
    )

    return TEMPLATES.TemplateResponse(
        "partials/mention_weekday.html",
        {
            "request": request,
            "file_id": file_id,
            "rows": rows,
            "plot_json": json.dumps(plot_spec, default=str),
        },
    )


@router.get("/partials/hashtag_years", response_class=HTMLResponse)
async def hashtag_years_partial(request: Request, file_id: str) -> HTMLResponse:
    """ハッシュタグ×年の推移を返す。"""

    try:
        session = _get_session(file_id)
    except HTTPException:
        return _render_error(request, "セッションが見つかりません", 404)

    frame = _filtered_frame(session)
    trend = text_analysis.hashtag_year_trend(frame, top_n=8)
    rows = trend.to_dicts() if not trend.is_empty() else []
    plot_spec = plotly_charts.plotly_multi_line(
        trend,
        x="year",
        y="occurrences",
        series="hashtag",
        title="ハッシュタグ年次推移",
        x_title="年",
        y_title="投稿数",
    )

    return TEMPLATES.TemplateResponse(
        "partials/hashtag_years.html",
        {
            "request": request,
            "file_id": file_id,
            "rows": rows,
            "plot_json": json.dumps(plot_spec, default=str),
        },
    )


@router.get("/partials/hashtag_cooccurrence", response_class=HTMLResponse)
async def hashtag_cooccurrence_partial(request: Request, file_id: str) -> HTMLResponse:
    """ハッシュタグ共起のランキングと簡易ネットワークを返す。"""

    try:
        session = _get_session(file_id)
    except HTTPException:
        return _render_error(request, "セッションが見つかりません", 404)

    frame = _filtered_frame(session)
    pairs = text_analysis.hashtag_cooccurrence(frame, top_n=40, min_count=2)
    pair_rows = pairs.to_dicts() if not pairs.is_empty() else []
    nodes = sorted({row["tag_a"] for row in pair_rows} | {row["tag_b"] for row in pair_rows})
    edges = [
        {"source": row["tag_a"], "target": row["tag_b"], "weight": row["count"]}
        for row in pair_rows
    ]
    network_plot = plotly_charts.plotly_network(nodes, edges, title="ハッシュタグ共起ネットワーク")

    return TEMPLATES.TemplateResponse(
        "partials/hashtag_cooccurrence.html",
        {
            "request": request,
            "file_id": file_id,
            "pairs": pair_rows,
            "network_plot": json.dumps(network_plot, default=str),
        },
    )


@router.get("/partials/keyword_categories", response_class=HTMLResponse)
async def keyword_categories_partial(request: Request, file_id: str) -> HTMLResponse:
    """キーワード辞書のカテゴリ別投稿数を返す。"""

    try:
        session = _get_session(file_id)
    except HTTPException:
        return _render_error(request, "セッションが見つかりません", 404)

    frame = _filtered_frame(session)
    raw_dict = (session.options or {}).get("keyword_dict")
    keyword_dict = text_analysis.parse_keyword_dictionary(raw_dict)
    counts = text_analysis.keyword_category_counts(frame, keyword_dict)
    rows = counts.to_dicts() if not counts.is_empty() else []
    plot_spec = plotly_charts.plotly_bar(
        counts,
        x="category",
        y="posts",
        title="カテゴリ別投稿数",
        x_title="カテゴリ",
        y_title="投稿数",
        category_order="array",
        category_array=counts["category"].cast(pl.Utf8).to_list() if not counts.is_empty() else [],
        margin_bottom=90,
        xaxis_title_standoff=20,
    )

    return TEMPLATES.TemplateResponse(
        "partials/keyword_categories.html",
        {
            "request": request,
            "file_id": file_id,
            "rows": rows,
            "raw_dict": raw_dict or "",
            "plot_json": json.dumps(plot_spec, default=str),
        },
    )


@router.get("/partials/word_trend", response_class=HTMLResponse)
async def word_trend_partial(
    request: Request, file_id: str, term: str | None = None
) -> HTMLResponse:
    """指定語の月次出現推移を返す。"""

    try:
        session = _get_session(file_id)
    except HTTPException:
        return _render_error(request, "セッションが見つかりません", 404)

    frame = _filtered_frame(session)
    analyzer = _build_text_analyzer(session)
    term_value = term.strip() if term else ""
    monthly = (
        text_analysis.word_monthly_counts(frame, term_value, analyzer)
        if term_value
        else pl.DataFrame()
    )
    rows = monthly.to_dicts() if not monthly.is_empty() else []
    plot_spec = plotly_charts.plotly_line(
        monthly,
        x="year_month",
        y="posts",
        title="単語の月次推移",
        x_title="年月",
        y_title="投稿数",
        legend_bottom=True,
        trace_name=term_value or "出現数",
        margin_bottom=90,
        xaxis_title_standoff=30,
    )

    return TEMPLATES.TemplateResponse(
        "partials/word_trend.html",
        {
            "request": request,
            "file_id": file_id,
            "term": term_value,
            "rows": rows,
            "plot_json": json.dumps(plot_spec, default=str),
        },
    )


@router.get("/partials/word_cooccurrence", response_class=HTMLResponse)
async def word_cooccurrence_partial(
    request: Request, file_id: str, term: str | None = None
) -> HTMLResponse:
    """指定語の共起語ランキングを返す。"""

    try:
        session = _get_session(file_id)
    except HTTPException:
        return _render_error(request, "セッションが見つかりません", 404)

    frame = _filtered_frame(session)
    analyzer = _build_text_analyzer(session)
    term_value = term.strip() if term else ""
    cooccur = (
        text_analysis.word_cooccurrence_for_term(frame, analyzer, term_value, top_n=30)
        if term_value
        else pl.DataFrame()
    )
    rows = cooccur.to_dicts() if not cooccur.is_empty() else []
    plot_spec = plotly_charts.plotly_bar(
        cooccur,
        x="word",
        y="count",
        title="共起語ランキング",
        x_title="語",
        y_title="回数",
        category_order="array",
        category_array=cooccur["word"].cast(pl.Utf8).to_list() if not cooccur.is_empty() else [],
        margin_bottom=90,
        xaxis_title_standoff=20,
    )

    return TEMPLATES.TemplateResponse(
        "partials/word_cooccurrence.html",
        {
            "request": request,
            "file_id": file_id,
            "term": term_value,
            "rows": rows,
            "plot_json": json.dumps(plot_spec, default=str),
        },
    )


@router.get("/partials/word_network", response_class=HTMLResponse)
async def word_network_partial(request: Request, file_id: str) -> HTMLResponse:
    """語の共起ネットワークを返す。"""

    try:
        session = _get_session(file_id)
    except HTTPException:
        return _render_error(request, "セッションが見つかりません", 404)

    frame = _filtered_frame(session)
    analyzer = _build_text_analyzer(session)
    pairs = text_analysis.word_cooccurrence_pairs(frame, analyzer, top_n=40, min_count=2)
    rows = pairs.to_dicts() if not pairs.is_empty() else []
    nodes = sorted({row["word_a"] for row in rows} | {row["word_b"] for row in rows})
    edges = [
        {"source": row["word_a"], "target": row["word_b"], "weight": row["count"]}
        for row in rows
    ]
    network_plot = plotly_charts.plotly_network(nodes, edges, title="語の共起ネットワーク")

    return TEMPLATES.TemplateResponse(
        "partials/word_network.html",
        {
            "request": request,
            "file_id": file_id,
            "rows": rows,
            "plot_json": json.dumps(network_plot, default=str),
        },
    )


@router.get("/partials/sudachi_compare", response_class=HTMLResponse)
async def sudachi_compare_partial(request: Request, file_id: str) -> HTMLResponse:
    """Sudachi分割モード比較を返す。"""

    try:
        session = _get_session(file_id)
    except HTTPException:
        return _render_error(request, "セッションが見つかりません", 404)

    frame = _filtered_frame(session)
    stopwords = _parse_stopwords((session.options or {}).get("stopwords"))
    pos_filter = (session.options or {}).get("pos_filter")

    analyzer_a = text_analysis.TextAnalyzer(mode="A", pos_filter=pos_filter, stopwords=stopwords)
    analyzer_b = text_analysis.TextAnalyzer(mode="B", pos_filter=pos_filter, stopwords=stopwords)
    analyzer_c = text_analysis.TextAnalyzer(mode="C", pos_filter=pos_filter, stopwords=stopwords)

    freq_a = analyzer_a.get_top_words(analyzer_a.get_word_frequency(frame), top_n=20).to_dicts()
    freq_b = analyzer_b.get_top_words(analyzer_b.get_word_frequency(frame), top_n=20).to_dicts()
    freq_c = analyzer_c.get_top_words(analyzer_c.get_word_frequency(frame), top_n=20).to_dicts()

    return TEMPLATES.TemplateResponse(
        "partials/sudachi_compare.html",
        {
            "request": request,
            "file_id": file_id,
            "rows_a": freq_a,
            "rows_b": freq_b,
            "rows_c": freq_c,
        },
    )


@router.get("/partials/topics_monthly", response_class=HTMLResponse)
async def topics_monthly_partial(request: Request, file_id: str) -> HTMLResponse:
    """月ごとの代表語を返す。"""

    try:
        session = _get_session(file_id)
    except HTTPException:
        return _render_error(request, "セッションが見つかりません", 404)

    frame = _filtered_frame(session)
    analyzer = _build_text_analyzer(session)
    topics = topic_analysis.monthly_tfidf_top(frame, analyzer, top_n=5)
    rows = topics.to_dicts() if not topics.is_empty() else []

    return TEMPLATES.TemplateResponse(
        "partials/topics_monthly.html",
        {"request": request, "file_id": file_id, "rows": rows},
    )


@router.get("/partials/clusters", response_class=HTMLResponse)
async def clusters_partial(
    request: Request, file_id: str, clusters: int = 4
) -> HTMLResponse:
    """クラスタリングの要約を返す。"""

    try:
        session = _get_session(file_id)
    except HTTPException:
        return _render_error(request, "セッションが見つかりません", 404)

    frame = _filtered_frame(session)
    analyzer = _build_text_analyzer(session)
    clusters = max(2, min(clusters, 8))
    summary_df, counts_df = topic_analysis.kmeans_cluster_summary(
        frame, analyzer, k=clusters, max_features=200, max_iter=12
    )
    rows = summary_df.to_dicts() if not summary_df.is_empty() else []
    plot_spec = plotly_charts.plotly_bar(
        counts_df.with_columns(pl.col("cluster_id").cast(pl.Utf8).alias("cluster_label"))
        if not counts_df.is_empty()
        else counts_df,
        x="cluster_label",
        y="size",
        title="クラスタ規模",
        x_title="クラスタ",
        y_title="投稿数",
        category_order="array",
        category_array=counts_df["cluster_id"].cast(pl.Utf8).to_list()
        if not counts_df.is_empty()
        else [],
        margin_bottom=80,
    )

    return TEMPLATES.TemplateResponse(
        "partials/clusters.html",
        {
            "request": request,
            "file_id": file_id,
            "rows": rows,
            "clusters": clusters,
            "plot_json": json.dumps(plot_spec, default=str),
        },
    )


@router.get("/partials/domain_years", response_class=HTMLResponse)
async def domain_years_partial(request: Request, file_id: str) -> HTMLResponse:
    """ドメイン×年の推移を返す。"""

    try:
        session = _get_session(file_id)
    except HTTPException:
        return _render_error(request, "セッションが見つかりません", 404)

    frame = _filtered_frame(session)
    trends = link_analysis.domain_year_trend(frame, top_n=5)
    rows = trends.to_dicts() if not trends.is_empty() else []
    plot_spec = plotly_charts.plotly_multi_line(
        trends,
        x="year",
        y="occurrences",
        series="domain",
        title="ドメイン別 年次推移",
        x_title="年",
        y_title="投稿数",
    )

    return TEMPLATES.TemplateResponse(
        "partials/domain_years.html",
        {
            "request": request,
            "file_id": file_id,
            "rows": rows,
            "plot_json": json.dumps(plot_spec, default=str),
        },
    )


@router.get("/partials/self_reference", response_class=HTMLResponse)
async def self_reference_partial(request: Request, file_id: str) -> HTMLResponse:
    """自己参照URLの比率を返す。"""

    try:
        session = _get_session(file_id)
    except HTTPException:
        return _render_error(request, "セッションが見つかりません", 404)

    frame = _filtered_frame(session)
    stats = link_analysis.self_reference_stats(frame)
    chart_df = pl.DataFrame(
        {
            "label": ["twitter/x参照", "外部リンク"],
            "posts": [stats["self_ref"], stats["external"]],
        }
    )
    plot_spec = plotly_charts.plotly_bar(
        chart_df,
        x="label",
        y="posts",
        title="自己参照URL比率",
        x_title="種別",
        y_title="投稿数",
        category_order="array",
        category_array=["twitter/x参照", "外部リンク"],
        margin_bottom=80,
    )

    return TEMPLATES.TemplateResponse(
        "partials/self_reference.html",
        {
            "request": request,
            "file_id": file_id,
            "stats": stats,
            "plot_json": json.dumps(plot_spec, default=str),
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
    keyword_dict: Optional[str] = Form(None),
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
        "keyword_dict": keyword_dict,
    }
    UPLOAD_STORE[file_id] = session

    return TEMPLATES.TemplateResponse(
        "partials/options_status.html",
        {"request": request, "message": "設定を保存しました。必要に応じてパネルを再読み込みしてください。"},
    )


def _df_to_csv_response(df: pl.DataFrame, filename: str) -> StreamingResponse:
    """Polars DataFrameをCSVで返すレスポンスを作成する。"""

    csv_bytes = df.write_csv().encode("utf-8")
    return StreamingResponse(
        iter([csv_bytes]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/download/{kind}")
async def download_csv(kind: str, file_id: str, term: Optional[str] = None) -> StreamingResponse:
    """集計結果をCSVでダウンロードする。"""

    session = _get_session(file_id)
    frame = _filtered_frame(session)

    if kind == "yearly":
        df = statistics.yearly_counts(frame)
        fname = "yearly_counts.csv"
    elif kind == "monthly":
        df = statistics.monthly_counts(frame)
        fname = "monthly_counts.csv"
    elif kind == "avg_yearly":
        df = statistics.average_yearly_counts(frame)
        fname = "average_yearly_counts.csv"
    elif kind == "avg_monthly":
        df = statistics.average_monthly_counts(frame)
        fname = "average_monthly_counts.csv"
    elif kind == "avg_weekday":
        df = _weekday_full_counts(statistics.average_weekday_counts(frame), value_col="avg_posts")
        fname = "average_weekday_counts.csv"
    elif kind == "hourly":
        df = statistics.hourly_counts(frame)
        fname = "hourly_counts.csv"
    elif kind == "weekday":
        df = _weekday_full_counts(statistics.weekday_counts(frame))
        fname = "weekday_counts.csv"
    elif kind == "weekday_hour":
        df = timeseries.weekday_hour_counts(frame).with_columns(WEEKDAY_LABEL_EXPR)
        fname = "weekday_hour_counts.csv"
    elif kind == "daily":
        df = timeseries.daily_counts(frame)
        fname = "daily_counts.csv"
    elif kind == "weekly":
        df = timeseries.weekly_counts(frame)
        fname = "weekly_counts.csv"
    elif kind == "daily_calendar":
        df = timeseries.daily_counts_full(frame)
        fname = "daily_counts_full.csv"
    elif kind == "domains":
        df = link_analysis.domain_ranking(frame, top_n=None)
        fname = "domain_ranking.csv"
    elif kind == "word_freq":
        analyzer = _build_text_analyzer(session)
        word_freq = analyzer.get_word_frequency(frame, text_column="text")
        df = analyzer.get_top_words(word_freq, top_n=None)
        fname = "word_ranking.csv"
    elif kind == "tfidf":
        analyzer = _build_text_analyzer(session)
        df = analyzer.get_tfidf_ranking(frame, text_column="text", top_n=None)
        fname = "tfidf_ranking.csv"
    elif kind == "hashtags":
        df = text_analysis.hashtag_ranking(frame, top_n=None)
        fname = "hashtag_ranking.csv"
    elif kind == "lengths":
        df = statistics.text_length_distribution(frame, bin_size=20)
        if not df.is_empty():
            df = df.select(["bin_label", "posts"])
        fname = "text_length_distribution.csv"
    elif kind == "tlds":
        df = link_analysis.tld_distribution(frame, top_n=None)
        fname = "tld_distribution.csv"
    elif kind == "intervals":
        df = statistics.posting_interval_distribution(frame, bin_size=30)
        if not df.is_empty():
            df = df.select(["bin_label", "posts"])
        fname = "posting_interval_distribution.csv"
    elif kind == "sessions":
        df = statistics.session_length_distribution(frame, gap_minutes=30)
        fname = "session_length_distribution.csv"
    elif kind == "url_counts":
        df = statistics.url_count_distribution(frame)
        fname = "url_count_distribution.csv"
    elif kind == "url_rate":
        stats = statistics.url_presence_stats(frame)
        df = pl.DataFrame(
            {
                "with_url": [stats["with_url"]],
                "without_url": [stats["without_url"]],
                "rate": [stats["rate"]],
            }
        )
        fname = "url_presence_rate.csv"
    elif kind == "long_texts":
        df = statistics.long_text_ranking(frame, top_n=100)
        fname = "long_text_ranking.csv"
    elif kind == "path_depth":
        df = link_analysis.path_depth_distribution(frame)
        fname = "url_path_depth.csv"
    elif kind == "domain_month":
        df = link_analysis.domain_month_trend(frame, term or "")
        fname = "domain_month_trend.csv"
    elif kind == "mentions":
        df = text_analysis.mention_ranking(frame, top_n=200)
        fname = "mention_ranking.csv"
    elif kind == "mention_weekday":
        df = text_analysis.mention_weekday_counts(frame)
        if not df.is_empty():
            df = df.group_by("weekday").agg(pl.col("occurrences").sum().alias("occurrences"))
        fname = "mention_weekday_counts.csv"
    elif kind == "hashtag_years":
        df = text_analysis.hashtag_year_trend(frame, top_n=20)
        fname = "hashtag_year_trend.csv"
    elif kind == "hashtag_pairs":
        df = text_analysis.hashtag_cooccurrence(frame, top_n=200, min_count=2)
        fname = "hashtag_cooccurrence.csv"
    elif kind == "keyword_categories":
        raw_dict = (session.options or {}).get("keyword_dict")
        keyword_dict = text_analysis.parse_keyword_dictionary(raw_dict)
        df = text_analysis.keyword_category_counts(frame, keyword_dict)
        fname = "keyword_category_counts.csv"
    elif kind == "word_trend":
        analyzer = _build_text_analyzer(session)
        df = text_analysis.word_monthly_counts(frame, term or "", analyzer)
        fname = "word_monthly_counts.csv"
    elif kind == "domain_years":
        df = link_analysis.domain_year_trend(frame, top_n=10)
        fname = "domain_year_trend.csv"
    elif kind == "self_reference":
        stats = link_analysis.self_reference_stats(frame)
        df = pl.DataFrame(
            {
                "self_ref": [stats["self_ref"]],
                "external": [stats["external"]],
                "rate": [stats["rate"]],
            }
        )
        fname = "self_reference_rate.csv"
    elif kind == "word_cooccurrence":
        analyzer = _build_text_analyzer(session)
        df = text_analysis.word_cooccurrence_for_term(frame, analyzer, term or "", top_n=200)
        fname = "word_cooccurrence.csv"
    elif kind == "word_network":
        analyzer = _build_text_analyzer(session)
        df = text_analysis.word_cooccurrence_pairs(frame, analyzer, top_n=200, min_count=2)
        fname = "word_cooccurrence_pairs.csv"
    elif kind == "sudachi_compare":
        stopwords = _parse_stopwords((session.options or {}).get("stopwords"))
        pos_filter = (session.options or {}).get("pos_filter")
        analyzer_a = text_analysis.TextAnalyzer(mode="A", pos_filter=pos_filter, stopwords=stopwords)
        analyzer_b = text_analysis.TextAnalyzer(mode="B", pos_filter=pos_filter, stopwords=stopwords)
        analyzer_c = text_analysis.TextAnalyzer(mode="C", pos_filter=pos_filter, stopwords=stopwords)
        top_a = analyzer_a.get_top_words(analyzer_a.get_word_frequency(frame), top_n=50).with_columns(
            pl.lit("A").alias("mode")
        )
        top_b = analyzer_b.get_top_words(analyzer_b.get_word_frequency(frame), top_n=50).with_columns(
            pl.lit("B").alias("mode")
        )
        top_c = analyzer_c.get_top_words(analyzer_c.get_word_frequency(frame), top_n=50).with_columns(
            pl.lit("C").alias("mode")
        )
        df = pl.concat([top_a, top_b, top_c], how="vertical")
        fname = "sudachi_compare.csv"
    elif kind == "topics_monthly":
        analyzer = _build_text_analyzer(session)
        df = topic_analysis.monthly_tfidf_top(frame, analyzer, top_n=5)
        fname = "monthly_topics.csv"
    elif kind == "clusters":
        analyzer = _build_text_analyzer(session)
        summary_df, _ = topic_analysis.kmeans_cluster_summary(
            frame, analyzer, k=max(2, min(int(term or 4), 8)), max_features=200, max_iter=12
        )
        df = summary_df
        fname = "cluster_summary.csv"
    else:
        raise HTTPException(status_code=400, detail="unknown download kind")

    if df.is_empty():
        raise HTTPException(status_code=404, detail="データがありません")

    return _df_to_csv_response(df, fname)


@router.get("/wordcloud")
async def wordcloud_image(
    file_id: str, max_words: int = 150, pos_filter: Optional[str] = "\u540d\u8a5e"
) -> StreamingResponse:
    """名詞ワードクラウド画像を返す。"""

    session = _get_session(file_id)
    frame = _filtered_frame(session)
    analyzer = _build_text_analyzer(session, pos_filter_override=pos_filter)
    word_freq = analyzer.get_word_frequency(frame, text_column="text")

    generator = wordcloud_viz.WordCloudGenerator()
    max_words = _clamp_max_words(max_words)
    image = generator.generate_wordcloud(word_freq, width=900, height=500, max_words=max_words)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")
