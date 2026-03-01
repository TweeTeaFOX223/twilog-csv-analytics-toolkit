# twilog-csv-analytics-toolkit
X(旧Twitter)のツイートを保存する「Twilog」から出力した自分のポストデータを分析するGUIアプリ。htmx・FastAPI。自分のTwitter歴の振り返りに有用。

## uvでの実行方法
- 依存インストール: `uv sync`
- 開発サーバ: `uv run uvicorn app.main:app --reload`
- テスト実行: `uv run pytest -q`

## プロジェクト構成と動作概要

### ディレクトリ構成（役割）
- `app/main.py`  
  FastAPIアプリのエントリポイント。`/static` をマウントし、`app/routes/views.py` のルーターを登録。
- `app/routes/views.py`  
  画面ルーティングとHTMX用エンドポイント。CSVアップロード処理、セッション内オプション更新、各分析パネル（`/partials/*`）のレスポンス生成を担当。
- `app/templates/`  
  Jinja2テンプレート。`base.html` + 画面テンプレート（`index.html`, `dashboard.html`）+ パーシャル（`partials/*.html`）でSSR/部分更新を実現。
- `app/static/`  
  CSS/フロント補助アセット。
- `src/twilog_analytics/data/`  
  Twilog CSVの読み込み・前処理（列推定、日時/本文/メンション等の整形）。
- `src/twilog_analytics/analysis/`  
  集計ロジック（時系列、単語、ハッシュタグ、メンション、URL等）。FastAPI非依存の純粋な分析関数群。
- `src/twilog_analytics/visualization/`  
  Plotly（インタラクティブ）、Matplotlib（静的）、WordCloud（画像）を生成する可視化ロジック。

### リクエスト時の処理フロー
1. `index.html` でCSVと設定を送信。
2. `views.py` がCSVを読み込み、`data` 層で前処理した結果をセッション（`file_id`）に保持。
3. `dashboard.html` を表示し、分析タブはHTMXで `/partials/*` を順次取得。
4. 各 `/partials/*` で `analysis` 層の集計結果を作成し、`visualization` 層でグラフ仕様/画像を生成。
5. パーシャルHTMLを差し替え描画し、ページ全体リロードなしでパネルを切り替え。

### 技術スタック
- ウェブ: FastAPI + Jinja2 + HTMX（SSR + 部分更新）
- DataFrame: Polars
- 形態素解析: SudachiPy + sudachidict-full
- 可視化: Plotly / Matplotlib / WordCloud

## 設定サンプル（辞書/ストップワード/品詞フィルタ）

### キーワード辞書（カテゴリ:語1,語2）
```
Cloudflare: WAF,ゼロトラスト,Workers
Next.js: App Router,ISR,Server Actions
Hono: Hono,Edge,Middleware
Auth: 認証,OAuth,JWT
Database: Postgres,MySQL,Redis
```

### ストップワード（改行区切り）
```
ログ
メモ
やった
対応
調整
バグ
修正
レビュー
リリース
```

### 品詞フィルタ（例: 名詞 または 名詞,動詞）
```
名詞,動詞
```
