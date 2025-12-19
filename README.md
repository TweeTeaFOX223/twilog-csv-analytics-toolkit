# twilog-csv-analytics-toolkit
X(旧Twitter)のツイートを保存する「Twilog」から出力した自分のポストデータを分析するGUIアプリ。htmx・FastAPI。自分のTwitter歴の振り返りに有用。

## uvでの実行方法
- 依存インストール: `uv sync`
- 開発サーバ: `uv run uvicorn app.main:app --reload`
- テスト実行: `uv run pytest -q`
