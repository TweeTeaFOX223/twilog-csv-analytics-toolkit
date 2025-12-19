# twilog-csv-analytics-toolkit
X(旧Twitter)のツイートを保存する「Twilog」から出力した自分のポストデータを分析するGUIアプリ。htmx・FastAPI。自分のTwitter歴の振り返りに有用。

## uvでの実行方法
- 依存インストール: `uv sync`
- 開発サーバ: `uv run uvicorn app.main:app --reload`
- テスト実行: `uv run pytest -q`

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
