# twilog-csv-analytics-toolkit
![Twilog Analytics App](https://raw.githubusercontent.com/TweeTeaFOX223/twilog-csv-analytics-toolkit/refs/heads/main/docs/app1.png)

  
## 概要
X(旧Twitter)のツイートを保存する「Twilog」から出力した自分のポストデータを分析するGUIアプリ。htmx・FastAPI。自分のTwitter歴の振り返りに有用。思い付いた限りの全ての分析手法を入れた。    
  
[「FastAPI + htmxが最強説」- AIエンジニアがモック作るならReactは不要、Streamlitも捨てよう](https://zenn.dev/livetoon/articles/04dccf642d324c)の記事を見て、htmx・FastAPIを実際に試すために作成した。AIエージェントとしてGPT-5.3-Codexを使用している。    

ほぼ同じUIを持つアプリをStreamlitで作成したことがあるが、それと比べると「レンダリング順序と状態管理」が非常に簡単だった。→「[bookmark-analytics-toolkit > streamlitのUI作成(地獄)](https://github.com/TweeTeaFOX223/bookmark-analytics-toolkit?tab=readme-ov-file#streamlit%E3%81%AEui%E4%BD%9C%E6%88%90%E5%9C%B0%E7%8D%84)」を参照。  
  
## uvでの実行方法
- 依存インストール: `uv sync`
- 開発サーバ: `uv run uvicorn app.main:app --reload`
- テスト実行: `uv run pytest -q`

## PyInstallerでexe化（Windows）
- ビルド依存を含めて同期: `uv sync --extra dev`
- exeビルド: `uv run pyinstaller twilog_analytics.spec --noconfirm`
- 実行ファイル: `dist\\twilog-analytics\\twilog-analytics.exe`
- 起動後に `http://127.0.0.1:8000` をブラウザで開く
- ライセンス一覧はビルド時に自動生成され、exe同梱されます

配布/実行形態について:
- 現在の設定は `one-dir` ビルドです。`dist\\twilog-analytics\\` フォルダ一式が必要です。
- `twilog-analytics.exe` 単体だけを別の場所へコピーすると動作しません（同梱DLL・データ参照のため）。
- 配布時は `dist\\twilog-analytics\\` フォルダごと配布してください。
- 単体exe（`one-file`）にしたい場合は `.spec` を `COLLECT` なし構成に変更する必要があります。

`dist\\twilog-analytics\\` の中身:
- `twilog-analytics.exe`: 起動用の実行ファイル本体。
- `_internal\\`: 実行に必要な同梱ファイル群（このフォルダごと必要）。
- `_internal\\app\\`: `templates` / `static` などアプリのHTML・CSS資産。
- `_internal\\app\\static\\THIRD_PARTY_LICENSES.txt`: 同梱ライブラリのライセンス一覧（テキスト）。
- `_internal\\app\\static\\THIRD_PARTY_LICENSES.json`: 同梱ライブラリのライセンス一覧（JSON）。
- `_internal\\sudachipy`, `_internal\\sudachidict_full`: 形態素解析（SudachiPy）用の本体と辞書データ。
- `_internal\\*.dll`, `_internal\\*.pyd`, `_internal\\python313.dll`: Pythonランタイムとネイティブ拡張（NumPy/Polars/Matplotlib等）の実行モジュール。
- `_internal\\base_library.zip`: 標準ライブラリの同梱アーカイブ。

ライセンス表示:
- アプリ起動後にヘッダーの `ライセンス` を開くと、同梱済みライセンス一覧を表示できます。
- 画面から `TXT` / `JSON` をそのままダウンロードできます。

ライセンス一覧の手動生成（開発時）:
- `uv run python scripts/generate_licenses.py`
- 生成先: `app/static/THIRD_PARTY_LICENSES.txt` と `app/static/THIRD_PARTY_LICENSES.json`

環境変数で待ち受け先を変更可能:
- `TWILOG_HOST`（既定: `127.0.0.1`）
- `TWILOG_PORT`（既定: `8000`）
- `TWILOG_OPEN_BROWSER`（既定: `1`。`0` で自動ブラウザ起動を無効化）
  
## サンプルデータ(CSV)
`sample_csv\TweeTeaFOX223-260301_only.csv`に私のXのポストのCSVがあります。リポスト(他の人の投稿)を除外して私のポストだけにしたやつです。これを読み込ませればすぐに試せます。  
  
## プロジェクト構成と動作概要
  
### 技術スタック
- 言語：Python
- パッケージ管理：uv
- AIエージェント：GPT-5.3-Codex
- アプリ: FastAPI + Jinja2 + HTMX（SSR + 部分更新）
- データフレーム: Polars
- 形態素解析: SudachiPy + sudachidict-full
- 可視化: Plotly / Matplotlib / WordCloud
  
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
  

## 機能一覧

### 基本
- CSVアップロード: Twilog形式CSVを読み込み、解析セッションを作成。
- 設定保存: 対象年、Sudachiモード、品詞フィルタ、ストップワード、キーワード辞書を保存。
- 部分更新UI: 分析パネルをHTMXで切り替え（フルリロード不要）。

### 投稿量・時系列
- 平均: 年/月/曜日ごとの平均投稿数を表示。
- 時間帯/曜日: 時間別・曜日別・日別の投稿分布を可視化。
- 週別/カレンダー: 週次推移と曜日×週のヒートマップを表示。
- 月内日偏り: 月内の日付ごとの偏りを確認。
- 深夜比率: 0-5時の投稿割合を集計。
- 投稿間隔: 投稿間隔の分布とセッション分割の傾向を表示。
- 間隔要約: 投稿間隔の要約統計（最小・中央値・平均など）。

### テキスト・語彙分析
- 単語ランキング: 出現頻度の高い語をランキング表示。
- ワードクラウド: 頻出語を画像で可視化。
- ワードクラウドN+V: 名詞+動詞を中心にワードクラウド生成。
- TF-IDF: 文書単位で特徴語を抽出。
- 単語推移: 指定語の時系列推移を表示。
- 共起語: 指定語と同時出現しやすい語を集計。
- 共起ネット: 語の共起関係をネットワーク表示。
- Sudachi比較: 分割モード差による語の違いを比較。
- 月代表語: 月ごとの代表語を抽出。
- クラスタ: 投稿を特徴語ベースでクラスタリング。

### ハッシュタグ分析
- ハッシュタグ: 使用頻度ランキングを表示。
- ハッシュタグ共起: 一緒に使われるタグ関係を可視化。
- ハッシュタグ年次: 年ごとの出現推移を表示。
- タグ数分布: 1投稿あたりのタグ数分布を集計。

### URL・ドメイン分析
- URL分布: URL有無やURL数の分布を可視化。
- ドメイン: 参照先ドメインの出現頻度を表示。
- TLD分布: トップレベルドメインの分布を表示。
- パス深さ: URLパスの深さ分布を集計。
- ドメイン年次: ドメイン利用の年次推移を表示。
- ドメイン月次: 指定ドメインの月次推移を表示。
- 自己参照URL: Twitter/X参照URLの比率を可視化。
- 自分ツイURL: 自分の投稿URL参照の割合を集計。

### メンション・反応
- リプライ比率: リプライ投稿と通常投稿の比率を表示。
- メンション: メンション先ランキングを表示。
- メンション曜日: 曜日ごとのメンション総数を表示。
- メンション曜日詳細: 上位メンション先別の曜日分布を表示。
- メンション数分布: 1投稿あたりメンション数の分布を集計。

### その他の投稿特性
- 文字数分布: 投稿文字数の分布を表示。
- 長文ランキング: 文字数が多い投稿を抽出。
- 改行数: 改行数の分布を集計。
- 絵文字数: 絵文字使用数の分布を表示。
- カテゴリ辞書: ユーザー定義キーワード辞書でカテゴリ別集計。
- 代表語推移: 代表語の時系列推移を表示。

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
