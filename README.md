---
title: 一目均衡表 節目判定（細田哲生理論）
emoji: 📈
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# ichimoku_hosoda ― 細田哲生理論ベースの一目均衡表 節目判定ツール

株式コードを入れると、日足・週足・月足で

- 直観的判断（各線の位置関係、どの線が押し目／戻りとして機能しているか、横ばい＝相場水準化）
- 直近の波形（N / P / Y / S）と三波動構成候補（AB=BD, AB=CD, AC=CD の時間、E/N/V/NT/P 計算値、達成有無）
- 価格の節目（計算値の集中帯、相場水準候補とその起点、水準×2−極値のもみ合い限界）
- 時間の節目（基本数値・対等数値・三波動・相場水準起点からの変化日候補。根拠の重なりでスコア化）
- 基本数値足（3日足・9日足・26日足 / 9週足・26週足 / 9ヶ月足・26ヶ月足）の陰連陽連と、今後の陰陽の分岐値

を Markdown レポート・JSON・PNG チャートで出力します。理論の根拠は `ichimoku_hosoda_theory.md` を参照。

## インストール

```bash
pip install pandas numpy matplotlib yfinance matplotlib-fontja fastapi uvicorn python-multipart plotly
```

## Web アプリ（推奨）

```bash
python -m ichimoku_hosoda.web          # → http://127.0.0.1:8000
python -m ichimoku_hosoda.web --port 8080
```
ブラウザでコード（9201 / N225 / USDJPY）を入力して「分析」。日足・週足・月足のタブ、対話チャート（極・相場水準・変化日を重ね描き）、
総合所見・変化日候補・計算値・相場水準・三波動候補・基本数値足の一覧、テキストレポート全文を表示します。
「CSV を読む」で手元の OHLC CSV も分析できます。動作確認は `DEMO` と入力。
取得結果は 15 分キャッシュします。plotly.js はパッケージ同梱のものを配信するためオフラインでも表示できます。


## Web に公開する（無料ホスティング）

### 方法 A: Hugging Face Spaces（推奨・無料、スリープあり）
1. https://huggingface.co/new-space で Space を作成（SDK: **Docker**、Public/Private どちらでも可）
2. 作成した Space の「Files」→「Add file」→「Upload files」で、このリポジトリの全ファイル（`ichimoku_hosoda/`、`Dockerfile`、`requirements.txt`、`README.md`、`ichimoku_hosoda_theory.md`）をアップロード
   - または Space の git リモートに push: `git remote add hf https://huggingface.co/spaces/<user>/<space> && git push hf main`
3. 数分でビルドされ `https://<user>-<space>.hf.space` で公開されます（README 冒頭の front matter が Space 設定です）

### 方法 B: Render（無料プラン、15 分無アクセスでスリープ）
1. このリポジトリを GitHub に置く
2. https://dashboard.render.com → New → **Blueprint** → リポジトリを選択（`render.yaml` を自動検出）
3. デプロイ後に `https://ichimoku-hosoda.onrender.com` のような URL が発行されます

### 方法 C: 自前サーバー / VPS
```bash
docker build -t ichimoku-hosoda . && docker run -p 7860:7860 ichimoku-hosoda
```

※ データ取得は yfinance（Yahoo Finance）を第一に、失敗時は stooq を使います。クラウド上では Yahoo 側に制限されることがあるため、その場合 stooq 経由になります（日足のみ）。
※ 公開 URL は誰でも使えます。細田先生の理論・原著の内容を含むため、公開範囲（Private Space、Basic 認証など）はご判断ください。

## 使い方（コマンドライン）

```bash
# 東証銘柄（4桁コードは自動で .T を補完）
python -m ichimoku_hosoda 7203 --out out

# 日経平均 / ドル円
python -m ichimoku_hosoda N225 --out out
python -m ichimoku_hosoda USDJPY --out out

# 手元の CSV（Date,Open,High,Low,Close[,Volume]。日本語列名 日付/始値/高値/安値/終値 も可）
python -m ichimoku_hosoda 7203 --csv data/7203.csv --out out

# 擬似データで動作確認
python -m ichimoku_hosoda DEMO --demo --out out_demo
```

オプション: `--period 6y`（取得期間）、`--horizon 60`（日足で何営業日先まで変化日を出すか）、`--no-chart`

出力: `out/<code>_report.md`、`out/<code>_summary.json`、`out/<code>_daily.png` / `_weekly.png` / `_monthly.png`

## Python から

```python
from ichimoku_hosoda import data, analyze
df = data.fetch("7203", period="6y")          # or data.load_csv("7203.csv")
res = analyze(df, "日足", "日", horizon=60, nbar_specs=((3, "3日足"),))
print(res.to_markdown())
res.change_clusters[:5]   # 変化日候補
res.levels[:5]            # 相場水準候補
res.waves[:5]             # 三波動候補
```

## 設計上の約束（細田理論に従う）

- 出力は「売買シグナル」ではなく「暫定判断＋検証条件」。変化日は当てる日ではなく、その日の反応の有無を確認するポイント
- 三役好転・悪化は「26 のスパンでのもみ合い放れ／下げ（上げ）の影響力脱却」の示唆として扱い、出口を三役悪化にしない
- 時間の数え方は起点日を 1 日目とする営業日数（細田の数え方）。日付の将来分は営業日近似で日本の祝日は未考慮
- 基本数値には幅を持たせる（〜50: ±1、〜150: ±2、それ以上: ±3）
- 9週足・3日足の始値は「当日を含めて n 本目」を既定（`nbars.nbar(..., include_current=False)` で「n 本前」に切替可）

## 既知の限界・今後

- 極（高値・安値）の抽出はジグザグ（既定 3/6/12/25%）に依存。銘柄のボラティリティで閾値調整が必要
- 「償却」「中間波動 P/Y/S の活用」「受動面・能動面」は細田氏が著書で詳述を避けた領域のため未実装
- 月足の先行スパン(52) は 78 ヶ月以上のデータが必要
- 変化日の「反応の有無」を日次で記録し、優先順位を更新する仕組み（細田：相場は毎日追う）は次段階
