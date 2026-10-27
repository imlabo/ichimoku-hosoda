"""ローカル Web アプリ:  python -m ichimoku_hosoda.web  → http://127.0.0.1:8000

株式コードを入力 → yfinance で取得 → 日足・週足・月足を分析 → 対話チャートとレポートを表示。
CSV アップロードにも対応。
"""
from __future__ import annotations

import io
import re
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from . import data, margin, names
from .engine import analyze
from .serialize import _level, clusters_to_json, result_to_json, waves_to_json
from .engine import manual_wave
from . import waves as wavemod

_results: dict[tuple, dict] = {}   # (code, period) -> {frame_key: AnalysisResult}

app = FastAPI(title="一目均衡表 節目判定")
_cache: dict[tuple, tuple[float, dict]] = {}
CACHE_SEC = 900


INV_NOTE = "値 = 基準×2 − 実際の価格。反転上の『上げ』は実際の下げ、『押し目』は実際の戻り。y 軸目盛とチャート上のラベルは実際の価格"


_LAST: dict = {}


def _with_inverse(src: pd.DataFrame, label: str, unit: str, horizon: int, nbar_specs, zz=None, extra_lag=None, bars=300, key=None):
    try:
        return _with_inverse_inner(src, label, unit, horizon, nbar_specs, zz, extra_lag, bars, key)
    except Exception as e:  # noqa: BLE001  1 つの足種の失敗で全体を止めない
        import traceback
        traceback.print_exc()
        _ERRORS.append(f"{label}: 分析エラー（{type(e).__name__}: {e}）")
        return None, None


_ERRORS: list = []


def _with_inverse_inner(src, label, unit, horizon, nbar_specs, zz, extra_lag, bars, key):
    kw = dict(nbar_specs=nbar_specs, extra_lag=extra_lag)
    if zz:
        kw["zz_thresholds"] = zz
    r1 = analyze(src, label, unit, horizon, **kw)
    normal = result_to_json(r1, bars=bars)
    inv_df, ref = data.invert(src)
    r2 = analyze(inv_df, "反転" + label, unit, horizon, **kw)
    inv = result_to_json(r2, bars=bars)
    inv["inverted"] = {"ref": ref, "note": INV_NOTE}
    if key:
        _LAST[key] = r1
        _LAST[key + "_inv"] = r2
    return normal, inv


def analyze_all(df: pd.DataFrame, name: str, horizon: int = 60, intraday: dict | None = None,
                full_df: pd.DataFrame | None = None) -> dict:
    """df: 日足タブに使う範囲、full_df: 週足・月足の作成に使う長期日足（省略時は df）。"""
    full = full_df if full_df is not None and len(full_df) > len(df) else df
    out = {"name": name, "from": str(df.index[0].date()), "to": str(df.index[-1].date()), "bars": len(df), "frames": {},
           "notes": []}
    for key, (idf, label, zz) in (intraday or {}).items():
        if idf is None or len(idf) < 80:
            out["notes"].append(f"{label}: データ不足または取得不可")
            continue
        out["frames"][key], out["frames"][key + "_inv"] = _with_inverse(
            idf, label, "本", 60, ((3, "3本足"), (9, "9本足"), (26, "26本足")), zz, None, 400, key=key)
    out["frames"]["daily"], out["frames"]["daily_inv"] = _with_inverse(
        df, "日足", "日", horizon, ((3, "3日足"), (9, "9日足"), (26, "26日足")), None, 3, min(320, len(df)), key="daily")
    if len(full) >= 60:
        wdf = data.resample(full, "W-FRI")
        if len(wdf) >= 60:
            out["frames"]["weekly"], out["frames"]["weekly_inv"] = _with_inverse(
                wdf, "週足", "週", 26, ((9, "9週足"), (26, "26週足")), (0.05, 0.10, 0.20, 0.35), 9, 260, key="weekly")
    if len(full) >= 300:
        mdf = data.resample(full, "ME")
        if len(mdf) >= 40:
            out["frames"]["monthly"], out["frames"]["monthly_inv"] = _with_inverse(
                mdf, "月足", "ヶ月", 18, ((9, "9ヶ月足"), (26, "26ヶ月足")), (0.08, 0.15, 0.30, 0.50), None, 200, key="monthly")
    return out


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


# データは screener-data ブランチ。main へのコミットは Render の再ビルドを起こすため分けている。
WATCHLIST_RAW = ("https://raw.githubusercontent.com/imlabo/ichimoku-hosoda/"
                 "screener-data/screener/watchlist.json")
_wl_cache: tuple[float, list] | None = None


def _load_watchlist() -> list:
    """監視リストを取得。GitHub の raw を15分キャッシュし、失敗時は同梱ファイルにフォールバック。

    Docker イメージには screener/ を同梱していないため、raw 取得が主経路。
    こうしておくと日々の更新のたびに再デプロイしなくてもページが最新になる。
    """
    global _wl_cache
    import json as _json

    now = time.time()
    if _wl_cache and now - _wl_cache[0] < 900:
        return _wl_cache[1]

    wl: list = []
    try:
        import requests
        r = requests.get(WATCHLIST_RAW, timeout=10)
        r.raise_for_status()
        wl = r.json()
    except Exception:
        try:
            path = Path(__file__).resolve().parent.parent / "screener" / "watchlist.json"
            wl = _json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            wl = []

    if wl:
        _wl_cache = (now, wl)
    return wl


def mini_chart(ch: dict, w: int = 300, h: int = 128) -> str:
    """ローソク足のミニチャートを SVG で描く。転換線・基準線と、ストップ高▲／ストップ安▼を重ねる。"""
    if not ch or not ch.get("c"):
        return ""
    o, hi, lo, c = ch["o"], ch["h"], ch["l"], ch["c"]
    n = len(c)
    vals = [v for v in (hi + lo + [x for x in (ch.get("tk") or []) if x is not None]
                        + [x for x in (ch.get("kj") or []) if x is not None]) if v is not None]
    if not vals:
        return ""
    vmax, vmin = max(vals), min(vals)
    pad = (vmax - vmin) * 0.06 or 1.0
    vmax, vmin = vmax + pad, vmin - pad
    PL, PR, PT, PB = 2, 2, 8, 2
    iw, ih = w - PL - PR, h - PT - PB
    step = iw / max(n, 1)
    bw = max(1.3, step * 0.62)

    def X(i):
        return round(PL + step * (i + 0.5), 2)

    def Y(v):
        return round(PT + ih * (vmax - v) / (vmax - vmin), 2)

    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" preserveAspectRatio="none" '
             f'xmlns="http://www.w3.org/2000/svg" class="mc" role="img">']

    def line(key, cls):
        pts = [f"{X(i)},{Y(v)}" for i, v in enumerate(ch.get(key) or []) if v is not None]
        if len(pts) > 1:
            parts.append(f'<polyline class="{cls}" points="{" ".join(pts)}"/>')

    line("kj", "kj")
    line("tk", "tk")
    for i in range(n):
        if None in (o[i], hi[i], lo[i], c[i]):
            continue
        up = c[i] >= o[i]
        cls = "up" if up else "dn"
        x = X(i)
        parts.append(f'<line class="wk {cls}" x1="{x}" y1="{Y(hi[i])}" x2="{x}" y2="{Y(lo[i])}"/>')
        y1, y2 = Y(max(o[i], c[i])), Y(min(o[i], c[i]))
        parts.append(f'<rect class="bd {cls}" x="{round(x - bw / 2, 2)}" y="{y1}" '
                     f'width="{round(bw, 2)}" height="{max(round(y2 - y1, 2), 1.0)}"/>')
    for i in ch.get("lu") or []:
        if i < n and hi[i] is not None:
            parts.append(f'<polygon class="mu" points="{X(i)},{Y(hi[i]) - 6.5} '
                         f'{X(i) - 3.2},{Y(hi[i]) - 1.5} {X(i) + 3.2},{Y(hi[i]) - 1.5}"/>')
    for i in ch.get("ld") or []:
        if i < n and lo[i] is not None:
            parts.append(f'<polygon class="md" points="{X(i)},{Y(lo[i]) + 6.5} '
                         f'{X(i) - 3.2},{Y(lo[i]) + 1.5} {X(i) + 3.2},{Y(lo[i]) + 1.5}"/>')
    parts.append("</svg>")
    return "".join(parts)


def chart_grid(rows: list, limit: int = 24) -> str:
    """候補のチャートを並べる。クリックで節目判定ページへ。"""
    import html as _h
    cards = []
    for x in rows[:limit]:
        svg = mini_chart(x.get("chart"))
        if not svg:
            continue
        name = _h.escape(str(x.get("name", "")))
        code = _h.escape(str(x.get("code", "")))
        sub = []
        if x.get("kind"):
            sub.append("継続" if x["kind"] == "continuation" else "前兆")
        if x.get("spike_date"):
            sub.append(str(x["spike_date"]))
        sub.append(f"スコア{x.get('score', '')}")
        cards.append(
            f'<a class="card" href="/?code={code}" title="{name} の節目判定を開く">'
            f'<div class="ct"><b>{code}</b> {name}</div>'
            f'{svg}<div class="cs">{" / ".join(_h.escape(s) for s in sub)}</div></a>')
    if not cards:
        return ""
    return f'<div class="grid">{"".join(cards)}</div>'


@app.get("/screener", response_class=HTMLResponse)
def screener():
    """大出来高 × 一目 監視リスト。screener/watchlist.json（毎営業日 17:30 JST 更新）を表示。"""
    import html as _html
    import json as _json

    wl = _load_watchlist()

    def norm(x):
        x.setdefault("direction", "down")
        if "spike_chg_pct" not in x and "spike_drop_pct" in x:
            x["spike_chg_pct"] = x.pop("spike_drop_pct")
        if "confirmed" not in x and "confirmed_white_candle" in x:
            x["confirmed"] = x.pop("confirmed_white_candle")
        if "level_held" not in x and "spike_low_held" in x:
            x["level_held"] = x.pop("spike_low_held")
        return x

    wl = [norm(x) for x in wl]
    asof = max((x.get("last_date", "") for x in wl), default="")
    mark = lambda b: '<span class="o">\u25cb</span>' if b else '<span class="x">\u2014</span>'
    CLOUD = {"above": "雲上", "in": "雲中", "below": "雲下"}

    def table(rows, direction):
        if not rows:
            return '<p class="none">該当なし</p>'
        d = direction == "down"
        cells = "".join(
            f"<tr><td class=\"code\"><a href=\"/?code={_html.escape(str(x['code']))}\">"
            f"{_html.escape(str(x['code']))}</a></td>"
            f"<td>{_html.escape(str(x['name']))}</td>"
            f"<td>{_html.escape(str(x['spike_date']))}</td>"
            f"<td class=\"num {'dn' if x['spike_chg_pct'] < 0 else 'upc'}\">{x['spike_chg_pct']:+.2f}%</td>"
            f"<td class=\"num\">\u00d7{x['spike_vol_mult']}</td>"
            f"<td class=\"num\">{x['days_since_spike']}</td>"
            f"<td>{mark(x['confirmed'])}</td><td>{mark(x['level_held'])}</td>"
            f"<td>{mark(x['above_tenkan'])}</td><td>{mark(x['above_kijun'])}</td>"
            f"<td>{CLOUD.get(x['cloud_position'], x['cloud_position'])}</td>"
            f"<td>{mark(x['chikou_bullish'])}</td>"
            f"<td class=\"num\">{x['basic_number_hit'] or '\u2014'}</td>"
            f"<td class=\"num score\">{x['score']}</td></tr>"
            for x in rows)
        return (
            "<div class=\"wrap\"><table><thead><tr>"
            "<th>コード</th><th>銘柄</th>"
            f"<th>{'急落日' if d else '急騰日'}</th><th>{'下落率' if d else '上昇率'}</th>"
            "<th>出来高倍率</th><th>経過日</th>"
            f"<th>{'陽線確認' if d else '終値維持'}</th><th>安値維持</th>"
            "<th>転換線</th><th>基準線</th><th>雲</th><th>遅行</th><th>日柄</th><th>スコア</th>"
            f"</tr></thead><tbody>{cells}</tbody></table></div>")

    def sq_table(rows):
        if not rows:
            return '<p class="none">該当なし</p>'
        KIND = {"continuation": "継続", "whipsaw": "往復", "precursor": "前兆"}
        def pct(v, suf="%"):
            return f"{v}{suf}" if v is not None else "\u2014"
        def mk(v):
            return (str(v).replace("プライム（内国株式）", "プライム")
                    .replace("スタンダード（内国株式）", "スタンダード")
                    .replace("グロース（内国株式）", "グロース"))
        cells = "".join(
            f"<tr><td class=\"code\"><a href=\"/?code={_html.escape(str(x['code']))}\">"
            f"{_html.escape(str(x['code']))}</a></td>"
            f"<td>{_html.escape(str(x['name']))}</td>"
            f"<td class=\"sm\">{_html.escape(mk(x.get('market', ''))[:8])}</td>"
            f"<td class=\"sm\">{KIND.get(x.get('kind'), '')}</td>"
            f"<td>{_html.escape(str(x['spike_date']))}</td>"
            f"<td class=\"num\">{pct(x.get('limit_pct'))}</td>"
            f"<td class=\"num\">\u00d7{x.get('spike_vol_mult') if x.get('spike_vol_mult') is not None else '\u2014'}</td>"
            f"<td class=\"num\">{pct(x.get('atr14_pct'))}</td>"
            f"<td class=\"num {'upc' if (x.get('ret_20d') or 0) > 0 else 'dn'}\">{pct(x.get('ret_20d'))}</td>"
            f"<td class=\"num {'upc' if (x.get('pullback_pct') or 0) > 0 else 'dn'}\">{pct(x.get('pullback_pct'))}</td>"
            f"<td class=\"num\">{x.get('lu_count_20d') if x.get('lu_count_20d') is not None else '\u2014'}</td>"
            f"<td class=\"num\">{pct(x.get('turnover20_oku'), '億')}</td>"
            f"<td class=\"num\">{x.get('prior_limitups_20d', 0)}</td>"
            f"<td>{mark(x.get('sanyaku'))}</td>"
            f"<td class=\"num\">{x.get('margin_ratio') if x.get('margin_ratio') is not None else '\u2014'}</td>"
            f"<td class=\"num\">{pct(x.get('margin_sell_chg4w'))}</td>"
            f"<td class=\"num {'dn' if (x.get('margin_sell_pl') or 0) < 0 else ''}\">{pct(x.get('margin_sell_pl'))}</td>"
            f"<td class=\"sm\">{_html.escape(str(x.get('sector', ''))[:6])}"
            f"{'<span class=\"o\">\u2605</span>' if x.get('theme') else ''}</td>"
            f"<td class=\"num score\">{x['score']}</td></tr>"
            for x in rows)
        return (
            "<div class=\"wrap\"><table><thead><tr>"
            "<th>コード</th><th>銘柄</th><th>市場</th><th>種別</th><th>起点日</th>"
            "<th>制限値幅</th><th>当日出来高</th><th>ATR</th><th>20日騰落</th>"
            "<th title=\"ストップ高が始まる前の終値に対する現在値。プラスなら押しが浅い\">押し</th>"
            "<th title=\"直近20日のストップ高回数\">反復</th><th>代金</th>"
            "<th>履歴</th><th>三役</th><th>貸借倍率</th><th>売残4週</th><th>売り方損益</th>"
            "<th>業種</th><th>スコア</th>"
            f"</tr></thead><tbody>{cells}</tbody></table></div>")

    down = [x for x in wl if x["direction"] == "down"]
    up = [x for x in wl if x["direction"] == "up"]
    sq = [x for x in wl if x["direction"] == "squeeze"]

    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>大出来高 × 一目 監視リスト</title>
<style>
:root {{ --bg:#fbfaf8; --fg:#1b1a18; --muted:#6b6862; --line:#e3e0da;
        --card:#fff; --up:#b3261e; --dn:#1d5c9e; --accent:#8a6a3b; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg:#17161a; --fg:#eceae6; --muted:#9a968e; --line:#302e33;
  --card:#201f24; --up:#e8705f; --dn:#79a9de; --accent:#c6a978; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg);
  font-family:-apple-system,BlinkMacSystemFont,"Hiragino Sans","Noto Sans JP",sans-serif;
  line-height:1.6; }}
.page {{ max-width:1100px; margin:0 auto; padding:32px 16px 64px; }}
h1 {{ font-size:1.5rem; margin:0 0 4px; letter-spacing:.02em; }}
.asof {{ color:var(--muted); font-size:.85rem; margin:0 0 28px; }}
h2 {{ font-size:1.05rem; margin:36px 0 12px; padding-bottom:6px;
  border-bottom:2px solid var(--accent); display:inline-block; }}
h2 .n {{ color:var(--muted); font-weight:400; font-size:.85rem; margin-left:8px; }}
.wrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:8px; background:var(--card); }}
table {{ border-collapse:collapse; width:100%; font-size:.85rem; }}
th, td {{ padding:8px 10px; text-align:left; white-space:nowrap;
  border-bottom:1px solid var(--line); }}
th {{ background:color-mix(in srgb, var(--card) 88%, var(--fg));
  font-weight:600; font-size:.78rem; color:var(--muted); }}
tbody tr:last-child td {{ border-bottom:none; }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.sm {{ font-size:.78rem; color:var(--muted); }}
.code {{ font-variant-numeric:tabular-nums; color:var(--accent); font-weight:600; }}
.score {{ font-weight:700; }}
.upc {{ color:var(--up); }} .dn {{ color:var(--dn); }}
.o {{ color:var(--accent); font-weight:700; }} .x {{ color:var(--muted); }}
.none {{ color:var(--muted); font-size:.9rem; padding:12px 0; }}
.grid {{ display:grid; gap:12px; margin-top:14px;
  grid-template-columns:repeat(auto-fill, minmax(230px, 1fr)); }}
.card {{ display:block; border:1px solid var(--line); border-radius:8px; background:var(--card);
  padding:8px 8px 6px; text-decoration:none; color:inherit; }}
.card:hover {{ border-color:var(--accent); }}
.ct {{ font-size:.78rem; margin-bottom:2px; white-space:nowrap; overflow:hidden;
  text-overflow:ellipsis; }}
.ct b {{ color:var(--accent); font-variant-numeric:tabular-nums; }}
.cs {{ font-size:.7rem; color:var(--muted); margin-top:2px; }}
.mc {{ display:block; height:112px; }}
.mc .wk {{ stroke-width:1; }}
.mc .bd {{ stroke-width:.6; }}
.mc .up {{ stroke:var(--up); fill:var(--up); }}
.mc .dn {{ stroke:var(--dn); fill:var(--dn); }}
.mc .tk {{ fill:none; stroke:var(--accent); stroke-width:1.2; }}
.mc .kj {{ fill:none; stroke:var(--muted); stroke-width:1.2; stroke-dasharray:3 2; }}
.mc .mu {{ fill:var(--up); }}
.mc .md {{ fill:var(--dn); }}
.code a {{ color:inherit; text-decoration:none; border-bottom:1px dotted currentColor; }}
.note {{ margin-top:36px; padding-top:16px; border-top:1px solid var(--line);
  color:var(--muted); font-size:.8rem; }}
.note li {{ margin:4px 0; }}
a {{ color:var(--accent); }}
</style></head><body><div class="page">
<h1>大出来高 × 一目 監視リスト</h1>
<p class="asof">{_html.escape(asof) or "データなし"} 引け後 / 東証プライム・スタンダード・グロース / 毎営業日 17:30 JST 更新
 &middot; <a href="/">節目判定にもどる</a></p>

<h2>ストップ高・踏み上げ候補<span class="n">{len(sq)} 件</span></h2>
{sq_table(sq)}
{chart_grid(sq)}

<h2>下落（底打ち候補）<span class="n">{len(down)} 件</span></h2>
{table(down, "down")}
{chart_grid(down, 12)}

<h2>上昇（ブレイク候補）<span class="n">{len(up)} 件</span></h2>
{table(up, "up")}
{chart_grid(up, 12)}

<div class="note"><ul>
<li><b>往復（ストップ高→ストップ安→再ストップ高）</b>の配点: 反復3回以上 +2 / 押しが浅い（ストップ高前の水準を維持）+2 / 売買代金20億円以下 +1 / 同50億円超 &minus;2 / プライム &minus;1。過去2年・往復378件の実測で、ストップ安から10日以内の再ストップ高は<b>34.4%</b>。ただし<b>ストップ安を挟まない場合も31.1%</b>なので、ストップ安自体の寄与は小さく、効くのは反復回数（3回以上37.0% / 2回23.5% / 1回以下21.1%）と押しの浅さ（維持37.3% / 下回り28.5%）でした。ストップ安までの日数とATRは効きませんでした。再ストップ高までは1〜7営業日が大半（3〜4日が最多）。このパターンでは一目の各項目を別途検証していないため、表示のみで配点していません</li>
<li><b>継続・前兆</b>の配点: 直近20日にストップ高の履歴 +3 / ATR14が7%以上 +2 / ストップ高当日の出来高が20日平均の2倍未満 +2 / 三役好転 +2 / 20日騰落+15%以上 +1 / 売買代金20億円以下 +1 / 同50億円超 &minus;2 / プライム &minus;1 / 雲上+1・雲下&minus;1 / 基本数値一致 +1。需給は 売残が4週で+20%以上 +2 / 同&minus;20%以下 &minus;2 / 貸借倍率1〜3 +1。同業種で2件以上ヒットすると +1（業種名に★）</li>
<li>この配点は過去1年・ストップ高763件の<b>翌日継続率</b>から決めています。全体16.5%に対し、履歴あり24.0% / ATR7%以上21.8% / 当日出来高2倍未満23.9% / 履歴&times;三役好転&times;ATR 30.6%。<b>最良の組み合わせでも3割程度</b>なので、スコアは確度ではなく条件の重なり具合として見てください</li>
<li>需給の配点も過去のストップ高177件の信用残の実測から決めています。基準15.4%に対し、<b>売残が4週で+20%以上は20.7%、&minus;20%以下は8.9%</b>、貸借倍率1〜3は19.8%。一方<b>「売残&gt;買残」は6.6%、「売り方が含み損」は14.4%で効きませんでした</b>（含み損は既に上昇した銘柄なら当然そうなるため、20日騰落とほぼ同じことを測っていた）。この2つは配点せず表示のみです</li>
<li>チャートは直近60営業日。実線が転換線、破線が基準線、▲がストップ高、▼がストップ安。コードまたはチャートをクリックすると節目判定が開きます</li>
<li>「継続」は直近3営業日にストップ高があった銘柄、「前兆」はストップ高履歴とATRの条件が揃っているがまだストップ高していない銘柄</li>
<li>抽出条件（下落・上昇）: 出来高が20日平均の3倍以上 &times; 下落側は7%超の下落または大陰線かつ20日高値から15%以上下落／上昇側は7%超の上昇または大陽線かつ20日高値を更新</li>
<li>スコア: 確認+2 / 安値維持+2（割れ-3） / 転換線上+2 / 基準線上+1 / 遅行好転+1 / 基本数値一致+1 / 雲上+1・雲下-1</li>
<li>確認: 下落側は急落後に陽線が出たか、上昇側は急騰日の終値以上を維持しているか</li>
<li>急落／急騰日から約26営業日、ストップ高候補は7日で自動除外</li>
<li>本ページは個人的な研究用の集計であり、投資助言ではありません。売買の判断はご自身の責任でお願いします。</li>
</ul></div>
</div></body></html>"""


@app.get("/guide", response_class=HTMLResponse)
def guide():
    return (Path(__file__).parent / "static" / "guide.html").read_text(encoding="utf-8")


@app.get("/static/i18n.js")
def i18n_js():
    return FileResponse(Path(__file__).parent / "static" / "i18n.js", media_type="application/javascript")


@app.get("/static/plotly.min.js")
def plotly_js():
    """plotly パッケージ同梱の JS を配信（オフラインでも動く）。"""
    import plotly
    return FileResponse(Path(plotly.__file__).parent / "package_data" / "plotly.min.js", media_type="application/javascript")


@app.get("/api/search")
def api_search(q: str, limit: int = 15):
    return JSONResponse({"results": names.search(q, limit), "status": names.status()})


@app.on_event("startup")
def _warm():
    names.warmup()


@app.get("/api/analyze")
def api_analyze(code: str, period: str = "6y", horizon: int = 60, intraday: int = 1, market: str = "auto"):
    raw_code = code
    c0 = code.strip()
    m = re.fullmatch(r"(?i)vn[:：]\s*([A-Za-z0-9]{1,10})", c0)
    if m:
        code = m.group(1).upper() + ".VN"
    elif market == "vn" and re.fullmatch(r"[A-Za-z0-9]{1,10}", c0) and not re.fullmatch(r"[0-9][0-9A-Za-z]{3}", c0) and c0.upper() != "DEMO":
        code = c0.upper() + (".VN" if c0.upper() not in ("VNINDEX",) else "")
    code, disp = names.resolve(code)
    key = (code.strip().upper(), period, horizon, intraday)
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_SEC:
        return JSONResponse(hit[1])
    demo = code.strip().upper() == "DEMO"
    try:
        # 週足・月足のため日足は常に長期（6年）を取得し、日足タブは選択期間に切り出す
        full = data.synthetic() if demo else data.fetch(code, period="6y" if period in ("6mo", "1y", "3y", "6y") else period)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"データ取得に失敗しました: {e}") from e
    cut = {"6mo": pd.DateOffset(months=6), "1y": pd.DateOffset(years=1), "3y": pd.DateOffset(years=3)}.get(period)
    df = full[full.index >= full.index[-1] - cut] if cut else full
    intra = {}
    if intraday:
        for key2, interval, label, zz in (("m1", "1m", "1分足", (0.002, 0.004, 0.008, 0.016)),
                                           ("m5", "5m", "5分足", (0.004, 0.008, 0.016, 0.03)),
                                           ("h1", "1h", "1時間足", (0.01, 0.02, 0.04, 0.08))):
            try:
                idf = data.synthetic_intraday(interval) if demo else data.fetch_intraday(code, interval)
            except Exception:  # noqa: BLE001
                idf = None
            intra[key2] = (idf, label, zz)
    _LAST.clear()
    _ERRORS.clear()
    res = analyze_all(df, data.normalize_symbol(code), horizon, intra, full_df=full)
    try:   # 水平線（支持線・抵抗線）：週足・日足・1時間足から
        from . import srlines
        h1 = intra.get("h1", (None,))[0] if intra else None
        res["sr"] = srlines.compute(full, data.resample(full, "W-FRI") if len(full) >= 60 else None, h1)
    except Exception as e:  # noqa: BLE001
        res["sr"] = {"levels": [], "error": str(e)}
    res["frames"] = {k: v for k, v in res["frames"].items() if v is not None}
    res["notes"] = res.get("notes", []) + list(_ERRORS)
    res["display_name"] = disp or (names.name_for(code) if not demo else "デモデータ")
    res["input"] = raw_code
    if "daily" not in res["frames"]:
        raise HTTPException(500, "日足の分析に失敗しました: " + " / ".join(_ERRORS))
    _results[(key[0], period)] = dict(_LAST)
    res["market"] = "demo" if demo else data.market_of(data.normalize_symbol(code))
    res["api_code"] = code     # 以降の再計算などでサーバーに渡すコード
    res["margin"] = (margin.synthetic_margin(end=full.index[-1]) if demo else
                     margin.fetch_margin(code) if res["market"] == "jp" else {"skip": True})
    try:
        if res["margin"].get("rows"):
            res["margin"]["pl"] = margin.estimate_pl(res["margin"], full)
    except Exception as e:  # noqa: BLE001
        res["margin"]["pl_error"] = str(e)
    _cache[key] = (time.time(), res)
    return JSONResponse(res)


def _get_result(code: str, frame: str, period: str):
    """保存済みの分析結果。サーバー再起動などで消えていたら分析をやり直す。"""
    code2, _ = names.resolve(code)
    k = (code2.strip().upper(), period)
    if frame not in (_results.get(k) or {}):
        for ck in [ck for ck in _cache if ck[0] == k[0] and ck[1] == period]:
            _cache.pop(ck, None)
        api_analyze(code, period)
    r = (_results.get(k) or {}).get(frame)
    if r is None:
        raise HTTPException(404, "分析結果が見つかりません。もう一度「分析」を押してください")
    return r


@app.get("/api/wave_custom")
def api_wave_custom(code: str, frame: str, a: int = -1, b: int = -1, c: int = -1, dir: str = "", kinds: str = "",
                    dates: str = "", period: str = "6y"):
    """チャート上で A・B・C をドラッグして指定した三波動（a/b/c は足の位置）。計算値・D・変化日を返す。"""
    r = _get_result(code, frame, period)
    if dates:   # 保存した日付から足の位置を引く（期間を変えても同じ点を指せるように）
        from .serialize import _d, _FMT
        _FMT["fmt"] = "%m-%d %H:%M" if r.unit == "本" else "%Y-%m-%d"
        idx = [_d(t) for t in r.df.index]
        pos = []
        for ds in dates.split(","):
            hit = [k for k, v in enumerate(idx) if v == ds]
            if not hit:
                raise HTTPException(404, f"{ds} の足がこの期間にありません")
            pos.append(hit[-1])
        a, b, c = pos
    try:
        w, cl = manual_wave(r, a, b, c, dir or None, kinds or None)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    wj = waves_to_json([w], r, len(r.df))[0]
    wj["manual"] = True
    wj["note"] = w.note
    wj["cd"] = clusters_to_json(cl[:12], len(r.df))
    wj["kinds"] = "".join(p.kind for p in (w.A, w.B, w.C))
    wj["half"] = _level(w.half, r, len(r.df)) if w.half is not None else None
    return JSONResponse({"wave": wj})


@app.get("/api/waves_from")
def api_waves_from(code: str, frame: str, a_i: int, period: str = "6y"):
    """チャート上で選んだ極（pivot index a_i）を起点 A にした三波動候補を返す。"""
    r = _get_result(code, frame, period)
    pos = {p.i: k for k, p in enumerate(r.pivots)}
    if a_i not in pos:
        raise HTTPException(400, "その点は極として登録されていません")
    ws = wavemod.build_waves(r.pivots, r.df, max_recent=10, force_a=pos[a_i])
    ws = [w for w in ws if "不成立" not in w.note]
    return JSONResponse({"waves": waves_to_json(ws, r, len(r.df))})


@app.post("/api/analyze_csv")
async def api_analyze_csv(file: UploadFile = File(...), name: str = Form("CSV"), horizon: int = Form(60)):
    raw = await file.read()
    try:
        df = data.load_csv(io.BytesIO(raw))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"CSV を読めませんでした: {e}") from e
    _ERRORS.clear()
    res = analyze_all(df, name or file.filename, horizon)
    res["frames"] = {k: v for k, v in res["frames"].items() if v is not None}
    res["notes"] = res.get("notes", []) + list(_ERRORS)
    return JSONResponse(res)


def main():
    import argparse

    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    print(f"→ ブラウザで http://{a.host}:{a.port} を開いてください")
    uvicorn.run("ichimoku_hosoda.web:app", host=a.host, port=a.port, reload=False)


if __name__ == "__main__":
    main()
