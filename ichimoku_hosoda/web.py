"""ローカル Web アプリ:  python -m ichimoku_hosoda.web  → http://127.0.0.1:8000

株式コードを入力 → yfinance で取得 → 日足・週足・月足を分析 → 対話チャートとレポートを表示。
CSV アップロードにも対応。
"""
from __future__ import annotations

import io
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
def api_analyze(code: str, period: str = "6y", horizon: int = 60, intraday: int = 1):
    raw_code = code
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
        for key2, interval, label, zz in (("m5", "5m", "5分足", (0.004, 0.008, 0.016, 0.03)),
                                           ("h1", "1h", "1時間足", (0.01, 0.02, 0.04, 0.08))):
            try:
                idf = data.synthetic_intraday(interval) if demo else data.fetch_intraday(code, interval)
            except Exception:  # noqa: BLE001
                idf = None
            intra[key2] = (idf, label, zz)
    _LAST.clear()
    _ERRORS.clear()
    res = analyze_all(df, data.normalize_symbol(code), horizon, intra, full_df=full)
    res["frames"] = {k: v for k, v in res["frames"].items() if v is not None}
    res["notes"] = res.get("notes", []) + list(_ERRORS)
    res["display_name"] = disp or (names.name_for(code) if not demo else "デモデータ")
    res["input"] = raw_code
    if "daily" not in res["frames"]:
        raise HTTPException(500, "日足の分析に失敗しました: " + " / ".join(_ERRORS))
    _results[(key[0], period)] = dict(_LAST)
    res["margin"] = margin.synthetic_margin(end=full.index[-1]) if demo else margin.fetch_margin(code)
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
