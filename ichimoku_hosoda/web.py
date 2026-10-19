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

from . import data, margin
from .engine import analyze
from .serialize import result_to_json

app = FastAPI(title="一目均衡表 節目判定（細田哲生理論）")
_cache: dict[tuple, tuple[float, dict]] = {}
CACHE_SEC = 900


INV_NOTE = "値 = 基準×2 − 実際の価格。反転上の『上げ』は実際の下げ、『押し目』は実際の戻り。y 軸目盛とチャート上のラベルは実際の価格"


def _with_inverse(src: pd.DataFrame, label: str, unit: str, horizon: int, nbar_specs, zz=None, extra_lag=None, bars=300):
    kw = dict(nbar_specs=nbar_specs, extra_lag=extra_lag)
    if zz:
        kw["zz_thresholds"] = zz
    normal = result_to_json(analyze(src, label, unit, horizon, **kw), bars=bars)
    inv_df, ref = data.invert(src)
    inv = result_to_json(analyze(inv_df, "反転" + label, unit, horizon, **kw), bars=bars)
    inv["inverted"] = {"ref": ref, "note": INV_NOTE}
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
            idf, label, "本", 60, ((3, "3本足"), (9, "9本足"), (26, "26本足")), zz, None, 400)
    out["frames"]["daily"], out["frames"]["daily_inv"] = _with_inverse(
        df, "日足", "日", horizon, ((3, "3日足"), (9, "9日足"), (26, "26日足")), None, 3, min(320, len(df)))
    if len(full) >= 60:
        wdf = data.resample(full, "W-FRI")
        if len(wdf) >= 60:
            out["frames"]["weekly"], out["frames"]["weekly_inv"] = _with_inverse(
                wdf, "週足", "週", 26, ((9, "9週足"), (26, "26週足")), (0.05, 0.10, 0.20, 0.35), 9, 260)
    if len(full) >= 300:
        mdf = data.resample(full, "ME")
        if len(mdf) >= 40:
            out["frames"]["monthly"], out["frames"]["monthly_inv"] = _with_inverse(
                mdf, "月足", "ヶ月", 18, ((9, "9ヶ月足"), (26, "26ヶ月足")), (0.08, 0.15, 0.30, 0.50), None, 200)
    return out


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/static/plotly.min.js")
def plotly_js():
    """plotly パッケージ同梱の JS を配信（オフラインでも動く）。"""
    import plotly
    return FileResponse(Path(plotly.__file__).parent / "package_data" / "plotly.min.js", media_type="application/javascript")


@app.get("/api/analyze")
def api_analyze(code: str, period: str = "6y", horizon: int = 60, intraday: int = 1):
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
    res = analyze_all(df, data.normalize_symbol(code), horizon, intra, full_df=full)
    res["margin"] = margin.synthetic_margin() if demo else margin.fetch_margin(code)
    _cache[key] = (time.time(), res)
    return JSONResponse(res)


@app.post("/api/analyze_csv")
async def api_analyze_csv(file: UploadFile = File(...), name: str = Form("CSV"), horizon: int = Form(60)):
    raw = await file.read()
    try:
        df = data.load_csv(io.BytesIO(raw))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"CSV を読めませんでした: {e}") from e
    return JSONResponse(analyze_all(df, name or file.filename, horizon))


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
