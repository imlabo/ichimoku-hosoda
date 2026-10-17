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

from . import data
from .engine import analyze
from .serialize import result_to_json

app = FastAPI(title="一目均衡表 節目判定（細田哲生理論）")
_cache: dict[tuple, tuple[float, dict]] = {}
CACHE_SEC = 900


def analyze_all(df: pd.DataFrame, name: str, horizon: int = 60) -> dict:
    out = {"name": name, "from": str(df.index[0].date()), "to": str(df.index[-1].date()), "bars": len(df), "frames": {}}
    out["frames"]["daily"] = result_to_json(
        analyze(df, "日足", "日", horizon, nbar_specs=((3, "3日足"), (9, "9日足"), (26, "26日足"))), bars=320)
    if len(df) >= 60:
        wdf = data.resample(df, "W-FRI")
        if len(wdf) >= 60:
            out["frames"]["weekly"] = result_to_json(
                analyze(wdf, "週足", "週", 26, nbar_specs=((9, "9週足"), (26, "26週足")),
                        zz_thresholds=(0.05, 0.10, 0.20, 0.35)), bars=260)
    if len(df) >= 300:
        mdf = data.resample(df, "ME")
        if len(mdf) >= 40:
            out["frames"]["monthly"] = result_to_json(
                analyze(mdf, "月足", "ヶ月", 18, nbar_specs=((9, "9ヶ月足"), (26, "26ヶ月足")),
                        zz_thresholds=(0.08, 0.15, 0.30, 0.50)), bars=200)
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
def api_analyze(code: str, period: str = "6y", horizon: int = 60):
    key = (code.strip().upper(), period, horizon)
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_SEC:
        return JSONResponse(hit[1])
    try:
        df = data.synthetic() if code.strip().upper() == "DEMO" else data.fetch(code, period=period)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"データ取得に失敗しました: {e}") from e
    res = analyze_all(df, data.normalize_symbol(code), horizon)
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
