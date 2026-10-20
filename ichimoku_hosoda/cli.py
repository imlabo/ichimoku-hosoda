"""コマンドライン:  python -m ichimoku_hosoda 7203 [--csv path] [--out dir] [--period 5y]

日足・週足・月足を順に分析し、Markdown レポートと PNG チャートを出力する。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import data
from .chart import plot
from .engine import analyze


def run(df: pd.DataFrame, name: str, out_dir: Path, horizon_d: int = 60, make_chart: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    results = {}

    daily = analyze(df, "日足", "日", horizon_d, nbar_specs=((3, "3日足"), (9, "9日足"), (26, "26日足")))
    results["daily"] = daily
    reports.append(daily.to_markdown())

    if len(df) >= 60:
        wdf = data.resample(df, "W-FRI")
        if len(wdf) >= 60:
            weekly = analyze(wdf, "週足", "週", 26, nbar_specs=((9, "9週足"), (26, "26週足")),
                             zz_thresholds=(0.05, 0.10, 0.20, 0.35))
            results["weekly"] = weekly
            reports.append(weekly.to_markdown(max_dates=8))
    if len(df) >= 300:
        mdf = data.resample(df, "ME")
        if len(mdf) >= 40:
            monthly = analyze(mdf, "月足", "ヶ月", 18, nbar_specs=((9, "9ヶ月足"), (26, "26ヶ月足")),
                              zz_thresholds=(0.08, 0.15, 0.30, 0.50))
            results["monthly"] = monthly
            reports.append(monthly.to_markdown(max_dates=6, max_levels=5))

    head = [f"# {name} ― 一目均衡表 節目判定",
            f"データ: {df.index[0].date()} 〜 {df.index[-1].date()}（{len(df)} 本）",
            "",
            "> 読み方: ①均衡表で上げ／下げ／もみ合いを直観 → ②三波動構成（起点と中心点）を定める → "
            "③変化日と計算値を出し、その日の反応の有無・計算値達成の有無で前提を検証する。"
            "節目は売買シグナルではなく「ここで相場の現れ方を確認する」ポイント。",
            ""]
    md = "\n".join(head) + "\n\n".join(reports)
    md_path = out_dir / f"{name}_report.md"
    md_path.write_text(md, encoding="utf-8")
    paths = {"report": str(md_path)}
    if make_chart:
        for key, bars in (("daily", 260), ("weekly", 200), ("monthly", 120)):
            if key in results:
                paths[key + "_chart"] = plot(results[key], str(out_dir / f"{name}_{key}.png"), bars=bars,
                                             title=f"{name} {results[key].label}")
    (out_dir / f"{name}_summary.json").write_text(json.dumps(_summary(results), ensure_ascii=False, indent=1, default=str),
                                                  encoding="utf-8")
    paths["json"] = str(out_dir / f"{name}_summary.json")
    return {"results": results, "paths": paths, "markdown": md}


def _summary(results: dict) -> dict:
    s = {}
    for k, r in results.items():
        s[k] = {
            "last": str(r.df.index[-1].date()), "close": float(r.df["Close"].iloc[-1]),
            "position": r.position.__dict__,
            "change_dates": [{"date": str(c["date"].date()), "score": round(c["score"], 2),
                              "details": [i.detail for i in c["items"]]} for c in r.change_clusters[:12]],
            "calc_levels": [{"price": round(c["price"], 2), "kinds": c["kinds"], "n": c["n"], "achieved": c["achieved"]}
                            for c in r.calc_clusters],
            "levels": [{"price": round(l.price, 2), "source": l.source, "origin": str(l.origin.date()),
                        "score": round(l.score, 2), "upper_limit": round(l.upper_limit, 2),
                        "lower_limit": round(l.lower_limit, 2)} for l in r.levels[:10]],
            "nbars": [{"label": n["label"], "current": n["current"], "history": n["history"]} for n in r.nbar_summaries],
        }
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description="一目均衡表 節目判定")
    ap.add_argument("code", help="株式コード（7203 / ^N225 / USDJPY）または名前（--csv 併用時はラベル）")
    ap.add_argument("--csv", help="OHLC CSV を使う（Date,Open,High,Low,Close）")
    ap.add_argument("--period", default="6y", help="yfinance 取得期間（既定 6y）")
    ap.add_argument("--out", default="out", help="出力先ディレクトリ")
    ap.add_argument("--horizon", type=int, default=60, help="日足で何営業日先まで変化日を出すか")
    ap.add_argument("--no-chart", action="store_true")
    ap.add_argument("--demo", action="store_true", help="擬似データで動作確認")
    a = ap.parse_args(argv)

    if a.demo:
        df = data.synthetic()
    elif a.csv:
        df = data.load_csv(a.csv)
    else:
        df = data.fetch(a.code, period=a.period)
    out = run(df, a.code.replace("^", "").replace("=", ""), Path(a.out), a.horizon, not a.no_chart)
    print(out["markdown"])
    print("\n---\n出力:", json.dumps(out["paths"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.exit(main())
