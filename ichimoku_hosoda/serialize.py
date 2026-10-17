"""AnalysisResult → ブラウザ描画用 JSON。"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .engine import AnalysisResult


def _f(v):
    if v is None:
        return None
    try:
        if isinstance(v, (float, np.floating)) and (math.isnan(v) or math.isinf(v)):
            return None
        return float(v)
    except Exception:
        return None


def _d(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def result_to_json(res: AnalysisResult, bars: int = 300) -> dict:
    df = res.df.iloc[-bars:]
    L = res.L.loc[df.index[0]:]
    n = len(res.df)
    out = {
        "label": res.label, "unit": res.unit, "horizon_bars": res.horizon,
        "last_date": _d(df.index[-1]), "close": _f(df["Close"].iloc[-1]),
        "ohlc": {"date": [_d(d) for d in df.index],
                 "open": [_f(v) for v in df["Open"]], "high": [_f(v) for v in df["High"]],
                 "low": [_f(v) for v in df["Low"]], "close": [_f(v) for v in df["Close"]]},
        "lines": {"date": [_d(d) for d in L.index]},
        "pivots": [{"date": _d(p.date), "price": _f(p.price), "kind": p.kind, "rank": p.rank}
                   for p in res.pivots if p.date >= df.index[0]],
        "levels": [{"price": _f(l.price), "source": l.source.split(" / ")[0], "origin": _d(l.origin),
                    "score": round(l.score, 2), "upper": _f(l.upper_limit), "lower": _f(l.lower_limit),
                    "hits": [{"n": h["n"], "date": _d(h["date"]), "what": h["what"]} for h in l.hits[:8]]}
                   for l in res.levels[:8]],
        "calc": [{"price": _f(c["price"]), "kinds": c["kinds"], "n": c["n"], "achieved": bool(c["achieved"]),
                  "pct": round(c["pct_from_close"], 2)} for c in res.calc_clusters],
        "change_dates": [{"date": _d(c["date"]), "offset": int(c["i_min"] - (n - 1)), "score": round(c["score"], 1),
                          "n": c["n_sources"], "kinds": c["kinds"], "details": [i.detail for i in c["items"][:6]]}
                         for c in res.change_clusters[:15]],
        "waves": [{"dir": w.direction, "A": [_d(w.A.date), _f(w.A.price)], "B": [_d(w.B.date), _f(w.B.price)],
                   "C": [_d(w.C.date), _f(w.C.price)], "ab": w.ab, "bc": w.bc, "ac": w.ac,
                   "calc": {k: _f(v) for k, v in w.calc.items()}, "achieved": w.achieved,
                   "d_dates": {k: _d(res.index_ext[i]) if i < len(res.index_ext) else None for k, i in w.d_dates.items()},
                   "d_info": [_dinfo(w, k, i, n, res) for k, i in w.d_dates.items()]}
                  for w in res.waves if "不成立" not in w.note][:8],
        "waveform": res.waveform["form"] if res.waveform else None,
        "p_value": ({"A": [_d(res.p_value["A"].date), _f(res.p_value["A"].price)],
                     "B": [_d(res.p_value["B"].date), _f(res.p_value["B"].price)],
                     "C": [_d(res.p_value["C"].date), _f(res.p_value["C"].price)],
                     "P": _f(res.p_value["P"]), "P_down": _f(res.p_value["P_down"])} if res.p_value else None),
        "position": res.position.__dict__,
        "verdict": res.verdict(),
        "nbars": [{"label": s["label"], "current": {"sign": s["current"]["sign"], "len": s["current"]["len"],
                                                   "start": _d(s["current"]["start"])} if s["current"] else None,
                   "history": s["history"],
                   "thresholds": [{"ahead": t["bars_ahead"], "open": _f(t["ref_open"])} for t in s["future_thresholds"][:8]]}
                  for s in res.nbar_summaries],
        "markdown": res.to_markdown(),
    }
    for col in ("tenkan", "kijun", "span_tk", "span_52", "chikou"):
        out["lines"][col] = [_f(v) for v in L[col]]
    return out


def _dinfo(w, rel, i, n, res):
    from .numbers import nearest_basic
    from .swings import bars_between
    ad = bars_between(w.A.i, i)
    nb = nearest_basic(ad)
    return {"rel": rel, "date": _d(res.index_ext[i]) if i < len(res.index_ext) else None,
            "offset": int(i - (n - 1)), "ad": ad, "ad_basic": nb[0] if nb else None,
            "days": w.ab if rel != "AC=CD" else w.ac}
