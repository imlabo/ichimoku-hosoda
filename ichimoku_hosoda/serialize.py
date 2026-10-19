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


_FMT = {"fmt": "%Y-%m-%d"}


def _d(ts) -> str:
    return pd.Timestamp(ts).strftime(_FMT["fmt"])


def result_to_json(res: AnalysisResult, bars: int = 300) -> dict:
    _FMT["fmt"] = "%m-%d %H:%M" if res.unit == "本" else "%Y-%m-%d"
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
        "pivots": [{"date": _d(p.date), "price": _f(p.price), "kind": p.kind, "rank": p.rank, "i": p.i}
                   for p in res.pivots if p.date >= df.index[0]],
        "levels": [_level(l, res, n) for l in res.levels[:8]],
        "future_dates": [_d(d) for d in res.index_ext[n:n + res.horizon]],
        "future_all": [_d(d) for d in res.index_ext[n:n + 260]],   # D 予測日など遠い日付の並び順用
        "calc": [{"price": _f(c["price"]), "kinds": c["kinds"], "n": c["n"], "achieved": bool(c["achieved"]),
                  "pct": round(c["pct_from_close"], 2)} for c in res.calc_clusters],
        "change_dates": clusters_to_json(res.change_clusters[:15], n),
        "n_total": n, "ohlc_from": n - len(df),
        "waves": [{"dir": w.direction, "tag": w.tag, "A": [_d(w.A.date), _f(w.A.price)], "A_rank": w.A.rank, "A_i": w.A.i, "B_i": w.B.i, "C_i": w.C.i,
                   "B": [_d(w.B.date), _f(w.B.price)],
                   "C": [_d(w.C.date), _f(w.C.price)], "ab": w.ab, "bc": w.bc, "ac": w.ac,
                   "calc": {k: _f(v) for k, v in w.calc.items()}, "achieved": w.achieved,
                   "d_dates": {k: _d(res.index_ext[i]) if i < len(res.index_ext) else None for k, i in w.d_dates.items()},
                   "d_info": [_dinfo(w, k, i, n, res) for k, i in w.d_dates.items()]}
                  for w in res.waves if "不成立" not in w.note][:16],
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
    if res.extra_lag and f"chikou_{res.extra_lag}" in L.columns:
        k = res.extra_lag
        out["lines"]["chikou_x"] = [_f(v) for v in L[f"chikou_{k}"]]
        out["extra_lag"] = {"n": k, "state": _lag_state(res.df, k)}
    return out


def _lag_state(df: pd.DataFrame, k: int) -> dict:
    """k 本遅行スパン＝本日終値 vs k 本前の実線。陽連・陰連（＝k本足）も返す。"""
    c = df["Close"].values
    n = len(df)
    if n <= k:
        return {}
    ref = df.iloc[-1 - k]
    rel = "up" if c[-1] > ref["High"] else "down" if c[-1] < ref["Low"] else "touch"
    sign = np.sign(c[k:] - c[:-k])
    run, cur = 0, sign[-1]
    for v in sign[::-1]:
        if v == cur and v != 0:
            run += 1
        else:
            break
    return {"rel": rel, "ref_date": _d(df.index[-1 - k]), "ref_high": _f(ref["High"]), "ref_low": _f(ref["Low"]),
            "run": int(run), "run_sign": "陽" if cur > 0 else "陰" if cur < 0 else "同値"}


def _dinfo(w, rel, i, n, res):
    from .numbers import nearest_basic
    from .swings import bars_between
    ad = bars_between(w.A.i, i)
    nb = nearest_basic(ad)
    return {"rel": rel, "date": _d(res.index_ext[i]) if i < len(res.index_ext) else None,
            "offset": int(i - (n - 1)), "ad": ad, "ad_basic": nb[0] if nb else None,
            "days": w.ab if rel != "AC=CD" else w.ac}


def _level(l, res, n):
    from .numbers import basic_numbers
    df = res.df
    c = float(df["Close"].iloc[-1])
    elapsed = n - l.origin_i  # 起点を 1 本目とした現在の本数 = elapsed
    nxt = [b for b in basic_numbers() if b > elapsed][:3]
    hi_since = float(df["High"].iloc[l.origin_i:].max())
    lo_since = float(df["Low"].iloc[l.origin_i:].min())
    # 直近 26 本で水準を跨いだ回数（押し戻りとして機能せず水準化しているか）
    seg = df.iloc[-26:]
    cross = int(((seg["Low"] <= l.price) & (seg["High"] >= l.price)).sum())
    return {"price": _f(l.price), "source": l.source.split(" / ")[0], "sources": l.source.split(" / "),
            "origin": _d(l.origin), "elapsed": int(elapsed), "score": round(l.score, 2),
            "upper": _f(l.upper_limit), "lower": _f(l.lower_limit),
            "hi_since": hi_since, "lo_since": lo_since,
            "rel": "above" if c > l.price else "below", "pct": round((c / l.price - 1) * 100, 2),
            "touch26": cross,
            "next_basic": [{"n": b, "date": _d(res.index_ext[l.origin_i + b - 1]) if l.origin_i + b - 1 < len(res.index_ext) else None,
                            "ahead": l.origin_i + b - 1 - (n - 1)} for b in nxt],
            "hits": [{"n": h["n"], "date": _d(h["date"]), "what": h["what"]} for h in l.hits[:10]]}


def waves_to_json(waves, res, n):
    import copy
    tmp = copy.copy(res)
    tmp.waves = waves
    return result_to_json(tmp, bars=10)["waves"]


def clusters_to_json(clusters, n):
    return [{"date": _d(c["date"]), "offset": int(c["i_min"] - (n - 1)), "score": round(c["score"], 1),
             "n": c["n_sources"], "kinds": c["kinds"], "details": [i.detail for i in c["items"][:6]]}
            for c in clusters]
