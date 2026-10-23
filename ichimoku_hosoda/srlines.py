"""支持線・抵抗線（水平線）。GOLD 短期売買チャート（imlabo/gold-chart）の考え方を株向けに移植。

1. 「お椀の水平線」が主役：過去のトレンドの起点（戻り売り・押し目買いの起点）から大きく離れたあと、
   価格が全戻ししてきた価格。週足・日足・1時間足で探し、深く・大きく・初回の戻りほど強い。
2. 移動平均線（20・210、ほか 5・75）との重なり：近くにあるほど加点。別の時間足の 20・210 が重なればさらに加点。
3. そのほか：山・谷（反転した価格）、三尊・逆三尊のネックライン候補、はっきりした一波の半値戻し、
   大きなローソク足の四本値、基準線・先行スパン2の横ばい、キリの良い節目。
4. 減点：その後に何度も終値で抜けた価格。
現在値の上下それぞれ最大 4 本。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .lines import half_line

SIDE = 4
MA_P = (5, 20, 75, 210)
MA_W = {"W": {20: 1.8, 210: 1.6, 75: 0.8, 5: 0.2},
        "D": {20: 1.6, 210: 1.6, 75: 0.7, 5: 0.2},
        "H": {20: 1.1, 210: 1.2, 75: 0.5, 5: 0.1}}
TF = {"W": dict(w=3.5, piv=3, bowl=7, label="週足", days=7),
      "D": dict(w=3.0, piv=3, bowl=6, label="日足", days=1),
      "H": dict(w=1.2, piv=5, bowl=2.5, label="1時間足", days=1 / 6)}


def _pivots(h, l, k):
    out = []
    n = len(h)
    for i in range(k, n - k):
        if h[i] == max(h[i - k:i + k + 1]):
            out.append((i, 1, float(h[i])))
        if l[i] == min(l[i - k:i + k + 1]):
            out.append((i, -1, float(l[i])))
    return out


def _atr(df, n=14):
    h, l, c = df["High"].values, df["Low"].values, df["Close"].values
    if len(df) < n + 2:
        return float("nan")
    tr = np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))
    return float(tr[-n:].mean())


def _flat_runs(arr, idx, min_len=9, max_back=400):
    out = []
    n = len(arr)
    i, stop = n - 1, max(0, n - max_back)
    while i > stop:
        v = arr[i]
        if not np.isfinite(v):
            i -= 1
            continue
        j = i
        while j - 1 >= stop and arr[j - 1] == v:
            j -= 1
        if i - j + 1 >= min_len:
            out.append((float(v), i - j + 1, idx[i]))
        i = j - 1
    return out


def _nice(x):
    p = 10 ** math.floor(math.log10(x))
    m = x / p
    return (1 if m < 1.5 else 2 if m < 3.5 else 5 if m < 7.5 else 10) * p


def _bowls(df, f, price, rng, tol, now, k_tf, rec_days):
    h, l, idx = df["High"].values, df["Low"].values, df.index
    L = len(df) - 1
    atr = _atr(df)
    if not np.isfinite(atr) or atr <= 0:
        return []
    out = []
    rec = lambda t: 0.3 + 0.7 * math.exp(-(now - t).total_seconds() / 86400 / rec_days)
    for i, t, p in _pivots(h, l, f["piv"]):
        if i >= L - 2 or abs(p - price) > rng:
            continue
        if t == 1:                                # 戻り売りの起点 → 上の抵抗
            if price >= p:
                continue
            mnI = i + 1 + int(np.argmin(l[i + 1:]))
            mn = l[mnI]
            depth = p - mn
            if depth < 1.5 * atr or (price - mn) / depth < 0.5:
                continue
            on, retests = False, 0
            for j in range(mnI, L):
                tt = h[j] >= p - tol
                if tt and not on:
                    retests += 1
                on = tt
            pre = max(0, i - f["piv"] * 6)
            pre_lo = l[pre:i].min() if i > pre else np.nan
            origin = min(1, max(0.3, (p - pre_lo) / atr / 2)) if np.isfinite(pre_lo) else 0.6
        else:                                     # 押し目買いの起点 → 下の支持
            if price <= p:
                continue
            mxI = i + 1 + int(np.argmax(h[i + 1:]))
            mx = h[mxI]
            depth = mx - p
            if depth < 1.5 * atr or (mx - price) / depth < 0.5:
                continue
            on, retests = False, 0
            for j in range(mxI, L):
                tt = l[j] <= p + tol
                if tt and not on:
                    retests += 1
                on = tt
            pre = max(0, i - f["piv"] * 6)
            pre_hi = h[pre:i].max() if i > pre else np.nan
            origin = min(1, max(0.3, (pre_hi - p) / atr / 2)) if np.isfinite(pre_hi) else 0.6
        size = min(3, max(0.75, depth / atr / 2)) * min(1.8, max(0.6, (L - i) / (f["piv"] * 5)))
        decay = 1 if retests == 0 else 0.85 if retests == 1 else 0.7 if retests == 2 else 0.5
        out.append(dict(p=p, w=f["bowl"] * size * origin * rec(idx[i]) * decay, tf=k_tf, kind="B", t=idx[i],
                        depth=float(depth), depth_atr=float(depth / atr), retests=retests, first=retests == 0))
    return out


def compute(daily: pd.DataFrame, weekly: pd.DataFrame | None = None, hourly: pd.DataFrame | None = None) -> dict:
    if hourly is not None and len(hourly) and daily is not None and len(daily):
        # 1時間足が日足と同じ値動き（直近の終値・日付が揃っている）ときだけ使う
        same_px = abs(float(hourly["Close"].iloc[-1]) / float(daily["Close"].iloc[-1]) - 1) < 0.03
        same_day = abs((hourly.index[-1].normalize() - daily.index[-1].normalize()).days) <= 4
        if not (same_px and same_day):
            hourly = None
    frames = {k: v for k, v in (("W", weekly), ("D", daily), ("H", hourly)) if v is not None and len(v) >= 40}
    if "D" not in frames:
        return {"levels": []}
    D = frames["D"]
    price = float(D["Close"].iloc[-1])
    now = D.index[-1]
    atr = _atr(D)
    if not np.isfinite(atr) or atr <= 0:
        atr = price * 0.02
    tol = max(atr * 0.12, price * 0.0006)
    rng = atr * 6
    rec_days = 120
    rec = lambda t: 0.3 + 0.7 * math.exp(-max(0, (now - t).total_seconds()) / 86400 / rec_days)
    cands = []
    mas = []
    for k, df in frames.items():
        f = TF[k]
        h, l, o, c, idx = df["High"].values, df["Low"].values, df["Open"].values, df["Close"].values, df.index
        L = len(df) - 1
        piv = _pivots(h, l, f["piv"])
        for i, t, p in piv:
            if abs(p - price) <= rng:
                cands.append(dict(p=p, w=f["w"] * rec(idx[i]), tf=k, kind="H" if t == 1 else "L", t=idx[i]))
        if f["bowl"]:
            cands += _bowls(df, f, price, rng, tol, now, k, rec_days)
        if k in ("W", "D"):
            kij = half_line(df, 26).values
            spb = half_line(df, 52).values
            for arr, nm, mult in ((kij, "基準線", 1.0), (spb, "先行スパン2", 1.2)):
                for v, ln, tt in _flat_runs(arr, idx):
                    if abs(v - price) <= rng:
                        cands.append(dict(p=v, w=f["w"] * mult * min(2, ln / 9) * rec(tt), tf=k, kind="K", nm=nm, t=tt))
            # ネックライン候補：直近の極値の手前の押し目（戻り目）
            P = piv[-8:]
            if len(P) >= 4:
                hi = max((x for x in P if x[1] == 1), key=lambda x: x[2], default=None)
                lo = min((x for x in P if x[1] == -1), key=lambda x: x[2], default=None)
                def before(x, t):
                    r = None
                    for y in P:
                        if y[0] < x[0] and y[1] == t:
                            r = y
                    return r
                nk1 = before(hi, -1) if hi else None
                nk2 = before(lo, 1) if lo else None
                if nk1 and price > nk1[2] and abs(nk1[2] - price) <= rng:
                    cands.append(dict(p=nk1[2], w=f["w"] * 0.9, tf=k, kind="N", t=idx[nk1[0]]))
                if nk2 and price < nk2[2] and abs(nk2[2] - price) <= rng:
                    cands.append(dict(p=nk2[2], w=f["w"] * 0.9, tf=k, kind="N", t=idx[nk2[0]]))
            # 半値戻し：直近のはっきりした一波
            legs = [(piv[j], piv[j + 1]) for j in range(max(0, len(piv) - 12), len(piv) - 1) if piv[j][1] != piv[j + 1][1]]
            a14 = _atr(df)
            if legs and np.isfinite(a14):
                big = max(legs, key=lambda g: abs(g[1][2] - g[0][2]))
                if abs(big[1][2] - big[0][2]) >= 3 * a14:
                    half = (big[0][2] + big[1][2]) / 2
                    if abs(half - price) <= rng:
                        cands.append(dict(p=half, w=f["w"] * 0.6, tf=k, kind="F", t=idx[big[1][0]]))
            # 大きなローソク足の四本値
            if np.isfinite(a14):
                for j in range(max(1, L - 30), L):
                    if abs(c[j] - o[j]) < 1.3 * a14:
                        continue
                    nm0 = "大陽線" if c[j] >= o[j] else "大陰線"
                    for v, nm in ((o[j], "始値"), (c[j], "終値"), (h[j], "高値"), (l[j], "安値")):
                        if abs(v - price) <= rng:
                            cands.append(dict(p=float(v), w=f["w"] * 0.35 * rec(idx[j]), tf=k, kind="C", nm=nm0 + nm, t=idx[j]))
        # 移動平均線（現在値）
        for per in MA_P:
            if len(df) >= per:
                v = float(df["Close"].iloc[-per:].mean())
                w = MA_W[k].get(per, 0)
                if w:
                    mas.append(dict(k=k, per=per, v=v, w=w))
    if not cands:
        return {"levels": [], "price": price, "tol": tol}
    cands.sort(key=lambda x: x["p"])
    cl = []
    for x in cands:
        last = cl[-1] if cl else None
        if last and x["p"] - last["lo"] <= tol * 2 and abs(x["p"] - last["p"]) <= tol:
            last["items"].append(x)
            last["sw"] += x["w"]
            last["p"] = sum(y["p"] * y["w"] for y in last["items"]) / last["sw"]
        else:
            cl.append(dict(items=[x], sw=x["w"], p=x["p"], lo=x["p"]))
    rb = _nice(price / 10)
    rs = rb / 2
    closes = D["Close"].values
    for z in cl:
        tfs = {x["tf"] for x in z["items"]}
        has_h = any(x["kind"] == "H" for x in z["items"])
        has_l = any(x["kind"] == "L" for x in z["items"])
        sc = z["sw"]
        if has_h and has_l:
            sc *= 1.3
        sc *= 1 + 0.2 * (len(tfs) - 1)
        r1, r2 = abs(z["p"] - round(z["p"] / rb) * rb), abs(z["p"] - round(z["p"] / rs) * rs)
        z["round"] = rb if r1 <= tol else rs if r2 <= tol else 0
        sc += 2 if z["round"] == rb else 1 if z["round"] == rs else 0
        z["mas"] = []
        bonus = 0
        for m in mas:
            d = abs(m["v"] - z["p"])
            if d > tol * 2:
                continue
            close = 1 - d / (tol * 2)
            bonus += m["w"] * (0.4 + 0.6 * close)
            z["mas"].append(dict(m, d=d))
        if len({m["k"] for m in z["mas"] if m["per"] in (20, 210)}) >= 2:
            bonus *= 1.35
        sc *= 1 + min(1.6, 0.45 * bonus)
        last_t = max(x["t"] for x in z["items"])
        cross, side = 0, 0
        for t, cv in zip(D.index, closes):
            if t <= last_t:
                continue
            sd = 1 if cv > z["p"] + tol * 0.5 else -1 if cv < z["p"] - tol * 0.5 else 0
            if sd and side and sd != side:
                cross += 1
            if sd:
                side = sd
        z["cross"] = cross
        sc *= 1 / (1 + 0.18 * cross)
        bowls = [x for x in z["items"] if x["kind"] == "B"]
        z["bowl"] = max(bowls, key=lambda x: x["w"]) if bowls else None
        z["score"] = sc
        z["tfs"] = tfs
        z["flip"] = has_h and has_l
    above = sorted([z for z in cl if z["p"] > price + tol * 0.25], key=lambda z: -z["score"])[:SIDE]
    below = sorted([z for z in cl if z["p"] < price - tol * 0.25], key=lambda z: -z["score"])[:SIDE]
    levels = [dict(z, side="res") for z in above] + [dict(z, side="sup") for z in below]
    mx = max([z["score"] for z in levels] + [1e-9])
    for z in levels:
        z["str"] = z["score"] / mx
    for i, z in enumerate(sorted([z for z in levels if z["side"] == "res"], key=lambda z: z["p"])):
        z["name"] = f"R{i + 1}"
    for i, z in enumerate(sorted([z for z in levels if z["side"] == "sup"], key=lambda z: -z["p"])):
        z["name"] = f"S{i + 1}"
    return {"levels": [_out(z, price) for z in sorted(levels, key=lambda z: -z["p"])], "price": price, "tol": tol,
            "used": [TF[k]["label"] for k in frames]}


def _d(t):
    t = pd.Timestamp(t)
    return t.strftime("%m-%d %H:%M") if (t.hour or t.minute) else t.strftime("%Y-%m-%d")


def _out(z, price):
    tags, why = [], []
    b = z["bowl"]
    if b:
        tags.append(f"お椀（{TF[b['tf']]['label']}{'・初回' if b['first'] else ''}）")
        why.append(f"{TF[b['tf']]['label']}のお椀：起点 {_d(b['t'])} から {b['depth']:,.1f}（ATR×{b['depth_atr']:.1f}）離れて全戻し"
                   + ("・初回の戻り（最も効きやすい）" if b["first"] else f"・{b['retests'] + 1}回目の接触"))
    if z["mas"]:
        best = max(z["mas"], key=lambda m: m["w"] - m["d"] / (abs(price) + 1))
        tags.append(f"MA重なり {TF[best['k']]['label']}{best['per']}" + (f" 他{len(z['mas']) - 1}" if len(z["mas"]) > 1 else ""))
        for m in sorted(z["mas"], key=lambda m: m["d"])[:3]:
            why.append(f"{TF[m['k']]['label']}{m['per']}MA が " + ("ほぼ同値" if m["d"] < price * 0.0005 else f"±{m['d']:,.1f}"))
    for k in ("W", "D", "H"):
        it = [x for x in z["items"] if x["tf"] == k]
        lb = TF[k]["label"]
        nh, nl = sum(x["kind"] == "H" for x in it), sum(x["kind"] == "L" for x in it)
        if nh:
            why.append(f"{lb}高値×{nh}")
        if nl:
            why.append(f"{lb}安値×{nl}")
        for nm in sorted({x["nm"] for x in it if x["kind"] == "K"}):
            why.append(f"{lb}{nm}横ばい")
        if any(x["kind"] == "N" for x in it):
            why.append(f"{lb}ネックライン候補")
        if any(x["kind"] == "F" for x in it):
            why.append(f"{lb}半値戻し")
        for nm in sorted({x["nm"] for x in it if x["kind"] == "C"}):
            why.append(f"{lb}{nm}")
    if any(x["kind"] == "N" for x in z["items"]):
        tags.append("ネック候補")
    if any(x["kind"] == "F" for x in z["items"]):
        tags.append("半値戻し")
    if z["flip"]:
        why.append("役割の入れ替わり")
    if z["round"]:
        tags.append(f"{z['round']:,.0f}の節目")
        why.append(f"{z['round']:,.0f}の節目")
    if z["cross"] >= 3:
        why.append(f"その後{z['cross']}回抜けている")
    return {"name": z["name"], "side": z["side"], "price": float(z["p"]), "str": round(float(z["str"]), 3),
            "pct": round((z["p"] / price - 1) * 100, 2), "tags": tags, "why": "・".join(why),
            "bowl": ({"date": _d(b["t"]), "first": bool(b["first"])} if b else None)}
