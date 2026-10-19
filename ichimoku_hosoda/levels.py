"""相場水準の候補と評価。

細田の見つけ方：
  ① 基準線・先行スパン(52)が横ばいを演じている値段
  ② 遅行スパンと実線が交わる（触れる）ポイント（値段＝その日の終値、起点＝26本前の足）
  ③ 転換線と基準線の交点、2本の先行スパンの交点
  ④ 重要な変化日に高値・安値となった足
  ⑤ 変化日・計算値で反応がなかった位置（本実装では三波動の計算値未達で経過した C を候補に）
  ⑥ 半値水準（まとめ資料 III-1）：ある期間の高値と安値の半値を出し、その期間内で初めてその値段に
     期間の始まりから過去へさかのぼって最初にその値段を付けた日（図の A）を起点とする。起点からの基本数値日は 高値・安値・水準そのものに止まりやすい
評価：水準を付けた起点からの基本数値日に、同水準・高値・安値が出ているか。
限界値：水準×2 − 反対側の極値。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .lines import SHIFT, flat_runs
from .numbers import ChangeDate, basic_numbers, tolerance
from .swings import Pivot


@dataclass
class Level:
    price: float
    origin_i: int
    origin: pd.Timestamp
    source: str
    hits: list = field(default_factory=list)       # 起点からの基本数値日での現れ方
    score: float = 0.0
    upper_limit: float = float("nan")
    lower_limit: float = float("nan")
    kind: str = ""          # "half" = 期間の高安の半値水準
    range: tuple = ()       # (i0, i1, 高値の足, 安値の足)


def find_levels(df: pd.DataFrame, L: pd.DataFrame, pivs: list[Pivot], lookback: int = 400,
                ranges: list[tuple[int, int, str]] | None = None) -> list[Level]:
    n = len(df)
    start = max(0, n - lookback)
    sub_idx = df.index[start:]
    levels: list[Level] = []
    pos = {d: i for i, d in enumerate(df.index)}

    # ① 横ばい線
    for name, min_len in (("kijun", 6), ("span_52", 8), ("span_tk", 8)):
        s = L[name].reindex(sub_idx)
        for r in flat_runs(s, min_len=min_len):
            levels.append(Level(r["price"], pos[r["start"]], r["start"], f"{_jp(name)}横ばい{r['bars']}本"))

    # ② 遅行スパンが実線に触れる／交わる
    c = df["Close"].values
    for t in range(max(start, SHIFT), n):
        ref = t - SHIFT
        if df["Low"].iloc[ref] <= c[t] <= df["High"].iloc[ref]:
            levels.append(Level(float(c[t]), ref, df.index[ref], f"遅行スパンが実線に触れる（{df.index[t].date()} の終値）"))

    # ③ 線の交点
    for a, b, label in (("tenkan", "kijun", "転換線×基準線"), ("span_tk", "span_52", "先行スパン同士")):
        sa, sb = L[a].reindex(sub_idx).values, L[b].reindex(sub_idx).values
        for k in range(1, len(sub_idx)):
            if np.isnan(sa[k]) or np.isnan(sb[k]) or np.isnan(sa[k - 1]) or np.isnan(sb[k - 1]):
                continue
            if (sa[k] - sb[k]) * (sa[k - 1] - sb[k - 1]) < 0:
                levels.append(Level(float((sa[k] + sb[k]) / 2), pos[sub_idx[k]], sub_idx[k], f"{label}の交点"))

    # ④ 重要な極（rank>=3）とその足の半値
    for p in pivs:
        if p.rank >= 3 and p.i >= start:
            levels.append(Level(p.price, p.i, p.date, f"重要{ '高値' if p.kind=='H' else '安値'}"))
            levels.append(Level(float((df['High'].iloc[p.i] + df['Low'].iloc[p.i]) / 2), p.i, p.date,
                                f"重要{'高値' if p.kind=='H' else '安値'}日の足の半値"))

    levels = _merge(levels)
    halves = half_levels(df, ranges)
    for lv in levels + halves:
        _evaluate(lv, df, pivs)
    for h in halves:
        h.score += 3.0     # 起点のはっきりした水準として優先
    return sorted(levels + halves, key=lambda x: -x.score)


def half_level(df: pd.DataFrame, i0: int, i1: int, label: str) -> Level | None:
    """df[i0..i1] の高値・安値の半値。起点は、期間の始まりの極（高値・安値の先に付いた方）から時間軸を過去へさかのぼって
    最初にその値段を付けた足（まとめ資料 III-1 の図の A）。過去に無ければ期間内で最初に付けた足。"""
    n = len(df)
    i0, i1 = max(0, i0), min(n - 1, i1)
    if i1 - i0 < 4:
        return None
    hi, lo = df["High"].values, df["Low"].values
    seg_hi, seg_lo = hi[i0:i1 + 1], lo[i0:i1 + 1]
    jh, jl = i0 + int(np.argmax(seg_hi)), i0 + int(np.argmin(seg_lo))
    mid = (hi[jh] + lo[jl]) / 2
    o, how = None, "さかのぼって最初に到達"
    for k in range(min(jh, jl), -1, -1):   # 期間の高値・安値のうち先に付いた方（＝期間の始まりの極）からさかのぼる
        if lo[k] <= mid <= hi[k]:
            o = k
            break
    if o is None:
        touch = np.where((seg_lo <= mid) & (seg_hi >= mid))[0]
        if not len(touch):
            return None
        o, how = i0 + int(touch[0]), "期間内で最初に到達"
    d = lambda k: df.index[k].strftime("%Y-%m-%d" if df.index[k].hour == 0 else "%m-%d %H:%M")
    lv = Level(float(mid), o, df.index[o],
               f"半値水準（{label}の高値 {hi[jh]:,.1f}［{d(jh)}］と安値 {lo[jl]:,.1f}［{d(jl)}］の半値、{how} {d(o)}）")
    lv.kind = "half"
    lv.range = (int(i0), int(i1), int(jh), int(jl))
    return lv


def half_levels(df: pd.DataFrame, ranges: list[tuple[int, int, str]] | None = None) -> list[Level]:
    n = len(df)
    rs = list(ranges or [])
    for b in (26, 52, 76, 129, 172, 257):
        if b < n:
            rs.append((n - b, n - 1, f"直近{b}本"))
    out: list[Level] = []
    for i0, i1, lab in rs:
        lv = half_level(df, i0, i1, lab)
        if lv is None:
            continue
        dup = next((x for x in out if abs(x.price - lv.price) / lv.price <= 0.004 and x.origin_i == lv.origin_i), None)   # 高値・安値が同じ期間は 1 本にまとまる
        if dup is None:
            out.append(lv)
    return out


def _jp(name: str) -> str:
    return {"kijun": "基準線", "tenkan": "転換線", "span_52": "先行スパン(52)", "span_tk": "先行スパン(基準転換)"}[name]


def _merge(levels: list[Level], tol: float = 0.004) -> list[Level]:
    levels.sort(key=lambda x: x.price)
    out: list[Level] = []
    for lv in levels:
        if out and abs(lv.price - out[-1].price) / out[-1].price <= tol:
            keep = out[-1]
            # 起点は最も古いものを採用、根拠は連結
            if lv.origin_i < keep.origin_i:
                keep.origin_i, keep.origin = lv.origin_i, lv.origin
            if lv.source not in keep.source:
                keep.source += " / " + lv.source
            keep.price = (keep.price + lv.price) / 2
        else:
            out.append(lv)
    return out


def _evaluate(lv: Level, df: pd.DataFrame, pivs: list[Pivot], same_tol: float = 0.01):
    n = len(df)
    table = basic_numbers()
    piv_pos = {p.i: p for p in pivs}
    hits = []
    for b in table:
        j = lv.origin_i + b - 1
        if j >= n:
            break
        tag = None
        for dj in range(-tolerance(b), tolerance(b) + 1):
            k = j + dj
            if 0 <= k < n:
                if k in piv_pos and piv_pos[k].rank >= 2:
                    tag = ("高値" if piv_pos[k].kind == "H" else "安値", k)
                    break
        if tag is None:
            row = df.iloc[j]
            if row["Low"] <= lv.price <= row["High"]:
                tag = ("同水準", j)
        if tag:
            hits.append({"n": b, "date": df.index[tag[1]], "what": tag[0]})
    lv.hits = hits
    elapsed = n - lv.origin_i
    expected = max(1, sum(1 for b in table if b <= elapsed))
    lv.score = len(hits) / expected * 10 + (2 if "横ばい" in lv.source else 0) + (1.5 if "遅行" in lv.source else 0)
    hi_since = df["High"].iloc[lv.origin_i:].max()
    lo_since = df["Low"].iloc[lv.origin_i:].min()
    lv.upper_limit = 2 * lv.price - lo_since   # 安値を反対側とした場合の上限
    lv.lower_limit = 2 * lv.price - hi_since   # 高値を反対側とした場合の下限


def level_change_dates(levels: list[Level], index_ext: pd.DatetimeIndex, n_bars: int, horizon: int,
                       top: int = 6, unit: str = "日") -> list[ChangeDate]:
    out = []
    for lv in levels[:top]:
        for b in basic_numbers():
            j = lv.origin_i + b - 1
            if n_bars - 3 <= j < n_bars + horizon:
                out.append(ChangeDate(j, index_ext[j], "level", b, lv.origin, "level",
                                      f"{'半値水準' if lv.kind == 'half' else '相場水準'} {lv.price:,.1f}（{lv.source.split(' / ')[0] if lv.kind != 'half' else '起点'}）起点 {lv.origin.date()} から {b}{unit}目",
                                      (0.5 + min(lv.score, 10) / 10) * (1.5 if lv.kind == "half" else 1.0)))
    return out
