"""三波動構成（時間関係・計算値）と波形分類。

定型（細田）: A→B 第一波動、B→C 第二波動（押し/戻り）、C→D 第三波動
  ① 時間: AB=BD, AB=CD, AC=CD      ② A〜D が基本数値      ③ D が計算値を達成
計算値（上げ）: E = B+(B−A), N = C+(B−A), V = B+(B−C), NT = C+(C−A)
  E 達成 → 方向持続しやすい／N・V は BC 半値・B を中心とするもみ合いの範疇（警戒）／NT は限界
P 計算値: 内はらみ（A 高値 > B 高値、C 安値）→ C + (A−B)、割り込めば下げ三波動を考える
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .numbers import ChangeDate, nearest_basic
from .swings import Pivot, bars_between


@dataclass
class Wave3:
    A: Pivot
    B: Pivot
    C: Pivot
    direction: str                       # 'up' / 'down'
    ab: int
    bc: int
    ac: int
    d_dates: dict = field(default_factory=dict)      # {'AB=BD': i, 'AB=CD': i, 'AC=CD': i}
    calc: dict = field(default_factory=dict)         # {'E':..,'N':..,'V':..,'NT':..}
    achieved: dict = field(default_factory=dict)     # 計算値ごとの達成有無
    extreme_since_c: float = float("nan")
    note: str = ""

    def ad_basic(self, i_d: int):
        return nearest_basic(bars_between(self.A.i, i_d))


def build_waves(pivs: list[Pivot], df: pd.DataFrame, max_recent: int = 8, max_back: int = 4) -> list[Wave3]:
    """直近 max_recent 個の極を C（第二波動の終点）候補とし、起点 A はそれ以前の同種の極から選ぶ。"""
    out: list[Wave3] = []
    n = len(pivs)
    hi, lo = df["High"].values, df["Low"].values
    for ci in range(max(2, n - max_recent), n):
        C = pivs[ci]
        for ai in range(ci - 2, max(-1, ci - 2 - 2 * max_back), -2):
            A = pivs[ai]
            if A.kind != C.kind:
                continue
            # B = A〜C 間の極値
            seg = slice(A.i, C.i + 1)
            if A.kind == "L":
                bi = A.i + int(np.argmax(hi[seg]))
                B = Pivot(bi, df.index[bi], float(hi[bi]), "H", 1)
                direction = "up"
                if B.price <= max(A.price, C.price):
                    continue
            else:
                bi = A.i + int(np.argmin(lo[seg]))
                B = Pivot(bi, df.index[bi], float(lo[bi]), "L", 1)
                direction = "down"
                if B.price >= min(A.price, C.price):
                    continue
            for p in pivs:
                if p.i == bi:
                    B = p
            ab, bc, ac = bars_between(A.i, B.i), bars_between(B.i, C.i), bars_between(A.i, C.i)
            if ab < 3:
                continue
            w = Wave3(A, B, C, direction, ab, bc, ac)
            w.d_dates = {"AB=BD": B.i + ab - 1, "AB=CD": C.i + ab - 1, "AC=CD": C.i + ac - 1}
            a, b, c = A.price, B.price, C.price
            if direction == "up":
                w.calc = {"E": b + (b - a), "N": c + (b - a), "V": b + (b - c), "NT": c + (c - a)}
                w.extreme_since_c = float(hi[C.i:].max())
                w.achieved = {k: w.extreme_since_c >= v for k, v in w.calc.items()}
                w.note = "N候補（押し目が起点より上）" if c > a else "押しが起点を割る→上げ三波動としては不成立"
            else:
                w.calc = {"E": b - (a - b), "N": c - (a - b), "V": b - (c - b), "NT": c - (a - c)}
                w.extreme_since_c = float(lo[C.i:].min())
                w.achieved = {k: w.extreme_since_c <= v for k, v in w.calc.items()}
                w.note = "N候補（戻りが起点より下）" if c < a else "戻りが起点を超える→下げ三波動としては不成立"
            out.append(w)
    out.sort(key=lambda w: -(2 * (w.A.rank + w.C.rank) + min(w.ab + w.bc, 120) / 20 + (3 if w.C.i >= len(df) - 60 else 0)))
    return out


def wave_change_dates(waves: list[Wave3], index_ext: pd.DatetimeIndex, n_bars: int, horizon: int,
                      unit: str = "日") -> list[ChangeDate]:
    out = []
    for w in waves:
        if "不成立" in w.note:
            continue
        for rel, j in w.d_dates.items():
            if n_bars - 3 <= j < n_bars + horizon:
                nb = w.ad_basic(j)
                extra = f"、A〜D {bars_between(w.A.i, j)}{unit}≒基本数値{nb[0]}" if nb else ""
                out.append(ChangeDate(j, index_ext[j], "wave", j - (w.B.i if rel == "AB=BD" else w.C.i) + 1,
                                      w.C.date, w.C.kind,
                                      f"{'上げ' if w.direction=='up' else '下げ'}三波動 A{w.A.date.date()} B{w.B.date.date()} "
                                      f"C{w.C.date.date()} の {rel}（{w.ab if rel!='AC=CD' else w.ac}{unit}）{extra}",
                                      1.2 + 0.15 * (w.A.rank + w.C.rank) + (0.5 if nb else 0)))
    return out


def calc_levels(waves: list[Wave3], last_close: float) -> list[dict]:
    """計算値を価格でクラスタ化（±0.4%）。達成済み／未達と、現値からの位置を付ける。"""
    items = []
    for w in waves:
        if "不成立" in w.note:
            continue
        for k, v in w.calc.items():
            items.append({"price": float(v), "kind": k, "wave": w, "achieved": w.achieved[k]})
    items.sort(key=lambda x: x["price"])
    clusters: list[dict] = []
    for it in items:
        if clusters and abs(it["price"] - clusters[-1]["price"]) / clusters[-1]["price"] <= 0.004:
            cl = clusters[-1]
            cl["items"].append(it)
            cl["price"] = float(np.mean([x["price"] for x in cl["items"]]))
        else:
            clusters.append({"price": it["price"], "items": [it]})
    for cl in clusters:
        cl["n"] = len(cl["items"])
        cl["kinds"] = "/".join(sorted(set(x["kind"] for x in cl["items"])))
        cl["achieved"] = any(x["achieved"] for x in cl["items"])
        cl["pct_from_close"] = (cl["price"] / last_close - 1) * 100
    return clusters


def classify_last_waveform(pivs: list[Pivot]) -> dict | None:
    """直近 4 極（A,B,C,D）で N / P / Y / S / 下げN を判定。"""
    if len(pivs) < 4:
        return None
    A, B, C, D = pivs[-4:]
    if A.kind == "L":   # 上げ基調で見る
        hh, hl = D.price > B.price, C.price > A.price
        if hh and hl:
            form = "N（上げの基本波動）"
        elif (not hh) and hl:
            form = "P（内はらみ・中間波動）"
        elif hh and not hl:
            form = "Y（外はらみ・中間波動）"
        else:
            form = "下げ N（高値・安値切り下げ）" if abs(C.price - A.price) / A.price > 0.01 else "S（前安値で支え・中間波動）"
    else:
        ll, lh = D.price < B.price, C.price < A.price
        if ll and lh:
            form = "N（下げの基本波動）"
        elif (not ll) and lh:
            form = "P（内はらみ・中間波動）"
        elif ll and not lh:
            form = "Y（外はらみ・中間波動）"
        else:
            form = "上げ N（高値・安値切り上げ）" if abs(C.price - A.price) / A.price > 0.01 else "S（前高値で抑え・中間波動）"
    return {"A": A, "B": B, "C": C, "D": D, "form": form,
            "times": (bars_between(A.i, B.i), bars_between(B.i, C.i), bars_between(C.i, D.i))}


def p_value(pivs: list[Pivot]) -> dict | None:
    """直近の内はらみ（高値切り下げ）に対する P 計算値。"""
    highs = [p for p in pivs if p.kind == "H"][-2:]
    lows = [p for p in pivs if p.kind == "L"]
    if len(highs) < 2 or not lows:
        return None
    A, B = highs
    if B.price >= A.price:
        return None
    C = min((p for p in lows if p.i > A.i), key=lambda p: p.price, default=None)
    if C is None:
        return None
    return {"A": A, "B": B, "C": C, "P": C.price + (A.price - B.price),
            "P_down": C.price - (A.price - B.price)}
