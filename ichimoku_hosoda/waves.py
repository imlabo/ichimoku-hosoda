"""三波動構成（時間関係・計算値）と波形分類。

定型: A→B 第一波動、B→C 第二波動（押し/戻り）、C→D 第三波動
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
    tag: str = ""          # 定型（期間の最高値・最安値から機械的に決めた三波動）なら説明文

    def ad_basic(self, i_d: int):
        return nearest_basic(bars_between(self.A.i, i_d))


def _origin_candidates(pivs: list[Pivot], ci: int, max_back: int, start: int, majors: list[int], ext: dict) -> list[int]:
    """C（pivs[ci]）に対する起点 A の候補 index。
    ① 直近の同種の極（max_back 個）  ② 評価範囲内の重要な極（rank>=3、直近 12 個まで）
    ③ 評価範囲内の最安値／最高値の極  — 起点と中心点を変えつつ複数の三波動を確認する。"""
    C = pivs[ci]
    cand = set()
    for ai in range(ci - 2, max(-1, ci - 2 - 2 * max_back), -2):
        cand.add(ai)
    cand.update(ai for ai in majors if ai < ci and pivs[ai].kind == C.kind)
    e = ext.get(C.kind)
    if e is not None and e < ci:
        cand.add(e)
    return sorted(cand, reverse=True)


def _mk_wave(A: Pivot, B: Pivot, C: Pivot, direction: str, df: pd.DataFrame) -> Wave3:
    hi, lo = df["High"].values, df["Low"].values
    ab, bc, ac = bars_between(A.i, B.i), bars_between(B.i, C.i), bars_between(A.i, C.i)
    w = Wave3(A, B, C, direction, ab, bc, ac)
    w.d_dates = {"AB=BD": B.i + ab - 1, "AB=CD": C.i + ab - 1, "AC=CD": C.i + ac - 1}
    a, b, c = A.price, B.price, C.price
    if direction == "up":
        w.calc = {"E": b + (b - a), "N": c + (b - a), "V": b + (b - c), "NT": c + (c - a)}
        w.extreme_since_c = float(hi[C.i:].max())
        w.achieved = {k: w.extreme_since_c >= v for k, v in w.calc.items()}
        w.note = "N候補（押し目が起点より上）"
    else:
        w.calc = {"E": b - (a - b), "N": c - (a - b), "V": b - (c - b), "NT": c - (a - c)}
        w.extreme_since_c = float(lo[C.i:].min())
        w.achieved = {k: w.extreme_since_c <= v for k, v in w.calc.items()}
        w.note = "N候補（戻りが起点より下）"
    return w


def rule_waves(df: pd.DataFrame, pivs: list[Pivot], windows: tuple[int | None, ...] = (None, 250, 120)) -> list[Wave3]:
    """定型の三波動（ユーザー定義のシンプルな決め方）。
    下げ: A = 期間の最高値、B = A 以降の最安値、C = B 以降の最高値
    上げ: A = 期間の最安値、B = A 以降の最高値、C = B 以降の最安値
    B が直近（C の余地がない＝第一波動が進行中）のときは、A 以降で確定した最大の山（谷）を B とし、
    その後の最安値（最高値）を C とする内側の三波動も出す。"""
    hi, lo = df["High"].values, df["Low"].values
    n = len(df)
    rank_at = {p.i: p.rank for p in pivs}
    confirmed = {p.i for p in pivs[:-1]}            # 最後の極は進行中
    out: list[Wave3] = []

    def piv(i: int, kind: str) -> Pivot:
        return Pivot(i, df.index[i], float(hi[i] if kind == "H" else lo[i]), kind, max(rank_at.get(i, 0), 3))

    for win in windows:
        if win is not None and win >= n:
            continue
        s0 = 0 if win is None else n - win
        label = "表示期間全体" if win is None else f"直近{win}本"
        for direction in ("down", "up"):
            if direction == "down":
                ai = s0 + int(np.argmax(hi[s0:]))
                if ai >= n - 3:
                    continue
                bi = ai + int(np.argmin(lo[ai:]))
                kA, kB = "H", "L"
            else:
                ai = s0 + int(np.argmin(lo[s0:]))
                if ai >= n - 3:
                    continue
                bi = ai + int(np.argmax(hi[ai:]))
                kA, kB = "L", "H"
            tag = f"定型（{label}）"
            ci_forced = None
            if bi >= n - 2:
                # B が直近で更新中（＝第三波動が進行中）→ A 以降の山（谷）のうち、
                # その後の押し（戻り）が最も深かったものを B、その押しの底（戻りの天井）を C とする
                best = None
                for p in pivs:
                    if p.kind != kB or not (ai < p.i < n - 2) or p.i not in confirmed:
                        continue
                    if kB == "H":
                        brk = next((k for k in range(p.i + 1, n) if hi[k] > p.price), n)
                        seg_lo = p.i + 1 + int(np.argmin(lo[p.i + 1:brk])) if brk > p.i + 1 else None
                        if seg_lo is None:
                            continue
                        depth = (p.price - lo[seg_lo]) / max(p.price - lo[ai], 1e-9)
                    else:
                        brk = next((k for k in range(p.i + 1, n) if lo[k] < p.price), n)
                        seg_lo = p.i + 1 + int(np.argmax(hi[p.i + 1:brk])) if brk > p.i + 1 else None
                        if seg_lo is None:
                            continue
                        depth = (hi[seg_lo] - p.price) / max(hi[ai] - p.price, 1e-9)
                    if depth >= 0.2 and (best is None or depth > best[0] + 1e-9):
                        best = (depth, p.i, seg_lo)
                if best is None:
                    continue
                _, bi, ci_forced = best
                tag = f"定型（{label}・B は最も深い押し{'' if kB == 'H' else '（戻り）'}の前の{'山' if kB == 'H' else '谷'}／現在は第三波動進行中）"
            if bi - ai < 2 or bi >= n - 1:
                continue
            ci = ci_forced if ci_forced is not None else \
                bi + 1 + (int(np.argmax(hi[bi + 1:])) if direction == "down" else int(np.argmin(lo[bi + 1:])))
            A, B, C = piv(ai, kA), piv(bi, kB), piv(ci, kA)
            if (direction == "down" and C.price >= A.price) or (direction == "up" and C.price <= A.price):
                continue
            w = _mk_wave(A, B, C, direction, df)
            w.tag = tag + ("（C は形成中）" if ci >= n - 2 else "")
            out.append(w)
    # 重複除去（同じ A・B・C）
    uniq, seen = [], set()
    # 表示期間全体 → 直近250本 → 直近120本 の順、同じ期間内では起点 A が新しい順
    order = lambda w: (0 if "表示期間全体" in w.tag else 1 if "250" in w.tag else 2, -w.A.i)
    for w in sorted(out, key=order):
        k = (w.A.i, w.B.i, w.C.i)
        if k not in seen:
            seen.add(k)
            uniq.append(w)
    return uniq


def build_waves(pivs: list[Pivot], df: pd.DataFrame, max_recent: int = 20, max_back: int = 4,
                force_a: int | None = None, lookback: int = 400) -> list[Wave3]:
    """C（第二波動の終点）候補 = 直近 max_recent 個の極＋評価範囲内の重要極。
    起点 A = _origin_candidates。force_a を与えるとその極を起点にした三波動のみ。"""
    out: list[Wave3] = []
    n = len(pivs)
    if n < 3:
        return out
    hi, lo = df["High"].values, df["Low"].values
    n_bars = len(df)
    start = max(0, n_bars - lookback)
    in_win = [k for k in range(n) if pivs[k].i >= start]
    majors = [k for k in in_win if pivs[k].rank >= 3][-24:]
    lows = [k for k in in_win if pivs[k].kind == "L"]
    highs = [k for k in in_win if pivs[k].kind == "H"]
    ext = {"L": min(lows, key=lambda k: pivs[k].price) if lows else None,
           "H": max(highs, key=lambda k: pivs[k].price) if highs else None}
    c_idx = sorted(set(range(max(2, n - max_recent), n)) | {k for k in majors if k >= 2})
    for ci in c_idx:
        C = pivs[ci]
        a_list = [force_a] if force_a is not None else _origin_candidates(pivs, ci, max_back, start, majors, ext)
        for ai in a_list:
            if ai is None or ai >= ci or ai < 0:
                continue
            A = pivs[ai]
            if A.kind != C.kind:
                continue
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
                    break
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

    # 同じ A・C の組は 1 つに
    uniq, seen = [], set()
    for w in out:
        k = (w.A.i, w.C.i)
        if k not in seen:
            seen.add(k)
            uniq.append(w)

    # B 以降の「最大の戻り高値／最深の押し安値」を事前計算（後ろからの累積）
    nat_after_h: dict[int, int] = {}
    nat_after_l: dict[int, int] = {}
    best_h = best_l = None
    pos_list = sorted(range(n), key=lambda k: pivs[k].i, reverse=True)
    for k in pos_list:
        p = pivs[k]
        nat_after_h[p.i] = best_h
        nat_after_l[p.i] = best_l
        if p.kind == "H" and (best_h is None or p.price > pivs[best_h].price):
            best_h = k
        if p.kind == "L" and (best_l is None or p.price < pivs[best_l].price):
            best_l = k

    def score(w: Wave3) -> float:
        sc = 2.5 * w.A.rank + 1.5 * w.C.rank
        if (w.direction == "up" and ext["L"] is not None and w.A.i == pivs[ext["L"]].i) or \
           (w.direction == "down" and ext["H"] is not None and w.A.i == pivs[ext["H"]].i):
            sc += 4            # 期間の最安値（最高値）を起点とする三波動
        if w.C.i >= n_bars - 60:
            sc += 3            # C が直近
        nat = nat_after_h.get(w.B.i) if w.direction == "down" else nat_after_l.get(w.B.i)
        if nat is not None and pivs[nat].i == w.C.i:
            sc += 3            # C が B 以降の最大の戻り（最深の押し）＝素直な第二波動
        if w.A.i < start:
            sc -= 3
        if abs(w.B.price - w.A.price) >= abs(w.C.price - w.B.price):
            sc += 2
        if w.ab / max(w.ac, 1) < 0.25:
            sc -= 3
        if w.bc > 2 * w.ab:
            sc -= 2
        return sc

    scored = sorted(((score(w), w) for w in uniq), key=lambda t: -t[0])
    per_a: dict[int, int] = {}
    picked = []
    for _, w in scored:
        if force_a is None and per_a.get(w.A.i, 0) >= 3:
            continue
        per_a[w.A.i] = per_a.get(w.A.i, 0) + 1
        picked.append(w)
        if len(picked) >= 14:
            break
    return picked


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
                                      f"{'【定型】' if w.tag else ''}{'上げ' if w.direction=='up' else '下げ'}三波動 A{w.A.date.date()} B{w.B.date.date()} "
                                      f"C{w.C.date.date()} の {rel}（{w.ab if rel!='AC=CD' else w.ac}{unit}）{extra}",
                                      (1.2 + 0.15 * (w.A.rank + w.C.rank) + (0.5 if nb else 0)) * (2.0 if w.tag else 1.0)))
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


def d_origin_change_dates(waves: list[Wave3], index_ext: pd.DatetimeIndex, n_bars: int, horizon: int,
                          unit: str = "日", rel: str = "AC=CD", prefix: str = "", mult: float = 1.0) -> list[ChangeDate]:
    """三波動の終了予測日 D（既定 AC=CD）を起点に、そこからの基本数値日を変化日候補にする（D を 1 本目と数える）。"""
    from .numbers import NAMES, basic_numbers
    out = []
    for w in waves:
        j0 = w.d_dates.get(rel)
        if j0 is None or j0 >= len(index_ext):
            continue
        for b, cat in basic_numbers().items():
            j = j0 + b - 1
            if j >= len(index_ext) or not (n_bars - 3 <= j < n_bars + horizon):
                continue
            wt = {"core": 1.0, "sub": 0.6, "plus100": 0.5, "small": 0.3}[cat] * 1.3 * mult
            out.append(ChangeDate(j, index_ext[j], "dbasic", b, index_ext[j0], "D",
                                  f"{prefix}{'上げ' if w.direction == 'up' else '下げ'}三波動の {rel}（{index_ext[j0].date()}）から {b}{unit}目"
                                  f"{'（' + NAMES[b] + '）' if b in NAMES else ''}", wt))
    return out
