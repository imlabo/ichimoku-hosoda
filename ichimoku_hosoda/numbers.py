"""基本数値・対等数値・変化日の生成。

基本数値：9 と 26 を絶対数とし、その組み合わせ（接点重複は −1/−2）。
併せて重視: 47, 67, 13, 37, 55, 21, 小基本数値 3・5・7, 100+基本数値,
一環 226 のほか経験上 247・257、一巡環 676。
数え方：起点日を 1 日目とする（起点日を含む営業日数）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

CORE = [9, 17, 26, 33, 42, 51, 65, 76, 83, 97, 101, 129, 172, 200, 226, 257]
SUB = [47, 67, 13, 37, 55, 21, 133, 142, 151, 165, 183, 247, 676]
PLUS100 = [100 + n for n in (9, 17, 26, 33, 42, 47, 51, 65, 67, 76, 83, 97)]
SMALL = [3, 5, 7]

def _dt(ts) -> str:
    """日付表示。分足・時間足（時刻あり）は「MM-DD HH:MM」、日足以上は「YYYY-MM-DD」。"""
    t = pd.Timestamp(ts)
    return t.strftime("%m-%d %H:%M") if (t.hour or t.minute) else t.strftime("%Y-%m-%d")


NAMES = {9: "一節", 26: "一期", 76: "一巡", 226: "一環", 676: "一巡環"}


def basic_numbers(max_n: int = 700, include_small: bool = False) -> dict[int, str]:
    """{数値: 区分} を返す。core=原著の基本数値, sub=併せて重視, plus100=100+基本数値。"""
    d: dict[int, str] = {}
    for n in CORE:
        d[n] = "core"
    for n in SUB:
        d.setdefault(n, "sub")
    for n in PLUS100:
        d.setdefault(n, "plus100")
    if include_small:
        for n in SMALL:
            d.setdefault(n, "small")
    return {k: v for k, v in sorted(d.items()) if k <= max_n}


def tolerance(n: int) -> int:
    """9 なら 7〜11、17 なら 13〜21 と幅を持たせる。ここでは控えめに ±1（〜50）、±2（〜150）、±3。"""
    return 1 if n <= 50 else 2 if n <= 150 else 3


def nearest_basic(n: int, table: dict[int, str] | None = None) -> tuple[int, str] | None:
    table = table or basic_numbers()
    best = None
    for b, cat in table.items():
        if abs(n - b) <= tolerance(b):
            if best is None or abs(n - b) < abs(n - best[0]):
                best = (b, cat)
    return best


@dataclass
class ChangeDate:
    """変化日候補。"""
    i: int                       # バー位置（将来は len(df) 以上）
    date: pd.Timestamp
    kind: str                    # 'basic' / 'equal' / 'wave' / 'level'
    n: int                       # 日数
    origin: pd.Timestamp         # 起点
    origin_kind: str             # 'H'/'L'/'level'
    detail: str = ""
    weight: float = 1.0
    sources: list = field(default_factory=list)


def future_dates(index: pd.DatetimeIndex, horizon: int) -> pd.DatetimeIndex:
    """index の末尾から horizon 本分の将来日付（日足は営業日近似。日本の祝日は未考慮）。"""
    from .lines import _extend_index
    return _extend_index(index, horizon)


def basic_change_dates(pivs, index_ext: pd.DatetimeIndex, n_bars: int, horizon: int,
                       table: dict[int, str] | None = None, unit: str = "日") -> list[ChangeDate]:
    """各極（起点）からの基本数値日を、現在〜horizon 本先まで列挙。"""
    table = table or basic_numbers()
    out = []
    for p in pivs:
        for b, cat in table.items():
            j = p.i + b - 1
            if n_bars - 3 <= j < n_bars + horizon:
                w = {"core": 1.0, "sub": 0.6, "plus100": 0.5, "small": 0.3}[cat] * (0.6 + 0.2 * p.rank)
                out.append(ChangeDate(j, index_ext[j], "basic", b, p.date, p.kind,
                                      f"{'高値' if p.kind=='H' else '安値'}{_dt(p.date)} から {b}{unit}目{'（'+NAMES[b]+'）' if b in NAMES else ''}", w))
    return out


def equal_change_dates(pivs, index_ext: pd.DatetimeIndex, n_bars: int, horizon: int,
                       max_pairs: int = 6, unit: str = "日") -> list[ChangeDate]:
    """対等数値：直近の各極を中心点に、その前の騰落（極→極）の日数を取る。"""
    out = []
    recent = pivs[-max_pairs - 1:]
    for k in range(1, len(recent)):
        center = recent[k]
        for m in range(k):
            prev = recent[m]
            n = center.i - prev.i + 1
            j = center.i + n - 1
            if n >= 3 and n_bars - 3 <= j < n_bars + horizon:
                out.append(ChangeDate(j, index_ext[j], "equal", n, center.date, center.kind,
                                      f"対等数値: {_k(prev)}{_dt(prev.date)}→{_k(center)}{_dt(center.date)} の {n}{unit}に対し "
                                      f"{_k(center)}{_dt(center.date)} から {n}{unit}目", 0.8 + 0.1 * center.rank))
    return out


def cluster_dates(cands: list[ChangeDate]) -> list[dict]:
    """同じ日の候補をまとめ、重なり度（スコア）で並べる。隣接日（±1）は
    連鎖させず、スコアの 1/2 を参考加点するだけにとどめる（1 日違いで重なる日を重要視）。"""
    by_i: dict[int, list[ChangeDate]] = {}
    for c in cands:
        by_i.setdefault(c.i, []).append(c)
    clusters: list[dict] = []
    for i, items in by_i.items():
        # 同一起点・同一種類の重複（基本数値と対等数値が同じ数になる等）は 1 件に数える
        seen, uniq = set(), []
        for c in sorted(items, key=lambda x: -x.weight):
            key = (c.kind, c.origin, c.n)
            if key not in seen:
                seen.add(key)
                uniq.append(c)
        own = sum(c.weight for c in uniq)
        clusters.append({"i_min": i, "i_max": i, "items": uniq, "own_score": own})
    for cl in clusters:
        neigh = sum(o["own_score"] for o in clusters if abs(o["i_min"] - cl["i_min"]) == 1)
        cl["score"] = cl["own_score"] + 0.5 * neigh
        cl["date"] = cl["items"][0].date
        cl["kinds"] = sorted(set(x.kind for x in cl["items"]))
        cl["n_sources"] = len(cl["items"])
        cl["items"].sort(key=lambda x: -x.weight)
    return sorted(clusters, key=lambda c: -c["score"])


def _k(p) -> str:
    return "高値" if p.kind == "H" else "安値"
