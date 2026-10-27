"""東証の制限値幅（ストップ高・ストップ安）判定。

基準値段（前日終値）から値幅を引き、その日の終値が上限／下限に達したかを判定する。
表の出典: https://kabu.com/rule/nehaba.html （全取引所共通）
連続ストップ配分による値幅拡大措置は考慮していない。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# (基準値段の上限（未満）, 値幅)
TABLE: list[tuple[float, float]] = [
    (100, 30), (200, 50), (500, 80), (700, 100), (1_000, 150),
    (1_500, 300), (2_000, 400), (3_000, 500), (5_000, 700),
    (7_000, 1_000), (10_000, 1_500), (15_000, 3_000), (20_000, 4_000),
    (30_000, 5_000), (50_000, 7_000), (70_000, 10_000), (100_000, 15_000),
    (150_000, 30_000), (200_000, 40_000), (300_000, 50_000), (500_000, 70_000),
    (700_000, 100_000), (1_000_000, 150_000), (1_500_000, 300_000),
    (2_000_000, 400_000), (3_000_000, 500_000), (5_000_000, 700_000),
    (7_000_000, 1_000_000), (10_000_000, 1_500_000), (15_000_000, 3_000_000),
    (20_000_000, 4_000_000), (30_000_000, 5_000_000), (50_000_000, 7_000_000),
]
LAST_WIDTH = 10_000_000  # 5,000万円以上

_EDGES = np.array([t[0] for t in TABLE])
_WIDTHS = np.array([t[1] for t in TABLE] + [LAST_WIDTH])


def width(base: float | pd.Series):
    """基準値段に対する制限値幅。Series を渡すと Series で返す。"""
    idx = np.searchsorted(_EDGES, np.asarray(base, dtype=float), side="right")
    w = _WIDTHS[np.clip(idx, 0, len(_WIDTHS) - 1)]
    return pd.Series(w, index=base.index) if isinstance(base, pd.Series) else float(w)


def limit_flags(d: pd.DataFrame, tol: float = 0.995) -> pd.DataFrame:
    """日足から各日のストップ高／ストップ安フラグと、値幅に対する到達率を返す。

    tol: 終値が上限（下限）の tol 倍以上動いていればストップとみなす。
         端数処理や呼値の差を吸収するための許容。
    """
    c = d["Close"]
    prev = c.shift(1)
    w = width(prev)
    chg = c - prev
    up_room = w.where(w > 0)
    reach = chg / up_room                     # +1.0 でストップ高、-1.0 でストップ安
    return pd.DataFrame({
        "prev_close": prev,
        "limit_width": w,
        "limit_pct": w / prev * 100,
        "reach": reach,
        "limit_up": reach >= tol,
        "limit_down": reach <= -tol,
    }, index=d.index)


def streaks(flags: pd.Series) -> list[tuple[int, int]]:
    """True の連続区間を (開始位置, 長さ) のリストで返す。"""
    out: list[tuple[int, int]] = []
    start = None
    vals = flags.to_numpy()
    for i, v in enumerate(vals):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - start))
            start = None
    if start is not None:
        out.append((start, len(vals) - start))
    return out
