"""高値・安値（極）の抽出。

細田：押し戻りの限界を小さく設定するほど多くの波形が確認できる（カギ足）。
複数の反転幅でジグザグを取り、大きな幅でも生き残る極ほど rank を高くする。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Pivot:
    i: int                 # バー位置
    date: pd.Timestamp
    price: float
    kind: str              # 'H' / 'L'
    rank: int              # 何段階目の閾値まで生き残ったか（大きいほど重要）


def zigzag(df: pd.DataFrame, pct: float) -> list[tuple[int, str, float]]:
    """高値は High、安値は Low を使う単純ジグザグ。pct は反転率(0.05=5%)。"""
    hi, lo = df["High"].values, df["Low"].values
    n = len(df)
    if n < 3:
        return []
    piv: list[tuple[int, str, float]] = []
    # 初期方向: 最初の数本で決める
    trend = 1 if hi[min(5, n - 1)] >= hi[0] else -1
    ext_i, ext_p = (0, hi[0]) if trend == 1 else (0, lo[0])
    for k in range(1, n):
        if trend == 1:
            if hi[k] >= ext_p:
                ext_i, ext_p = k, hi[k]
            elif lo[k] <= ext_p * (1 - pct):
                piv.append((ext_i, "H", float(ext_p)))
                trend, ext_i, ext_p = -1, k, lo[k]
        else:
            if lo[k] <= ext_p:
                ext_i, ext_p = k, lo[k]
            elif hi[k] >= ext_p * (1 + pct):
                piv.append((ext_i, "L", float(ext_p)))
                trend, ext_i, ext_p = 1, k, hi[k]
    piv.append((ext_i, "H" if trend == 1 else "L", float(ext_p)))  # 進行中の極（暫定）
    return piv


def pivots(df: pd.DataFrame, thresholds: tuple[float, ...] = (0.03, 0.06, 0.12, 0.25)) -> list[Pivot]:
    """複数閾値で抽出し rank 付け。最小閾値で得た極を基準に、より大きい閾値でも
    同じ位置（±2本）に極があれば rank を加算。"""
    base = zigzag(df, thresholds[0])
    higher = [set(i for i, _, _ in zigzag(df, t)) for t in thresholds[1:]]
    out = []
    for i, kind, price in base:
        rank = 1
        for s in higher:
            if any(abs(i - j) <= 2 for j in s):
                rank += 1
            else:
                break
        out.append(Pivot(i, df.index[i], price, kind, rank))
    return out


def bars_between(i0: int, i1: int) -> int:
    """起点日を含む営業日数（細田の数え方: 5/27起点→6/18 で 17 日間）。"""
    return i1 - i0 + 1
