"""均衡表（転換線・基準線）とスパン、任意期間の半値線・遅行スパン。

細田：「一目均衡表とは基本数値での半値関係を表す表」。
  転換線 = 過去9日間の半値、基準線 = 過去26日間の半値、
  先行スパン(基準転換) = (基準値+転換値)/2 を26日先行、
  先行スパン(52) = 過去52日間の半値を26日先行（＝76日間の半値関係の先見せ）、
  遅行スパン = 終値を26日遅行。
原著は先行スパンを「上限／下限」と呼び、上げ下げで入れ替わる。本実装では
  span_tk（基準転換）/ span_52 と名付け、どちらが上かは都度判定する。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TENKAN, KIJUN, SENKOU_B, SHIFT = 9, 26, 52, 26


def half_line(df: pd.DataFrame, n: int) -> pd.Series:
    """過去 n 本（当日含む）の高値・安値の半値。"""
    return (df["High"].rolling(n).max() + df["Low"].rolling(n).min()) / 2.0


def lagging_span(df: pd.DataFrame, n: int = SHIFT) -> pd.Series:
    """終値を n 本過去にずらした線（index は元の index のまま、値が前に移動）。"""
    return df["Close"].shift(-n)


def compute(df: pd.DataFrame, extra_half: tuple[int, ...] = (51, 65, 76),
            extra_lag: tuple[int, ...] = (9, 13, 17, 33, 42, 65)) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["tenkan"] = half_line(df, TENKAN)
    out["kijun"] = half_line(df, KIJUN)
    # 先行スパンは 26 本先に描くため、未来分の index を足す
    future_idx = _extend_index(df.index, SHIFT)
    span_tk = ((out["tenkan"] + out["kijun"]) / 2.0)
    span_52 = half_line(df, SENKOU_B)
    out = out.reindex(future_idx)
    out["span_tk"] = span_tk.reindex(future_idx).shift(SHIFT)
    out["span_52"] = span_52.reindex(future_idx).shift(SHIFT)
    out["chikou"] = lagging_span(df, SHIFT).reindex(future_idx)
    for n in extra_half:
        out[f"half_{n}"] = half_line(df, n).reindex(future_idx)
    for n in extra_lag:
        out[f"chikou_{n}"] = lagging_span(df, n).reindex(future_idx)
    return out


def _extend_index(idx: pd.DatetimeIndex, n: int) -> pd.DatetimeIndex:
    """足種を推定して n 本分の将来日付を補う（日足=営業日、週足=7日、月足=月末）。"""
    if len(idx) < 3:
        return idx
    step_min = np.median(np.diff(idx.values).astype("timedelta64[m]").astype(int))
    if step_min < 24 * 60:  # 分足・時間足: 直近に現れた「時刻の並び」を翌営業日以降に繰り返す
        recent = idx[idx >= idx[-1].normalize() - pd.Timedelta(days=7)]
        tod = sorted(set((t.hour, t.minute) for t in recent))
        if not tod:
            tod = [(idx[-1].hour, idx[-1].minute)]
        fut, day, last = [], idx[-1].normalize(), idx[-1]
        while len(fut) < n:
            for h, m in tod:
                t = day + pd.Timedelta(hours=h, minutes=m)
                if t > last:
                    fut.append(t)
                    if len(fut) >= n:
                        break
            day = (day + pd.offsets.BDay(1)).normalize()
        return idx.append(pd.DatetimeIndex(fut))
    step = step_min / (24 * 60)
    if step <= 1.5:
        fut = pd.bdate_range(idx[-1], periods=n + 1)[1:]
    elif step <= 8:
        fut = pd.date_range(idx[-1], periods=n + 1, freq="7D")[1:]
    else:
        fut = pd.date_range(idx[-1], periods=n + 1, freq="ME")[1:]
    return idx.append(fut)


def flat_runs(s: pd.Series, min_len: int = 5, tol: float = 0.0) -> list[dict]:
    """線が値段を変えない区間（横ばい）。細田：横ばいの値段は相場水準になりやすい。"""
    s = s.dropna()
    runs = []
    if s.empty:
        return runs
    vals = s.values
    start = 0
    for i in range(1, len(vals) + 1):
        if i == len(vals) or abs(vals[i] - vals[start]) > tol * max(abs(vals[start]), 1e-9):
            if i - start >= min_len:
                runs.append({"start": s.index[start], "end": s.index[i - 1], "bars": i - start,
                             "price": float(vals[start])})
            start = i
    return runs


@dataclass
class Position:
    tenkan_vs_kijun: str      # 'up' 転換線>基準線 / 'down' / 'cross'
    kijun_slope: str          # 'up' / 'down' / 'flat'
    chikou_vs_price: str      # 遅行スパン vs 26本前の実線
    price_vs_cloud: str       # 実線 vs 先行スパン帯 'above'/'below'/'inside'
    order: str                # 並び順の文字列
    sanyaku: str              # '好転' / '悪化' / '中立'


def position(df: pd.DataFrame, L: pd.DataFrame) -> Position:
    i = df.index[-1]
    c = df["Close"].iloc[-1]
    tk, kj = L.loc[i, "tenkan"], L.loc[i, "kijun"]
    sa, sb = L.loc[i, "span_tk"], L.loc[i, "span_52"]
    top, bot = max(sa, sb), min(sa, sb)
    tvk = "up" if tk > kj else "down" if tk < kj else "cross"
    kj_prev = L["kijun"].loc[:i].dropna()
    slope = "flat"
    if len(kj_prev) >= 2:
        d = kj_prev.iloc[-1] - kj_prev.iloc[-2]
        slope = "up" if d > 0 else "down" if d < 0 else "flat"
    # 遅行スパン: 本日終値 vs 26本前の実線
    if len(df) > SHIFT:
        ref = df.iloc[-1 - SHIFT]
        cvp = "up" if c > ref["High"] else "down" if c < ref["Low"] else "touch"
    else:
        cvp = "n/a"
    pvc = "n/a" if np.isnan(top) or np.isnan(bot) else "above" if c > top else "below" if c < bot else "inside"
    names = {k: v for k, v in {"実線": c, "転換線": tk, "基準線": kj, "先行(基準転換)": sa, "先行(52)": sb}.items() if not np.isnan(v)}
    order = " > ".join(k for k, _ in sorted(names.items(), key=lambda kv: -kv[1]))
    if tvk == "up" and cvp == "up" and pvc == "above":
        san = "好転"
    elif tvk == "down" and cvp == "down" and pvc == "below":
        san = "悪化"
    else:
        san = "中立"
    return Position(tvk, slope, cvp, pvc, order, san)


def support_resistance_events(df: pd.DataFrame, L: pd.DataFrame, lookback: int = 60) -> list[dict]:
    """均衡表各線が「押し目／戻り」として機能した跡（線に触れて反転）と
    「貫いた」跡（相場水準候補）を直近 lookback 本から拾う。"""
    ev = []
    sub = df.iloc[-lookback:]
    for name in ("tenkan", "kijun", "span_tk", "span_52"):
        line = L[name].reindex(sub.index)
        for k in range(1, len(sub) - 1):
            d, v = sub.index[k], line.iloc[k]
            if np.isnan(v):
                continue
            lo, hi, cl = sub["Low"].iloc[k], sub["High"].iloc[k], sub["Close"].iloc[k]
            prev_c, next_c = sub["Close"].iloc[k - 1], sub["Close"].iloc[k + 1]
            if lo <= v <= hi:
                if prev_c > v and cl >= v and next_c > cl:
                    ev.append({"date": d, "line": name, "price": float(v), "kind": "押し目として機能"})
                elif prev_c < v and cl <= v and next_c < cl:
                    ev.append({"date": d, "line": name, "price": float(v), "kind": "戻りとして機能"})
                elif (prev_c - v) * (next_c - v) < 0:
                    ev.append({"date": d, "line": name, "price": float(v), "kind": "貫通（相場水準候補）"})
    return ev
