"""基本数値足（3日足・9週足・26週足・9ヶ月足・26ヶ月足）。

「当日終値と三本前の寄り付きで作成した、ローソク足の陰陽」(3日足)
      「当週の終値と 9 週前の始値で作る」(9週足) ＝ 9 の遅行スパンとほぼ同じ考え方
  ・陰連陽連の本数は基本数値・対等数値に律せられやすい
  ・9週足は 9 本以上の連続で「上げ時代／下げ時代」、3日足は 9 以上まれで軸は 5〜6
  ・始値は既知なので、今後 n−1 本分「いくらで引ければ陽線／陰線か」が事前に分かる
include_current=True: 当日を含めて n 本（t−(n−1) の始値）／False: n 本前（t−n の始値）
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def nbar(df: pd.DataFrame, n: int, include_current: bool = True) -> pd.DataFrame:
    k = n - 1 if include_current else n
    o = df["Open"].shift(k)
    out = pd.DataFrame({"ref_open": o, "close": df["Close"]}, index=df.index)
    out["sign"] = np.sign(out["close"] - out["ref_open"])
    out["sign"] = out["sign"].where(out["ref_open"].notna())
    return out


def runs(sign: pd.Series) -> list[dict]:
    s = sign.dropna()
    res = []
    if s.empty:
        return res
    cur, start = s.iloc[0], 0
    vals = s.values
    for i in range(1, len(vals) + 1):
        if i == len(vals) or vals[i] != cur or vals[i] == 0:
            if cur != 0:
                res.append({"sign": "陽" if cur > 0 else "陰", "len": i - start,
                            "start": s.index[start], "end": s.index[i - 1]})
            if i < len(vals):
                cur, start = vals[i], i
    return res


def summarize(df: pd.DataFrame, n: int, label: str, include_current: bool = True, future: int | None = None) -> dict:
    nb = nbar(df, n, include_current)
    rs = runs(nb["sign"])
    cur = rs[-1] if rs else None
    prev_same = [r for r in rs[:-1] if cur and r["sign"] != cur["sign"]][-1:] if cur else []
    k = n - 1 if include_current else n
    fut = []
    horizon = future if future is not None else k
    for step in range(1, min(horizon, k) + 1):
        # step 本後の足が比較する始値 = 当日から (k − step) 本前の始値
        j = len(df) - 1 - (k - step)
        if 0 <= j < len(df):
            fut.append({"bars_ahead": step, "ref_open": float(df["Open"].iloc[j]), "ref_date": df.index[j]})
    hist = [{"sign": r["sign"], "len": r["len"]} for r in rs[-8:]]
    return {"label": label, "n": n, "current": cur, "previous_opposite": prev_same[0] if prev_same else None,
            "history": hist, "future_thresholds": fut, "last_close": float(df["Close"].iloc[-1])}
