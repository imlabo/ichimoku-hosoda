"""データ取得・読み込み。

株式コード → yfinance（お手元の環境で動作。4桁数字は東証 .T を補う）
CSV → Date,Open,High,Low,Close[,Volume] 列を持つファイル
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ["Open", "High", "Low", "Close"]


def normalize_symbol(code: str) -> str:
    """'7203' → '7203.T'、'N225'/'日経平均' → '^N225'、'USDJPY' → 'USDJPY=X'。"""
    c = code.strip()
    if c in ("日経平均", "N225", "n225", "nikkei", "NIKKEI"):
        return "^N225"
    if re.fullmatch(r"\d{4}", c):
        return f"{c}.T"
    if re.fullmatch(r"[A-Za-z]{6}", c) and c.upper() in {"USDJPY", "EURUSD", "EURJPY", "GBPJPY", "AUDJPY"}:
        return f"{c.upper()}=X"
    return c


def fetch(code: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """OHLC を取得。yfinance → 失敗時は stooq の順に試す。"""
    sym = normalize_symbol(code)
    errors = []
    try:
        return _fetch_yf(sym, period, interval)
    except Exception as e:  # noqa: BLE001
        errors.append(f"yfinance: {e}")
    try:
        return _fetch_stooq(sym, period)
    except Exception as e:  # noqa: BLE001
        errors.append(f"stooq: {e}")
    raise RuntimeError(f"{sym}: データを取得できませんでした（" + " / ".join(errors) + "）")


def _fetch_yf(sym: str, period: str, interval: str) -> pd.DataFrame:
    import yfinance as yf  # 遅延 import

    df = yf.download(sym, period=period, interval=interval, progress=False, auto_adjust=False, threads=False)
    if df is None or len(df) == 0:
        raise RuntimeError("空の応答")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df[REQUIRED + (["Volume"] if "Volume" in df.columns else [])].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    return _clean(df)


def _stooq_symbol(sym: str) -> str:
    if sym.endswith(".T"):
        return sym[:-2] + ".jp"
    if sym == "^N225":
        return "^nkx"
    if sym.endswith("=X"):
        return sym[:-2].lower()
    return sym.lower()


def _fetch_stooq(sym: str, period: str) -> pd.DataFrame:
    import urllib.request

    url = f"https://stooq.com/q/d/l/?s={_stooq_symbol(sym)}&i=d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        raw = r.read().decode("utf-8", "replace")
    if "No data" in raw or raw.count("\n") < 30:
        raise RuntimeError("データなし")
    import io as _io
    df = pd.read_csv(_io.StringIO(raw))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")[REQUIRED + (["Volume"] if "Volume" in df.columns else [])]
    years = {"1y": 1, "2y": 2, "3y": 3, "5y": 5, "6y": 6, "10y": 10}.get(period)
    if years:
        df = df[df.index >= df.index[-1] - pd.DateOffset(years=years)]
    return _clean(df)


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date") or cols.get("日付") or df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    df.index.name = "Date"
    rename = {}
    for want in REQUIRED + ["Volume"]:
        for c in df.columns:
            if c.lower() == want.lower():
                rename[c] = want
    jp = {"始値": "Open", "高値": "High", "安値": "Low", "終値": "Close", "出来高": "Volume"}
    for c in df.columns:
        if c in jp:
            rename[c] = jp[c]
    df = df.rename(columns=rename)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"CSV に列が足りません: {missing}")
    return _clean(df)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=REQUIRED)
    for c in REQUIRED:
        df[c] = df[c].astype(float)
    return df


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """日足 → 週足('W-FRI') / 月足('ME')。足の日付は期間末日。"""
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in df.columns:
        agg["Volume"] = "sum"
    out = df.resample(rule).agg(agg).dropna(subset=["Close"])
    # 期間内に実データのある最終日を足の日付にする
    last_dates = df["Close"].resample(rule).apply(lambda s: s.index[-1] if len(s) else pd.NaT)
    out.index = pd.DatetimeIndex(last_dates.dropna().values)
    out.index.name = "Date"
    return out


def synthetic(n: int = 1300, seed: int = 7, start: str = "2021-01-04", s0: float = 2500.0) -> pd.DataFrame:
    """動作確認用の擬似株価（三波動っぽいレジームを含むランダムウォーク）。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    drift = np.zeros(n)
    # 上げ・押し・上げ・もみ合い・下げ … のレジームを基本数値っぽい長さで配置
    segs = [(76, 0.0022), (26, -0.0025), (65, 0.0018), (51, 0.0), (42, -0.003), (33, 0.0015),
            (101, 0.0012), (26, -0.002), (83, 0.001), (76, -0.0015), (129, 0.0014), (65, 0.0),
            (42, -0.0022), (76, 0.002), (300, 0.0005)]
    i = 0
    for ln, d in segs:
        drift[i:i + ln] = d
        i += ln
        if i >= n:
            break
    rets = drift + rng.normal(0, 0.011, n)
    close = s0 * np.exp(np.cumsum(rets))
    opn = close * (1 + rng.normal(0, 0.004, n))
    hi = np.maximum(close, opn) * (1 + np.abs(rng.normal(0, 0.006, n)))
    lo = np.minimum(close, opn) * (1 - np.abs(rng.normal(0, 0.006, n)))
    df = pd.DataFrame({"Open": opn, "High": hi, "Low": lo, "Close": close,
                       "Volume": rng.integers(1e5, 5e6, n)}, index=dates)
    df.index.name = "Date"
    return df.round(1)
