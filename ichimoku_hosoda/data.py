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
    # 東証コード: 4桁数字、または 285A のような英字入り新形式（数字3桁＋英数字、など）
    if re.fullmatch(r"[0-9][0-9A-Za-z]{3}", c) and not c.isalpha():
        return f"{c.upper()}.T"
    if re.fullmatch(r"[0-9][0-9A-Za-z]{3}\.T", c, flags=re.I):
        return c.upper()
    if re.fullmatch(r"[A-Za-z]{6}", c) and c.upper() in {"USDJPY", "EURUSD", "EURJPY", "GBPJPY", "AUDJPY", "USDVND"}:
        return f"{c.upper()}=X"
    # ベトナム株: "VN:FPT" / "FPT.VN"、指数 VNINDEX
    m = re.fullmatch(r"(?i)vn[:：]\s*([A-Za-z0-9]{1,10})", c)
    if m:
        c = m.group(1).upper() + ".VN"
    if c.upper() in ("VNINDEX", "VN-INDEX", "VNINDEX.VN", "^VNINDEX"):
        return "^VNINDEX.VN"
    if re.fullmatch(r"(?i)[A-Z0-9]{1,10}\.VN", c):
        return c.upper()
    # 米国株: 英字ティッカー（BRK.B / BRK-B も可）
    if re.fullmatch(r"[A-Za-z]{1,5}([.\-][A-Za-z])?", c):
        return c.upper().replace(".", "-")
    return c


def market_of(sym: str) -> str:
    """'jp' / 'us' / 'vn' / 'other'（為替・指数など）。"""
    s = sym.upper()
    if s.endswith(".T") or s == "^N225":
        return "jp"
    if s.endswith(".VN"):
        return "vn"
    if s.endswith("=X"):
        return "other"
    if re.fullmatch(r"\^?[A-Z][A-Z\-]{0,6}", s):
        return "us"
    return "other"


MARKET_TZ = {"jp": "Asia/Tokyo", "vn": "Asia/Ho_Chi_Minh", "us": "America/New_York", "other": "Asia/Tokyo"}


def fetch(code: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """OHLC を取得。yfinance → 失敗時は stooq の順に試す。"""
    sym = normalize_symbol(code)
    errors = []
    try:
        return _fetch_yf(sym, period, interval)
    except Exception as e:  # noqa: BLE001
        errors.append(f"yfinance: {e}")
    if market_of(sym) == "vn":
        try:
            return _fetch_vndirect(sym, period)
        except Exception as e:  # noqa: BLE001
            errors.append(f"VNDirect: {e}")
        raise RuntimeError(f"{sym}: データを取得できませんでした（" + " / ".join(errors) + "）")
    try:
        return _fetch_stooq(sym, period)
    except Exception as e:  # noqa: BLE001
        errors.append(f"stooq: {e}")
    raise RuntimeError(f"{sym}: データを取得できませんでした（" + " / ".join(errors) + "）")


INTRADAY_PERIOD = {"1m": "7d", "5m": "59d", "15m": "59d", "30m": "59d", "1h": "729d"}


def fetch_intraday(code: str, interval: str = "5m") -> pd.DataFrame:
    """分足・時間足（yfinance のみ。1m は直近7日、5m は直近約60日、1h は約2年が上限）。時刻は日本時間に変換。"""
    sym = normalize_symbol(code)
    df = _fetch_yf(sym, INTRADAY_PERIOD.get(interval, "59d"), interval, keep_tz=MARKET_TZ.get(market_of(sym), "Asia/Tokyo"))
    return df


def _fetch_yf(sym: str, period: str, interval: str, keep_tz: bool | str = False) -> pd.DataFrame:
    import yfinance as yf  # 遅延 import

    df = yf.download(sym, period=period, interval=interval, progress=False, auto_adjust=False, threads=False)
    if df is None or len(df) == 0:
        raise RuntimeError("空の応答")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df[REQUIRED + (["Volume"] if "Volume" in df.columns else [])].copy()
    idx = pd.to_datetime(df.index)
    if keep_tz and idx.tz is not None:
        idx = idx.tz_convert(keep_tz if isinstance(keep_tz, str) else "Asia/Tokyo").tz_localize(None)   # 市場の現地時刻
    elif idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx
    df.index.name = "Date"
    return _clean(df)


def _fetch_vndirect(sym: str, period: str) -> pd.DataFrame:
    """ベトナム株の予備: VNDirect の公開 API（株価は千ドン単位で返るので 1000 倍してドンに揃える）。"""
    import json
    import urllib.request
    code = "VNINDEX" if sym.upper().startswith("^VNINDEX") else sym.upper().replace(".VN", "")
    years = {"6mo": 1, "1y": 1, "2y": 2, "3y": 3, "5y": 5, "6y": 6, "10y": 10}.get(period, 6)
    since = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    url = (f"https://finfo-api.vndirect.com.vn/v4/stock_prices?sort=date&q=code:{code}~date:gte:{since}"
           f"&size=3000&page=1")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        js = json.loads(r.read().decode("utf-8", "replace"))
    rows = js.get("data") or []
    if len(rows) < 30:
        raise RuntimeError("データなし")
    df = pd.DataFrame(rows)
    mult = 1.0 if code == "VNINDEX" else 1000.0
    pick = lambda a, b: df[a] if a in df.columns else df[b]
    out = pd.DataFrame({"Open": pick("adOpen", "open").astype(float) * mult, "High": pick("adHigh", "high").astype(float) * mult,
                        "Low": pick("adLow", "low").astype(float) * mult, "Close": pick("adClose", "close").astype(float) * mult,
                        "Volume": pick("nmVolume", "volume").astype(float)},
                       index=pd.to_datetime(df["date"]))
    out.index.name = "Date"
    return _clean(out)


def _stooq_symbol(sym: str) -> str:
    if sym.endswith(".T"):
        return sym[:-2] + ".jp"
    if sym == "^N225":
        return "^nkx"
    if sym.endswith("=X"):
        return sym[:-2].lower()
    if market_of(sym) == "us" and not sym.startswith("^"):
        return sym.lower() + ".us"
    return sym.lower()


def _fetch_stooq(sym: str, period: str) -> pd.DataFrame:
    import urllib.request

    url = f"https://stooq.com/q/d/l/?s={_stooq_symbol(sym)}&i=d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=40) as r:
        raw = r.read().decode("utf-8", "replace")
    if "No data" in raw or raw.count("\n") < 30:
        raise RuntimeError("データなし")
    import io as _io
    df = pd.read_csv(_io.StringIO(raw))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")[REQUIRED + (["Volume"] if "Volume" in df.columns else [])]
    years = {"1y": 1, "2y": 2, "3y": 3, "5y": 5, "6y": 6, "10y": 10}.get(period)
    if period == "6mo":
        df = df[df.index >= df.index[-1] - pd.DateOffset(months=6)]
    elif years:
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


def synthetic_intraday(interval: str = "5m", seed: int = 11) -> pd.DataFrame:
    """動作確認用の擬似分足（東証の立会時間 9:00-11:30 / 12:30-15:30 を模す）。"""
    rng = np.random.default_rng(seed)
    step = {"1m": 1, "5m": 5}.get(interval, 60)
    days = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods={1: 5, 5: 40}.get(step, 300))
    times = []
    for d in days:
        for h, m in [(9, 0), (11, 30), (12, 30), (15, 30)][:0]:
            pass
        t = d + pd.Timedelta(hours=9)
        while t < d + pd.Timedelta(hours=15, minutes=30):
            if not (d + pd.Timedelta(hours=11, minutes=30) <= t < d + pd.Timedelta(hours=12, minutes=30)):
                times.append(t)
            t += pd.Timedelta(minutes=step)
    n = len(times)
    rets = rng.normal(0, {1: 0.0007, 5: 0.0015}.get(step, 0.004), n) + np.sin(np.arange(n) / {1: 120, 5: 76}.get(step, 26)) * 0.0006
    close = 1800 * np.exp(np.cumsum(rets))
    opn = np.r_[close[0], close[:-1]]
    hi = np.maximum(close, opn) * (1 + np.abs(rng.normal(0, 0.0008, n)))
    lo = np.minimum(close, opn) * (1 - np.abs(rng.normal(0, 0.0008, n)))
    df = pd.DataFrame({"Open": opn, "High": hi, "Low": lo, "Close": close}, index=pd.DatetimeIndex(times, name="Date"))
    return df.round(1)


def invert(df: pd.DataFrame, ref: float | None = None) -> tuple[pd.DataFrame, float]:
    """上下反転: 価格' = ref×2 − 価格（高値と安値、始値と終値の役割を入れ替える）。ref は期間の高値+安値の半値。"""
    if ref is None:
        ref = (float(df["High"].max()) + float(df["Low"].min())) / 2.0
    out = pd.DataFrame({"Open": 2 * ref - df["Open"], "High": 2 * ref - df["Low"],
                        "Low": 2 * ref - df["High"], "Close": 2 * ref - df["Close"]}, index=df.index)
    if "Volume" in df.columns:
        out["Volume"] = df["Volume"]
    return out, ref
