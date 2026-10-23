"""信用残（週次）の推移。

取得元（上から順に試す）:
  1. 株探（スマホ版）週次信用残 https://s.kabutan.jp/stocks/<code>/historical_margins/weekly/?page=N
     表「日付(YY/MM/DD) / 終値 / 前週比率 / 信用倍率 / 売り残(株) / 買い残(株) / …」、1ページ30週
  2. Yahoo!ファイナンス（日本）の信用残ページ https://finance.yahoo.co.jp/quote/<code>.T/margin
  表「日付 / 売り残 / 買い残 / 信用倍率 / 前週比」をそのまま読む（週次・ページにある分（最大約1年）を返す）。
ページ構成が変わった場合は取得不可として notes に理由を返す（アプリ本体は動く）。
"""
from __future__ import annotations

import io
import re
import urllib.request

import pandas as pd

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


def _num(x) -> float | None:
    if x is None:
        return None
    s = str(x).replace(",", "").replace("株", "").replace("倍", "").replace("+", "").strip()
    s = s.replace("−", "-").replace("－", "-").replace("―", "")
    if s in ("", "-", "---", "nan", "None"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "ja"})
    with urllib.request.urlopen(req, timeout=25) as r:
        return r.read().decode("utf-8", "replace")


def _finish(rows: list[dict], weeks: int, url: str) -> dict:
    rows = [x for x in rows if x["buy"] is not None or x["sell"] is not None]
    seen, uniq = set(), []
    for x in sorted(rows, key=lambda x: x["date"]):
        if x["date"] not in seen:
            seen.add(x["date"])
            uniq.append(x)
    rows = uniq[-weeks:]
    for i, x in enumerate(rows):
        prev = rows[i - 1] if i else None
        x["buy_chg"] = (x["buy"] - prev["buy"]) if prev and x["buy"] is not None and prev["buy"] is not None else None
        x["sell_chg"] = (x["sell"] - prev["sell"]) if prev and x["sell"] is not None and prev["sell"] is not None else None
    if not rows:
        return {"error": "信用残の行を読めませんでした", "source": url}
    return {"rows": rows, "source": url}


def fetch_kabutan(c: str, weeks: int = 60) -> dict:
    base = f"https://s.kabutan.jp/stocks/{c}/historical_margins/weekly/"
    rows: list[dict] = []
    for page in range(1, (weeks + 29) // 30 + 1):
        html = _get(base + (f"?page={page}" if page > 1 else ""))
        got = 0
        for t in pd.read_html(io.StringIO(html)):
            cols = [str(x) for x in t.columns]
            if not any("売り残" in x for x in cols) or not any("買い残" in x for x in cols):
                continue
            col = lambda key: next((t.columns[i] for i, x in enumerate(cols) if key in x and "額" not in x), None)
            dcol, scol, bcol, rcol = col("日付"), col("売り残"), col("買い残"), col("倍率")
            pcol = col("売買単価")
            for _, r in t.iterrows():
                m = re.search(r"(\d{2,4})/(\d{1,2})/(\d{1,2})", str(r[dcol]))
                if not m:
                    continue
                y = int(m.group(1)); y = y + 2000 if y < 100 else y
                rows.append({"date": f"{y}-{int(m.group(2)):02d}-{int(m.group(3)):02d}",
                             "sell": _num(r[scol]), "buy": _num(r[bcol]), "ratio": _num(r[rcol]) if rcol is not None else None,
                             "avg_price": _num(r[pcol]) if pcol is not None else None})
                got += 1
        if got < 30:
            break
    if not rows:
        raise RuntimeError("株探の表を読めませんでした")
    return _finish(rows, weeks, base)


def fetch_margin(code: str, weeks: int = 60) -> dict:
    """{'rows': [{'date','sell','buy','ratio','sell_chg','buy_chg'}...], 'source': url} / 失敗時は {'error': ...}。"""
    c = re.sub(r"\.T$", "", code.strip().upper())
    if not re.fullmatch(r"[0-9][0-9A-Z]{3}", c):
        return {"error": "信用残は東証個別銘柄のみ"}
    errs = []
    try:
        return fetch_kabutan(c, weeks)
    except Exception as e:  # noqa: BLE001
        errs.append(f"株探: {e}")
    res = _fetch_yahoo(c, weeks)
    if "error" in res:
        res["error"] = " / ".join(errs + ["Yahoo: " + res["error"]])
    return res


def _fetch_yahoo(c: str, weeks: int) -> dict:
    url = f"https://finance.yahoo.co.jp/quote/{c}.T/margin"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "ja"})
        with urllib.request.urlopen(req, timeout=25) as r:
            html = r.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return {"error": f"取得失敗: {e}", "source": url}
    try:
        tables = pd.read_html(io.StringIO(html))
    except Exception as e:  # noqa: BLE001
        return {"error": f"表の解析失敗: {e}", "source": url}
    target = None
    for t in tables:
        cols = " ".join(str(c) for c in t.columns)
        if "売り残" in cols and "買い残" in cols and ("日付" in cols or "日" in cols):
            target = t
            break
    if target is None:
        return {"error": "信用残の表が見つかりません（ページ構成の変更の可能性）", "source": url}
    cols = {str(c): c for c in target.columns}

    def find(key):
        for k, v in cols.items():
            if key in k:
                return v
        return None

    dcol, scol, bcol, rcol = find("日"), find("売り残"), find("買い残"), find("倍率")
    rows = []
    for _, r in target.iterrows():
        d = str(r[dcol]) if dcol is not None else ""
        m = re.search(r"(\d{4})[/年.-](\d{1,2})[/月.-](\d{1,2})", d) or re.search(r"(\d{1,2})[/月](\d{1,2})", d)
        if not m:
            continue
        if len(m.groups()) == 3:
            date = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
        else:
            date = f"{int(m.group(1)):02d}-{int(m.group(2)):02d}"
        rows.append({"date": date, "sell": _num(r[scol]) if scol is not None else None,
                     "buy": _num(r[bcol]) if bcol is not None else None,
                     "ratio": _num(r[rcol]) if rcol is not None else None})
    # 年のない日付（MM-DD）は直近から遡って年を補う
    today = pd.Timestamp.today()
    for x in rows:
        if re.fullmatch(r"\d{2}-\d{2}", x["date"]):
            mm = int(x["date"][:2])
            y = today.year if mm <= today.month else today.year - 1
            x["date"] = f"{y}-{x['date']}"
    return _finish(rows, weeks, url)


def synthetic_margin(end=None, weeks: int = 30) -> dict:
    import numpy as np
    rng = np.random.default_rng(5)
    dates = pd.date_range(end=pd.Timestamp(end) if end is not None else pd.Timestamp.today(), periods=weeks, freq="W-FRI")
    buy, sell = 1_200_000, 300_000
    rows = []
    for d in dates:
        buy += int(rng.normal(0, 80_000)); sell += int(rng.normal(0, 30_000))
        rows.append({"date": d.strftime("%Y-%m-%d"), "sell": sell, "buy": buy, "ratio": round(buy / sell, 2)})
    for i, x in enumerate(rows):
        x["buy_chg"] = rows[i]["buy"] - rows[i - 1]["buy"] if i else None
        x["sell_chg"] = rows[i]["sell"] - rows[i - 1]["sell"] if i else None
    return {"rows": rows, "source": "synthetic"}


def estimate_pl(m: dict, daily: pd.DataFrame, window: int = 26) -> dict | None:
    """信用の買い方・売り方の含み益率（推定）。
    週ごとの残高の増加分を、その週の平均売買単価（無ければその週の終値の平均）で建てたと見なして平均取得単価を積み上げる。
    減少分は平均単価を変えずに差し引く。制度信用の期日（6か月）に合わせて直近 window 週から積み上げ始める。
    含み益率: 買い方 = 現値 ÷ 平均取得 − 1、売り方 = 1 − 現値 ÷ 平均取得。"""
    rows = (m or {}).get("rows") or []
    if len(rows) < 4 or daily is None or not len(daily):
        return None
    close = daily["Close"]
    def wk_price(r):
        if r.get("avg_price"):
            return float(r["avg_price"])
        d = pd.Timestamp(r["date"])
        seg = close[(close.index > d - pd.Timedelta(days=7)) & (close.index <= d)]
        return float(seg.mean()) if len(seg) else None
    rs = rows[-window:]
    out = {}
    for side in ("buy", "sell"):
        bal = val = None
        hist = []
        for r in rs:
            q, p = r.get(side), wk_price(r)
            if q is None or p is None:
                continue
            if bal is None:                       # 最初の週の残高はその週の単価で建てたと見なす
                bal, val = q, q * p
            elif q > bal:
                val += (q - bal) * p
                bal = q
            else:
                val = val * (q / bal) if bal else 0
                bal = q
            if bal:
                hist.append({"date": r["date"], "cost": val / bal})
        if not hist:
            continue
        cost = hist[-1]["cost"]
        px = float(close.iloc[-1])
        pl = (px / cost - 1) if side == "buy" else (1 - px / cost)
        # 週ごとの含み益率の推移（その週の終値で評価）
        series = []
        for h in hist:
            d = pd.Timestamp(h["date"])
            seg = close[close.index <= d]
            if len(seg):
                p0 = float(seg.iloc[-1])
                series.append({"date": h["date"], "pl": round(((p0 / h["cost"] - 1) if side == "buy" else (1 - p0 / h["cost"])) * 100, 2)})
        out[side] = {"cost": round(cost, 2), "pl": round(pl * 100, 2), "series": series}
    if not out:
        return None
    out["basis"] = "売買単価" if any(r.get("avg_price") for r in rs) else "終値の平均"
    out["weeks"] = len(rs)
    return out
