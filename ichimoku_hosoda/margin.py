"""信用残（週次）の直近推移。

取得元: Yahoo!ファイナンス（日本）の信用残ページ https://finance.yahoo.co.jp/quote/<code>.T/margin
  表「日付 / 売り残 / 買い残 / 信用倍率 / 前週比」をそのまま読む（週次・直近約1か月分を返す）。
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


def fetch_margin(code: str, weeks: int = 5) -> dict:
    """{'rows': [{'date','sell','buy','ratio','sell_chg','buy_chg'}...], 'source': url} / 失敗時は {'error': ...}。"""
    c = re.sub(r"\.T$", "", code.strip().upper())
    if not re.fullmatch(r"[0-9][0-9A-Z]{3}", c):
        return {"error": "信用残は東証個別銘柄のみ"}
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
    rows = [x for x in rows if x["buy"] is not None or x["sell"] is not None]
    rows.sort(key=lambda x: x["date"])
    rows = rows[-weeks:]
    for i, x in enumerate(rows):
        prev = rows[i - 1] if i else None
        x["buy_chg"] = (x["buy"] - prev["buy"]) if prev and x["buy"] is not None and prev["buy"] is not None else None
        x["sell_chg"] = (x["sell"] - prev["sell"]) if prev and x["sell"] is not None and prev["sell"] is not None else None
    if not rows:
        return {"error": "信用残の行を読めませんでした", "source": url}
    return {"rows": rows, "source": url}


def synthetic_margin() -> dict:
    import numpy as np
    rng = np.random.default_rng(5)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=5, freq="W-FRI")
    buy, sell = 1_200_000, 300_000
    rows = []
    for d in dates:
        buy += int(rng.normal(0, 80_000)); sell += int(rng.normal(0, 30_000))
        rows.append({"date": d.strftime("%Y-%m-%d"), "sell": sell, "buy": buy, "ratio": round(buy / sell, 2)})
    for i, x in enumerate(rows):
        x["buy_chg"] = rows[i]["buy"] - rows[i - 1]["buy"] if i else None
        x["sell_chg"] = rows[i]["sell"] - rows[i - 1]["sell"] if i else None
    return {"rows": rows, "source": "synthetic"}
