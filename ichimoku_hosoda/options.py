"""個別銘柄オプションの建玉（行使価格別）と損益分岐。

米国株: yfinance の option_chain（建玉 openInterest・出来高・価格・IV）。
日本株: JPX「先物・オプション清算値段」（rbYYYYMMDD.csv）の清算価格・IV・原資産価格・残日数。
        JPX の建玉残高表は有価証券オプションを合計でしか載せないため、行使価格別の建玉は出せない（価格・IV・損益分岐のみ）。

誰が買って誰が売ったかは公表されないので、出すのは「行使価格ごとの残っている枚数（建玉）」と、
その価格（プレミアム）から計算した損益分岐（コール＝行使価格＋プレミアム、プット＝行使価格−プレミアム）。
"""
from __future__ import annotations

import io
import math
import re
import time
import urllib.request
from datetime import date, datetime

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
_CACHE: dict = {}


def _get(url: str, binary: bool = False, timeout: int = 30):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "ja"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        b = r.read()
    return b if binary else b.decode("utf-8", "replace")


def _cached(key, ttl, fn):
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    v = fn()
    _CACHE[key] = (time.time(), v)
    return v


# ---------------------------------------------------------------- 集計（共通）
def summarize(spot: float, exp: str, days: int, strikes: list[dict]) -> dict:
    """strikes: [{strike, call_oi, put_oi, call_px, put_px, call_iv, put_iv, call_vol, put_vol, call_chg, put_chg}]"""
    S = sorted([s for s in strikes if s.get("strike")], key=lambda s: s["strike"])
    for s in S:
        k = s["strike"]
        s["call_be"] = k + s["call_px"] if s.get("call_px") is not None else None
        s["put_be"] = k - s["put_px"] if s.get("put_px") is not None else None
    coi = sum(s.get("call_oi") or 0 for s in S)
    poi = sum(s.get("put_oi") or 0 for s in S)

    def wavg(side):
        w = [(s.get(side + "_oi") or 0, s.get(side + "_be")) for s in S if s.get(side + "_be") is not None]
        tot = sum(a for a, _ in w)
        return sum(a * b for a, b in w) / tot if tot else None

    # 最大苦痛価格（満期にその値段で終わると、オプションの買い手が受け取る額の合計が最小）
    mp = None
    if coi + poi:
        best = None
        for p in [s["strike"] for s in S]:
            pay = sum((s.get("call_oi") or 0) * max(p - s["strike"], 0) + (s.get("put_oi") or 0) * max(s["strike"] - p, 0) for s in S)
            if best is None or pay < best:
                best, mp = pay, p
    above = [s for s in S if s["strike"] >= spot and (s.get("call_oi") or 0) > 0]
    below = [s for s in S if s["strike"] <= spot and (s.get("put_oi") or 0) > 0]
    cw = max(above, key=lambda s: s["call_oi"])["strike"] if above else None
    pw = max(below, key=lambda s: s["put_oi"])["strike"] if below else None
    atm = min(S, key=lambda s: abs(s["strike"] - spot)) if S else None
    ivs = [x for x in ((atm or {}).get("call_iv"), (atm or {}).get("put_iv")) if x]
    iv = sum(ivs) / len(ivs) if ivs else None
    sig = spot * iv * math.sqrt(max(days, 1) / 365) if iv else None
    return {"exp": exp, "days": int(days), "strikes": S, "call_oi": coi, "put_oi": poi,
            "pcr": (poi / coi) if coi else None, "max_pain": mp, "call_wall": cw, "put_wall": pw,
            "call_be_avg": wavg("call"), "put_be_avg": wavg("put"), "atm_iv": iv, "sigma": sig}


# ---------------------------------------------------------------- 米国株
def us_options(symbol: str, max_exp: int = 4) -> dict:
    import yfinance as yf
    t = yf.Ticker(symbol)
    exps = list(t.options or [])
    if not exps:
        return {"error": "この銘柄の上場オプションが見つかりません"}
    try:
        h = t.history(period="5d")
        spot = float(h["Close"].iloc[-1])
    except Exception:  # noqa: BLE001
        spot = None
    today = date.today()
    out = []
    for e in exps[:max_exp + 1]:
        days = (datetime.strptime(e, "%Y-%m-%d").date() - today).days
        if days < 0:
            continue
        ch = t.option_chain(e)
        rows: dict[float, dict] = {}
        for side, df in (("call", ch.calls), ("put", ch.puts)):
            for _, r in df.iterrows():
                k = float(r["strike"])
                bid, ask, last = (float(r.get(x) or 0) for x in ("bid", "ask", "lastPrice"))
                px = (bid + ask) / 2 if bid > 0 and ask > 0 else (last if last > 0 else None)
                d = rows.setdefault(k, {"strike": k})
                oi = r.get("openInterest")
                d[side + "_oi"] = int(oi) if oi == oi and oi is not None else 0
                v = r.get("volume")
                d[side + "_vol"] = int(v) if v == v and v is not None else 0
                d[side + "_px"] = px
                iv = r.get("impliedVolatility")
                d[side + "_iv"] = float(iv) if iv == iv and iv and iv > 0.001 else None
        if spot is None:
            continue
        out.append(summarize(spot, e, days, list(rows.values())))
        if len(out) >= max_exp:
            break
    if not out:
        return {"error": "オプションの建玉を読めませんでした"}
    return {"market": "us", "spot": spot, "source": "Yahoo Finance（yfinance）", "unit": "枚（1枚＝100株）",
            "asof": today.isoformat(), "expiries": out}


# ---------------------------------------------------------------- 日本株（JPX）
JPX = "https://www.jpx.co.jp"
OI_PAGE = JPX + "/markets/derivatives/trading-volume/index.html"
RB_PAGE = JPX + "/markets/derivatives/settlement-price/index.html"


def _link(page: str, pat: str) -> tuple[str, str] | None:
    html = _get(page)
    for m in re.finditer(r'href="([^"]+)"', html):
        d = re.search(pat, m.group(1))
        if d:
            u = m.group(1)
            return (u if u.startswith("http") else JPX + u), f"{d.group(1)}-{d.group(2)}-{d.group(3)}"
    return None


def _jpx_files() -> dict:
    """最新の清算値段 CSV を読み、有価証券オプションの行だけを残す（1 時間キャッシュ）。
    行使価格別の建玉は、JPX の建玉残高表（open_interest.xlsx）では有価証券オプションが合計だけのため使えない。"""
    def load():
        res = {"rb": [], "errors": []}
        try:
            u, d = _link(RB_PAGE, r"rb(\d{4})(\d{2})(\d{2})\.csv$")
            res["rb_date"], res["rb_url"] = d, u
            res["rb"] = parse_rb_csv(_get(u, binary=True, timeout=60))
        except Exception as e:  # noqa: BLE001
            res["errors"].append(f"清算値段: {e}")
        return res
    return _cached("jpx", 3600, load)


def _num(x):
    try:
        s = str(x).replace(",", "").strip()
        if s in ("", "-", "nan", "None"):
            return None
        return float(s)
    except ValueError:
        return None


def parse_rb_csv(buf: bytes) -> list[dict]:
    """清算値段 CSV（rbYYYYMMDD.csv、Shift_JIS）:
    銘柄コード,銘柄名称,PUT/CAL,限月,権利行使価格,清算価格,理論価格,原資産価格,ボラティリティ(%),金利,残日数,原資産名称
    有価証券オプションの銘柄名称は「CAL_285A_261212_9000」（末尾 A は権利行使価格の調整銘柄）。証券コードの行だけ残す。"""
    try:
        text = buf.decode("cp932")
    except UnicodeDecodeError:
        text = buf.decode("utf-8", "replace")
    out = []
    for line in text.splitlines():
        c = [x.strip().strip('"') for x in line.split(",")]
        if len(c) < 11 or c[2] not in ("PUT", "CAL"):
            continue
        m = re.match(r"^(PUT|CAL)_([0-9][0-9A-Z]{3})_(\d{6})_([\d.]+)([A-Z]?)$", c[1])
        if not m:
            continue
        iv = _num(c[8])
        out.append({"name": c[1], "code": m.group(2), "pc": "P" if m.group(1) == "PUT" else "C",
                    "month": c[3][:6], "exp": "20" + m.group(3), "strike": _num(c[4]), "adj": bool(m.group(5)),
                    "px": _num(c[5]), "spot": _num(c[7]), "iv": iv / 100 if iv and iv > 0 else None,
                    "days": int(_num(c[10]) or 0)})
    return out


def jp_options(code: str) -> dict:
    c = re.sub(r"\.T$", "", code.strip().upper())
    J = _jpx_files()
    rb = [x for x in J["rb"] if x["code"] == c and x["strike"]]
    if not rb:
        msg = "JPX の有価証券オプションに、この銘柄の清算値段がありません（オプションが上場していない銘柄）"
        if J["errors"]:
            msg += "／" + " / ".join(J["errors"])
        return {"error": msg, "source": "JPX"}
    spot = next((x["spot"] for x in rb if x.get("spot")), None)
    exps = []
    for exp in sorted({x["exp"] for x in rb}):
        rows: dict[float, dict] = {}
        days = 0
        # 同じ行使価格に調整銘柄（末尾 A）と通常銘柄がある場合は通常銘柄を優先
        for x in sorted([x for x in rb if x["exp"] == exp], key=lambda x: not x["adj"]):
            side = "put" if x["pc"] == "P" else "call"
            d = rows.setdefault(x["strike"], {"strike": x["strike"]})
            d[side + "_px"] = x["px"] if x["px"] and x["px"] > 0 else None
            d[side + "_iv"] = x["iv"] if x["iv"] and x["iv"] < 3 else None
            d["adj"] = x["adj"]
            days = max(days, x["days"])
        if spot is None or days < 1:
            continue
        e = summarize(spot, f"{exp[:4]}-{exp[4:6]}-{exp[6:]}", days, list(rows.values()))
        e["no_oi"] = True
        exps.append(e)
    if not exps:
        return {"error": "JPX のデータから組み立てられませんでした", "source": "JPX"}
    return {"market": "jp", "spot": spot, "source": "JPX（先物・オプション清算値段）", "unit": "枚",
            "asof": J.get("rb_date"), "urls": [J["rb_url"]] if J.get("rb_url") else [], "no_oi": True,
            "expiries": exps[:4]}


# ---------------------------------------------------------------- デモ
def synthetic(spot: float) -> dict:
    import numpy as np
    rng = np.random.default_rng(3)
    step = 10 ** math.floor(math.log10(spot * 0.05)) or 1
    step = step * (5 if spot * 0.05 / step > 4 else 2 if spot * 0.05 / step > 2 else 1)
    exps = []
    for j, days in enumerate((9, 37, 65)):
        rows = []
        iv = 0.32 + 0.02 * j
        for k in np.arange(round(spot * 0.75 / step) * step, spot * 1.25, step):
            k = float(k)
            t = days / 365
            sd = spot * iv * math.sqrt(t)
            from math import erf, exp, sqrt
            N = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
            d1 = (math.log(spot / k) + 0.5 * iv * iv * t) / (iv * math.sqrt(t))
            d2 = d1 - iv * math.sqrt(t)
            call = spot * N(d1) - k * N(d2)
            put = call - spot + k
            w = exp(-((k - spot) / (sd * 1.4)) ** 2)
            rows.append({"strike": k, "call_px": round(max(call, 0.1), 1), "put_px": round(max(put, 0.1), 1),
                         "call_iv": iv, "put_iv": iv + 0.02,
                         "call_oi": int(w * 3000 * (1.4 if k > spot else 0.6) * rng.uniform(.5, 1.5) * (2 if abs(k / spot - 1.1) < .02 else 1)),
                         "put_oi": int(w * 2600 * (1.4 if k < spot else 0.5) * rng.uniform(.5, 1.5) * (2 if abs(k / spot - .92) < .02 else 1)),
                         "call_vol": int(rng.uniform(0, 400) * w), "put_vol": int(rng.uniform(0, 400) * w)})
        exps.append(summarize(spot, f"満期{j + 1}", days, rows))
    return {"market": "demo", "spot": spot, "source": "デモデータ", "unit": "枚", "expiries": exps}


def fetch(api_code: str, market: str, spot_hint: float | None = None) -> dict:
    if market == "demo":
        return synthetic(spot_hint or 1800.0)
    if market == "us":
        return _cached(("us", api_code), 1800, lambda: us_options(api_code))
    if market == "jp":
        return jp_options(api_code)
    return {"error": "オプションは米国株と日本株（JPX 上場の有価証券オプション）のみ"}
