"""銘柄名 ⇄ コードの解決。

一次: JPX「東証上場銘柄一覧」（毎月第3営業日更新、Excel）をサーバー起動後に取得して 24 時間キャッシュ。
     https://www.jpx.co.jp/markets/statistics-equities/misc/01.html
二次: Yahoo Finance の検索 API（英字名・海外銘柄向け）。
指数・為替は固定の別名表で解決。
"""
from __future__ import annotations

import io
import json
import re
import threading
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
PAGE = "https://www.jpx.co.jp/markets/statistics-equities/misc/01.html"
KNOWN = ["https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xlsx",
         "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"]
CACHE = Path("/tmp/ichimoku_jpx_list.pkl")
TTL = 24 * 3600

SPECIAL = [  # (表示名, シンボル, 別名…)
    ("日経平均", "^N225", ["日経", "日経225", "N225", "NIKKEI"]),
    ("TOPIX連動ETF(1306)", "1306.T", ["TOPIX", "トピックス"]),
    ("米ドル/円", "USDJPY=X", ["ドル円", "USDJPY", "ドル"]),
    ("ユーロ/円", "EURJPY=X", ["ユーロ円", "EURJPY"]),
    ("ユーロ/米ドル", "EURUSD=X", ["ユーロドル", "EURUSD"]),
    ("英ポンド/円", "GBPJPY=X", ["ポンド円", "GBPJPY"]),
    ("豪ドル/円", "AUDJPY=X", ["豪ドル円", "AUDJPY"]),
    ("NYダウ", "^DJI", ["ダウ", "DOW"]),
    ("S&P500", "^GSPC", ["SP500", "S&P"]),
    ("ナスダック総合", "^IXIC", ["ナスダック", "NASDAQ"]),
    ("VN-Index（ベトナム）", "^VNINDEX.VN", ["VNINDEX", "VN-INDEX", "ベトナム指数", "ホーチミン指数"]),
    ("米ドル/ベトナムドン", "USDVND=X", ["USDVND", "ドン"]),
]

# 米国株（主要銘柄）: (表示名, シンボル, 別名)
US_LIST = [
    ('Apple', 'AAPL', ['アップル']),
    ('Microsoft', 'MSFT', ['マイクロソフト']),
    ('NVIDIA', 'NVDA', ['エヌビディア']),
    ('Amazon.com', 'AMZN', ['アマゾン']),
    ('Alphabet (Google) A', 'GOOGL', ['グーグル', 'アルファベット']),
    ('Meta Platforms', 'META', ['メタ', 'フェイスブック']),
    ('Tesla', 'TSLA', ['テスラ']),
    ('Berkshire Hathaway B', 'BRK-B', ['バークシャー']),
    ('Broadcom', 'AVGO', ['ブロードコム']),
    ('JPMorgan Chase', 'JPM', ['JPモルガン']),
    ('Visa', 'V', ['ビザ']),
    ('Mastercard', 'MA', ['マスターカード']),
    ('UnitedHealth', 'UNH', ['ユナイテッドヘルス']),
    ('Exxon Mobil', 'XOM', ['エクソン']),
    ('Eli Lilly', 'LLY', ['イーライリリー']),
    ('Johnson & Johnson', 'JNJ', ['ジョンソン']),
    ('Walmart', 'WMT', ['ウォルマート']),
    ('Procter & Gamble', 'PG', ['P&G', 'プロクター']),
    ('Home Depot', 'HD', ['ホームデポ']),
    ('Costco', 'COST', ['コストコ']),
    ('Netflix', 'NFLX', ['ネットフリックス']),
    ('AMD', 'AMD', ['エーエムディー']),
    ('Intel', 'INTC', ['インテル']),
    ('Coca-Cola', 'KO', ['コカコーラ']),
    ('PepsiCo', 'PEP', ['ペプシ']),
    ('Walt Disney', 'DIS', ['ディズニー']),
    ('Boeing', 'BA', ['ボーイング']),
    ('Nike', 'NKE', ['ナイキ']),
    ('Oracle', 'ORCL', ['オラクル']),
    ('Salesforce', 'CRM', ['セールスフォース']),
    ('Adobe', 'ADBE', ['アドビ']),
    ('Qualcomm', 'QCOM', ['クアルコム']),
    ('Palantir', 'PLTR', ['パランティア']),
    ('Taiwan Semiconductor ADR', 'TSM', ['TSMC', '台湾積体電路']),
    ('SPDR S&P500 ETF', 'SPY', ['エスアンドピー', 'ETF']),
    ('Invesco QQQ', 'QQQ', ['ナスダック100', 'ETF']),
]
# ベトナム株（VN30 ほか）: (表示名, シンボル, 別名)
VN_LIST = [
    ('Ngân hàng Á Châu (ACB)', 'ACB.VN', ['ACB', 'VN:ACB', 'Asia', 'Commercial', 'Bank']),
    ('Becamex IDC', 'BCM.VN', ['BCM', 'VN:BCM']),
    ('BIDV', 'BID.VN', ['BID', 'VN:BID', 'Ngân', 'hàng', 'Đầu', 'tư', 'và', 'Phát', 'triển']),
    ('Bảo Việt', 'BVH.VN', ['BVH', 'VN:BVH', 'Bao', 'Viet', 'Holdings']),
    ('VietinBank', 'CTG.VN', ['CTG', 'VN:CTG', 'Ngân', 'hàng', 'Công', 'Thương']),
    ('FPT', 'FPT.VN', ['FPT', 'VN:FPT', 'FPT', 'Corporation']),
    ('PV Gas', 'GAS.VN', ['GAS', 'VN:GAS', 'Tổng', 'công', 'ty', 'Khí']),
    ('Cao su Việt Nam', 'GVR.VN', ['GVR', 'VN:GVR', 'Vietnam', 'Rubber', 'Group']),
    ('HDBank', 'HDB.VN', ['HDB', 'VN:HDB']),
    ('Hòa Phát', 'HPG.VN', ['HPG', 'VN:HPG', 'Hoa', 'Phat', 'Group', 'thép']),
    ('MB Bank', 'MBB.VN', ['MBB', 'VN:MBB', 'Ngân', 'hàng', 'Quân', 'đội']),
    ('Masan', 'MSN.VN', ['MSN', 'VN:MSN', 'Masan', 'Group']),
    ('Thế Giới Di Động', 'MWG.VN', ['MWG', 'VN:MWG', 'Mobile', 'World']),
    ('Petrolimex', 'PLX.VN', ['PLX', 'VN:PLX']),
    ('PV Power', 'POW.VN', ['POW', 'VN:POW']),
    ('Sabeco', 'SAB.VN', ['SAB', 'VN:SAB', 'bia', 'Sài', 'Gòn']),
    ('SHB', 'SHB.VN', ['SHB', 'VN:SHB', 'Ngân', 'hàng', 'Sài', 'Gòn', 'Hà', 'Nội']),
    ('SeABank', 'SSB.VN', ['SSB', 'VN:SSB']),
    ('Chứng khoán SSI', 'SSI.VN', ['SSI', 'VN:SSI', 'SSI', 'Securities']),
    ('Sacombank', 'STB.VN', ['STB', 'VN:STB']),
    ('Techcombank', 'TCB.VN', ['TCB', 'VN:TCB']),
    ('TPBank', 'TPB.VN', ['TPB', 'VN:TPB']),
    ('Vietcombank', 'VCB.VN', ['VCB', 'VN:VCB', 'Ngân', 'hàng', 'Ngoại', 'thương']),
    ('Vinhomes', 'VHM.VN', ['VHM', 'VN:VHM']),
    ('VIB', 'VIB.VN', ['VIB', 'VN:VIB']),
    ('Vingroup', 'VIC.VN', ['VIC', 'VN:VIC']),
    ('Vietjet Air', 'VJC.VN', ['VJC', 'VN:VJC', 'ベトジェット']),
    ('Vinamilk', 'VNM.VN', ['VNM', 'VN:VNM', 'ビナミルク', 'sữa']),
    ('VPBank', 'VPB.VN', ['VPB', 'VN:VPB']),
    ('Vincom Retail', 'VRE.VN', ['VRE', 'VN:VRE']),
    ('Hóa chất Đức Giang', 'DGC.VN', ['DGC', 'VN:DGC']),
    ('Chứng khoán VNDirect', 'VND.VN', ['VND', 'VN:VND']),
    ('PNJ', 'PNJ.VN', ['PNJ', 'VN:PNJ', 'vàng', 'bạc', 'đá', 'quý', 'Phú', 'Nhuận']),
    ('REE', 'REE.VN', ['REE', 'VN:REE']),
    ('Khang Điền', 'KDH.VN', ['KDH', 'VN:KDH']),
    ('Novaland', 'NVL.VN', ['NVL', 'VN:NVL']),
    ('Đất Xanh', 'DXG.VN', ['DXG', 'VN:DXG']),
    ('Chứng khoán Vietcap', 'VCI.VN', ['VCI', 'VN:VCI']),
    ('Gemadept', 'GMD.VN', ['GMD', 'VN:GMD']),
]


_lock = threading.Lock()
_df: pd.DataFrame | None = None
_loaded_at = 0.0
_last_error = ""
_fail_at = 0.0
RETRY_AFTER = 600   # 取得失敗後は 10 分間再試行しない（検索のたびに待たせない）


def norm(s: str) -> str:
    """NFKC＋小文字＋ひらがな→カタカナ＋空白・記号除去。"""
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in s)
    return re.sub(r"[\s　・\.\-/（）()&＆,，株式会社]", "", s)


def _get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "ja"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _download() -> pd.DataFrame:
    urls = list(KNOWN)
    try:
        html = _get(PAGE).decode("utf-8", "replace")
        for m in re.findall(r'href="([^"]+data_j\.xlsx?)"', html):
            urls.insert(0, urllib.parse.urljoin(PAGE, m))
    except Exception:  # noqa: BLE001
        pass
    last = None
    for u in urls:
        try:
            raw = _get(u, 60)
            df = pd.read_excel(io.BytesIO(raw), dtype=str)
            cols = {c: c for c in df.columns}
            code_c = next(c for c in cols if "コード" in c and "業種" not in c and "規模" not in c)
            name_c = next(c for c in cols if "銘柄名" in c)
            mkt_c = next((c for c in cols if "市場" in c), None)
            ind_c = next((c for c in cols if "33業種区分" in c), None)
            out = pd.DataFrame({
                "code": df[code_c].astype(str).str.strip().str.upper().str.replace(r"\.0$", "", regex=True),
                "name": df[name_c].astype(str).str.strip(),
                "market": df[mkt_c].astype(str).str.strip() if mkt_c else "",
                "industry": df[ind_c].astype(str).str.strip() if ind_c else "",
            })
            out = out[out["code"].str.fullmatch(r"[0-9][0-9A-Z]{3}")]
            out["key"] = out["name"].map(norm)
            return out.reset_index(drop=True)
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"JPX 銘柄一覧を取得できませんでした: {last}")


def _ensure(block: bool = True) -> pd.DataFrame | None:
    global _df, _loaded_at, _last_error, _fail_at
    if _df is not None and time.time() - _loaded_at < TTL:
        return _df
    if _df is None and CACHE.exists() and time.time() - CACHE.stat().st_mtime < TTL * 7:
        try:
            _df, _loaded_at = pd.read_pickle(CACHE), CACHE.stat().st_mtime
            if time.time() - _loaded_at < TTL:
                return _df
        except Exception:  # noqa: BLE001
            pass
    if time.time() - _fail_at < RETRY_AFTER:
        return _df
    if not block:
        threading.Thread(target=_ensure, daemon=True).start()
        return _df
    if not _lock.acquire(timeout=0.5):   # 別スレッドが取得中なら待たない
        return _df
    try:
        if _df is not None and time.time() - _loaded_at < TTL:
            return _df
        try:
            df = _download()
            _df, _loaded_at, _last_error = df, time.time(), ""
            try:
                df.to_pickle(CACHE)
            except Exception:  # noqa: BLE001
                pass
        except Exception as e:  # noqa: BLE001
            _last_error, _fail_at = str(e), time.time()
    finally:
        _lock.release()
    return _df


def warmup() -> None:
    threading.Thread(target=_ensure, daemon=True).start()


def _yahoo_search(q: str, n: int = 8) -> list[dict]:
    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search?" + urllib.parse.urlencode(
            {"q": q, "quotesCount": n, "newsCount": 0, "lang": "ja-JP", "region": "JP"})
        j = json.loads(_get(url, 10).decode("utf-8"))
        out = []
        for it in j.get("quotes", []):
            sym = it.get("symbol")
            if sym:
                out.append({"code": sym.replace(".T", "") if sym.endswith(".T") else sym, "symbol": sym,
                            "name": it.get("shortname") or it.get("longname") or sym,
                            "market": it.get("exchDisp", ""), "industry": it.get("typeDisp", "")})
        return out
    except Exception:  # noqa: BLE001
        return []


def search(q: str, limit: int = 15) -> list[dict]:
    q = (q or "").strip()
    if not q:
        return []
    k = norm(q)
    res: list[dict] = []
    for disp, sym, aliases in SPECIAL:
        if any(k and (k in norm(a) or norm(a).startswith(k)) for a in [disp, sym] + aliases):
            res.append({"code": sym, "symbol": sym, "name": disp, "market": "指数・為替", "industry": ""})
    for lst, mk in ((US_LIST, "米国株"), (VN_LIST, "ベトナム株")):
        for disp, sym, aliases in lst:
            tick = sym.replace(".VN", "")
            if (k and any(norm(x).startswith(k) or (len(k) >= 3 and k in norm(x)) for x in [disp, sym, tick] + aliases)):
                res.append({"code": sym, "symbol": sym, "name": disp, "market": mk, "industry": ""})
    df = _ensure(block=True)
    if df is not None and len(df):
        uq = unicodedata.normalize("NFKC", q).upper()
        by_code = df[df["code"].str.startswith(uq)] if re.fullmatch(r"[0-9][0-9A-Z]{0,3}", uq) else df.iloc[0:0]
        by_name = df[df["key"].str.contains(re.escape(k), na=False)] if k else df.iloc[0:0]
        # 前方一致を優先
        by_name = by_name.assign(_p=~by_name["key"].str.startswith(k)).sort_values(["_p", "code"])
        for _, r in pd.concat([by_code, by_name]).drop_duplicates("code").head(limit).iterrows():
            res.append({"code": r["code"], "symbol": r["code"] + ".T", "name": r["name"],
                        "market": r["market"], "industry": r["industry"]})
    if len(res) < 3 and not re.fullmatch(r"[0-9][0-9A-Za-z]{3}", q):
        seen = {x["symbol"] for x in res}
        res += [x for x in _yahoo_search(q) if x["symbol"] not in seen]
    return res[:limit]


def name_for(code: str) -> str | None:
    c = (code or "").strip().upper()
    for disp, sym, aliases in SPECIAL + US_LIST + VN_LIST:
        if c in (sym.upper(), sym.upper().replace(".VN", ""), *(a.upper() for a in aliases)):
            return disp
    c = re.sub(r"\.T$", "", c)
    df = _ensure(block=False)
    if df is not None:
        hit = df[df["code"] == c]
        if len(hit):
            return str(hit.iloc[0]["name"])
    return None


def resolve(q: str) -> tuple[str, str | None]:
    """入力（コード or 名前）→ (取得用コード, 表示名)。名前なら検索の先頭を採用。"""
    s = (q or "").strip()
    if not s or s.upper() == "DEMO" or re.fullmatch(r"[0-9][0-9A-Za-z]{3}(\.T)?", s, flags=re.I) \
            or re.fullmatch(r"[\^A-Za-z0-9=.\-]+", s):
        return s, name_for(s)
    hits = search(s, 1)
    if hits:
        return hits[0]["symbol"], hits[0]["name"]
    return s, None


def status() -> dict:
    return {"loaded": _df is not None, "count": 0 if _df is None else int(len(_df)),
            "loaded_at": _loaded_at, "error": _last_error}
