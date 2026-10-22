"""
大出来高急落 × 一目均衡表 スクリーナー（引け後実行）

- 対象: JPX 上場銘柄（プライム / スタンダード / グロース）。tickers.txt があればそちらを優先
- 検出: 直近 LOOKBACK 営業日以内に「出来高急増 + 急落」があった銘柄（セリング・クライマックス候補）
- 評価: 一目均衡表（転換線 / 基準線 / 雲 / 遅行スパン）と日柄（基本数値 9・17・26）でスコア付け
- 出力: watchlist.json（累積・26日で自動除外）/ watchlist.md（人が読む用）/ daily/YYYY-MM-DD.json

※ 投資助言ではありません。条件は冒頭の定数で調整してください。
"""
from __future__ import annotations

import io
import json
import sys
import time
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ---------------- 条件（ここを調整） ----------------
VOL_MULT = 3.0          # 当日出来高 >= 20日平均 × VOL_MULT
DROP_PCT = -7.0         # 当日終値 <= 前日終値 × (1 + DROP_PCT/100)
PRIOR_DROP_PCT = -15.0  # 直近20日高値からの下落率（これ以下で「下げ続けた後」とみなす）
BIG_BLACK_BODY = 0.70   # 大陰線: 実体 / (高値-安値) >= 0.70（下落率条件とどちらか満たせばOK）
LOOKBACK = 5            # 何営業日前までの急落日を拾うか
MIN_TURNOVER = 1e8      # 20日平均売買代金 >= 1億円（薄商い除外）
MIN_PRICE = 100         # 低位株除外
EXPIRE_DAYS = 26        # 監視リストから外すまでの日柄
BASIC_NUMBERS = (9, 17, 26)
MIN_DATA_RATE = 0.5     # 株価取得率がこれ未満なら「0件」ではなく異常終了させる
# ---------------------------------------------------

ROOT = Path(__file__).resolve().parent
OUT_JSON = ROOT / "watchlist.json"
OUT_MD = ROOT / "watchlist.md"
DAILY_DIR = ROOT / "daily"
# JPX「東証上場銘柄一覧」。2026年に .xls → .xlsx へ変わったため両方試す
JPX_URLS = [
    "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xlsx",
    "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls",
]
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36")


def _pick_col(df: pd.DataFrame, *keywords: str) -> str:
    """列名に keywords のいずれかを含む列を返す（JPX の列名変更に耐えるため）"""
    for col in df.columns:
        s = str(col)
        if any(k in s for k in keywords):
            return col
    raise SystemExit(f"JPX 銘柄一覧に {keywords} を含む列が見つかりません。列: {list(df.columns)}")


def load_universe() -> pd.DataFrame:
    """code, name, market の DataFrame を返す"""
    txt = ROOT / "tickers.txt"
    if txt.exists():
        codes = [l.strip() for l in txt.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
        print(f"universe source: {txt.name}（{len(codes)} 銘柄）", file=sys.stderr)
        return pd.DataFrame({"code": codes, "name": codes, "market": ""})

    df = None
    last_err: Exception | None = None
    for url in JPX_URLS:
        try:
            r = requests.get(url, timeout=60, headers={"User-Agent": UA})
            r.raise_for_status()
            df = pd.read_excel(io.BytesIO(r.content))
            print(f"universe source: {url}", file=sys.stderr)
            break
        except Exception as e:
            print(f"  {url} → {type(e).__name__}: {e}", file=sys.stderr)
            last_err = e
    if df is None:
        raise SystemExit(f"JPX の銘柄一覧を取得できませんでした: {last_err}")

    print(f"columns: {list(df.columns)}", file=sys.stderr)
    c_code = _pick_col(df, "コード")
    c_name = _pick_col(df, "銘柄名")
    c_market = _pick_col(df, "市場", "区分")
    df = df.rename(columns={c_code: "code", c_name: "name", c_market: "market"})
    df["code"] = df["code"].astype(str).str.strip()
    df = df[df["market"].astype(str).str.contains("プライム|スタンダード|グロース")]
    df = df[df["code"].str.fullmatch(r"[0-9A-Za-z]{4}")]
    return df[["code", "name", "market"]].reset_index(drop=True)


def make_session():
    """ブラウザを装ったセッション。Yahoo が GitHub Actions の IP を弾くのを回避する狙い。"""
    try:
        from curl_cffi import requests as cffi_requests
        s = cffi_requests.Session(impersonate="chrome")
        print("session: curl_cffi (impersonate=chrome)", file=sys.stderr)
        return s
    except Exception as e:
        print(f"session: curl_cffi 不可（{e}）→ requests にフォールバック", file=sys.stderr)
        s = requests.Session()
        s.headers.update({
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                           "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
        })
        return s


def _yf_download(chunk: list[str], period: str, session):
    """session 引数を受けない yfinance でも動くように２段構えで呼ぶ。"""
    kw = dict(period=period, group_by="ticker", threads=True,
              auto_adjust=False, progress=False)
    if session is not None:
        try:
            return yf.download(chunk, session=session, **kw)
        except TypeError:
            pass  # 新しめの yfinance は session を受けない（内部で curl_cffi を使う）
    return yf.download(chunk, **kw)


def download(codes: list[str], period: str = "6mo",
             chunk_size: int = 100, max_retries: int = 3) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    session = make_session()
    tickers = [f"{c}.T" for c in codes]
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        got = 0
        for attempt in range(1, max_retries + 1):
            try:
                data = _yf_download(chunk, period, session)
            except Exception as e:
                print(f"  chunk {i // chunk_size + 1}: 例外 {type(e).__name__}: {e}", file=sys.stderr)
                data = None
            if data is not None and len(data):
                for t in chunk:
                    try:
                        d = data[t] if len(chunk) > 1 else data
                    except KeyError:
                        continue
                    d = d.dropna(subset=["Close", "Volume"])
                    if len(d) >= 60:
                        out[t[:-2]] = d
                        got += 1
            if got:
                break
            if attempt < max_retries:
                wait = 5 * 2 ** (attempt - 1)  # 5s, 10s
                print(f"  chunk {i // chunk_size + 1}: 取得0件 → {wait}秒待って再試行 "
                      f"({attempt}/{max_retries})", file=sys.stderr)
                time.sleep(wait)
        print(f"downloaded {min(i + chunk_size, len(tickers))}/{len(tickers)} "
              f"（このチャンク {got}/{len(chunk)} 件・累計 {len(out)} 件）", file=sys.stderr)
    return out


def ichimoku(d: pd.DataFrame) -> pd.DataFrame:
    h, l, c = d["High"], d["Low"], d["Close"]
    x = pd.DataFrame(index=d.index)
    x["tenkan"] = (h.rolling(9).max() + l.rolling(9).min()) / 2
    x["kijun"] = (h.rolling(26).max() + l.rolling(26).min()) / 2
    x["senkou_a"] = ((x["tenkan"] + x["kijun"]) / 2).shift(26)
    x["senkou_b"] = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    x["chikou"] = c.shift(-26)  # 遅行スパン（当日終値を26日前へ）
    return x


def find_spike_days(d: pd.DataFrame) -> list[pd.Timestamp]:
    c, v, h, l, o = d["Close"], d["Volume"], d["High"], d["Low"], d["Open"]
    avg_vol = v.rolling(20).mean().shift(1)
    chg = c.pct_change() * 100
    body = (o - c) / (h - l).replace(0, np.nan)
    hi20 = h.rolling(20).max().shift(1)
    prior = (c / hi20 - 1) * 100

    cond = (v >= avg_vol * VOL_MULT) & ((chg <= DROP_PCT) | ((body >= BIG_BLACK_BODY) & (chg < 0))) & (prior <= PRIOR_DROP_PCT)
    recent = d.index[-LOOKBACK:]
    return [t for t in cond[cond].index if t in recent]


def evaluate(code: str, name: str, market: str, d: pd.DataFrame, spike: pd.Timestamp) -> dict:
    ich = ichimoku(d)
    last = d.index[-1]
    i_spike = d.index.get_loc(spike)
    days_since = len(d) - 1 - i_spike  # 急落日=0

    c_last = float(d["Close"].iloc[-1])
    o_last = float(d["Open"].iloc[-1])
    low_spike = float(d["Low"].loc[spike])
    after = d.iloc[i_spike + 1:]

    # 確認条件
    confirmed = bool((after["Close"] > after["Open"]).any()) if len(after) else False  # 急落後に陽線あり
    low_held = bool((after["Low"] >= low_spike).all()) if len(after) else True        # 急落日安値を割っていない
    above_tenkan = c_last > float(ich["tenkan"].iloc[-1])
    above_kijun = c_last > float(ich["kijun"].iloc[-1])
    cloud_top = float(np.nanmax([ich["senkou_a"].iloc[-1], ich["senkou_b"].iloc[-1]]))
    cloud_bot = float(np.nanmin([ich["senkou_a"].iloc[-1], ich["senkou_b"].iloc[-1]]))
    cloud_pos = "above" if c_last > cloud_top else ("in" if c_last >= cloud_bot else "below")
    # 遅行スパン: 当日終値 vs 26日前の終値
    chikou_up = c_last > float(d["Close"].iloc[-27]) if len(d) > 27 else False
    # 日柄: 急落日からの経過が基本数値±1
    hidara_hit = next((n for n in BASIC_NUMBERS if abs(days_since - n) <= 1), None)

    score = 0
    score += 2 if confirmed else 0
    score += 2 if low_held else -3
    score += 2 if above_tenkan else 0
    score += 1 if above_kijun else 0
    score += 1 if chikou_up else 0
    score += 1 if hidara_hit else 0
    score += {"above": 1, "in": 0, "below": -1}[cloud_pos]

    return {
        "code": code, "name": name, "market": market,
        "spike_date": spike.strftime("%Y-%m-%d"),
        "spike_drop_pct": round(float(d["Close"].pct_change().loc[spike] * 100), 2),
        "spike_vol_mult": round(float(d["Volume"].loc[spike] / d["Volume"].rolling(20).mean().shift(1).loc[spike]), 1),
        "spike_low": low_spike,
        "last_date": last.strftime("%Y-%m-%d"),
        "close": c_last,
        "days_since_spike": days_since,
        "confirmed_white_candle": confirmed,
        "spike_low_held": low_held,
        "above_tenkan": above_tenkan,
        "above_kijun": above_kijun,
        "cloud_position": cloud_pos,
        "chikou_bullish": chikou_up,
        "basic_number_hit": hidara_hit,
        "score": score,
        "avg_turnover_20d": float((d["Close"] * d["Volume"]).rolling(20).mean().iloc[-1]),
    }


def screen() -> tuple[list[dict], str]:
    uni = load_universe()
    print(f"universe: {len(uni)} 銘柄", file=sys.stderr)
    data = download(uni["code"].tolist())
    rate = len(data) / max(len(uni), 1)
    print(f"price data: {len(data)}/{len(uni)} 銘柄（{rate:.1%}）", file=sys.stderr)
    if rate < MIN_DATA_RATE:
        raise SystemExit(
            f"株価データの取得率が {rate:.1%}（{len(data)}/{len(uni)}）しかありません。"
            f"Yahoo 側でブロックされている可能性が高いため、監視リストは更新せず異常終了します。"
        )
    today = None
    found: list[dict] = []
    for _, row in uni.iterrows():
        d = data.get(row["code"])
        if d is None:
            continue
        if float(d["Close"].iloc[-1]) < MIN_PRICE:
            continue
        if float((d["Close"] * d["Volume"]).rolling(20).mean().iloc[-1]) < MIN_TURNOVER:
            continue
        today = d.index[-1].strftime("%Y-%m-%d")
        for spike in find_spike_days(d):
            found.append(evaluate(row["code"], row["name"], row["market"], d, spike))
    return found, today or dt.date.today().isoformat()


def merge_watchlist(new: list[dict], today: str) -> list[dict]:
    old = json.loads(OUT_JSON.read_text(encoding="utf-8")) if OUT_JSON.exists() else []
    by_key = {(x["code"], x["spike_date"]): x for x in old}
    for x in new:
        by_key[(x["code"], x["spike_date"])] = x  # 最新評価で上書き
    kept = []
    for x in by_key.values():
        age = (pd.Timestamp(today) - pd.Timestamp(x["spike_date"])).days
        if age <= EXPIRE_DAYS * 1.5:  # 暦日換算でざっくり 26 営業日
            kept.append(x)
    kept.sort(key=lambda x: (-x["score"], x["days_since_spike"]))
    return kept


def to_markdown(wl: list[dict], today: str) -> str:
    lines = [f"# 大出来高急落 × 一目 監視リスト（{today} 引け後）", "",
             "| コード | 銘柄 | 急落日 | 下落率 | 出来高倍率 | 経過日 | 陽線確認 | 安値維持 | 転換線 | 基準線 | 雲 | 遅行 | 日柄 | スコア |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    ok = lambda b: "○" if b else "－"
    for x in wl:
        lines.append(
            f"| {x['code']} | {x['name']} | {x['spike_date']} | {x['spike_drop_pct']}% | ×{x['spike_vol_mult']} | {x['days_since_spike']} "
            f"| {ok(x['confirmed_white_candle'])} | {ok(x['spike_low_held'])} | {ok(x['above_tenkan'])} | {ok(x['above_kijun'])} "
            f"| {x['cloud_position']} | {ok(x['chikou_bullish'])} | {x['basic_number_hit'] or '－'} | {x['score']} |")
    lines += ["", "- スコア: 陽線確認+2 / 安値維持+2（割れ-3） / 転換線上+2 / 基準線上+1 / 遅行好転+1 / 基本数値一致+1 / 雲上+1・雲下-1",
              "- 急落日から約26営業日で自動除外。投資助言ではありません。"]
    return "\n".join(lines)


if __name__ == "__main__":
    new, today = screen()
    DAILY_DIR.mkdir(exist_ok=True)
    (DAILY_DIR / f"{today}.json").write_text(json.dumps(new, ensure_ascii=False, indent=2), encoding="utf-8")
    wl = merge_watchlist(new, today)
    OUT_JSON.write_text(json.dumps(wl, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(to_markdown(wl, today), encoding="utf-8")
    print(f"{today}: 新規 {len(new)} 件 / 監視リスト {len(wl)} 件")
