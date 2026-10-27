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
import os
import sys
import time
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import limits  # noqa: E402

# ---------------- 条件（ここを調整） ----------------
VOL_MULT = 3.0          # 当日出来高 >= 20日平均 × VOL_MULT（上昇・下落 共通）
# 下落側（セリング・クライマックス候補）
DROP_PCT = -7.0         # 当日終値 <= 前日終値 × (1 + DROP_PCT/100)
PRIOR_DROP_PCT = -15.0  # 直近20日高値からの下落率（これ以下で「下げ続けた後」とみなす）
BIG_BODY = 0.70         # 大陰線/大陽線: 実体 / (高値-安値) >= この値（騰落率条件とどちらか満たせばOK）
# 上昇側（出来高を伴うブレイク候補）
RISE_PCT = 7.0          # 当日終値 >= 前日終値 × (1 + RISE_PCT/100)
BREAKOUT_MARGIN = 0.0   # 当日終値 >= 直近20日高値 × (1 + BREAKOUT_MARGIN/100)（新高値ブレイク）
LOOKBACK = 5            # 何営業日前までの急落日を拾うか
MIN_TURNOVER = 1e8      # 20日平均売買代金 >= 1億円（薄商い除外）
MIN_PRICE = 100         # 低位株除外
EXPIRE_DAYS = 26        # 監視リストから外すまでの日柄
BASIC_NUMBERS = (9, 17, 26)
MIN_DATA_RATE = 0.5     # 株価取得率がこれ未満なら「0件」ではなく異常終了させる

# ストップ高・踏み上げ候補（squeeze）
# しきい値は screener/research_limitup.py による過去1年・全763イベントの集計から決めた。
# 全体の翌日継続率 16.5% に対し、ストップ高履歴あり 24.0% / ATR>=7% 21.8% /
# 当日出来高2倍未満 23.9% / 履歴×三役好転×ATR 30.6%。
SQ_ATR_PCT = 7.0            # ATR14 / 終値（%）
SQ_PRIOR_DAYS = 20          # ストップ高履歴を見る営業日数
SQ_QUIET_VOL = 2.0          # ストップ高当日の出来高 / 20日平均（これ未満だと継続しやすい）
SQ_RET20 = 15.0             # 20日リターン（%）
SQ_TURNOVER_MAX = 50e8      # 20日平均売買代金がこれを超えると継続率が落ちる（50億円）
SQ_TURNOVER_SMALL = 20e8    # 20億円以下は継続率が高め
SQ_LOOKBACK = 3             # 何営業日前までのストップ高を拾うか
SQ_EXPIRE_DAYS = 7          # squeeze は短命なので暦日7日で落とす
SQ_MARGIN_TOP = int(os.environ.get("SQ_MARGIN_TOP", "20"))  # 需給（信用残）を取る上位件数（シャードごと）

# 対象市場。環境変数 SCREENER_MARKETS でシャードごとに切り替える（カンマ区切り）。
# Yahoo の時間あたりリクエスト上限に当たるため、1ジョブ1市場に分けて並列実行する。
MARKETS = tuple(x for x in os.environ.get("SCREENER_MARKETS", "プライム").split(",") if x)
CHUNK_SLEEP = 5             # チャンク間の待ち（秒）
RATE_LIMIT_WAITS = (60, 180)  # 取得0件のときの待ち（秒）。Yahoo の制限は分単位で解けるため長めに
TIME_BUDGET_SEC = 25 * 60   # これを超えたら取得を打ち切る（Actions の無料枠を守る）
# ---------------------------------------------------

ROOT = Path(__file__).resolve().parent
OUT_JSON = ROOT / "watchlist.json"
OUT_MD = ROOT / "watchlist.md"
DAILY_DIR = ROOT / "daily"
SHARD_DIR = ROOT / "shards"
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
    c_market = _pick_col(df, "市場・商品区分", "市場")
    c_sector = next((c for c in df.columns if "33業種区分" in str(c)), None)
    ren = {c_code: "code", c_name: "name", c_market: "market"}
    if c_sector is not None:
        ren[c_sector] = "sector"
    df = df.rename(columns=ren)
    if "sector" not in df.columns:
        df["sector"] = ""
    df["code"] = df["code"].astype(str).str.strip()
    df = df[df["market"].astype(str).str.contains("|".join(MARKETS))]
    df = df[df["code"].str.fullmatch(r"[0-9A-Za-z]{4}")]
    return df[["code", "name", "market", "sector"]].reset_index(drop=True)


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
             chunk_size: int = 100) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    session = make_session()
    tickers = [f"{c}.T" for c in codes]
    started = time.monotonic()
    n_chunks = (len(tickers) + chunk_size - 1) // chunk_size

    for i in range(0, len(tickers), chunk_size):
        if time.monotonic() - started > TIME_BUDGET_SEC:
            print(f"時間切れ（{TIME_BUDGET_SEC // 60}分）で取得を打ち切ります", file=sys.stderr)
            break
        chunk = tickers[i:i + chunk_size]
        c_no = i // chunk_size + 1
        got = 0
        for attempt in range(len(RATE_LIMIT_WAITS) + 1):
            try:
                data = _yf_download(chunk, period, session)
            except Exception as e:
                print(f"  chunk {c_no}: 例外 {type(e).__name__}: {e}", file=sys.stderr)
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
            if got or attempt == len(RATE_LIMIT_WAITS):
                break
            wait = RATE_LIMIT_WAITS[attempt]
            print(f"  chunk {c_no}: 取得0件 → {wait}秒待って再試行 "
                  f"({attempt + 1}/{len(RATE_LIMIT_WAITS)})", file=sys.stderr)
            time.sleep(wait)
        print(f"downloaded {min(i + chunk_size, len(tickers))}/{len(tickers)} "
              f"（chunk {c_no}/{n_chunks}: {got}/{len(chunk)} 件・累計 {len(out)} 件）", file=sys.stderr)
        if CHUNK_SLEEP and i + chunk_size < len(tickers):
            time.sleep(CHUNK_SLEEP)
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


def find_spike_days(d: pd.DataFrame) -> list[tuple[pd.Timestamp, str]]:
    """直近 LOOKBACK 営業日の「出来高急増 × 急落／急騰」を (日付, 方向) で返す"""
    c, v, h, l, o = d["Close"], d["Volume"], d["High"], d["Low"], d["Open"]
    avg_vol = v.rolling(20).mean().shift(1)
    chg = c.pct_change() * 100
    rng = (h - l).replace(0, np.nan)
    black = (o - c) / rng   # 陰線の実体比
    white = (c - o) / rng   # 陽線の実体比
    hi20 = h.rolling(20).max().shift(1)
    prior = (c / hi20 - 1) * 100

    big_vol = v >= avg_vol * VOL_MULT
    down = big_vol & ((chg <= DROP_PCT) | ((black >= BIG_BODY) & (chg < 0))) & (prior <= PRIOR_DROP_PCT)
    up = big_vol & ((chg >= RISE_PCT) | ((white >= BIG_BODY) & (chg > 0))) & (c >= hi20 * (1 + BREAKOUT_MARGIN / 100))

    recent = set(d.index[-LOOKBACK:])
    hits = [(t, "down") for t in down[down].index if t in recent]
    hits += [(t, "up") for t in up[up].index if t in recent]
    return hits


def evaluate(code: str, name: str, market: str, d: pd.DataFrame,
             spike: pd.Timestamp, direction: str) -> dict:
    ich = ichimoku(d)
    last = d.index[-1]
    i_spike = d.index.get_loc(spike)
    days_since = len(d) - 1 - i_spike  # 急落／急騰日 = 0

    c_last = float(d["Close"].iloc[-1])
    low_spike = float(d["Low"].loc[spike])
    high_spike = float(d["High"].loc[spike])
    close_spike = float(d["Close"].loc[spike])
    after = d.iloc[i_spike + 1:]

    # 一目均衡表の位置関係
    above_tenkan = c_last > float(ich["tenkan"].iloc[-1])
    above_kijun = c_last > float(ich["kijun"].iloc[-1])
    cloud_top = float(np.nanmax([ich["senkou_a"].iloc[-1], ich["senkou_b"].iloc[-1]]))
    cloud_bot = float(np.nanmin([ich["senkou_a"].iloc[-1], ich["senkou_b"].iloc[-1]]))
    cloud_pos = "above" if c_last > cloud_top else ("in" if c_last >= cloud_bot else "below")
    chikou_up = c_last > float(d["Close"].iloc[-27]) if len(d) > 27 else False
    hidara_hit = next((n for n in BASIC_NUMBERS if abs(days_since - n) <= 1), None)

    # 方向ごとの確認条件
    if direction == "down":
        # 急落後に陽線が出たか / 急落日の安値を守っているか
        confirmed = bool((after["Close"] > after["Open"]).any()) if len(after) else False
        level_held = bool((after["Low"] >= low_spike).all()) if len(after) else True
    else:
        # 急騰日の終値以上を維持しているか / 急騰日の安値を守っているか
        confirmed = c_last >= close_spike
        level_held = bool((after["Low"] >= low_spike).all()) if len(after) else True

    score = 0
    score += 2 if confirmed else 0
    score += 2 if level_held else -3
    score += 2 if above_tenkan else 0
    score += 1 if above_kijun else 0
    score += 1 if chikou_up else 0
    score += 1 if hidara_hit else 0
    score += {"above": 1, "in": 0, "below": -1}[cloud_pos]

    return {
        "code": code, "name": name, "market": market,
        "direction": direction,
        "spike_date": spike.strftime("%Y-%m-%d"),
        "spike_chg_pct": round(float(d["Close"].pct_change().loc[spike] * 100), 2),
        "spike_vol_mult": round(float(d["Volume"].loc[spike] / d["Volume"].rolling(20).mean().shift(1).loc[spike]), 1),
        "spike_low": low_spike,
        "spike_high": high_spike,
        "spike_close": close_spike,
        "last_date": last.strftime("%Y-%m-%d"),
        "close": c_last,
        "days_since_spike": days_since,
        "confirmed": confirmed,
        "level_held": level_held,
        "above_tenkan": above_tenkan,
        "above_kijun": above_kijun,
        "cloud_position": cloud_pos,
        "chikou_bullish": chikou_up,
        "basic_number_hit": hidara_hit,
        "score": score,
        "avg_turnover_20d": float((d["Close"] * d["Volume"]).rolling(20).mean().iloc[-1]),
    }


def _atr_pct(d: pd.DataFrame, i: int, n: int = 14) -> float | None:
    h, l, c = d["High"], d["Low"], d["Close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    v = tr.rolling(n).mean().iloc[i]
    px = float(c.iloc[i])
    return None if (pd.isna(v) or not px) else round(float(v) / px * 100, 2)


def find_squeeze(d: pd.DataFrame) -> list[tuple[pd.Timestamp, str]]:
    """ストップ高の継続候補と、その前兆候補を (日付, 種別) で返す。

    continuation: 直近 SQ_LOOKBACK 営業日にストップ高があった（翌日の継続を狙う）
    precursor:    まだストップ高していないが、履歴と ATR が揃っている
    """
    flags = limits.limit_flags(d)
    lu = flags["limit_up"].fillna(False)
    recent = d.index[-SQ_LOOKBACK:]
    hits = [(t, "continuation") for t in lu[lu].index if t in recent]
    if hits:
        return hits

    i = len(d) - 1
    prior = int(lu.iloc[max(0, i - SQ_PRIOR_DAYS):i].sum())
    atr = _atr_pct(d, i)
    if prior >= 1 and atr is not None and atr >= SQ_ATR_PCT:
        return [(d.index[i], "precursor")]
    return []


def evaluate_squeeze(code: str, name: str, market: str, sector: str,
                     d: pd.DataFrame, spike: pd.Timestamp, kind: str) -> dict:
    ich = ichimoku(d)
    flags = limits.limit_flags(d)
    lu = flags["limit_up"].fillna(False)
    i = len(d) - 1
    i_spike = d.index.get_loc(spike)
    days_since = i - i_spike

    c_last = float(d["Close"].iloc[-1])
    avg20_prev = float(d["Volume"].rolling(20).mean().shift(1).iloc[i_spike])
    vol_ratio = round(float(d["Volume"].iloc[i_spike]) / avg20_prev, 2) if avg20_prev else None
    turnover = float((d["Close"] * d["Volume"]).rolling(20).mean().iloc[i])
    atr = _atr_pct(d, i)
    ret20 = round(c_last / float(d["Close"].iloc[i - 20]) * 100 - 100, 2) if i >= 20 else None
    # 起点日より前のストップ高回数。起点日自身は数えない（continuation で必ず1以上になるのを防ぐ）
    prior_lu = int(lu.iloc[max(0, i_spike - SQ_PRIOR_DAYS):i_spike].sum())

    above_tenkan = c_last > float(ich["tenkan"].iloc[-1])
    above_kijun = c_last > float(ich["kijun"].iloc[-1])
    cloud_top = float(np.nanmax([ich["senkou_a"].iloc[-1], ich["senkou_b"].iloc[-1]]))
    cloud_bot = float(np.nanmin([ich["senkou_a"].iloc[-1], ich["senkou_b"].iloc[-1]]))
    cloud_pos = "above" if c_last > cloud_top else ("in" if c_last >= cloud_bot else "below")
    chikou_up = c_last > float(d["Close"].iloc[-27]) if len(d) > 27 else False
    sanyaku = bool(above_tenkan and above_kijun and cloud_pos == "above" and chikou_up)
    hidara_hit = next((n for n in BASIC_NUMBERS if abs(days_since - n) <= 1), None)

    # 値動きスコア（過去1年の継続率の差をもとに配点）
    score = 0
    score += 3 if prior_lu >= 1 else 0                      # 24.0% vs 12.1%
    score += 2 if (atr or 0) >= SQ_ATR_PCT else 0           # 21.8% vs 10.8%（4%未満）
    score += 2 if (vol_ratio is not None and vol_ratio < SQ_QUIET_VOL) else 0   # 23.9%
    score += 2 if sanyaku else 0                            # 18.2% vs 15.4%、重ねると効く
    score += 1 if (ret20 or 0) >= SQ_RET20 else 0           # 19.0%
    score += 1 if turnover <= SQ_TURNOVER_SMALL else 0      # 17.7%
    score -= 2 if turnover > SQ_TURNOVER_MAX else 0         # 10.2%
    score -= 1 if "プライム" in str(market) else 0            # 9.5% vs 20.7%/17.0%
    score += {"above": 1, "in": 0, "below": -1}[cloud_pos]
    score += 1 if hidara_hit else 0

    return {
        "code": code, "name": name, "market": market, "sector": sector or "",
        "direction": "squeeze", "kind": kind,
        "spike_date": spike.strftime("%Y-%m-%d"),
        "spike_chg_pct": round(float(d["Close"].pct_change().iloc[i_spike] * 100), 2),
        "spike_vol_mult": vol_ratio,
        "limit_pct": round(float(flags["limit_pct"].iloc[i_spike]), 2),
        "last_date": d.index[-1].strftime("%Y-%m-%d"),
        "close": c_last,
        "days_since_spike": days_since,
        "prior_limitups_20d": prior_lu,
        "atr14_pct": atr,
        "ret_20d": ret20,
        "turnover20_oku": round(turnover / 1e8, 2),
        "above_tenkan": above_tenkan, "above_kijun": above_kijun,
        "cloud_position": cloud_pos, "chikou_bullish": chikou_up,
        "sanyaku": sanyaku, "basic_number_hit": hidara_hit,
        "quiet_volume": bool(vol_ratio is not None and vol_ratio < SQ_QUIET_VOL),
        "price_score": score, "score": score,
        "avg_turnover_20d": turnover,
    }


def add_margin(rows: list[dict], data: dict[str, pd.DataFrame], top: int = SQ_MARGIN_TOP) -> None:
    """上位候補にだけ信用残を取得して需給スコアを足す（株探を1銘柄ずつ叩くため件数を絞る）。

    売残 > 買残（貸借倍率 < 1）、売残の増加、売り方の含み損は踏み上げ余地を示す。
    含み損の推定は ichimoku_hosoda.margin.estimate_pl をそのまま使う。
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from ichimoku_hosoda import margin as mg
    except Exception as e:
        print(f"margin モジュールを読めません: {e}", file=sys.stderr)
        return

    targets = sorted(rows, key=lambda x: -x["price_score"])[:top]
    print(f"信用残を取得: {len(targets)} 銘柄", file=sys.stderr)
    for n, x in enumerate(targets, 1):
        try:
            m = mg.fetch_margin(x["code"], weeks=30)
            if "error" in m:
                x["margin_note"] = "取得失敗"
                continue
            rs = [r for r in m.get("rows", []) if r.get("sell") is not None and r.get("buy") is not None]
            if not rs:
                x["margin_note"] = "データなし"
                continue
            last = rs[-1]
            ratio = last.get("ratio")
            if ratio is None and last.get("sell"):
                ratio = last["buy"] / last["sell"]
            sell_chg = None
            if len(rs) >= 5 and rs[-5].get("sell"):
                sell_chg = round(last["sell"] / rs[-5]["sell"] * 100 - 100, 1)

            sell_pl = None
            try:
                pl = mg.estimate_pl(m, data.get(x["code"]))
                if pl and pl.get("sell"):
                    sell_pl = pl["sell"]["pl"]      # 売り方の含み益率（% / マイナスなら含み損）
            except Exception:
                pass

            # 配点は research_margin.py の実測（過去のストップ高177件・サンプリング重み補正後、
            # 基準15.4%）に合わせている。売残の「増加」は効くが、「売残>買残」「売り方の含み損」は
            # 効かなかったので点を与えない（含み損は表示だけ残す）。
            add = 0
            if sell_chg is not None:
                if sell_chg >= 20:
                    add += 2                        # 20.7%：売り方が積み上がっている
                elif sell_chg <= -20:
                    add -= 2                        # 8.9%：すでに買い戻されている
            if ratio is not None and 1 <= ratio < 3:
                add += 1                            # 19.8%
            x.update({"margin_date": last.get("date"),
                      "margin_sell": last.get("sell"), "margin_buy": last.get("buy"),
                      "margin_ratio": round(ratio, 2) if ratio is not None else None,
                      "margin_sell_chg4w": sell_chg,
                      "margin_sell_pl": sell_pl,
                      "margin_score": add,
                      "score": x["price_score"] + add})
        except Exception as e:
            x["margin_note"] = f"例外: {type(e).__name__}"
        finally:
            time.sleep(1.2)                         # 株探への負荷を抑える
        if n % 10 == 0:
            print(f"  信用残 {n}/{len(targets)}", file=sys.stderr)


def add_theme(rows: list[dict]) -> None:
    """同業種で複数ヒットしていればテーマ化の兆候として加点"""
    from collections import Counter
    cnt = Counter(x.get("sector", "") for x in rows if x.get("sector"))
    for x in rows:
        n = cnt.get(x.get("sector", ""), 0)
        x["sector_hits"] = n
        if n >= 2:
            x["score"] = x.get("score", x["price_score"]) + 1
            x["theme"] = True


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
    sq: list[dict] = []
    for _, row in uni.iterrows():
        d = data.get(row["code"])
        if d is None:
            continue
        if float(d["Close"].iloc[-1]) < MIN_PRICE:
            continue
        if float((d["Close"] * d["Volume"]).rolling(20).mean().iloc[-1]) < MIN_TURNOVER:
            continue
        today = d.index[-1].strftime("%Y-%m-%d")
        for spike, direction in find_spike_days(d):
            found.append(evaluate(row["code"], row["name"], row["market"], d, spike, direction))
        for spike, kind in find_squeeze(d):
            sq.append(evaluate_squeeze(row["code"], row["name"], row["market"],
                                       row.get("sector", ""), d, spike, kind))

    # 同一銘柄で複数日ヒットするので、起点日が最新のものだけ残す
    if sq:
        latest: dict[str, dict] = {}
        for x in sq:
            cur = latest.get(x["code"])
            if cur is None or x["spike_date"] > cur["spike_date"]:
                latest[x["code"]] = x
        sq = list(latest.values())

    if sq:
        add_theme(sq)
        add_margin(sq, data)
        print(f"squeeze: {len(sq)} 件（継続 "
              f"{sum(1 for x in sq if x['kind'] == 'continuation')} / 前兆 "
              f"{sum(1 for x in sq if x['kind'] == 'precursor')}）", file=sys.stderr)
    found += sq
    return found, today or dt.date.today().isoformat()


def _normalize(x: dict) -> dict:
    """旧フォーマット（方向なし・spike_drop_pct）の記録を読めるようにする"""
    x.setdefault("direction", "down")
    if "spike_chg_pct" not in x and "spike_drop_pct" in x:
        x["spike_chg_pct"] = x.pop("spike_drop_pct")
    if "confirmed" not in x and "confirmed_white_candle" in x:
        x["confirmed"] = x.pop("confirmed_white_candle")
    if "level_held" not in x and "spike_low_held" in x:
        x["level_held"] = x.pop("spike_low_held")
    return x


def merge_watchlist(new: list[dict], today: str) -> list[dict]:
    old = json.loads(OUT_JSON.read_text(encoding="utf-8")) if OUT_JSON.exists() else []
    key = lambda x: (x["code"], x["spike_date"], x["direction"])
    by_key = {key(_normalize(x)): x for x in old}
    for x in new:
        by_key[key(x)] = x  # 最新評価で上書き
    kept = []
    for x in by_key.values():
        age = (pd.Timestamp(today) - pd.Timestamp(x["spike_date"])).days
        limit = SQ_EXPIRE_DAYS if x["direction"] == "squeeze" else EXPIRE_DAYS * 1.5
        if age <= limit:  # 暦日換算。squeeze は短命なので短く
            kept.append(x)
    # squeeze は銘柄ごとに最新の起点だけ残す（前日までの記録が残って重複するのを防ぐ）
    newest: dict[str, str] = {}
    for x in kept:
        if x["direction"] != "squeeze":
            continue
        if x["spike_date"] > newest.get(x["code"], ""):
            newest[x["code"]] = x["spike_date"]
    kept = [x for x in kept
            if x["direction"] != "squeeze" or x["spike_date"] == newest[x["code"]]]

    kept.sort(key=lambda x: (-x["score"], x["days_since_spike"]))
    return kept


SCORE_NOTE = ("スコア: 確認+2 / 水準維持+2（割れ-3） / 転換線上+2 / 基準線上+1 / "
              "遅行好転+1 / 基本数値一致+1 / 雲上+1・雲下-1")
CONFIRM_LABEL = {"down": "陽線確認", "up": "終値維持"}


def _table(rows: list[dict], direction: str) -> list[str]:
    head = "急落日" if direction == "down" else "急騰日"
    chg = "下落率" if direction == "down" else "上昇率"
    out = [f"| コード | 銘柄 | {head} | {chg} | 出来高倍率 | 経過日 | "
           f"{CONFIRM_LABEL[direction]} | 安値維持 | 転換線 | 基準線 | 雲 | 遅行 | 日柄 | スコア |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    ok = lambda b: "○" if b else "－"
    for x in rows:
        out.append(
            f"| {x['code']} | {x['name']} | {x['spike_date']} | {x['spike_chg_pct']}% "
            f"| ×{x['spike_vol_mult']} | {x['days_since_spike']} "
            f"| {ok(x['confirmed'])} | {ok(x['level_held'])} | {ok(x['above_tenkan'])} | {ok(x['above_kijun'])} "
            f"| {x['cloud_position']} | {ok(x['chikou_bullish'])} | {x['basic_number_hit'] or '－'} | {x['score']} |")
    if len(out) == 2:
        out = ["該当なし"]
    return out


def _squeeze_table(rows: list[dict]) -> list[str]:
    if not rows:
        return ["該当なし"]
    out = ["| コード | 銘柄 | 市場 | 種別 | 起点日 | 制限値幅 | 当日出来高 | ATR | 20日騰落 | 代金 | 履歴 | 三役 | 貸借倍率 | 売残4週 | 売り方損益 | 業種 | スコア |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    ok = lambda b: "○" if b else "－"
    kind = {"continuation": "継続", "precursor": "前兆"}
    for x in rows:
        mk = str(x.get("market", "")).replace("プライム（内国株式）", "プライム") \
            .replace("スタンダード（内国株式）", "スタンダード").replace("グロース（内国株式）", "グロース")
        out.append(
            f"| {x['code']} | {x['name']} | {mk[:6]} | {kind.get(x.get('kind'), '')} "
            f"| {x['spike_date']} | {x.get('limit_pct', '－')}% "
            f"| ×{x.get('spike_vol_mult') or '－'} | {x.get('atr14_pct') or '－'}% "
            f"| {x.get('ret_20d') or '－'}% | {x.get('turnover20_oku') or '－'}億 "
            f"| {x.get('prior_limitups_20d', 0)}回 | {ok(x.get('sanyaku'))} "
            f"| {x.get('margin_ratio') if x.get('margin_ratio') is not None else '－'} "
            f"| {str(x.get('margin_sell_chg4w')) + '%' if x.get('margin_sell_chg4w') is not None else '－'} "
            f"| {str(x.get('margin_sell_pl')) + '%' if x.get('margin_sell_pl') is not None else '－'} "
            f"| {str(x.get('sector', ''))[:6]}{'★' if x.get('theme') else ''} | {x['score']} |")
    return out


def to_markdown(wl: list[dict], today: str) -> str:
    down = [x for x in wl if x["direction"] == "down"]
    up = [x for x in wl if x["direction"] == "up"]
    sq = [x for x in wl if x["direction"] == "squeeze"]
    lines = [f"# 大出来高 × 一目 監視リスト（{today} 引け後）", ""]

    lines += [f"## ストップ高・踏み上げ候補 {len(sq)} 件", ""]
    lines += _squeeze_table(sq)
    lines += ["", f"## 下落（底打ち候補） {len(down)} 件", ""]
    lines += _table(down, "down")
    lines += ["", f"## 上昇（ブレイク候補） {len(up)} 件", ""]
    lines += _table(up, "up")
    lines += ["",
              "### ストップ高候補のスコア",
              "履歴（直近20日にストップ高）+3 / ATR14が7%以上 +2 / 当日出来高が20日平均の2倍未満 +2 / "
              "三役好転 +2 / 20日騰落+15%以上 +1 / 売買代金20億円以下 +1 / 同50億円超 -2 / プライム -1 / "
              "雲上+1・雲下-1 / 基本数値一致 +1",
              "需給: 売残が4週で+20%以上 +2 / 同-20%以下 -2 / 貸借倍率1〜3 +1"
              "（売残177件の実測。基準15.4%に対し 売残増20.7% / 売残減8.9% / 倍率1〜3 19.8%。"
              "『売残>買残』6.6%・『売り方の含み損』14.4% は効かなかったので配点せず、表示のみ）",
              "テーマ: 同業種で2件以上ヒット +1（業種名に★）",
              "",
              "配点は過去1年・ストップ高763件の翌日継続率から決めています（全体16.5% → 履歴あり24.0% / "
              "ATR7%以上21.8% / 当日出来高2倍未満23.9% / 履歴×三役×ATR 30.6%）。"
              "スコアは確度ではなく条件の重なり具合を表すものとして見てください。",
              "",
              f"- {SCORE_NOTE}",
              "- 下落側の確認＝急落後に陽線が出たか。上昇側の確認＝急騰日の終値以上を維持しているか",
              "- 急落／急騰日から約26営業日、ストップ高候補は7日で自動除外。投資助言ではありません。"]
    return "\n".join(lines)


def load_shards() -> tuple[list[dict], str]:
    """市場ごとのシャード結果を読み込んで結合する"""
    found: list[dict] = []
    days: list[str] = []
    files = sorted(SHARD_DIR.glob("*.json")) if SHARD_DIR.exists() else []
    if not files:
        raise SystemExit("シャードの結果がありません（screener/shards/*.json）")
    for f in files:
        obj = json.loads(f.read_text(encoding="utf-8"))
        found += obj.get("found", [])
        if obj.get("today"):
            days.append(obj["today"])
        print(f"shard {f.stem}: {len(obj.get('found', []))} 件 / {obj.get('today')}", file=sys.stderr)
    return found, max(days) if days else dt.date.today().isoformat()


if __name__ == "__main__":
    # SCREENER_MODE: all（単独実行）/ shard（市場ごとに取得して中間ファイルへ）/ merge（シャードを結合）
    mode = os.environ.get("SCREENER_MODE", "all")

    if mode == "merge":
        new, today = load_shards()
    else:
        new, today = screen()

    if mode == "shard":
        SHARD_DIR.mkdir(exist_ok=True)
        tag = "-".join(MARKETS) or "all"
        (SHARD_DIR / f"{tag}.json").write_text(
            json.dumps({"today": today, "markets": list(MARKETS), "found": new},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"shard {tag}: {len(new)} 件 / {today}", file=sys.stderr)
        raise SystemExit(0)

    DAILY_DIR.mkdir(exist_ok=True)
    (DAILY_DIR / f"{today}.json").write_text(json.dumps(new, ensure_ascii=False, indent=2), encoding="utf-8")
    wl = merge_watchlist(new, today)
    OUT_JSON.write_text(json.dumps(wl, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(to_markdown(wl, today), encoding="utf-8")
    nd = sum(1 for x in new if x["direction"] == "down")
    nq = sum(1 for x in new if x["direction"] == "squeeze")
    nu = len(new) - nd - nq
    wd = sum(1 for x in wl if x["direction"] == "down")
    wq = sum(1 for x in wl if x["direction"] == "squeeze")
    wu = len(wl) - wd - wq
    print(f"{today}: 新規 {len(new)} 件（下落 {nd} / 上昇 {nu} / ストップ高 {nq}）"
          f" / 監視リスト {len(wl)} 件（下落 {wd} / 上昇 {wu} / ストップ高 {wq}）")
