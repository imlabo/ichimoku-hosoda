"""調査用（単発実行）: 過去のストップ高連発銘柄の「直前の姿」を集計する。

ストップ高の連続（1日・2日以上）を洗い出し、その直前日の特徴量を記録。
同じ特徴量を無作為抽出した平常日（ベースライン）でも集計し、
どの指標がストップ高連発の前兆として効くかを分位点で比較できるようにする。

実行: SCREENER_MARKETS=グロース python screener/research_limitup.py
出力: screener/research/<市場>.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import limits  # noqa: E402
import screener as scr  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "research"
PERIOD = os.environ.get("RESEARCH_PERIOD", "1y")
BASELINE_EVERY = 10          # 何営業日ごとにベースラインを採るか
MIN_TURNOVER = 3e7           # 調査段階ではゆるめ（3,000万円）
FEATURES = ["close", "turnover20", "vol_ratio", "vol_ratio_event", "gap_event",
            "ret_5d", "ret_20d", "ret_60d", "pos_20d", "pos_52w", "up_from_52w_low",
            "atr14_pct", "up_days_5", "prior_limitups_20d"]
FLAGS = ["above_tenkan", "above_kijun", "chikou_bullish", "sanyaku", "cloud_above"]


def prep(d: pd.DataFrame) -> dict:
    """銘柄ごとに一度だけ計算する系列をまとめる"""
    c, h, l, v, o = d["Close"], d["High"], d["Low"], d["Volume"], d["Open"]
    ich = scr.ichimoku(d)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    cloud_top = pd.concat([ich["senkou_a"], ich["senkou_b"]], axis=1).max(axis=1)
    cloud_bot = pd.concat([ich["senkou_a"], ich["senkou_b"]], axis=1).min(axis=1)
    return {
        "d": d, "c": c, "o": o, "v": v,
        "avg20": v.rolling(20).mean(),
        "turnover20": (c * v).rolling(20).mean(),
        "hi20": h.rolling(20).max(), "hi252": h.rolling(252, min_periods=60).max(),
        "lo252": l.rolling(252, min_periods=60).min(),
        "atr14": tr.rolling(14).mean(),
        "tenkan": ich["tenkan"], "kijun": ich["kijun"],
        "cloud_top": cloud_top, "cloud_bot": cloud_bot,
        "flags": limits.limit_flags(d),
    }


def features_at(p: dict, i: int) -> dict | None:
    """位置 i（参照日）の特徴量。i+1 が起点日になる想定。"""
    c, v, o = p["c"], p["v"], p["o"]
    n = len(c)
    if i < 70 or i >= n:
        return None
    px = float(c.iloc[i])
    avg20_prev = float(p["avg20"].iloc[i - 1]) if not np.isnan(p["avg20"].iloc[i - 1]) else None
    turnover = float(p["turnover20"].iloc[i]) if not np.isnan(p["turnover20"].iloc[i]) else None
    if not turnover or turnover < MIN_TURNOVER or not avg20_prev:
        return None

    def ratio(a, b):
        return None if (b is None or b == 0 or np.isnan(b)) else round(float(a) / float(b), 4)

    nxt = i + 1
    f = {
        "close": round(px, 1),
        "turnover20": round(turnover / 1e8, 2),                      # 億円
        "vol_ratio": round(float(v.iloc[i]) / avg20_prev, 2),
        "vol_ratio_event": (round(float(v.iloc[nxt]) / float(p["avg20"].iloc[i]), 2)
                            if nxt < n and p["avg20"].iloc[i] else None),
        "gap_event": (round(float(o.iloc[nxt]) / px * 100 - 100, 2) if nxt < n else None),
        "ret_5d": round(px / float(c.iloc[i - 5]) * 100 - 100, 2),
        "ret_20d": round(px / float(c.iloc[i - 20]) * 100 - 100, 2),
        "ret_60d": round(px / float(c.iloc[i - 60]) * 100 - 100, 2),
        "pos_20d": round(px / float(p["hi20"].iloc[i]) * 100 - 100, 2),
        "pos_52w": (round(px / float(p["hi252"].iloc[i]) * 100 - 100, 2)
                    if not np.isnan(p["hi252"].iloc[i]) else None),
        "up_from_52w_low": (round(px / float(p["lo252"].iloc[i]) * 100 - 100, 2)
                            if not np.isnan(p["lo252"].iloc[i]) else None),
        "atr14_pct": (round(float(p["atr14"].iloc[i]) / px * 100, 2)
                      if not np.isnan(p["atr14"].iloc[i]) else None),
        "up_days_5": int((c.iloc[i - 4:i + 1].to_numpy() > o.iloc[i - 4:i + 1].to_numpy()).sum()),
        "prior_limitups_20d": int(p["flags"]["limit_up"].iloc[max(0, i - 19):i + 1].sum()),
        "above_tenkan": bool(px > float(p["tenkan"].iloc[i])),
        "above_kijun": bool(px > float(p["kijun"].iloc[i])),
        "cloud_above": bool(px > float(p["cloud_top"].iloc[i])) if not np.isnan(p["cloud_top"].iloc[i]) else None,
        "chikou_bullish": bool(px > float(c.iloc[i - 26])) if i >= 26 else None,
    }
    f["sanyaku"] = bool(f["above_tenkan"] and f["above_kijun"]
                        and f["cloud_above"] and f["chikou_bullish"])
    f["ref_date"] = str(c.index[i].date())
    return f


def main() -> None:
    markets = tuple(x for x in os.environ.get("SCREENER_MARKETS", "プライム,スタンダード,グロース").split(",") if x)
    scr.MARKETS = markets
    uni = scr.load_universe()
    print(f"universe: {len(uni)} 銘柄 / markets={markets}", file=sys.stderr)
    data = scr.download(uni["code"].tolist(), period=PERIOD)
    print(f"price data: {len(data)}/{len(uni)}", file=sys.stderr)

    events: list[dict] = []
    baseline: list[dict] = []
    for _, row in uni.iterrows():
        d = data.get(row["code"])
        if d is None or len(d) < 120:
            continue
        p = prep(d)
        lu = p["flags"]["limit_up"].fillna(False)

        for start, length in limits.streaks(lu):
            f = features_at(p, start - 1)
            if f is None:
                continue
            f.update({"code": row["code"], "name": row["name"], "market": row["market"],
                      "event_date": str(d.index[start].date()), "streak_len": int(length),
                      "run_pct": round(float(d["Close"].iloc[min(start + length - 1, len(d) - 1)])
                                       / float(d["Close"].iloc[start - 1]) * 100 - 100, 2)})
            events.append(f)

        for i in range(70, len(d) - 1, BASELINE_EVERY):
            if lu.iloc[i + 1]:          # 翌日がストップ高の日はベースラインから除く
                continue
            f = features_at(p, i)
            if f is not None:
                f["market"] = row["market"]
                baseline.append(f)

    def quantiles(rows: list[dict]) -> dict:
        out = {"n": len(rows)}
        for k in FEATURES:
            vals = [r[k] for r in rows if r.get(k) is not None]
            if vals:
                q = np.percentile(vals, [10, 25, 50, 75, 90])
                out[k] = {"p10": round(q[0], 2), "p25": round(q[1], 2), "median": round(q[2], 2),
                          "p75": round(q[3], 2), "p90": round(q[4], 2)}
        for k in FLAGS:
            vals = [r[k] for r in rows if r.get(k) is not None]
            if vals:
                out[k] = {"true_rate": round(sum(vals) / len(vals), 3)}
        return out

    multi = [e for e in events if e["streak_len"] >= 2]
    res = {
        "markets": list(markets),
        "period": PERIOD,
        "universe": int(len(uni)),
        "with_data": len(data),
        "counts": {"events": len(events), "streak_ge2": len(multi), "baseline": len(baseline)},
        "quantiles": {
            "streak_ge2": quantiles(multi),
            "streak_1": quantiles([e for e in events if e["streak_len"] == 1]),
            "baseline": quantiles(baseline),
        },
        "streak_len_hist": {str(k): int(v) for k, v in
                            pd.Series([e["streak_len"] for e in events]).value_counts().sort_index().items()},
        "market_hist": {k: int(v) for k, v in
                        pd.Series([e["market"] for e in multi]).value_counts().items()},
        "events": sorted(events, key=lambda e: e["event_date"], reverse=True)[:400],
    }
    OUT_DIR.mkdir(exist_ok=True)
    tag = "-".join(markets)
    (OUT_DIR / f"{tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"events {len(events)} / streak>=2 {len(multi)} / baseline {len(baseline)}", file=sys.stderr)


if __name__ == "__main__":
    main()
