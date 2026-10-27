"""調査用（単発実行）: ストップ高 → ストップ安 → ストップ高 の往復パターンを測る。

「ストップ高のあと W 営業日以内にストップ安」という並びを洗い出し、
そのストップ安から K 営業日以内に再びストップ高したかを集計する。
比較のため「ストップ高のあとストップ安が来なかった」場合の再ストップ高率も出す。

実行: SCREENER_MARKETS=グロース python screener/research_whipsaw.py
出力: screener/research/whipsaw-<市場>.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import limits  # noqa: E402
import screener as scr  # noqa: E402

PERIOD = os.environ.get("RESEARCH_PERIOD", "1y")
W = int(os.environ.get("WHIP_W", "10"))     # ストップ高からストップ安までの許容日数
K = int(os.environ.get("WHIP_K", "10"))     # ストップ安から再ストップ高までの観測日数
MIN_TURNOVER = 3e7


def main() -> None:
    markets = tuple(x for x in os.environ.get("SCREENER_MARKETS", "グロース").split(",") if x)
    scr.MARKETS = markets
    uni = scr.load_universe()
    print(f"universe: {len(uni)} / markets={markets}", file=sys.stderr)
    data = scr.download(uni["code"].tolist(), period=PERIOD)
    print(f"price data: {len(data)}/{len(uni)}", file=sys.stderr)

    whip: list[dict] = []        # ストップ高→ストップ安 の並び
    plain: list[dict] = []       # ストップ高のみ（W日以内にストップ安なし）
    for _, row in uni.iterrows():
        d = data.get(row["code"])
        if d is None or len(d) < 120:
            continue
        turn = (d["Close"] * d["Volume"]).rolling(20).mean()
        fl = limits.limit_flags(d)
        lu = fl["limit_up"].fillna(False).to_numpy()
        ld = fl["limit_down"].fillna(False).to_numpy()
        c = d["Close"]
        h, l = d["High"], d["Low"]
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean() / c * 100
        n = len(d)

        for i in range(70, n - 1):
            if not lu[i]:
                continue
            if pd.isna(turn.iloc[i]) or turn.iloc[i] < MIN_TURNOVER:
                continue
            # ストップ高 i の後 W 日以内の最初のストップ安
            j = next((k for k in range(i + 1, min(i + 1 + W, n)) if ld[k]), None)
            base = {"code": row["code"], "name": row["name"], "market": row["market"],
                    "up_date": str(d.index[i].date()),
                    "atr14_pct": None if pd.isna(atr.iloc[i]) else round(float(atr.iloc[i]), 2),
                    "turnover20_oku": round(float(turn.iloc[i]) / 1e8, 2),
                    "prior_lu_20d": int(lu[max(0, i - 20):i].sum()),
                    "prior_ld_20d": int(ld[max(0, i - 20):i].sum())}
            if j is None:
                # 比較群: ストップ高の翌日から K 日以内に再ストップ高したか
                nxt = lu[i + 1:min(i + 1 + K, n)]
                base["again"] = bool(nxt.any())
                base["days_to_again"] = int(np.argmax(nxt)) + 1 if nxt.any() else None
                plain.append(base)
                continue
            # ストップ安 j のあと K 日以内に再ストップ高したか
            nxt = lu[j + 1:min(j + 1 + K, n)]
            rec = dict(base)
            rec.update({
                "down_date": str(d.index[j].date()),
                "gap_days": j - i,
                "drawdown_pct": round(float(c.iloc[j]) / float(c.iloc[i]) * 100 - 100, 2),
                "close_vs_pre_up": round(float(c.iloc[j]) / float(c.iloc[i - 1]) * 100 - 100, 2),
                "again": bool(nxt.any()),
                "days_to_again": int(np.argmax(nxt)) + 1 if nxt.any() else None,
                "atr_at_down": None if pd.isna(atr.iloc[j]) else round(float(atr.iloc[j]), 2),
                "ld_count_20d": int(ld[max(0, j - 20):j + 1].sum()),
                "lu_count_20d": int(lu[max(0, j - 20):j + 1].sum()),
            })
            whip.append(rec)

    def rate(rows, label):
        if not rows:
            return None
        a = sum(1 for r in rows if r["again"])
        print(f"{label:<46} n={len(rows):>4}  再ストップ高 {a:>3} = {a/len(rows)*100:>5.1f}%", file=sys.stderr)
        return {"label": label, "n": len(rows), "again": a, "rate": round(a / len(rows) * 100, 1)}

    summary = [
        rate(plain, f"ストップ安を挟まない（ストップ高から{K}日以内に再ストップ高）"),
        rate(whip, f"ストップ高→ストップ安（ストップ安から{K}日以内に再ストップ高）"),
        rate([r for r in whip if r["gap_days"] <= 2], "  うち ストップ安までが2日以内"),
        rate([r for r in whip if r["gap_days"] >= 5], "  うち ストップ安までが5日以上"),
        rate([r for r in whip if (r.get("atr_at_down") or 0) >= 12], "  うち ストップ安時のATRが12%以上"),
        rate([r for r in whip if (r.get("atr_at_down") or 99) < 8], "  うち ストップ安時のATRが8%未満"),
        rate([r for r in whip if r["lu_count_20d"] >= 2], "  うち 20日内のストップ高が2回以上"),
        rate([r for r in whip if (r.get("close_vs_pre_up") or 0) > 0], "  うち ストップ高前の水準を維持"),
        rate([r for r in whip if (r.get("close_vs_pre_up") or 0) <= 0], "  うち ストップ高前の水準を下回った"),
        rate([r for r in whip if r["turnover20_oku"] <= 20], "  うち 売買代金20億円以下"),
    ]
    res = {"markets": list(markets), "period": PERIOD, "W": W, "K": K,
           "counts": {"whipsaw": len(whip), "plain": len(plain)},
           "summary": [x for x in summary if x],
           "days_to_again_hist": {str(k): int(v) for k, v in
                                  pd.Series([r["days_to_again"] for r in whip if r["days_to_again"]])
                                  .value_counts().sort_index().items()},
           "whipsaw": sorted(whip, key=lambda r: r["down_date"], reverse=True)[:300]}
    (ROOT / "research").mkdir(exist_ok=True)
    (ROOT / "research" / f"whipsaw-{'-'.join(markets)}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
