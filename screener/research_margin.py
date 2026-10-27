"""調査用（単発実行）: ストップ高の翌日継続に信用需給が効くかを検証する。

research_limitup.py が出したイベント一覧の各銘柄について株探から信用残の週次履歴を取り、
イベント直前の週の 貸借倍率・売残の4週増減・売り方の含み益率 を割り出して、
翌日継続（streak>=2）との関係を集計する。

実行: python screener/research_margin.py
出力: screener/research/margin.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
import screener as scr  # noqa: E402
from ichimoku_hosoda import margin as mg  # noqa: E402

MAX_EVENTS = int(os.environ.get("MARGIN_MAX_EVENTS", "260"))
SLEEP = 1.2


def pick_week(rows: list[dict], before: str) -> int | None:
    """before より前の最後の週のインデックス"""
    idx = [i for i, r in enumerate(rows) if r.get("date") and r["date"] < before]
    return idx[-1] if idx else None


def main() -> None:
    events: list[dict] = []
    for f in (ROOT / "research").glob("*.json"):
        if f.stem == "margin":
            continue
        events += json.loads(f.read_text(encoding="utf-8"))["events"]

    # 継続した事例を全部入れ、継続しなかった事例を同程度サンプリングして偏りを避ける
    cont = [e for e in events if e["streak_len"] >= 2]
    nocont = [e for e in events if e["streak_len"] == 1]
    rng = np.random.default_rng(0)
    if len(nocont) > MAX_EVENTS - len(cont):
        pick = rng.choice(len(nocont), size=max(0, MAX_EVENTS - len(cont)), replace=False)
        nocont = [nocont[i] for i in pick]
    target = cont + nocont
    print(f"検証対象 {len(target)} 件（継続 {len(cont)} / 非継続 {len(nocont)}）", file=sys.stderr)

    # 含み益率の推定には日足が必要なので、対象銘柄だけ取得
    codes = sorted({e["code"] for e in target})
    scr.CHUNK_SLEEP = 2
    data = scr.download(codes, period="1y", chunk_size=100)
    print(f"日足 {len(data)}/{len(codes)}", file=sys.stderr)

    cache: dict[str, dict] = {}
    out: list[dict] = []
    for n, e in enumerate(target, 1):
        code = e["code"]
        if code not in cache:
            try:
                cache[code] = mg.fetch_margin(code, weeks=60)
            except Exception as ex:
                cache[code] = {"error": str(ex)}
            time.sleep(SLEEP)
        m = cache[code]
        rec = {"code": code, "event_date": e["event_date"], "streak_len": e["streak_len"],
               "market": e["market"], "cont": e["streak_len"] >= 2}
        rows = [r for r in (m.get("rows") or []) if r.get("sell") is not None and r.get("buy") is not None]
        i = pick_week(rows, e["event_date"]) if rows else None
        if i is not None:
            r = rows[i]
            ratio = r.get("ratio")
            if ratio is None and r.get("sell"):
                ratio = r["buy"] / r["sell"]
            rec["ratio"] = round(ratio, 2) if ratio is not None else None
            rec["sell"] = r.get("sell")
            rec["buy"] = r.get("buy")
            if i >= 4 and rows[i - 4].get("sell"):
                rec["sell_chg4w"] = round(r["sell"] / rows[i - 4]["sell"] * 100 - 100, 1)
            d = data.get(code)
            if d is not None:
                # イベント前日までの日足で含み益率を推定する
                dd = d[d.index < pd.Timestamp(e["event_date"])]
                if len(dd) > 30:
                    try:
                        pl = mg.estimate_pl({"rows": rows[:i + 1]}, dd)
                        if pl:
                            if pl.get("sell"):
                                rec["sell_pl"] = pl["sell"]["pl"]
                            if pl.get("buy"):
                                rec["buy_pl"] = pl["buy"]["pl"]
                    except Exception:
                        pass
        else:
            rec["note"] = m.get("error", "該当週なし")[:60]
        out.append(rec)
        if n % 25 == 0:
            print(f"  {n}/{len(target)}", file=sys.stderr)

    def rate(rows, label):
        if not rows:
            return None
        c = sum(1 for r in rows if r["cont"])
        print(f"{label:<40} n={len(rows):>4}  継続 {c:>3} = {c/len(rows)*100:>5.1f}%", file=sys.stderr)
        return {"label": label, "n": len(rows), "cont": c, "rate": round(c / len(rows) * 100, 1)}

    have = [r for r in out if r.get("ratio") is not None]
    summary = [
        rate(have, "信用残が取れた全体（基準）"),
        rate([r for r in have if r["ratio"] < 1], "貸借倍率 < 1（売残 > 買残）"),
        rate([r for r in have if 1 <= r["ratio"] < 3], "貸借倍率 1〜3"),
        rate([r for r in have if r["ratio"] >= 10], "貸借倍率 10以上（買残が厚い）"),
        rate([r for r in have if (r.get("sell_chg4w") or -99) >= 20], "売残が4週で+20%以上"),
        rate([r for r in have if (r.get("sell_chg4w") or 99) <= -20], "売残が4週で-20%以下"),
        rate([r for r in have if (r.get("sell_pl") or 99) < 0], "売り方が含み損"),
        rate([r for r in have if (r.get("sell_pl") or 99) <= -20], "売り方の含み損が20%以上"),
        rate([r for r in have if (r.get("buy_pl") or 99) < 0], "買い方が含み損"),
    ]
    res = {"n_target": len(target), "n_with_margin": len(have),
           "summary": [x for x in summary if x], "rows": out}
    (ROOT / "research").mkdir(exist_ok=True)
    (ROOT / "research" / "margin.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
