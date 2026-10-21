"""著書の「思考の手順」に沿った確認項目。

1. 過ぎた変化日の結果（反応＝高値・安値になったか／反応なし＝その方向の動きが強い）
2. 均衡表の先読み（実線が高値・安値を更新しないと仮定した転換線・基準線の値。先行スパンは既に分かっている）
3. 次の変化日までの検証条件（「いつまでに・いくら以上（以下）なら前提どおり／割れば前提が誤り」）
4. 準備構成（重要な安値（高値）からの基本数値日にその安値（高値）を割らないか。割れば失敗）
5. 三役の内訳（①転換線×基準線・②遅行スパン×実線＝26 のスパンでのもみ合い放れ、③実線×先行スパン＝下げ（上げ）の影響力脱却）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import numbers, waves as wavemod
from .lines import KIJUN, SHIFT, TENKAN


def projected_lines(df: pd.DataFrame, k: int = 60) -> pd.DataFrame:
    """実線が今後、終値のまま高値・安値を更新しないと仮定した転換線・基準線の値（k 本先まで）。
    i 本先の転換線 = 直近 (9−i) 本の高値・安値と仮定値の半値（9 本を過ぎれば終値に収束）。"""
    hi, lo, c = df["High"].values, df["Low"].values, float(df["Close"].iloc[-1])
    out = []
    for i in range(1, k + 1):
        row = {}
        for name, w in (("tenkan", TENKAN), ("kijun", KIJUN)):
            m = w - i
            if m > 0:
                h, l = max(hi[-m:].max(), c), min(lo[-m:].min(), c)
            else:
                h = l = c
            row[name] = (h + l) / 2
        out.append(row)
    return pd.DataFrame(out, index=range(1, k + 1))


def _reaction(df: pd.DataFrame, pivs: list, j: int) -> dict:
    """変化日 j（±1 本）の反応。重要な極（rank>=2）があれば高値・安値、無ければ反応なしと前後の動き。"""
    n = len(df)
    for p in pivs:
        if abs(p.i - j) <= 1 and p.rank >= 2:
            return {"kind": "high" if p.kind == "H" else "low", "price": float(p.price), "i": int(p.i)}
    c = df["Close"].values
    a, b = max(0, j - 5), min(n - 1, j + 5)
    move = "up" if c[b] > c[a] else "down"
    return {"kind": "none", "move": move}


def past_change_results(df, pivs, ws, levels, index_ext, unit: str, lookback: int = 80, top: int = 8) -> list[dict]:
    """直近 lookback 本の間にあった変化日候補（重なりの大きい順）と、その日の反応。
    最後の 4 本は極が未確定なので評価しない。"""
    n = len(df)
    lo_i = max(30, n - lookback)
    span = n - 4 - lo_i
    if span <= 3:
        return []
    fake_n = lo_i + 3                      # 生成関数の「n_bars−3 ≦ j」を過去にずらす
    piv_before = [p for p in pivs if p.i < n - 4]
    cands = (numbers.basic_change_dates(piv_before[-14:], index_ext, fake_n, span, unit=unit)
             + numbers.equal_change_dates(piv_before, index_ext, fake_n, span, unit=unit)
             + wavemod.wave_change_dates(ws, index_ext, fake_n, span, unit=unit))
    cands = [c for c in cands if lo_i <= c.i < n - 4 and c.origin < df.index[c.i]]
    cl = numbers.cluster_dates(cands)
    pick = []
    for x in sorted(cl, key=lambda x: -x["score"]):      # 重なりの大きい順に、3 本以上離れたものだけ
        if all(abs(x["i_min"] - y["i_min"]) >= 3 for y in pick):
            pick.append(x)
        if len(pick) >= top:
            break
    cl = pick
    out = []
    for c in sorted(cl, key=lambda x: x["i_min"]):
        j = int(c["i_min"])
        r = _reaction(df, pivs, j)
        out.append({"i": j, "date": df.index[j], "score": float(c["score"]), "reaction": r,
                    "details": [x.detail for x in c["items"][:3]]})
    return out


def prep_check(df: pd.DataFrame, pivs: list, lookback: int = 160) -> dict | None:
    """準備構成：直近の重要な安値（高値）D を起点に、基本数値日（9・17・26・33・42…）まで D を割らないか。
    割れば「準備構成の失敗」（下げ（上げ）の三波動を明確にする）。"""
    n = len(df)
    s0 = max(0, n - lookback)
    major = [p for p in pivs if p.rank >= 3 and s0 <= p.i < n - 2]
    if not major:
        return None
    D = major[-1]
    bottom = D.kind == "L"
    hi, lo = df["High"].values, df["Low"].values
    after = lo[D.i + 1:] if bottom else hi[D.i + 1:]
    elapsed = n - D.i
    BN = (9, 17, 26, 33, 42, 51, 65, 76)
    passed = [b for b in BN if b <= elapsed]
    nxt = [b for b in BN if b > elapsed][:2]
    brk = None
    if len(after):
        idx = np.where(after < D.price)[0] if bottom else np.where(after > D.price)[0]
        if len(idx):
            jb = D.i + 1 + int(idx[0])
            brk = {"i": jb, "date": df.index[jb], "n": jb - D.i + 1}
    return {"side": "bottom" if bottom else "top", "i": D.i, "date": D.date, "price": float(D.price),
            "elapsed": int(elapsed), "status": "failed" if brk else ("early" if elapsed < 9 else "holding"),
            "broke": brk, "passed": passed, "next": nxt}


def sanyaku_detail(df: pd.DataFrame, L: pd.DataFrame) -> dict:
    """三役の内訳と、それぞれが今の状態になってから何本たったか。"""
    idx = df.index
    c = df["Close"]
    tk, kj = L["tenkan"].reindex(idx), L["kijun"].reindex(idx)
    top = np.maximum(L["span_tk"].reindex(idx), L["span_52"].reindex(idx))
    bot = np.minimum(L["span_tk"].reindex(idx), L["span_52"].reindex(idx))
    s1 = np.sign(tk - kj)
    # 遅行スパン（当日終値）と 26 本前の実線
    past = c.shift(SHIFT - 1)
    s2 = np.sign(c - past)
    s3 = pd.Series(np.where(c > top, 1, np.where(c < bot, -1, 0)), index=idx)

    def since(s):
        s = s.dropna()
        if not len(s):
            return 0, 0
        v = s.iloc[-1]
        k = 0
        for x in s.iloc[::-1]:
            if x == v:
                k += 1
            else:
                break
        return int(v), k
    return {"tk_kj": since(s1), "chikou": since(s2), "cloud": since(s3)}


def _span_note(c: float, span) -> str:
    if not span:
        return ""
    lo, hi = span
    if c > hi:
        return f"。その日の先行スパン（{lo:,.1f}〜{hi:,.1f}）は大きな押し目の限界。割れば上げの前提が崩れる"
    if c < lo:
        return f"。その日の先行スパン（{lo:,.1f}〜{hi:,.1f}）を上抜ければ下げの影響力を脱する"
    return f"。実線はその日の先行スパン（{lo:,.1f}〜{hi:,.1f}）の中。上下どちらへ抜けるかで方向が決まる"


def conditions(res, proj: pd.DataFrame, prep: dict | None) -> list[str]:
    """次の変化日までの検証条件（文章）。"""
    df, L, u, n = res.df, res.L, res.unit, len(res.df)
    c = float(df["Close"].iloc[-1])
    nxt = sorted([cl for cl in res.change_clusters if cl["i_min"] >= n], key=lambda x: x["i_min"])
    if not nxt:
        return []
    top = max(nxt[:5], key=lambda x: x["score"])
    k = int(top["i_min"] - (n - 1))
    d = top["date"].date()
    row = proj.loc[min(k, len(proj))] if k >= 1 else proj.loc[1]
    tk, kj = float(row["tenkan"]), float(row["kijun"])
    j = n - 1 + k
    span = None
    if 0 <= j < len(L.index) or True:
        try:
            dt = res.index_ext[j]
            a, b = float(L["span_tk"].get(dt, np.nan)), float(L["span_52"].get(dt, np.nan))
            if not (np.isnan(a) or np.isnan(b)):
                span = (min(a, b), max(a, b))
        except Exception:  # noqa: BLE001
            pass
    out = []
    pos = res.position
    if pos.tenkan_vs_kijun == "up" or c >= max(tk, kj):
        out.append(f"重要な変化日 {d}（{k:+d}{u}）まで：転換線の見込み {tk:,.1f}・基準線の見込み {kj:,.1f}（値動きがない場合の値）。"
                   f"この両線を押し目に保てば上げの前提どおり。変化日に高値となるか、反応なく続伸（延長・加速）かを確認")
        out.append(f"基準線 {kj:,.1f} を割り込めば「もみ合いか転換か」をまず考える" + _span_note(c, span))
    elif pos.tenkan_vs_kijun == "down" or c <= min(tk, kj):
        out.append(f"重要な変化日 {d}（{k:+d}{u}）まで：転換線の見込み {tk:,.1f}・基準線の見込み {kj:,.1f}（値動きがない場合の値）。"
                   f"この両線に戻りを抑えられれば下げの前提どおり。変化日に安値となるか、反応なく続落かを確認")
        out.append(f"転換線・基準線（{max(tk, kj):,.1f}）を上抜けば下げの前提を見直す" + _span_note(c, span))
    else:
        out.append(f"重要な変化日 {d}（{k:+d}{u}）まで：実線は転換線・基準線（{min(tk, kj):,.1f}〜{max(tk, kj):,.1f}）に挟まれている。"
                   f"どちらの線から放れるかを待つ（上なら上げ、下ならもみ合いの継続か下げ）")
    if prep and prep.get("status") == "failed":
        side = "安値" if prep["side"] == "bottom" else "高値"
        b = prep["broke"]
        out.append(f"準備構成の失敗：{side} {prep['price']:,.1f}（{prep['date'].date()}）を {b['n']}{u}目（{b['date'].date()}）に"
                   f"{'割り込んだ' if side == '安値' else '上抜いた'}。{'下げ' if side == '安値' else '上げ'}の三波動をより明確に考える")
    if prep and prep.get("status") in ("holding", "early"):
        bottom = prep["side"] == "bottom"
        nn = prep["next"][0] if prep["next"] else None
        if bottom:
            msg = (f"準備構成（底値固め）：重要な安値 {prep['price']:,.1f}（{prep['date'].date()}）から {prep['elapsed']}{u}目で、まだ割っていない。"
                   + (f"{nn}{u}目まで割らなければ底値固めが続く。その後、間を置かずに均衡表が好転すれば中間波動からの上放れ。割れば準備構成の失敗（下げの三波動を考える）" if nn else ""))
        else:
            msg = (f"準備構成（天井固め）：重要な高値 {prep['price']:,.1f}（{prep['date'].date()}）から {prep['elapsed']}{u}目で、まだ超えていない。"
                   + (f"{nn}{u}目まで超えなければ天井固め（下げの準備構成）が続く。超えれば失敗（上げの三波動を考える）" if nn else ""))
        out.append(msg)
    return out
