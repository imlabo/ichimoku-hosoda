"""1 つの足種（日足／週足／月足）に対する総合分析。"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import lines, levels as lv, nbars, numbers, swings, waves


@dataclass
class AnalysisResult:
    label: str
    df: pd.DataFrame
    L: pd.DataFrame
    pivots: list
    position: lines.Position
    events: list
    waves: list
    waveform: dict | None
    p_value: dict | None
    calc_clusters: list
    levels: list
    change_clusters: list
    nbar_summaries: list
    horizon: int
    index_ext: pd.DatetimeIndex
    unit: str = "日"
    extra_lag: int | None = None   # チャートに追加表示する遅行スパンの本数（日足 3、週足 9）

    # ---------- 出力 ----------
    def to_markdown(self, max_dates: int = 12, max_levels: int = 8, max_calc: int = 10) -> str:
        df, pos = self.df, self.position
        last = df.index[-1]
        c = df["Close"].iloc[-1]
        u = self.unit
        md = [f"## {self.label}（最終足 {last.date()}  終値 {c:,.1f}）", ""]
        md += ["### 0. 総合所見（暫定判断と検証条件）"] + [f"- {x}" for x in self.verdict()] + [""]

        # 直観的判断
        md += ["### 1. 直観的判断（均衡表の位置関係）",
               f"- 並び順: {pos.order}",
               f"- 転換線 vs 基準線: {_jp(pos.tenkan_vs_kijun)}／基準線の向き: {_jp(pos.kijun_slope)}／"
               f"遅行スパン vs 実線: {_jp(pos.chikou_vs_price)}／実線 vs 先行スパン帯: {_jp(pos.price_vs_cloud)}",
               f"- 三役: **{pos.sanyaku}**（※細田：三役好転・悪化は売買サインではなく「26 のスパンでのもみ合い放れ＋下げ（上げ）の影響力脱却」の示唆）"]
        i = last
        md.append(f"- 現在値: 転換線 {self.L.loc[i,'tenkan']:,.1f} / 基準線 {self.L.loc[i,'kijun']:,.1f} / "
                  f"先行(基準転換) {self.L.loc[i,'span_tk']:,.1f} / 先行(52) {self.L.loc[i,'span_52']:,.1f}")
        fut = self.L.loc[self.L.index > last, ["span_tk", "span_52"]].dropna()
        if len(fut):
            md.append(f"- 今後 {len(fut)}{u} の先行スパン帯: {fut.min().min():,.1f} 〜 {fut.max().max():,.1f}"
                      f"（26{u}先: 基準転換 {fut['span_tk'].iloc[-1]:,.1f} / 52 {fut['span_52'].iloc[-1]:,.1f}）")
        flats = []
        for name in ("kijun", "span_52"):
            s = self.L[name].loc[:last].dropna()
            k = 1
            while k < len(s) and s.iloc[-1 - k] == s.iloc[-1]:
                k += 1
            if k >= 4:
                flats.append(f"{lv._jp(name)} が {k}{u}横ばい（{s.iloc[-1]:,.1f}）→ 相場水準として機能している可能性")
        md += [f"- {x}" for x in flats]
        recent_ev = [e for e in self.events if (last - e["date"]).days <= 45 * (1 if u == '日' else 7 if u == '週' else 30)]
        if recent_ev:
            md.append("- 直近の各線の働き: " + "、".join(
                f"{e['date'].date()} {lv._jp(e['line'])}{e['kind']}" for e in recent_ev[-6:]))
        md.append("")

        # 波形
        md.append("### 2. 直近の波形と三波動構成")
        if self.waveform:
            w = self.waveform
            md.append(f"- 直近4極 A{w['A'].kind}{w['A'].date.date()} {w['A'].price:,.1f} → B {w['B'].price:,.1f} → "
                      f"C {w['C'].price:,.1f} → D {w['D'].price:,.1f}（{w['times'][0]}{u}・{w['times'][1]}{u}・{w['times'][2]}{u}）: **{w['form']}**")
            if "中間波動" in w["form"]:
                md.append("  - 中間波動＝原則もみ合いとして認識（転換をも示唆）。放れの方向を相場水準からの基本数値で確認する")
        good = [w for w in self.waves if "不成立" not in w.note]
        for w in good[:6]:
            ach = "・".join(k for k, v in w.achieved.items() if v) or "未達"
            md.append(f"- {'上げ' if w.direction=='up' else '下げ'}三波動 A {w.A.date.date()} {w.A.price:,.1f} / B {w.B.date.date()} {w.B.price:,.1f} / "
                      f"C {w.C.date.date()} {w.C.price:,.1f}  時間 AB={w.ab} BC={w.bc} AC={w.ac}{u}  "
                      f"計算値 E {w.calc['E']:,.1f} N {w.calc['N']:,.1f} V {w.calc['V']:,.1f} NT {w.calc['NT']:,.1f}  達成: {ach}")
        if self.p_value:
            p = self.p_value
            md.append(f"- P 計算値（内はらみ A高 {p['A'].price:,.1f} → B高 {p['B'].price:,.1f}、安値 {p['C'].price:,.1f}）: "
                      f"上 {p['P']:,.1f} / 下 {p['P_down']:,.1f} ― 下を割り込めば下げ三波動構成を考える")
        md.append("")

        # 価格の節目
        md.append("### 3. 価格の節目")
        md.append("**計算値の集中（±0.4%でまとめ）**")
        up = [x for x in self.calc_clusters if x["price"] > c][:max_calc // 2]
        dn = [x for x in self.calc_clusters if x["price"] <= c][-(max_calc // 2):]
        for x in sorted(up + dn, key=lambda z: -z["price"]):
            md.append(f"- {x['price']:,.1f}（{x['pct_from_close']:+.1f}%）{x['kinds']} ×{x['n']}{'  達成済' if x['achieved'] else ''}")
        md.append("")
        md.append("**相場水準候補（起点からの基本数値の現れ方で評価）**")
        for l in self.levels[:max_levels]:
            hits = "、".join(f"{h['n']}{u}目{h['what']}" for h in l.hits[:6])
            md.append(f"- {l.price:,.1f}（{(l.price/c-1)*100:+.1f}%） {l.source.split(' / ')[0]}  起点 {l.origin.date()}  "
                      f"[{hits or '—'}]  もみ合い限界: 上 {l.upper_limit:,.1f} / 下 {l.lower_limit:,.1f}")
        md.append("")

        # 時間の節目
        md.append(f"### 4. 時間の節目 ― 変化日候補（今後 {self.horizon}{u}、根拠の重なりが多い順）")
        md.append("※ 日付は営業日近似（祝日未考慮）。変化日は「当たる日」ではなく、その日の反応の有無を必ず確認するポイント。")
        n_bars = len(df)
        for cl in self.change_clusters[:max_dates]:
            when = cl["i_min"] - (n_bars - 1)
            tag = f"{when:+d}{u}" if when != 0 else "本日"
            md.append(f"- **{cl['date'].date()}**（{tag}）スコア {cl['score']:.1f}、根拠 {cl['n_sources']} 件 [{'/'.join(cl['kinds'])}]")
            for it in cl["items"][:4]:
                md.append(f"    - {it.detail}")
            if len(cl["items"]) > 4:
                md.append(f"    - …他 {len(cl['items'])-4} 件")
        md.append("")

        # 基本数値足
        if self.nbar_summaries:
            md.append("### 5. 基本数値足（陰連・陽連）")
            for s in self.nbar_summaries:
                cur = s["current"]
                if not cur:
                    continue
                hist = " ".join(f"{h['sign']}{h['len']}" for h in s["history"])
                line = f"- **{s['label']}**: 現在 {cur['sign']}{cur['len']}連（{cur['start'].date()}〜）  履歴: {hist}"
                if s["previous_opposite"]:
                    po = s["previous_opposite"]
                    line += f"  ← 直前の{po['sign']}{po['len']}連に対する対等 {po['len']} 本目に注目"
                md.append(line)
                th = s["future_thresholds"][:4]
                if th:
                    md.append("    - 今後の陰陽の分岐（終値がこの値より上なら陽線）: " +
                              "、".join(f"{t['bars_ahead']}本後 {t['ref_open']:,.1f}" for t in th))
            md.append("")
        return "\n".join(md)


    def verdict(self) -> list[str]:
        """細田の思考順序に沿った暫定判断：①動くか動かないか ②上げか下げか ③次に確認すべき節目。"""
        pos, df, u = self.position, self.df, self.unit
        c = float(df["Close"].iloc[-1])
        n = len(df)
        out = []
        # 動く／動かない
        kij = self.L["kijun"].loc[:df.index[-1]].dropna()
        flat_k = int((kij.iloc[-8:] == kij.iloc[-1]).sum()) if len(kij) >= 8 else 0
        recent = self.events[-10:]
        pierce = sum(1 for e in recent if "貫通" in e["kind"])
        func = sum(1 for e in recent if "機能" in e["kind"])
        form = self.waveform["form"] if self.waveform else ""
        if pos.tenkan_vs_kijun == "cross" or flat_k >= 5 or (pierce > func and pierce >= 3) or "中間波動" in form:
            move = "動かない相場（もみ合い）寄り"
            why = []
            if flat_k >= 5:
                why.append(f"基準線が{flat_k}{u}値段を変えていない")
            if pierce >= 3:
                why.append(f"直近で各線を貫通する動きが{pierce}回（押し戻りとして機能せず相場水準化）")
            if "中間波動" in form:
                why.append(f"直近波形が{form.split('（')[0]}")
            out.append(f"**{move}**" + (f"：{'、'.join(why)}" if why else "") + "。問題は「もみ合い放れはいつか」。相場水準からの基本数値日と、各線が収束する位置からの放れ方を見る")
        else:
            d = "上げ" if (pos.tenkan_vs_kijun == "up" and pos.chikou_vs_price in ("up", "touch")) else \
                "下げ" if (pos.tenkan_vs_kijun == "down" and pos.chikou_vs_price in ("down", "touch")) else "方向未確定"
            s1 = f"**動く相場・{d}寄り**：並び順 {pos.order}"
            if func:
                s1 += f"、直近で各線が押し目／戻りとして{func}回機能"
            out.append(s1 + "。問題は「いつ・いくらまで」。三波動の変化日と計算値で上限（下限）を測る")
        # 三役の扱い
        if pos.sanyaku != "中立":
            out.append(f"三役{pos.sanyaku}の状態だが、乗ってよいかは「残された時間と値幅」次第。直近の起点からの経過と、未達の計算値までの距離を見る")
        # 次の節目
        nxt = [cl for cl in self.change_clusters if cl["i_min"] >= n - 1]
        nxt.sort(key=lambda x: x["i_min"])
        if nxt:
            top = max(nxt[:6], key=lambda x: x["score"])
            out.append(f"次の変化日候補: {nxt[0]['date'].date()}（{nxt[0]['i_min']-(n-1):+d}{u}）、"
                       f"重なりが最も厚いのは {top['date'].date()}（スコア {top['score']:.1f}）。その日に高値・安値となるか、反応がないかを確認")
        up = [x for x in self.calc_clusters if x["price"] > c and not x["achieved"]]
        dn = [x for x in self.calc_clusters if x["price"] < c and not x["achieved"]]
        if up:
            x = min(up, key=lambda z: z["price"])
            out.append(f"上の未達計算値: {x['price']:,.1f}（{x['pct_from_close']:+.1f}%、{x['kinds']}）。E 達成なら勢い継続、N・V 止まりならもみ合いの範疇として警戒")
        if dn:
            x = max(dn, key=lambda z: z["price"])
            out.append(f"下の未達計算値: {x['price']:,.1f}（{x['pct_from_close']:+.1f}%、{x['kinds']}）")
        i = df.index[-1]
        tk, kj = self.L.loc[i, "tenkan"], self.L.loc[i, "kijun"]
        out.append(f"押し目／戻りの限界の目安: 転換線 {tk:,.1f}、基準線 {kj:,.1f}。"
                   + ("転換線・基準線を割れば「もみ合いか転換か」をまず考える" if c > kj else "転換線・基準線を上抜くまで上げを急いで買う理由はない"))
        if self.levels:
            lv0 = self.levels[0]
            out.append(f"最有力の相場水準: {lv0.price:,.1f}（{lv0.source.split(' / ')[0]}、起点 {lv0.origin.date()}）。"
                       f"この水準×2−極値 ＝ もみ合い限界 上 {lv0.upper_limit:,.1f} / 下 {lv0.lower_limit:,.1f}")
        return out


def _jp(s: str) -> str:
    return {"up": "上", "down": "下", "cross": "交差", "flat": "横ばい", "touch": "触れている",
            "above": "上", "below": "下", "inside": "帯の中", "n/a": "-"}.get(s, s)


def analyze(df: pd.DataFrame, label: str = "日足", unit: str = "日", horizon: int = 60,
            nbar_specs: tuple[tuple[int, str], ...] = ((3, "3日足"),),
            zz_thresholds: tuple[float, ...] = (0.03, 0.06, 0.12, 0.25), extra_lag: int | None = None) -> AnalysisResult:
    L = lines.compute(df)
    pivs = swings.pivots(df, zz_thresholds)
    # 進行中の極（最後）は暫定なので三波動の C には使うが rank は 1 のまま
    n = len(df)
    index_ext = numbers.future_dates(df.index, max(horizon + 5, 320))  # D 予測日が遠い大きな三波動にも日付を付ける
    pos = lines.position(df, L)
    events = lines.support_resistance_events(df, L)
    rw = waves.rule_waves(df, pivs)
    seen_ac = {(w.A.i, w.C.i) for w in rw}
    ws = rw + [w for w in waves.build_waves(pivs, df) if (w.A.i, w.C.i) not in seen_ac]
    wf = waves.classify_last_waveform(pivs)
    pv = waves.p_value(pivs)
    calc = waves.calc_levels(ws, float(df["Close"].iloc[-1]))
    lvs = lv.find_levels(df, L, pivs)
    # 定型三波動の A・B・C からの基本数値／対等数値（細田：重要な高値・安値からの基本数値・対等数値で変化日を定める）
    rule_pts, seen_pt = [], set()
    for w in rw:
        for p in (w.A, w.B, w.C):
            if p.i not in seen_pt:
                seen_pt.add(p.i)
                rule_pts.append(p)
    rule_cands = []
    for c in numbers.basic_change_dates(rule_pts, index_ext, n, horizon, unit=unit):
        c.weight *= 1.8
        c.detail = "【定型】" + c.detail
        rule_cands.append(c)
    for w in rw:   # 対等数値: B から AB、C から BC（もみ合い時間）、C から AB
        for base, length, lab in ((w.B, w.ab, "AB"), (w.C, w.bc, "BC"), (w.C, w.ab, "AB")):
            j = base.i + length - 1
            if n - 3 <= j < n + horizon:
                rule_cands.append(numbers.ChangeDate(j, index_ext[j], "equal", length, base.date, base.kind,
                    f"【定型】対等数値: {lab} {length}{unit} を {'高値' if base.kind=='H' else '安値'}{base.date.date()} から", 2.0))
    cands = (rule_cands + numbers.basic_change_dates(pivs[-10:], index_ext, n, horizon, unit=unit)
             + numbers.equal_change_dates(pivs, index_ext, n, horizon, unit=unit)
             + waves.wave_change_dates(ws, index_ext, n, horizon, unit=unit)
             + lv.level_change_dates(lvs, index_ext, n, horizon, unit=unit))
    clusters = numbers.cluster_dates(cands)
    nbs = [nbars.summarize(df, k, lab) for k, lab in nbar_specs]
    return AnalysisResult(label, df, L, pivs, pos, events, ws, wf, pv, calc, lvs, clusters, nbs, horizon, index_ext, unit, extra_lag)
