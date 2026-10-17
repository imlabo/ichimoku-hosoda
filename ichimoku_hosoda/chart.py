"""均衡表チャート（PNG）。極・変化日・相場水準を書き込む。"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .engine import AnalysisResult  # noqa: E402

try:  # 日本語フォント（あれば）
    import matplotlib_fontja  # noqa: F401
except Exception:  # pragma: no cover
    try:
        import japanize_matplotlib  # noqa: F401
    except Exception:
        pass

C = {"up": "#c0392b", "down": "#1f5f8b", "tenkan": "#d35400", "kijun": "#2c3e50",
     "cloud": "#8e9aaf", "chikou": "#27ae60", "level": "#7d3c98", "date": "#b9770e"}


def plot(res: AnalysisResult, path: str, bars: int = 260, title: str | None = None,
         max_dates: int = 6, max_levels: int = 6) -> str:
    df = res.df.iloc[-bars:]
    L = res.L.loc[df.index[0]:]
    fig, ax = plt.subplots(figsize=(15, 7.5), dpi=130)
    x = mdates.date2num(df.index.to_pydatetime())
    w = 0.6 * np.median(np.diff(x)) if len(x) > 1 else 0.6
    for xi, (o, h, l, c) in zip(x, df[["Open", "High", "Low", "Close"]].values):
        col = C["up"] if c >= o else C["down"]
        ax.vlines(xi, l, h, color=col, lw=0.7)
        ax.add_patch(plt.Rectangle((xi - w / 2, min(o, c)), w, max(abs(c - o), 1e-9), color=col, lw=0))
    xl = mdates.date2num(L.index.to_pydatetime())
    ax.plot(xl, L["tenkan"], color=C["tenkan"], lw=1.1, label="転換線(9)")
    ax.plot(xl, L["kijun"], color=C["kijun"], lw=1.3, label="基準線(26)")
    ax.plot(xl, L["span_tk"], color=C["cloud"], lw=0.8, label="先行スパン(基準転換)")
    ax.plot(xl, L["span_52"], color=C["cloud"], lw=0.8, ls="--", label="先行スパン(52)")
    ax.fill_between(xl, L["span_tk"], L["span_52"], color=C["cloud"], alpha=0.18)
    ax.plot(xl, L["chikou"], color=C["chikou"], lw=1.0, label="遅行スパン(26)")
    # 極
    for p in res.pivots:
        if p.date < df.index[0] or p.rank < 2:
            continue
        ax.annotate(f"{p.date.strftime('%m/%d')}\n{p.price:,.0f}", (mdates.date2num(p.date), p.price),
                    textcoords="offset points", xytext=(0, 8 if p.kind == "H" else -16),
                    ha="center", fontsize=6 + p.rank, color="#333", alpha=0.85)
    # 相場水準
    c0 = df["Close"].iloc[-1]
    for lv in res.levels[:max_levels]:
        ax.axhline(lv.price, color=C["level"], lw=0.8, ls=":", alpha=0.8)
        ax.text(x[-1], lv.price, f" 水準 {lv.price:,.0f}", color=C["level"], fontsize=7, va="center")
    # 変化日
    ymin, ymax = df["Low"].min(), df["High"].max()
    span = ymax - ymin
    for k, cl in enumerate(sorted(res.change_clusters[:max_dates], key=lambda z: z["i_min"])):
        xd = mdates.date2num(cl["date"].to_pydatetime())
        ax.axvline(xd, color=C["date"], lw=0.9, ls="--", alpha=0.8)
        ax.text(xd, ymin + span * (0.02 + 0.05 * (k % 4)), f"{cl['date'].strftime('%m/%d')}({cl['score']:.0f})",
                color=C["date"], fontsize=7, ha="center", va="bottom", rotation=90,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    ax.set_xlim(x[0] - 2 * w, xl[-1] + 2 * w)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m"))
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, ncol=3, framealpha=0.9)
    ax.set_title(title or f"{res.label}  一目均衡表（細田理論）  終値 {c0:,.1f}", fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
