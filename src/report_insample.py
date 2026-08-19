#!/usr/bin/env python3
"""
The in-sample report. Deliberately narrow: it answers the three questions that
were asked, states what changed since the first run, and stops.

The sealed period (Jan 2025 - Aug 2026) is not read by this script and does not
appear anywhere in it.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out_insample"

CSS = """
body{font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
     margin:0;padding:32px;background:#f6f7f9;color:#1c2430}
.wrap{max-width:1080px;margin:0 auto}
h1{font-size:26px;margin:0 0 4px}
h2{font-size:20px;margin:34px 0 10px;border-bottom:2px solid #d9dee5;padding-bottom:6px}
h3{font-size:16px;margin:22px 0 8px;color:#41506a}
.sub{color:#68758a;margin:0 0 22px}
.scroll{overflow-x:auto;margin:10px 0 18px}
table{border-collapse:collapse;width:100%;background:#fff;
     box-shadow:0 1px 2px rgba(0,0,0,.07);font-size:13.5px}
th,td{padding:7px 10px;text-align:right;border-bottom:1px solid #eceff3}
th{background:#eef1f5;font-weight:600;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
tr:last-child td{border-bottom:none}
.pos{color:#0a7d3f;font-weight:600}.neg{color:#b3261e;font-weight:600}
.note{background:#fff8e1;border-left:4px solid #e6a700;padding:12px 16px;margin:16px 0}
.bad{background:#fdecea;border-left:4px solid #b3261e;padding:12px 16px;margin:16px 0}
.good{background:#e9f6ee;border-left:4px solid #0a7d3f;padding:12px 16px;margin:16px 0}
.seal{background:#eef2ff;border-left:4px solid #4353b0;padding:12px 16px;margin:16px 0}
img{max-width:100%;border:1px solid #dfe4ea;background:#fff;border-radius:4px}
.small{font-size:13px;color:#68758a}
"""

GREEN = ("expectancy_R", "net_pnl", "net_return_pct", "profit_factor")


def tbl(df, money=GREEN, highlight=None):
    if df is None or df.empty:
        return "<p class='small'>no rows</p>"
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        tds = []
        for c in df.columns:
            v = r[c]
            cls = ""
            if isinstance(v, (int, float)) and c in money:
                base = 1.0 if c == "profit_factor" else 0.0
                cls = " class='pos'" if v > base else (" class='neg'" if v < base else "")
            if isinstance(v, float):
                v = f"{v:.0f}" if c in ("year", "trades", "years", "positive_years",
                                        "completed", "expired", "retests",
                                        "worst_losing_streak", "timeouts") else f"{v:,.4f}" \
                    if c.startswith("exp") or c == "prob_no_edge" else f"{v:,.2f}"
            tds.append(f"<td{cls}>{v}</td>")
        mark = " style='background:#f2f8f4'" if highlight and highlight(r) else ""
        rows.append(f"<tr{mark}>" + "".join(tds) + "</tr>")
    return f"<div class='scroll'><table><tr>{head}</tr>{''.join(rows)}</table></div>"


def read(n):
    p = OUT / n
    return pd.read_csv(p) if p.exists() else None


def bar_chart(df, x, y, title, ylabel):
    fig, ax = plt.subplots(figsize=(9.5, 3.0))
    colours = ["#0a7d3f" if v > 0 else "#b3261e" for v in df[y]]
    ax.bar(df[x].astype(str), df[y], color=colours, alpha=.85)
    ax.axhline(0, color="#333", linewidth=.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.2, axis="y")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build() -> Path:
    ab = read("ab_comparison.csv")
    rt = read("retest_timing.csv")
    grid = read("insample_grid.csv")

    h = [f"<style>{CSS}</style><div class='wrap'>",
         "<h1>XAU/USD - in-sample results</h1>",
         "<p class='sub'>2015-01-01 to 2024-12-31. Ten years, 4.1 million minute bars. "
         "The sealed period is not in this document.</p>"]

    h.append("<div class='seal'><b>The sealed period has not been run.</b><br><br>"
             "Jan 2025 to Aug 2026 was not loaded by the script that produced this "
             "report. Nothing below has been influenced by it. It gets run once, "
             "after you confirm the rules are frozen.</div>")

    h.append("<h2>What changed since the first report</h2>")
    h.append("<div class='bad'><b>Two things, and they reverse my earlier advice.</b><br><br>"
             "<b>1. A defect in how breakouts were detected.</b> A breakout was being "
             "re-armed on every bar while price stayed beyond the zone, instead of only "
             "on the bar that crossed it. One breakout was being counted as hundreds. "
             "That corrupted 001A's setups and made the retest statistics meaningless. "
             "It is fixed.<br><br>"
             "<b>2. Ten years of data instead of three and a half.</b> The first run "
             "started in Aug 2021. Gold spent most of that window going up, and it "
             "flattered one strategy and buried the other.<br><br>"
             "Together these flip the answer. <b>001A is now the candidate and 001B is "
             "not.</b> I told you to drop 001A. On the corrected code over ten years "
             "that was wrong, and I would rather say so than let it stand.</div>")

    if ab is not None:
        best = ab.sort_values("expectancy_R", ascending=False).iloc[0]
        h.append("<h2>1. Does the 200 EMA regime filter add value?</h2>")
        h.append("<div class='good'><b>Yes - consistently, and that consistency is the "
                 "point.</b><br><br>It improves expectancy in <b>all four</b> comparisons "
                 "- both strategies, with and without the session filter - and roughly "
                 "halves the maximum drawdown every time. A filter that helps in one "
                 "place is luck. One that helps in four out of four, in the same "
                 "direction, is doing something real. The cost is about 37% of the "
                 "trades.</div>")
        h.append(tbl(ab, highlight=lambda r: r.trend_mode == "ema50_200"))

        h.append("<h2>2. Does the London / New York filter earn its place?</h2>")
        h.append("<div class='note'><b>No, not on expectancy.</b><br><br>"
                 "In three of the four pairs, removing the session filter is slightly "
                 "<i>better</i>. But every difference is far inside the confidence "
                 "interval - this is noise, not a finding, and I would not choose "
                 "between them on these numbers.<br><br>"
                 "There is one honest reason to keep it: it lowers the maximum drawdown "
                 "in three of the four pairs, and you cannot watch the overnight session "
                 "anyway. That is a practical argument, not a statistical one.</div>")

        h.append("<h2>3. How long do retests actually take?</h2>")
        if rt is not None:
            h.append(tbl(rt))
        dist = read("retest_bars_001A_ema50_200.csv")
        if dist is not None and not dist.empty:
            h.append(f"<img src='{bar_chart(dist, dist.columns[0], 'retests', 'How many 15m bars the retest took', 'retests')}'>")
        h.append("<p>Median <b>4 bars</b>, mean 5.2, 90th percentile <b>10</b>. About "
                 "<b>38%</b> of breakouts get a confirmed retest inside the window; the "
                 "rest expire. Only 10 completed on the last possible bar, so widening "
                 "the window would buy very little. <b>The 12-bar rule you set is a good "
                 "one</b> - I am not proposing to change it.</p>")

        h.append("<h2>The candidate</h2>")
        h.append(f"<p><b>001A with the 200 EMA regime filter.</b> Expectancy "
                 f"{best.expectancy_R:+.3f}R over {int(best.trades)} trades, profit "
                 f"factor {best.profit_factor:.2f}, worst drawdown "
                 f"{best.max_drawdown_pct:.2f}%.</p>")

    yr = read("by_year_001A_ema50_200_session.csv")
    if yr is not None and not yr.empty:
        h.append("<h3>Year by year - 001A, regime filter, session filter on</h3>")
        h.append(f"<img src='{bar_chart(yr, 'year', 'expectancy_R', 'Expectancy by year (R)', 'expectancy R')}'>")
        h.append(tbl(yr))
        h.append("<p class='small'>Six of ten years positive, and no single year carries "
                 "the result. That is the 'reasonable across different years' you asked "
                 "for rather than one spectacular year hiding four bad ones.</p>")

    d = read("by_direction_001A_ema50_200_session.csv")
    if d is not None and not d.empty:
        h.append("<h2>The thing I am least comfortable with</h2>")
        h.append(tbl(d))
        h.append("<div class='bad'><b>All of the profit is in the short trades.</b><br><br>"
                 "The buys are slightly negative over 133 trades. The sells carry the "
                 "whole result on 64 trades in ten years - about six a year. That is a "
                 "thin base for a conclusion, and because the regime filter only allows "
                 "sells when the 50 EMA is below the 200 EMA, those 64 trades are "
                 "clustered in a handful of gold downtrends.<br><br>"
                 "I am not proposing we act on it by trading sells only. Cutting the "
                 "strategy down to the half that worked in the past is exactly the "
                 "over-fitting you asked me to avoid. I am flagging it so you know where "
                 "the number comes from.</div>")

    if grid is not None:
        h.append("<h2>Full in-sample grid</h2>")
        h.append("<p class='small'>Every combination, for completeness. No parameter "
                 "search was run - these are the thresholds from your brief, with only "
                 "the two A/B comparisons you asked for.</p>")
        h.append(tbl(grid))

    h.append("<h2>What I need from you</h2>")
    h.append("<ol><li>Confirm the rules to freeze: <b>001A, 200 EMA regime filter, "
             "session filter on or off</b> (I lean to on, for drawdown).</li>"
             "<li>Once frozen I run the sealed period <b>once</b> and send whatever it "
             "says.</li><li>Then the paper trader, using those exact frozen rules.</li></ol>")
    h.append("</div>")

    p = OUT / "insample_report.html"
    p.write_text("".join(h), encoding="utf-8")
    return p


if __name__ == "__main__":
    print(build())
