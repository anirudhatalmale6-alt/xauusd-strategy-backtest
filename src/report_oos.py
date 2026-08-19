#!/usr/bin/env python3
"""
The sealed-period report. Run once, reports what happened, changes nothing.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out_oos"
IS = ROOT / "out_insample"

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
.kpi{display:flex;flex-wrap:wrap;gap:12px;margin:18px 0}
.kpi div{background:#fff;border:1px solid #e3e8ee;border-radius:6px;padding:12px 16px;min-width:132px}
.kpi b{display:block;font-size:22px;margin-bottom:2px}
.kpi span{font-size:12px;color:#68758a;text-transform:uppercase;letter-spacing:.4px}
img{max-width:100%;border:1px solid #dfe4ea;background:#fff;border-radius:4px}
.small{font-size:13px;color:#68758a}
code{background:#eef1f5;padding:1px 5px;border-radius:3px;font-size:12.5px}
"""

GREEN = ("expectancy_R", "net_pnl", "net_return_pct", "profit_factor", "pnl", "r")
INT_COLS = ("year", "trades", "wins", "losses", "worst_losing_streak",
            "best_winning_streak", "timeouts", "skipped_overlapping",
            "zone_touches", "bars_held", "bars_to_retest", "count")


def tbl(df, highlight=None):
    if df is None or df.empty:
        return "<p class='small'>no rows</p>"
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        tds = []
        for c in df.columns:
            v = r[c]
            cls = ""
            if isinstance(v, (int, float)) and not isinstance(v, bool) and c in GREEN:
                base = 1.0 if c == "profit_factor" else 0.0
                cls = " class='pos'" if v > base else (" class='neg'" if v < base else "")
            if isinstance(v, float):
                v = (f"{v:.0f}" if c in INT_COLS
                     else f"{v:,.4f}" if c.startswith("exp") or c == "prob_no_edge"
                     else f"{v:,.2f}")
            tds.append(f"<td{cls}>{v}</td>")
        mark = " style='background:#f2f8f4'" if highlight and highlight(r) else ""
        rows.append(f"<tr{mark}>" + "".join(tds) + "</tr>")
    return f"<div class='scroll'><table><tr>{head}</tr>{''.join(rows)}</table></div>"


def read(name, base=OUT):
    p = base / name
    return pd.read_csv(p) if p.exists() else None


def equity_chart(eq):
    fig, ax = plt.subplots(figsize=(9.5, 3.0))
    t = pd.to_datetime(eq.exit_ts, utc=True)
    ax.plot(t, eq.equity, color="#4353b0", linewidth=1.6)
    ax.axhline(10_000, color="#999", linewidth=.8, linestyle="--")
    ax.fill_between(t, 10_000, eq.equity,
                    where=eq.equity >= 10_000, color="#0a7d3f", alpha=.15)
    ax.fill_between(t, 10_000, eq.equity,
                    where=eq.equity < 10_000, color="#b3261e", alpha=.15)
    ax.set_title("Sealed period equity curve - $10,000 start, 0.25% risk")
    ax.set_ylabel("account")
    ax.grid(alpha=.2)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build() -> Path:
    grid = read("oos_grid.csv")
    head = grid[grid.headline].iloc[0]
    meta = json.loads((OUT / "oos_meta.json").read_text())
    pre = meta["frozen_config"]

    h = [f"<style>{CSS}</style><div class='wrap'>",
         "<h1>XAU/USD - sealed out-of-sample result</h1>",
         "<p class='sub'>2025-01-01 to 2026-08-19. Strategy 001A, 200 EMA regime "
         "filter, London/New York session filter, 2R target, 0.25% risk. "
         "Run once, on the rules you froze.</p>"]

    h.append("<div class='seal'><b>The rules were frozen before this ran.</b><br><br>"
             f"They are written in <code>out_oos/preregistered.json</code>, which "
             f"is committed alongside the results: <b>{pre['headline']}</b>. "
             "Nothing was changed after seeing the numbers below, and nothing "
             "will be.<br><br>"
             "<span class='small'>Note on one thing I did not do: 3R scored "
             "better than 2R over the ten in-sample years (+0.186R vs +0.112R). "
             "I kept 2R, because 2R is what your brief specified and moving to "
             "the best in-sample number is exactly the over-fitting you ruled "
             "out.</span></div>")

    h.append("<h2>The short answer</h2>")
    h.append("<div class='bad'><b>It did not hold up - but read the next section "
             "before drawing the obvious conclusion, because the sealed period "
             "only tested half the strategy.</b><br><br>"
             f"31 trades, {head.win_rate_pct:.2f}% win rate, profit factor "
             f"{head.profit_factor:.3f}, expectancy <b>{head.expectancy_R:+.4f}R</b>, "
             f"net <b>{head.net_return_pct:+.2f}%</b> on the account. The in-sample "
             "expectancy was +0.112R. This is a clear miss.</div>")

    h.append("<div class='kpi'>"
             f"<div><b>{int(head.trades)}</b><span>trades</span></div>"
             f"<div><b>{head.win_rate_pct:.1f}%</b><span>win rate</span></div>"
             f"<div><b>{head.profit_factor:.2f}</b><span>profit factor</span></div>"
             f"<div><b class='neg'>{head.expectancy_R:+.3f}R</b><span>expectancy</span></div>"
             f"<div><b class='neg'>{head.net_return_pct:+.2f}%</b><span>net return</span></div>"
             f"<div><b>{head.max_drawdown_pct:.2f}%</b><span>max drawdown</span></div>"
             f"<div><b>{int(head.worst_losing_streak)}</b><span>worst losing streak</span></div>"
             "</div>")

    # ---------------------------------------------------------------- why ----
    h.append("<h2>Why this result is narrower than it looks</h2>")
    d_oos = read("by_direction_oos.csv")
    d_is = read("by_direction_001A_ema50_200_session.csv", IS)
    h.append("<div class='note'><b>Every single trade in the sealed period was a "
             "BUY. Not one sell.</b><br><br>"
             "Gold's 50-day EMA stayed above its 200-day EMA for essentially the "
             "whole of 2025 and 2026 so far. The regime filter you approved only "
             "permits sells when that is the other way round, so the short side "
             "of the strategy - the side that produced <i>all</i> of the "
             "in-sample profit - never got a single opportunity to trade.<br><br>"
             "All 45 qualified setups were longs. All 31 trades were longs. "
             "The sealed period is a test of the long side only.</div>")

    if d_is is not None:
        h.append("<h3>In-sample, by direction (2015-2024)</h3>")
        h.append(tbl(d_is))
    if d_oos is not None:
        h.append("<h3>Sealed period, by direction (2025-2026)</h3>")
        h.append(tbl(d_oos))

    h.append("<div class='good'><b>Put side by side, the two periods actually "
             "agree with each other.</b><br><br>"
             "The long side lost money in-sample: <b>-0.049R over 133 trades</b>. "
             "The long side lost money out-of-sample: <b>-0.131R over 31 trades</b>. "
             "That is the same answer twice, on data eleven years apart. The "
             "sealed period did not contradict the in-sample study - it confirmed "
             "the half of it that was already known to be weak, and stayed silent "
             "on the half that carried the result.<br><br>"
             "So the honest verdict is not \"the strategy failed\". It is: "
             "<b>the long side is now tested and does not work, and the short "
             "side is still untested.</b></div>")

    # ------------------------------------------------------------ breakdown --
    h.append("<h2>Everything you asked for</h2>")
    h.append("<h3>Headline result and the surrounding grid</h3>")
    h.append("<p class='small'>The highlighted row is the frozen configuration and "
             "is the result. The other rows are context only - they were not used "
             "to choose anything.</p>")
    h.append(tbl(grid, highlight=lambda r: bool(r.headline)))

    eq = read("equity_oos.csv")
    if eq is not None and not eq.empty:
        h.append("<h3>Equity curve</h3>")
        h.append(f"<img src='{equity_chart(eq)}'>")

    for title, fname in (("Year by year", "by_year_oos.csv"),
                         ("London vs New York vs overlap", "by_session_oos.csv"),
                         ("BUY vs SELL", "by_direction_oos.csv"),
                         ("Market regime", "by_regime_oos.csv"),
                         ("How trades ended", "by_exit_reason_oos.csv")):
        df = read(fname)
        if df is not None and not df.empty:
            h.append(f"<h3>{title}</h3>")
            h.append(tbl(df))

    h.append("<p class='small'>2026 covers January to 19 August only, and contains "
             "just five trades. It is far too small a sample to read as a trend.</p>")

    rr = read("rejection_reasons.csv")
    if rr is not None and not rr.empty:
        h.append("<h3>Setups that were refused, and why</h3>")
        h.append(tbl(rr))
        h.append("<p class='small'><code>retest_expired</code> - price broke the "
                 "zone but never came back inside 12 bars. <code>stop_too_wide</code> "
                 "- the structural stop was more than 4x ATR away, so the trade was "
                 "skipped rather than sized down. <code>out_of_session</code> - a "
                 "valid setup outside London/New York hours.</p>")

    # -------------------------------------------------------- trade rate -----
    h.append("<h2>One practical problem for the paper-trading plan</h2>")
    h.append("<div class='note'><b>This strategy trades about 20 times a year, and "
             "it went from 21 April to 19 August 2026 without a single qualified "
             "setup.</b><br><br>"
             "197 trades over ten in-sample years. 31 over the twenty sealed "
             "months. At that rate, your target of 50 to 100 more qualified "
             "trades on a live paper feed would take somewhere between "
             "<b>two and a half and five years</b> to collect.<br><br>"
             "That is worth knowing before we build the harness, because it "
             "changes what paper trading is for. It cannot be the thing that "
             "proves the edge - there is not enough time. What it can do is prove "
             "that the live engine reproduces the backtest exactly, on maybe a "
             "dozen trades over a few months, which is a smaller but still "
             "worthwhile job.</div>")

    h.append("<h2>The full trade log</h2>")
    tr = read("trades_oos.csv")
    if tr is not None and not tr.empty:
        show = [c for c in ["entry_ts", "exit_ts", "direction", "session", "regime",
                            "entry", "stop", "target", "exit", "reason",
                            "structural_stop", "atr_buffer", "stop_distance",
                            "atr15", "spread", "zone_lo", "zone_hi", "zone_touches",
                            "bars_to_retest", "body_pct", "close_pos",
                            "r", "pnl", "equity"] if c in tr.columns]
        h.append(tbl(tr[show]))
        h.append("<p class='small'>Every column from your brief is here: structural "
                 "stop and ATR buffer recorded separately, the zone that produced "
                 "the setup, how many 15m bars the retest took, the trigger "
                 "candle's shape, the spread paid, and the result in R. The same "
                 "file is <code>out_oos/trades_oos.csv</code> in the repository.</p>")

    h.append("<h2>What I am not doing</h2>")
    h.append("<p>Not changing a rule, not re-running anything, not switching to 3R, "
             "and not proposing a sells-only strategy on the back of this. The "
             "sealed period has been used, and it cannot be used again.</p>")
    h.append("</div>")

    p = OUT / "oos_report.html"
    p.write_text("".join(h), encoding="utf-8")
    return p


if __name__ == "__main__":
    print(build())
