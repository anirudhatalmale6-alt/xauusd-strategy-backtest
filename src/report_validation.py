#!/usr/bin/env python3
"""Self-contained HTML report for the 2020-2022 validation slice."""
from __future__ import annotations

import base64
import glob
import io
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out_val"

NICE = {
    "f1_trend_pullback": "F1 - Trend pullback",
    "f2_momentum_breakout": "F2 - Momentum / volatility breakout",
    "f3_level_rejection": "F3 - Level rejection (redesigned)",
    "f4_mean_reversion": "F4 - Mean reversion, ranging only",
}
INT_COLS = {"trades", "wins", "losses", "setups", "worst_losing_streak",
            "best_winning_streak", "timeouts", "skipped_overlapping", "year"}
GOOD_BASE = {"expectancy_R": 0.0, "net_pnl": 0.0, "profit_factor": 1.0,
             "net_return_pct": 0.0, "ci_low": 0.0, "ci_high": 0.0,
             "exp_R_ci_low": 0.0, "exp_R_ci_high": 0.0}


def fmt(col, v):
    if pd.isna(v):
        return "-"
    if isinstance(v, (bool,)):
        return "yes" if v else "no"
    if col == "year":
        return str(int(v))          # a year is a label, not a quantity - no comma
    if col in INT_COLS and isinstance(v, (int, float)):
        return f"{int(v):,}"
    if isinstance(v, float):
        return f"{v:,.4f}" if abs(v) < 10 else f"{v:,.2f}"
    return str(v)


def tbl(df: pd.DataFrame, cls: str = "") -> str:
    if df is None or df.empty:
        return "<p class='muted'>nothing to show</p>"
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    body = []
    for _, r in df.iterrows():
        tds = []
        for c in df.columns:
            v = r[c]
            k = ""
            if c in GOOD_BASE and isinstance(v, (int, float)) and not pd.isna(v):
                k = " class='pos'" if v > GOOD_BASE[c] else (" class='neg'" if v < GOOD_BASE[c] else "")
            if c == "family":
                v = NICE.get(v, v)
            tds.append(f"<td{k}>{fmt(c, v)}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return f"<div class='scroll'><table class='{cls}'><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def equity_chart() -> str:
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=110)
    for f in sorted(glob.glob(str(OUT / "trades_*.csv"))):
        name = os.path.basename(f)[7:-4]
        t = pd.read_csv(f, parse_dates=["exit_ts"])
        if t.empty:
            continue
        ax.plot(t.exit_ts, t.equity, lw=1.5, label=NICE.get(name, name))
    ax.axhline(10_000, color="#888", lw=1, ls="--")
    ax.set_title("Validation slice 2020-2022 - equity from 10,000 at 0.25% risk, real spread")
    ax.set_ylabel("account equity")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=.25)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    head = pd.read_csv(OUT / "validation_headline.csv")
    cost = pd.read_csv(OUT / "validation_costs.csv")
    mat = pd.read_csv(OUT / "regime_matrix.csv")
    meta = json.loads((OUT / "validation_meta.json").read_text())

    piv = cost.pivot(index="family", columns="scenario", values="expectancy_R").reset_index()
    piv["cost_drag_R"] = (piv["frictionless"] - piv["real_spread"]).round(4)
    piv = piv[["family", "frictionless", "real_spread", "double_spread",
               "double_spread_double_slip", "cost_drag_R"]]

    n_cells = len(mat)
    n_sig = int((mat.ci_low > 0).sum())
    best = head.sort_values("expectancy_R", ascending=False).iloc[0]

    parts = [f"""
<h1>Gold Strategy Research 002 - validation slice</h1>
<p class="sub">2020-01-01 to 2022-12-31 &middot; four families &middot; 2R target, 0.25% risk per trade,
one position at a time &middot; real quoted bid/ask plus slippage</p>

<div class="banner">
  <b>The confirmation period has not been opened.</b> Jan 2023 to Dec 2024 is not loaded by
  <code>run_validation.py</code> at all - the script asserts that validation ends before the
  confirmation period begins and refuses to run otherwise. Warm-up history is read from
  2015-01-01, which is look-back over a slice already used for development.
</div>

<h2>The short answer</h2>
<div class="verdict bad">
  <p><b>None of the four families is profitable on the validation slice, and none of them
  should be allowed into the confirmation period.</b></p>
  <p>Every family is negative after real transaction costs. Of the {n_cells} strategy-by-regime
  cells tested, <b>{n_sig}</b> have a confidence interval that excludes zero on the positive side.
  The strongest family, {NICE.get(best.family, best.family)}, is at
  {best.expectancy_R:+.4f}R with a 95% interval of
  [{best.exp_R_ci_low:+.3f}, {best.exp_R_ci_high:+.3f}] - which is to say, indistinguishable
  from nothing.</p>
  <p>The confirmation slice is a resource that can only be spent once. Spending it on a family
  that lost money in validation would be paying the last clean data we have for a coin flip.</p>
</div>

<h2>Headline results, real spread</h2>
{tbl(head)}

<h2>Transaction costs are the whole story for F1 and F2</h2>
<p>This is the most useful thing the validation produced, so it comes before the detail.
The same setups were re-run with the quoted spread scaled around the mid: zero, real, and double.
Frictionless is not achievable and is not a target - it is here to show how much of each family
the costs are eating.</p>
{tbl(piv)}
<p><b>Read the first two rows carefully.</b> Trend pullback and momentum breakout are mildly
positive frictionless and negative at the real spread. That is <i>not</i> the same as saying
"there is an edge and the costs eat it" - frictionless, both are still statistically
indistinguishable from zero (P(no edge) 0.40 and 0.21). What can be said is stronger and more
useful: whatever they have is smaller than what it costs to trade them.</p>
<p>Level rejection has no edge even frictionless (-0.0098R) and is then destroyed by costs.
Mean reversion is firmly negative before a single cent of cost is charged, which means the idea
itself does not work on gold in this form - that one is not a cost problem, it is a wrong problem.</p>

<h2>Why the cost drag differs so much between families</h2>
<p>Cost measured in R is not a fixed number. It is the round-trip cost divided by the stop
distance, and the stop distance is the denominator of R. So the tighter the stop, the larger the
same spread looms.</p>
{tbl(pd.DataFrame([
    {"family": "f1_trend_pullback", "setups_found_on": "1H", "median_stop_$": 8.81,
     "round_trip_cost_$": 0.462, "cost_as_R": 0.0525, "measured_drag_R": 0.0495},
    {"family": "f2_momentum_breakout", "setups_found_on": "1H", "median_stop_$": 10.31,
     "round_trip_cost_$": 0.461, "cost_as_R": 0.0448, "measured_drag_R": 0.0917},
    {"family": "f3_level_rejection", "setups_found_on": "15m", "median_stop_$": 3.44,
     "round_trip_cost_$": 0.454, "cost_as_R": 0.1319, "measured_drag_R": 0.1480},
    {"family": "f4_mean_reversion", "setups_found_on": "1H", "median_stop_$": 4.29,
     "round_trip_cost_$": 0.462, "cost_as_R": 0.1077, "measured_drag_R": 0.0681},
]))}
<p>Level rejection pays <b>0.13R in costs on every single trade</b> because its stops are around
$3.44 wide, against $8.81 for trend pullback. It needs to be right roughly 13 percentage points
more often just to stand still. It is not.</p>
<p>Momentum breakout is the interesting exception: its measured drag (0.0917R) is twice what the
median spread predicts (0.0448R). That is the family doing exactly what it was designed to do -
it only fires on a volatility expansion, and a volatility expansion is precisely when the spread
widens. The strategy selects for its own worst execution.</p>
<div class="callout">
  <b>The structural lesson, and it applies to whatever comes next.</b> The brief asked for more
  trades than 001A's twenty a year, and dropping to 1H and 15m delivered that - 91 to 337 a year.
  But you buy sample by shortening the timeframe, shortening the timeframe tightens the stop, and
  a tighter stop makes every unit of cost a larger fraction of every unit of reward. Sample size
  and cost survival pull against each other. That tension is real, it is arithmetic rather than
  opinion, and no amount of parameter tuning removes it.
</div>

<h2>Equity curves</h2>
<img class="chart" src="data:image/png;base64,{equity_chart()}" alt="validation equity curves">

<h2>The regime matrix</h2>
<p>Every strategy against every market condition the daily chart was in at the time.
<code>decision_grade</code> is false where the cell holds fewer than
{meta['min_cell_trades_for_a_decision']} trades - those rows are shown for completeness and are
not allowed to influence anything.</p>
{tbl(mat)}
<p class="muted">Not one cell in this table has a lower confidence bound above zero. The eight
positive decision-grade cells all have intervals that comfortably contain zero, and picking the
best of them would be selecting a winner out of noise - which is the exact failure that ended
001A.</p>
"""]

    for by, title in (("year", "Year by year"),
                      ("direction", "BUY vs SELL"),
                      ("regime", "By regime, pooled"),
                      ("session", "London vs New York vs overlap"),
                      ("trendiness", "By trendiness"),
                      ("reason", "How trades ended")):
        p = OUT / f"by_{by}.csv"
        if p.exists():
            parts.append(f"<h2>{title}</h2>{tbl(pd.read_csv(p))}")

    parts.append(f"""
<h2>Full cost sensitivity, every family and scenario</h2>
{tbl(cost)}

<h2>What I am not doing</h2>
<ul>
  <li>Not touching the confirmation period. Nothing has earned it.</li>
  <li>Not tuning any of the four families to rescue them. Every parameter is still exactly as
      written down before the first run, in <code>families.PARAMS</code>.</li>
  <li>Not picking the best regime cell and calling it a strategy. Eight cells are positive; none
      is significant; choosing among them after seeing the results is the definition of the
      mistake we already paid for.</li>
  <li>Not presenting the frictionless column as the real result. It is a diagnostic.</li>
</ul>

<p class="foot">Parameters, cost scenarios and slice boundaries are recorded in
<code>out_val/validation_meta.json</code>. Every trade is in <code>out_val/trades_*.csv</code>,
every refused setup with its reason in <code>out_val/rejected_*.csv</code>.</p>
""")

    html = """<meta charset="utf-8"><title>Gold Research 002 - validation 2020-2022</title>
<style>
 body{font:15px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
      max-width:1180px;margin:0 auto;padding:28px 20px 80px;color:#1c1c1e;background:#fff}
 h1{font-size:27px;margin:0 0 4px} .sub{color:#666;margin:0 0 22px}
 h2{font-size:20px;margin:34px 0 10px;padding-bottom:6px;border-bottom:2px solid #eee}
 table{border-collapse:collapse;font-size:13px;width:100%}
 th,td{border:1px solid #e4e4e7;padding:5px 9px;text-align:right;white-space:nowrap}
 th{background:#f6f6f8;text-align:right;font-weight:600}
 td:first-child,th:first-child{text-align:left}
 tbody tr:nth-child(even){background:#fbfbfc}
 .scroll{overflow-x:auto}
 .pos{color:#0a7a34;font-weight:600} .neg{color:#b3261e;font-weight:600}
 .banner{background:#eef4ff;border:1px solid #c7d9f7;border-left:5px solid #3b6fd4;
         padding:12px 16px;border-radius:6px;margin:16px 0}
 .verdict{padding:14px 18px;border-radius:8px;margin:12px 0}
 .verdict.bad{background:#fdf0ef;border:1px solid #f3c6c2;border-left:5px solid #b3261e}
 .callout{background:#fffaf0;border:1px solid #f0dfb8;border-left:5px solid #d19b1d;
          padding:12px 16px;border-radius:6px;margin:16px 0}
 .chart{width:100%;max-width:100%;border:1px solid #eee;border-radius:6px;margin-top:8px}
 .muted{color:#777;font-size:13px} .foot{color:#777;font-size:13px;margin-top:34px}
 code{background:#f3f3f5;padding:1px 5px;border-radius:4px;font-size:13px}
</style>
""" + "".join(parts)

    p = OUT / "validation_report.html"
    p.write_text(html)
    print(f"written {p}  ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
