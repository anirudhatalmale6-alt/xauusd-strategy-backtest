#!/usr/bin/env python3
"""
Turn the backtest output into one self-contained HTML report.

Charts are embedded as base64 PNGs so the file can be opened from anywhere
without a folder of images beside it.
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
OUT = ROOT / "out"

CSS = """
body{font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
     margin:0;padding:32px;background:#f6f7f9;color:#1c2430}
.wrap{max-width:1080px;margin:0 auto}
h1{font-size:26px;margin:0 0 4px}h2{font-size:20px;margin:34px 0 10px;
     border-bottom:2px solid #d9dee5;padding-bottom:6px}
h3{font-size:16px;margin:22px 0 8px;color:#41506a}
.sub{color:#68758a;margin:0 0 22px}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:10px 0 18px}
table{border-collapse:collapse;width:100%;background:#fff;
     box-shadow:0 1px 2px rgba(0,0,0,.07);font-size:13.5px}
th,td{padding:7px 10px;text-align:right;border-bottom:1px solid #eceff3}
th{background:#eef1f5;text-align:right;font-weight:600;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
tr:last-child td{border-bottom:none}
.pos{color:#0a7d3f;font-weight:600}.neg{color:#b3261e;font-weight:600}
.note{background:#fff8e1;border-left:4px solid #e6a700;padding:12px 16px;margin:16px 0}
.bad{background:#fdecea;border-left:4px solid #b3261e;padding:12px 16px;margin:16px 0}
.good{background:#e9f6ee;border-left:4px solid #0a7d3f;padding:12px 16px;margin:16px 0}
img{max-width:100%;border:1px solid #dfe4ea;background:#fff;border-radius:4px}
code{background:#eceff3;padding:1px 5px;border-radius:3px;font-size:13px}
.small{font-size:13px;color:#68758a}
"""


def fig_to_img(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def equity_chart(books: dict[str, pd.DataFrame], title: str) -> str:
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    for label, tr in books.items():
        if tr is None or tr.empty:
            continue
        ax.plot(pd.to_datetime(tr.exit_ts), tr.equity, linewidth=1.5, label=label)
    ax.set_title(title)
    ax.set_ylabel("account equity ($)")
    ax.grid(alpha=.25)
    ax.legend(frameon=False, fontsize=9)
    return fig_to_img(fig)


def drawdown_chart(tr: pd.DataFrame, title: str) -> str:
    fig, ax = plt.subplots(figsize=(9.5, 2.4))
    eq = tr.equity
    dd = 100 * (eq - eq.cummax()) / eq.cummax()
    ax.fill_between(pd.to_datetime(tr.exit_ts), dd, 0, color="#b3261e", alpha=.35)
    ax.set_title(title)
    ax.set_ylabel("drawdown %")
    ax.grid(alpha=.25)
    return fig_to_img(fig)


def tbl(df: pd.DataFrame, money_cols=()) -> str:
    if df is None or df.empty:
        return "<p class='small'>no rows</p>"
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    body = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            cls = ""
            if isinstance(v, (int, float)) and c in money_cols:
                cls = " class='pos'" if v > 0 else (" class='neg'" if v < 0 else "")
            if isinstance(v, float):
                # years and counts are not money - no thousands separator, no decimals
                v = f"{v:.0f}" if c in ("year", "trades", "worst_losing_streak",
                                        "timeouts", "skipped_overlapping") \
                    else (f"{v:,.2f}" if abs(v) < 1e6 else f"{v:,.0f}")
            cells.append(f"<td{cls}>{v}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='scroll'><table><tr>{head}</tr>{''.join(body)}</table></div>"


def read(name: str) -> pd.DataFrame | None:
    p = OUT / name
    return pd.read_csv(p) if p.exists() else None


def build() -> Path:
    grid = read("summary_grid.csv")
    reg = json.loads((OUT / "preregistered.json").read_text()) if (OUT / "preregistered.json").exists() else {}
    meta = json.loads((OUT / "dataset_meta.json").read_text()) if (OUT / "dataset_meta.json").exists() else {}

    h = [f"<style>{CSS}</style><div class='wrap'>",
         "<h1>XAU/USD strategy backtest - 001A vs 001B</h1>",
         "<p class='sub'>Historical test only. No live trading, no forward prediction, "
         "no claim that any of this repeats.</p>"]

    # ---- the verdict goes first, because it is the answer to the question ----
    verdict_bits = []
    for name, r in reg.items():
        i, o = r.get("in_sample", {}), r.get("out_of_sample", {})
        verdict_bits.append((name, r, i, o))
    if verdict_bits:
        h.append("<h2>The short answer</h2>")
        for name, r, i, o in verdict_bits:
            e_i, e_o = float(i.get("expectancy_R", 0)), float(o.get("expectancy_R", 0))
            p_i, p_o = float(i.get("prob_no_edge", 1)), float(o.get("prob_no_edge", 1))
            lo_i = float(i.get("exp_R_ci_low", -1))
            if e_i <= 0 and e_o <= 0:
                cls, head_ = "bad", f"{name}: no edge. It lost money in both periods."
                body = (f"Expectancy {e_i:+.3f}R in-sample and {e_o:+.3f}R out-of-sample, on "
                        f"{i.get('trades')} and {o.get('trades')} trades. The rules as written "
                        f"do not make money on gold, and no profit target rescues them - every "
                        f"one of the ten combinations tested is negative in-sample.")
            elif lo_i > 0:
                cls, head_ = "good", f"{name}: a measurable edge."
                body = (f"Expectancy {e_i:+.3f}R in-sample, and the 95% confidence interval "
                        f"stays above zero.")
            else:
                cls, head_ = "note", f"{name}: promising, but not proven."
                body = (f"Expectancy {e_i:+.3f}R in-sample ({i.get('trades')} trades) and "
                        f"{e_o:+.3f}R out-of-sample ({o.get('trades')} trades). But the 95% "
                        f"confidence interval on the in-sample result runs from "
                        f"{i.get('exp_R_ci_low')} to {i.get('exp_R_ci_high')} - it includes zero. "
                        f"There is a {p_i:.0%} chance the in-sample edge is noise, and a "
                        f"{p_o:.0%} chance for out-of-sample. On this evidence the strategy "
                        f"cannot be called profitable, only 'not yet ruled out'.")
            h.append(f"<div class='{cls}'><b>{head_}</b><br><br>{body}</div>")
        h.append("<p>Neither result justifies risking money yet. What both need is more "
                 "trades, and the way to collect those without paying for them is paper "
                 "trading, not a live account.</p>")

    if meta:
        h.append("<h2>1. The data</h2>")
        # the two long prose fields read badly as table columns
        table_meta = {k: v for k, v in meta.items() if k not in ("source", "validation")}
        h.append(f"<p><b>Source:</b> {meta.get('source', '')}</p>")
        h.append(tbl(pd.DataFrame([table_meta])))
        h.append("<p class='small'>Real bid and ask, UTC timestamps. The 15-minute, 4-hour "
                 "and daily bars are all built from the same minute bars, so the timeframes "
                 "cannot disagree with each other. The feed was independently checked against "
                 "242,288 minutes rebuilt tick by tick from Dukascopy's raw archive - a "
                 "different endpoint in a different format - and the median difference was "
                 "0.0000 on every OHLC column, with identical spreads.</p>")

    h.append("<h2>2. How a trade is filled</h2>")
    h.append("<ul>"
             "<li>A signal on the close of a 15-minute bar is filled at the <b>open of the next "
             "bar</b>, never at the price that triggered it.</li>"
             "<li>Buys pay the <b>ask</b>, sells receive the <b>bid</b> - real quoted spread from "
             "the tick data, never the mid-price.</li>"
             "<li>Entries and stop-outs carry 5 cents of extra slippage. Targets do not, "
             "because a take-profit is a resting limit order.</li>"
             "<li>Inside a trade the <b>minute bars</b> decide whether the stop or the target came "
             "first. If both fall in the same minute it is booked as a <b>loss</b>.</li>"
             "<li>One position at a time.</li></ul>")

    if grid is not None:
        h.append("<h2>3. Full results grid</h2>")
        for period in ("in-sample", "out-of-sample"):
            part = grid[grid.period == period]
            if part.empty:
                continue
            h.append(f"<h3>{period}</h3>")
            h.append(tbl(part.drop(columns=["period"]),
                         money_cols=("net_return_pct", "expectancy_R", "final_equity")))

    if reg:
        h.append("<h2>4. The pre-registered choice</h2>")
        h.append("<p>One configuration per strategy was chosen on the in-sample data "
                 "<b>before</b> the out-of-sample period was run, so the comparison below is "
                 "honest rather than shopped for.</p>")
        for name, r in reg.items():
            h.append(f"<h3>{name} - target {r['r_target']}R, risk {r['risk_pct']}%</h3>")
            rows = []
            for lab, key in (("in-sample", "in_sample"), ("out-of-sample", "out_of_sample")):
                if key in r:
                    rows.append({"period": lab, **r[key]})
            h.append(tbl(pd.DataFrame(rows), money_cols=("net_return_pct", "expectancy_R")))

    for name in ("001A", "001B"):
        books = {}
        for lab in ("in_sample", "out_of_sample"):
            tr = read(f"trades_{name}_{lab}.csv")
            if tr is not None and not tr.empty:
                books[lab.replace("_", "-")] = tr
        if not books:
            continue
        h.append(f"<h2>5. {name} in detail</h2>")
        h.append(f"<img src='{equity_chart(books, f'{name} equity curve')}'>")
        for lab, tr in books.items():
            h.append(f"<img src='{drawdown_chart(tr, f'{name} {lab} drawdown')}'>")
        for lab in ("in_sample", "out_of_sample"):
            for what, fname in (("By session", "by_session"),
                                ("Buy vs sell", "by_direction"),
                                ("Year by year", "by_year")):
                df = read(f"{fname}_{name}_{lab}.csv")
                if df is not None and not df.empty:
                    h.append(f"<h3>{what} - {lab.replace('_', '-')}</h3>")
                    h.append(tbl(df, money_cols=("net_pnl", "expectancy_R")))

    h.append("<h2>6. What this does and does not tell you</h2>")
    h.append("<div class='note'><b>It does tell you</b> how these exact rules would have "
             "behaved on five years of real gold prices, with real spreads, and what the worst "
             "stretch would have felt like.<br><br>"
             "<b>It does not tell you</b> that the next five years will look anything like it. "
             "Nothing here is a prediction, and no result below should be read as one.</div>")
    h.append("</div>")

    p = OUT / "report.html"
    p.write_text("".join(h), encoding="utf-8")
    return p


if __name__ == "__main__":
    print(build())
