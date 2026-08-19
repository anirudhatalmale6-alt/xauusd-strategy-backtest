#!/usr/bin/env python3
"""
Cross-check the two independent Dukascopy endpoints against each other.

The backtest runs on the aggregated chart feed, because it is the only way to
get five years of minute bars in reasonable time. But an aggregated feed is
somebody else's arithmetic, and the whole exercise is worthless if those numbers
are wrong.

So a slice of the same period was also built the hard way, straight from the raw
tick archive, by summing individual ticks into minute bars here. Two different
endpoints, two different formats, one underlying market. If they agree bar for
bar, the aggregated feed can be trusted for the rest.

This is the check that would catch a decimal in the wrong place, a timezone
shift, or bars stamped at the close instead of the open - the errors that
produce a perfectly believable equity curve out of garbage.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TICKS = ROOT / "data" / "m1"
FEED = ROOT / "data" / "m1_freeserv"


def load(folder: Path) -> pd.DataFrame:
    files = sorted(folder.rglob("XAUUSD_*.parquet"))
    if not files:
        raise SystemExit(f"nothing in {folder}")
    df = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
    return df[~df.index.duplicated(keep="first")]


def main() -> None:
    a = load(TICKS)      # built here from raw ticks
    b = load(FEED)       # Dukascopy's own aggregation
    print(f"tick-built : {len(a):,} bars  {a.index.min()} -> {a.index.max()}")
    print(f"chart feed : {len(b):,} bars  {b.index.min()} -> {b.index.max()}")

    both = a.index.intersection(b.index)
    print(f"overlap    : {len(both):,} minutes")
    if len(both) < 1000:
        raise SystemExit("not enough overlap to draw a conclusion")

    a2, b2 = a.loc[both], b.loc[both]
    cover_a = len(both) / len(a) * 100
    print(f"the tick-built set is {cover_a:.1f}% covered by the feed\n")

    worst = {}
    for col in ("bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c"):
        d = (a2[col] - b2[col]).abs()
        worst[col] = (d.mean(), d.median(), d.quantile(.999), d.max(), (d > 0.10).mean() * 100)

    print(f"{'column':8s} {'mean':>9s} {'median':>9s} {'p99.9':>9s} {'max':>9s} {'>10c %':>8s}")
    for col, (mean, med, p999, mx, pct) in worst.items():
        print(f"{col:8s} {mean:9.4f} {med:9.4f} {p999:9.4f} {mx:9.4f} {pct:8.3f}")

    sa = (a2.ask_c - a2.bid_c)
    sb = (b2.ask_c - b2.bid_c)
    print(f"\nspread  tick-built median {sa.median():.3f}   feed median {sb.median():.3f}")
    print(f"        tick-built p99    {sa.quantile(.99):.3f}   feed p99    {sb.quantile(.99):.3f}")

    # the verdict
    close = max(w[1] for w in worst.values())          # worst median difference
    tail = max(w[2] for w in worst.values())           # worst p99.9
    print()
    if close <= 0.02 and tail <= 0.50:
        print("VERDICT: the two sources agree. The chart feed is safe to backtest on.")
    elif close <= 0.05:
        print("VERDICT: broadly consistent, with a fatter tail than expected - "
              "worth looking at the outliers before trusting fine detail.")
    else:
        print("VERDICT: the sources DISAGREE. Do not backtest on this until it is explained.")

    diff = (a2.bid_c - b2.bid_c).abs()
    bad = diff.sort_values(ascending=False).head(8)
    if bad.iloc[0] > 0.10:
        print("\nlargest bid-close disagreements:")
        for ts, v in bad.items():
            print(f"  {ts}  tick {a2.bid_c[ts]:.3f}  feed {b2.bid_c[ts]:.3f}  diff {v:.3f}")


if __name__ == "__main__":
    sys.exit(main())
