#!/usr/bin/env python3
"""
Transaction-cost scenarios.

The backtest already trades against real quoted bid and ask from the tick data
and adds slippage on entries and on stopped exits. That is the honest baseline.
But "does this survive its costs" is a different question from "what did it
make", and the only way to answer it is to move the costs and watch what breaks.

A scenario is a single multiplier k applied to the half-spread around the mid:

    ask = mid + k * (ask - mid)
    bid = mid - k * (mid - bid)

    k = 0   frictionless. Not achievable, and not a target. It exists so the
            report can show how much of the edge the costs are eating - a
            strategy that only works at k = 0 is a spreadsheet, not a strategy.
    k = 1   the real historical spread, as quoted. This is the headline.
    k = 2   double spread. Gold's spread widens on news, at the roll, and in
            exactly the fast conditions a breakout strategy wants to trade in,
            so this is the stress test that matters.

Slippage is separate and is moved with cfg.slippage, because widening the quote
and getting a worse fill inside it are two different failures.
"""
from __future__ import annotations

import pandas as pd

# (label, spread multiplier, slippage multiplier)
SCENARIOS = [
    ("frictionless", 0.0, 0.0),
    ("real_spread", 1.0, 1.0),
    ("double_spread", 2.0, 1.0),
    ("double_spread_double_slip", 2.0, 2.0),
]
HEADLINE = "real_spread"


def rescale(df: pd.DataFrame, k: float) -> pd.DataFrame:
    """
    Widen or narrow the quotes around the mid by a factor k.

    Only the bid/ask columns move. The mid OHLC the strategy looks at is left
    exactly as it was, because changing the spread does not change the chart -
    it changes what you pay to act on it.
    """
    if k == 1.0:
        return df
    out = df.copy()
    for side in ("o", "h", "l", "c"):
        b, a = f"bid_{side}", f"ask_{side}"
        if b not in out.columns or a not in out.columns:
            continue
        mid = out[side]
        out[a] = mid + k * (out[a] - mid)
        out[b] = mid - k * (mid - out[b])
    if "spread" in out.columns:
        out["spread"] = out["spread"] * k
    if "spread_mean" in out.columns:
        out["spread_mean"] = out["spread_mean"] * k
    return out
