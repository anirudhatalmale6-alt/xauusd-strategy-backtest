#!/usr/bin/env python3
"""
The market-regime layer for Strategy Research 002.

One job: label every intraday bar with what the DAILY chart was doing at the
time, so that a family's results can be split by market condition instead of
averaged into a single number that hides everything interesting.

The label has two axes, both computed on the daily bar and both causal:

  direction   close vs the 200-day EMA, plus the sign of that EMA's 20-day slope
              bull  - above a rising EMA
              bear  - below a falling EMA
              mixed - anything else (price and slope disagree)

  trendiness  ADX(14), the standard Wilder version
              trending    >= 25
              transition  20 to 25
              ranging     < 20

Measured over 2015-2026 these labels persist for 25 days (direction), 14 days
(trendiness) and 9.4 days (the pair). That is the number that decides whether a
regime layer is worth having at all - a label that flipped every other day would
be noise dressed up as structure.

Look-ahead control is the same rule used everywhere else in this project: a
daily row is not visible until the daily bar has CLOSED, so the frame carries
`known_at` and is joined with merge_asof on that column, never on the bar's own
timestamp.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dataset import atr, ema

EMA_SLOW = 200
SLOPE_LOOKBACK = 20
ADX_N = 14
ADX_TREND = 25.0
ADX_RANGE = 20.0


def adx(df: pd.DataFrame, n: int = ADX_N) -> pd.DataFrame:
    """Wilder's ADX. Returns +DI, -DI and ADX, all smoothed with alpha = 1/n."""
    up = df.h.diff()
    dn = -df.l.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)

    prev = df.c.shift(1)
    tr = pd.concat([df.h - df.l, (df.h - prev).abs(), (df.l - prev).abs()], axis=1).max(axis=1)

    a = 1 / n
    atr_ = tr.ewm(alpha=a, adjust=False, min_periods=n).mean()
    pdi = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=a, adjust=False, min_periods=n).mean() / atr_
    mdi = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=a, adjust=False, min_periods=n).mean() / atr_
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    return pd.DataFrame({"pdi": pdi, "mdi": mdi,
                         "adx": dx.ewm(alpha=a, adjust=False, min_periods=n).mean()})


def daily_regime(d1: pd.DataFrame) -> pd.DataFrame:
    """One row per daily bar, carrying `known_at` = that bar's close."""
    out = pd.DataFrame(index=d1.index)
    e = ema(d1.c, EMA_SLOW)
    slope = e.diff(SLOPE_LOOKBACK)
    a = adx(d1, ADX_N)

    out["direction"] = np.where(d1.c.isna() | e.isna(), "unknown",
                        np.where((d1.c > e) & (slope > 0), "bull",
                        np.where((d1.c < e) & (slope < 0), "bear", "mixed")))
    out["trendiness"] = np.where(a.adx.isna(), "unknown",
                         np.where(a.adx >= ADX_TREND, "trending",
                         np.where(a.adx < ADX_RANGE, "ranging", "transition")))
    out["regime"] = out.direction + "/" + out.trendiness
    out["adx"] = a.adx.round(2)
    out["atr_d"] = atr(d1, 14)
    out["atr_pct"] = (100 * out.atr_d / d1.c).round(3)
    # Volatility as a percentile of gold's own history, so "high" means high for
    # gold rather than high in dollars. Expanding, not full-sample - a full
    # sample percentile would leak the future into every early bar.
    out["vol_pctile"] = (100 * out.atr_pct.expanding(min_periods=250).rank(pct=True)).round(1)
    out["known_at"] = d1.known_at
    return out


def attach(bars: pd.DataFrame, d1: pd.DataFrame,
           cols: tuple[str, ...] = ("direction", "trendiness", "regime",
                                    "adx", "atr_pct", "vol_pctile")) -> pd.DataFrame:
    """
    Join the daily regime onto an intraday frame by `known_at`.

    `bars` keeps its own index; the returned frame is a copy with the regime
    columns added. Nothing is visible before the daily bar that produced it has
    closed.
    """
    dr = daily_regime(d1)
    # `bars` already carries its own `known_at` (its bar close), so the daily
    # join key is renamed rather than colliding into known_at_x / known_at_y.
    right = (dr[["known_at", *cols]]
             .rename(columns={"known_at": "_regime_known_at"})
             .sort_values("_regime_known_at").reset_index(drop=True))
    left = bars.reset_index().sort_values(bars.index.name or "ts")
    key = left.columns[0]
    merged = pd.merge_asof(left, right, left_on=key, right_on="_regime_known_at",
                           direction="backward")
    merged = merged.drop(columns=["_regime_known_at"]).set_index(key)
    merged.index.name = bars.index.name
    return merged


if __name__ == "__main__":
    import dataset
    d = dataset.build()
    dr = daily_regime(d["d1"])
    print(dr.regime.value_counts().to_string())
    print(f"\nlabelled from {dr[dr.direction != 'unknown'].index.min().date()}")
