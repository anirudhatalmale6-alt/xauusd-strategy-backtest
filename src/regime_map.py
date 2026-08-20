#!/usr/bin/env python3
"""
Describe what gold actually DID, year by year and quarter by quarter, 2015-2026.

This is deliberately not a strategy and it fits nothing. It exists so that the
development / validation split for Strategy Research 002 can be argued from the
market's own behaviour instead of chosen by eye. If a walk-forward fold contains
only bull quarters, a trend strategy will look brilliant in it and the split
itself will have done the over-fitting for us.

Three descriptive measures, all on the daily bar, all causal:

  direction  close vs the 200-day EMA, plus the sign of that EMA's 20-day slope
  trendiness ADX(14) - the standard Wilder version, >25 trending, <20 ranging
  volatility ATR(14) / close, expressed as a percentile of its own 2015-2026
             distribution so "high vol" means high for gold, not high in dollars

The regime label is the pair (direction, trendiness), which is the same shape as
the regime layer the client asked to investigate - so this doubles as a sanity
check that such a layer would produce usable, persistent states rather than
flip-flopping every other day.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out_regime"


def adx(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
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
    return pd.DataFrame({"pdi": pdi, "mdi": mdi, "adx": dx.ewm(alpha=a, adjust=False, min_periods=n).mean()})


def build() -> pd.DataFrame:
    d = dataset.build()
    d1 = d["d1"].copy()

    d1["ema200"] = dataset.ema(d1.c, 200)
    d1["ema200_slope"] = d1.ema200.diff(20)
    d1["atr14"] = dataset.atr(d1, 14)
    d1["atr_pct"] = 100 * d1.atr14 / d1.c
    d1 = d1.join(adx(d1, 14))

    d1["vol_pctile"] = d1.atr_pct.rank(pct=True) * 100

    d1["direction"] = np.where(d1.c.isna() | d1.ema200.isna(), "unknown",
                       np.where((d1.c > d1.ema200) & (d1.ema200_slope > 0), "bull",
                       np.where((d1.c < d1.ema200) & (d1.ema200_slope < 0), "bear", "mixed")))
    d1["trendiness"] = np.where(d1.adx.isna(), "unknown",
                        np.where(d1.adx >= 25, "trending",
                        np.where(d1.adx < 20, "ranging", "transition")))
    d1["regime"] = d1.direction + "/" + d1.trendiness
    return d1


def persistence(labels: pd.Series) -> float:
    """Mean run length in days. A regime layer that flips every 2 days is useless."""
    s = labels.dropna()
    runs = (s != s.shift()).cumsum()
    return float(s.groupby(runs).size().mean())


def main() -> None:
    OUT.mkdir(exist_ok=True)
    d1 = build()
    d1.to_csv(OUT / "daily_regime.csv")

    core = d1[d1.regime.str.contains("unknown") == False]  # noqa: E712

    print(f"daily bars {len(d1):,}   {d1.index.min().date()} -> {d1.index.max().date()}")
    print(f"regime-labelled bars {len(core):,}\n")

    print("=== how much of the sample each regime is worth ===")
    share = (core.regime.value_counts(normalize=True) * 100).round(1)
    cnt = core.regime.value_counts()
    print(pd.DataFrame({"days": cnt, "pct": share}).to_string())
    print(f"\nmean regime run length: {persistence(core.regime):.1f} days")
    print(f"mean direction run length: {persistence(core.direction):.1f} days")
    print(f"mean trendiness run length: {persistence(core.trendiness):.1f} days\n")

    # ---- year by year ----
    rows = []
    for y, g in core.groupby(core.index.year):
        r = {"year": y, "days": len(g),
             "return_pct": round(100 * (g.c.iloc[-1] / g.c.iloc[0] - 1), 1),
             "atr_pct_med": round(g.atr_pct.median(), 2),
             "adx_med": round(g.adx.median(), 1)}
        for k in ("bull", "bear", "mixed"):
            r[k] = round(100 * (g.direction == k).mean(), 0)
        for k in ("trending", "ranging", "transition"):
            r[k] = round(100 * (g.trendiness == k).mean(), 0)
        rows.append(r)
    yr = pd.DataFrame(rows)
    yr.to_csv(OUT / "by_year_regime.csv", index=False)
    print("=== year by year (columns after adx_med are % of days) ===")
    print(yr.to_string(index=False))

    # ---- quarter by quarter, which is the granularity a fold is built from ----
    q = core.copy()
    q["quarter"] = q.index.tz_convert(None).to_period("Q").astype(str)
    rows = []
    for qq, g in q.groupby("quarter"):
        rows.append({"quarter": qq, "days": len(g),
                     "return_pct": round(100 * (g.c.iloc[-1] / g.c.iloc[0] - 1), 1),
                     "dominant_direction": g.direction.mode().iloc[0],
                     "dominant_trendiness": g.trendiness.mode().iloc[0],
                     "atr_pct_med": round(g.atr_pct.median(), 2),
                     "vol_pctile_med": round(g.vol_pctile.median(), 0)})
    qt = pd.DataFrame(rows)
    qt.to_csv(OUT / "by_quarter_regime.csv", index=False)
    print("\n=== quarter by quarter ===")
    print(qt.to_string(index=False))

    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
