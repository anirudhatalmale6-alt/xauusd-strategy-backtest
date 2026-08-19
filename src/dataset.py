#!/usr/bin/env python3
"""
Load the M1 bid/ask dataset and build every timeframe the strategies need.

Everything downstream depends on two rules that are enforced here:

  1. All bar timestamps are the bar's OPEN time, in UTC.
  2. A bar on a higher timeframe is not allowed to be "known" until it has
     closed. Each higher timeframe frame therefore carries a `known_at` column
     - the moment the strategy is first permitted to look at that row - and the
     joins in strategy.py use it. This is where look-ahead bias would creep in
     if we were careless, so it is handled once, here.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# The backtest runs on the aggregated chart feed (five years of it). The
# tick-built set in data/m1 covers a slice of the same period and exists purely
# as a control - validate_sources.py shows the two agree to 0.000 on the median.
DATA = Path(__file__).resolve().parent.parent / "data" / "m1_freeserv"

# Sessions are defined in LOCAL exchange time on purpose. Defining them in UTC
# would silently drift by an hour twice a year, and the two markets change their
# clocks on different dates.
LONDON = ("Europe/London", "08:00", "16:30")
NEWYORK = ("America/New_York", "08:00", "17:00")


def load_m1(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    files = sorted(DATA.rglob("XAUUSD_*.parquet"))
    if not files:
        raise SystemExit(f"no M1 parquet files in {DATA} - run fetch_dukascopy.py first")
    df = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.index.name = "ts"
    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df.index <= pd.Timestamp(end, tz="UTC")]
    # Mid is what the strategy "sees" on a chart; bid/ask are what it trades at.
    df["o"] = (df.bid_o + df.ask_o) / 2
    df["h"] = (df.bid_h + df.ask_h) / 2
    df["l"] = (df.bid_l + df.ask_l) / 2
    df["c"] = (df.bid_c + df.ask_c) / 2
    return df


def resample(m1: pd.DataFrame, rule: str, *, origin: str | pd.Timestamp = "start_day") -> pd.DataFrame:
    """Aggregate mid-price OHLC. Index is the bar OPEN time; `known_at` is its close."""
    g = m1.resample(rule, origin=origin, label="left", closed="left")
    out = pd.DataFrame({
        "o": g.o.first(), "h": g.h.max(), "l": g.l.min(), "c": g.c.last(),
        # the actual quotes the engine has to trade against, not the mid
        "bid_o": g.bid_o.first(), "ask_o": g.ask_o.first(),
        "bid_c": g.bid_c.last(), "ask_c": g.ask_c.last(),
        "ticks": g.ticks.sum(),
        "spread": g.spread_mean.mean(),
    }).dropna(subset=["o"])
    out = out[out.ticks > 0]
    out["known_at"] = out.index + pd.Timedelta(rule)
    return out


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df.c.shift(1)
    tr = pd.concat([df.h - df.l, (df.h - prev).abs(), (df.l - prev).abs()], axis=1).max(axis=1)
    # Wilder smoothing, the ATR everybody actually means when they say ATR.
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def tag_sessions(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Which cash session each bar-open falls in. DST handled by the tz database."""
    out = pd.DataFrame(index=idx)
    for name, (tz, t0, t1) in {"london": LONDON, "newyork": NEWYORK}.items():
        local = idx.tz_convert(tz)
        t = pd.Series(local.time, index=idx)
        h0, m0 = (int(x) for x in t0.split(":"))
        h1, m1_ = (int(x) for x in t1.split(":"))
        import datetime as _dt
        out[name] = (t >= _dt.time(h0, m0)) & (t < _dt.time(h1, m1_))
        # Weekends never count, whatever the clock says.
        out[name] &= pd.Series(local.weekday, index=idx) < 5
    out["session"] = "off"
    out.loc[out.london, "session"] = "london"
    out.loc[out.newyork, "session"] = "newyork"
    # The overlap is genuinely both; label it so the split report is not misleading.
    out.loc[out.london & out.newyork, "session"] = "overlap"
    out["in_session"] = out.session != "off"
    return out


def build(start: str | None = None, end: str | None = None) -> dict[str, pd.DataFrame]:
    m1 = load_m1(start, end)
    m15 = resample(m1, "15min")
    h4 = resample(m1, "4h")
    d1 = resample(m1, "1D")
    m15 = m15.join(tag_sessions(m15.index))
    return {"m1": m1, "m15": m15, "h4": h4, "d1": d1}


if __name__ == "__main__":
    d = build()
    for k, v in d.items():
        print(f"{k:4s} {len(v):9,d} bars  {v.index.min()}  ->  {v.index.max()}")
    m15 = d["m15"]
    print("\nsession split of 15m bars:")
    print(m15.session.value_counts())
    print(f"\nmedian spread on 15m bars: {m15.spread.median():.3f}")
