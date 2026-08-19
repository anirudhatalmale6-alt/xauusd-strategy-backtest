#!/usr/bin/env python3
"""
Strategy 001A (trend + breakout + retest) and 001B (trend + zone rejection).

Both share the same top-down structure Gloria described:

    Daily  -> which way are we allowed to trade at all
    4H     -> where the major support / resistance zones are
    15m    -> the trigger

Signals are produced independently of the profit target, so 001A and 001B are
compared on an identical set of setups and the R-multiple grid does not change
which trades are found - only where they are closed.

Look-ahead control
------------------
* A 4H swing pivot is only "known" k bars after it printed, because that is when
  it is confirmed. Zones are built from confirmed pivots only.
* Daily and 4H values are attached to a 15m bar by `known_at`, the higher
  timeframe bar's CLOSE, never its open.
* A signal is decided on the CLOSE of a 15m bar and filled on the OPEN of the
  next one. Nothing is ever filled at the price that triggered it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dataset import atr, ema

# --- parameters, all fixed in advance from the brief rather than tuned --------
PIVOT_K = 3            # bars either side that make a 4H swing pivot
ZONE_MIN_TOUCHES = 2   # "major" means price respected it more than once
ZONE_TOL_ATR = 0.50    # pivots within this many 4H ATRs are the same zone
ZONE_PAD_ATR = 0.25    # zone edges padded by this many 4H ATRs
ZONE_MAX_AGE = 120     # 4H bars a zone survives untouched (~20 trading days)
RETEST_WINDOW = 12     # 15m bars a breakout has to be retested within (3 hours)
CONFIRM_FRACTION = 1 / 3  # close must sit in this fraction of the bar's range
STOP_ATR_BUFFER = 0.50  # ATR(14) padding beyond the swing point
STOP_MIN_ATR = 0.30     # reject setups whose stop is unrealistically tight
STOP_MAX_ATR = 4.00     # ... or so wide the position size becomes meaningless
SWING_LOOKBACK = 8      # 15m bars used to find the protective swing point
TREND_EMA = 50
STRUCTURE_PIVOT_K = 2   # fractal size for the daily higher-high/higher-low test


# ---------------------------------------------------------------- trend ------
def _pivots(h: pd.Series, l: pd.Series, k: int) -> tuple[pd.Series, pd.Series]:
    """Fractal swing highs/lows. True on the bar that PRINTED the pivot."""
    win = 2 * k + 1
    hi = h.rolling(win, center=True).max()
    lo = l.rolling(win, center=True).min()
    return (h >= hi) & h.notna(), (l <= lo) & l.notna()


def daily_trend(d1: pd.DataFrame) -> pd.DataFrame:
    """+1 uptrend, -1 downtrend, 0 no opinion - and only ever from closed bars."""
    out = pd.DataFrame(index=d1.index)
    out["ema"] = ema(d1.c, TREND_EMA)
    sh, sl = _pivots(d1.h, d1.l, STRUCTURE_PIVOT_K)

    # A pivot is not confirmed until k more bars have closed, so shift it forward.
    k = STRUCTURE_PIVOT_K
    ph = d1.h.where(sh).shift(k)
    pl = d1.l.where(sl).shift(k)
    last_h, prev_h = ph.ffill(), ph.ffill().shift(1)
    # shift(1) on the forward-filled series is wrong when the value repeats, so
    # compare against the previous DISTINCT pivot instead.
    prev_h = ph.dropna().shift(1).reindex(ph.index).ffill()
    prev_l = pl.dropna().shift(1).reindex(pl.index).ffill()
    last_l = pl.ffill()

    higher = (last_h > prev_h) & (last_l > prev_l)
    lower = (last_h < prev_h) & (last_l < prev_l)

    up = (d1.c > out.ema) & higher
    dn = (d1.c < out.ema) & lower
    out["trend"] = np.where(up, 1, np.where(dn, -1, 0))
    out["known_at"] = d1.known_at
    return out[["trend", "known_at"]]


def h4_trend(h4: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=h4.index)
    e = ema(h4.c, TREND_EMA)
    out["trend"] = np.where(h4.c > e, 1, np.where(h4.c < e, -1, 0))
    out["known_at"] = h4.known_at
    return out[["trend", "known_at"]]


# ---------------------------------------------------------------- zones ------
@dataclass
class Zone:
    kind: str                 # "high" (resistance-born) or "low" (support-born)
    prices: list[float] = field(default_factory=list)
    lo: float = 0.0
    hi: float = 0.0
    touches: int = 0
    last_seen: int = 0        # index of the 4H bar that last confirmed a pivot
    zid: int = 0

    def rebuild(self, pad: float) -> None:
        self.lo = min(self.prices) - pad
        self.hi = max(self.prices) + pad
        self.touches = len(self.prices)


def build_zones(h4: pd.DataFrame) -> pd.DataFrame:
    """
    Walk the 4H series once, maintaining a live set of zones, and record the
    zone set as at every bar close. Returns one row per (bar close, zone).
    """
    a = atr(h4, 14)
    sh, sl = _pivots(h4.h, h4.l, PIVOT_K)
    # confirmation lag: the pivot at i is only visible at i + PIVOT_K
    conf_high = h4.h.where(sh).shift(PIVOT_K)
    conf_low = h4.l.where(sl).shift(PIVOT_K)

    zones: list[Zone] = []
    rows: list[tuple] = []
    next_id = 0

    for i, (ts, row) in enumerate(h4.iterrows()):
        av = a.iloc[i]
        if not np.isfinite(av) or av <= 0:
            continue
        tol, pad = ZONE_TOL_ATR * av, ZONE_PAD_ATR * av

        for price, kind in ((conf_high.iloc[i], "high"), (conf_low.iloc[i], "low")):
            if not np.isfinite(price):
                continue
            hit = None
            for z in zones:
                if z.kind == kind and abs(np.mean(z.prices) - price) <= tol:
                    hit = z
                    break
            if hit is None:
                next_id += 1
                hit = Zone(kind=kind, zid=next_id)
                zones.append(hit)
            hit.prices.append(float(price))
            hit.last_seen = i
            hit.rebuild(pad)

        zones = [z for z in zones if i - z.last_seen <= ZONE_MAX_AGE]

        for z in zones:
            if z.touches >= ZONE_MIN_TOUCHES:
                rows.append((row.known_at, z.zid, z.kind, z.lo, z.hi, z.touches))

    return pd.DataFrame(rows, columns=["known_at", "zid", "kind", "lo", "hi", "touches"])


# -------------------------------------------------------------- signals ------
def _confirms(o: float, h: float, l: float, c: float, direction: int) -> bool:
    """Close in the top (or bottom) third of the bar's range, in our direction."""
    rng = h - l
    if rng <= 0:
        return False
    if direction > 0:
        return c > o and (c - l) / rng >= 1 - CONFIRM_FRACTION
    return c < o and (h - c) / rng >= 1 - CONFIRM_FRACTION


def generate_signals(m15: pd.DataFrame, h4: pd.DataFrame, d1: pd.DataFrame,
                     mode: str) -> pd.DataFrame:
    """mode is 'A' (breakout + retest) or 'B' (zone rejection)."""
    assert mode in ("A", "B")

    dt_ = daily_trend(d1)
    ht_ = h4_trend(h4)
    zones = build_zones(h4)

    # Attach the higher timeframes by their CLOSE time - this is the join that
    # keeps the whole thing honest.
    bars = m15.reset_index()
    for src, col in ((dt_, "d_trend"), (ht_, "h_trend")):
        right = (src[["known_at", "trend"]]
                 .rename(columns={"trend": col})
                 .sort_values("known_at")
                 .reset_index(drop=True))
        bars = pd.merge_asof(bars.sort_values("ts"), right,
                             left_on="ts", right_on="known_at", direction="backward")
        bars = bars.drop(columns=[c for c in ("known_at",) if c in bars.columns])
    bars["atr15"] = atr(m15, 14).values
    bars = bars.set_index("ts")

    # zone snapshots, grouped by the moment they became visible
    zone_times = pd.DatetimeIndex(sorted(zones.known_at.unique()))
    zones_by_time = {t: g for t, g in zones.groupby("known_at")}

    highs = bars.h.values
    lows = bars.l.values
    opens = bars.o.values
    closes = bars.c.values
    atr15 = bars.atr15.values
    idx = bars.index

    # rolling swing points for the stop
    swing_lo = bars.l.rolling(SWING_LOOKBACK).min().values
    swing_hi = bars.h.rolling(SWING_LOOKBACK).max().values

    pending: dict[tuple[int, int], int] = {}  # (zid, direction) -> bar index of breakout
    out: list[dict] = []

    for i in range(len(bars)):
        ts = idx[i]
        if not np.isfinite(atr15[i]) or atr15[i] <= 0:
            continue
        d_tr = bars.d_trend.iat[i]
        h_tr = bars.h_trend.iat[i]
        if not np.isfinite(d_tr) or d_tr == 0 or d_tr != h_tr:
            continue  # daily and 4H must agree, else stand aside
        direction = int(d_tr)

        pos = zone_times.searchsorted(ts, side="right") - 1
        if pos < 0:
            continue
        zs = zones_by_time.get(zone_times[pos])
        if zs is None or zs.empty:
            continue

        for z in zs.itertuples():
            price = closes[i]
            # A zone only interests us from one side, and only in the trend direction.
            if direction > 0:
                is_support = price > z.hi
                is_resistance = price < z.lo
            else:
                is_support = price > z.hi
                is_resistance = price < z.lo

            if mode == "A":
                key = (z.zid, direction)
                if direction > 0 and closes[i] > z.hi and highs[i] > z.hi:
                    # broke up through it - arm a retest, unless already armed
                    if is_support and key not in pending:
                        pending[key] = i
                elif direction < 0 and closes[i] < z.lo and lows[i] < z.lo:
                    if is_resistance and key not in pending:
                        pending[key] = i

                if key in pending:
                    b = pending[key]
                    if i - b > RETEST_WINDOW:
                        pending.pop(key, None)
                        continue
                    if i == b:
                        continue
                    touched = (lows[i] <= z.hi) if direction > 0 else (highs[i] >= z.lo)
                    if not touched:
                        continue
                    if _confirms(opens[i], highs[i], lows[i], closes[i], direction):
                        out.append(_make(i, ts, direction, z, bars, swing_lo, swing_hi,
                                         atr15, "retest"))
                        pending.pop(key, None)

            else:  # mode B - rejection from the zone, no break required
                if direction > 0:
                    if not is_support:
                        continue
                    wick_in = lows[i] <= z.hi and lows[i] >= z.lo - 0.25 * atr15[i]
                    body_out = closes[i] > z.hi
                    ok = wick_in and body_out
                else:
                    if not is_resistance:
                        continue
                    wick_in = highs[i] >= z.lo and highs[i] <= z.hi + 0.25 * atr15[i]
                    body_out = closes[i] < z.lo
                    ok = wick_in and body_out
                if ok and _confirms(opens[i], highs[i], lows[i], closes[i], direction):
                    out.append(_make(i, ts, direction, z, bars, swing_lo, swing_hi,
                                     atr15, "rejection"))

    sig = pd.DataFrame([s for s in out if s is not None])
    if sig.empty:
        return sig
    sig = sig.drop_duplicates(subset=["signal_ts", "direction"]).sort_values("signal_ts")
    return sig.reset_index(drop=True)


def _make(i, ts, direction, z, bars, swing_lo, swing_hi, atr15, kind):
    if i + 1 >= len(bars):
        return None
    if not bars.in_session.iat[i]:
        return None
    a = atr15[i]
    if direction > 0:
        raw_stop = min(swing_lo[i], z.lo)
        stop = raw_stop - STOP_ATR_BUFFER * a
    else:
        raw_stop = max(swing_hi[i], z.hi)
        stop = raw_stop + STOP_ATR_BUFFER * a
    if not np.isfinite(stop):
        return None
    ref = bars.c.iat[i]
    dist = abs(ref - stop)
    if dist < STOP_MIN_ATR * a or dist > STOP_MAX_ATR * a:
        return None
    return {
        "signal_ts": ts,
        "entry_ts": bars.index[i + 1],   # filled on the NEXT bar's open
        "direction": direction,
        "stop": float(stop),
        "ref_close": float(ref),
        "atr15": float(a),
        "zone_id": int(z.zid),
        "zone_lo": float(z.lo),
        "zone_hi": float(z.hi),
        "setup": kind,
        "session": bars.session.iat[i],
    }
