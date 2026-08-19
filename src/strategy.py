#!/usr/bin/env python3
"""
Strategy 001A (trend + breakout + retest) and 001B (trend + zone rejection).

Both share the same top-down structure from the brief:

    Daily  -> which way are we allowed to trade at all
    4H     -> where the major support / resistance zones are
    15m    -> the trigger

Signals are produced independently of the profit target, so 001A and 001B are
compared on an identical set of setups and the R-multiple grid changes only
where trades are closed, never which ones are found.

Two things are recorded besides the signals themselves:

  * every setup that triggered the pattern but was then refused, with the
    reason - so "why did it not trade" is answerable rather than guessed at.
  * how many 15m bars each retest took, for the ones that completed AND the
    ones that timed out, so the 12-bar window can be judged on evidence later
    instead of being tuned now.

Look-ahead control
------------------
* A 4H swing pivot is only "known" k bars after it printed, because that is when
  it is confirmed. Zones are built from confirmed pivots only.
* Daily and 4H values reach a 15m bar by `known_at`, the higher timeframe bar's
  CLOSE, never its open.
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
STOP_ATR_BUFFER = 0.50  # ATR(14) padding beyond the structural stop
STOP_MIN_ATR = 0.30     # reject setups whose stop is unrealistically tight
STOP_MAX_ATR = 4.00     # ... or so wide the position size becomes meaningless
SWING_LOOKBACK = 8      # 15m bars used to find the structural stop
TREND_EMA = 50
TREND_EMA_SLOW = 200    # only used by trend_mode="ema50_200"
STRUCTURE_PIVOT_K = 2   # fractal size for the daily higher-high/higher-low test

TREND_MODES = ("ema50", "ema50_200")


# ---------------------------------------------------------------- trend ------
def _pivots(h: pd.Series, l: pd.Series, k: int) -> tuple[pd.Series, pd.Series]:
    """Fractal swing highs/lows. True on the bar that PRINTED the pivot."""
    win = 2 * k + 1
    hi = h.rolling(win, center=True).max()
    lo = l.rolling(win, center=True).min()
    return (h >= hi) & h.notna(), (l <= lo) & l.notna()


def daily_trend(d1: pd.DataFrame, mode: str = "ema50") -> pd.DataFrame:
    """
    +1 uptrend, -1 downtrend, 0 no opinion - always from closed bars only.

    mode "ema50"      : price vs the 50 EMA, confirmed by market structure.
    mode "ema50_200"  : the same, plus the long-term regime (50 EMA above or
                        below the 200 EMA) has to agree. This is the comparison
                        Gloria asked for - does the extra filter earn its place,
                        or does it just remove trades?
    """
    if mode not in TREND_MODES:
        raise ValueError(f"trend mode must be one of {TREND_MODES}")

    out = pd.DataFrame(index=d1.index)
    fast = ema(d1.c, TREND_EMA)
    slow = ema(d1.c, TREND_EMA_SLOW)
    sh, sl = _pivots(d1.h, d1.l, STRUCTURE_PIVOT_K)

    # A pivot is not confirmed until k more bars have closed, so shift it forward.
    k = STRUCTURE_PIVOT_K
    ph = d1.h.where(sh).shift(k)
    pl = d1.l.where(sl).shift(k)
    last_h = ph.ffill()
    last_l = pl.ffill()
    # compare against the previous DISTINCT pivot, not the forward-filled value
    prev_h = ph.dropna().shift(1).reindex(ph.index).ffill()
    prev_l = pl.dropna().shift(1).reindex(pl.index).ffill()

    higher = (last_h > prev_h) & (last_l > prev_l)
    lower = (last_h < prev_h) & (last_l < prev_l)

    up = (d1.c > fast) & higher
    dn = (d1.c < fast) & lower

    if mode == "ema50_200":
        regime_up = fast > slow
        up = up & regime_up
        dn = dn & ~regime_up

    out["trend"] = np.where(up, 1, np.where(dn, -1, 0))
    out["regime"] = np.where(fast > slow, "bull", "bear")
    out["known_at"] = d1.known_at
    return out[["trend", "regime", "known_at"]]


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
    last_seen: int = 0
    zid: int = 0

    def rebuild(self, pad: float) -> None:
        self.lo = min(self.prices) - pad
        self.hi = max(self.prices) + pad
        self.touches = len(self.prices)


def build_zones(h4: pd.DataFrame) -> pd.DataFrame:
    """
    Walk the 4H series once, maintaining a live set of zones, and record the
    zone set as at every bar close. One row per (bar close, zone).
    """
    a = atr(h4, 14)
    sh, sl = _pivots(h4.h, h4.l, PIVOT_K)
    conf_high = h4.h.where(sh).shift(PIVOT_K)   # confirmation lag
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


# ------------------------------------------------------------- candles -------
def candle_stats(o: float, h: float, l: float, c: float) -> dict:
    """Shape of the trigger candle, kept so setups can be reviewed later."""
    rng = h - l
    if rng <= 0:
        return {"body_pct": 0.0, "upper_wick_pct": 0.0,
                "lower_wick_pct": 0.0, "close_pos": 0.5, "range": 0.0}
    return {
        "body_pct": round(abs(c - o) / rng, 4),
        "upper_wick_pct": round((h - max(o, c)) / rng, 4),
        "lower_wick_pct": round((min(o, c) - l) / rng, 4),
        "close_pos": round((c - l) / rng, 4),      # 1.0 = closed on the high
        "range": round(rng, 3),
    }


def _confirms(o: float, h: float, l: float, c: float, direction: int) -> bool:
    """Close in the top (or bottom) third of the bar's range, in our direction."""
    rng = h - l
    if rng <= 0:
        return False
    if direction > 0:
        return c > o and (c - l) / rng >= 1 - CONFIRM_FRACTION
    return c < o and (h - c) / rng >= 1 - CONFIRM_FRACTION


# -------------------------------------------------------------- signals ------
def generate_signals(m15: pd.DataFrame, h4: pd.DataFrame, d1: pd.DataFrame,
                     mode: str, trend_mode: str = "ema50",
                     session_filter: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    mode         'A' (breakout + retest) or 'B' (zone rejection)
    trend_mode   'ema50' or 'ema50_200'
    session_filter  False runs the identical strategy with no session
                    restriction, which is the benchmark for whether the
                    London/New York filter earns its place.

    Returns (signals, rejections).
    """
    assert mode in ("A", "B")

    dt_ = daily_trend(d1, trend_mode)
    ht_ = h4_trend(h4)
    zones = build_zones(h4)

    bars = m15.reset_index()
    for src, cols in ((dt_, {"trend": "d_trend", "regime": "regime"}),
                      (ht_, {"trend": "h_trend"})):
        right = (src[["known_at", *cols.keys()]]
                 .rename(columns=cols)
                 .sort_values("known_at")
                 .reset_index(drop=True))
        bars = pd.merge_asof(bars.sort_values("ts"), right,
                             left_on="ts", right_on="known_at", direction="backward")
        bars = bars.drop(columns=[c for c in ("known_at",) if c in bars.columns])
    bars["atr15"] = atr(m15, 14).values
    bars = bars.set_index("ts")

    zone_times = pd.DatetimeIndex(sorted(zones.known_at.unique()))
    zones_by_time = {t: g for t, g in zones.groupby("known_at")}

    highs, lows = bars.h.values, bars.l.values
    opens, closes = bars.o.values, bars.c.values
    atr15 = bars.atr15.values
    idx = bars.index
    swing_lo = bars.l.rolling(SWING_LOOKBACK).min().values
    swing_hi = bars.h.rolling(SWING_LOOKBACK).max().values

    pending: dict[tuple[int, int], int] = {}   # (zid, direction) -> breakout bar
    out: list[dict] = []
    rejected: list[dict] = []

    def refuse(i, direction, z, reason, extra=None):
        row = {"ts": idx[i], "direction": direction, "zone_id": int(z.zid) if z else -1,
               "reason": reason, "session": bars.session.iat[i],
               "price": float(closes[i])}
        if extra:
            row.update(extra)
        rejected.append(row)

    for i in range(len(bars)):
        ts = idx[i]
        if not np.isfinite(atr15[i]) or atr15[i] <= 0:
            continue
        d_tr = bars.d_trend.iat[i]
        h_tr = bars.h_trend.iat[i]
        if not np.isfinite(d_tr) or d_tr == 0:
            continue          # no daily opinion - not a setup, so not a rejection
        if d_tr != h_tr:
            continue          # 4H disagrees with Daily - likewise
        direction = int(d_tr)

        pos = zone_times.searchsorted(ts, side="right") - 1
        if pos < 0:
            continue
        zs = zones_by_time.get(zone_times[pos])
        if zs is None or zs.empty:
            continue

        for z in zs.itertuples():
            price = closes[i]
            is_support = price > z.hi
            is_resistance = price < z.lo

            if mode == "A":
                key = (z.zid, direction)
                # Arm only on the bar that actually CROSSES the zone edge. Without
                # the previous-bar test a breakout re-arms on every bar while price
                # sits beyond the zone, which turns one breakout into hundreds of
                # "expired retests" and makes the timing statistics meaningless.
                prev_c = closes[i - 1] if i > 0 else closes[i]
                crossed_up = closes[i] > z.hi and prev_c <= z.hi
                crossed_dn = closes[i] < z.lo and prev_c >= z.lo
                if direction > 0 and crossed_up and is_support:
                    pending.setdefault(key, i)
                elif direction < 0 and crossed_dn and is_resistance:
                    pending.setdefault(key, i)

                if key in pending:
                    b = pending[key]
                    age = i - b
                    if age > RETEST_WINDOW:
                        pending.pop(key, None)
                        refuse(i, direction, z, "retest_expired",
                               {"bars_to_retest": age, "retest_completed": False})
                        continue
                    if age == 0:
                        continue
                    touched = (lows[i] <= z.hi) if direction > 0 else (highs[i] >= z.lo)
                    if not touched:
                        continue
                    if not _confirms(opens[i], highs[i], lows[i], closes[i], direction):
                        continue
                    made = _make(i, ts, direction, z, bars, swing_lo, swing_hi,
                                 atr15, "retest", session_filter, refuse,
                                 bars_to_retest=age)
                    if made:
                        out.append(made)
                        pending.pop(key, None)

            else:   # mode B - rejection from the zone, no break required
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
                if not ok:
                    continue
                if not _confirms(opens[i], highs[i], lows[i], closes[i], direction):
                    continue
                made = _make(i, ts, direction, z, bars, swing_lo, swing_hi,
                             atr15, "rejection", session_filter, refuse)
                if made:
                    out.append(made)

    sig = pd.DataFrame([s for s in out if s is not None])
    rej = pd.DataFrame(rejected)
    if not sig.empty:
        sig = (sig.drop_duplicates(subset=["signal_ts", "direction"])
                  .sort_values("signal_ts").reset_index(drop=True))
    return sig, rej


def _make(i, ts, direction, z, bars, swing_lo, swing_hi, atr15, kind,
          session_filter, refuse, bars_to_retest=None):
    """Apply the remaining filters. Anything refused here is logged with a reason."""
    if i + 1 >= len(bars):
        return None
    if session_filter and not bars.in_session.iat[i]:
        refuse(i, direction, z, "out_of_session", {"bars_to_retest": bars_to_retest})
        return None

    a = atr15[i]
    if direction > 0:
        structural = min(swing_lo[i], z.lo)
        stop = structural - STOP_ATR_BUFFER * a
    else:
        structural = max(swing_hi[i], z.hi)
        stop = structural + STOP_ATR_BUFFER * a
    if not np.isfinite(stop):
        return None

    ref = bars.c.iat[i]
    dist = abs(ref - stop)
    if dist < STOP_MIN_ATR * a:
        refuse(i, direction, z, "stop_too_tight",
               {"stop_distance": round(dist, 3), "atr15": round(a, 3)})
        return None
    if dist > STOP_MAX_ATR * a:
        refuse(i, direction, z, "stop_too_wide",
               {"stop_distance": round(dist, 3), "atr15": round(a, 3)})
        return None

    cs = candle_stats(bars.o.iat[i], bars.h.iat[i], bars.l.iat[i], bars.c.iat[i])
    return {
        "signal_ts": ts,
        "entry_ts": bars.index[i + 1],       # filled on the NEXT bar's open
        "direction": direction,
        # the stop, broken into its two parts, as asked
        "structural_stop": round(float(structural), 3),
        "atr_buffer": round(float(STOP_ATR_BUFFER * a), 3),
        "stop": round(float(stop), 3),
        "stop_distance": round(float(dist), 3),
        "ref_close": round(float(ref), 3),
        "atr15": round(float(a), 3),
        "spread": round(float(bars.spread.iat[i]), 4),
        "zone_id": int(z.zid), "zone_lo": round(float(z.lo), 3),
        "zone_hi": round(float(z.hi), 3), "zone_touches": int(z.touches),
        "setup": kind,
        "session": bars.session.iat[i],
        "regime": bars.regime.iat[i] if "regime" in bars.columns else "",
        "bars_to_retest": bars_to_retest,
        "retest_completed": None if bars_to_retest is None else True,
        **cs,
    }
