#!/usr/bin/env python3
"""
The four strategy families of Research 002.

Every number these families use is declared once, in PARAMS below, and none of
them is searched over. That is the whole point: the brief asks for a small set
of economically sensible pre-defined rules, so the honest way to build this is
to write the numbers down first and then find out what they are worth. If a
family only works after its parameters are hunted for, it does not work.

The families deliberately do NOT share a trend filter, a session filter or a
stop style beyond what each one's own logic requires. They are meant to be four
different bets on how gold moves, not four skins on the same idea:

  F1 trend_pullback    buy dips inside an established daily trend
  F2 momentum_breakout trade a range break, but only when volatility expands
  F3 level_rejection   fade a failed push through a level real orders sit at
  F4 mean_reversion    fade a stretched move, but only when the daily is ranging

Shared machinery
----------------
* Setups are found on 1H (F1, F2, F4) or 15m (F3). Execution always happens on
  the 15m grid, because that is where engine.py fills and walks trades, and it
  is the more pessimistic of the two.
* A setup decided on the CLOSE of its bar is filled at the OPEN of the first 15m
  bar at or after that close. Nothing is filled at the price that triggered it.
* Daily context reaches an intraday bar through regime.attach, which joins on
  the daily bar's CLOSE. See regime.py.
* Every refusal is logged with a reason, same as 001, so "why did it not trade"
  is answerable rather than guessed at.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dataset import atr, ema, tag_sessions
import regime as regime_mod

# ----------------------------------------------------------------- params ----
# Fixed in advance. Not tuned, not swept. Changing anything here after seeing a
# result is the failure mode this project already paid for once.
PARAMS = {
    "f1": {
        "ema": 20,             # the pullback destination on 1H
        "touch_atr": 0.10,     # bar low must reach within this many ATRs of it
        "swing_bars": 3,       # stop sits under the last N bars
        "stop_atr": 0.50,      # ... plus this much ATR of air
        "no_chase_bars": 10,   # must be buying below the recent high, not chasing
    },
    "f2": {
        "channel": 20,         # prior N-bar extreme defines the break
        "range_mult": 1.50,    # breaking bar's range vs the N-bar average range
        "stop_atr": 0.25,      # stop beyond the breaking bar's own extreme
        "cooldown": 5,         # bars before the same direction can fire again
    },
    "f3": {
        "max_pierce_atr": 1.50,  # a rejection, not a collapse through the level
        "stop_atr": 0.25,
        "session_only": True,    # these levels only matter when the desks are on
    },
    "f4": {
        "ma": 20,
        "sd": 2.50,            # a genuinely stretched close, not a routine one
        "rsi_n": 2,
        "rsi_lo": 10.0,
        "rsi_hi": 90.0,
        "stop_atr": 0.50,
        "hold_h1": 24,         # time stop, in 1H bars
    },
    # applied to every family, so no family gets to trade an unusable stop
    "stop_min_atr": 0.30,
    "stop_max_atr": 4.00,
    "confirm_fraction": 1 / 3,   # close must sit in this third of the bar's range
}


# ------------------------------------------------------------- helpers -------
def rsi(s: pd.Series, n: int) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _confirms(o, h, l, c, direction) -> bool:
    """Close in the top (or bottom) third of the bar's own range, our way."""
    rng = h - l
    if rng <= 0:
        return False
    f = PARAMS["confirm_fraction"]
    if direction > 0:
        return c > o and (c - l) / rng >= 1 - f
    return c < o and (h - c) / rng >= 1 - f


def candle_stats(o, h, l, c) -> dict:
    rng = h - l
    if rng <= 0:
        return {"body_pct": 0.0, "upper_wick_pct": 0.0, "lower_wick_pct": 0.0,
                "close_pos": 0.5, "range": 0.0}
    return {"body_pct": round(abs(c - o) / rng, 4),
            "upper_wick_pct": round((h - max(o, c)) / rng, 4),
            "lower_wick_pct": round((min(o, c) - l) / rng, 4),
            "close_pos": round((c - l) / rng, 4),
            "range": round(rng, 3)}


def prepare(m1: pd.DataFrame, m15: pd.DataFrame, d1: pd.DataFrame):
    """Build the 1H frame and hang the daily regime off both intraday frames."""
    import dataset
    h1 = dataset.resample(m1, "1h")
    h1 = h1.join(tag_sessions(h1.index))
    h1 = regime_mod.attach(h1, d1)
    m15 = regime_mod.attach(m15, d1)
    return h1, m15


class _Collector:
    """Accumulates signals and refusals, and maps a decision time to a fill bar."""

    def __init__(self, m15_index: pd.DatetimeIndex, family: str):
        self.idx = m15_index
        self.family = family
        self.sig: list[dict] = []
        self.rej: list[dict] = []

    def fill_bar(self, decided_at: pd.Timestamp):
        """First 15m bar open at or after the decision. None if we ran off the end."""
        p = self.idx.searchsorted(decided_at, side="left")
        return self.idx[p] if p < len(self.idx) else None

    def refuse(self, ts, direction, reason, **extra):
        self.rej.append({"ts": ts, "family": self.family, "direction": direction,
                         "reason": reason, **extra})

    def add(self, *, ts, direction, stop, ref, atr_v, setup, **extra) -> bool:
        """Apply the stop sanity gate common to all families, then record."""
        dist = abs(ref - stop)
        if not np.isfinite(stop) or dist <= 0:
            return False
        if dist < PARAMS["stop_min_atr"] * atr_v:
            self.refuse(ts, direction, "stop_too_tight",
                        stop_distance=round(dist, 3), atr=round(atr_v, 3))
            return False
        if dist > PARAMS["stop_max_atr"] * atr_v:
            self.refuse(ts, direction, "stop_too_wide",
                        stop_distance=round(dist, 3), atr=round(atr_v, 3))
            return False

        entry_ts = self.fill_bar(ts)
        if entry_ts is None:
            return False

        self.sig.append({
            "signal_ts": ts, "entry_ts": entry_ts, "direction": direction,
            "stop": round(float(stop), 3), "stop_distance": round(float(dist), 3),
            "ref_close": round(float(ref), 3), "atr": round(float(atr_v), 3),
            "family": self.family, "setup": setup, "zone_id": -1,
            **extra,
        })
        return True

    def frames(self):
        sig = pd.DataFrame(self.sig)
        if not sig.empty:
            sig = (sig.drop_duplicates(subset=["signal_ts", "direction"])
                      .sort_values("signal_ts").reset_index(drop=True))
        return sig, pd.DataFrame(self.rej)


CTX_COLS = ("session", "regime", "direction", "trendiness", "adx", "vol_pctile", "spread")


def ctx_arrays(frame: pd.DataFrame) -> dict:
    """
    Pull the descriptive columns out as numpy arrays once.

    Reading them per bar with .iloc[i] is fine on 26,000 1H bars and painful on
    400,000 15m ones, and F3 runs on the 15m frame.
    """
    d = {c: frame[c].to_numpy() for c in CTX_COLS if c in frame.columns}
    d["in_session"] = frame["in_session"].to_numpy() if "in_session" in frame else None
    return d


def _ctx(ca: dict, i: int) -> dict:
    """The descriptive columns every signal carries, whatever the family."""
    return {"session": ca["session"][i], "regime": ca["regime"][i],
            "direction_regime": ca["direction"][i], "trendiness": ca["trendiness"][i],
            "adx": ca["adx"][i], "vol_pctile": ca["vol_pctile"][i],
            "spread": round(float(ca["spread"][i]), 4)}


# ------------------------------------------------------- F1 trend pullback ---
def f1_trend_pullback(h1: pd.DataFrame, m15_index: pd.DatetimeIndex):
    """
    Buy the dip in an uptrend, sell the rally in a downtrend.

    Daily direction decides which side we are allowed to take at all. On 1H,
    price has to actually reach the 20 EMA and then close back away from it in
    the trend's direction, in the top (or bottom) third of its own range. The
    no-chase rule keeps us from calling a fresh breakout a "pullback".
    """
    p = PARAMS["f1"]
    c = _Collector(m15_index, "f1_trend_pullback")
    step = pd.Timedelta("1h")

    e = ema(h1.c, p["ema"]).values
    a = atr(h1, 14).values
    lo, hi, op, cl = h1.l.values, h1.h.values, h1.o.values, h1.c.values
    swing_lo = h1.l.rolling(p["swing_bars"]).min().values
    swing_hi = h1.h.rolling(p["swing_bars"]).max().values
    recent_hi = h1.h.rolling(p["no_chase_bars"]).max().shift(1).values
    recent_lo = h1.l.rolling(p["no_chase_bars"]).min().shift(1).values
    ca = ctx_arrays(h1)

    for i in range(len(h1)):
        av = a[i]
        if not np.isfinite(av) or av <= 0 or not np.isfinite(e[i]):
            continue
        if ca["direction"][i] == "bull":
            direction = 1
        elif ca["direction"][i] == "bear":
            direction = -1
        else:
            continue

        ts = h1.index[i]
        if direction > 0:
            reached = lo[i] <= e[i] + p["touch_atr"] * av
            back = cl[i] > e[i]
            not_chasing = np.isfinite(recent_hi[i]) and cl[i] < recent_hi[i]
            stop = swing_lo[i] - p["stop_atr"] * av
        else:
            reached = hi[i] >= e[i] - p["touch_atr"] * av
            back = cl[i] < e[i]
            not_chasing = np.isfinite(recent_lo[i]) and cl[i] > recent_lo[i]
            stop = swing_hi[i] + p["stop_atr"] * av

        if not (reached and back):
            continue
        if not not_chasing:
            c.refuse(ts, direction, "chasing_extreme")
            continue
        if not _confirms(op[i], hi[i], lo[i], cl[i], direction):
            c.refuse(ts, direction, "no_close_confirmation")
            continue

        c.add(ts=ts + step, direction=direction, stop=stop, ref=cl[i], atr_v=av,
              setup="pullback_to_ema",
              structural_stop=round(float(swing_lo[i] if direction > 0 else swing_hi[i]), 3),
              atr_buffer=round(float(p["stop_atr"] * av), 3),
              dist_to_ema_atr=round(float(abs(cl[i] - e[i]) / av), 3),
              **_ctx(ca, i), **candle_stats(op[i], hi[i], lo[i], cl[i]))

    return c.frames()


# --------------------------------------------------- F2 momentum breakout ----
def f2_momentum_breakout(h1: pd.DataFrame, m15_index: pd.DatetimeIndex):
    """
    Break of the prior 20-bar extreme, but only on a bar whose range is at least
    1.5x the recent average - so it takes a genuine volatility expansion, not a
    quiet drift through the level.

    No retest is required. That single decision is the main reason 001A starved:
    only 38% of its breakouts were ever retested inside the window, so the
    strategy spent most of its life waiting for a second event that never came.

    Both directions are allowed in every regime on purpose. Whether breakouts
    only pay in trending markets is exactly the question the regime matrix is
    there to answer, so it must not be assumed here.
    """
    p = PARAMS["f2"]
    c = _Collector(m15_index, "f2_momentum_breakout")
    step = pd.Timedelta("1h")

    a = atr(h1, 14).values
    lo, hi, op, cl = h1.l.values, h1.h.values, h1.o.values, h1.c.values
    ch_hi = h1.h.rolling(p["channel"]).max().shift(1).values
    ch_lo = h1.l.rolling(p["channel"]).min().shift(1).values
    avg_rng = (h1.h - h1.l).rolling(p["channel"]).mean().shift(1).values
    bar_rng = (hi - lo)
    ca = ctx_arrays(h1)

    last = {1: -10 ** 9, -1: -10 ** 9}
    for i in range(len(h1)):
        av = a[i]
        if not np.isfinite(av) or av <= 0 or not np.isfinite(ch_hi[i]) or not np.isfinite(avg_rng[i]):
            continue
        ts = h1.index[i]

        for direction in (1, -1):
            broke = cl[i] > ch_hi[i] if direction > 0 else cl[i] < ch_lo[i]
            if not broke:
                continue
            if i - last[direction] < p["cooldown"]:
                c.refuse(ts, direction, "cooldown")
                continue
            if bar_rng[i] < p["range_mult"] * avg_rng[i]:
                c.refuse(ts, direction, "no_volatility_expansion",
                         range_mult=round(float(bar_rng[i] / avg_rng[i]), 2))
                continue
            if not _confirms(op[i], hi[i], lo[i], cl[i], direction):
                c.refuse(ts, direction, "no_close_confirmation")
                continue

            structural = lo[i] if direction > 0 else hi[i]
            stop = structural - direction * p["stop_atr"] * av
            if c.add(ts=ts + step, direction=direction, stop=stop, ref=cl[i], atr_v=av,
                     setup="channel_break",
                     structural_stop=round(float(structural), 3),
                     atr_buffer=round(float(p["stop_atr"] * av), 3),
                     range_mult=round(float(bar_rng[i] / avg_rng[i]), 2),
                     channel_level=round(float(ch_hi[i] if direction > 0 else ch_lo[i]), 3),
                     **_ctx(ca, i), **candle_stats(op[i], hi[i], lo[i], cl[i])):
                last[direction] = i

    return c.frames()


# ------------------------------------------------------ F3 level rejection ---
def f3_level_rejection(m15: pd.DataFrame, d1: pd.DataFrame):
    """
    The redesign of 001B.

    001B's zones were built from swing pivots, which are a chart construct - the
    market has no idea they exist. This version only uses levels that real
    orders genuinely rest at and that every desk is looking at the same way:

        prior day high, prior day low
        prior week high, prior week low
        today's opening price

    A rejection is a bar that pierces the level and closes back on the original
    side, in the top or bottom third of its range, during a cash session. The
    pierce has to be a rejection rather than a collapse, so it is capped at 1.5
    ATR. Each level fires at most once per day per direction.
    """
    p = PARAMS["f3"]
    c = _Collector(m15.index, "f3_level_rejection")

    # Daily levels, each stamped with the moment it becomes knowable.
    dd = d1.copy()
    dd["pdh"], dd["pdl"] = dd.h.shift(1), dd.l.shift(1)
    wk = d1.resample("W").agg({"h": "max", "l": "min"})
    # A weekly bar labelled at its own week-end is not knowable until that week
    # has actually finished, so push the label forward a day before forward
    # filling. Without this a Sunday would see its own week's extremes.
    wk.index = wk.index + pd.Timedelta("1D")
    dd["pwh"] = wk.h.reindex(dd.index, method="ffill")
    dd["pwl"] = wk.l.reindex(dd.index, method="ffill")
    # prior day / prior week levels are known from the day's first tick; the
    # day's own open is known once the day has opened, which is the same instant
    dd["dopen"] = dd.o
    lv = dd[["pdh", "pdl", "pwh", "pwl", "dopen"]].copy()
    lv["day"] = lv.index.tz_convert("UTC").date

    by_day = {r.day: r for r in lv.itertuples()}

    a = atr(m15, 14).values
    lo, hi, op, cl = m15.l.values, m15.h.values, m15.o.values, m15.c.values
    prev_cl = m15.c.shift(1).values
    days = m15.index.tz_convert("UTC").date
    fired: set = set()
    ca = ctx_arrays(m15)
    in_sess = ca["in_session"]

    for i in range(len(m15)):
        av = a[i]
        if not np.isfinite(av) or av <= 0 or not np.isfinite(prev_cl[i]):
            continue
        if p["session_only"] and not in_sess[i]:
            continue

        lvrow = by_day.get(days[i])
        if lvrow is None:
            continue
        ts = m15.index[i]

        for name in ("pdh", "pdl", "pwh", "pwl", "dopen"):
            L = getattr(lvrow, name)
            if not np.isfinite(L):
                continue

            # bullish: the level was support, price dipped below it and closed back above
            if prev_cl[i] > L and lo[i] < L <= cl[i]:
                direction = 1
                pierce = L - lo[i]
            # bearish: the level was resistance, price poked above and closed back below
            elif prev_cl[i] < L and hi[i] > L >= cl[i]:
                direction = -1
                pierce = hi[i] - L
            else:
                continue

            key = (days[i], name, direction)
            if key in fired:
                continue
            if pierce > p["max_pierce_atr"] * av:
                c.refuse(ts, direction, "pierce_too_deep", level=name,
                         pierce_atr=round(float(pierce / av), 2))
                continue
            if not _confirms(op[i], hi[i], lo[i], cl[i], direction):
                c.refuse(ts, direction, "no_close_confirmation", level=name)
                continue

            structural = lo[i] if direction > 0 else hi[i]
            stop = structural - direction * p["stop_atr"] * av
            if c.add(ts=ts + pd.Timedelta("15min"), direction=direction, stop=stop,
                     ref=cl[i], atr_v=av, setup=f"reject_{name}",
                     structural_stop=round(float(structural), 3),
                     atr_buffer=round(float(p["stop_atr"] * av), 3),
                     level=name, level_price=round(float(L), 3),
                     pierce_atr=round(float(pierce / av), 3),
                     **_ctx(ca, i), **candle_stats(op[i], hi[i], lo[i], cl[i])):
                fired.add(key)

    return c.frames()


# ------------------------------------------------------- F4 mean reversion ---
def f4_mean_reversion(h1: pd.DataFrame, m15_index: pd.DatetimeIndex):
    """
    Fade a stretched 1H close, but ONLY when the daily chart is objectively
    ranging (ADX under 20). This is the family that most needs the regime layer:
    fading a trending market is how accounts die, and the gate is the strategy.

    Exit is the band midpoint, not an R multiple. Mean reversion aims at a place,
    not a distance, and forcing a 2R target on it would be testing a different
    strategy and then blaming this one for the result. engine.Config carries
    exit_mode="level" for exactly this.
    """
    p = PARAMS["f4"]
    c = _Collector(m15_index, "f4_mean_reversion")
    step = pd.Timedelta("1h")

    ma = h1.c.rolling(p["ma"]).mean()
    sd = h1.c.rolling(p["ma"]).std()
    upper, lower = (ma + p["sd"] * sd).values, (ma - p["sd"] * sd).values
    mid = ma.values
    r = rsi(h1.c, p["rsi_n"]).values
    a = atr(h1, 14).values
    lo, hi, op, cl = h1.l.values, h1.h.values, h1.o.values, h1.c.values
    ca = ctx_arrays(h1)

    for i in range(len(h1)):
        av = a[i]
        if not np.isfinite(av) or av <= 0 or not np.isfinite(mid[i]) or not np.isfinite(r[i]):
            continue
        ts = h1.index[i]

        if cl[i] < lower[i] and r[i] < p["rsi_lo"]:
            direction = 1
        elif cl[i] > upper[i] and r[i] > p["rsi_hi"]:
            direction = -1
        else:
            continue

        if ca["trendiness"][i] != "ranging":
            c.refuse(ts, direction, "regime_not_ranging",
                     trendiness=ca["trendiness"][i], adx=ca["adx"][i])
            continue

        structural = lo[i] if direction > 0 else hi[i]
        stop = structural - direction * p["stop_atr"] * av
        target = float(mid[i])
        if (target - cl[i]) * direction <= 0:
            c.refuse(ts, direction, "target_wrong_side")
            continue

        c.add(ts=ts + step, direction=direction, stop=stop, ref=cl[i], atr_v=av,
              setup="band_fade", target=round(target, 3),
              structural_stop=round(float(structural), 3),
              atr_buffer=round(float(p["stop_atr"] * av), 3),
              band_width_atr=round(float((upper[i] - lower[i]) / av), 2),
              rsi=round(float(r[i]), 1),
              **_ctx(ca, i), **candle_stats(op[i], hi[i], lo[i], cl[i]))

    return c.frames()


FAMILIES = {
    "f1_trend_pullback": ("h1", f1_trend_pullback),
    "f2_momentum_breakout": ("h1", f2_momentum_breakout),
    "f3_level_rejection": ("m15", f3_level_rejection),
    "f4_mean_reversion": ("h1", f4_mean_reversion),
}
