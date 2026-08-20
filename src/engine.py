#!/usr/bin/env python3
"""
Trade simulation and performance metrics.

Execution rules, all chosen to err against the strategy rather than for it:

  * A signal decided on the close of a 15m bar is filled at the OPEN of the
    next 15m bar. Nothing is filled at the price that triggered it.
  * Buys pay the ASK, sells receive the BID - real quoted prices from the
    tick data, never the mid.
  * Entries and stop exits carry additional slippage. Target exits do not,
    because a take-profit is a resting limit order.
  * Once in a trade, the minute bars are walked to see which of the stop and
    the target was reached first. If BOTH sit inside the same minute, the
    trade is recorded as a loss. Assuming the good one came first is the
    classic way a backtest flatters itself.
  * One position at a time. A signal arriving while a trade is open is skipped
    and counted, not queued.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SLIPPAGE = 0.05          # dollars per ounce, on entry and on stopped exits
MAX_HOLD_BARS = 96 * 3   # a trade is abandoned after three days at the M15 close


@dataclass
class Config:
    r_multiple: float
    risk_pct: float
    start_equity: float = 10_000.0
    slippage: float = SLIPPAGE
    # Research 002 additions. Both default to the 001 behaviour so the frozen
    # 001A run still reproduces bar for bar.
    max_hold_bars: int = MAX_HOLD_BARS
    exit_mode: str = "r_target"   # "level" = the signal carries its own target


def _walk(m1: pd.DataFrame, direction: int, entry: float, stop: float, target: float,
          t_from: pd.Timestamp, t_to: pd.Timestamp, slip: float):
    """
    Find the first minute at which the stop or the target is reached.
    Longs are closed against the BID, shorts against the ASK - you exit a long
    by selling, and you sell at the bid.
    """
    seg = m1.loc[t_from:t_to]
    if seg.empty:
        return None
    if direction > 0:
        hit_stop = seg.bid_l.values <= stop
        hit_tgt = seg.bid_h.values >= target
    else:
        hit_stop = seg.ask_h.values >= stop
        hit_tgt = seg.ask_l.values <= target

    both = hit_stop | hit_tgt
    if not both.any():
        return None
    i = int(np.argmax(both))
    ts = seg.index[i]
    if hit_stop[i]:
        # stops slip against you; that is what a stop is
        px = stop - slip if direction > 0 else stop + slip
        return ts, float(px), "stop"
    return ts, float(target), "target"


def simulate(signals: pd.DataFrame, m15: pd.DataFrame, m1: pd.DataFrame,
             cfg: Config) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()

    equity = cfg.start_equity
    open_until: pd.Timestamp | None = None
    skipped = 0
    rows = []

    m15_idx = m15.index
    for s in signals.itertuples():
        t_entry = s.entry_ts
        if t_entry not in m15.index:
            continue
        if open_until is not None and t_entry <= open_until:
            skipped += 1
            continue

        bar = m15.loc[t_entry]
        # buy at the ask, sell at the bid, then add slippage on top
        if s.direction > 0:
            entry = float(bar.ask_o) + cfg.slippage
        else:
            entry = float(bar.bid_o) - cfg.slippage

        stop = float(s.stop)
        risk_per_oz = abs(entry - stop)
        if risk_per_oz <= 0:
            continue
        if cfg.exit_mode == "level":
            # Mean reversion exits at a price the market defines (the band
            # midpoint), not at a multiple of the stop. Forcing an R target on
            # it would handicap the family rather than test it.
            target = float(getattr(s, "target", float("nan")))
            if not np.isfinite(target) or (target - entry) * s.direction <= 0:
                continue
        else:
            target = entry + s.direction * cfg.r_multiple * risk_per_oz

        # position size straight from the risk budget
        risk_cash = equity * cfg.risk_pct / 100.0
        size = risk_cash / risk_per_oz

        pos = m15_idx.get_loc(t_entry)
        t_limit = m15_idx[min(pos + cfg.max_hold_bars, len(m15_idx) - 1)]
        res = _walk(m1, s.direction, entry, stop, target, t_entry, t_limit, cfg.slippage)

        if res is None:
            exit_ts = t_limit
            exit_px = float(m15.loc[t_limit].bid_c if s.direction > 0 else m15.loc[t_limit].ask_c)
            reason = "timeout"
        else:
            exit_ts, exit_px, reason = res

        pnl = (exit_px - entry) * s.direction * size
        equity += pnl
        open_until = exit_ts

        row = {
            "signal_ts": s.signal_ts, "entry_ts": t_entry, "exit_ts": exit_ts,
            "direction": "BUY" if s.direction > 0 else "SELL",
            "session": s.session, "setup": s.setup, "zone_id": s.zone_id,
            "entry": round(entry, 3), "stop": round(stop, 3), "target": round(target, 3),
            "exit": round(exit_px, 3), "reason": reason,
            "risk_per_oz": round(risk_per_oz, 3), "size_oz": round(size, 4),
            "pnl": round(pnl, 2), "r": round(pnl / risk_cash, 3) if risk_cash else 0.0,
            "equity": round(equity, 2),
            "bars_held": int(m15_idx.get_loc(exit_ts) - pos) if exit_ts in m15.index else np.nan,
        }
        # carry the setup's own detail into the trade log so a trade can be
        # reviewed later without joining back to the signals file
        for f in ("structural_stop", "atr_buffer", "stop_distance", "atr15", "spread",
                  "zone_lo", "zone_hi", "zone_touches", "regime", "bars_to_retest",
                  "body_pct", "upper_wick_pct", "lower_wick_pct", "close_pos", "range"):
            if hasattr(s, f):
                row[f] = getattr(s, f)
        # Research 002 families invent their own descriptive columns (level_kind,
        # trendiness, adx, band width and so on). Carry anything else through
        # rather than making every new family edit this list.
        for f in signals.columns:
            if f not in row and f not in ("direction", "stop", "entry_ts", "signal_ts"):
                row[f] = getattr(s, f, None)
        rows.append(row)

    out = pd.DataFrame(rows)
    out.attrs["skipped_overlapping"] = skipped
    return out


# ------------------------------------------------------------- metrics -------
def _streak(mask: pd.Series) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def bootstrap_expectancy(r: pd.Series, draws: int = 20_000, seed: int = 7) -> tuple[float, float, float]:
    """
    Resample the trade list with replacement and see how often the edge survives.

    A handful of trades can produce a flattering average by luck alone, and a
    profit factor quoted off 29 trades tells you almost nothing. This returns a
    95% interval for expectancy and the share of resamples that came out at or
    below zero - effectively, the chance this "edge" is noise.
    """
    if len(r) < 5:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sample = rng.choice(r.to_numpy(), size=(draws, len(r)), replace=True).mean(axis=1)
    return (float(np.percentile(sample, 2.5)),
            float(np.percentile(sample, 97.5)),
            float((sample <= 0).mean()))


def metrics(trades: pd.DataFrame, cfg: Config) -> dict:
    if trades.empty:
        return {"trades": 0}
    wins = trades[trades.pnl > 0]
    losses = trades[trades.pnl <= 0]
    gross_win = wins.pnl.sum()
    gross_loss = -losses.pnl.sum()

    eq = pd.concat([pd.Series([cfg.start_equity]), trades.equity], ignore_index=True)
    peak = eq.cummax()
    dd = (eq - peak) / peak
    net = trades.equity.iloc[-1] - cfg.start_equity

    days = (trades.exit_ts.max() - trades.entry_ts.min()).days or 1
    years = days / 365.25
    ci_lo, ci_hi, p_noise = bootstrap_expectancy(trades.r)

    return {
        "exp_R_ci_low": round(ci_lo, 4),
        "exp_R_ci_high": round(ci_hi, 4),
        "prob_no_edge": round(p_noise, 4),
        "trades": int(len(trades)),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate_pct": round(100 * len(wins) / len(trades), 2),
        "profit_factor": round(gross_win / gross_loss, 3) if gross_loss > 0 else float("inf"),
        "expectancy_R": round(trades.r.mean(), 4),
        "expectancy_cash": round(trades.pnl.mean(), 2),
        "net_return_pct": round(100 * net / cfg.start_equity, 2),
        "cagr_pct": round(100 * ((trades.equity.iloc[-1] / cfg.start_equity) ** (1 / years) - 1), 2)
        if years > 0.2 else None,
        "max_drawdown_pct": round(100 * dd.min(), 2),
        "worst_losing_streak": _streak(trades.pnl <= 0),
        "best_winning_streak": _streak(trades.pnl > 0),
        "avg_win": round(wins.pnl.mean(), 2) if len(wins) else 0.0,
        "avg_loss": round(losses.pnl.mean(), 2) if len(losses) else 0.0,
        "avg_win_R": round(wins.r.mean(), 3) if len(wins) else 0.0,
        "avg_loss_R": round(losses.r.mean(), 3) if len(losses) else 0.0,
        "avg_bars_held": round(trades.bars_held.mean(), 1),
        "timeouts": int((trades.reason == "timeout").sum()),
        # setups that were valid but arrived while another trade was still open
        "skipped_overlapping": int(trades.attrs.get("skipped_overlapping", 0)),
        "final_equity": round(trades.equity.iloc[-1], 2),
    }


def split_table(trades: pd.DataFrame, by: str) -> pd.DataFrame:
    """Per-group breakdown - used for London vs New York, BUY vs SELL, and by year."""
    if trades.empty:
        return pd.DataFrame()
    g = trades.groupby(by)
    out = pd.DataFrame({
        "trades": g.size(),
        "win_rate_pct": (100 * g.pnl.apply(lambda s: (s > 0).mean())).round(2),
        "expectancy_R": g.r.mean().round(4),
        "net_pnl": g.pnl.sum().round(2),
    })
    pf = g.pnl.apply(lambda s: s[s > 0].sum() / -s[s <= 0].sum() if (s <= 0).any() else np.inf)
    out["profit_factor"] = pf.round(3)
    return out.reset_index()
