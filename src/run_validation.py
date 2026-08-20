#!/usr/bin/env python3
"""
VALIDATION SLICE: 2020-01-01 to 2022-12-31. All four families, one run each.

This is the slice where families are compared and the shortlist is cut. It is
allowed to be looked at more than once - that is what it is for, and it is why
Jan 2023 to Dec 2024 is not loaded anywhere in this file.

Why 2020-2022 and not the most recent three years: it is the only stretch in the
whole dataset that contains a volatility spike inside an uptrend (2020), a
genuine bear year (2021) and a sideways grind (2022). A validation slice made of
three bull years would tell us which family likes bull markets, which we already
know.

Warm-up runs from 2015-01-01 so the 200-day EMA, the ADX and the expanding
volatility percentile are identical to what they would be in a full-history run.
Every setup dated before 2020-01-01 is then discarded. That is look-back over a
slice we have already used for development, which is legitimate; nothing here
looks forward, and the confirmation period is never opened.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import costs
import dataset
import engine
import families

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out_val"

WARMUP_START = "2015-01-01"
VAL_START, VAL_END = "2020-01-01", "2022-12-31"
CONFIRMATION_IS_SEALED = ("2023-01-01", "2024-12-31")   # never loaded here

RISK_PCT = 0.25
R_TARGET = 2.0
MIN_CELL_TRADES = 30      # below this a regime cell is reported but not decided on


def cfg_for(name: str, slip_mult: float) -> engine.Config:
    return engine.Config(
        r_multiple=R_TARGET, risk_pct=RISK_PCT,
        slippage=engine.SLIPPAGE * slip_mult,
        exit_mode="level" if name == "f4_mean_reversion" else "r_target",
        max_hold_bars=(families.PARAMS["f4"]["hold_h1"] * 4
                       if name == "f4_mean_reversion" else engine.MAX_HOLD_BARS),
    )


def regime_matrix(trades: pd.DataFrame, family: str) -> pd.DataFrame:
    """Trade count, expectancy and a bootstrap interval for every regime cell."""
    rows = []
    for regime, g in trades.groupby("regime"):
        lo, hi, p = engine.bootstrap_expectancy(g.r)
        rows.append({
            "family": family, "regime": regime, "trades": len(g),
            "win_rate_pct": round(100 * (g.pnl > 0).mean(), 1),
            "expectancy_R": round(g.r.mean(), 4),
            "ci_low": round(lo, 4) if np.isfinite(lo) else None,
            "ci_high": round(hi, 4) if np.isfinite(hi) else None,
            "prob_no_edge": round(p, 3) if np.isfinite(p) else None,
            "net_pnl": round(g.pnl.sum(), 2),
            "decision_grade": len(g) >= MIN_CELL_TRADES,
        })
    return pd.DataFrame(rows).sort_values("regime")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    assert pd.Timestamp(VAL_END) < pd.Timestamp(CONFIRMATION_IS_SEALED[0]), \
        "validation must end before the confirmation period begins"

    print(f"loading {WARMUP_START} .. {VAL_END} (warm-up + validation slice) ...", flush=True)
    d = dataset.build(WARMUP_START, VAL_END)
    m1, m15, d1 = d["m1"], d["m15"], d["d1"]
    print(f"  M1 {len(m1):,}  M15 {len(m15):,}  D1 {len(d1):,}")
    print(f"  {m1.index.min()}  ->  {m1.index.max()}")

    h1, m15r = families.prepare(m1, m15, d1)
    print(f"  H1 {len(h1):,}  regime attached\n")

    # cost variants, built once and reused across families
    variants = {}
    for label, ks, _ in costs.SCENARIOS:
        variants[label] = (costs.rescale(m1, ks), costs.rescale(m15, ks))

    cut = pd.Timestamp(VAL_START, tz="UTC")
    years = (pd.Timestamp(VAL_END, tz="UTC") - cut).days / 365.25

    headline_rows, cost_rows, matrix_rows = [], [], []
    splits = {k: [] for k in ("year", "direction", "regime", "session",
                              "trendiness", "reason")}

    for name, (_kind, fn) in families.FAMILIES.items():
        print(f"=== {name} ===", flush=True)
        sig, rej = fn(m15r, d1) if name == "f3_level_rejection" else fn(h1, m15.index)

        n_warm = int((sig.signal_ts < cut).sum()) if not sig.empty else 0
        if not sig.empty:
            sig = sig[sig.signal_ts >= cut].reset_index(drop=True)
        if not rej.empty:
            rej = rej[rej.ts >= cut].reset_index(drop=True)
        sig.to_csv(OUT / f"signals_{name}.csv", index=False)
        rej.to_csv(OUT / f"rejected_{name}.csv", index=False)
        print(f"  {len(sig):,} setups ({n_warm} warm-up discarded), {len(rej):,} refused")

        if sig.empty:
            continue

        # ---- every cost scenario, so "after costs" is a number not a claim ----
        headline_trades = None
        for label, _ks, slip in costs.SCENARIOS:
            v_m1, v_m15 = variants[label]
            c = cfg_for(name, slip)
            tr = engine.simulate(sig, v_m15, v_m1, c)
            if tr.empty:
                continue
            m = engine.metrics(tr, c)
            m.update({"family": name, "scenario": label})
            cost_rows.append(m)
            if label == costs.HEADLINE:
                headline_trades = tr

        tr = headline_trades
        if tr is None or tr.empty:
            print("  no trades at real spread")
            continue

        tr["year"] = tr.entry_ts.dt.year
        tr.to_csv(OUT / f"trades_{name}.csv", index=False)

        c = cfg_for(name, 1.0)
        m = engine.metrics(tr, c)
        m.update({"family": name, "trades_per_year": round(len(tr) / years, 1),
                  "setups": int(len(sig))})
        headline_rows.append(m)

        for by in splits:
            if by in tr.columns:
                t = engine.split_table(tr, by)
                t.insert(0, "family", name)
                splits[by].append(t)

        matrix_rows.append(regime_matrix(tr, name))

        print(f"  {len(tr):,} trades  {len(tr)/years:,.0f}/yr   "
              f"expectancy {m['expectancy_R']:+.4f}R  "
              f"CI [{m['exp_R_ci_low']:+.3f}, {m['exp_R_ci_high']:+.3f}]  "
              f"P(no edge) {m['prob_no_edge']:.3f}\n")

    # ------------------------------------------------------------- write ----
    head = pd.DataFrame(headline_rows)
    cols = ["family", "setups", "trades", "trades_per_year", "wins", "losses",
            "win_rate_pct", "profit_factor", "expectancy_R", "exp_R_ci_low",
            "exp_R_ci_high", "prob_no_edge", "net_return_pct", "max_drawdown_pct",
            "worst_losing_streak", "best_winning_streak", "avg_win_R", "avg_loss_R",
            "avg_bars_held", "timeouts", "skipped_overlapping", "final_equity"]
    head = head[[c for c in cols if c in head.columns]]
    head.to_csv(OUT / "validation_headline.csv", index=False)

    cost = pd.DataFrame(cost_rows)
    ccols = ["family", "scenario", "trades", "win_rate_pct", "profit_factor",
             "expectancy_R", "exp_R_ci_low", "exp_R_ci_high", "prob_no_edge",
             "net_return_pct", "max_drawdown_pct"]
    cost = cost[[c for c in ccols if c in cost.columns]]
    cost.to_csv(OUT / "validation_costs.csv", index=False)

    mat = pd.concat(matrix_rows, ignore_index=True) if matrix_rows else pd.DataFrame()
    mat.to_csv(OUT / "regime_matrix.csv", index=False)

    for by, frames in splits.items():
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(OUT / f"by_{by}.csv", index=False)

    (OUT / "validation_meta.json").write_text(json.dumps({
        "warmup_start": WARMUP_START,
        "validation": [VAL_START, VAL_END],
        "confirmation_period_untouched": list(CONFIRMATION_IS_SEALED),
        "risk_pct": RISK_PCT, "r_target": R_TARGET,
        "min_cell_trades_for_a_decision": MIN_CELL_TRADES,
        "params": families.PARAMS,
        "cost_scenarios": [{"label": a, "spread_mult": b, "slippage_mult": c}
                           for a, b, c in costs.SCENARIOS],
    }, indent=2, default=str))

    # ------------------------------------------------------------- print ----
    pd.set_option("display.width", 250)
    print("=== VALIDATION 2020-2022, headline (real spread, 2R, 0.25% risk) ===")
    print(head.to_string(index=False))
    print("\n=== transaction cost sensitivity ===")
    print(cost.to_string(index=False))
    print("\n=== regime matrix ===")
    print(mat.to_string(index=False))
    for by in ("year", "direction", "session"):
        p = OUT / f"by_{by}.csv"
        if p.exists():
            print(f"\n--- by {by} ---")
            print(pd.read_csv(p).to_string(index=False))
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
