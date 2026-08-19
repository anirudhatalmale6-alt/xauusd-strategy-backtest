#!/usr/bin/env python3
"""
IN-SAMPLE ONLY. This script cannot see the sealed period - it is not given the
dates, and dataset.build() is called with an end of 2024-12-31.

It answers the three comparisons asked for, and nothing else:

  1. Does the 200 EMA regime filter add value, or just remove trades?
  2. Does the London / New York session filter earn its place?
  3. How long do retests actually take, for the ones that complete and the
     ones that expire?

Deliberately NOT done here: searching for better parameters. Every threshold is
the one written in the brief. The comparisons above are two-way A/B tests, not
an optimiser, and the thing being looked for is consistency across years rather
than the best headline number.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset
import engine
from strategy import generate_signals

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out_insample"

IS_START, IS_END = "2015-01-01", "2024-12-31"      # the sealed period starts the day after

R_TARGETS = [1.0, 1.5, 2.0, 2.5, 3.0]
RISKS = [0.25, 0.5]
REFERENCE = (2.0, 0.25)        # the fixed point used for all A/B comparisons
STRATEGIES = {"001A": "A", "001B": "B"}
TREND_MODES = ["ema50", "ema50_200"]
SESSION_MODES = [True, False]


def run_one(sig, m15, m1, r, risk):
    cfg = engine.Config(r_multiple=r, risk_pct=risk)
    tr = engine.simulate(sig, m15, m1, cfg)
    return tr, engine.metrics(tr, cfg)


def yearly(tr: pd.DataFrame) -> pd.DataFrame:
    if tr.empty:
        return pd.DataFrame()
    x = tr.copy()
    x["year"] = x.entry_ts.dt.year
    return engine.split_table(x, "year")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    print("loading in-sample data (2015-01-01 .. 2024-12-31) ...", flush=True)
    d = dataset.build(IS_START, IS_END)
    m1, m15, h4, d1 = d["m1"], d["m15"], d["h4"], d["d1"]
    print(f"  M1 {len(m1):,}  M15 {len(m15):,}  H4 {len(h4):,}  D1 {len(d1):,}")
    print(f"  {m1.index.min()}  ->  {m1.index.max()}\n")

    grid_rows, ab_rows, retest_rows = [], [], []
    cache = {}

    for name, mode in STRATEGIES.items():
        for tmode in TREND_MODES:
            for sess_on in SESSION_MODES:
                sig, rej = generate_signals(m15, h4, d1, mode, tmode, sess_on)
                tag = f"{name}_{tmode}_{'session' if sess_on else 'allhours'}"
                cache[(name, tmode, sess_on)] = (sig, rej)
                n = len(sig)
                print(f"{tag:34s} {n:4d} setups   {len(rej):5d} refused", flush=True)
                if n == 0:
                    continue
                sig.to_csv(OUT / f"signals_{tag}.csv", index=False)
                rej.to_csv(OUT / f"rejected_{tag}.csv", index=False)

                for r in R_TARGETS:
                    for risk in RISKS:
                        tr, m = run_one(sig, m15, m1, r, risk)
                        m.update({"strategy": name, "trend_mode": tmode,
                                  "session_filter": sess_on,
                                  "r_target": r, "risk_pct": risk})
                        grid_rows.append(m)
                        if (r, risk) == REFERENCE:
                            tr.to_csv(OUT / f"trades_{tag}.csv", index=False)
                            yr = yearly(tr)
                            if not yr.empty:
                                yr.to_csv(OUT / f"by_year_{tag}.csv", index=False)
                                pos_years = int((yr.expectancy_R > 0).sum())
                                m2 = dict(m)
                                m2.update({"years": len(yr),
                                           "positive_years": pos_years,
                                           "worst_year_R": float(yr.expectancy_R.min()),
                                           "best_year_R": float(yr.expectancy_R.max())})
                                ab_rows.append(m2)
                            for by, fname in (("session", "by_session"),
                                              ("direction", "by_direction"),
                                              ("regime", "by_regime")):
                                if by in tr.columns:
                                    engine.split_table(tr, by).to_csv(
                                        OUT / f"{fname}_{tag}.csv", index=False)

        # ---- retest timing, question 3 (only 001A has retests) ----
        if mode == "A":
            for tmode in TREND_MODES:
                sig, rej = cache[(name, tmode, True)]
                done = sig.bars_to_retest.dropna() if "bars_to_retest" in sig else pd.Series(dtype=float)
                exp = (rej[rej.reason == "retest_expired"].bars_to_retest.dropna()
                       if "reason" in rej and not rej.empty else pd.Series(dtype=float))
                retest_rows.append({
                    "trend_mode": tmode,
                    "completed": int(len(done)),
                    "expired": int(len(exp)),
                    "completion_rate_pct": round(100 * len(done) / max(1, len(done) + len(exp)), 2),
                    "bars_median": float(done.median()) if len(done) else None,
                    "bars_mean": round(float(done.mean()), 2) if len(done) else None,
                    "bars_p90": float(done.quantile(.90)) if len(done) else None,
                    "bars_max": float(done.max()) if len(done) else None,
                })
                if len(done):
                    dist = done.value_counts().sort_index()
                    dist.rename("retests").to_frame().to_csv(
                        OUT / f"retest_bars_{name}_{tmode}.csv")

    grid = pd.DataFrame(grid_rows)
    cols = ["strategy", "trend_mode", "session_filter", "r_target", "risk_pct",
            "trades", "win_rate_pct", "profit_factor", "expectancy_R",
            "exp_R_ci_low", "exp_R_ci_high", "prob_no_edge", "net_return_pct",
            "max_drawdown_pct", "worst_losing_streak", "timeouts",
            "skipped_overlapping", "final_equity"]
    grid = grid[[c for c in cols if c in grid.columns]]
    grid.to_csv(OUT / "insample_grid.csv", index=False)

    if ab_rows:
        ab = pd.DataFrame(ab_rows)
        abc = ["strategy", "trend_mode", "session_filter", "trades", "win_rate_pct",
               "profit_factor", "expectancy_R", "exp_R_ci_low", "exp_R_ci_high",
               "prob_no_edge", "years", "positive_years", "worst_year_R",
               "best_year_R", "max_drawdown_pct"]
        ab = ab[[c for c in abc if c in ab.columns]]
        ab.to_csv(OUT / "ab_comparison.csv", index=False)
        print(f"\n=== A/B comparison at {REFERENCE[0]}R, {REFERENCE[1]}% risk ===")
        print(ab.to_string(index=False))

    if retest_rows:
        rt = pd.DataFrame(retest_rows)
        rt.to_csv(OUT / "retest_timing.csv", index=False)
        print("\n=== 001A retest timing (12-bar window) ===")
        print(rt.to_string(index=False))

    print("\n=== full in-sample grid ===")
    print(grid.to_string(index=False))

    (OUT / "insample_meta.json").write_text(json.dumps({
        "in_sample_start": IS_START, "in_sample_end": IS_END,
        "sealed_period": "2025-01-01 .. 2026-08-19 - NOT run by this script",
        "reference_config": {"r_target": REFERENCE[0], "risk_pct": REFERENCE[1]},
        "note": "no parameter search was performed; all thresholds are the ones "
                "written in the brief. The only variations are the two A/B "
                "comparisons requested.",
    }, indent=2))
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
