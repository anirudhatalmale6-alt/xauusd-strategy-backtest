#!/usr/bin/env python3
"""
Run both strategies over the whole grid and write the results out.

The out-of-sample discipline matters more than any single number here:

  1. Everything is fitted, chosen and argued over on the IN-SAMPLE period only.
  2. From the in-sample results one configuration per strategy is picked - the
     "pre-registered" choice - and written down before the out-of-sample data
     is touched.
  3. The out-of-sample period is then run once. The full out-of-sample grid is
     also printed, but only so you can see whether the pre-registered pick was
     lucky or representative. It is not there to shop for a better number.

If the out-of-sample result is much worse than the in-sample one, the strategy
was fitted to the past. That is the single most useful thing this whole exercise
can tell us, and it is the reason for holding data back.
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
OUT = ROOT / "out"

IS_START, IS_END = "2021-08-01", "2024-12-31"
OOS_START, OOS_END = "2025-01-01", "2026-08-19"

R_TARGETS = [1.0, 1.5, 2.0, 2.5, 3.0]
RISKS = [0.25, 0.5]
STRATEGIES = {"001A": "A", "001B": "B"}


def grid(signals, m15, m1, label):
    rows = []
    trades_by_key = {}
    for r in R_TARGETS:
        for risk in RISKS:
            cfg = engine.Config(r_multiple=r, risk_pct=risk)
            tr = engine.simulate(signals, m15, m1, cfg)
            m = engine.metrics(tr, cfg)
            m.update({"period": label, "r_target": r, "risk_pct": risk})
            rows.append(m)
            trades_by_key[(r, risk)] = tr
    return pd.DataFrame(rows), trades_by_key


def main() -> None:
    OUT.mkdir(exist_ok=True)
    print("loading data ...", flush=True)
    d = dataset.build(IS_START, OOS_END)
    m1, m15, h4, d1 = d["m1"], d["m15"], d["h4"], d["d1"]
    print(f"  M1 {len(m1):,}  M15 {len(m15):,}  H4 {len(h4):,}  D1 {len(d1):,}")
    print(f"  {m1.index.min()}  ->  {m1.index.max()}")

    summary_all = []
    registry = {}

    for name, mode in STRATEGIES.items():
        print(f"\n=== strategy {name} ===", flush=True)
        sig = generate_signals(m15, h4, d1, mode)
        if sig.empty:
            print("  no signals at all")
            continue
        sig.to_csv(OUT / f"signals_{name}.csv", index=False)
        print(f"  {len(sig)} raw setups  "
              f"({(sig.direction > 0).sum()} long / {(sig.direction < 0).sum()} short)")

        is_mask = (sig.signal_ts >= pd.Timestamp(IS_START, tz="UTC")) & \
                  (sig.signal_ts <= pd.Timestamp(IS_END, tz="UTC"))
        sig_is, sig_oos = sig[is_mask], sig[~is_mask]

        m15_is = m15.loc[:IS_END]
        m15_oos = m15.loc[OOS_START:]
        m1_is = m1.loc[:IS_END]
        m1_oos = m1.loc[OOS_START:]

        gis, tis = grid(sig_is, m15_is, m1_is, "in-sample")
        goos, toos = grid(sig_oos, m15_oos, m1_oos, "out-of-sample")
        gis["strategy"] = goos["strategy"] = name

        # pre-register the in-sample pick, by expectancy, before looking at OOS
        valid = gis[gis.trades >= 20]
        pool = valid if not valid.empty else gis
        best = pool.sort_values("expectancy_R", ascending=False).iloc[0]
        pick = (float(best.r_target), float(best.risk_pct))
        registry[name] = {
            "picked_on": "in-sample expectancy (R)",
            "r_target": pick[0], "risk_pct": pick[1],
            "in_sample": {k: best[k] for k in
                          ("trades", "win_rate_pct", "profit_factor", "expectancy_R",
                           "exp_R_ci_low", "exp_R_ci_high", "prob_no_edge",
                           "net_return_pct", "max_drawdown_pct", "worst_losing_streak")},
        }
        oos_row = goos[(goos.r_target == pick[0]) & (goos.risk_pct == pick[1])]
        if not oos_row.empty:
            registry[name]["out_of_sample"] = {
                k: oos_row.iloc[0][k] for k in
                ("trades", "win_rate_pct", "profit_factor", "expectancy_R",
                 "exp_R_ci_low", "exp_R_ci_high", "prob_no_edge",
                 "net_return_pct", "max_drawdown_pct", "worst_losing_streak")}

        summary_all += [gis, goos]

        # full trade logs for the pre-registered configuration
        for label, book in (("in_sample", tis), ("out_of_sample", toos)):
            tr = book.get(pick)
            if tr is not None and not tr.empty:
                tr.to_csv(OUT / f"trades_{name}_{label}.csv", index=False)
                for by, fname in (("session", "by_session"), ("direction", "by_direction")):
                    engine.split_table(tr, by).to_csv(
                        OUT / f"{fname}_{name}_{label}.csv", index=False)
                yr = tr.copy()
                yr["year"] = yr.entry_ts.dt.year
                engine.split_table(yr, "year").to_csv(
                    OUT / f"by_year_{name}_{label}.csv", index=False)

    if summary_all:
        allg = pd.concat(summary_all, ignore_index=True)
        cols = ["strategy", "period", "r_target", "risk_pct", "trades", "win_rate_pct",
                "profit_factor", "expectancy_R", "exp_R_ci_low", "exp_R_ci_high",
                "prob_no_edge", "net_return_pct", "max_drawdown_pct",
                "worst_losing_streak", "avg_win", "avg_loss", "timeouts",
                "skipped_overlapping", "final_equity"]
        allg = allg[[c for c in cols if c in allg.columns]]
        allg.to_csv(OUT / "summary_grid.csv", index=False)
        print("\n" + allg.to_string(index=False))

    (OUT / "preregistered.json").write_text(json.dumps(registry, indent=2, default=str))
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
