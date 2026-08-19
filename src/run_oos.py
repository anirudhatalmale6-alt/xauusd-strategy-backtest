#!/usr/bin/env python3
"""
THE SEALED PERIOD. 2025-01-01 to 2026-08-19. Run once, on frozen rules.

Two things about how this is done that matter:

  1. Indicators need history. A 200-period daily EMA cannot exist on the first
     day of the test, and 4H zones need prior swings to have formed. So price
     data is loaded from 2023-06-01 to warm the indicators up, and then every
     signal dated before 2025-01-01 is thrown away. That is look-BACK, which is
     legitimate; nothing in here looks forward.

  2. Equity starts at 10,000 on the first sealed trade. The in-sample equity
     curve is not carried over - the two periods are reported separately so the
     out-of-sample drawdown is measured against its own peak, not flattered by
     ten years of prior gains.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset
import engine
from preregister import FROZEN, write as write_prereg
from strategy import generate_signals

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out_oos"

WARMUP_START = "2023-06-01"          # indicator warm-up only, discarded afterwards
OOS_START, OOS_END = "2025-01-01", "2026-08-19"


def main() -> None:
    prereg = write_prereg()
    cfg_f = json.loads(prereg.read_text())
    print(f"frozen config read from {prereg}")
    print(f"  {cfg_f['headline']}\n")

    print(f"loading {WARMUP_START} .. {OOS_END} (warm-up + sealed) ...", flush=True)
    d = dataset.build(WARMUP_START, OOS_END)
    m1, m15, h4, d1 = d["m1"], d["m15"], d["h4"], d["d1"]
    print(f"  M1 {len(m1):,}  M15 {len(m15):,}  H4 {len(h4):,}  D1 {len(d1):,}")
    print(f"  {m1.index.min()}  ->  {m1.index.max()}\n")

    sig, rej = generate_signals(
        m15, h4, d1,
        "A" if cfg_f["strategy"] == "001A" else "B",
        cfg_f["trend_mode"],
        cfg_f["session_filter"],
    )

    cut = pd.Timestamp(OOS_START, tz="UTC")
    n_warm = int((sig.signal_ts < cut).sum()) if not sig.empty else 0
    sig = sig[sig.signal_ts >= cut].reset_index(drop=True)
    if not rej.empty and "signal_ts" in rej:
        rej = rej[rej.signal_ts >= cut].reset_index(drop=True)
    print(f"{len(sig)} qualified setups in the sealed period "
          f"({n_warm} warm-up setups discarded), {len(rej)} refused\n")

    sig.to_csv(OUT / "signals_oos.csv", index=False)
    rej.to_csv(OUT / "rejected_oos.csv", index=False)
    if not rej.empty and "reason" in rej:
        rej.reason.value_counts().rename("count").to_frame().to_csv(OUT / "rejection_reasons.csv")
        print(rej.reason.value_counts().to_string(), "\n")

    # ---- the headline run, and only then the rest of the grid for context ----
    rows = []
    combos = ([(cfg_f["r_target"], cfg_f["risk_pct"])]
              + [(r, cfg_f["risk_pct"]) for r in cfg_f["secondary_r_targets"]]
              + [(r, s) for s in cfg_f["secondary_risk_pct"]
                 for r in [cfg_f["r_target"]] + cfg_f["secondary_r_targets"]])

    headline_trades = None
    for r, risk in combos:
        c = engine.Config(r_multiple=r, risk_pct=risk)
        tr = engine.simulate(sig, m15, m1, c)
        m = engine.metrics(tr, c)
        m.update({"r_target": r, "risk_pct": risk,
                  "headline": (r, risk) == (cfg_f["r_target"], cfg_f["risk_pct"])})
        rows.append(m)
        if m["headline"]:
            headline_trades = tr

    grid = pd.DataFrame(rows)
    cols = ["headline", "r_target", "risk_pct", "trades", "wins", "losses",
            "win_rate_pct", "profit_factor", "expectancy_R", "exp_R_ci_low",
            "exp_R_ci_high", "prob_no_edge", "net_return_pct", "max_drawdown_pct",
            "worst_losing_streak", "best_winning_streak", "avg_win", "avg_loss",
            "avg_win_R", "avg_loss_R", "avg_bars_held", "timeouts",
            "skipped_overlapping", "final_equity"]
    grid = grid[[c for c in cols if c in grid.columns]]
    grid.to_csv(OUT / "oos_grid.csv", index=False)

    tr = headline_trades
    if tr is None or tr.empty:
        print("no trades in the sealed period")
        return

    tr.to_csv(OUT / "trades_oos.csv", index=False)
    x = tr.copy()
    x["year"] = x.entry_ts.dt.year
    for by, fname in (("year", "by_year"), ("session", "by_session"),
                      ("direction", "by_direction"), ("regime", "by_regime"),
                      ("reason", "by_exit_reason")):
        if by in x.columns:
            engine.split_table(x, by).to_csv(OUT / f"{fname}_oos.csv", index=False)

    # equity curve for the report
    eq = pd.concat([pd.DataFrame({"exit_ts": [tr.entry_ts.iloc[0]], "equity": [10_000.0]}),
                    tr[["exit_ts", "equity"]]], ignore_index=True)
    eq.to_csv(OUT / "equity_oos.csv", index=False)

    head = grid[grid.headline].iloc[0]
    print("=== SEALED PERIOD RESULT - 001A, regime filter, session filter, 2R, 0.25% ===")
    print(head.to_string())
    print("\n=== full grid (context only, the headline row is the result) ===")
    print(grid.to_string(index=False))
    for f in ("by_year", "by_direction", "by_session", "by_regime", "by_exit_reason"):
        p = OUT / f"{f}_oos.csv"
        if p.exists():
            print(f"\n--- {f} ---")
            print(pd.read_csv(p).to_string(index=False))

    (OUT / "oos_meta.json").write_text(json.dumps({
        "warmup_start": WARMUP_START,
        "oos_start": OOS_START, "oos_end": OOS_END,
        "warmup_setups_discarded": n_warm,
        "qualified_setups": int(len(sig)),
        "frozen_config": cfg_f,
        "run_count": 1,
    }, indent=2, default=str))
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
