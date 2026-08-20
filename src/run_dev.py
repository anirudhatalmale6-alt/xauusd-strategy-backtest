#!/usr/bin/env python3
"""
Development slice only: 2015-10-01 to 2019-12-31.

This is a SMOKE TEST, not a result. Its job is to prove that each family finds
setups at a sane rate, that its stops are the size they were meant to be, that
trades open and close for the reasons they should, and that the regime layer is
attached correctly. Nothing here is allowed to select a family.

To keep that honest rather than merely stated, this script deliberately does not
print expectancy, profit factor or net return. A broken family gives itself away
in the trade count, the stop distribution, the hold time and the exit mix - none
of which require knowing whether it made money. If I can see the P&L on the
development slice, I will start choosing with it, and then the validation slice
is no longer independent of me.

Warm-up: price is loaded from 2014-06-01 so the 200-day EMA and the ADX exist on
the first day of the slice. Every setup dated before 2015-10-01 is discarded.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset
import engine
import families

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out_dev"

WARMUP_START = "2014-06-01"
DEV_START, DEV_END = "2015-10-01", "2019-12-31"

RISK_PCT = 0.25
R_TARGET = 2.0


def main() -> None:
    OUT.mkdir(exist_ok=True)
    print(f"loading {WARMUP_START} .. {DEV_END} (warm-up + development slice) ...", flush=True)
    d = dataset.build(WARMUP_START, DEV_END)
    m1, m15, d1 = d["m1"], d["m15"], d["d1"]
    print(f"  M1 {len(m1):,}  M15 {len(m15):,}  D1 {len(d1):,}")

    h1, m15r = families.prepare(m1, m15, d1)
    print(f"  H1 {len(h1):,}  regime attached\n")

    cut = pd.Timestamp(DEV_START, tz="UTC")
    years = (pd.Timestamp(DEV_END, tz="UTC") - cut).days / 365.25

    summary = []
    for name, (frame_kind, fn) in families.FAMILIES.items():
        if name == "f3_level_rejection":
            sig, rej = fn(m15r, d1)
        else:
            sig, rej = fn(h1, m15.index)

        n_warm = int((sig.signal_ts < cut).sum()) if not sig.empty else 0
        if not sig.empty:
            sig = sig[sig.signal_ts >= cut].reset_index(drop=True)
        if not rej.empty:
            rej = rej[rej.ts >= cut].reset_index(drop=True)

        cfg = engine.Config(
            r_multiple=R_TARGET, risk_pct=RISK_PCT,
            exit_mode="level" if name == "f4_mean_reversion" else "r_target",
            max_hold_bars=(families.PARAMS["f4"]["hold_h1"] * 4
                           if name == "f4_mean_reversion" else engine.MAX_HOLD_BARS),
        )
        tr = engine.simulate(sig, m15, m1, cfg) if not sig.empty else pd.DataFrame()

        sig.to_csv(OUT / f"signals_{name}.csv", index=False)
        rej.to_csv(OUT / f"rejected_{name}.csv", index=False)
        if not tr.empty:
            tr.to_csv(OUT / f"trades_{name}.csv", index=False)

        print(f"=== {name} ===")
        print(f"  setups {len(sig):,} ({n_warm} warm-up discarded)"
              f"   trades {len(tr):,}   {len(tr) / years:,.0f} per year")
        if not rej.empty:
            print("  refused: " + ", ".join(f"{k} {v}" for k, v in
                                            rej.reason.value_counts().items()))
        if not tr.empty:
            print("  exits:   " + ", ".join(f"{k} {v}" for k, v in
                                            tr.reason.value_counts().items()))
            print(f"  stop distance in ATRs: median "
                  f"{(tr.stop_distance / tr.atr).median():.2f}, "
                  f"p10 {(tr.stop_distance / tr.atr).quantile(.10):.2f}, "
                  f"p90 {(tr.stop_distance / tr.atr).quantile(.90):.2f}")
            print(f"  bars held: median {tr.bars_held.median():.0f}, "
                  f"p90 {tr.bars_held.quantile(.90):.0f}   "
                  f"skipped overlapping {tr.attrs.get('skipped_overlapping', 0)}")
            print("  direction: " + ", ".join(f"{k} {v}" for k, v in
                                              tr.direction.value_counts().items()))
            print("  regime:    " + ", ".join(f"{k} {v}" for k, v in
                                              tr.regime.value_counts().items()))
        print()

        summary.append({
            "family": name, "setups": len(sig), "trades": len(tr),
            "trades_per_year": round(len(tr) / years, 1),
            "meets_rate_gate": len(tr) / years >= 100,
            "skipped_overlapping": int(tr.attrs.get("skipped_overlapping", 0)) if not tr.empty else 0,
            "regimes_seen": tr.regime.nunique() if not tr.empty else 0,
        })

    s = pd.DataFrame(summary)
    s.to_csv(OUT / "dev_summary.csv", index=False)
    print("=== development slice summary (diagnostics only, nothing selected here) ===")
    print(s.to_string(index=False))
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
