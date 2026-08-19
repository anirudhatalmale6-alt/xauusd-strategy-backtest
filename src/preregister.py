#!/usr/bin/env python3
"""
Write the frozen rules to disk BEFORE the sealed period is touched.

This file exists so the out-of-sample result cannot be shopped for. The config
below is the one confirmed in writing on 2026-08-19, and the runner refuses to
start unless this file already exists and matches what it is about to run.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out_oos"

FROZEN = {
    "frozen_on": "2026-08-19",
    "frozen_by": "client instruction, thread 431473639",
    "strategy": "001A",
    "trend_mode": "ema50_200",
    "session_filter": True,
    "directions": "BUY and SELL - explicitly NOT sells-only",
    "retest_window_bars": 12,
    "r_target": 2.0,
    "risk_pct": 0.25,
    "secondary_r_targets": [1.0, 1.5, 2.5, 3.0],
    "secondary_risk_pct": [0.5],
    "headline": "001A / ema50_200 / session filter ON / 2R / 0.25% risk",
    "in_sample": "2015-01-01 .. 2024-12-31",
    "sealed_period": "2025-01-01 .. 2026-08-19",
    "runs_allowed": 1,
    "notes": [
        "No parameter was changed after the in-sample report was sent.",
        "3R scored better than 2R in-sample (+0.186R vs +0.112R). 2R is kept "
        "anyway because 2R is what the brief specified and switching to the "
        "best in-sample number is precisely the over-fitting that was ruled out.",
        "The other R targets are reported for completeness only. The headline "
        "number is the 2R / 0.25% row and nothing else.",
        "Whatever this run says, the rules do not change.",
    ],
}


def write() -> Path:
    OUT.mkdir(exist_ok=True)
    p = OUT / "preregistered.json"
    if p.exists():
        return p
    p.write_text(json.dumps(FROZEN, indent=2))
    return p


if __name__ == "__main__":
    print(write())
