#!/usr/bin/env python3
"""
Build the M1 XAU/USD dataset from Dukascopy's aggregated chart feed.

This is the same data as the tick archive, already aggregated to one-minute
bars, and it comes 30,000 bars at a time instead of one hour per request. The
tick route needed roughly 35,000 requests for five years; this needs about 120.

Both sides are pulled separately - `offer_side=B` and `offer_side=A` - so the
spread is a measured bid/ask difference rather than an assumption, which is what
makes the fills in engine.py honest.

The tick-built dataset in data/m1 is kept as an independent control: two
different endpoints, same underlying market. validate_sources.py compares them
bar for bar, and any disagreement is a reason to stop and look rather than to
carry on.

Response row: [epoch_ms, open, high, low, close, volume].
"""
from __future__ import annotations

import datetime as dt
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "m1_freeserv"
URL = "https://freeserv.dukascopy.com/2.0/index.php"
REFERER = "https://freeserv.dukascopy.com/2.0/?path=chart/index&instrument=XAU/USD"

START = dt.datetime(2015, 1, 1, tzinfo=dt.UTC)
END = dt.datetime(2026, 8, 19, 23, 59, tzinfo=dt.UTC)

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126 Safari/537.36",
    "Referer": REFERER,
})


def chunk(ts_ms: int, side: str, interval: str = "1MIN", retries: int = 5) -> list[list]:
    params = {
        "path": "chart/json3", "instrument": "XAU/USD", "offer_side": side,
        "interval": interval, "splits": "true", "stocks": "true",
        "time_direction": "P", "timestamp": str(ts_ms), "jsonp": "_callbacks____0",
    }
    for attempt in range(retries):
        try:
            r = session.get(URL, params=params, timeout=90)
            if r.status_code == 200:
                t = r.text
                body = t[t.find("(") + 1: t.rfind(")")]
                data = json.loads(body)
                if data and data != [None]:
                    return data
                if data == [] or data == [None]:
                    return []
        except (requests.RequestException, json.JSONDecodeError, ValueError):
            pass
        time.sleep(min(15.0, 2 ** attempt) * (0.7 + random.random() * 0.6))
    raise RuntimeError(f"freeserv failed for {side} @ {ts_ms}")


def pull_side(side: str) -> pd.DataFrame:
    """Page backwards from END to START."""
    cursor = int(END.timestamp() * 1000)
    floor_ms = int(START.timestamp() * 1000)
    frames: list[pd.DataFrame] = []
    seen_oldest = None

    while cursor > floor_ms:
        rows = chunk(cursor, side)
        if not rows:
            print(f"  {side}: empty chunk at {dt.datetime.fromtimestamp(cursor / 1000, dt.UTC)}, stopping")
            break
        df = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c", "v"])
        df["ts"] = pd.to_datetime(df.ts, unit="ms", utc=True)
        df = df.sort_values("ts")
        oldest = int(df.ts.iloc[0].timestamp() * 1000)
        newest = df.ts.iloc[-1]
        frames.append(df)
        print(f"  {side}: {len(df):6d} bars  {df.ts.iloc[0]}  ->  {newest}", flush=True)

        if seen_oldest is not None and oldest >= seen_oldest:
            print(f"  {side}: feed stopped going back at {df.ts.iloc[0]}")
            break
        seen_oldest = oldest
        cursor = oldest - 60_000          # step one minute past the oldest bar
        time.sleep(0.3)

    out = pd.concat(frames).drop_duplicates(subset=["ts"]).set_index("ts").sort_index()
    return out[(out.index >= START) & (out.index <= END)]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("bid side")
    bid = pull_side("B")
    print("ask side")
    ask = pull_side("A")

    df = bid.join(ask, how="inner", lsuffix="_bidside", rsuffix="_askside")
    out = pd.DataFrame({
        "bid_o": df.o_bidside, "bid_h": df.h_bidside, "bid_l": df.l_bidside, "bid_c": df.c_bidside,
        "ask_o": df.o_askside, "ask_h": df.h_askside, "ask_l": df.l_askside, "ask_c": df.c_askside,
        "ticks": (df.v_bidside + df.v_askside),
    })
    out["spread_mean"] = out.ask_c - out.bid_c
    out["spread_med"] = out.spread_mean

    # A crossed quote means the two sides came from different moments - drop it
    # rather than let the engine trade a negative spread.
    bad = (out.spread_mean < 0).sum()
    if bad:
        print(f"dropping {bad} bars where ask < bid")
        out = out[out.spread_mean >= 0]

    for year, part in out.groupby(out.index.year):
        p = OUT / f"XAUUSD_{year}.parquet"
        part.to_parquet(p, compression="zstd")
        print(f"{year}: {len(part):8,d} bars -> {p.name}")

    print(f"\ntotal {len(out):,} M1 bars  {out.index.min()} -> {out.index.max()}"
          f"   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    sys.exit(main())
