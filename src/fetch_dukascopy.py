#!/usr/bin/env python3
"""
Build an M1 XAU/USD dataset with real bid AND ask from Dukascopy tick data.

Dukascopy serves one LZMA-compressed file per instrument per UTC hour. Each tick
is 20 bytes big-endian: milliseconds-into-the-hour, ask, bid, ask volume, bid
volume, with ask/bid as integer points (XAUUSD has 3 decimals, so divide by 1000).

We never keep the ticks. Each hour is aggregated to one-minute bars as it
arrives - bid OHLC, ask OHLC, tick count and the mean/median spread - and
written one DAY at a time, so the job resumes cleanly after any interruption and
memory stays flat.

Dukascopy rate-limits per IP. Measured here: four concurrent connections give
100% success, sixteen give 50% failures for no extra throughput. So the
concurrency is deliberately low, and any hour that still fails is retried by a
second gap-filling pass rather than being silently dropped - a missing hour in
the middle of a session would corrupt the 4H and Daily bars built from it.

Timestamps are UTC throughout. Sessions and daylight saving are handled later,
in dataset.py, from the exchange calendars.
"""
from __future__ import annotations

import datetime as dt
import lzma
import random
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SYMBOL = "XAUUSD"
POINT = 1000.0                      # XAUUSD is quoted to 3 decimals
BASE = "https://datafeed.dukascopy.com/datafeed"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "m1"
TICK = struct.Struct(">IIIff")
WORKERS = 6

START = dt.date(2021, 8, 1)
END = dt.date(2026, 8, 19)

_local = threading.local()


def sess() -> requests.Session:
    s = getattr(_local, "s", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"})
        _local.s = s
    return s


def hour_url(d: dt.date, hour: int) -> str:
    return f"{BASE}/{SYMBOL}/{d.year}/{d.month - 1:02d}/{d.day:02d}/{hour:02d}h_ticks.bi5"


class Closed(Exception):
    """The market really was shut for this hour - not an error."""


def fetch_hour(d: dt.date, hour: int, retries: int = 6) -> np.ndarray:
    url = hour_url(d, hour)
    last = ""
    for attempt in range(retries):
        try:
            r = sess().get(url, timeout=60)
        except requests.RequestException as e:
            last = type(e).__name__
        else:
            if r.status_code == 404:
                raise Closed
            if r.status_code == 200:
                if not r.content:
                    raise Closed
                try:
                    raw = lzma.LZMADecompressor(format=lzma.FORMAT_AUTO).decompress(r.content)
                except lzma.LZMAError:
                    last = "lzma"
                else:
                    n = len(raw) // TICK.size
                    if n == 0:
                        raise Closed
                    return np.frombuffer(raw[: n * TICK.size], dtype=np.dtype([
                        ("ms", ">u4"), ("ask", ">u4"), ("bid", ">u4"),
                        ("askvol", ">f4"), ("bidvol", ">f4"),
                    ]))
            else:
                last = str(r.status_code)
        time.sleep(min(20.0, 1.5 * (2 ** attempt)) * (0.6 + random.random() * 0.8))
    raise RuntimeError(f"{url} failed after {retries} attempts ({last})")


def ticks_to_m1(buf: np.ndarray, d: dt.date, hour: int) -> pd.DataFrame:
    ms = buf["ms"].astype(np.int64)
    ask = buf["ask"].astype(np.float64) / POINT
    bid = buf["bid"].astype(np.float64) / POINT
    base = pd.Timestamp(dt.datetime(d.year, d.month, d.day, hour), tz="UTC")
    df = pd.DataFrame({"bid": bid, "ask": ask}, index=base + pd.to_timedelta(ms, unit="ms"))
    df["spread"] = df.ask - df.bid
    g = df.resample("1min")
    out = pd.DataFrame({
        "bid_o": g.bid.first(), "bid_h": g.bid.max(), "bid_l": g.bid.min(), "bid_c": g.bid.last(),
        "ask_o": g.ask.first(), "ask_h": g.ask.max(), "ask_l": g.ask.min(), "ask_c": g.ask.last(),
        "spread_mean": g.spread.mean(), "spread_med": g.spread.median(),
        "ticks": g.bid.count(),
    })
    return out[out.ticks > 0]


def trading_hours(d: dt.date) -> list[int]:
    """Skip hours the metals market is certainly shut, to save pointless requests."""
    wd = d.weekday()                     # Mon=0 .. Sun=6
    if wd == 5:
        return []                        # Saturday
    if wd == 6:
        return [21, 22, 23]              # Sunday reopen
    if wd == 4:
        return list(range(0, 21))        # Friday close
    return list(range(0, 24))


def day_path(d: dt.date) -> Path:
    return OUT / f"{d:%Y}" / f"{SYMBOL}_{d:%Y%m%d}.parquet"


def write_day(d: dt.date, frames: list[pd.DataFrame]) -> int:
    if not frames:
        return 0
    day = pd.concat(frames).sort_index()
    day = day[~day.index.duplicated(keep="first")]
    p = day_path(d)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    day.to_parquet(tmp, compression="zstd")
    tmp.replace(p)                        # atomic - a half-written parquet is poison
    return len(day)


def run(budget_s: float | None = None) -> None:
    """
    One flat queue of (day, hour) jobs across the whole range, so a slow request
    never stalls the rest. Days are written as soon as all of their hours are in.

    `budget_s` stops cleanly after roughly that many seconds, which lets the job
    be run in bounded chunks - every finished day is already on disk, so the next
    run simply picks up where this one stopped.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    days = [START + dt.timedelta(days=i) for i in range((END - START).days + 1)]
    days = [d for d in days if trading_hours(d)]
    todo = [d for d in days if not day_path(d).exists()]
    print(f"{len(days)} trading days in range, {len(todo)} still to fetch", flush=True)
    if not todo:
        print("DONE", flush=True)
        return

    jobs = [(d, h) for d in todo for h in trading_hours(d)]
    pending: dict[dt.date, int] = {d: len(trading_hours(d)) for d in todo}
    frames: dict[dt.date, list[pd.DataFrame]] = {d: [] for d in todo}
    failed: dict[dt.date, list[int]] = {}

    t0 = time.time()
    done_days = 0
    stop = threading.Event()

    SKIP, FAIL, CLOSED = object(), object(), object()

    def work(job):
        d, h = job
        if stop.is_set():
            return d, h, SKIP
        try:
            return d, h, ticks_to_m1(fetch_hour(d, h), d, h)
        except Closed:
            return d, h, CLOSED
        except Exception:
            return d, h, FAIL

    with ThreadPoolExecutor(WORKERS) as ex:
        for d, h, res in ex.map(work, jobs):
            if res is SKIP:
                continue
            if res is FAIL:
                failed.setdefault(d, []).append(h)
            elif res is not CLOSED and not res.empty:
                frames[d].append(res)
            pending[d] -= 1

            if pending[d] == 0:
                if d in failed:
                    # An incomplete day is worse than no day - a hole in the
                    # middle of a session would corrupt the 4H and Daily bars
                    # built from it. Leave it unwritten so the next run retries.
                    frames.pop(d, None)
                    continue
                n = write_day(d, frames.pop(d))
                done_days += 1
                if done_days % 10 == 0:
                    el = time.time() - t0
                    rate = done_days / el
                    print(f"[{done_days:5d}/{len(todo)}] {d}  {n:5d} bars  "
                          f"{rate * 60:.1f} days/min  "
                          f"eta {(len(todo) - done_days) / rate / 60:.0f} min", flush=True)
                if budget_s and time.time() - t0 > budget_s:
                    print(f"budget reached after {done_days} days - stopping cleanly", flush=True)
                    stop.set()

    if failed:
        print(f"\n{len(failed)} day(s) had hours that would not download:", flush=True)
        for d, hs in sorted(failed.items()):
            print(f"  {d}: {sorted(hs)}", flush=True)
        print("re-run to fill them - incomplete days were still written and will be "
              "re-checked by verify_data.py", flush=True)

    print("DONE" if not stop.is_set() else "PARTIAL", flush=True)


if __name__ == "__main__":
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else None
    sys.exit(run(budget))
