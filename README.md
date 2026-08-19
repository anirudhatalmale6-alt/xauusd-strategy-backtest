# XAU/USD backtest - strategy 001A vs 001B

A historical test of two gold strategies. It answers one question: do these
rules make money on real prices, or not?

**It is not a trading bot, it does not connect to a broker, and it cannot place
an order.** Nothing here predicts the future.

## The result

| | 001A - breakout + retest | 001B - zone rejection |
|---|---|---|
| in-sample expectancy | **-0.022R** | **+0.095R** |
| out-of-sample expectancy | **-0.127R** | **+0.447R** |
| 95% interval (in-sample) | -0.240 to +0.207 | -0.199 to +0.389 |
| chance the edge is noise | 59% | 25% |

**001A has no edge.** It lost money in-sample and out-of-sample, and all ten
combinations of profit target and risk were negative in-sample.

**001B is not proven.** It made money in both periods, but the confidence
interval includes zero in both. On 97 in-sample trades it cannot be told apart
from luck. The out-of-sample stretch looks strong, but it is 29 trades.

Neither justifies risking money. The next step is paper trading to gather more
trades, not a live account.

## What the numbers rest on

- **Data**: Dukascopy Bank, five years of XAU/USD one-minute bars with real bid
  and ask, 1.79 million bars, Aug 2021 to Aug 2026.
- **Independently checked**: 242,288 of those minutes were rebuilt from the raw
  tick archive - a different endpoint, a different format - and compared bar for
  bar. Median difference **0.0000** on all eight OHLC columns, identical spreads.
- **No look-ahead**: 4H and Daily values only become visible after the bar
  closes; 4H zones are built from swing pivots only once confirmed; a signal on
  a 15m close is filled at the next bar's open.
- **Costs that err against the strategy**: buys pay the ask, sells receive the
  bid, entries and stops carry 5c slippage, and if a stop and a target fall in
  the same minute it is booked as a **loss**.
- **Out-of-sample held back**: one configuration per strategy was chosen on
  in-sample data and written to `out/preregistered.json` before the
  out-of-sample period was run.

## Layout

```
src/fetch_freeserv.py     build the 5-year M1 bid/ask dataset
src/fetch_dukascopy.py    the same thing the slow way, from raw ticks (the control)
src/validate_sources.py   compare the two, bar for bar
src/verify_data.py        gaps, crossed quotes, spread sanity, session split
src/dataset.py            timeframes + session tagging (DST from the tz database)
src/strategy.py           trend, zones, and the 001A / 001B triggers
src/engine.py             fills, position sizing, metrics, bootstrap significance
src/run_backtest.py       the full grid, in-sample then out-of-sample
src/report.py             the HTML report
out/report.html           <- open this
```

## Running it

```bash
pip install pandas numpy pyarrow requests matplotlib
python3 src/fetch_freeserv.py     # ~75 seconds for five years
python3 src/verify_data.py
python3 src/run_backtest.py
python3 src/report.py
```

## Parameters

Every threshold was fixed in advance from the written brief, not tuned to make
the results look better. They are constants at the top of `strategy.py`:
pivot size, zone tolerance and padding, retest window, confirmation fraction,
ATR stop buffer, and the min/max stop distance.
