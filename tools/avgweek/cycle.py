#!/usr/bin/env python3
"""
cycle.py  --  where is this name in its own history of run-ups and grinds, and what did past waits look like?

  python3 cycle.py ASTS RKLB HIMS
  python3 cycle.py ASTS RKLB HIMS --universe universe_highbeta.txt     # adds pooled base rates across the basket

Per ticker:
  1. State now: distance below the 52-week and all-time highs, weeks since each, the current leg (down since a peak,
     or up from a trough) with its depth and age, and where today's 26-week trailing return sits in the name's own history.
  2. The catalogue: every trough-to-peak run-up of at least --run (default 2x) on weekly closes, found with a zigzag of
     --threshold (default 30%) reversals; for each, the grind that preceded it (depth and weeks) and the run-up's size and length.
  3. Waiting times: weeks from a peak to the trough that started the next run-up, in this name and (with --universe) across
     the basket; today's elapsed wait against that distribution, and what the waits that outlasted today's still had to run.
  4. Base rates: from weeks at least as far below the 52-week high as today, how often the next 13 / 26 / 52 weeks held a
     +50% (and a 2x) move from that week's close, against the same rate from any week. Own history and pooled.

Nothing here is a forecast. A name has three to six episodes; the pooled numbers have power, the single-name ones do not.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from avgweek import BENCHMARKS, _gbm_path, drop_shell_comment, load_prices, weekly_series  # noqa: E402

__version__ = "0.1.0"
HORIZONS = (13, 26, 52)


# ----------------------------------------------------------------------------------------------------- swings
def zigzag(close: np.ndarray, thr: float) -> list[tuple[int, float, str]]:
    """Confirmed swing pivots on a close series: a peak once price has fallen thr below the running high,
    a trough once it has risen thr above the running low. Returns [(index, price, 'H'|'L'), ...] in order."""
    n = len(close)
    if n < 2:
        return []
    piv: list[tuple[int, float, str]] = []
    direction = 0
    hi_i = lo_i = 0
    hi_p = lo_p = float(close[0])
    for i in range(1, n):
        p = float(close[i])
        if direction >= 0 and p > hi_p:
            hi_i, hi_p = i, p
        if direction <= 0 and p < lo_p:
            lo_i, lo_p = i, p
        if direction >= 0 and p <= hi_p * (1.0 - thr):
            piv.append((hi_i, hi_p, "H"))
            direction, lo_i, lo_p = -1, i, p
        elif direction <= 0 and p >= lo_p * (1.0 + thr):
            piv.append((lo_i, lo_p, "L"))
            direction, hi_i, hi_p = 1, i, p
    return piv


def episodes(wk: pd.DataFrame, thr: float, min_run: float) -> tuple[list[dict], dict]:
    """Run-ups of at least `min_run` (1.0 = a double) with the grind before each, plus the state of the current leg."""
    close = wk["close"].values.astype(float)
    dates = wk["date"].values
    piv = zigzag(close, thr)
    runs = []
    for k in range(1, len(piv)):
        (i0, p0, k0), (i1, p1, k1) = piv[k - 1], piv[k]
        if k0 == "L" and k1 == "H" and p1 / p0 - 1.0 >= min_run:
            prev_peak = piv[k - 2] if k >= 2 else None
            runs.append({"start": pd.Timestamp(dates[i0]), "end": pd.Timestamp(dates[i1]), "gain": p1 / p0 - 1.0, "weeks": i1 - i0,
                         "grind_weeks": (i0 - prev_peak[0]) if prev_peak else None,
                         "grind_depth": (1.0 - p0 / prev_peak[1]) if prev_peak else None,
                         "prev_peak_date": pd.Timestamp(dates[prev_peak[0]]) if prev_peak else None})
    # the current, unconfirmed leg
    last = len(close) - 1
    state = {"leg": "flat", "since_idx": 0}
    if piv:
        li, lp, lk = piv[-1]
        if lk == "H":
            trough_i = li + int(np.argmin(close[li:]))
            state = {"leg": "down", "peak_date": pd.Timestamp(dates[li]), "peak_price": lp, "weeks_since_peak": last - li,
                     "depth_so_far": 1.0 - float(close[trough_i]) / lp, "trough_date": pd.Timestamp(dates[trough_i]),
                     "off_trough": float(close[last]) / float(close[trough_i]) - 1.0}
        else:
            state = {"leg": "up", "trough_date": pd.Timestamp(dates[li]), "trough_price": lp, "weeks_since_trough": last - li,
                     "gain_so_far": float(close[last]) / lp - 1.0}
    else:
        hi_i = int(np.argmax(close))
        state = {"leg": "down" if hi_i < last else "up", "peak_date": pd.Timestamp(dates[hi_i]), "peak_price": float(close[hi_i]),
                 "weeks_since_peak": last - hi_i, "depth_so_far": 1.0 - float(close[last:].min()) / float(close[hi_i]),
                 "trough_date": pd.Timestamp(dates[last]), "off_trough": 0.0}
    return runs, state


# ----------------------------------------------------------------------------------------------------- state and base rates
def state_frame(wk: pd.DataFrame) -> pd.DataFrame:
    """Per week: drawdown from the trailing 52-week high, weeks since that high, 26-week trailing return, and the
    largest gain available over the next 13 / 26 / 52 weeks from this week's close."""
    c = wk["close"].astype(float)
    f = pd.DataFrame(index=wk.index)
    f["close"] = c
    hi52 = c.rolling(52, min_periods=26).max()
    f["dd52"] = 1.0 - c / hi52
    f["ret26"] = c / c.shift(26) - 1.0
    # weeks since the 52-week high
    pos = np.arange(len(c))
    argmax = c.rolling(52, min_periods=26).apply(lambda x: len(x) - 1 - int(np.argmax(x)), raw=True)
    f["wks_since_hi52"] = argmax
    vals = c.values
    for h in HORIZONS:
        fwd = np.full(len(vals), np.nan)
        for i in range(len(vals) - 1 - h):              # full windows only; the last h weeks are censored, not failures
            fwd[i] = vals[i + 1:i + 1 + h].max() / vals[i] - 1.0
        f[f"fwd{h}"] = fwd
    return f


def base_rates(f: pd.DataFrame, dd_now: float, slack: float = 0.10) -> dict:
    """P(+50%) and P(2x) within each horizon, from weeks at least (dd_now - slack) below the 52-week high, vs any week."""
    out = {}
    like = f[f["dd52"] >= max(dd_now - slack, 0.0)]
    for h in HORIZONS:
        a = f[f"fwd{h}"].dropna()
        l = like[f"fwd{h}"].dropna()
        out[h] = {"n_all": int(len(a)), "p50_all": float((a >= 0.5).mean()) if len(a) else float("nan"), "p100_all": float((a >= 1.0).mean()) if len(a) else float("nan"),
                  "n_like": int(len(l)), "p50_like": float((l >= 0.5).mean()) if len(l) else float("nan"), "p100_like": float((l >= 1.0).mean()) if len(l) else float("nan")}
    return out


def pct_rank(series: pd.Series, x: float) -> float:
    s = series.dropna()
    return float((s < x).mean()) if len(s) and x == x else float("nan")


# ----------------------------------------------------------------------------------------------------- presentation
def _p(x, d=0):
    return "n/a" if x is None or x != x else f"{x*100:+.{d}f}%"


def _u(x, d=0):
    return "n/a" if x is None or x != x else f"{x*100:.{d}f}%"


def residual_waits(gaps: list[int], elapsed: int) -> tuple[int, int, list[int]]:
    longer = sorted(g - elapsed for g in gaps if g > elapsed)
    return len([g for g in gaps if g <= elapsed]), len(gaps), longer


def describe(ticker: str, wk: pd.DataFrame, runs: list[dict], state: dict, f: pd.DataFrame, rates: dict, pooled: dict | None,
             thr: float, min_run: float) -> str:
    c = float(wk["close"].iloc[-1])
    last = f.iloc[-1]
    ath_i = int(np.argmax(wk["close"].values))
    ath, ath_date = float(wk["close"].iloc[ath_i]), wk["date"].iloc[ath_i]
    weeks_since_ath = len(wk) - 1 - ath_i
    L = [f"{ticker}  ${c:,.2f} on {wk['date'].iloc[-1].date()}   {_u(last['dd52'])} below the 52-week high ({int(last['wks_since_hi52']) if last['wks_since_hi52'] == last['wks_since_hi52'] else 'n/a'} weeks ago), "
         f"{_u(1 - c / ath)} below the all-time high ${ath:,.2f} ({ath_date.date()}, {weeks_since_ath} weeks ago)   {len(wk)} weeks of history"]
    if state["leg"] == "down":
        L.append(f"  Current leg: DOWN from ${state['peak_price']:,.2f} on {state['peak_date'].date()}, {state['weeks_since_peak']} weeks; "
                 f"low so far {_u(state['depth_so_far'])} below it on {state['trough_date'].date()}; now {_p(state['off_trough'])} off that low")
    elif state["leg"] == "up":
        past = (f"; finished run-ups here ran {_p(min(r['gain'] for r in runs))} to {_p(max(r['gain'] for r in runs))} over {min(r['weeks'] for r in runs)} to {max(r['weeks'] for r in runs)} weeks" if runs else "")
        L.append(f"  Current leg: UP from ${state['trough_price']:,.2f} on {state['trough_date'].date()}, {state['weeks_since_trough']} weeks, {_p(state['gain_so_far'])} so far{past}")
    r26 = float(last["ret26"]) if last["ret26"] == last["ret26"] else float("nan")
    L.append(f"  26-week return now {_p(r26)}: lower than {_u(pct_rank(f['ret26'], r26))} of this name's own 26-week windows")
    # catalogue
    L.append(f"  Run-ups of {min_run + 1:.0f}x or more, trough to peak ({thr:.0%} swing filter): {len(runs)}")
    for r in runs:
        g = (f"after a {r['grind_weeks']}-week grind, {_u(r['grind_depth'])} down from the {r['prev_peak_date'].date()} peak" if r["grind_weeks"] is not None else "from the start of the data")
        L.append(f"    {r['start'].date()} -> {r['end'].date()}   {_p(r['gain'])} in {r['weeks']} wk   {g}")
    gaps = [r["grind_weeks"] for r in runs if r["grind_weeks"] is not None]
    elapsed = state.get("weeks_since_peak") if state["leg"] == "down" else None
    if gaps:
        txt = ", ".join(str(g) for g in sorted(gaps))
        L.append(f"  Peak-to-trough waits before those run-ups: {txt} weeks (median {int(np.median(gaps))})")
        if elapsed is not None:
            done, tot, longer = residual_waits(gaps, elapsed)
            more = (f"the {len(longer)} that lasted longer still had {min(longer)} to {max(longer)} weeks to run (median {int(np.median(longer))})" if longer
                    else "every past wait was shorter than this one")
            L.append(f"  Today's wait: {elapsed} weeks since the peak; {done} of {tot} past waits had ended by now; {more}")
    elif elapsed is not None:
        L.append(f"  Today's wait: {elapsed} weeks since the peak; no completed {min_run + 1:.0f}x run-up in this name's history to compare with")
    if pooled and pooled.get("gaps") and elapsed is not None:
        pg = pooled["gaps"]
        done, tot, longer = residual_waits(pg, elapsed)
        more = (f"those that lasted longer still had a median {int(np.median(longer))} weeks to run (quartiles {int(np.percentile(longer, 25))} to {int(np.percentile(longer, 75))})" if longer
                else "none lasted longer")
        L.append(f"  Across the other {pooled['names']} names ({tot} waits, median {int(np.median(pg))} weeks): {done} of {tot} had ended by {elapsed} weeks; {more}")
    # base rates
    dd_now = float(last["dd52"]) if last["dd52"] == last["dd52"] else 0.0
    def line(tag, R):
        parts = []
        for h in HORIZONS:
            r = R[h]
            parts.append(f"{h}wk {_u(r['p50_like'])} (n={r['n_like']}) vs {_u(r['p50_all'])} any week")
        return f"  {tag}: " + "; ".join(parts)
    L.append(f"  Chance of a +50% move from here, from weeks at least {_u(max(dd_now - 0.10, 0))} below the 52-week high:")
    L.append(line("    own history ", rates))
    if pooled and pooled.get("rates"):
        L.append(line("    basket      ", pooled["rates"]))
    parts = [f"{h}wk {_u(rates[h]['p100_like'])} vs {_u(rates[h]['p100_all'])}" for h in HORIZONS]
    L.append("  Same for a 2x move, own history: " + "; ".join(parts))
    return "\n".join(L)


# ----------------------------------------------------------------------------------------------------- IO
def read_universe(path: str) -> list[str]:
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            out.extend(line.replace(",", " ").split())
    return out


def load_daily(ticker: str, args) -> pd.DataFrame:
    csv = args.csv
    if args.csv_dir:
        cand = Path(args.csv_dir) / f"{ticker}.csv"
        if cand.exists():
            csv = str(cand)
        elif not csv:
            raise SystemExit(f"{cand} not found")
    daily, _ = load_prices(ticker, Path(args.cache_dir).expanduser(), csv=csv, refresh=not args.no_refresh, source=args.source)
    if args.asof:
        daily = daily[daily.index <= pd.Timestamp(args.asof)]
    return daily


def analyse(ticker: str, daily: pd.DataFrame, thr: float, min_run: float) -> dict:
    wk = weekly_series(daily)
    runs, state = episodes(wk, thr, min_run)
    f = state_frame(wk)
    return {"ticker": ticker, "wk": wk, "runs": runs, "state": state, "f": f}


def pooled_from(results: list[dict], dd_now: float) -> dict:
    gaps = [r["grind_weeks"] for res in results for r in res["runs"] if r["grind_weeks"] is not None]
    frames = [res["f"] for res in results]
    f = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return {"names": len(results), "gaps": gaps, "rates": base_rates(f, dd_now) if len(f) else None}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tickers", nargs="*")
    p.add_argument("--universe", help="file with one ticker per line; pools the waits and base rates across all of them")
    p.add_argument("--threshold", type=float, default=0.30, help="zigzag reversal that confirms a swing, as a fraction (default 0.30)")
    p.add_argument("--run", type=float, default=1.0, help="minimum trough-to-peak gain that counts as a run-up (default 1.0 = a double)")
    p.add_argument("--asof", help="pretend today is this date (YYYY-MM-DD)")
    p.add_argument("--csv")
    p.add_argument("--csv-dir")
    p.add_argument("--cache-dir", default="~/.cache/avgweek")
    p.add_argument("--source", default="auto", choices=["auto", "yfinance", "stooq", "alphavantage"])
    p.add_argument("--no-refresh", action="store_true")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        return selftest()
    shown = [t.strip(",;").upper() for t in drop_shell_comment(args.tickers) if t.strip(",;")]
    pool_names = list(dict.fromkeys(shown + ([t.upper() for t in read_universe(args.universe)] if args.universe else [])))
    if not pool_names:
        p.print_help()
        return 2
    results: dict[str, dict] = {}
    for t in pool_names:
        if t in BENCHMARKS:
            continue
        try:
            daily = load_daily(t, args)
        except (SystemExit, Exception) as exc:  # noqa: BLE001
            print(f"  {t}: could not load prices: {exc}")
            continue
        if len(daily) < 300:
            print(f"  {t}: only {len(daily)} daily bars, skipped")
            continue
        results[t] = analyse(t, daily, args.threshold, args.run)
    if not results:
        return 1
    show = shown or list(results)
    for t in show:
        if t not in results:
            continue
        res = results[t]
        last = res["f"].iloc[-1]
        dd_now = float(last["dd52"]) if last["dd52"] == last["dd52"] else 0.0
        rates = base_rates(res["f"], dd_now)
        pooled = pooled_from([r for k, r in results.items() if k != t], dd_now) if (args.universe and len(results) > 1) else None
        print(describe(t, res["wk"], res["runs"], res["state"], res["f"], rates, pooled, args.threshold, args.run))
        print()
    print("Read: a name has a handful of episodes, so its own waits and rates are anecdotes; the basket rows carry the statistics. "
          "A base rate near the any-week rate means the depth of the drawdown says nothing about timing.")
    return 0


# ----------------------------------------------------------------------------------------------------- self-test
def selftest() -> int:
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
        if not ok:
            fails += 1

    print("Self-test 1: a sawtooth with a known cycle (40 weeks grinding -60%, then 12 weeks tripling)")
    weeks = [100.0]
    price = 100.0
    for cyc in range(5):
        for w in range(40):
            price *= (0.4) ** (1 / 40)
            weeks.append(price)
        for w in range(12):
            price *= 3.0 ** (1 / 12)
            weeks.append(price)
    # cut inside a grind: 25 weeks into the 6th grind
    for w in range(25):
        price *= (0.4) ** (1 / 40)
        weeks.append(price)
    idx = pd.bdate_range(end=pd.Timestamp("2026-10-09"), periods=len(weeks) * 5)
    daily = pd.DataFrame({"close": np.repeat(weeks, 5)}, index=idx)
    daily["open"] = daily["high"] = daily["low"] = daily["close"]
    wk = weekly_series(daily, drop_partial=False)
    runs, state = episodes(wk, 0.30, 1.0)
    check("finds the 5 tripling run-ups", len(runs) == 5, f"found {len(runs)}: {[round(r['gain'], 2) for r in runs]}")
    gaps = [r["grind_weeks"] for r in runs if r["grind_weeks"] is not None]
    check("the grind before each run-up measures 40 weeks", all(g == 40 for g in gaps), f"gaps {gaps}")
    check("current leg is DOWN, 25 weeks old", state["leg"] == "down" and state["weeks_since_peak"] == 25, f"{state.get('leg')} {state.get('weeks_since_peak')}")
    done, tot, longer = residual_waits(gaps, 25)
    check("residual wait from 25 weeks in is 15 weeks for every past cycle", len(longer) == 5 and all(x == 15 for x in longer) and done == 0, f"{longer}")
    f = state_frame(wk)
    r = base_rates(f, float(f["dd52"].iloc[-1]))
    check("from a grind this deep, a 2x within 26 weeks was far more likely than from any week",
          r[26]["p100_like"] >= 0.6 and r[26]["p100_like"] > r[26]["p100_all"] + 0.2, f"like {r[26]['p100_like']:.2f} all {r[26]['p100_all']:.2f}")

    print("Self-test 2: a random walk has no cycle: depth of drawdown does not change the +50% base rate much")
    rng = np.random.default_rng(3)
    diffs = []
    for _ in range(12):
        d = _gbm_path(8 * 252, 0.1, 0.8, rng, S0=1000.0)
        w = weekly_series(d, drop_partial=False)
        ff = state_frame(w)
        rr = base_rates(ff, 0.5)
        if rr[26]["n_like"] >= 30:
            diffs.append(rr[26]["p50_like"] - rr[26]["p50_all"])
    check("average (like minus any-week) across 12 paths is within 10 points", len(diffs) >= 6 and abs(float(np.mean(diffs))) < 0.10,
          f"mean diff {np.mean(diffs)*100:+.1f} pts over {len(diffs)} paths")
    z = zigzag(np.array([1, 2, 3, 2.0, 1.4, 2.1, 3.5]), 0.3)
    check("zigzag confirms the opening low, a peak at 3 and a trough at 1.4", [(i, k) for i, _, k in z] == [(0, "L"), (2, "H"), (4, "L")], str(z))
    print(f"\nSelf-test {'PASSED' if fails == 0 else 'FAILED'} ({fails} failures)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
