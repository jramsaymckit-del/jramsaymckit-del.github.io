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
  4. The grind now, for a name in a down leg: the leg's depth at its low and its pace against the name's past legs; among
     past legs (own and basket) that got as deep, how many more weeks and how much further they fell before the trough, and
     how long the legs that lasted as long ran in total. Measured from the leg's low so far, which may be weeks ago.
  5. Base rates: from weeks at least as far below the 52-week high as today, how often the next 13 / 26 / 52 weeks held a
     +50% move, a 2x, a -30% fall and a -50% fall from that week's close, against the same rate from any week, the quartiles
     of the 26-week return, and the years those weeks come from. Own history and pooled; with --universe a last line counts
     how many names in the basket are in a down leg today.

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

__version__ = "0.2.1"
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
                     "weeks_to_low": trough_i - li, "depth_so_far": 1.0 - float(close[trough_i]) / lp,
                     "trough_date": pd.Timestamp(dates[trough_i]), "off_trough": float(close[last]) / float(close[trough_i]) - 1.0}
        else:
            state = {"leg": "up", "trough_date": pd.Timestamp(dates[li]), "trough_price": lp, "weeks_since_trough": last - li,
                     "gain_so_far": float(close[last]) / lp - 1.0}
    else:                                   # no swing confirmed yet: measure from the highest (or lowest) close
        hi_i = int(np.argmax(close))
        if hi_i < last:
            trough_i = hi_i + int(np.argmin(close[hi_i:]))
            state = {"leg": "down", "peak_date": pd.Timestamp(dates[hi_i]), "peak_price": float(close[hi_i]), "weeks_since_peak": last - hi_i,
                     "weeks_to_low": trough_i - hi_i, "depth_so_far": 1.0 - float(close[trough_i]) / float(close[hi_i]),
                     "trough_date": pd.Timestamp(dates[trough_i]), "off_trough": float(close[last]) / float(close[trough_i]) - 1.0}
        else:
            lo_i = int(np.argmin(close))
            state = {"leg": "up", "trough_date": pd.Timestamp(dates[lo_i]), "trough_price": float(close[lo_i]),
                     "weeks_since_trough": last - lo_i, "gain_so_far": float(close[last]) / float(close[lo_i]) - 1.0}
    return runs, state


# ----------------------------------------------------------------------------------------------------- state and base rates
def state_frame(wk: pd.DataFrame) -> pd.DataFrame:
    """Per week: drawdown from the trailing 52-week high, weeks since that high, 26-week trailing return, and the
    largest gain available over the next 13 / 26 / 52 weeks from this week's close."""
    c = wk["close"].astype(float)
    f = pd.DataFrame(index=wk.index)
    f["close"] = c
    f["year"] = np.asarray(pd.DatetimeIndex(wk.index).year)
    hi52 = c.rolling(52, min_periods=26).max()
    f["dd52"] = 1.0 - c / hi52
    f["ret26"] = c / c.shift(26) - 1.0
    # weeks since the 52-week high
    argmax = c.rolling(52, min_periods=26).apply(lambda x: len(x) - 1 - int(np.argmax(x)), raw=True)
    f["wks_since_hi52"] = argmax
    vals = c.values
    for h in HORIZONS:
        fwd, fmin, fret = (np.full(len(vals), np.nan) for _ in range(3))
        for i in range(len(vals) - 1 - h):              # full windows only; the last h weeks are censored, not failures
            win = vals[i + 1:i + 1 + h]
            fwd[i], fmin[i], fret[i] = win.max() / vals[i] - 1.0, win.min() / vals[i] - 1.0, vals[i + h] / vals[i] - 1.0
        f[f"fwd{h}"], f[f"fwdmin{h}"], f[f"fwdret{h}"] = fwd, fmin, fret
    return f


def base_rates(f: pd.DataFrame, dd_now: float, slack: float = 0.10) -> dict:
    """Within each horizon, from weeks at least (dd_now - slack) below the 52-week high and from any week: the chance of a
    +50% and a 2x move (forward maximum), of a -30% and a -50% move (forward minimum), and the quartiles of the point return."""
    out = {}
    like = f[f["dd52"] >= max(dd_now - slack, 0.0)]
    def share(series, test):
        return float(test(series).mean()) if len(series) else float("nan")
    def quart(series):
        return tuple(float(np.percentile(series, q)) for q in (25, 50, 75)) if len(series) else (float("nan"),) * 3
    for h in HORIZONS:
        rows = {}
        for tag, sub in (("all", f), ("like", like)):
            mx, mn, rt = sub[f"fwd{h}"].dropna(), sub[f"fwdmin{h}"].dropna(), sub[f"fwdret{h}"].dropna()
            rows[f"n_{tag}"] = int(len(mx))
            rows[f"p50_{tag}"], rows[f"p100_{tag}"] = share(mx, lambda x: x >= 0.5), share(mx, lambda x: x >= 1.0)
            rows[f"d30_{tag}"], rows[f"d50_{tag}"] = share(mn, lambda x: x <= -0.3), share(mn, lambda x: x <= -0.5)
            rows[f"q_{tag}"] = quart(rt)
        yrs = like.loc[like[f"fwdret{h}"].notna(), "year"] if "year" in like.columns else pd.Series(dtype=float)
        rows["years_like"] = [(int(y), float(v)) for y, v in yrs.value_counts(normalize=True).head(3).items()]
        out[h] = rows
    return out


def down_legs(wk: pd.DataFrame, thr: float) -> list[dict]:
    """Every completed peak-to-trough leg (zigzag H -> L) with its week-by-week drawdown path from the peak."""
    close = wk["close"].values.astype(float)
    piv = zigzag(close, thr)
    legs = []
    for k in range(1, len(piv)):
        (i0, p0, k0), (i1, _, k1) = piv[k - 1], piv[k]
        if k0 == "H" and k1 == "L":
            path = 1.0 - close[i0:i1 + 1] / p0
            legs.append({"peak_idx": i0, "trough_idx": i1, "depth": float(path[-1]), "weeks": i1 - i0, "path": path,
                         "slope": math.log(1.0 - float(path[-1])) / max(i1 - i0, 1)})
    return legs


def grind_context(legs: list[dict], depth_now: float, age_now: int) -> dict:
    """Among completed down legs: how many got as deep as today's, and from the week they first did, how many more weeks
    and how much further they fell to the trough; and how long the legs that were still falling at today's age ran in total."""
    remaining, further, final = [], [], []
    for leg in legs:
        hits = np.nonzero(leg["path"] >= depth_now - 1e-9)[0]
        if not len(hits):
            continue
        h = int(hits[0])
        remaining.append(leg["weeks"] - h)
        further.append(1.0 - (1.0 - leg["depth"]) / (1.0 - float(leg["path"][h])))
        final.append(leg["depth"])
    older = [leg["weeks"] for leg in legs if leg["weeks"] >= age_now]
    return {"n": len(legs), "reached": len(remaining), "remaining": remaining, "further": further, "final": final, "older": older,
            "age": age_now, "slopes": [leg["slope"] for leg in legs]}


def basket_now(results: dict, slack_days: int = 7) -> dict:
    """Today's leg for every name whose prices run to the latest week: how many are in a down leg, and how old and deep."""
    if not results:
        return {}
    lasts = {t: pd.Timestamp(res["wk"]["date"].iloc[-1]) for t, res in results.items()}
    latest = max(lasts.values())
    cur = [res["state"] for t, res in results.items() if (latest - lasts[t]).days <= slack_days]
    down = [st for st in cur if st["leg"] == "down"]
    return {"names": len(cur), "latest": latest, "stale": len(results) - len(cur), "down": len(down),
            "up": sum(st["leg"] == "up" for st in cur),
            "down_weeks": [st["weeks_since_peak"] for st in down], "down_depth": [st["depth_so_far"] for st in down]}


def pct_rank(series: pd.Series, x: float) -> float:
    s = series.dropna()
    return float((s < x).mean()) if len(s) and x == x else float("nan")


# ----------------------------------------------------------------------------------------------------- presentation
def _p(x, d=0):
    return "n/a" if x is None or x != x else f"{x*100:+.{d}f}%"


def _u(x, d=0):
    return "n/a" if x is None or x != x else f"{x*100:.{d}f}%"


def basket_line(bn: dict) -> str:
    dw, dd = bn["down_weeks"], bn["down_depth"]
    down = (f"{bn['down']} are in a down leg (median {np.median(dw):.0f} weeks since the peak and {_u(float(np.median(dd)))} deep at the low; "
            f"{sum(d >= 0.5 for d in dd)} of them 50% or deeper)" if dw else "none is in a down leg")
    stale = f"; {bn['stale']} with older prices left out" if bn["stale"] else ""
    return f"Basket now: of {bn['names']} names with prices to {bn['latest'].date()}, {down}, and {bn['up']} in an up leg{stale}."


def residual_waits(gaps: list[int], elapsed: int) -> tuple[int, int, list[int]]:
    longer = sorted(g - elapsed for g in gaps if g > elapsed)
    return len([g for g in gaps if g <= elapsed]), len(gaps), longer


def describe(ticker: str, wk: pd.DataFrame, runs: list[dict], state: dict, f: pd.DataFrame, rates: dict, pooled: dict | None,
             thr: float, min_run: float, legs_ctx: tuple | None = None) -> str:
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
    L.append(f"  26-week return now {_p(r26)}: {_u(pct_rank(f['ret26'], r26))} of this name's own 26-week windows were lower")
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
                    else f"none lasted longer than this one (the longest was {max(gaps)} weeks)")
            L.append(f"  Today's wait: {elapsed} weeks since the peak; {done} of {tot} past waits had ended by now; {more}")
    elif elapsed is not None:
        L.append(f"  Today's wait: {elapsed} weeks since the peak; no completed {min_run + 1:.0f}x run-up in this name's history to compare with")
    if pooled and pooled.get("gaps") and elapsed is not None:
        pg = pooled["gaps"]
        done, tot, longer = residual_waits(pg, elapsed)
        more = (f"those that lasted longer still had a median {int(np.median(longer))} weeks to run (quartiles {int(np.percentile(longer, 25))} to {int(np.percentile(longer, 75))})" if longer
                else "none lasted longer")
        L.append(f"  Across the other {pooled['names']} names ({tot} waits, median {int(np.median(pg))} weeks): {done} of {tot} had ended by {elapsed} weeks; {more}")
    # the grind itself
    def q(xs, d=0, pct=True):
        if not xs:
            return "n/a"
        one = (lambda x: f"{x*100:.{d}f}%") if pct else (lambda x: f"{x:.0f}")
        if len(xs) == 1:
            return one(xs[0])
        if len(xs) == 2:
            return f"{one(min(xs))} and {one(max(xs))}"
        lo, md, hi = (np.percentile(xs, 25), np.percentile(xs, 50), np.percentile(xs, 75))
        return (f"{md*100:.{d}f}% ({lo*100:.{d}f} to {hi*100:.{d}f})" if pct else f"{md:.0f} ({lo:.0f} to {hi:.0f})")
    def grind_line(tag, gc):
        if not gc or not gc["n"]:
            return f"  {tag}: no completed down legs"
        reached = f"{gc['reached']} of {gc['n']} legs got this deep" if gc["n"] < 40 else f"{gc['reached']/gc['n']*100:.0f}% of {gc['n']} legs got this deep"
        rest = (f"; from that point the trough came {q(gc['remaining'], pct=False)} weeks later and {q(gc['further'])} lower, final depth {q(gc['final'])}"
                if gc["reached"] else "")
        older = (f"; legs that lasted at least {gc['age']} weeks ran {q(gc['older'], pct=False)} weeks in total" if gc["older"]
                 else f"; none lasted {gc['age']} weeks")
        legend = " [median (quartiles)]" if max(len(gc["remaining"]), len(gc["older"])) >= 3 else ""
        return f"  {tag}: {reached}{rest}{older}{legend}"
    if state["leg"] == "down" and legs_ctx is not None:
        own_gc, pool_gc = legs_ctx
        a_low = max(int(state.get("weeks_to_low", state["weeks_since_peak"])), 1)
        pace = math.exp(math.log(max(1.0 - state["depth_so_far"], 1e-9)) / a_low) - 1.0
        med = float(np.median(own_gc["slopes"])) if own_gc and own_gc.get("slopes") else float("nan")
        past = f"; this name's past legs: median {(math.exp(med) - 1.0)*100:.1f}% a week, peak to trough" if med == med else ""
        if a_low >= state["weeks_since_peak"]:
            L.append(f"  This down leg: {_u(state['depth_so_far'])} deep at its low this week, week {a_low}, an average of {pace*100:.1f}% a week{past}")
        else:
            off = state["off_trough"]
            where = "level with it" if off < 0.005 else f"{_p(off)} above it"
            L.append(f"  This down leg: {_u(state['depth_so_far'])} deep at its low in week {a_low} ({state['trough_date'].date()}), "
                     f"an average of {pace*100:.1f}% a week{past}. The two lines below start from that low; today, week "
                     f"{state['weeks_since_peak']}, is {where}")
        L.append(grind_line("    own history ", own_gc))
        if pool_gc:
            L.append(grind_line("    basket      ", pool_gc))
    elif state["leg"] == "up" and legs_ctx is not None and legs_ctx[0] and legs_ctx[0].get("last_leg"):
        ll = legs_ctx[0]["last_leg"]
        L.append(f"  The grind before this leg: {ll['weeks']} weeks, {_u(ll['depth'])} deep")
    # base rates
    dd_now = float(last["dd52"]) if last["dd52"] == last["dd52"] else 0.0
    def trio(R, key):
        return " / ".join(f"{R[h][key]*100:.0f}" for h in HORIZONS) + "%"
    def rate_line(label, key):
        own = f"own {trio(rates, key):<16} (any week {trio(rates, key.replace('_like', '_all')):<14})"
        bask = (f"   basket {trio(pooled['rates'], key):<14} (any week {trio(pooled['rates'], key.replace('_like', '_all'))})" if pooled and pooled.get("rates") else "")
        return f"      {label:<16}{own}{bask}"
    def ns(R):
        return "/".join(f"{R[h]['n_like']:,}" for h in HORIZONS)
    n_b = f", basket n={ns(pooled['rates'])}" if pooled and pooled.get("rates") else ""
    L.append(f"  From weeks at least {_u(max(dd_now - 0.10, 0))} below the 52-week high (own n={ns(rates)}{n_b} at the three horizons), "
             f"within 13 / 26 / 52 weeks:")
    L.append(rate_line("+50% or more", "p50_like"))
    L.append(rate_line("2x or more", "p100_like"))
    L.append(rate_line("-30% or more", "d30_like"))
    L.append(rate_line("-50% or more", "d50_like"))
    def qline(R):
        lo, md, hi = R[26]["q_like"]; alo, amd, ahi = R[26]["q_all"]
        return f"median {_p(md)} (quartiles {_p(lo)} to {_p(hi)}; any week {_p(amd)}, {_p(alo)} to {_p(ahi)})"
    L.append(f"      26 weeks later   own {qline(rates)}")
    if pooled and pooled.get("rates"):
        L.append(f"                       basket {qline(pooled['rates'])}")
    def yrs(R):
        ys = R[26].get("years_like") or []
        return ", ".join(f"{y} ({v*100:.0f}%)" for y, v in ys) if ys else "n/a"
    L.append(f"      These weeks by year, top three: own {yrs(rates)}" + (f"; basket {yrs(pooled['rates'])}" if pooled and pooled.get("rates") else ""))
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
    return {"ticker": ticker, "wk": wk, "runs": runs, "state": state, "f": f, "legs": down_legs(wk, thr)}


def pooled_from(results: list[dict], dd_now: float, depth_now: float | None = None, age_now: int | None = None) -> dict:
    gaps = [r["grind_weeks"] for res in results for r in res["runs"] if r["grind_weeks"] is not None]
    frames = [res["f"] for res in results]
    f = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    legs = [leg for res in results for leg in res["legs"]]
    gc = grind_context(legs, depth_now, age_now) if (depth_now is not None and age_now is not None) else None
    return {"names": len(results), "gaps": gaps, "rates": base_rates(f, dd_now) if len(f) else None, "grind": gc}


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
        st = res["state"]
        depth_now, age_now = (st["depth_so_far"], st.get("weeks_to_low", st["weeks_since_peak"])) if st["leg"] == "down" else (None, None)
        pooled = pooled_from([r for k, r in results.items() if k != t], dd_now, depth_now, age_now) if (args.universe and len(results) > 1) else None
        own_gc = grind_context(res["legs"], depth_now, age_now) if depth_now is not None else {"last_leg": res["legs"][-1] if res["legs"] else None}
        legs_ctx = (own_gc, pooled["grind"] if pooled else None)
        print(describe(t, res["wk"], res["runs"], res["state"], res["f"], rates, pooled, args.threshold, args.run, legs_ctx))
        print()
    bn = basket_now(results) if args.universe else {}
    if bn and bn["names"] > 1:
        print(basket_line(bn))
        print()
    print("Read: a name has a handful of episodes, so its own waits, legs and rates are anecdotes; the basket rows carry the statistics. "
          "A rate near its any-week rate means the depth of the drawdown says nothing about what comes next; the grind lines say how past legs that "
          "were already this deep finished, which is a different question from whether this one is near its end. Each horizon counts only weeks "
          "with a full forward window, so the sample shrinks from left to right and the three rates in a row are not nested. When most of the "
          "basket is in a down leg at once, the grind is the sector's rather than the name's, and positions across those names are one bet.")
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
    legs = down_legs(wk, 0.30)
    gc = grind_context(legs, state["depth_so_far"], state["weeks_since_peak"])
    check("five completed down legs of 40 weeks and -60%", len(legs) == 5 and all(l["weeks"] == 40 for l in legs) and all(abs(l["depth"] - 0.6) < 1e-9 for l in legs),
          f"{[(l['weeks'], round(l['depth'], 3)) for l in legs]}")
    check("every past leg reached today's depth with 15 weeks and a further 29% to go", gc["reached"] == 5 and all(x == 15 for x in gc["remaining"]) and all(abs(x - (1 - 0.4 / 0.4 ** (25 / 40))) < 1e-6 for x in gc["further"]),
          f"remaining {gc['remaining']} further {[round(x, 3) for x in gc['further']]}")
    f = state_frame(wk)
    r = base_rates(f, float(f["dd52"].iloc[-1]))
    check("from a grind this deep a further -30% within 26 weeks was LESS likely than from any week (closer to the trough)",
          r[26]["d30_like"] < r[26]["d30_all"] and r[26]["d30_like"] < 0.45, f"like {r[26]['d30_like']:.2f} all {r[26]['d30_all']:.2f}")
    check("from a grind this deep, a 2x within 26 weeks was far more likely than from any week",
          r[26]["p100_like"] >= 0.6 and r[26]["p100_like"] > r[26]["p100_all"] + 0.2, f"like {r[26]['p100_like']:.2f} all {r[26]['p100_all']:.2f}")

    txt = describe("SAW", wk, runs, state, f, r, None, 0.30, 1.0, (gc, None))
    line = lambda t, key: next((ln.strip()[:140] for ln in t.splitlines() if key in ln), "")
    check("pct_rank is the share of windows below the value", abs(pct_rank(pd.Series([1.0, 2.0, 3.0, 4.0]), 3.5) - 0.75) < 1e-12)
    pr = pct_rank(f["ret26"], float(f["ret26"].iloc[-1]))
    check("the 26-week line reads that share as the windows that were lower", f"{pr*100:.0f}% of this name's own 26-week windows were lower" in txt,
          line(txt, "26-week return now"))
    check("a leg at its low this week: pace -2.3% a week, the same as every past leg",
          "deep at its low this week, week 25, an average of -2.3% a week" in txt and "median -2.3% a week" in txt, line(txt, "This down leg"))
    ys = r[26]["years_like"]
    check("the like-today weeks carry their years, largest share first", bool(ys) and all(a[1] >= b[1] for a, b in zip(ys, ys[1:]))
          and sum(v for _, v in ys) <= 1 + 1e-9 and line(txt, "These weeks by year") != "", str(ys))

    print("Self-test 1b: the same grind, then a 5-week bounce of +10% that does not confirm a trough")
    weeks_b = list(weeks)
    pb = weeks_b[-1]
    for w in range(5):
        pb *= 1.10 ** (1 / 5)
        weeks_b.append(pb)

    def mk(series, end):
        ix = pd.bdate_range(end=pd.Timestamp(end), periods=len(series) * 5)
        d = pd.DataFrame({"close": np.repeat(series, 5)}, index=ix)
        d["open"] = d["high"] = d["low"] = d["close"]
        w = weekly_series(d, drop_partial=False)
        rn, st = episodes(w, 0.30, 1.0)
        return {"wk": w, "runs": rn, "state": st, "f": state_frame(w), "legs": down_legs(w, 0.30)}
    b = mk(weeks_b, "2026-10-09")
    st_b = b["state"]
    check("still a DOWN leg, 30 weeks since the peak, its low in week 25, +10% off it",
          st_b["leg"] == "down" and st_b["weeks_since_peak"] == 30 and st_b.get("weeks_to_low") == 25 and abs(st_b["off_trough"] - 0.10) < 1e-9,
          f"{st_b.get('leg')} {st_b.get('weeks_since_peak')} {st_b.get('weeks_to_low')} {st_b.get('off_trough', 0):.3f}")
    gc_b = grind_context(b["legs"], st_b["depth_so_far"], st_b["weeks_to_low"])
    r_b = base_rates(b["f"], float(b["f"]["dd52"].iloc[-1]))
    txt_b = describe("SAWB", b["wk"], b["runs"], st_b, b["f"], r_b, None, 0.30, 1.0, (gc_b, None))
    check("the pace runs to the low, not to today, and the grind lines say they start from that low",
          "deep at its low in week 25" in txt_b and "an average of -2.3% a week" in txt_b and "today, week 30, is +10% above it" in txt_b,
          line(txt_b, "This down leg"))
    check("legs that lasted as long as this one did to its low: all five, 40 weeks each", gc_b["age"] == 25 and gc_b["older"] == [40] * 5
          and "legs that lasted at least 25 weeks ran 40 (40 to 40) weeks in total" in txt_b, str(gc_b["older"]))

    print("Self-test 1c: the basket-now line counts legs on the latest common week")
    up_path = [100.0 * 2.0 ** (i / 50) for i in range(51)]
    up_path += [up_path[-1] * 0.6 ** (i / 20) for i in range(1, 21)]
    up_path += [up_path[-1] * 1.5 ** (i / 30) for i in range(1, 31)]
    pool = {"A": mk(weeks, "2026-10-09"), "B": b, "C": mk(up_path, "2026-10-09"), "D": mk(up_path, "2026-08-28")}
    bn = basket_now(pool)
    check("two names down, one up, one left out for older prices", (bn["names"], bn["down"], bn["up"], bn["stale"]) == (3, 2, 1, 1), basket_line(bn))
    pr_all = pooled_from(list(pool.values()), 0.4)
    check("pooled years come through the concat", bool(pr_all["rates"][26]["years_like"]), str(pr_all["rates"][26]["years_like"]))

    print("Self-test 2: a random walk has no cycle: depth of drawdown does not change the +50% base rate much")
    rng = np.random.default_rng(3)
    diffs, ddiffs = [], []
    for _ in range(12):
        d = _gbm_path(8 * 252, 0.1, 0.8, rng, S0=1000.0)
        w = weekly_series(d, drop_partial=False)
        ff = state_frame(w)
        rr = base_rates(ff, 0.5)
        if rr[26]["n_like"] >= 30:
            diffs.append(rr[26]["p50_like"] - rr[26]["p50_all"])
            ddiffs.append(rr[26]["d30_like"] - rr[26]["d30_all"])
    check("average (like minus any-week) for +50% across 12 paths is within 10 points", len(diffs) >= 6 and abs(float(np.mean(diffs))) < 0.10,
          f"mean diff {np.mean(diffs)*100:+.1f} pts over {len(diffs)} paths")
    check("the same for a -30% fall", len(ddiffs) >= 6 and abs(float(np.mean(ddiffs))) < 0.10, f"mean diff {np.mean(ddiffs)*100:+.1f} pts")
    z = zigzag(np.array([1, 2, 3, 2.0, 1.4, 2.1, 3.5]), 0.3)
    check("zigzag confirms the opening low, a peak at 3 and a trough at 1.4", [(i, k) for i, _, k in z] == [(0, "L"), (2, "H"), (4, "L")], str(z))
    print(f"\nSelf-test {'PASSED' if fails == 0 else 'FAILED'} ({fails} failures)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
