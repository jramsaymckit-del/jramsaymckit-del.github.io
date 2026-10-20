#!/usr/bin/env python3
"""
swing.py  --  can timing a high-beta name beat holding it? Three simple rules against buy and hold, on weekly closes.

  python3 swing.py --universe universe_highbeta.txt     # every name in the list, its full history
  python3 swing.py ASTS RKLB HIMS --years 5             # your names, the last five years
  python3 swing.py --universe universe_highbeta.txt --dd 40 --take 50

Each rule decides on a Friday close and trades at the next week's open, paying --cost (0.1%) per trade; cash earns
approximate T-bill yields.
  buy and hold   own it throughout
  dip buyer      buy when the close is --dd (50%) or more below its 52-week high; sell once it is up --take (100%) from the buy
                 (--stop sells after a fall of that much from the buy)
  trend          own it while the close is above its --ma (30)-week average, cash otherwise
  cycle          buy once it has rallied --swing (30%) off its lowest close since the last sale; sell once it has fallen --swing
                 from its highest close since the buy (the same swings cycle.py counts)

Survivorship warning: Yahoo only has names that still trade, so every list here is made of survivors. That flatters any rule
that buys falling stocks, because the ones that kept falling to zero are not in the data. Being in cash also cuts volatility
drag, so on a very volatile name with little drift a rule can beat holding just by sitting out; the "in mkt" column shows how
much of the time each rule was invested.

Two checks follow the table. The no-skill check prices random in/out timing that is invested as often as each rule and holds
for as long on average (pooled over all the names, so it knows nothing about any one stock): if random timing beats holding
on as many names as the rule does, the rule's win is just time in cash (--paths sets the paths per name, 0 skips it). The
year table shows where each rule's lead over holding came from, calendar year by calendar year.
"""
from __future__ import annotations

import argparse
import math
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from avgweek import BENCHMARKS, _gbm_path, drop_shell_comment, load_prices, weekly_series  # noqa: E402
from coveredcall import RATES as RECENT_RATES  # noqa: E402

__version__ = "0.2.0"

# Approximate average 3-month T-bill yields by year (placed at mid-year), then the quarterly table from coveredcall.py.
OLD_RATES = {1985: 0.075, 1986: 0.060, 1987: 0.058, 1988: 0.067, 1989: 0.081, 1990: 0.075, 1991: 0.054, 1992: 0.034, 1993: 0.030,
             1994: 0.043, 1995: 0.055, 1996: 0.050, 1997: 0.051, 1998: 0.048, 1999: 0.047, 2000: 0.059, 2001: 0.034, 2002: 0.016,
             2003: 0.010, 2004: 0.014, 2005: 0.032, 2006: 0.047, 2007: 0.044, 2008: 0.014, 2009: 0.0015, 2010: 0.0014, 2011: 0.0005,
             2012: 0.0009, 2013: 0.0006, 2014: 0.0003, 2015: 0.0005, 2016: 0.0032, 2017: 0.0093}
EPOCH = pd.Timestamp("1970-01-01")
_RX = [(pd.Timestamp(f"{y}-07-01") - EPOCH).days for y in OLD_RATES] + [(pd.Timestamp(d) - EPOCH).days for d, _ in RECENT_RATES]
_RY = list(OLD_RATES.values()) + [v for _, v in RECENT_RATES]
RULES = ("hold", "dip", "trend", "cycle")
LABELS = {"hold": "buy & hold", "dip": "dip buyer", "trend": "trend", "cycle": "cycle"}


def rate_at(date, fixed: float | None = None) -> float:
    if fixed is not None:
        return fixed
    return float(np.interp((pd.Timestamp(date) - EPOCH).days, _RX, _RY))


def simulate(wk: pd.DataFrame, rule: str, args, wants=None) -> dict:
    """Run one rule over a weekly frame. Decisions use data up to each Friday close; fills are at the next week's open.
    wants, if given, replaces the rule with a fixed in/out decision for each week."""
    o, c = wk["open"].values.astype(float), wk["close"].values.astype(float)
    dates = pd.to_datetime(wk["date"].values)
    n = len(c)
    hi52 = pd.Series(c).rolling(52, min_periods=52).max().values
    ma = pd.Series(c).rolling(args.ma, min_periods=args.ma).mean().values
    start = max(52, args.ma) - 1
    if n - start < 10:
        return {"n": 0}
    eq, curve, pos, entry, trades, weeks_in, log, decided = 1.0, [1.0], False, None, 0, 0, [], []
    ref_low, ref_high = c[start], None
    for t in range(start, n - 1):
        if wants is not None:
            want = bool(wants[t - start])
        elif rule == "hold":
            want = True
        elif rule == "dip":
            if not pos:
                want = c[t] <= hi52[t] * (1.0 - args.dd)
            else:
                want = c[t] < entry * (1.0 + args.take) and not (args.stop and c[t] <= entry * (1.0 - args.stop))
        elif rule == "trend":
            want = c[t] > ma[t]
        else:                                                     # cycle
            if not pos:
                ref_low = min(ref_low, c[t])
                want = c[t] >= ref_low * (1.0 + args.swing)
            else:
                ref_high = max(ref_high, c[t])
                want = c[t] > ref_high * (1.0 - args.swing)
        decided.append(want)
        rc = rate_at(dates[t], args.rate) * max((dates[t + 1] - dates[t]).days, 1) / 365.0
        if want and pos:
            r = c[t + 1] / c[t] - 1.0
        elif want:
            r = (c[t + 1] / o[t + 1]) * (1.0 - args.cost) - 1.0
            pos, entry, trades = True, o[t + 1], trades + 1
            ref_high = max(o[t + 1], c[t + 1])
            log.append({"buy": dates[t + 1], "buy_px": o[t + 1]})
        elif pos:
            r = (o[t + 1] / c[t]) * (1.0 - args.cost) - 1.0 + rc
            pos, entry = False, None
            ref_low = min(o[t + 1], c[t + 1])
            log[-1].update({"sell": dates[t + 1], "sell_px": o[t + 1]})
        else:
            r = rc
        weeks_in += int(pos)
        eq *= 1.0 + r
        curve.append(eq)
    curve = np.array(curve)
    years = max((dates[n - 1] - dates[start]).days / 365.25, 1e-9)
    return {"n": n - 1 - start, "years": years, "cagr": eq ** (1.0 / years) - 1.0 if eq > 0 else -1.0, "final": eq,
            "maxdd": float(np.max(1.0 - curve / np.maximum.accumulate(curve))), "in_mkt": weeks_in / max(n - 1 - start, 1),
            "trades": trades, "log": log, "first": dates[start], "last": dates[n - 1],
            "wants": np.array(decided, dtype=bool), "curve": curve, "dates": dates[start:n]}


def legs(wk: pd.DataFrame, args) -> dict | None:
    """Each week's return for the four things a position can do (stay in, buy, sell, stay out), so any in/out sequence can
    be priced without the loop. Same conventions as simulate()."""
    o, c = wk["open"].values.astype(float), wk["close"].values.astype(float)
    dates = pd.to_datetime(wk["date"].values)
    n, start = len(c), max(52, args.ma) - 1
    if n - start < 10:
        return None
    t = np.arange(start, n - 1)
    d0, d1 = dates[t], dates[t + 1]
    rate = np.full(len(t), args.rate) if args.rate is not None else np.interp(np.asarray((d0 - EPOCH).days), _RX, _RY)
    rc = rate * np.maximum(np.asarray((d1 - d0).days), 1) / 365.0
    return {"stay": c[t + 1] / c[t] - 1.0, "buy": (c[t + 1] / o[t + 1]) * (1.0 - args.cost) - 1.0,
            "sell": (o[t + 1] / c[t]) * (1.0 - args.cost) - 1.0 + rc, "out": rc,
            "years": max((dates[n - 1] - dates[start]).days / 365.25, 1e-9)}


def price_paths(lg: dict, W) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compounded yearly return, final wealth and worst drawdown of each row of W (True = own it the next week)."""
    W = np.atleast_2d(np.asarray(W, dtype=bool))
    prev = np.concatenate([np.zeros((len(W), 1), dtype=bool), W[:, :-1]], axis=1)
    r = np.where(W, np.where(prev, lg["stay"], lg["buy"]), np.where(prev, lg["sell"], lg["out"]))
    curve = np.concatenate([np.ones((len(W), 1)), np.cumprod(1.0 + r, axis=1)], axis=1)
    eq = curve[:, -1]
    cagr = np.where(eq > 0, np.maximum(eq, 1e-300) ** (1.0 / lg["years"]) - 1.0, -1.0)
    return cagr, eq, np.max(1.0 - curve / np.maximum.accumulate(curve, axis=1), axis=1)


def random_timing(n: int, p_in: float, p_out: float, rng, paths: int) -> np.ndarray:
    """paths x n in/out sequences from a two-state chain: buy with probability p_in each week while out, sell with probability
    p_out each week while in. The first week is drawn from the chain's long-run share of time in."""
    u = rng.random((paths, n))
    W = np.empty((paths, n), dtype=bool)
    W[:, 0] = u[:, 0] < p_in / (p_in + p_out)
    for t in range(1, n):
        W[:, t] = np.where(W[:, t - 1], u[:, t] >= p_out, u[:, t] < p_in)
    return W


def random_check(results: dict, lgs: dict, args) -> dict:
    """For each timing rule: the median compounded return per name of random timing that is invested as often as the rule and
    holds for as long on average, both pooled over all the names."""
    names = [t for t in results if results[t]["hold"]["n"] and lgs.get(t)]
    out = {}
    for k in RULES[1:]:
        w_all = sum(results[t][k]["n"] for t in names)
        w_in = sum(results[t][k]["in_mkt"] * results[t][k]["n"] for t in names)
        trades = sum(results[t][k]["trades"] for t in names)
        if trades == 0 or w_in <= 0 or w_in >= w_all:
            continue
        p_out, p_in = min(trades / w_in, 1.0), min(trades / (w_all - w_in), 1.0)
        med = {}
        for t in names:
            rng = np.random.default_rng(zlib.crc32(f"{t}:{k}".encode()))
            med[t] = float(np.median(price_paths(lgs[t], random_timing(len(lgs[t]["stay"]), p_in, p_out, rng, args.paths))[0]))
        out[k] = {"in_mkt": w_in / w_all, "p_in": p_in, "p_out": p_out, "median": med}
    return out


def yearly(res: dict) -> pd.Series:
    """Calendar-year returns of a strategy's value, without the first, partial year; the last year is to date."""
    s = pd.Series(res["curve"], index=res["dates"])
    ye = s.groupby(s.index.year).last()
    return (ye / ye.shift(1) - 1.0).iloc[1:]


# ----------------------------------------------------------------------------------------------------- presentation
def _p(x, d=1):
    return "n/a" if x is None or x != x else f"{x*100:+.{d}f}%"


def _u(x, d=0):
    return "n/a" if x is None or x != x else f"{x*100:.{d}f}%"


def year_lines(results: dict, names: list) -> list[str]:
    per = {}
    for t in names:
        ys = {k: yearly(results[t][k]) for k in RULES}
        for y, h in ys["hold"].items():
            per.setdefault(y, []).append([h] + [ys[k].get(y, np.nan) - h for k in RULES[1:]])
    keep = [y for y in sorted(per) if len(per[y]) >= 5]
    if not keep:
        return []
    L = ["", "  By calendar year: the median name's buy-and-hold return, then each rule's lead over holding (median across names, points).",
         "  Each name's first, partial year is left out; the last year is to date.",
         "    year  names  buy & hold  " + "  ".join(f"{LABELS[k]:>9}" for k in RULES[1:])]
    ahead = dict.fromkeys(RULES[1:], 0)
    for y in keep:
        med = np.nanmedian(np.array(per[y], dtype=float), axis=0)
        for i, k in enumerate(RULES[1:]):
            ahead[k] += int(med[i + 1] > 0)
        L.append(f"    {y}  {len(per[y]):>5}  {med[0]*100:>+9.0f}%  " + "  ".join(f"{m*100:>+9.0f}" for m in med[1:]))
    L.append("    years ahead of holding: " + ", ".join(f"{LABELS[k]} {ahead[k]}" for k in RULES[1:]) + f" of {len(keep)}")
    return L


def render(results: dict, bench: dict | None, args, rc: dict | None = None) -> str:
    names = [t for t, r in results.items() if r["hold"]["n"]]
    head = ["ticker", "years"] + [f"{LABELS[k]} /yr" for k in RULES] + [f"{LABELS[k]} worst" for k in RULES] + [f"in mkt {LABELS[k]}" for k in RULES[1:]] + \
           [f"trades {LABELS[k]}" for k in RULES[1:]]
    rows = []
    for t in names:
        r = results[t]
        rows.append([t, f"{r['hold']['years']:.1f}"] + [_p(r[k]["cagr"]) for k in RULES] + [_u(-r[k]["maxdd"]) for k in RULES] +
                    [_u(r[k]["in_mkt"]) for k in RULES[1:]] + [str(r[k]["trades"]) for k in RULES[1:]])
    if bench:
        rows.append([f"{args.benchmark} (ref)", f"{bench['hold']['years']:.1f}"] + [_p(bench[k]["cagr"]) for k in RULES] + [_u(-bench[k]["maxdd"]) for k in RULES] +
                    [_u(bench[k]["in_mkt"]) for k in RULES[1:]] + [str(bench[k]["trades"]) for k in RULES[1:]])
    widths = [max(len(x[i]) for x in [head] + rows) for i in range(len(head))]
    L = [f"Timing rules against buy and hold, weekly closes, fills at the next week's open, {args.cost*100:.1f}% a trade, cash at T-bill yields"
         + (f", last {args.years} years" if args.years else ", full history per name"),
         f"dip buyer: buy {args.dd*100:.0f}%+ below the 52-week high, sell at +{args.take*100:.0f}%" + (f" or -{args.stop*100:.0f}%" if args.stop else "")
         + f";  trend: above the {args.ma}-week average;  cycle: in after a {args.swing*100:.0f}% rally off the low, out after a {args.swing*100:.0f}% fall from the high"]
    for i, row in enumerate([head] + rows):
        L.append("  " + "  ".join(x.ljust(widths[j]) if j == 0 else x.rjust(widths[j]) for j, x in enumerate(row)))
        if i == 0:
            L.append("  " + "  ".join("-" * w for w in widths))
    if names:
        L.append("")
        for k in RULES[1:]:
            d = [results[t][k]["cagr"] - results[t]["hold"]["cagr"] for t in names]
            dd = [results[t]["hold"]["maxdd"] - results[t][k]["maxdd"] for t in names]
            L.append(f"  {LABELS[k]:<10} beat buy and hold in {sum(x > 0 for x in d)} of {len(d)} names; median {np.median(d)*100:+.1f} pts a year "
                     f"(average {np.mean(d)*100:+.1f}); worst drawdown {np.median(dd)*100:.0f} pts smaller at the median; invested {np.mean([results[t][k]['in_mkt'] for t in names])*100:.0f}% of the time")
        L.append(f"  buy and hold: median {np.median([results[t]['hold']['cagr'] for t in names])*100:+.1f}% a year, median worst drawdown {np.median([results[t]['hold']['maxdd'] for t in names])*100:.0f}%")
        if rc:
            L.append("")
            L.append(f"  No-skill check: random in/out timing, invested as often as each rule and for as long on average (pooled over all the names), "
                     f"{args.paths} paths per name, median path")
            for k, v in rc.items():
                ts = [t for t in names if t in v["median"]]
                d_rand = [v["median"][t] - results[t]["hold"]["cagr"] for t in ts]
                d_rule = [results[t][k]["cagr"] - v["median"][t] for t in ts]
                L.append(f"  {LABELS[k]:<10} random timing like it (in {v['in_mkt']*100:.0f}% of the time, {1/v['p_out']:.0f} weeks in and {1/v['p_in']:.0f} out "
                         f"on average) beat buy and hold in {sum(x > 0 for x in d_rand)} of {len(ts)} names (median {np.median(d_rand)*100:+.1f} pts a year); "
                         f"the rule beat random timing in {sum(x > 0 for x in d_rule)} of {len(ts)} (median {np.median(d_rule)*100:+.1f} pts)")
        L.extend(year_lines(results, names))
    L.append("  /yr = compounded a year; worst = deepest fall from a high in the strategy's own value; in mkt = share of weeks invested. Survivors only: "
             "names that went to zero are missing, which flatters the dip buyer most.")
    return "\n".join(L)


# ----------------------------------------------------------------------------------------------------- IO
def read_universe(path: str) -> list[str]:
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            out.extend(line.replace(",", " ").split())
    return out


def load_weekly(ticker: str, args) -> pd.DataFrame:
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
    wk = weekly_series(daily).reset_index(drop=True)
    if args.years:
        wk = wk.tail(int(args.years * 52) + max(52, args.ma)).reset_index(drop=True)
    return wk


def run_all(wk: pd.DataFrame, args) -> dict:
    return {k: simulate(wk, k, args) for k in RULES}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tickers", nargs="*")
    p.add_argument("--universe", help="file with one ticker per line (# comments allowed)")
    p.add_argument("--years", type=float, default=None, help="only the last N years (default: each name's full history)")
    p.add_argument("--dd", type=float, default=0.50, help="dip buyer: buy this far below the 52-week high, 0.5 or 50 (default)")
    p.add_argument("--take", type=float, default=1.00, help="dip buyer: sell at this gain from the buy, 1.0 or 100 (default)")
    p.add_argument("--stop", type=float, default=None, help="dip buyer: also sell after this fall from the buy, e.g. 0.4")
    p.add_argument("--ma", type=int, default=30, help="trend: weeks in the moving average (default 30)")
    p.add_argument("--swing", type=float, default=0.30, help="cycle: the rally and the fall that switch it, 0.3 or 30 (default)")
    p.add_argument("--cost", type=float, default=0.001, help="cost per trade as a fraction (default 0.001)")
    p.add_argument("--rate", type=float, default=None, help="a fixed cash rate instead of the T-bill table")
    p.add_argument("--paths", type=int, default=200, help="random-timing paths per name for the no-skill check, 0 to skip (default 200)")
    p.add_argument("--benchmark", default="SPY")
    p.add_argument("--asof")
    p.add_argument("--csv")
    p.add_argument("--csv-dir")
    p.add_argument("--cache-dir", default="~/.cache/avgweek")
    p.add_argument("--source", default="auto", choices=["auto", "yfinance", "stooq", "alphavantage"])
    p.add_argument("--no-refresh", action="store_true")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        return selftest()
    for k in ("dd", "take", "stop", "swing"):
        v = getattr(args, k)
        if v is not None and v >= 1.0 and k != "take":
            setattr(args, k, v / 100.0)
    if args.take >= 5.0:
        args.take = args.take / 100.0
    toks = [t.strip(",;").upper() for t in drop_shell_comment(args.tickers) if t.strip(",;")]
    if args.universe:
        toks += [t.upper() for t in read_universe(args.universe)]
    toks = list(dict.fromkeys(toks))
    if not toks:
        p.print_help()
        return 2
    results, lgs = {}, {}
    for t in toks:
        if t in BENCHMARKS:
            continue
        try:
            wk = load_weekly(t, args)
        except (SystemExit, Exception) as exc:  # noqa: BLE001
            print(f"  {t}: could not load prices: {exc}")
            continue
        results[t] = run_all(wk, args)
        lgs[t] = legs(wk, args)
    rc = random_check(results, lgs, args) if args.paths else {}
    bench = None
    try:
        bench = run_all(load_weekly(args.benchmark, args), args)
    except (SystemExit, Exception) as exc:  # noqa: BLE001
        print(f"  ({args.benchmark} not available for the reference line: {exc})")
    print(render(results, bench, args, rc))
    return 0


# ----------------------------------------------------------------------------------------------------- self-test
def _ns(**kw):
    base = dict(dd=0.50, take=1.00, stop=None, ma=30, swing=0.30, cost=0.001, rate=0.04, paths=50)
    base.update(kw)
    return argparse.Namespace(**base)


def _weekly(closes, opens=None, end="2026-10-16") -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    opens = closes if opens is None else np.asarray(opens, dtype=float)
    dates = pd.date_range(end=pd.Timestamp(end), periods=len(closes), freq="W-FRI")
    return pd.DataFrame({"open": opens, "close": closes, "date": dates})


def selftest() -> int:
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
        if not ok:
            fails += 1

    print("Self-test 1: mechanics on a hand-built path")
    closes = [100.0] * 60 + [45.0] + [46.0 + 5.0 * i for i in range(1, 12)] + [100.0] * 10
    opens = list(closes)
    opens[61] = 46.0                                   # the week after the 55% drawdown opens at 46
    wk = _weekly(closes, opens)
    r = simulate(wk, "dip", _ns())
    lg = r["log"][0] if r["log"] else {}
    first_double = next(i for i in range(62, len(closes)) if closes[i] >= 92.0)
    check("the dip buyer buys at the next week's open after the close that was 55% below the high", r["trades"] == 1 and lg.get("buy") == wk["date"][61] and lg.get("buy_px") == 46.0,
          f"{r['trades']} trade(s), bought {lg.get('buy')} at {lg.get('buy_px')}")
    check("and sells at the open after the first close at or above double the buy price", lg.get("sell") == wk["date"][first_double + 1],
          f"sold {lg.get('sell')} at {lg.get('sell_px')}")

    print("Self-test 2: a sawtooth with clear cycles (40 weeks grinding -60%, 12 weeks tripling)")
    w, price = [100.0], 100.0
    for _ in range(6):
        for _ in range(40):
            price *= 0.4 ** (1 / 40)
            w.append(price)
        for _ in range(12):
            price *= 3.0 ** (1 / 12)
            w.append(price)
    wk = _weekly(w)
    res = run_all(wk, _ns())
    check("the dip buyer and the cycle rider both beat holding a market that cycles", res["dip"]["cagr"] > res["hold"]["cagr"] and res["cycle"]["cagr"] > res["hold"]["cagr"],
          f"hold {res['hold']['cagr']*100:+.1f}%  dip {res['dip']['cagr']*100:+.1f}%  cycle {res['cycle']['cagr']*100:+.1f}%  trend {res['trend']['cagr']*100:+.1f}%")
    check("one round trip per cycle for the cycle rider", 4 <= res["cycle"]["trades"] <= 6, f"{res['cycle']['trades']} trades")

    print("Self-test 3: on fair random walks (drift = the cash rate, no costs) no rule earns anything extra")
    a = _ns(cost=0.0)
    # rules that sit in cash avoid volatility drag, so compare average final wealth, not compounded growth
    finals = {k: [] for k in RULES}
    rng = np.random.default_rng(9)
    for _ in range(300):
        d = _gbm_path(4 * 252, 0.04, 0.8, rng, S0=100.0)
        wkk = weekly_series(d, drop_partial=False).reset_index(drop=True)
        for k in RULES:
            finals[k].append(simulate(wkk, k, a)["final"])
    for k in RULES[1:]:
        diff = np.array(finals[k]) - np.array(finals["hold"])
        se = float(np.std(diff, ddof=1) / math.sqrt(len(diff)))
        check(f"{LABELS[k]}: average final wealth matches holding within 3 SE (no look-ahead)", abs(np.mean(diff)) < 3 * se,
              f"{np.mean(diff):+.3f} (SE {se:.3f})")

    print("Self-test 4: pricing a rule's own in/out sequence without the loop gives the loop's numbers")
    walk = 100.0 * np.exp(np.cumsum(np.random.default_rng(4).normal(0.002, 0.09, 400)))
    worst = 0.0
    for path in (w, walk):
        wkk = _weekly(path)
        lg = legs(wkk, _ns())
        for k in RULES:
            r = simulate(wkk, k, _ns())
            cg, eq, dd = price_paths(lg, r["wants"])
            again = simulate(wkk, "hold", _ns(), wants=r["wants"])
            worst = max(worst, abs(eq[0] / r["final"] - 1.0), abs(dd[0] - r["maxdd"]), abs(cg[0] - r["cagr"]), abs(again["final"] / r["final"] - 1.0))
    check("final wealth, compounded return and worst drawdown agree for every rule on two paths", worst < 1e-9, f"largest gap {worst:.1e}")

    print("Self-test 5: random timing is invested as often, and holds as long, as it was told to")
    W = random_timing(5000, 0.10, 0.05, np.random.default_rng(5), 40)
    spells = int((W[:, 1:] & ~W[:, :-1]).sum() + W[:, 0].sum())
    check("invested two thirds of the time (buy 10% a week while out, sell 5% a week while in)", abs(W.mean() - 2 / 3) < 0.02, f"{W.mean():.3f}")
    check("20 weeks in per spell on average", abs(W.sum() / spells - 20.0) < 1.5, f"{W.sum() / spells:.1f}")
    wkk = _weekly(walk)
    res = run_all(wkk, _ns())
    rc = random_check({"X": res}, {"X": legs(wkk, _ns())}, _ns())
    ok = all(abs(rc[k]["in_mkt"] - res[k]["in_mkt"]) < 1e-9 for k in rc)
    check("the no-skill check copies each rule's share of time invested", ok and len(rc) > 0, ", ".join(f"{k} {rc[k]['in_mkt']:.2f}" for k in rc))

    print("Self-test 6: calendar-year returns compound back to the whole")
    r = res["trend"]
    s = pd.Series(r["curve"], index=r["dates"])
    base = s[s.index.year == s.index[0].year].iloc[-1]
    ys = yearly(r)
    check("the product of the yearly returns equals the growth after the first year-end", abs(float(np.prod(1.0 + ys.values)) - s.iloc[-1] / base) < 1e-9,
          f"{len(ys)} years, {float(np.prod(1.0 + ys.values)):.4f} vs {s.iloc[-1] / base:.4f}")
    print(f"\nSelf-test {'PASSED' if fails == 0 else 'FAILED'} ({fails} failures)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
