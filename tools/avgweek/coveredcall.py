#!/usr/bin/env python3
"""
coveredcall.py  --  own the shares, sell a call above the price once a year: what would it have returned?

  python3 coveredcall.py                          # Mag 7, four mid caps and four high-beta names, last 5 years
  python3 coveredcall.py MSFT AAPL --target 10    # your own list, calls 10% out of the money
  python3 coveredcall.py --m-grid                 # premiums at IV = 0.9, 1.0 and 1.15 x realized vol
  python3 coveredcall.py --detail                 # every yearly cycle, name by name

Each cycle: hold the shares and sell one call expiring 52 weeks later, struck --target above the price (12% by default).
At expiry the position is worth the lower of the price and the strike, plus the premium and the interest it earned; that
amount rolls into the next cycle, so shares called away are bought back at the market. Buy and hold is the same shares
with no call. Prices are Yahoo's total-return closes, so dividends count on both sides.

No free source has option price history, so each call is priced with Black-Scholes at IV = m x the trailing one-year
realized vol (m = 1 by default), with the same wing skew as the other tools, approximate one-year Treasury yields, and a
haircut on the premium for the bid-ask spread. Real one-year quotes can differ by 10 to 20%; --m-grid shows how much
that matters. Every one-year window inside the span is also scored, so a lucky start date cannot carry the result.
"""
from __future__ import annotations

import argparse
import html as _html
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from avgweek import BENCHMARKS, _gbm_path, bs_call, bs_call_delta, drop_shell_comment, load_prices, weekly_series  # noqa: E402

__version__ = "0.1.0"

GROUPS = {
    "Mag 7": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"],
    "Mid caps": ["AAL", "ETSY", "ROKU", "LYFT"],
    "High beta": ["PLTR", "COIN", "MSTR", "RKLB"],
}
# Approximate one-year US Treasury yields at the start of each quarter, interpolated in between; --rate overrides.
RATES = (("2018-01-01", 0.018), ("2019-01-01", 0.026), ("2020-01-01", 0.016), ("2020-04-01", 0.002), ("2021-01-01", 0.001),
         ("2021-10-01", 0.001), ("2022-01-01", 0.004), ("2022-04-01", 0.017), ("2022-07-01", 0.028), ("2022-10-01", 0.040),
         ("2023-01-01", 0.047), ("2023-04-01", 0.047), ("2023-07-01", 0.054), ("2023-10-01", 0.054), ("2024-01-01", 0.048),
         ("2024-04-01", 0.050), ("2024-07-01", 0.050), ("2024-10-01", 0.040), ("2025-01-01", 0.042), ("2025-04-01", 0.040),
         ("2025-07-01", 0.040), ("2025-10-01", 0.036), ("2026-01-01", 0.035), ("2026-07-01", 0.034))
EPOCH = pd.Timestamp("1970-01-01")
MIN_VOL_DAYS = 60     # daily closes needed before the first call can be priced


# ----------------------------------------------------------------------------------------------------- model
def rate_at(date, fixed: float | None = None) -> float:
    if fixed is not None:
        return fixed
    x = (pd.Timestamp(date) - EPOCH).days
    xs = [(pd.Timestamp(d) - EPOCH).days for d, _ in RATES]
    return float(np.interp(x, xs, [v for _, v in RATES]))


def default_skew(vol: float) -> float:
    """Same convention as avgweek and decide: upside call wing on volatile names, index-like downside skew otherwise."""
    return 0.3 if (vol == vol and vol >= 0.35) else -0.2


def trailing_vol(close: pd.Series, asof, days: int = 252, min_days: int = MIN_VOL_DAYS) -> float:
    """Annualized realized vol of daily log returns over the year to `asof` (at least `min_days` returns)."""
    s = close.loc[:pd.Timestamp(asof)].tail(days + 1)
    if len(s) < min_days + 1:
        return float("nan")
    lr = np.diff(np.log(s.values.astype(float)))
    return float(np.std(lr, ddof=1) * math.sqrt(252))


def call_premium(S: float, K: float, T: float, r: float, iv: float, skew: float) -> float:
    """Black-Scholes call with the tools' wing skew: IV(K) = IV x (1 + skew x (0.5 - delta(K)))."""
    d = bs_call_delta(S, K, T, r, iv)
    return float(bs_call(S, K, T, r, max(iv * (1.0 + skew * (0.5 - d)), 0.01)))


def one_cycle(close: pd.Series, wk: pd.DataFrame, pos: int, args) -> dict | None:
    """Sell the call at weekly close `pos`, hold to expiry `args.tenor` weeks later. None when it cannot be priced."""
    end = pos + args.tenor
    if pos < 0 or end >= len(wk):
        return None
    d0, d1 = pd.Timestamp(wk["date"].iloc[pos]), pd.Timestamp(wk["date"].iloc[end])
    S0, ST = float(wk["close"].iloc[pos]), float(wk["close"].iloc[end])
    rv = trailing_vol(close, d0)
    if not (rv == rv and rv > 0):
        return None
    T = max((d1 - d0).days, 1) / 365.0
    r = rate_at(d0, args.rate)
    K = S0 * (1.0 + args.target)
    iv = args.m * rv
    skew = args.skew if args.skew is not None else default_skew(rv)
    prem = call_premium(S0, K, T, r, iv, skew) * (1.0 - args.slip)
    cc_end = min(ST, K) + prem * (1.0 + r * T)
    return {"pos": pos, "start": d0, "end": d1, "S0": S0, "ST": ST, "K": K, "iv": iv, "rate": r, "premium": prem / S0,
            "capped": ST > K, "cc": cc_end / S0 - 1.0, "bh": ST / S0 - 1.0, "years": T}


def run_path(close: pd.Series, wk: pd.DataFrame, n_cycles: int, args) -> list[dict]:
    """Back-to-back cycles ending on the last complete week; a name with too little history starts a whole cycle later."""
    last = len(wk) - 1
    start = last - n_cycles * args.tenor
    while start + args.tenor <= last and one_cycle(close, wk, start, args) is None:
        start += args.tenor
    out, pos = [], start
    while pos + args.tenor <= last:
        c = one_cycle(close, wk, pos, args)
        if c is None:
            break
        out.append(c)
        pos += args.tenor
    return out


def window_stats(close: pd.Series, wk: pd.DataFrame, first_pos: int, args) -> dict:
    """Every one-year window starting on any week inside the span: how often the covered call beat holding, and was capped."""
    rows = [c for c in (one_cycle(close, wk, p, args) for p in range(max(first_pos, 0), len(wk) - args.tenor)) if c]
    if not rows:
        return {"n": 0, "ahead": float("nan"), "capped": float("nan")}
    return {"n": len(rows), "ahead": float(np.mean([c["cc"] > c["bh"] for c in rows])), "capped": float(np.mean([c["capped"] for c in rows])),
            "cc": float(np.mean([c["cc"] for c in rows])), "bh": float(np.mean([c["bh"] for c in rows]))}


def summarize(cycles: list[dict]) -> dict:
    if not cycles:
        return {"n": 0}
    cc = float(np.prod([1.0 + c["cc"] for c in cycles]) - 1.0)
    bh = float(np.prod([1.0 + c["bh"] for c in cycles]) - 1.0)
    yrs = sum(c["years"] for c in cycles)

    def ann(tot):
        return (1.0 + tot) ** (1.0 / yrs) - 1.0 if yrs > 0 and tot > -1.0 else float("nan")
    return {"n": len(cycles), "cc_total": cc, "bh_total": bh, "cc_ann": ann(cc), "bh_ann": ann(bh), "capped": int(sum(c["capped"] for c in cycles)),
            "premium": float(np.mean([c["premium"] for c in cycles])), "worst_cc": min(c["cc"] for c in cycles),
            "worst_bh": min(c["bh"] for c in cycles), "first": cycles[0]["start"], "last": cycles[-1]["end"]}


# ----------------------------------------------------------------------------------------------------- presentation
def _pct(x, d=0):
    return "n/a" if x is None or x != x else f"{x*100:+.{d}f}%"


def _upct(x, d=1):
    return "n/a" if x is None or x != x else f"{x*100:.{d}f}%"


COLS = ["group", "ticker", "CC total", "CC/yr", "B&H total", "B&H/yr", "diff/yr", "capped", "premium", "worst CC", "worst B&H", "CC ahead"]


def table_rows(results: list[dict]) -> list[list[str]]:
    rows, groups = [], list(dict.fromkeys(r["group"] for r in results))
    for g in groups:
        rs = [r for r in results if r["group"] == g and r["sum"]["n"]]
        for r in rs:
            s, w = r["sum"], r["win"]
            rows.append([g, r["ticker"] + ("" if s["n"] == r["want"] else f" ({s['n']}y)"), _pct(s["cc_total"]), _pct(s["cc_ann"], 1), _pct(s["bh_total"]),
                         _pct(s["bh_ann"], 1), f"{(s['cc_ann'] - s['bh_ann'])*100:+.1f} pts", f"{s['capped']} of {s['n']}", _upct(s["premium"]),
                         _pct(s["worst_cc"]), _pct(s["worst_bh"]), _upct(w["ahead"], 0)])
        if len(rs) > 1:
            m = lambda k: float(np.mean([r["sum"][k] for r in rs]))     # noqa: E731
            rows.append(["", "average", "", _pct(m("cc_ann"), 1), "", _pct(m("bh_ann"), 1), f"{(m('cc_ann') - m('bh_ann'))*100:+.1f} pts",
                         f"{sum(r['sum']['capped'] for r in rs)} of {sum(r['sum']['n'] for r in rs)}", _upct(m("premium")), "", "",
                         _upct(float(np.mean([r["win"]["ahead"] for r in rs])), 0)])
    return rows


def render_text(results: list[dict], args, m: float, bench: dict | None) -> str:
    ok = [r for r in results if r["sum"]["n"]]
    if not ok:
        return "no name had enough history"
    first = min(r["sum"]["first"] for r in ok).date()
    last = max(r["sum"]["last"] for r in ok).date()
    L = [f"Covered calls: own the shares, sell a call {args.target*100:.0f}% above the price every {args.tenor} weeks, IV = {m:.2f} x trailing one-year realized vol",
         f"{ok[0]['want']} cycles from {first} to {last}; a name with less history shows its cycle count. Total-return prices, premiums less {args.slip*100:.0f}% for the spread."]
    rows = [COLS] + table_rows(results)
    widths = [max(len(r[i]) for r in rows) for i in range(len(COLS))]
    for i, r in enumerate(rows):
        L.append("  " + "  ".join(c.ljust(widths[j]) if j < 2 else c.rjust(widths[j]) for j, c in enumerate(r)))
        if i == 0:
            L.append("  " + "  ".join("-" * w for w in widths))
    if bench:
        L.append(f"  {bench['ticker']} buy and hold over the same span: {_pct(bench['total'])} total, {_pct(bench['ann'], 1)} a year")
    L.append("  CC = covered call, B&H = buy and hold; /yr = compounded a year; diff = CC minus B&H; capped = years the stock finished above the "
             "strike; premium = average call premium as a share of the price; worst = worst single cycle; CC ahead = share of all one-year "
             "windows in the span, starting any week, where the covered call beat holding.")
    return "\n".join(L)


def render_detail(results: list[dict]) -> str:
    L = []
    for r in results:
        if not r["cycles"]:
            continue
        L.append(f"  {r['ticker']}")
        for c in r["cycles"]:
            L.append(f"    {c['start'].date()} to {c['end'].date()}   price {c['S0']:,.2f} to {c['ST']:,.2f}   strike {c['K']:,.2f}   IV {c['iv']*100:.0f}%   "
                     f"premium {c['premium']*100:.1f}%   CC {_pct(c['cc'], 1)}   B&H {_pct(c['bh'], 1)}" + ("   called away" if c["capped"] else ""))
    return "\n".join(L)


def render_page(blocks: list[dict], args) -> str:
    from decide import PAGE_CSS
    asof = pd.Timestamp.today().strftime("%Y-%m-%d")
    out = [f'<header><h1>covered calls</h1><span class="meta">{asof} &middot; a call {args.target*100:.0f}% above the price, sold every {args.tenor} weeks '
           f'&middot; total-return prices</span></header>']
    for b in blocks:
        ok = [r for r in b["results"] if r["sum"]["n"]]
        if not ok:
            continue
        first, last = min(r["sum"]["first"] for r in ok).date(), max(r["sum"]["last"] for r in ok).date()
        out.append(f'<h2>IV assumed {b["m"]:.2f}&times; realized</h2><p class="sub">{ok[0]["want"]} cycles, {first} to {last}. '
                   f'Each year the position is worth the lower of the price and the strike, plus the premium.</p>')
        out.append('<div class="tablewrap"><table><tr>' + "".join(f"<th>{_html.escape(c)}</th>" for c in COLS) + "</tr>")
        for row in table_rows(b["results"]):
            avg = row[1] == "average"
            cells = []
            for j, c in enumerate(row):
                cls = ""
                if j == 6 and c:
                    val = float(c.split()[0])
                    cls = ' class="ok"' if val >= 0.05 else ' class="no"' if val <= -0.05 else ""
                txt = f"<b>{_html.escape(c)}</b>" if avg and c else _html.escape(c)
                style = ' style="text-align:left"' if j < 2 else ""
                cells.append(f"<td{cls}{style}>{txt}</td>")
            out.append("<tr>" + "".join(cells) + "</tr>")
        out.append("</table></div>")
        if b.get("bench"):
            bb = b["bench"]
            out.append(f'<p class="sub">{_html.escape(bb["ticker"])} buy and hold over the same span: {_pct(bb["total"])} total, {_pct(bb["ann"], 1)} a year.</p>')
    out.append('<p class="note">CC = covered call, B&amp;H = buy and hold. /yr is compounded a year; diff is CC minus B&amp;H; capped counts the years the '
               'stock finished above the strike; premium is the average call premium as a share of the price; worst is the worst single cycle; CC ahead is '
               'the share of all one-year windows in the span, starting any week, where the covered call beat holding. Premiums are modelled with '
               'Black-Scholes at IV = m times realized vol, since no free source has option price history; real quotes can differ by 10 to 20%.</p>')
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>covered calls</title><style>{PAGE_CSS}</style></head><body><div class="wrap">{"".join(out)}</div></body></html>')


# ----------------------------------------------------------------------------------------------------- IO
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


def bench_span(daily: pd.DataFrame, first, last, ticker: str) -> dict | None:
    c = daily["close"]
    a, b = c.loc[:pd.Timestamp(first)], c.loc[:pd.Timestamp(last)]
    if not len(a) or not len(b):
        return None
    tot = float(b.iloc[-1] / a.iloc[-1] - 1.0)
    yrs = (pd.Timestamp(last) - pd.Timestamp(first)).days / 365.25
    return {"ticker": ticker, "total": tot, "ann": (1.0 + tot) ** (1.0 / yrs) - 1.0 if yrs > 0 else float("nan")}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tickers", nargs="*", help="tickers to test; none = the built-in Mag 7, mid-cap and high-beta lists")
    p.add_argument("--target", type=float, default=0.12, help="strike above the price at each sale: 0.12 or 12 for 12%% (default)")
    p.add_argument("--years", type=int, default=5, help="years of back-to-back cycles, ending at the last complete week (default 5)")
    p.add_argument("--tenor", type=int, default=52, help="weeks from sale to expiry (default 52)")
    p.add_argument("--m", type=float, default=1.0, help="IV as a multiple of trailing one-year realized vol (default 1.0)")
    p.add_argument("--m-grid", action="store_true", help="run at IV = 0.9, 1.0 and 1.15 x realized")
    p.add_argument("--skew", type=float, default=None, help="wing skew; default +0.3 when one-year vol >= 35%%, else -0.2")
    p.add_argument("--rate", type=float, default=None, help="a fixed interest rate instead of the built-in Treasury table")
    p.add_argument("--slip", type=float, default=0.03, help="haircut on each premium for the bid-ask spread: 0.03 or 3 (default)")
    p.add_argument("--detail", action="store_true", help="print every cycle")
    p.add_argument("--benchmark", default="SPY")
    p.add_argument("--asof")
    p.add_argument("--csv")
    p.add_argument("--csv-dir")
    p.add_argument("--cache-dir", default="~/.cache/avgweek")
    p.add_argument("--source", default="auto", choices=["auto", "yfinance", "stooq", "alphavantage"])
    p.add_argument("--no-refresh", action="store_true")
    p.add_argument("--html", default="coveredcall.html")
    p.add_argument("--no-html", action="store_true")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        return selftest()
    args.target = args.target / 100.0 if args.target >= 1.0 else args.target
    args.slip = args.slip / 100.0 if args.slip >= 1.0 else args.slip
    toks = [t.strip(",;").upper() for t in drop_shell_comment(args.tickers) if t.strip(",;")]
    groups = {"Your list": toks} if toks else GROUPS
    n_cycles = max(1, int(round(args.years * 52 / args.tenor)))
    data = {}
    for g, names in groups.items():
        for t in names:
            if t in BENCHMARKS:
                print(f"  {t}: index ETF, shown as the reference line only")
                continue
            try:
                daily = load_daily(t, args)
            except (SystemExit, Exception) as exc:  # noqa: BLE001
                print(f"  {t}: could not load prices: {exc}")
                continue
            data[t] = (g, daily, weekly_series(daily))
    if not data:
        return 1
    bdaily = None
    try:
        bdaily = load_daily(args.benchmark, args)
    except (SystemExit, Exception) as exc:  # noqa: BLE001
        print(f"  ({args.benchmark} not available for the reference line: {exc})")
    blocks = []
    for m in ([0.9, 1.0, 1.15] if args.m_grid else [args.m]):
        args.m = m
        results = []
        for t, (g, daily, wk) in data.items():
            cycles = run_path(daily["close"], wk, n_cycles, args)
            win = window_stats(daily["close"], wk, cycles[0]["pos"], args) if cycles else {"n": 0, "ahead": float("nan")}
            results.append({"ticker": t, "group": g, "cycles": cycles, "sum": summarize(cycles), "win": win, "want": n_cycles})
            if not cycles:
                print(f"  {t}: not enough history for one {args.tenor}-week cycle")
        ok = [r for r in results if r["sum"]["n"]]
        bench = None
        if bdaily is not None and ok:
            full = [r for r in ok if r["sum"]["n"] == n_cycles] or ok
            bench = bench_span(bdaily, full[0]["sum"]["first"], full[0]["sum"]["last"], args.benchmark)
        print(render_text(results, args, m, bench))
        if bdaily is not None and ok and bench is None:
            print(f"  ({args.benchmark} has no prices at the start of the span, so there is no reference line)")
        if args.detail:
            print(render_detail(results))
        print()
        blocks.append({"m": m, "results": results, "bench": bench})
    if not args.no_html and args.html:
        Path(args.html).write_text(render_page(blocks, args), encoding="utf-8")
        print(f"wrote {args.html}")
    return 0


# ----------------------------------------------------------------------------------------------------- self-test
def _ns(**kw):
    base = dict(target=0.12, tenor=52, m=1.0, skew=None, rate=None, slip=0.03)
    base.update(kw)
    return argparse.Namespace(**base)


def _daily_from_weekly(weekly_closes, end="2026-10-16", wiggle=0.01) -> pd.DataFrame:
    """Five sessions a week that end each week exactly on the given close, with a small alternating wiggle for some vol."""
    days = []
    for c in weekly_closes:
        days.extend([c * (1 + wiggle), c * (1 - wiggle), c * (1 + wiggle), c * (1 - wiggle), c])
    idx = pd.bdate_range(end=pd.Timestamp(end), periods=len(days))
    d = pd.DataFrame({"close": days}, index=idx)
    d["open"] = d["high"] = d["low"] = d["close"]
    return d


def selftest() -> int:
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
        if not ok:
            fails += 1

    print("Self-test 1: pricing")
    S, K, T, r, v = 100.0, 112.0, 1.0, 0.04, 0.30
    d1 = (math.log(S / K) + (r + 0.5 * v * v) * T) / (v * math.sqrt(T))
    N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))   # noqa: E731
    ref = S * N(d1) - K * math.exp(-r * T) * N(d1 - v * math.sqrt(T))
    check("a one-year call 12% out of the money at 30% vol matches the closed form (about 8.9% of the price)",
          abs(call_premium(S, K, T, r, v, 0.0) - ref) < 1e-9 and 8.8 < ref < 9.1, f"{call_premium(S, K, T, r, v, 0.0):.4f} vs {ref:.4f}")
    check("the rate table interpolates and holds its ends", abs(rate_at("2023-02-15") - 0.047) < 1e-9 and rate_at("2010-01-01") == 0.018
          and rate_at("2030-01-01") == 0.034 and rate_at("2020-01-01", 0.05) == 0.05)

    print("Self-test 2: mechanics on a hand-built path (flat warm-up, +30% in year one, -10% in year two)")
    weeks = [100.0] * 60 + [100.0 + 30.0 * i / 52 for i in range(1, 53)] + [130.0 - 13.0 * i / 52 for i in range(1, 53)]
    daily = _daily_from_weekly(weeks)
    wk = weekly_series(daily, drop_partial=False)
    a = _ns(rate=0.04)
    cyc = run_path(daily["close"], wk, 2, a)
    check("two cycles, from 100 to 130 and from 130 to 117", len(cyc) == 2 and abs(cyc[0]["S0"] - 100) < 1e-9 and abs(cyc[0]["ST"] - 130) < 1e-9
          and abs(cyc[1]["ST"] - 117) < 1e-9, str([(round(c["S0"], 2), round(c["ST"], 2)) for c in cyc]))
    c0, c1 = cyc[0], cyc[1]
    check("year one is called away: +12% plus the premium and its interest", c0["capped"] and abs(c0["cc"] - (0.12 + c0["premium"] * (1 + 0.04 * c0["years"]))) < 1e-12,
          f"CC {c0['cc']*100:+.2f}% premium {c0['premium']*100:.2f}%")
    check("year two keeps the shares: -10% plus the premium", (not c1["capped"]) and abs(c1["cc"] - (-0.10 + c1["premium"] * (1 + 0.04 * c1["years"]))) < 1e-12,
          f"CC {c1['cc']*100:+.2f}%")
    s = summarize(cyc)
    check("the two years compound, for the covered call and for holding", abs(s["cc_total"] - ((1 + c0["cc"]) * (1 + c1["cc"]) - 1)) < 1e-12 and abs(s["bh_total"] - 0.17) < 1e-9,
          f"CC {s['cc_total']*100:+.1f}%  B&H {s['bh_total']*100:+.1f}%")
    short = run_path(daily["close"].iloc[-(5 * 70):], weekly_series(daily.iloc[-(5 * 70):], drop_partial=False), 2, a)
    check("a name with too little history for the first sale starts a whole cycle later", len(short) == 1, f"{len(short)} cycle(s)")

    print("Self-test 3: on fair random walks (drift = the rate, IV = realized, no costs) the call changes nothing on average")
    rng = np.random.default_rng(5)
    fair = _ns(rate=0.04, skew=0.0, slip=0.0)
    diffs, capped = [], []
    for _ in range(80):
        d = _gbm_path(7 * 252, 0.04, 0.30, rng, S0=100.0)
        w = weekly_series(d, drop_partial=False)
        for c in run_path(d["close"], w, 5, fair):
            diffs.append(c["cc"] - c["bh"])
            capped.append(c["capped"])
    se = float(np.std(diffs, ddof=1) / math.sqrt(len(diffs)))
    check("covered call minus holding averages zero within 3 SE", abs(np.mean(diffs)) < 3 * se, f"{np.mean(diffs)*100:+.2f}% (SE {se*100:.2f}%, n {len(diffs)})")
    check("the call is exercised about as often as the model says (roughly a third of years at 30% vol)", 0.22 < np.mean(capped) < 0.48, f"{np.mean(capped)*100:.0f}%")

    print("Self-test 4: drift decides who wins")
    up, down = [], []
    for _ in range(40):
        for mu, bucket in ((0.50, up), (-0.30, down)):
            d = _gbm_path(7 * 252, mu, 0.30, rng, S0=100.0)
            w = weekly_series(d, drop_partial=False)
            bucket.extend(c["cc"] - c["bh"] for c in run_path(d["close"], w, 5, fair))
    check("a stock rising 50% a year: the covered call lags", np.mean(up) < -0.05, f"{np.mean(up)*100:+.1f}% a cycle")
    check("a stock falling 30% a year: the covered call leads", np.mean(down) > 0.03, f"{np.mean(down)*100:+.1f}% a cycle")
    d = _gbm_path(7 * 252, 0.10, 0.30, rng, S0=100.0)
    w = weekly_series(d, drop_partial=False)
    cyc = run_path(d["close"], w, 5, _ns())
    ws = window_stats(d["close"], w, cyc[0]["pos"], _ns())
    check("one-year windows: every week inside the span is a start, and the shares come out between 0 and 1", ws["n"] == len(w) - 52 - cyc[0]["pos"]
          and 0.0 <= ws["ahead"] <= 1.0 and 0.0 <= ws["capped"] <= 1.0, f"n {ws['n']} ahead {ws['ahead']*100:.0f}% capped {ws['capped']*100:.0f}%")
    print(f"\nSelf-test {'PASSED' if fails == 0 else 'FAILED'} ({fails} failures)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
