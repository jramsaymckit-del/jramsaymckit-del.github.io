#!/usr/bin/env python3
"""
decide.py  --  type a ticker, get a side: CCS (call credit spread), PCS (put credit spread) or NEITHER,
for this week, this month and the coming year.

  python3 decide.py ASTS                        # IV assumed 1.15 x realized
  python3 decide.py ASTS=105 RKLB=80            # today's IV from your screener, in percent
  python3 decide.py                             # prompt: type tickers one after another, q to quit
  python3 decide.py --test ASTS RKLB HIMS       # walk-forward: did the rule beat always-CCS, always-PCS and cash?
  python3 decide.py --test --universe universe_highbeta.txt

The determinant, identical for the three rows:
  1. The stock's own weekly closes. For horizon h (1 week, 4 weeks) every past h-week return in the trailing window
     (3 years; all data for the year row) is rescaled to today's realized vol (filtered historical simulation), so the
     shape of the past (rips, grinds, gaps) is measured in today's units.
  2. Today's 30-delta call spread and put spread (10% of spot wide, listed strike grid, wing skew, slippage,
     commissions: the same price_spread() the backtest uses), priced at today's IV: your screener's number, or m x realized.
  3. Expectancy of each side = average over the rescaled history of (net credit - capped payoff) / max risk.
  4. Hurdle = the market's average return over the horizon + 2.5 points, per unit of risk ("beat the market by 2.5%").
     A side is a candidate when its expectancy clears the hurdle; the verdict is the better candidate, else NEITHER.
The year row answers "which side, every week, for the coming year": the weekly determinant on all available history.

--test runs the identical rule at every past Friday using only data up to that Friday, then scores the realized
result of the side it chose against always-CCS, always-PCS, cash and the best side in hindsight. That is the only
evidence the verdicts deserve; read it before acting on them. No IV history exists in free data, so the test assumes
IV = m x realized throughout (--m, or --m-grid for 1.0 / 1.15 / 1.3).
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from avgweek import (BENCHMARKS, SpreadConfig, _gbm_path, load_prices, price_spread, realized_vol_series,  # noqa: E402
                     spread_pnl_at_expiry, weekly_series)

__version__ = "0.1.0"
MIN_WEEKS = 52            # analog returns needed before a verdict is given
WINDOW_WEEKS = 156        # trailing window for the week and month rows
SCALE_CLIP = (0.25, 4.0)  # bounds on the vol-rescaling ratio in the filtered historical simulation
ROWS = (("week", 1, True), ("month", 4, True), ("year", 1, False))   # name, horizon in weeks, windowed?


# ----------------------------------------------------------------------------------------------------- data shaping
def weekly_frame(daily: pd.DataFrame, drop_partial: bool = True) -> pd.DataFrame:
    """Weekly bars with the realized-vol estimate at each weekly close."""
    wk = weekly_series(daily, drop_partial=drop_partial).copy()
    rv = realized_vol_series(daily["close"], "blend")
    wk["rv"] = rv.reindex(wk["date"]).values
    return wk


def vol_1y(daily: pd.DataFrame, asof=None) -> float:
    lr = np.log(daily["close"]).diff().dropna()
    if asof is not None:
        lr = lr[lr.index <= asof]
    lr = lr.tail(252)
    return float(lr.std(ddof=1) * math.sqrt(252)) if len(lr) >= 60 else float("nan")


def default_skew(v1y: float) -> float:
    """Same convention as avgweek: upside call wing on the volatile names, index-like downside skew otherwise."""
    return 0.3 if (v1y == v1y and v1y >= 0.35) else -0.2


def analog_returns(wk: pd.DataFrame, pos: int, h: int, window_weeks: int | None, rv_now: float | None = None) -> tuple[np.ndarray, float]:
    """h-week forward simple returns of the weeks ending at or before position `pos`, each rescaled from the realized
    vol at its own start to the realized vol now. Returns spanning a hole in the data are dropped."""
    start = 0 if window_weeks is None else max(0, pos + 1 - window_weeks - h)
    closes = wk["close"].values[start:pos + 1]
    dates = wk["date"].values[start:pos + 1]
    rv = wk["rv"].values[start:pos + 1]
    if rv_now is None:
        rv_now = float(rv[-1]) if len(rv) else float("nan")
    if len(closes) <= h or not (rv_now == rv_now) or rv_now <= 0:
        return np.array([]), rv_now
    r = closes[h:] / closes[:-h] - 1.0
    span = (dates[h:] - dates[:-h]) / np.timedelta64(1, "D")
    rv_s = rv[:-h]
    ok = (span <= 7 * h + 3) & np.isfinite(rv_s) & (rv_s > 0) & np.isfinite(r) & (r > -1.0)
    scale = np.clip(rv_now / np.where(ok, rv_s, rv_now), *SCALE_CLIP)
    r_scaled = np.expm1(np.log1p(np.where(ok, r, 0.0)) * scale)
    return r_scaled[ok], rv_now


# ----------------------------------------------------------------------------------------------------- the determinant
def hurdle_for(h: int, hurdle_pts: float, market_drift: float) -> float:
    """Beat the market's average return over the horizon by `hurdle_pts`, per unit of risk."""
    return hurdle_pts + market_drift * 7.0 * h / 365.0


def side_eval(side: str, S0: float, T: float, sigma: float, r_scaled: np.ndarray, h: int, cfg: SpreadConfig) -> dict:
    c = replace(cfg, side=side)
    q = price_spread(S0, T, max(sigma, c.iv_floor), c)
    out = {"side": side, "q": q, "E": float("nan"), "SE": float("nan"), "n": int(len(r_scaled)), "win": float("nan"), "skipped": q["skipped"]}
    if q["skipped"] or len(r_scaled) == 0:
        return out
    pnl = spread_pnl_at_expiry(q, side, S0 * (1.0 + r_scaled), c.commission)
    ror = pnl / q["max_risk"]
    n_eff = max(len(ror) / h, 1.0)       # overlapping h-week windows carry about n/h independent observations
    out.update({"E": float(ror.mean()), "SE": float(ror.std(ddof=1) / math.sqrt(n_eff)) if len(ror) > 2 else float("nan"),
                "win": float((pnl > 0).mean()), "credit_frac": q["credit"] / q["width"]})
    return out


def pick(E_call: float, E_put: float, hurdle: float) -> str:
    cands = [(E, s) for E, s in ((E_call, "CCS"), (E_put, "PCS")) if E == E and E >= hurdle]
    return max(cands)[1] if cands else "NEITHER"


def decide_at(wk: pd.DataFrame, pos: int, cfg: SpreadConfig, m: float, iv: float | None, hurdle_pts: float, market_drift: float,
              window_weeks: int = WINDOW_WEEKS, S0: float | None = None, rv_now: float | None = None) -> list[dict]:
    """The three verdicts as of weekly position `pos` (spot and realized vol may be overridden with today's values)."""
    S0 = float(wk["close"].values[pos]) if S0 is None else S0
    rows = []
    for name, h, windowed in ROWS:
        r_s, rv0 = analog_returns(wk, pos, h, window_weeks if windowed else None, rv_now)
        sigma = iv if iv else m * rv0
        T = 7.0 * h / 365.0
        ec, ep = side_eval("call", S0, T, sigma, r_s, h, cfg), side_eval("put", S0, T, sigma, r_s, h, cfg)
        hd = hurdle_for(h, hurdle_pts, market_drift)
        v = pick(ec["E"], ep["E"], hd) if len(r_s) >= MIN_WEEKS else "n/a"
        rows.append({"row": name, "h": h, "verdict": v, "call": ec, "put": ep, "hurdle": hd, "n": int(len(r_s)), "sigma": sigma, "rv": rv0, "S0": S0})
    return rows


# ----------------------------------------------------------------------------------------------------- walk-forward test
def realized_ror(wk: pd.DataFrame, pos: int, h: int, side: str, sigma: float, cfg: SpreadConfig) -> float | None:
    """Return on risk of the side's spread sold at weekly position pos and held h weeks, from the actual closes."""
    closes, dates = wk["close"].values, wk["date"].values
    span = (dates[pos + h] - dates[pos]) / np.timedelta64(1, "D")
    if span > 7 * h + 3:
        return None
    q = price_spread(float(closes[pos]), 7.0 * h / 365.0, max(sigma, cfg.iv_floor), replace(cfg, side=side))
    if q["skipped"]:
        return None
    return float(spread_pnl_at_expiry(q, side, float(closes[pos + h]), cfg.commission) / q["max_risk"])


def walk_forward(wk: pd.DataFrame, cfg: SpreadConfig, m: float, hurdle_pts: float, market_drift: float,
                 window_weeks: int = WINDOW_WEEKS, min_weeks: int = MIN_WEEKS) -> dict[str, list[dict]]:
    """Verdict at each Friday from data up to that Friday only, scored on what happened next. Non-overlapping trades."""
    out: dict[str, list[dict]] = {"week": [], "month": [], "year": []}
    n = len(wk)
    rv = wk["rv"].values

    def first_eligible(h, windowed):
        """First weekly position with enough analog returns (the realized-vol warm-up eats the first weeks)."""
        for pos in range(min_weeks + h, n - h):
            r_s, rv_now = analog_returns(wk, pos, h, window_weeks if windowed else None)
            if len(r_s) >= min_weeks and rv_now == rv_now and rv_now > 0:
                return pos
        return n

    for name, h, windowed in ROWS[:2]:
        pos = first_eligible(h, windowed)
        while pos + h < n:
            r_s, rv_now = analog_returns(wk, pos, h, window_weeks if windowed else None)
            if len(r_s) >= min_weeks and rv_now == rv_now and rv_now > 0:
                sigma, T = m * rv_now, 7.0 * h / 365.0
                S0 = float(wk["close"].values[pos])
                ec, ep = side_eval("call", S0, T, sigma, r_s, h, cfg), side_eval("put", S0, T, sigma, r_s, h, cfg)
                v = pick(ec["E"], ep["E"], hurdle_for(h, hurdle_pts, market_drift))
                rc, rp = realized_ror(wk, pos, h, "call", sigma, cfg), realized_ror(wk, pos, h, "put", sigma, cfg)
                if rc is not None and rp is not None:
                    out[name].append({"date": wk["date"].values[pos], "verdict": v, "rule": rc if v == "CCS" else rp if v == "PCS" else 0.0,
                                      "ccs": rc, "pcs": rp, "E_call": ec["E"], "E_put": ep["E"]})
            pos += h
    pos = first_eligible(1, False)
    while pos + 52 < n:
        r_s, rv_now = analog_returns(wk, pos, 1, None)
        if len(r_s) >= min_weeks and rv_now == rv_now and rv_now > 0:
            sigma = m * rv_now
            S0 = float(wk["close"].values[pos])
            ec, ep = side_eval("call", S0, 7.0 / 365.0, sigma, r_s, 1, cfg), side_eval("put", S0, 7.0 / 365.0, sigma, r_s, 1, cfg)
            v = pick(ec["E"], ep["E"], hurdle_for(1, hurdle_pts, market_drift))
            cc, pp = [], []
            for p in range(pos, pos + 52):
                rvp = rv[p]
                if not (rvp == rvp) or rvp <= 0:
                    continue
                rc, rp = realized_ror(wk, p, 1, "call", m * rvp, cfg), realized_ror(wk, p, 1, "put", m * rvp, cfg)
                if rc is None or rp is None:
                    continue
                cc.append(rc)
                pp.append(rp)
            if len(cc) >= 26:
                ac, ap = float(np.mean(cc)), float(np.mean(pp))
                out["year"].append({"date": wk["date"].values[pos], "verdict": v, "rule": ac if v == "CCS" else ap if v == "PCS" else 0.0,
                                    "ccs": ac, "pcs": ap, "weeks": len(cc), "E_call": ec["E"], "E_put": ep["E"]})
        pos += 52
    return out


def score(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    df = pd.DataFrame(rows)
    n = len(df)
    traded = df[df["verdict"] != "NEITHER"]
    other = np.where(traded["verdict"] == "CCS", traded["pcs"], traded["ccs"])
    return {"n": n, "rule": float(df["rule"].mean()), "se": float(df["rule"].std(ddof=1) / math.sqrt(n)) if n > 2 else float("nan"),
            "ccs": float(df["ccs"].mean()), "pcs": float(df["pcs"].mean()), "best": float(np.maximum(np.maximum(df["ccs"], df["pcs"]), 0.0).mean()),
            "frac_ccs": float((df["verdict"] == "CCS").mean()), "frac_pcs": float((df["verdict"] == "PCS").mean()),
            "edge": float((traded["rule"].values - other).mean()) if len(traded) else float("nan"),
            "edge_se": float((traded["rule"].values - other).std(ddof=1) / math.sqrt(len(traded))) if len(traded) > 2 else float("nan"), "n_traded": int(len(traded)),
            "first": str(pd.Timestamp(df["date"].min()).date()), "last": str(pd.Timestamp(df["date"].max()).date())}


# ----------------------------------------------------------------------------------------------------- presentation
def _pct(x, d=1):
    return "n/a" if x is None or x != x else f"{x*100:+.{d}f}%"


def _upct(x, d=1):
    return "n/a" if x is None or x != x else f"{x*100:.{d}f}%"


def strikes_text(ev: dict) -> str:
    q = ev["q"]
    if ev.get("skipped") or "K_short" not in q:
        return ""
    suf = "c" if ev["side"] == "call" else "p"
    return f"sell {q['K_short']:g}{suf} / buy {q['K_long']:g}{suf}"


def card(ticker: str, rows: list[dict], asof, iv: float | None, m: float, first_year: str, n_weeks: int, why: bool = False) -> str:
    S0, rv = rows[0]["S0"], rows[0]["rv"]
    iv_txt = (f"IV {iv*100:.0f}% (screener), {iv/rv:.2f}x realized" if iv and rv == rv and rv > 0
              else f"IV assumed {m:.2f}x realized {rv*100:.0f}%")
    L = [f"{ticker}  ${S0:,.2f} on {asof}   {iv_txt}   {n_weeks} weeks since {first_year}"]
    for r in rows:
        v = r["verdict"]
        chosen = r["call"] if v == "CCS" else r["put"] if v == "PCS" else None
        act = strikes_text(chosen) if chosen else ""
        if r["row"] == "year" and chosen:
            act = "every week; this week " + act
        unit = "per trade" if r["row"] != "year" else "per week"
        L.append(f"  {r['row']:<6} {v:<8} {act:<38} CCS {_pct(r['call']['E']):>6}  PCS {_pct(r['put']['E']):>6}  of risk {unit} (hurdle {_pct(r['hurdle'])})")
        if why:
            c, p = r["call"], r["put"]
            L.append(f"         n={r['n']} analog {r['h']}-week returns; SE CCS {_upct(c['SE'])} PCS {_upct(p['SE'])}; win CCS {_upct(c['win'], 0)} PCS {_upct(p['win'], 0)}; "
                     f"credit CCS {c.get('credit_frac', float('nan'))*100:.0f}% / PCS {p.get('credit_frac', float('nan'))*100:.0f}% of width; sigma {r['sigma']*100:.0f}%"
                     + (f"; CCS {c['skipped']}" if c["skipped"] else "") + (f"; PCS {p['skipped']}" if p["skipped"] else ""))
    if not iv:
        L.append(f"  (pass {ticker}=<IV%> with your screener's implied vol; the verdict moves with it)")
    return "\n".join(L)


def test_report(pooled: dict[str, list[dict]], names: int, m: float, hurdle_pts: float, market_drift: float, spy: dict | None) -> str:
    L = [f"Walk-forward, {names} name(s), IV assumed {m:.2f}x realized, hurdle = market + {hurdle_pts*100:.1f} points per unit of risk"]
    for name, h, _ in ROWS:
        s = score(pooled[name])
        if s["n"] == 0:
            L.append(f"  {name:<6} no trades")
            continue
        unit = "trade" if name != "year" else "week"
        n_lbl = f"n {s['n']:,} {'name-years' if name == 'year' else 'trades'}"
        L.append(f"  {name:<6} rule {_pct(s['rule'])} of risk per {unit} (SE {_upct(s['se'])}, {n_lbl}, {s['first']} to {s['last']})   "
                 f"traded {(s['frac_ccs']+s['frac_pcs'])*100:.0f}% (CCS {s['frac_ccs']*100:.0f}%, PCS {s['frac_pcs']*100:.0f}%), "
                 f"chosen side minus the other {_pct(s['edge'])} (SE {_upct(s['edge_se'])})   "
                 f"always CCS {_pct(s['ccs'])}   always PCS {_pct(s['pcs'])}   best in hindsight {_pct(s['best'])}")
        if spy and name in spy and spy[name] == spy[name]:
            L.append(f"         SPY over the same span {_pct(spy[name])} per {unit}; rule minus SPY {(s['rule']-spy[name])*100:+.1f} points (target +{hurdle_pts*100:.1f})")
    L.append("  (n counts stock-weeks: the same week on several names is correlated, so the SE is optimistic; "
             "the year row cannot be more than a handful of observations per name)")
    return "\n".join(L)


# ----------------------------------------------------------------------------------------------------- IO
def parse_ticker_token(tok: str) -> tuple[str, float | None]:
    t, _, v = tok.replace(" ", "").partition("=")
    if not v:
        return t.upper(), None
    x = float(v.rstrip("%"))
    return t.upper(), (x / 100.0 if x > 3 else x)


def read_universe(path: str) -> list[str]:
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            out.extend(line.replace(",", " ").split())
    return out


def load_daily(ticker: str, args) -> tuple[pd.DataFrame, str]:
    csv = args.csv
    if args.csv_dir:
        cand = Path(args.csv_dir) / f"{ticker}.csv"
        if cand.exists():
            csv = str(cand)
        elif not csv:
            raise SystemExit(f"{cand} not found")
    daily, src = load_prices(ticker, Path(args.cache_dir).expanduser(), csv=csv, refresh=not args.no_refresh, source=args.source)
    if args.asof:
        daily = daily[daily.index <= pd.Timestamp(args.asof)]
    return daily, src


def make_cfg(args, skew: float) -> SpreadConfig:
    return SpreadConfig(side="call", delta=args.delta, width_pct=args.width, iv_mult=1.0, rv_mode="blend", skew=skew, r=args.rate,
                        snap_mode="up", slip_frac=args.slip, slip_floor_per_leg=0.03, commission=args.commission, mgmt="hold")


def run_one(ticker: str, iv: float | None, args) -> list[dict] | None:
    if ticker in BENCHMARKS:
        print(f"{ticker}  index ETF: reference only, never a candidate for selling spreads")
        return None
    try:
        daily, _ = load_daily(ticker, args)
    except (SystemExit, Exception) as exc:  # noqa: BLE001
        print(f"{ticker}  could not load prices: {exc}")
        return None
    if len(daily) < 70:
        print(f"{ticker}  only {len(daily)} daily bars; nothing to say")
        return None
    wk = weekly_frame(daily)
    if len(wk) < MIN_WEEKS + 1:
        print(f"{ticker}  only {len(wk)} complete weeks; {MIN_WEEKS} needed before a verdict")
        return None
    skew = args.skew if args.skew is not None else default_skew(vol_1y(daily))
    cfg = make_cfg(args, skew)
    # today's spot and realized vol from the last daily close; the analog distribution from the completed weeks
    rv_daily = realized_vol_series(daily["close"], "blend")
    S0, rv_now = float(daily["close"].iloc[-1]), float(rv_daily.iloc[-1])
    asof = daily.index[-1].date()
    rows = decide_at(wk, len(wk) - 1, cfg, args.m, iv, args.hurdle / 100.0, args.market_drift / 100.0, int(round(args.window * 52)), S0=S0, rv_now=rv_now)
    print(card(ticker, rows, asof, iv, args.m, str(wk.index[0].year), int(wk["ret"].notna().sum()), args.why))
    return rows


def run_test(tickers: list[str], args) -> int:
    ms = [1.0, 1.15, 1.3] if args.m_grid else [args.m]
    frames: dict[str, pd.DataFrame] = {}
    for t in tickers:
        if t in BENCHMARKS:
            continue
        try:
            daily, _ = load_daily(t, args)
        except (SystemExit, Exception) as exc:  # noqa: BLE001
            print(f"  {t}: could not load prices: {exc}")
            continue
        wk = weekly_frame(daily)
        if len(wk) < MIN_WEEKS + 10:
            print(f"  {t}: only {len(wk)} weeks, skipped")
            continue
        frames[t] = (daily, wk)
    if not frames:
        return 1
    spy = None
    try:
        sd, _ = load_daily(args.benchmark, args)
        sw = weekly_series(sd)
        spy = {"frame": sw}
    except (SystemExit, Exception) as exc:  # noqa: BLE001
        print(f"  ({args.benchmark} not available for the market comparison: {exc})")
    for m in ms:
        pooled: dict[str, list[dict]] = {"week": [], "month": [], "year": []}
        per_name = []
        for t, (daily, wk) in frames.items():
            skew = args.skew if args.skew is not None else default_skew(vol_1y(daily))
            res = walk_forward(wk, make_cfg(args, skew), m, args.hurdle / 100.0, args.market_drift / 100.0, int(round(args.window * 52)))
            for k in pooled:
                pooled[k].extend(res[k])
            s = score(res["week"])
            if s["n"]:
                per_name.append(f"  {t:<6} week: rule {_pct(s['rule'])} vs always CCS {_pct(s['ccs'])} / PCS {_pct(s['pcs'])}, traded {(s['frac_ccs']+s['frac_pcs'])*100:.0f}% of {s['n']} weeks")
        spy_means = None
        if spy:
            spy_means = {}
            sw = spy["frame"]
            for name, h, _ in ROWS:
                rows = pooled[name]
                if not rows:
                    continue
                lo, hi = pd.Timestamp(min(r["date"] for r in rows)), pd.Timestamp(max(r["date"] for r in rows))
                sub = sw[(sw["date"] >= lo) & (sw["date"] <= hi + pd.Timedelta(days=7 * (h if name != "year" else 52)))]
                c = sub["close"].values
                hh = h if name != "year" else 1
                spy_means[name] = float(np.mean(c[hh:] / c[:-hh] - 1.0)) if len(c) > hh else float("nan")
        print(test_report(pooled, len(frames), m, args.hurdle / 100.0, args.market_drift / 100.0, spy_means))
        if args.why and per_name:
            print("\n".join(per_name))
        print()
    return 0


# ----------------------------------------------------------------------------------------------------- self-test
def selftest() -> int:
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
        if not ok:
            fails += 1

    nocost = SpreadConfig(side="call", delta=0.30, width_pct=0.10, iv_mult=1.0, rv_mode="blend", skew=0.0, r=0.04, snap_mode="nearest",
                          slip_frac=0.0, slip_floor_per_leg=0.0, commission=0.0, mgmt="hold", iv_floor=0.0, strike_step=0.01)
    costs = replace(nocost, skew=0.3, snap_mode="up", slip_frac=0.02, slip_floor_per_leg=0.03, commission=0.65)
    rng = np.random.default_rng(11)
    hurdle, drift = 0.025, 0.10

    print("Self-test 1: a fair random walk (drift = r, IV = realized, no costs) gives no side and no edge")
    Ec, Ep, neither = [], [], 0
    paths = [_gbm_path(5 * 252, 0.04, 0.9, rng, S0=1000.0) for _ in range(30)]
    for d in paths:
        wk = weekly_frame(d, drop_partial=False)
        rows = decide_at(wk, len(wk) - 1, nocost, 1.0, None, hurdle, drift)
        yr = rows[2]
        Ec.append(yr["call"]["E"])
        Ep.append(yr["put"]["E"])
        neither += yr["verdict"] == "NEITHER"
    n_nan = int(np.isnan(Ec).sum() + np.isnan(Ep).sum())
    check("year-row expectancy of both sides within 1.5% of zero, averaged over 30 paths",
          n_nan == 0 and abs(np.mean(Ec)) < 0.015 and abs(np.mean(Ep)) < 0.015, f"CCS {np.nanmean(Ec)*100:+.2f}%  PCS {np.nanmean(Ep)*100:+.2f}%  unpriced {n_nan}")
    check("verdict NEITHER on at least 80% of fair paths at the +2.5 point hurdle", neither >= 24, f"{neither}/30")

    print("Self-test 2: rich IV (1.6x realized) makes both sides candidates")
    traded = 0
    for d in paths[:15]:
        wk = weekly_frame(d, drop_partial=False)
        rows = decide_at(wk, len(wk) - 1, nocost, 1.6, None, hurdle, drift)
        traded += rows[2]["verdict"] != "NEITHER"
    check("a side is chosen on at least 80% of paths", traded >= 12, f"{traded}/15")

    print("Self-test 3: drift decides the side at IV = 1.15x realized with full costs")
    def year_verdict(mu):
        wk = weekly_frame(_gbm_path(5 * 252, mu, 0.6, rng, S0=1e5), drop_partial=False)
        return decide_at(wk, len(wk) - 1, costs, 1.15, None, hurdle, drift)[2]["verdict"]
    ups = sum(year_verdict(1.5) == "PCS" for _ in range(10))
    downs = sum(year_verdict(-1.0) == "CCS" for _ in range(10))
    check("+150%/yr drift -> PCS on most paths", ups >= 7, f"{ups}/10")
    check("-100%/yr drift -> CCS on most paths", downs >= 7, f"{downs}/10")

    print("Self-test 4: filtered historical simulation rescales a quiet past to today's vol")
    quiet = _gbm_path(2 * 252, 0.0, 0.4, rng)
    loud = _gbm_path(2 * 252, 0.0, 0.8, rng, S0=float(quiet["close"].iloc[-1]))
    loud.index = pd.bdate_range(quiet.index[-1] + pd.Timedelta(days=1), periods=len(loud))
    both = pd.concat([quiet, loud])
    wk = weekly_frame(both, drop_partial=False)
    r_s, rv_now = analog_returns(wk, len(wk) - 1, 1, WINDOW_WEEKS, rv_now=0.8)   # today's vol known to be 80%
    raw = wk["ret"].dropna().tail(WINDOW_WEEKS).values
    target = 0.8 / math.sqrt(52)
    check("rescaled weekly returns have today's vol (within 12%) and more than the raw window", abs(np.std(r_s) / target - 1) < 0.12 and np.std(r_s) > np.std(raw),
          f"scaled {np.std(r_s)*100:.1f}%  raw {np.std(raw)*100:.1f}%  target {target*100:.1f}%")

    print("Self-test 5: walk-forward on fair random walks shows no false edge")
    pooled = {"week": [], "month": [], "year": []}
    for d in paths[:20]:
        res = walk_forward(weekly_frame(d, drop_partial=False), nocost, 1.0, 0.0, 0.0)
        for k in pooled:
            pooled[k].extend(res[k])
    s = score(pooled["week"])
    check("weekly rule (hurdle 0, always picks a side) is within 2.5 SE of zero", abs(s["rule"]) < 2.5 * s["se"], f"rule {s['rule']*100:+.2f}% SE {s['se']*100:.2f}% n={s['n']}")
    check("its chosen side is no better than the other side (within 2.5 SE)", abs(s["edge"]) < 2.5 * s["edge_se"], f"edge {s['edge']*100:+.2f}% SE {s['edge_se']*100:.2f}%")
    pooled2 = {"week": [], "month": [], "year": []}
    for d in paths[:20]:
        res = walk_forward(weekly_frame(d, drop_partial=False), nocost, 1.0, hurdle, drift)
        for k in pooled2:
            pooled2[k].extend(res[k])
    s2 = score(pooled2["week"])
    check("with the +2.5 point hurdle it stands aside most weeks", (s2["frac_ccs"] + s2["frac_pcs"]) < 0.35, f"traded {(s2['frac_ccs']+s2['frac_pcs'])*100:.0f}%")
    sm = score(pooled2["month"])
    check("month and year rows produce trades", sm["n"] > 0 and score(pooled2["year"])["n"] > 0, f"month n={sm['n']} year n={score(pooled2['year'])['n']}")
    print(f"\nSelf-test {'PASSED' if fails == 0 else 'FAILED'} ({fails} failures)")
    return 1 if fails else 0


# ----------------------------------------------------------------------------------------------------- CLI
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tickers", nargs="*", help="TICKER or TICKER=IV (IV in percent, e.g. ASTS=105). None: interactive prompt")
    p.add_argument("--universe", help="file with one ticker per line (# comments allowed)")
    p.add_argument("--m", type=float, default=1.15, help="IV / realized vol assumed when no IV is given (default 1.15)")
    p.add_argument("--hurdle", type=float, default=2.5, help="points over the market's return, per unit of risk (default 2.5)")
    p.add_argument("--market-drift", type=float, default=10.0, help="the market's average annual return used in the hurdle, percent (default 10)")
    p.add_argument("--window", type=float, default=3.0, help="trailing years of history for the week and month rows (default 3)")
    p.add_argument("--delta", type=float, default=0.30, help="short strike delta (default 0.30)")
    p.add_argument("--width", type=float, default=0.10, help="spread width as a fraction of spot (default 0.10)")
    p.add_argument("--skew", type=float, default=None, help="wing skew slope; default +0.3 when 1y vol >= 35%%, else -0.2")
    p.add_argument("--rate", type=float, default=0.04)
    p.add_argument("--slip", type=float, default=0.02, help="slippage as a fraction of width (default 0.02, floor $0.03 per leg)")
    p.add_argument("--commission", type=float, default=0.65, help="per contract per leg (default 0.65)")
    p.add_argument("--asof", help="pretend today is this date (YYYY-MM-DD)")
    p.add_argument("--csv", help="daily CSV for a single ticker")
    p.add_argument("--csv-dir", help="directory of TICKER.csv files")
    p.add_argument("--cache-dir", default="~/.cache/avgweek")
    p.add_argument("--source", default="auto", choices=["auto", "yfinance", "stooq", "alphavantage"])
    p.add_argument("--no-refresh", action="store_true")
    p.add_argument("--why", action="store_true", help="one extra line per row: sample size, standard errors, win rates, credits")
    p.add_argument("--test", action="store_true", help="walk-forward test of the rule on the given tickers")
    p.add_argument("--m-grid", action="store_true", help="with --test: run at IV = 1.0, 1.15 and 1.3 x realized")
    p.add_argument("--benchmark", default="SPY", help="market ticker for the --test comparison (default SPY)")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        return selftest()
    toks = list(args.tickers)
    if args.universe:
        toks += read_universe(args.universe)
    if args.test:
        return run_test([parse_ticker_token(t)[0] for t in toks], args)
    if toks:
        for tok in toks:
            t, iv = parse_ticker_token(tok)
            run_one(t, iv, args)
            print()
        return 0
    print("ticker, or ticker=IV% (e.g. ASTS=105); q to quit")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line or line.lower() in ("q", "quit", "exit"):
            break
        for tok in line.replace(",", " ").split():
            t, iv = parse_ticker_token(tok)
            run_one(t, iv, args)
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
