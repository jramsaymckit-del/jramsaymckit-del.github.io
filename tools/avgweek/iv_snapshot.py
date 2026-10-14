#!/usr/bin/env python3
"""
iv_snapshot.py  --  append today's weekly option-chain facts for a list of tickers to a CSV.

Why: free sources only give you TODAY's option chain. The backtest in avgweek.py has to assume the implied vol it
sells at (IV = m x realized vol). If you log the chain once a day, in a few months you own a real history of
m = IV/RV, skew and the actual credits for 50/30/20/10-delta weekly call spreads, and the assumption can be
replaced by data.

Run it once per trading day near the close (3:50pm ET), e.g. with cron or a scheduled task:
  python iv_snapshot.py ASTS RKLB HIMS --out iv_log.csv

Fields per ticker per day: date, spot, expiry, dte, atm_iv, iv_25d, iv_10d, skew_s, m = atm_iv / rv20,
strike/iv/mid for the ~50, 30, 20, 10-delta calls, and the mid credit of a 10%-wide spread off each of them.

IVs are recomputed from bid/ask mids with our own Black-Scholes inversion; Yahoo's impliedVolatility column is
kept for reference only (it is often stale on illiquid strikes).
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from avgweek import NY, bs_call, bs_call_delta, implied_vol  # noqa: E402

TARGET_DELTAS = (0.50, 0.30, 0.20, 0.10)


def snapshot(ticker: str, rate: float = 0.04, width_pct: float = 0.10) -> dict | None:
    import yfinance as yf
    tk = yf.Ticker(ticker)
    hist = tk.history(period="3mo", auto_adjust=True)
    if hist is None or hist.empty:
        print(f"  {ticker}: no price history")
        return None
    spot = float(hist["Close"].iloc[-1])
    rv20 = float(np.log(hist["Close"]).diff().dropna().tail(20).std(ddof=1) * math.sqrt(252))
    exps = tk.options
    if not exps:
        print(f"  {ticker}: no listed options")
        return None
    today = datetime.now(NY).date()
    cands = [(e, (datetime.strptime(e, "%Y-%m-%d").date() - today).days, datetime.strptime(e, "%Y-%m-%d").weekday()) for e in exps]
    window = [c for c in cands if 3 <= c[1] <= 12] or [c for c in cands if c[1] > 0]
    fridays = [c for c in window if c[2] == 4]
    exp, dte, _ = sorted(fridays or window, key=lambda c: abs(c[1] - 7))[0]
    ch = tk.option_chain(exp)
    calls = ch.calls.copy()
    try:
        live = float(ch.underlying.get("regularMarketPrice") or 0)
        if live > 0:
            spot = live
    except Exception:  # noqa: BLE001
        pass
    T = max(dte, 1) / 365.0
    calls["mid"] = (calls["bid"] + calls["ask"]) / 2.0
    calls = calls[(calls["mid"] > 0) & (calls["ask"] > 0)].copy()
    calls["iv"] = [implied_vol(m, spot, k, T, rate, True) for m, k in zip(calls["mid"], calls["strike"])]
    calls = calls.dropna(subset=["iv"]).sort_values("strike")
    if calls.empty:
        print(f"  {ticker}: could not invert any IVs")
        return None
    calls["delta"] = [bs_call_delta(spot, k, T, rate, v) for k, v in zip(calls["strike"], calls["iv"])]
    atm = calls.iloc[(calls["strike"] - spot).abs().argsort()[:2]]
    atm_iv = float(atm["iv"].mean())
    row = {"date": str(today), "ticker": ticker, "spot": round(spot, 4), "expiry": exp, "dte": dte,
           "atm_iv": round(atm_iv, 4), "rv20": round(rv20, 4), "m_iv_over_rv20": round(atm_iv / rv20, 3) if rv20 else "",
           "yahoo_atm_iv": round(float(atm["impliedVolatility"].mean()), 4)}
    def nearest(delta):
        return calls.iloc[(calls["delta"] - delta).abs().argsort()[:1]].iloc[0]
    c25 = nearest(0.25)
    row["iv_25d"] = round(float(c25["iv"]), 4)
    row["skew_s"] = round((float(c25["iv"]) / atm_iv - 1.0) / 0.25, 3) if atm_iv else ""
    for d in TARGET_DELTAS:
        c = nearest(d)
        k = float(c["strike"])
        tag = f"d{int(d*100)}"
        row[f"{tag}_strike"] = k
        row[f"{tag}_iv"] = round(float(c["iv"]), 4)
        row[f"{tag}_mid"] = round(float(c["mid"]), 3)
        row[f"{tag}_bid"] = float(c["bid"])
        row[f"{tag}_ask"] = float(c["ask"])
        # credit of a ~10%-of-spot wide spread off this strike, using the nearest listed long strike and its own mid
        target_long = k + width_pct * spot
        longs = calls[calls["strike"] > k]
        if len(longs):
            lc = longs.iloc[(longs["strike"] - target_long).abs().argsort()[:1]].iloc[0]
            row[f"{tag}_long_strike"] = float(lc["strike"])
            row[f"{tag}_spread_credit_mid"] = round(float(c["mid"] - lc["mid"]), 3)
        else:
            row[f"{tag}_long_strike"] = ""
            row[f"{tag}_spread_credit_mid"] = ""
    return row


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tickers", nargs="+")
    p.add_argument("--out", default="iv_log.csv")
    p.add_argument("--rate", type=float, default=0.04)
    p.add_argument("--width", type=float, default=0.10, help="spread width as a fraction of spot for the logged credits")
    args = p.parse_args()
    rows = []
    for t in args.tickers:
        try:
            r = snapshot(t.upper(), args.rate, args.width)
            if r:
                rows.append(r)
                print(f"  {t.upper():<6} spot {r['spot']:<9} {r['expiry']} ({r['dte']}d)  ATM IV {r['atm_iv']:.0%}  RV20 {r['rv20']:.0%}  m={r['m_iv_over_rv20']}  skew {r['skew_s']}  "
                      f"30d K {r['d30_strike']} credit(10%w) {r['d30_spread_credit_mid']}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {t}: failed: {exc}")
    if not rows:
        return 1
    out = Path(args.out)
    fields = list(rows[0].keys())
    existing = out.read_text().splitlines()[0].split(",") if out.exists() and out.stat().st_size > 0 else None
    if existing and existing != fields:
        # keep the file's own header so old and new rows stay aligned; new fields are dropped, missing ones left blank
        fields = existing
    with out.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if existing is None:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"appended {len(rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
