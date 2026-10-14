#!/usr/bin/env python3
"""
avgweek.py  --  "Average week" drift statistics + weekly call-credit-spread backtest, one ticker (or a basket) at a time.

Standalone. No paid data. Not wired into the screener (but writes a summary.json the screener could read).

  * Daily OHLCV from yfinance (fallback: Stooq CSV download, or a CSV you pass with --csv / --csv-dir).
  * Weeks are last-trading-day to last-trading-day (Friday to Friday, Thursday when Friday is a holiday),
    which is exactly the life of a weekly option.
  * Option prices are Black-Scholes on daily closes with an implied-vol PROXY (trailing realized vol x a
    multiplier m, plus a mild call skew). Hold-to-expiry results are therefore "exact history of the stock,
    assumed entry credit": the only assumed number per trade is the credit, and the tool solves for the
    break-even multiplier m_be instead of pretending to know the historical premium.

Typical use
  python avgweek.py ASTS
  python avgweek.py ASTS RKLB HIMS SPY --iv-from-chain --out reports        # basket + pooled view + benchmark
  python avgweek.py ASTS --csv my_asts_daily.csv                           # offline
  python avgweek.py --selftest                                             # proves the math on synthetic data

Dependencies: numpy, pandas. (pip install yfinance requests for live data.) scipy is NOT required.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
from dataclasses import dataclass, asdict, replace
from datetime import date, datetime, time as dtime
from zoneinfo import ZoneInfo
from pathlib import Path

import numpy as np
import pandas as pd

__version__ = "0.4.2"

# ======================================================================================================
# 1. Normal distribution + chi-square helpers (self-contained; trivially portable to JS)
# ======================================================================================================

SQRT2 = math.sqrt(2.0)


def ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT2))


def nppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation + one Halley refinement)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"nppf: p must be in (0,1), got {p}")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    elif p <= phigh:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    e = ncdf(x) - p
    u = e * math.sqrt(2.0 * math.pi) * math.exp(0.5 * x * x)
    return x - u / (1.0 + 0.5 * x * u)


def chi2_sf_df4(x: float) -> float:
    """Survival function of chi-square with 4 degrees of freedom (closed form for even df)."""
    if x <= 0:
        return 1.0
    return math.exp(-x / 2.0) * (1.0 + x / 2.0)


# ======================================================================================================
# 2. Black-Scholes (European, no dividends: none of the target names pay one)
# ======================================================================================================

def bs_d1(S, K, T, r, sigma):
    return (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))


def bs_call(S, K, T, r, sigma):
    if T <= 0.0 or sigma <= 0.0:
        return max(S - K, 0.0)
    d1 = bs_d1(S, K, T, r, sigma)
    return S * ncdf(d1) - K * math.exp(-r * T) * ncdf(d1 - sigma * math.sqrt(T))


def bs_put(S, K, T, r, sigma):
    if T <= 0.0 or sigma <= 0.0:
        return max(K - S, 0.0)
    return bs_call(S, K, T, r, sigma) - S + K * math.exp(-r * T)


def bs_call_delta(S, K, T, r, sigma):
    if T <= 0.0 or sigma <= 0.0:
        return 1.0 if S > K else 0.0
    return ncdf(bs_d1(S, K, T, r, sigma))


def strike_for_call_delta(S, T, r, sigma, delta):
    """Strike whose BS call delta equals `delta`. delta=0.5 is (almost exactly) at-the-money."""
    sq = sigma * math.sqrt(T)
    return S * math.exp(-(nppf(delta) * sq - (r + 0.5 * sigma * sigma) * T))


def strike_for_put_delta(S, T, r, sigma, delta):
    """Strike whose BS put delta equals -`delta` (delta given as a positive number)."""
    sq = sigma * math.sqrt(T)
    return S * math.exp(-(nppf(1.0 - delta) * sq - (r + 0.5 * sigma * sigma) * T))


def implied_vol(price, S, K, T, r, is_call=True, lo=0.01, hi=6.0):
    """Bisection BS inversion. Returns nan if the price is outside no-arbitrage bounds."""
    f = bs_call if is_call else bs_put
    if not (price > 0) or T <= 0:
        return float("nan")
    if f(S, K, T, r, lo) > price or f(S, K, T, r, hi) < price:
        return float("nan")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(S, K, T, r, mid) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def default_strike_step(S: float) -> float:
    """Typical listed increments for weeklies on liquid US single names. Override with --strike-step."""
    if S < 25:
        return 0.5
    if S < 100:
        return 1.0
    if S < 200:
        return 2.5
    return 5.0


def snap(x: float, step: float, mode: str = "nearest") -> float:
    if mode == "up":
        v = math.ceil(x / step - 1e-9) * step
    elif mode == "down":
        v = math.floor(x / step + 1e-9) * step
    else:
        v = round(x / step) * step
    return round(v, 4)


# ======================================================================================================
# 3. Data
# ======================================================================================================

def _normalise_daily(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.rename(columns={c: str(c).strip().lower().replace(" ", "_") for c in df.columns})
    if "adj_close" in df.columns and "close" in df.columns:
        factor = df["adj_close"] / df["close"]
        for c in ("open", "high", "low"):
            if c in df.columns:
                df[c] = df[c] * factor
        df["close"] = df["adj_close"]
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep].copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    for c in ("open", "high", "low"):
        if c not in df.columns:
            df[c] = df["close"]
        bad = ~(pd.to_numeric(df[c], errors="coerce") > 0)
        df.loc[bad, c] = df.loc[bad, "close"]
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)
    return df


NY = ZoneInfo("America/New_York")


def expected_last_session(now: datetime | None = None) -> date:
    """The most recent session whose close is final: today after 16:15 New York on a weekday, else the previous
    business day. Holidays are not special-cased (a holiday makes the cache look stale one day early: harmless)."""
    now = now or datetime.now(NY)
    d = now.date()
    if now.weekday() < 5 and now.time() >= dtime(16, 15):
        return d
    return (pd.Timestamp(d) - pd.offsets.BDay(1)).date()


def drop_live_bar(df: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """yfinance includes today's bar at the live price during the session; drop it until the close is final."""
    if df.empty:
        return df
    now = now or datetime.now(NY)
    if df.index[-1].date() == now.date() and now.time() < dtime(16, 15):
        return df.iloc[:-1]
    return df


def fetch_yfinance(ticker: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(ticker, period="max", interval="1d", auto_adjust=True, progress=False, threads=False)
    if df is None or len(df) == 0:
        raise RuntimeError(f"yfinance returned no rows for {ticker}")
    return drop_live_bar(_normalise_daily(df))


def fetch_stooq(ticker: str) -> pd.DataFrame:
    import requests
    from io import StringIO
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
    resp = requests.get(url, timeout=30, headers={"User-Agent": "avgweek/" + __version__})
    resp.raise_for_status()
    if "Date,Open" not in resp.text[:200]:
        raise RuntimeError(f"Stooq returned no data for {ticker}: {resp.text[:80]!r}")
    return drop_live_bar(_normalise_daily(pd.read_csv(StringIO(resp.text), parse_dates=["Date"], index_col="Date")))


def fetch_alphavantage(ticker: str) -> pd.DataFrame:
    """Free tier: ~25 requests/day, UNADJUSTED daily bars (fine for names that never split; check before trusting)."""
    import requests
    from io import StringIO
    key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not key:
        raise RuntimeError("set ALPHAVANTAGE_API_KEY to use Alpha Vantage")
    url = ("https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&outputsize=full&datatype=csv"
           f"&symbol={ticker}&apikey={key}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    if not resp.text.startswith("timestamp,"):
        raise RuntimeError(f"Alpha Vantage: {resp.text[:120]!r}")
    df = pd.read_csv(StringIO(resp.text), parse_dates=["timestamp"], index_col="timestamp")
    return drop_live_bar(_normalise_daily(df))


FETCHERS = {"yfinance": fetch_yfinance, "stooq": fetch_stooq, "alphavantage": fetch_alphavantage}


def read_csv_prices(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        raise SystemExit(f"CSV not found: {path}  (for --csv-dir the file must be named TICKER.csv)")
    df = pd.read_csv(path)
    if len(df) and str(df.columns[0]).strip().lower() == "price" and str(df.iloc[0, 0]).strip().lower() == "ticker":
        # yf.download(...).to_csv(): three header rows (Price / Ticker / Date)
        df = pd.read_csv(path, header=0, skiprows=[1, 2])
        df = df.rename(columns={df.columns[0]: "Date"})
    date_col = next((c for c in df.columns if str(c).lower() in ("date", "datetime", "timestamp")), df.columns[0])
    raw = df[date_col].astype(str)
    try:
        idx = pd.to_datetime(raw)
        if idx.dt.tz is not None:
            idx = idx.dt.tz_convert(NY).dt.tz_localize(None)
    except (ValueError, TypeError):
        idx = pd.to_datetime(raw, utc=True, errors="coerce").dt.tz_convert(NY).dt.tz_localize(None)
    df = df.set_index(pd.DatetimeIndex(idx)).drop(columns=[date_col])
    df = df[df.index.notna()]
    return _normalise_daily(df)


def load_prices(ticker: str, cache_dir: Path, csv: str | None = None, refresh: bool = True,
                source: str = "auto", force: bool = False) -> tuple[pd.DataFrame, str]:
    """Daily OHLCV. Cache is re-downloaded in full (never appended: a split rewrites adjusted history) when its
    last row is older than the last final session, or when force=True."""
    if csv:
        return read_csv_prices(csv), f"csv:{csv}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{ticker.upper()}.csv"
    cached = None
    if cache_file.exists():
        try:
            cached = _normalise_daily(pd.read_csv(cache_file, parse_dates=["date"], index_col="date"))
            if cached.empty:
                raise ValueError("empty cache")
        except Exception:  # noqa: BLE001 - a truncated or empty cache file is treated as missing
            cached = None
        if cached is not None:
            last = cached.index[-1].date()
            if not force and (not refresh or last >= expected_last_session()):
                return cached, f"cache:{cache_file} (through {last})"
    errors = []
    order = ["yfinance", "stooq", "alphavantage"] if source == "auto" else [source]
    fetchers = [(n, FETCHERS[n]) for n in order]
    for name, fn in fetchers:
        try:
            df = fn(ticker)
            if len(df) < 30:
                raise RuntimeError(f"{name} returned only {len(df)} rows")
            df.to_csv(cache_file, index_label="date")
            return df, f"{name} (cached to {cache_file})"
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")
    if cached is not None:
        return cached, f"cache:{cache_file} (STALE, refresh failed: {'; '.join(errors)})"
    raise SystemExit(f"Could not load {ticker}: " + " | ".join(errors) +
                     "\nTip: download a daily CSV (stooq.com or Yahoo 'Download') and pass --csv file.csv")


# ======================================================================================================
# 4. Weekly series and statistics
# ======================================================================================================

LOOKBACKS = [("1m", 1), ("2m", 2), ("3m", 3), ("6m", 6), ("1y", 12), ("2y", 24), ("3y", 36), ("5y", 60), ("all", None)]
EXCEEDANCE_K = (0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)
MIN_WEEKS_FOR_DIRECTION = 26


MAX_WEEK_GAP_DAYS = 10


def weekly_series(daily: pd.DataFrame, drop_partial: bool = True, now: datetime | None = None) -> pd.DataFrame:
    """Weekly bars labelled by the week's Friday. `date` = actual last trading day; `first_date` = first."""
    d = daily.copy()
    d["date"] = d.index
    g = d.resample("W-FRI")
    wk = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last(),
        "date": g["date"].last(), "first_date": g["date"].first(), "ndays": g["close"].count(),
    })
    wk = wk[wk["ndays"] > 0].copy()
    wk.index.name = "week_ending"
    if drop_partial and len(wk):
        label, last = wk.index[-1], wk["date"].iloc[-1]
        now = now or datetime.now(NY)
        today = pd.Timestamp(now.date())
        # the trailing bin is incomplete if its Friday has not closed yet; a past week ending Mon-Wed is a data cut
        incomplete_now = label > today or (label == today and now.time() < dtime(16, 15))
        if label > last and (last.weekday() < 3 or incomplete_now):
            wk = wk.iloc[:-1].copy()
    wk["prev_close"] = wk["close"].shift(1)
    wk["gap_days"] = (wk["date"] - wk["date"].shift(1)).dt.days
    wk["ret"] = wk["close"] / wk["prev_close"] - 1.0
    wk["logret"] = np.log(wk["close"] / wk["prev_close"])
    wk["gap_ret"] = wk["open"] / wk["prev_close"] - 1.0      # weekend gap: first open vs previous week's close
    wk["hold_ret"] = wk["close"] / wk["open"] - 1.0           # Monday open -> Friday close
    gap = wk["gap_days"] > MAX_WEEK_GAP_DAYS                   # a hole in the data is not a week
    wk.loc[gap, ["ret", "logret", "gap_ret", "hold_ret"]] = np.nan
    wk.attrs["data_gaps"] = int(gap.sum())
    return wk


def week_window(wk: pd.DataFrame, cutoff) -> pd.DataFrame:
    """Weeks whose whole span lies after `cutoff` (a week labelled L spans (L-7d, L])."""
    return wk if cutoff is None else wk[wk.index >= cutoff + pd.Timedelta(days=7)]


def wilson_ci(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def median_ci(x: pd.Series, z: float = 1.96) -> tuple[float, float]:
    """Distribution-free CI for the median from order statistics."""
    s = np.sort(np.asarray(x, dtype=float))
    n = len(s)
    if n < 6:
        return (float("nan"), float("nan"))
    L = max(int(math.floor((n - z * math.sqrt(n)) / 2)), 1)   # 1-based ranks, symmetric about the middle
    U = n - L + 1
    return (float(s[L - 1]), float(s[U - 1]))


def max_drawdown(prices: pd.Series) -> float:
    if len(prices) == 0:
        return float("nan")
    return float((prices / prices.cummax() - 1.0).min())


def lognormal_exceedance(k: float, sigma_w: float) -> float:
    """P(simple return > k) for a driftless lognormal week with log-sd sigma_w (the 'no edge' benchmark)."""
    if sigma_w <= 0:
        return float("nan")
    return ncdf((-math.log1p(k) - 0.5 * sigma_w * sigma_w) / sigma_w)


def lookback_cutoff(asof: pd.Timestamp, months: int | None):
    return None if months is None else asof - pd.DateOffset(months=months)


def window_metrics(wk_sub: pd.DataFrame, daily_sub: pd.DataFrame | None, pooled: bool = False) -> dict | None:
    """The 'average week' block for one window (a lookback rung, a calendar year, or a pooled stack)."""
    r = wk_sub["ret"].dropna()
    lr = wk_sub["logret"].dropna()
    n = int(len(r))
    if n < 4:
        return None
    sd_log = float(lr.std(ddof=1))
    up, down = r[r > 0], r[r < 0]
    avg_up = float(up.mean()) if len(up) else float("nan")
    avg_down = float(down.mean()) if len(down) else float("nan")
    win = float((r > 0).mean())
    wlo, whi = wilson_ci(win * n, n)
    mlo, mhi = median_ci(r)
    if daily_sub is not None and len(daily_sub) > 10:
        dlr = np.log(daily_sub["close"]).diff().dropna()
        ann_vol = float(dlr.std(ddof=1) * math.sqrt(252))
    else:
        ann_vol = sd_log * math.sqrt(52)
    sigma_w = ann_vol / math.sqrt(52)
    s2 = sd_log * sd_log
    return {
        "weeks": n,
        "cum_return": float("nan") if pooled else float(math.exp(lr.sum()) - 1.0),
        "mean_wk": float(r.mean()),
        "mean_log_wk": float(lr.mean()),
        "vol_drag_wk": 0.5 * s2,
        "median_wk": float(r.median()),
        "median_ci_lo": mlo, "median_ci_hi": mhi,
        "win_rate": win, "win_ci_lo": wlo, "win_ci_hi": whi,
        "win_null": lognormal_exceedance(0.0, sigma_w),
        "win_null_lo": lognormal_exceedance(0.0, sigma_w) - 1.96 * math.sqrt(0.25 / n),
        "win_null_hi": lognormal_exceedance(0.0, sigma_w) + 1.96 * math.sqrt(0.25 / n),
        "t_stat": float(lr.mean() / (sd_log / math.sqrt(n))) if sd_log > 0 else float("nan"),
        "direction_reliable": n >= MIN_WEEKS_FOR_DIRECTION,
        "avg_up_wk": avg_up, "avg_down_wk": avg_down,
        "payoff_ratio": (avg_up / abs(avg_down)) if (len(up) and len(down) and avg_down) else float("nan"),
        "profit_factor_wk": float(up.sum() / abs(down.sum())) if len(down) and down.sum() else float("nan"),
        "sd_wk": float(r.std(ddof=1)),
        "ann_vol": ann_vol,
        "skew": float(r.skew()) if n >= 3 else float("nan"),
        "skew_lognormal_null": float((math.exp(s2) + 2) * math.sqrt(math.exp(s2) - 1)) if s2 > 0 else float("nan"),
        "tail_ratio_5": float(r.nlargest(5).mean() / abs(r.nsmallest(5).mean())) if n >= 10 and r.nsmallest(5).mean() else float("nan"),
        "cum_ex_top5": float(math.exp(lr.sum() - lr.nlargest(5).sum()) - 1.0) if (n >= 10 and not pooled) else float("nan"),
        "n_wk_gt_15": int((r > 0.15).sum()),
        "exp_n_wk_gt_15_lognormal": float(n * lognormal_exceedance(0.15, sigma_w)),
        "pct_wk_gt_10": float((r > 0.10).mean()),
        "pct_wk_lt_m10": float((r < -0.10).mean()),
        "best_wk": float(r.max()), "worst_wk": float(r.min()),
        "max_dd": float("nan") if pooled else max_drawdown(wk_sub["close"]),
        "pooled": pooled,
        "lag1_autocorr": float(np.corrcoef(lr.values[1:], lr.values[:-1])[0, 1]) if n > 5 and sd_log > 0 else float("nan"),
    }


def weekly_stats(wk: pd.DataFrame, daily: pd.DataFrame | None, lookbacks=LOOKBACKS) -> pd.DataFrame:
    asof = wk.index[-1]
    rows = []
    for label, months in lookbacks:
        cutoff = lookback_cutoff(asof, months)
        sub = week_window(wk, cutoff)
        dsub = None if daily is None else (daily if cutoff is None else daily[daily.index > cutoff])
        m = window_metrics(sub, dsub)
        if m:
            rows.append({"window": label, **m})
    return pd.DataFrame(rows).set_index("window") if rows else pd.DataFrame()


def yearly_stats(wk: pd.DataFrame, daily: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    for y, sub in wk.groupby(wk.index.year):
        dsub = None if daily is None else daily[daily.index.year == y]
        m = window_metrics(sub, dsub)
        if m:
            rows.append({"window": str(y), **m})
    return pd.DataFrame(rows).set_index("window") if rows else pd.DataFrame()


def horizon_block(lr: pd.Series, h: int = 3, b_iid: int = 10000, b_block: int = 2000, block_len: float = 4.0,
                  seed: int = 0) -> dict:
    """P(up over h weeks) four ways + the h-week return distribution. lr = weekly log returns of the window."""
    x = np.asarray(lr.dropna(), dtype=float)
    n = len(x)
    out = {"weeks": n}
    if n < h + 4:
        return out
    rng = np.random.default_rng(seed)
    rh = np.convolve(x, np.ones(h), mode="valid")  # overlapping h-week log returns
    out.update({
        "p_up_overlap": float((rh > 0).mean()), "n_windows": int(len(rh)), "n_eff": float(len(rh) / h),
        "p_up_nonoverlap": float((rh[::h] > 0).mean()),
        "p_up_normal": ncdf(math.sqrt(h) * x.mean() / x.std(ddof=1)) if x.std(ddof=1) > 0 else float("nan"),
    })
    idx = rng.integers(0, n, size=(b_iid, h))
    s = x[idx].sum(axis=1)
    q = np.quantile(s, [0.05, 0.25, 0.50, 0.75, 0.95])
    out.update({
        "p_up_iid_boot": float((s > 0).mean()),
        "q05": float(math.expm1(q[0])), "q25": float(math.expm1(q[1])), "q50": float(math.expm1(q[2])),
        "q75": float(math.expm1(q[3])), "q95": float(math.expm1(q[4])),
        "mean_h": float(np.expm1(s).mean()),
        "p_gt_10": float((s > math.log(1.10)).mean()), "p_gt_20": float((s > math.log(1.20)).mean()),
        "p_gt_30": float((s > math.log(1.30)).mean()), "p_lt_m20": float((s < math.log(0.80)).mean()),
    })
    # stationary block bootstrap (Politis-Romano) CI for the overlapping P(up)
    stats = np.empty(b_block)
    p_new = 1.0 / block_len
    for b in range(b_block):
        starts = rng.integers(0, n, size=n)
        cont = rng.random(n) >= p_new
        ii = np.empty(n, dtype=int)
        ii[0] = starts[0]
        for t in range(1, n):
            ii[t] = (ii[t - 1] + 1) % n if cont[t] else starts[t]
        rb = np.convolve(x[ii], np.ones(h), mode="valid")
        stats[b] = (rb > 0).mean()
    out["p_up_block_ci_lo"] = float(np.percentile(stats, 2.5))
    out["p_up_block_ci_hi"] = float(np.percentile(stats, 97.5))
    return out


def exceedance_table(r: pd.Series, ann_vol: float, widths=(0.05, 0.10), ks=EXCEEDANCE_K, rate: float = 0.04,
                     T: float = 7 / 365, iv_today: float | None = None) -> pd.DataFrame:
    """Model-free core: how often and by how much weekly closes exceeded +k%, vs the lognormal benchmark,
    and what a spread sold k% OTM would have paid out vs what BS at m=1 would have credited."""
    r = r.dropna()
    n = len(r)
    sigma_w = ann_vol / math.sqrt(52)
    rows = []
    for k in ks:
        exc = r > k
        p = float(exc.mean()) if n else float("nan")
        lo, hi = wilson_ci(exc.sum(), n)
        over = (r[exc] - k)
        row = {"k_otm": k, "n": n, "realized_p": p, "ci_lo": lo, "ci_hi": hi,
               "lognormal_p": lognormal_exceedance(k, sigma_w),
               "avg_overshoot": float(over.mean()) if len(over) else 0.0}
        if iv_today:
            row["implied_p_today"] = lognormal_exceedance(k, iv_today / math.sqrt(52))
        for w in widths:
            payout = float(np.minimum(over, w).sum() / n) if n else float("nan")  # expected payout per week, % of spot
            credit = bs_call(1.0, 1.0 + k, T, rate, ann_vol) - bs_call(1.0, 1.0 + k + w, T, rate, ann_vol)
            row[f"payout_w{int(w*100)}"] = payout
            row[f"full_width_share_w{int(w*100)}"] = float((over >= w).mean()) if len(over) else float("nan")
            row[f"bs_credit_w{int(w*100)}"] = credit
            row[f"credit_over_payout_w{int(w*100)}"] = credit / payout if payout > 0 else float("nan")  # no breaches: undefined, see the CI
        rows.append(row)
    return pd.DataFrame(rows).set_index("k_otm")


def dependence_block(lr: pd.Series) -> dict:
    x = np.asarray(lr.dropna(), dtype=float)
    n = len(x)
    out = {"weeks": n, "acf_se": 1 / math.sqrt(n) if n else float("nan")}
    if n < 12:
        return out
    acf = {}
    xm = x - x.mean()
    den = float((xm * xm).sum())
    for k in (1, 2, 3, 4):
        acf[k] = float((xm[k:] * xm[:-k]).sum() / den) if den > 0 else float("nan")
        out[f"acf{k}"] = acf[k]
    q = n * (n + 2) * sum(acf[k] ** 2 / (n - k) for k in (1, 2, 3, 4))
    out["ljung_box_q4"] = float(q)
    out["ljung_box_p"] = chi2_sf_df4(q)
    v1 = x.var(ddof=1)
    for k in (2, 3, 4):
        rk = np.convolve(x, np.ones(k), mode="valid")
        out[f"vr{k}"] = float(rk.var(ddof=1) / (k * v1)) if v1 > 0 else float("nan")
        out[f"vr{k}_se"] = math.sqrt(2 * (2 * k - 1) * (k - 1) / (3 * k * n))
    rho = acf[1]
    out["n_eff_lag1"] = float(n * (1 - rho) / (1 + rho)) if rho > -1 else float("nan")
    return out


def conditional_stats(wk: pd.DataFrame, daily: pd.DataFrame | None, years: float = 2.0, big: float = 0.15) -> pd.DataFrame:
    """Next-week distribution conditional on what just happened. Compare rows to 'all weeks', not to zero."""
    asof = wk.index[-1]
    sub = week_window(wk, asof - pd.DateOffset(months=int(round(years * 12)))).copy()
    r = sub["ret"]
    prev = r.shift(1)
    r3_prev = sub["logret"].rolling(3).sum().shift(1)
    full_close = wk["close"]
    hi52_prev = full_close.rolling(52, min_periods=26).max().shift(1).reindex(sub.index)
    lo26_prev = full_close.rolling(26, min_periods=13).min().shift(1).reindex(sub.index)
    prev_close = sub["close"].shift(1)
    conds = {
        "all weeks": prev.notna(),
        "after up week": prev > 0,
        "after down week": prev < 0,
        f"after week > +{int(big*100)}%": prev > big,
        f"after week < -{int(big*100)}%": prev < -big,
        "after 2 up weeks": (prev > 0) & (r.shift(2) > 0),
        "after 2 down weeks": (prev < 0) & (r.shift(2) < 0),
        "after 3wk > +30%": r3_prev > math.log(1.30),
        "after 3wk < -25%": r3_prev < math.log(0.75),
        "after new 52wk high": prev_close >= hi52_prev,
        "after new 26wk low": prev_close <= lo26_prev,
    }
    if daily is not None:
        rv20 = np.log(daily["close"]).diff().rolling(20).std(ddof=1) * math.sqrt(252)
        sub["rv20"] = rv20.reindex(sub["date"]).values
        rv_prev = sub["rv20"].shift(1)
        if rv_prev.notna().sum() > 9:
            terc = rv_prev.quantile([1 / 3, 2 / 3]).values
            conds["RV20 low tercile at entry"] = rv_prev <= terc[0]
            conds["RV20 mid tercile at entry"] = (rv_prev > terc[0]) & (rv_prev <= terc[1])
            conds["RV20 high tercile at entry"] = rv_prev > terc[1]
    rows = []
    for name, mask in conds.items():
        x = r[mask.fillna(False) & r.notna()]
        n = int(len(x))
        win = float((x > 0).mean()) if n else float("nan")
        lo, hi = wilson_ci(win * n, n) if n else (float("nan"), float("nan"))
        rows.append({"condition": name, "n": n,
                     "next_wk_mean": float(x.mean()) if n else float("nan"),
                     "next_wk_median": float(x.median()) if n else float("nan"),
                     "next_wk_win_rate": win, "win_ci_lo": lo, "win_ci_hi": hi,
                     "next_wk_p_gt_10": float((x > 0.10).mean()) if n else float("nan"),
                     "next_wk_worst": float(x.min()) if n else float("nan")})
    out = pd.DataFrame(rows).set_index("condition")
    out.attrs["window_years"] = years
    return out


def daily_decomposition(daily: pd.DataFrame, lookbacks=LOOKBACKS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Close-to-close daily return split into overnight (prev close -> open) and intraday (open -> close),
    plus the gap-event counts a weekly spread cannot manage. Returns (table, top-10 sessions)."""
    d = daily.copy()
    d["cc"] = d["close"].pct_change()
    d["overnight"] = d["open"] / d["close"].shift(1) - 1.0
    d["intraday"] = d["close"] / d["open"] - 1.0
    d["r_on"] = np.log1p(d["overnight"])
    d["r_id"] = np.log1p(d["intraday"])
    asof = d.index[-1]
    rows = []
    for label, months in lookbacks:
        cutoff = lookback_cutoff(asof, months)
        sub = (d if cutoff is None else d[d.index > cutoff]).dropna(subset=["cc", "overnight", "intraday"])
        if len(sub) < 10:
            continue
        v_on, v_id = float((sub["r_on"] ** 2).sum()), float((sub["r_id"] ** 2).sum())
        rows.append({
            "window": label, "days": int(len(sub)),
            "daily_mean": float(sub["cc"].mean()), "daily_median": float(sub["cc"].median()),
            "daily_win_rate": float((sub["cc"] > 0).mean()),
            "overnight_mean": float(sub["overnight"].mean()), "overnight_win_rate": float((sub["overnight"] > 0).mean()),
            "intraday_mean": float(sub["intraday"].mean()), "intraday_win_rate": float((sub["intraday"] > 0).mean()),
            "overnight_total": float(np.prod(1 + sub["overnight"]) - 1), "intraday_total": float(np.prod(1 + sub["intraday"]) - 1),
            "overnight_var_share": v_on / (v_on + v_id) if (v_on + v_id) > 0 else float("nan"),
            "n_days_abs_gt_10": int((sub["cc"].abs() > 0.10).sum()), "n_days_abs_gt_20": int((sub["cc"].abs() > 0.20).sum()),
            "n_gaps_abs_gt_10": int((sub["overnight"].abs() > 0.10).sum()),
        })
    dd = d.dropna(subset=["cc"])
    top = dd.reindex(dd["cc"].abs().sort_values(ascending=False).index)[["cc", "overnight", "intraday"]].head(10)
    return pd.DataFrame(rows).set_index("window") if rows else pd.DataFrame(), top


def rolling_regime(wk: pd.DataFrame, window: int = 26) -> dict:
    r = wk["ret"].dropna()
    med = r.rolling(window).median().dropna()
    win = (r > 0).rolling(window).mean().dropna()
    out = {"window_weeks": window, "n": int(len(med))}
    if len(med) < 2:
        return out
    sign = np.sign(med)
    changes = med.index[1:][(sign.values[1:] != sign.values[:-1])]
    out.update({"median_min": float(med.min()), "median_max": float(med.max()), "median_now": float(med.iloc[-1]),
                "win_min": float(win.min()), "win_max": float(win.max()), "win_now": float(win.iloc[-1]),
                "last_sign_change": str(changes[-1].date()) if len(changes) else "never",
                "n_sign_changes": int(len(changes))})
    return out


def _pct(v, d: int = 1, sign: bool = True) -> str:
    if v is None or v != v:
        return "n/a"
    return f"{v*100:+.{d}f}%" if sign else f"{v*100:.{d}f}%"


def _num(v, d: int = 2) -> str:
    if v is None or v != v:
        return "n/a"
    if v == float("inf"):
        return ">4"
    return f"{v:.{d}f}"


def poisson_tail(k: int, lam: float) -> float:
    """P(X >= k) for X ~ Poisson(lam)."""
    if lam <= 0:
        return 1.0 if k <= 0 else 0.0
    cdf, term = 0.0, math.exp(-lam)
    for i in range(k):
        cdf += term
        term *= lam / (i + 1)
    return max(0.0, 1.0 - cdf)


def verdict_lines(stats: pd.DataFrame, horizon: dict, window: str = "1y") -> list[str]:
    if stats.empty or window not in stats.index:
        return ["(not enough data for a verdict)"]
    s = stats.loc[window]
    t = s["t_stat"]
    n = int(s["weeks"])
    if t != t:
        direction = "not computable (no variance)"
    else:
        sig_t = abs(t) >= 2
        win_sig = not (s["win_ci_lo"] < 0.5 < s["win_ci_hi"])
        if sig_t and win_sig and ((t > 0) != (s["win_rate"] > 0.5)):
            direction = "MIXED: the mean says " + ("up" if t > 0 else "down") + ", the typical week says the opposite"
        elif sig_t:
            direction = ("UP" if t > 0 else "DOWN") + " on the mean (t beyond 2)"
        elif win_sig:
            direction = ("UP" if s["win_rate"] > 0.5 else "DOWN") + " on the typical week (win-rate CI excludes 50%)"
        else:
            direction = "not distinguishable from zero"
    l1 = (f"DIRECTION ({window}, n={n}): median {_pct(s['median_wk'])}/wk [{_pct(s['median_ci_lo'])}, {_pct(s['median_ci_hi'])}], "
          f"win {_pct(s['win_rate'], 0, False)} [{_pct(s['win_ci_lo'], 0, False)}, {_pct(s['win_ci_hi'], 0, False)}], "
          f"mean log {_pct(s['mean_log_wk'], 2)}/wk (t={_num(t)}) -> {direction}.")
    shape = []
    se_skew = math.sqrt(6.0 / n) if n else float("nan")
    if s["skew"] == s["skew"] and s["skew_lognormal_null"] == s["skew_lognormal_null"] and s["skew"] > s["skew_lognormal_null"] + 2 * se_skew:
        shape.append("right-tailed beyond lognormal (skew more than 2 SE above the null)")
    k15 = int(s["n_wk_gt_15"])
    p_pois = poisson_tail(k15, float(s["exp_n_wk_gt_15_lognormal"])) if s["exp_n_wk_gt_15_lognormal"] == s["exp_n_wk_gt_15_lognormal"] else 1.0
    if k15 >= 3 and p_pois < 0.05:
        shape.append(f"excess +15% weeks (Poisson tail p={p_pois:.2f})")
    l2 = (f"SHAPE ({window}): skew {_num(s['skew'])} (lognormal null {_num(s['skew_lognormal_null'])}, SE {_num(se_skew)}), "
          f"{k15} weeks > +15% (lognormal expects {_num(s['exp_n_wk_gt_15_lognormal'], 1)}), tail ratio {_num(s['tail_ratio_5'])}, "
          f"window {_pct(s['cum_return'], 0)} vs {_pct(s['cum_ex_top5'], 0)} without its top 5 weeks "
          f"-> {', '.join(shape) if shape else 'not separable from a plain high-vol random walk at this sample size'}.")
    l3 = ""
    if horizon and "p_up_block_ci_lo" in horizon:
        l3 = (f"3-WEEK: P(up) {_pct(horizon['p_up_overlap'], 0, False)} overlapping (n_eff {horizon['n_eff']:.0f}), "
              f"{_pct(horizon['p_up_iid_boot'], 0, False)} bootstrap, block-bootstrap 95% CI [{_pct(horizon['p_up_block_ci_lo'], 0, False)}, {_pct(horizon['p_up_block_ci_hi'], 0, False)}]; "
              f"median 3-wk move {_pct(horizon['q50'])}, P(> +20%) {_pct(horizon['p_gt_20'], 0, False)}, P(< -20%) {_pct(horizon['p_lt_m20'], 0, False)}.")
    return [l1, l2] + ([l3] if l3 else [])


# ======================================================================================================
# 5. Weekly credit-spread backtest (call side = the thesis; put side = the mirror test)
# ======================================================================================================

@dataclass
class SpreadConfig:
    side: str = "call"             # 'call' (sell call credit spread) or 'put' (mirror test)
    delta: float = 0.30            # short-strike target |delta| (0.5 = ATM)
    width_pct: float = 0.10        # long strike this far from the short strike, as % of spot (snapped to grid)
    width_abs: float | None = None # or a fixed dollar width
    iv_mult: float = 1.0           # m: implied vol proxy = m x realized-vol estimate
    rv_mode: str = "blend"         # 'blend' = max(0.25 RV10 + 0.5 RV20 + 0.25 RV60, 0.6 RV252) | 'rv20' | 'longrun' = max(RV20, 0.75 RV252)
    iv_floor: float = 0.20
    skew: float = 0.3              # IV(K) = IV_atm x (1 + skew x (0.5 - |delta_K|)); 0 = flat vol
    r: float = 0.04
    strike_step: float | None = None
    snap_mode: str = "up"          # short strike snapped away from spot ('up' for calls, mirrored for puts) | 'nearest'
    slip_frac: float = 0.02        # haircut per open and per close, fraction of width ...
    slip_floor_per_leg: float = 0.03  # ... with this floor per leg (a one-tick haircut even on tiny spreads)
    commission: float = 0.65       # per contract per leg
    mgmt: str = "hold"             # hold | tp50 | sl2x | tp50_sl2x | sl_touch | close1dte  (daily-close triggers, model marks)
    mark_iv_mult: float = 1.25     # management marks use iv x this once spot is beyond the short strike (IV expands on rips)
    min_credit_frac: float = 0.03
    earnings: str = "skip"         # skip | include | bump   (only matters when earnings dates are known)
    earnings_move: float = 0.12    # implied move used by 'bump'
    risk_budget: float = 2000.0    # for the constant-risk sizing variant

    def label(self) -> str:
        w = f"${self.width_abs:g}" if self.width_abs else f"{self.width_pct:.0%}"
        return f"{self.side}_d{self.delta:.2f}_w{w}_m{self.iv_mult:g}_{self.mgmt}"


def realized_vol_series(close: pd.Series, mode: str = "blend") -> pd.Series:
    lr = np.log(close).diff()
    def rv(n):
        return lr.rolling(n).std(ddof=1) * math.sqrt(252)
    if mode == "rv20":
        return rv(20)
    if mode == "longrun":
        blend, floor = rv(20), 0.75 * rv(252)
    else:
        blend = 0.25 * rv(10) + 0.5 * rv(20) + 0.25 * rv(60)
        floor = 0.6 * rv(252)
    out = blend.copy()
    both = blend.notna() & floor.notna()
    out[both] = np.maximum(blend[both], floor[both])
    return out


def regime_frame(daily: pd.DataFrame, short_interest: pd.DataFrame | None = None, float_shares: float | None = None) -> pd.DataFrame:
    c = daily["close"]
    f = pd.DataFrame(index=daily.index)
    if short_interest is not None and len(short_interest):
        si = short_interest.sort_values("available_date")
        m = pd.merge_asof(pd.DataFrame({"date": daily.index}), si[["available_date", "short_interest", "days_to_cover"]].rename(columns={"available_date": "date"}),
                          on="date", direction="backward")
        f["si_days_to_cover"] = m["days_to_cover"].values
        f["si_shares"] = m["short_interest"].values
        f["si_pct_float"] = (m["short_interest"].values / float_shares) if float_shares else np.nan
    f["dma50"] = c.rolling(50).mean()
    f["dma50_slope"] = f["dma50"].diff(20)
    f["dma200"] = c.rolling(200).mean()
    f["high52"] = c.rolling(252, min_periods=60).max()
    f["pct_from_high52"] = c / f["high52"] - 1.0
    rv20 = np.log(c).diff().rolling(20).std(ddof=1) * math.sqrt(252)
    f["rv20"] = rv20
    def _pctile(x):
        past = x[:-1]
        past = past[~np.isnan(past)]
        return float((x[-1] > past).mean()) if len(past) >= 40 and x[-1] == x[-1] else float("nan")
    f["rv20_pctile"] = rv20.rolling(252, min_periods=60).apply(_pctile, raw=True)
    above = c > f["dma50"]
    rising = f["dma50_slope"] > 0
    f["trend"] = np.where(above & rising, "up", np.where(~above & ~rising, "down", "mixed"))
    f.loc[f["dma50"].isna() | f["dma50_slope"].isna(), "trend"] = "n/a"
    return f


def backtest_spreads(daily: pd.DataFrame, wk: pd.DataFrame, cfg: SpreadConfig, earnings: set | None = None,
                     regimes: pd.DataFrame | None = None) -> pd.DataFrame:
    """Sell one credit spread at each weekly close, expiring at the next weekly close. One row per week."""
    rv = realized_vol_series(daily["close"], cfg.rv_mode)
    closes, highs, lows = daily["close"], daily["high"], daily["low"]
    close_pos = {d: i for i, d in enumerate(closes.index)}
    close_vals = closes.values
    is_call = cfg.side == "call"
    rows = []
    idx = list(wk.index)
    for i in range(len(idx) - 1):
        e_lbl, x_lbl = idx[i], idx[i + 1]
        entry_date, exp_date = wk.at[e_lbl, "date"], wk.at[x_lbl, "date"]
        S0, ST = float(wk.at[e_lbl, "close"]), float(wk.at[x_lbl, "close"])
        prev_ret = wk.at[e_lbl, "ret"]
        rec = {"entry_date": entry_date, "expiry_date": exp_date, "S0": S0, "ST": ST, "stock_ret": ST / S0 - 1.0,
               "prev_week_ret": float(prev_ret) if prev_ret == prev_ret else float("nan"), "skipped": None}
        if regimes is not None and entry_date in regimes.index:
            rg = regimes.loc[entry_date]
            rec.update({"trend": rg["trend"], "pct_from_high52": rg["pct_from_high52"], "rv20_pctile": rg["rv20_pctile"],
                        "above_dma200": bool(S0 > rg["dma200"]) if rg["dma200"] == rg["dma200"] else None})
        if (exp_date - entry_date).days > MAX_WEEK_GAP_DAYS:
            rec["skipped"] = "data gap"
            rows.append(rec)
            continue
        rv0 = rv.get(entry_date, float("nan"))
        if not (rv0 == rv0) or rv0 <= 0:
            rec["skipped"] = "no realized vol yet"
            rows.append(rec)
            continue
        p0, p1 = close_pos[entry_date], close_pos[exp_date]
        n_sessions = p1 - p0
        T = max((exp_date - entry_date).days, 1) / 365.0   # calendar clock, same as the chain calibration (dte/365)
        sigma = max(cfg.iv_mult * rv0, cfg.iv_floor)
        # a report dated on the entry Friday could be after the close (lands in this week): treat it as in-week too
        ew = bool(earnings) and any(entry_date <= e <= exp_date for e in earnings)
        rec["earnings_week"] = ew
        if ew and cfg.earnings == "skip":
            rec["skipped"] = "earnings week"
            rows.append(rec)
            continue
        if ew and cfg.earnings == "bump":
            sigma = math.sqrt(sigma * sigma + (cfg.earnings_move / 0.8) ** 2 / T)
        step = cfg.strike_step or default_strike_step(S0)
        width = cfg.width_abs if cfg.width_abs else cfg.width_pct * S0
        width = max(step, snap(width, step))
        if is_call:
            K_raw = strike_for_call_delta(S0, T, cfg.r, sigma, cfg.delta)
            K_short = max(snap(K_raw, step, "up" if cfg.snap_mode == "up" else "nearest"), step)
            K_long = round(K_short + width, 4)
            d_s = bs_call_delta(S0, K_short, T, cfg.r, sigma)
            d_l = bs_call_delta(S0, K_long, T, cfg.r, sigma)
            iv_s, iv_l = sigma * (1 + cfg.skew * (0.5 - d_s)), sigma * (1 + cfg.skew * (0.5 - d_l))
            credit_mid = bs_call(S0, K_short, T, cfg.r, iv_s) - bs_call(S0, K_long, T, cfg.r, iv_l)
            payoff = max(0.0, min(ST - K_short, width))
            otm = K_short / S0 - 1.0
        else:
            K_raw = strike_for_put_delta(S0, T, cfg.r, sigma, cfg.delta)
            K_short = max(snap(K_raw, step, "down" if cfg.snap_mode == "up" else "nearest"), step)
            K_long = round(K_short - width, 4)
            if K_long <= 0:
                rec["skipped"] = "long put strike <= 0"
                rows.append(rec)
                continue
            d_s = 1.0 - bs_call_delta(S0, K_short, T, cfg.r, sigma)
            d_l = 1.0 - bs_call_delta(S0, K_long, T, cfg.r, sigma)
            iv_s, iv_l = sigma * (1 + cfg.skew * (0.5 - d_s)), sigma * (1 + cfg.skew * (0.5 - d_l))
            credit_mid = bs_put(S0, K_short, T, cfg.r, iv_s) - bs_put(S0, K_long, T, cfg.r, iv_l)
            payoff = max(0.0, min(K_short - ST, width))
            otm = 1.0 - K_short / S0
        slip = max(cfg.slip_frac * width, 2 * cfg.slip_floor_per_leg)
        credit = credit_mid - slip
        rec.update({"rv": rv0, "sigma": sigma, "sessions": n_sessions, "T_days": (exp_date - entry_date).days, "K_short": K_short, "K_long": K_long, "width": width,
                    "short_otm_pct": otm, "model_delta": d_s, "credit_mid": credit_mid, "credit": credit})
        if credit_mid < cfg.min_credit_frac * width or credit <= 0:
            rec["skipped"] = "credit too small"
            rows.append(rec)
            continue
        exit_date, exit_reason, exit_value = exp_date, "expiry", None
        if cfg.mgmt != "hold":
            tp, sl, c1, st = "tp50" in cfg.mgmt, "sl2x" in cfg.mgmt, "close1dte" in cfg.mgmt, "sl_touch" in cfg.mgmt
            for p in range(p0 + 1, p1):
                S_ = close_vals[p]
                d_ = closes.index[p]
                T_rem = T * (p1 - p) / max(n_sessions, 1)   # decay spread evenly over the remaining sessions
                beyond = (S_ > K_short) if is_call else (S_ < K_short)
                mk = cfg.mark_iv_mult if beyond else 1.0
                if is_call:
                    val = bs_call(S_, K_short, T_rem, cfg.r, iv_s * mk) - bs_call(S_, K_long, T_rem, cfg.r, iv_l * mk)
                else:
                    val = bs_put(S_, K_short, T_rem, cfg.r, iv_s * mk) - bs_put(S_, K_long, T_rem, cfg.r, iv_l * mk)
                if tp and val <= 0.5 * credit_mid:
                    exit_date, exit_reason, exit_value = d_, "take_profit", val
                    break
                if sl and val >= 3.0 * credit_mid:
                    exit_date, exit_reason, exit_value = d_, "stop_loss", val
                    break
                touched_today = (highs.iloc[p] >= K_short) if is_call else (lows.iloc[p] <= K_short)
                if st and touched_today:
                    exit_date, exit_reason, exit_value = d_, "stop_touch", val
                    break
                if c1 and p == p1 - 1 and ((is_call and S_ >= 0.99 * K_short) or (not is_call and S_ <= 1.01 * K_short)):
                    exit_date, exit_reason, exit_value = d_, "close_1dte_near_strike", val
                    break
        if exit_value is None:
            pnl_share = credit - payoff
            comm = 2 * cfg.commission + (2 * cfg.commission if payoff > 0 else 0.0)  # closing ITM legs costs a round
        else:
            pnl_share = credit - (exit_value + slip)
            comm = 4 * cfg.commission
        pnl = pnl_share * 100.0 - comm
        max_risk = (width - credit) * 100.0 + 4 * cfg.commission + slip * 100.0   # includes a worst-case closing round
        contracts = int(cfg.risk_budget // max_risk) if max_risk > 0 else 0
        path_hi = highs.iloc[p0 + 1:p1 + 1]
        path_lo = lows.iloc[p0 + 1:p1 + 1]
        touched = bool((path_hi > K_short).any()) if is_call else bool((path_lo < K_short).any())
        rec.update({"exit_date": exit_date, "exit_reason": exit_reason, "pnl": pnl, "max_risk": max_risk,
                    "ret_on_risk": pnl / max_risk if max_risk > 0 else float("nan"), "win": pnl > 0,
                    "contracts_const_risk": contracts, "pnl_const_risk": pnl * contracts,
                    "finished_beyond_short": (ST > K_short) if is_call else (ST < K_short), "touched_short": touched})
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.attrs["config"] = asdict(cfg)
    return out


def summarize_trades(trades: pd.DataFrame, years_hint: float | None = None) -> dict:
    if trades is None or trades.empty:
        return {"trades": 0, "skipped": 0}
    t = trades[trades["skipped"].isna()].copy()
    n = int(len(t))
    out = {"trades": n, "skipped": int(trades["skipped"].notna().sum()),
           "skip_reasons": ", ".join(f"{k}: {v}" for k, v in trades["skipped"].dropna().value_counts().items()) or "none"}
    if n == 0:
        return out
    wins, losses = t[t["pnl"] > 0]["pnl"], t[t["pnl"] <= 0]["pnl"]
    cum = pd.concat([pd.Series([0.0]), t["pnl"].cumsum()], ignore_index=True)  # equity starts at zero
    span_days = (t["expiry_date"].max() - t["entry_date"].min()).days
    years = years_hint or max(span_days / 365.25, 1e-9)
    avg_risk = float(t["max_risk"].mean())
    streak, worst_streak = 0, 0
    for w in (t["pnl"] > 0).values:
        streak = 0 if w else streak + 1
        worst_streak = max(worst_streak, streak)
    worst5 = losses.nsmallest(5).sum() if len(losses) else 0.0
    out.update({
        "win_rate": float((t["pnl"] > 0).mean()),
        "total_pnl": float(t["pnl"].sum()),
        "avg_pnl": float(t["pnl"].mean()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() else float("inf"),
        "expectancy_pct_risk": float((t["pnl"] / t["max_risk"]).mean()),
        "expectancy_se_pct_risk": float((t["pnl"] / t["max_risk"]).std(ddof=1) / math.sqrt(n)) if n > 2 else float("nan"),
        "expectancy_pct_notional": float((t["pnl"] / (t["S0"] * 100.0)).mean()),
        "full_loss_rate": float((t["pnl"] <= -0.9 * t["max_risk"]).mean()),
        "years_to_2sigma": float((2 * (t["pnl"] / t["max_risk"]).std(ddof=1) / abs((t["pnl"] / t["max_risk"]).mean())) ** 2 / 52) if n > 2 and (t["pnl"] / t["max_risk"]).mean() != 0 else float("nan"),
        "return_on_max_risk": float(t["pnl"].sum() / t["max_risk"].sum()),
        "return_on_avg_risk_annual": float(t["pnl"].sum() / avg_risk / years) if avg_risk > 0 else float("nan"),
        "breakeven_win_rate": float(abs(losses.mean()) / (wins.mean() + abs(losses.mean()))) if len(wins) and len(losses) else float("nan"),
        "max_drawdown_$": float((cum - cum.cummax()).min()),
        "longest_losing_streak": int(worst_streak),
        "worst5_share_of_losses": float(worst5 / losses.sum()) if len(losses) and losses.sum() else float("nan"),
        "sharpe_weekly": float(t["pnl"].mean() / t["pnl"].std(ddof=1) * math.sqrt(52)) if n > 2 and t["pnl"].std(ddof=1) > 0 else float("nan"),
        "pnl_const_risk": float(t["pnl_const_risk"].sum()),
        "avg_contracts_const_risk": float(t["contracts_const_risk"].mean()),
        "const_risk_zero_weeks": int((t["contracts_const_risk"] == 0).sum()),
        "avg_credit": float(t["credit"].mean()),
        "avg_width": float(t["width"].mean()),
        "avg_credit_pct_width": float((t["credit"] / t["width"]).mean()),
        "avg_short_otm_pct": float(t["short_otm_pct"].mean()),
        "avg_model_delta": float(t["model_delta"].mean()),
        "finished_beyond_short_rate": float(t["finished_beyond_short"].mean()),
        "touched_short_rate": float(t["touched_short"].mean()),
        "avg_sigma_used": float(t["sigma"].mean()),
        "stock_buy_hold": float(np.prod(1.0 + t["stock_ret"]) - 1.0),
        "years": float(years),
        "first_entry": str(t["entry_date"].min().date()),
        "last_expiry": str(t["expiry_date"].max().date()),
    })
    return out


def group_summary(trades: pd.DataFrame, key: str) -> pd.DataFrame:
    t = trades[trades["skipped"].isna()].copy()
    if t.empty or key not in t.columns:
        return pd.DataFrame()
    rows = []
    for g, sub in t.groupby(key, dropna=False):
        if sub.empty:
            continue
        s = summarize_trades(sub, years_hint=len(sub) / 52.0)
        rows.append({key: g, "trades": s["trades"], "win_rate": s["win_rate"], "total_pnl": s["total_pnl"],
                     "expectancy_pct_risk": s["expectancy_pct_risk"], "full_loss_rate": s["full_loss_rate"],
                     "profit_factor": s["profit_factor"], "max_dd_$": s["max_drawdown_$"], "worst_wk_$": float(sub["pnl"].min()),
                     "beyond_short": s["finished_beyond_short_rate"], "model_delta": s["avg_model_delta"], "stock_ret": s["stock_buy_hold"]})
    return pd.DataFrame(rows).set_index(key)


def per_year(trades: pd.DataFrame) -> pd.DataFrame:
    t = trades[trades["skipped"].isna()].copy()
    if t.empty:
        return pd.DataFrame()
    t["year"] = t["entry_date"].dt.year
    return group_summary(t, "year")


def slice_period(trades: pd.DataFrame, months: int | None, asof=None) -> pd.DataFrame:
    """Trades entered within the last `months` before `asof` (default: the frame's last expiry). Pass the
    unfiltered frame's as-of when slicing a filtered subset so both cover the same calendar window."""
    if months is None or trades.empty or "expiry_date" not in trades.columns:
        return trades
    asof = asof if asof is not None else trades["expiry_date"].max()
    return trades[trades["entry_date"] > asof - pd.DateOffset(months=months)]


TRADE_COLS = ["entry_date", "expiry_date", "S0", "ST", "stock_ret", "prev_week_ret", "skipped", "rv", "sigma", "K_short", "K_long",
              "width", "short_otm_pct", "model_delta", "credit_mid", "credit", "exit_date", "exit_reason", "pnl", "max_risk",
              "ret_on_risk", "win", "contracts_const_risk", "pnl_const_risk", "finished_beyond_short", "touched_short"]


def completed(trades: pd.DataFrame) -> pd.DataFrame:
    """Rows that actually traded, with every expected column present even when nothing traded."""
    if trades is None or trades.empty:
        return pd.DataFrame(columns=TRADE_COLS)
    t = trades[trades["skipped"].isna()].copy() if "skipped" in trades.columns else trades.copy()
    for c in TRADE_COLS:
        if c not in t.columns:
            t[c] = np.nan
    return t


def breakeven_mult(daily, wk, cfg: SpreadConfig, earnings, regimes, months: int | None, lo=0.5, hi=4.0, iters=14) -> float:
    """Smallest IV multiplier m at which the program breaks even over the period (bisection; re-runs the backtest)."""
    def total(m):
        tr = slice_period(backtest_spreads(daily, wk, replace(cfg, iv_mult=m), earnings, regimes), months)
        tr = tr[tr["skipped"].isna()]
        return float(tr["pnl"].sum()) if len(tr) else float("nan")
    f_lo, f_hi = total(lo), total(hi)
    if f_lo != f_lo or f_hi != f_hi:
        return float("nan")
    if f_lo > 0:
        return lo  # profitable even at 0.5x realized vol: the edge is not the vol premium
    if f_hi < 0:
        return float("inf")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if total(mid) > 0:
            hi = mid
        else:
            lo = mid
    return round(0.5 * (lo + hi), 3)


def leaky_short_benchmark(daily: pd.DataFrame, trades: pd.DataFrame) -> float:
    """Short 100 shares each week at the entry close, cover at the first close at or above the spread's long
    strike or at Friday. 'Leaky cap' because gaps can jump through it."""
    t = completed(trades)
    closes = daily["close"]
    total = 0.0
    for _, row in t.iterrows():
        path = closes[(closes.index > row["entry_date"]) & (closes.index <= row["expiry_date"])]
        cap = row["K_long"]
        hit = path[path >= cap]
        exit_px = float(hit.iloc[0]) if len(hit) else float(row["ST"])
        total += -(exit_px - row["S0"]) * 100.0
    return total


# ======================================================================================================
# 6. Today's option chain (optional): m_today, skew, strike grid, with our own IV inversion
# ======================================================================================================

def chain_calibration(ticker: str, spot: float, rv_now: float, rate: float = 0.04) -> dict | None:
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        exps = tk.options
        if not exps:
            return None
        today = datetime.now(NY).date()
        cands = [(e, (datetime.strptime(e, "%Y-%m-%d").date() - today).days, datetime.strptime(e, "%Y-%m-%d").weekday()) for e in exps]
        window = [c for c in cands if 3 <= c[1] <= 12] or [c for c in cands if c[1] > 0]
        fridays = [c for c in window if c[2] == 4]   # the backtest trades Friday expiries; SPY/QQQ also list Mon/Wed
        exp, dte, _ = sorted(fridays or window, key=lambda c: abs(c[1] - 7))[0]
        ch = tk.option_chain(exp)
        calls_all = ch.calls.copy()
        try:  # price the chain against the chain's own underlying quote, not a cached or historical close
            live = float(ch.underlying.get("regularMarketPrice") or 0)
            if live > 0:
                spot = live
        except Exception:  # noqa: BLE001
            pass
        T = max(dte, 1) / 365.0
        calls_all["mid"] = (calls_all["bid"] + calls_all["ask"]) / 2.0
        calls = calls_all[(calls_all["mid"] > 0) & (calls_all["ask"] > 0)].copy()
        if calls.empty:
            print(f"  (option chain for {exp} has no live bid/ask yet)")
            return None
        calls["iv_own"] = [implied_vol(m, spot, k, T, rate, True) for m, k in zip(calls["mid"], calls["strike"])]
        calls = calls.dropna(subset=["iv_own"])
        if calls.empty:
            return None
        listed = calls_all[(calls_all["strike"] > spot * 0.8) & (calls_all["strike"] < spot * 1.2)].sort_values("strike")
        steps = np.diff(listed["strike"].values)
        step = float(pd.Series(steps).mode().iloc[0]) if len(steps) else float("nan")
        atm = calls.iloc[(calls["strike"] - spot).abs().argsort()[:2]]
        iv_atm = float(atm["iv_own"].mean())
        if not iv_atm == iv_atm:
            return None
        calls["delta"] = [bs_call_delta(spot, k, T, rate, v) for k, v in zip(calls["strike"], calls["iv_own"])]
        c25 = calls.iloc[(calls["delta"] - 0.25).abs().argsort()[:1]]
        c10 = calls.iloc[(calls["delta"] - 0.10).abs().argsort()[:1]]
        iv25 = float(c25["iv_own"].iloc[0]) if len(c25) else float("nan")
        iv10 = float(c10["iv_own"].iloc[0]) if len(c10) else float("nan")
        skew = (iv25 / iv_atm - 1.0) / 0.25 if iv_atm and iv25 == iv25 else float("nan")
        short_pct, short_ratio = None, None
        try:
            info = tk.info or {}
            short_pct, short_ratio = info.get("shortPercentOfFloat"), info.get("shortRatio")
        except Exception:  # noqa: BLE001
            pass
        return {"expiry": exp, "dte": dte, "spot_used": spot, "iv_atm": iv_atm, "iv_25d_call": iv25, "iv_10d_call": iv10,
                "short_pct_float": short_pct, "short_ratio_days": short_ratio,
                "skew_s": float(min(max(skew, -0.5), 1.0)) if skew == skew else None,
                "strike_step": step, "rv_now": rv_now, "m_today": iv_atm / rv_now if rv_now else float("nan"),
                "yahoo_iv_atm": float(atm["impliedVolatility"].mean())}
    except Exception as exc:  # noqa: BLE001
        print(f"  (could not read option chain: {exc})")
        return None


SEC_UA = os.environ.get("AVGWEEK_CONTACT", "avgweek research tool (set AVGWEEK_CONTACT=name email for SEC fair-access policy)")


def harvest_edgar_8k(block: dict) -> set:
    """Earnings-release dates from an EDGAR submissions block: 8-K filings whose items include 2.02
    (Results of Operations). reportDate is the event date; filingDate is the fallback."""
    out = set()
    forms, items = block.get("form", []), block.get("items", [])
    rdates, fdates = block.get("reportDate", []), block.get("filingDate", [])
    for i, f in enumerate(forms):
        if f in ("8-K", "8-K/A") and "2.02" in str(items[i] if i < len(items) else ""):
            d = (rdates[i] if i < len(rdates) else "") or (fdates[i] if i < len(fdates) else "")
            if d:
                out.add(pd.Timestamp(d).normalize())
    return out


def earnings_from_edgar(ticker: str, years: int = 8) -> set:
    """Free and authoritative: SEC EDGAR 8-K item 2.02 filings. Needs network; returns an empty set on failure."""
    import requests
    h = {"User-Agent": SEC_UA, "Accept-Encoding": "gzip, deflate"}
    tick = requests.get("https://www.sec.gov/files/company_tickers.json", headers=h, timeout=30).json()
    cik = next((v["cik_str"] for v in tick.values() if str(v.get("ticker", "")).upper() == ticker.upper()), None)
    if cik is None:
        return set()
    sub = requests.get(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json", headers=h, timeout=30).json()
    dates = harvest_edgar_8k(sub.get("filings", {}).get("recent", {}))
    cutoff = pd.Timestamp.today() - pd.DateOffset(years=years)
    for extra in sub.get("filings", {}).get("files", []):
        if pd.Timestamp(extra.get("filingTo", "1900-01-01")) < cutoff:
            continue
        try:
            dates |= harvest_edgar_8k(requests.get(f"https://data.sec.gov/submissions/{extra['name']}", headers=h, timeout=30).json())
        except Exception:  # noqa: BLE001
            pass
    return dates


def parse_nasdaq_short_interest(payload: dict) -> pd.DataFrame:
    """Nasdaq's per-symbol short-interest table (bi-monthly settlement dates, about one year). Publication lags the
    settlement date by roughly nine business days, so `available_date` is what the backtest may know at entry."""
    rows = (((payload or {}).get("data") or {}).get("shortInterestTable") or {}).get("rows") or []
    def num(x):
        try:
            return float(str(x).replace(",", ""))
        except (TypeError, ValueError):
            return float("nan")
    out = pd.DataFrame({
        "settlement_date": [pd.Timestamp(r.get("settlementDate")) for r in rows],
        "short_interest": [num(r.get("interest")) for r in rows],
        "avg_daily_volume": [num(r.get("avgDailyShareVolume")) for r in rows],
        "days_to_cover": [num(r.get("daysToCover")) for r in rows],
    }).dropna(subset=["settlement_date"]).sort_values("settlement_date")
    out["available_date"] = out["settlement_date"] + pd.offsets.BDay(10)
    return out.reset_index(drop=True)


def short_interest_history(ticker: str) -> pd.DataFrame:
    import requests
    url = f"https://api.nasdaq.com/api/quote/{ticker.upper()}/short-interest?assetClass=stocks"
    h = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36",
         "Accept": "application/json, text/plain, */*", "Accept-Language": "en-US,en;q=0.9", "Origin": "https://www.nasdaq.com", "Referer": "https://www.nasdaq.com/"}
    return parse_nasdaq_short_interest(requests.get(url, headers=h, timeout=20).json())


def load_earnings(ticker: str, earnings_file: str | None, fetch: bool) -> set:
    dates: set = set()
    if earnings_file:
        for raw in Path(earnings_file).read_text().split():
            raw = raw.strip().strip(",")
            if raw:
                dates.add(pd.Timestamp(raw).normalize())
    if fetch:
        try:
            edgar = earnings_from_edgar(ticker)
            if edgar:
                print(f"  earnings dates from SEC EDGAR 8-K item 2.02: {len(edgar)}")
            dates |= edgar
        except Exception as exc:  # noqa: BLE001
            print(f"  (could not fetch EDGAR filings: {exc})")
        try:
            import yfinance as yf
            ed = yf.Ticker(ticker).get_earnings_dates(limit=60)
            if ed is not None and len(ed):
                for ts in ed.index:
                    ts = pd.Timestamp(ts)
                    dates.add((ts.tz_localize(None) if ts.tzinfo else ts).normalize())
        except Exception as exc:  # noqa: BLE001
            print(f"  (could not fetch earnings dates: {exc})")
    return dates


# ======================================================================================================
# 7. Formatting and report
# ======================================================================================================

PCT_COLS = {"cum_return", "mean_wk", "mean_log_wk", "vol_drag_wk", "median_wk", "median_ci_lo", "median_ci_hi", "win_rate",
            "win_ci_lo", "win_ci_hi", "avg_up_wk", "avg_down_wk", "sd_wk", "ann_vol", "cum_ex_top5", "pct_wk_gt_10",
            "pct_wk_lt_m10", "best_wk", "worst_wk", "max_dd", "next_wk_mean", "next_wk_median", "next_wk_win_rate",
            "next_wk_p_gt_10", "next_wk_worst", "daily_mean", "daily_median", "daily_win_rate", "overnight_mean",
            "overnight_win_rate", "intraday_mean", "intraday_win_rate", "overnight_total", "intraday_total",
            "overnight_var_share", "expectancy_pct_risk", "beyond_short", "stock_ret", "model_delta", "realized_p",
            "ci_lo", "ci_hi", "lognormal_p", "avg_overshoot", "payout_w5", "payout_w10", "bs_credit_w5", "bs_credit_w10",
            "full_width_share_w5", "full_width_share_w10", "win_null", "win_null_lo", "win_null_hi", "expectancy_se_pct_risk", "avg_sigma_used",
            "cc", "overnight", "intraday", "pct_from_high52", "return_on_max_risk", "full_loss_rate", "implied_p_today",
            "p_up_overlap", "p_up_nonoverlap", "p_up_iid_boot", "p_up_block_ci_lo", "p_up_block_ci_hi", "p_up_normal",
            "q05", "q25", "q50", "q75", "q95", "mean_h", "p_gt_10", "p_gt_20", "p_gt_30", "p_lt_m20",
            "call_win", "put_win", "call_exp_pct_risk", "put_exp_pct_risk", "call_minus_put_pct_risk"}


def _fmt_df(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in d.columns:
        cs = str(c)
        if cs in PCT_COLS:
            d[c] = d[c].map(lambda v: f"{v*100:+.1f}%" if isinstance(v, (float, np.floating)) and v == v else ("" if isinstance(v, float) else v))
        elif d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: f"{v:,.2f}" if v == v and abs(v) != float("inf") else ("inf" if v == float("inf") else ""))
        elif d[c].dtype.kind == "b":
            d[c] = d[c].map(lambda v: "yes" if v else "no")
    if d.index.name == "k_otm":
        d.index = [f"+{k*100:g}%" for k in d.index]
    return d


def fmt_table(df: pd.DataFrame, cols: list | None = None) -> str:
    if df is None or df.empty:
        return "(no data)"
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    return _fmt_df(df).to_string()


def print_summary_dict(title: str, s: dict) -> None:
    print(f"\n{title}")
    money = {"total_pnl", "avg_pnl", "avg_win", "avg_loss", "max_drawdown_$", "pnl_const_risk"}
    per_share = {"avg_credit", "avg_width"}
    pct = {"win_rate", "expectancy_pct_risk", "expectancy_se_pct_risk", "expectancy_pct_notional", "full_loss_rate", "return_on_max_risk", "return_on_avg_risk_annual", "breakeven_win_rate",
           "avg_short_otm_pct", "finished_beyond_short_rate", "touched_short_rate", "stock_buy_hold", "avg_sigma_used",
           "avg_model_delta", "worst5_share_of_losses", "avg_credit_pct_width"}
    for k, v in s.items():
        if isinstance(v, float):
            if k in money:
                v = f"${v:,.0f}"
            elif k in per_share:
                v = f"${v:,.2f} /share"
            elif k in pct:
                v = f"{v*100:.1f}%"
            else:
                v = f"{v:.2f}"
        print(f"  {k:<28} {v}")


def svg_line(series, width=760, height=240, title=""):
    allv = [v for _, vals in series for v in vals if v == v]
    if not allv:
        return ""
    lo, hi = min(allv), max(allv)
    if hi == lo:
        hi = lo + 1
    pad = 36
    n = max(len(vals) for _, vals in series)
    def x(i): return pad + (width - 2 * pad) * i / max(n - 1, 1)
    def y(v): return height - pad - (height - 2 * pad) * (v - lo) / (hi - lo)
    colours = ["#4cc9f0", "#f72585", "#ffd166", "#06d6a0"]
    paths = []
    for k, (name, vals) in enumerate(series):
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vals) if v == v)
        paths.append(f'<polyline fill="none" stroke="{colours[k % 4]}" stroke-width="1.6" points="{pts}"/>'
                     f'<text x="{pad + 8 + k * 300}" y="16" fill="{colours[k % 4]}" font-size="11">{name}</text>')
    zero = f'<line x1="{pad}" x2="{width - pad}" y1="{y(0):.1f}" y2="{y(0):.1f}" stroke="#666" stroke-dasharray="3,3"/>' if lo < 0 < hi else ""
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px;background:#12141a;border-radius:8px">'
            f'<text x="{width/2}" y="{height-8}" fill="#aaa" font-size="11" text-anchor="middle">{title}</text>'
            f'<text x="4" y="{y(hi)+4:.0f}" fill="#aaa" font-size="10">{hi:,.0f}</text>'
            f'<text x="4" y="{y(lo):.0f}" fill="#aaa" font-size="10">{lo:,.0f}</text>{zero}{"".join(paths)}</svg>')


def svg_hist(values, bins=30, width=760, height=220, title="", marks=()):
    vals = [v for v in values if v == v]
    if not vals:
        return ""
    counts, edges = np.histogram(vals, bins=bins)
    pad = 30
    bw = (width - 2 * pad) / bins
    mx = counts.max() or 1
    bars = []
    for i, c in enumerate(counts):
        h = (height - 2 * pad) * c / mx
        col = "#06d6a0" if edges[i] >= 0 else "#f72585"
        bars.append(f'<rect x="{pad + i*bw:.1f}" y="{height - pad - h:.1f}" width="{bw-1:.1f}" height="{h:.1f}" fill="{col}" opacity="0.85"/>')
    def xpos(v): return pad + (width - 2 * pad) * (v - edges[0]) / (edges[-1] - edges[0])
    mk = "".join(f'<line x1="{xpos(v):.1f}" x2="{xpos(v):.1f}" y1="{pad}" y2="{height-pad}" stroke="#ffd166" stroke-dasharray="4,3"/>'
                 f'<text x="{xpos(v)+3:.1f}" y="{pad+10}" fill="#ffd166" font-size="10">{lab}</text>' for v, lab in marks if edges[0] <= v <= edges[-1])
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px;background:#12141a;border-radius:8px">'
            f'<text x="{width/2}" y="{height-6}" fill="#aaa" font-size="11" text-anchor="middle">{title}</text>'
            f'<text x="{pad}" y="{height-pad+12}" fill="#aaa" font-size="10">{edges[0]*100:.0f}%</text>'
            f'<text x="{width-pad}" y="{height-pad+12}" fill="#aaa" font-size="10" text-anchor="end">{edges[-1]*100:.0f}%</text>'
            f'{"".join(bars)}{mk}</svg>')


def heatmap_html(sweep: pd.DataFrame, value: str = "expectancy_pct_risk", period: str = "1y", mgmt: str = "hold") -> str:
    if sweep is None or sweep.empty:
        return ""
    s = sweep[(sweep["period"] == period) & (sweep["mgmt"] == mgmt) & (sweep["side"] == "call")]
    if s.empty:
        return ""
    piv = s.pivot_table(index="delta", columns="iv_mult", values=value, aggfunc="first").sort_index(ascending=False)
    be = s.drop_duplicates("delta").set_index("delta")["m_be"] if "m_be" in s.columns else None
    cells = []
    for d, row in piv.iterrows():
        tds = []
        for m, v in row.items():
            if v != v:
                tds.append("<td></td>")
                continue
            a = min(abs(v) / 0.10, 1.0)
            col = f"rgba(6,214,160,{0.15 + 0.6*a:.2f})" if v > 0 else f"rgba(247,37,133,{0.15 + 0.6*a:.2f})"
            tds.append(f'<td style="background:{col}">{v*100:+.1f}%</td>')
        be_txt = ""
        if be is not None and d in be.index and be[d] == be[d]:
            be_txt = f"{be[d]:.2f}" if be[d] != float("inf") else "&gt;4"
        cells.append(f"<tr><td>{d:.2f}</td>{''.join(tds)}<td>{be_txt}</td></tr>")
    head = "".join(f"<th>m={m:g}</th>" for m in piv.columns)
    return (f'<h3>Last {period}, {mgmt}</h3><table><tr><th>short delta</th>{head}<th>break-even m</th></tr>{"".join(cells)}</table>'
            f'<p class="note">Expectancy per week as % of max risk, call spreads. Green = seller wins. '
            f'The right column is the IV/RV multiplier at which the program breaks even; compare it with m_today from the chain.</p>')


def html_report(ctx: dict) -> str:
    css = ("body{background:#0b0c10;color:#e6e6e6;font:14px/1.45 -apple-system,Segoe UI,Roboto,sans-serif;margin:24px;max-width:1500px}"
           "h1,h2,h3{font-weight:600} h2{margin-top:36px;border-bottom:1px solid #2a2d36;padding-bottom:6px}"
           "table{border-collapse:collapse;font-size:12.5px;margin:10px 0} th,td{padding:4px 8px;border-bottom:1px solid #22252e;text-align:right;white-space:nowrap}"
           "th{color:#9aa3b2;font-weight:600} td:first-child,th:first-child{text-align:left} .note{color:#9aa3b2;font-size:12.5px}"
           ".kpi{display:inline-block;margin:6px 18px 6px 0} .kpi b{display:block;font-size:20px} .verdict{background:#13161d;border-left:3px solid #c9a84c;padding:10px 14px;margin:12px 0;font-size:13.5px}"
           ".scroll{overflow-x:auto}")
    def tbl(df):
        if df is None or df.empty:
            return "<p class='note'>(no data)</p>"
        return '<div class="scroll">' + _fmt_df(df).to_html(border=0, escape=False) + "</div>"
    t = completed(ctx["base_trades"])
    series = [("Call spreads: cumulative P&L ($, 1 contract)", t["pnl"].cumsum().tolist())] if len(t) else []
    put_t = completed(ctx.get("put_trades"))
    if len(put_t):
        series.append(("Put spreads (mirror)", put_t["pnl"].cumsum().tolist()))
    bs = ctx["base_summary"]
    mbe1 = ctx["m_be"].get("1y", float("nan"))
    kpis = [("Weeks (1y)", str(bs.get("trades", 0))), ("Win rate", f"{bs.get('win_rate', float('nan'))*100:.0f}%"),
            ("P&L, 1 contract", f"${bs.get('total_pnl', 0):,.0f}"),
            (f"P&L, const. risk ${ctx['base_cfg'].risk_budget:,.0f}", f"${bs.get('pnl_const_risk', 0):,.0f}" + (f" ({bs['const_risk_zero_weeks']} wks over budget)" if bs.get('const_risk_zero_weeks') else "")),
            ("Expectancy / max risk", f"{bs.get('expectancy_pct_risk', float('nan'))*100:.1f}% ± {bs.get('expectancy_se_pct_risk', float('nan'))*100:.1f}%"),
            ("Full-width losses", f"{bs.get('full_loss_rate', float('nan'))*100:.0f}% of weeks"),
            ("Break-even m (1y)", _num(mbe1)),
            ("Max drawdown", f"${bs.get('max_drawdown_$', 0):,.0f}"), ("Stock buy & hold", f"{bs.get('stock_buy_hold', float('nan'))*100:+.0f}%")]
    kpi_html = "".join(f'<div class="kpi"><span class="note">{k}</span><b>{v}</b></div>' for k, v in kpis)
    wk = ctx["wk"]
    marks = [(float(t["short_otm_pct"].mean()), "avg short strike")] if len(t) and t["short_otm_pct"].notna().any() else []
    weekly_hist = svg_hist(wk["ret"].dropna().tolist(), title="Distribution of weekly returns (all data); dashed = average short-strike distance", marks=marks)
    chain = ctx.get("chain")
    chain_txt = (f"Today's chain ({chain['expiry']}, {chain['dte']}d): ATM IV {chain['iv_atm']:.0%}, 25d call IV {chain['iv_25d_call']:.0%}, "
                 f"skew s = {chain['skew_s']}, strike step {chain['strike_step']}, RV now {chain['rv_now']:.0%}, <b>m_today = {chain['m_today']:.2f}</b>.") if chain else \
        ctx.get("chain_note", "No chain calibration.")
    dep = ctx["dep"]
    cfg = ctx["base_cfg"]
    g = lambda k: dep.get(k, float("nan"))  # noqa: E731
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{ctx['ticker']} average week</title><style>{css}</style></head><body>
<h1>{ctx['ticker']}: the average week, and selling weekly call credit spreads</h1>
<p class="note">Generated {datetime.now():%Y-%m-%d %H:%M} by avgweek.py v{__version__}. Data: {ctx['source']}. Last weekly close used: {wk['date'].iloc[-1]:%Y-%m-%d} at {wk['close'].iloc[-1]:,.2f}.
Weeks run last-trading-day to last-trading-day. Option prices: Black-Scholes, IV = m x realized-vol estimate ({cfg.rv_mode}, floor {cfg.iv_floor:.0%}) with call skew s={cfg.skew}; hold-to-expiry results use the model ONLY for the entry credit.</p>
<div class="verdict">{"<br>".join(ctx['verdict'])}</div>
<p class="note">{chain_txt}</p>

<h2>1. The average week, by lookback</h2>
<p class="note">Nested windows (5y contains 1y). Rows under {MIN_WEEKS_FOR_DIRECTION} weeks cannot support a direction call; read their vol and worst-week columns only. mean_log_wk is what buy-and-hold compounds at; mean_wk minus mean_log_wk is the volatility drag. The skew null is a driftless lognormal with the same vol: a 100%-vol random walk already has median &lt; 0, win rate &lt; 50% and positive skew with zero edge.</p>
{tbl(ctx['stats'][ctx['dir_cols']])}
<h3>Shape and tails</h3>{tbl(ctx['stats'][ctx['shape_cols']])}
<h3>Calendar years (non-nested)</h3>{tbl(ctx['yearly'][[c for c in ctx['dir_cols'] + ['ann_vol', 'cum_ex_top5', 'n_wk_gt_15', 'max_dd'] if c in ctx['yearly'].columns]])}
<p class="note">Rolling 26-week median: now {_pct(ctx['rolling'].get('median_now'))}/wk, range [{_pct(ctx['rolling'].get('median_min'))}, {_pct(ctx['rolling'].get('median_max'))}], sign changes {ctx['rolling'].get('n_sign_changes', 0)}, last on {ctx['rolling'].get('last_sign_change', 'n/a')}. {ctx['regime_flag']}</p>
{weekly_hist}

<h2>2. The {ctx['h']}-week horizon</h2>{tbl(ctx['horizon_df'])}
<p class="note">Overlapping windows share most of their content (n_eff = n/{ctx['h']}). The block-bootstrap CI is the honest interval; expect it to span roughly 25 to 65% for a single high-vol name. q05..q95 are quantiles of the {ctx['h']}-week return from an iid bootstrap of weekly log returns.</p>

<h2>3. Exceedance vs implied: the model-free core</h2>
<p class="note">For each out-of-the-money distance k: how often the week closed above +k% (Wilson CI), what a driftless lognormal at the window's realized vol predicts, the average overshoot, the average weekly payout of a spread sold k% OTM (per $1 of spot) for 5% and 10% widths, and the credit Black-Scholes would have charged at m=1. credit_over_payout &gt; 1 means the seller was paid more than it cost at m=1. A strike that only works at m &gt; 1.3 relies on the variance premium, not on the drift.</p>
<h3>Last 1 year</h3>{tbl(ctx['exceed'].get('1y'))}
<h3>All data</h3>{tbl(ctx['exceed'].get('all'))}

<h2>4. What happens the week after... (last {ctx['cond'].attrs.get('window_years', 2):g} years)</h2>{tbl(ctx['cond'])}
<p class="note">Compare each row to 'all weeks', not to zero: conditioning on a +15% week selects lucky weeks, and the next week regresses regardless of any true signal. Dependence (last 2y): ACF(1..4) = {_num(g('acf1'))}, {_num(g('acf2'))}, {_num(g('acf3'))}, {_num(g('acf4'))} (SE {_num(g('acf_se'))}; need |rho| &gt; {_num(2*g('acf_se'))} to matter); Ljung-Box Q(4) p = {_num(g('ljung_box_p'))}; variance ratios VR2 {_num(g('vr2'))}, VR3 {_num(g('vr3'))}, VR4 {_num(g('vr4'))} (&lt;1 mean reversion, &gt;1 momentum; iid SE for VR3 {_num(g('vr3_se'))}).</p>

<h2>5. Daily decomposition: overnight gap vs intraday drift, and gap events</h2>{tbl(ctx['decomp'])}
<h3>Ten largest sessions</h3>{tbl(ctx['top_days'])}

<h2>6. Weekly call credit spread, baseline {cfg.label()}</h2>
<div>{kpi_html}</div>
{svg_line(series, title="Cumulative P&L by week, all data (1 contract)")}
<h3>Call side vs put side (mirror test)</h3>{tbl(ctx['mirror'])}
<p class="note">If put spreads do as well as call spreads under the same m, the result is the assumed vol premium, not the directional thesis. Only a call-side advantage supports "grind down".</p>
<h3>Same baseline priced off three realized-vol estimators</h3>{tbl(ctx['pricing_modes'])}
<p class="note">Trailing realized vol is low after calm weeks, which pulls strikes in just before the next rip; a long-run estimator keeps them out. A filter or a configuration whose edge survives only one pricing mode is a strike-placement artifact, not a thesis.</p>
<h3>By calendar year</h3>{tbl(ctx['base_year'])}
<h3>By trend state at entry (close vs 50-day average and its 20-day slope)</h3>{tbl(ctx['base_trend'])}
<h3>By realized-vol percentile at entry (trailing year)</h3>{tbl(ctx['base_rv'])}
<h3>Conditional on last week</h3>{tbl(ctx['base_cond'])}
{('<h3>Earnings weeks vs the rest</h3>' + tbl(ctx['base_earn'])) if not ctx['base_earn'].empty else ''}
{('<h3>Regime filter vs unfiltered</h3>' + tbl(ctx['regime_filtered'])) if not ctx['regime_filtered'].empty else ''}
<p class="note">Benchmarks over the same weeks, all data: buy &amp; hold 100 shares {ctx['bench']['buy_hold_100']:+,.0f} $; short 100 shares each week, covered at the first close at or above the spread's long strike ("leaky cap": gaps jump through it) {ctx['bench']['leaky_short']:+,.0f} $; call spreads {ctx['base_all'].get('total_pnl', 0):+,.0f} $ per contract. Management variants (tp50, sl2x, close1dte) are approximate: daily-close triggers, flat intra-week IV.</p>

<h2>7. Sweep: short delta x IV multiplier</h2>{heatmap_html(ctx['sweep'], period='1y')}{heatmap_html(ctx['sweep'], period='all')}
<p class="note">{ctx['n_variants']} configurations were evaluated for this ticker. At zero true edge the best of 16 variants on one year reads about +6% of risk per week, and a 16-delta program shows a &gt;90% win rate 62% of the time; the expectancy standard error is in summary.json (expectancy_se_pct_risk). Rank by expectancy, never by win rate.</p>
<h3>Full sweep (last 1y)</h3>{tbl(ctx['sweep'][ctx['sweep']['period'] == '1y'].drop(columns=['period']).set_index('config')) if not ctx['sweep'].empty else ''}

<h2>What this can and cannot claim</h2>
<p class="note">CAN: how often and by how much weekly closes exceeded each strike distance (exact history); the P&L of a hold-to-expiry program as a function of the assumed IV/RV multiplier and the break-even multiplier per period; whether call spreads beat put spreads under the same assumptions; the stock's average-week statistics with honest intervals.
CANNOT: the actual historical dollar result (entry credits depend on IV, skew and fills that are not in free data); that "the average week is down" is a property of the stock rather than of the window (one year of weeks cannot establish it); realistic stop/target performance without intraday prices; anything about the next regime. The identical program flips sign between years with no change in the stock's character, so the edge, if any, is a regime call.</p>
</body></html>"""


# ======================================================================================================
# 7b. Brief mode: the six lines that decide the thesis, nothing else
# ======================================================================================================

def _sgn(v, d=0):
    return "n/a" if v is None or v != v else f"{v*100:+.{d}f}%"


def _pts(v, d=1):
    return "n/a" if v is None or v != v else f"{v*100:.{d}f}"


def brief_fields(ticker, wk, stats, base_summary, base_all, m_be, mirror, cond, yr, sweep, chain, chain_note, m_today_summ, skipped_note,
                 decomp=None, si_now=None):
    s1 = stats.loc["1y"] if "1y" in stats.index else stats.iloc[-1]
    dec1 = decomp.loc["1y"] if (decomp is not None and "1y" in decomp.index) else None
    decA = decomp.loc["all"] if (decomp is not None and "all" in decomp.index) else None
    def split_label(r):
        if r is None:
            return "n/a"
        on, ind = r["overnight_total"], r["intraday_total"]
        if on > 0 and ind < 0:
            return "GAPS UP, BLEEDS INTRADAY"
        if on < 0 and ind > 0:
            return "GAINS INTRADAY, GAPS DOWN"
        return "BOTH UP" if on > 0 else "BOTH DOWN"
    win, lo, hi, null = s1["win_rate"], s1["win_ci_lo"], s1["win_ci_hi"], s1["win_null"]
    desc = "YES" if (s1["median_wk"] < 0 and win < 0.5) else ("PARTLY" if (s1["median_wk"] < 0 or win < 0.5) else "NO")
    noise = "inside" if (lo <= null <= hi) else ("below" if null > hi else "above")
    exp1, se1 = base_summary.get("expectancy_pct_risk", float("nan")), base_summary.get("expectancy_se_pct_risk", float("nan"))
    expA, seA = base_all.get("expectancy_pct_risk", float("nan")), base_all.get("expectancy_se_pct_risk", float("nan"))
    mbe1, mbeA = m_be.get("1y", float("nan")), m_be.get("all", float("nan"))
    m_today = chain["m_today"] if chain and chain.get("m_today") == chain.get("m_today") else None
    exp_mt1 = m_today_summ.get("1y", {}).get("expectancy_pct_risk", float("nan")) if m_today_summ else float("nan")
    exp_mtA = m_today_summ.get("all", {}).get("expectancy_pct_risk", float("nan")) if m_today_summ else float("nan")
    if expA == expA and seA == seA and expA > 2 * seA and exp1 == exp1 and exp1 > -se1:
        trade = "YES at fair pricing"
    elif m_today and exp_mt1 == exp_mt1 and exp_mtA == exp_mtA and exp_mt1 > 0 and exp_mtA > 0:
        trade = f"YES on today's premium (IV/RV {m_today:.2f})"
    elif m_today:
        trade = f"NO (IV/RV today {m_today:.2f}, needs {_num(max(mbe1, mbeA))})"
    else:
        trade = "NO at fair pricing"
    d1 = mirror.loc["1y", "call_minus_put_pct_risk"] if (not mirror.empty and "1y" in mirror.index) else float("nan")
    dA = mirror.loc["all", "call_minus_put_pct_risk"] if (not mirror.empty and "all" in mirror.index) else float("nan")
    if d1 == d1 and dA == dA and d1 > 0.02 and dA > 0.02:
        direction = "DIRECTION (calls beat puts)"
    elif d1 == d1 and dA == dA and d1 < -0.02 and dA < -0.02:
        direction = "PUT SIDE BETTER"
    else:
        direction = "PREMIUM, not direction"
    rip_row = cond.loc["after week > +15%"] if "after week > +15%" in cond.index else None
    all_row = cond.loc["all weeks"] if "all weeks" in cond.index else None
    if rip_row is not None and all_row is not None and rip_row["n"] >= 10:
        gap = rip_row["next_wk_win_rate"] - all_row["next_wk_win_rate"]
        after_rip = "MOMENTUM" if gap > 0.10 else ("MEAN REVERSION" if gap < -0.10 else "NO SIGNAL")
    else:
        after_rip, gap = "TOO FEW RIPS TO SAY", float("nan")
    years_all = [(int(y), float(r["expectancy_pct_risk"]), float(r["stock_ret"])) for y, r in yr.iterrows()] if not yr.empty else []
    years = years_all[-8:]
    years_cut = len(years_all) > 8
    sw = sweep[(sweep["period"] == "1y") & (sweep["mgmt"] == "hold") & (sweep["iv_mult"] == 1.0)] if not sweep.empty else pd.DataFrame()
    deltas = sorted(((float(r["delta"]), float(r["expectancy_pct_risk"])) for _, r in sw.iterrows()), reverse=True) if not sw.empty else []
    return dict(ticker=ticker, close=float(wk["close"].iloc[-1]), asof=str(wk["date"].iloc[-1].date()), ret_1y=s1["cum_return"], vol=s1["ann_vol"],
                weeks=int(s1["weeks"]), median=s1["median_wk"], mean=s1["mean_wk"], win=win, win_lo=lo, win_hi=hi, win_null=null, noise=noise,
                split_1y=split_label(dec1), split_all=split_label(decA),
                on_1y=dec1["overnight_total"] if dec1 is not None else float("nan"), in_1y=dec1["intraday_total"] if dec1 is not None else float("nan"),
                on_all=decA["overnight_total"] if decA is not None else float("nan"), in_all=decA["intraday_total"] if decA is not None else float("nan"),
                gap_share=dec1["overnight_var_share"] if dec1 is not None else float("nan"), si_now=si_now,
                cum_ex_top5=s1["cum_ex_top5"], desc=desc, exp1=exp1, se1=se1, expA=expA, seA=seA, mbe1=mbe1, mbeA=mbeA,
                m_today=m_today, exp_mt1=exp_mt1, exp_mtA=exp_mtA, trade=trade, chain_note=chain_note, d1=d1, dA=dA, direction=direction,
                rip_n=int(rip_row["n"]) if rip_row is not None else 0, rip_win=rip_row["next_wk_win_rate"] if rip_row is not None else float("nan"),
                rip_med=rip_row["next_wk_median"] if rip_row is not None else float("nan"), all_win=all_row["next_wk_win_rate"] if all_row is not None else float("nan"),
                after_rip=after_rip, years=years, years_cut=years_cut, deltas=deltas, full_loss=base_all.get("full_loss_rate", float("nan")),
                max_dd=base_all.get("max_drawdown_$", float("nan")), first=base_all.get("first_entry", ""), skipped_note=skipped_note)


def brief_text(b: dict) -> str:
    L = []
    si = b.get("si_now")
    si_txt = f"   short interest {si['pct_float']*100:.0f}% of float, {si['days_to_cover']:.1f} days to cover" if si and si.get("pct_float") else (f"   short interest {si['days_to_cover']:.1f} days to cover" if si else "")
    L.append(f"{b['ticker']}  ${b['close']:,.2f}   1y {_sgn(b['ret_1y'])}   vol {b['vol']*100:.0f}%{si_txt}   ({b['weeks']} weeks; {b['skipped_note']})")
    L.append(f"  Grinds down, rips up?      {b['desc']:<28} average week {_sgn(b['mean'], 1)} (mean), typical week {_sgn(b['median'], 1)} (median), up {b['win']*100:.0f}% of weeks; year {_sgn(b['ret_1y'])} but {_sgn(b['cum_ex_top5'])} without its best 5 weeks")
    L.append(f"                             {'':<28} (a {b['vol']*100:.0f}%-vol random walk gives {b['win_null']*100:.0f}% up-weeks; yours is {b['noise']} its noise band {b['win_lo']*100:.0f}-{b['win_hi']*100:.0f}%)")
    L.append(f"  Where the move happens     {b['split_1y']:<28} 1y: overnight {_sgn(b['on_1y'])}, intraday {_sgn(b['in_1y'])}; all data: overnight {_sgn(b['on_all'])}, intraday {_sgn(b['in_all'])}; gaps carry {b['gap_share']*100:.0f}% of daily variance")
    mt = f"IV/RV today {b['m_today']:.2f}: last 1y {_sgn(b['exp_mt1'], 1)}, all {_sgn(b['exp_mtA'], 1)}" if b["m_today"] else "IV/RV today unknown: pass --iv TICKER=<screener IV> or run --iv-from-chain between 9:30 and 16:00 New York"
    L.append(f"  Sell weekly call spreads?  {b['trade']:<28} at fair IV: last 1y {_sgn(b['exp1'], 1)} +/-{_pts(b['se1'])} of risk/wk, since {b['first'][:4]} {_sgn(b['expA'], 1)}/wk; breaks even at IV/RV {_num(b['mbe1'])} (1y) / {_num(b['mbeA'])} (all)")
    L.append(f"                             {'':<28} ({mt})")
    L.append(f"  Direction or premium?      {b['direction']:<28} call minus put spreads: 1y {_sgn(b['d1'], 1)}, all {_sgn(b['dA'], 1)} of risk/wk")
    L.append(f"  After a +15% week          {b['after_rip']:<28} next week up {_sgn(b['rip_win'], 0)[1:] if b['rip_win'] == b['rip_win'] else 'n/a'} of the time (all weeks {_sgn(b['all_win'], 0)[1:] if b['all_win'] == b['all_win'] else 'n/a'}), median {_sgn(b['rip_med'], 1)}, n={b['rip_n']}")
    if b["years"]:
        L.append(f"  By year (30d, fair IV)     {'last 8y: ' if b.get('years_cut') else ''}" + "  ".join(f"{y} {e*100:+.0f}%" for y, e, _ in b["years"]) + "   (stock: " + " ".join(f"{r*100:+.0f}%" for _, _, r in b["years"]) + ")")
    if b["deltas"]:
        best = max(b["deltas"], key=lambda t: t[1])
        L.append(f"  Best short delta, last 1y  {best[0]:.2f} {_sgn(best[1], 1)}/wk" + " " * 14 + "(" + ", ".join(f"{d:.2f}: {_sgn(e, 1)}" for d, e in b["deltas"]) + f"; full-width losses {b['full_loss']*100:.0f}% of weeks, worst run ${b['max_dd']:,.0f} per contract)")
    return "\n".join(L)


def brief_html(cards: list[dict], basket: str = "") -> str:
    css = ("body{background:#0b0c10;color:#e6e6e6;font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:28px;max-width:1100px}"
           ".card{background:#13161d;border:1px solid #252a38;border-radius:10px;padding:18px 22px;margin:0 0 18px}"
           ".h{font-size:22px;font-weight:600;margin:0 0 10px} .h small{color:#8c94a8;font-weight:400;font-size:14px;margin-left:12px}"
           ".row{display:grid;grid-template-columns:230px 250px 1fr;gap:10px;padding:7px 0;border-top:1px solid #1f2430} .row:first-of-type{border-top:0}"
           ".q{color:#8c94a8} .v{font-weight:600} .yes{color:#4caf82} .no{color:#e05c5c} .mid{color:#c9a84c} .d{color:#c8cdd8;font-size:13.5px}"
           ".basket{color:#c9a84c;font-size:14px;margin:0 0 14px}")
    def cls(v):
        v = v.upper()
        return "yes" if v.startswith("YES") or v.startswith("DIRECTION") else ("no" if v.startswith("NO") or v.startswith("PUT") else "mid")
    def row(q, v, d):
        return f'<div class="row"><div class="q">{q}</div><div class="v {cls(v)}">{v}</div><div class="d">{d}</div></div>'
    out = [f'<!doctype html><html><head><meta charset="utf-8"><title>avgweek brief</title><style>{css}</style></head><body>']
    if basket:
        out.append(f'<p class="basket">{basket}</p>')
    for b in cards:
        mt = f"IV/RV today {b['m_today']:.2f}: last 1y {_sgn(b['exp_mt1'], 1)}, all {_sgn(b['exp_mtA'], 1)}" if b["m_today"] else "IV/RV today unknown: pass --iv TICKER=&lt;screener IV&gt; or run --iv-from-chain during US market hours"
        si = b.get("si_now")
        si_txt = f" &middot; short interest {si['pct_float']*100:.0f}% of float, {si['days_to_cover']:.1f} days to cover" if si and si.get("pct_float") else (f" &middot; {si['days_to_cover']:.1f} days to cover" if si else "")
        years = "&nbsp;&nbsp;".join(f"{y} <b>{e*100:+.0f}%</b> <span class=q>({r*100:+.0f}%)</span>" for y, e, r in b["years"])
        deltas = ", ".join(f"{d:.2f}: {_sgn(e, 1)}" for d, e in b["deltas"])
        out.append(f'<div class="card"><div class="h">{b["ticker"]} <small>${b["close"]:,.2f} on {b["asof"]} &middot; 1y {_sgn(b["ret_1y"])} &middot; vol {b["vol"]*100:.0f}%{si_txt} &middot; {b["skipped_note"]}</small></div>'
                   + row("Grinds down, rips up?", b["desc"], f"average week {_sgn(b['mean'], 1)} (mean), typical week {_sgn(b['median'], 1)} (median), up {b['win']*100:.0f}% of weeks; year {_sgn(b['ret_1y'])} but {_sgn(b['cum_ex_top5'])} without its best 5 weeks. A {b['vol']*100:.0f}%-vol random walk gives {b['win_null']*100:.0f}% up-weeks; yours is {b['noise']} its noise band {b['win_lo']*100:.0f} to {b['win_hi']*100:.0f}%.")
                   + row("Where the move happens", b["split_1y"], f"Last year: overnight {_sgn(b['on_1y'])}, intraday {_sgn(b['in_1y'])}. All data: overnight {_sgn(b['on_all'])}, intraday {_sgn(b['in_all'])}. Overnight gaps carry {b['gap_share']*100:.0f}% of daily variance, and a weekly spread owns every gap.")
                   + row("Sell weekly call spreads?", b["trade"], f"At fair IV: last 1y {_sgn(b['exp1'], 1)} &plusmn;{_pts(b['se1'])} of risk per week, since {b['first'][:4]} {_sgn(b['expA'], 1)} per week. Breaks even at IV/RV {_num(b['mbe1'])} (1y) / {_num(b['mbeA'])} (all). {mt}.")
                   + row("Direction or premium?", b["direction"], f"Call minus put spreads: 1y {_sgn(b['d1'], 1)}, all {_sgn(b['dA'], 1)} of risk per week. If puts do as well, the edge is the premium, not the drift.")
                   + row("After a +15% week", b["after_rip"], f"Next week up {b['rip_win']*100:.0f}% of the time against {b['all_win']*100:.0f}% for all weeks, median {_sgn(b['rip_med'], 1)}, n={b['rip_n']}." if b["rip_win"] == b["rip_win"] else "Not enough rips in the window.")
                   + row("By year (30-delta, fair IV)" + (" , last 8" if b.get("years_cut") else ""), "", years + " <span class=q>(stock return in brackets)</span>")
                   + (row("Best short delta, last 1y", f"{max(b['deltas'], key=lambda t: t[1])[0]:.2f} {_sgn(max(b['deltas'], key=lambda t: t[1])[1], 1)}/wk", f"{deltas}. Full-width losses {b['full_loss']*100:.0f}% of weeks; worst run ${b['max_dd']:,.0f} per contract. Differences inside &plusmn;{_pts(b['se1'], 0)} points are noise.") if b["deltas"] else "")
                   + "</div>")
    out.append('<p class="q" style="font-size:12.5px">Option P&amp;L is synthetic: Black-Scholes at IV = m x realized vol. "Fair IV" is m = 1. Full tables are in report.html next to this file.</p></body></html>')
    return "".join(out)


# ======================================================================================================
# 8. Orchestration
# ======================================================================================================

DIR_COLS = ["weeks", "cum_return", "mean_wk", "mean_log_wk", "median_wk", "median_ci_lo", "median_ci_hi", "win_rate", "win_ci_lo",
            "win_ci_hi", "t_stat", "direction_reliable"]
SHAPE_COLS = ["weeks", "ann_vol", "sd_wk", "vol_drag_wk", "win_rate", "win_null", "win_null_lo", "win_null_hi", "skew", "skew_lognormal_null", "tail_ratio_5", "avg_up_wk", "avg_down_wk",
              "payoff_ratio", "n_wk_gt_15", "exp_n_wk_gt_15_lognormal", "pct_wk_gt_10", "pct_wk_lt_m10", "cum_return", "cum_ex_top5",
              "best_wk", "worst_wk", "max_dd", "lag1_autocorr"]
HORIZON_COLS = ["weeks", "n_windows", "n_eff", "p_up_overlap", "p_up_nonoverlap", "p_up_iid_boot", "p_up_block_ci_lo", "p_up_block_ci_hi",
                "p_up_normal", "q05", "q25", "q50", "q75", "q95", "p_gt_10", "p_gt_20", "p_gt_30", "p_lt_m20"]


def _json_safe(obj):
    """Recursively replace NaN/inf (not valid JSON) with None and numpy scalars with Python ones."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if (obj != obj or obj in (float("inf"), float("-inf"))) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return str(obj)
    return obj


def analyse_ticker(ticker: str, daily: pd.DataFrame, source: str, args, daily_full: pd.DataFrame | None = None) -> dict:
    """`daily` is the analysis window; `daily_full` (default: the same) carries the warm-up history for
    realized vol and regime tags so --start does not silently drop the first weeks of the backtest."""
    dfull = daily_full if daily_full is not None else daily
    wk = weekly_series(daily, drop_partial=not args.include_partial)
    h = args.horizon
    print(f"\n{'='*110}\n{ticker}  |  {source}\n{'='*110}")
    print(f"Daily bars: {len(daily)} ({daily.index[0].date()} .. {daily.index[-1].date()})  |  weekly bars: {len(wk)}  |  last weekly close {wk['close'].iloc[-1]:,.2f} on {wk['date'].iloc[-1].date()}")

    rv_now = float(realized_vol_series(dfull["close"], args.rv_mode).iloc[-1])
    offline = source.startswith("csv")
    chain = None
    chain_note = "No chain calibration (run with --iv-from-chain during US market hours to get m_today and the skew)."
    overrides = {}
    for tok in list(args.iv or []) + [f"m:{t}" for t in (args.m or [])]:
        kind = "m" if tok.startswith("m:") else "iv"
        body = tok[2:] if kind == "m" else tok
        if "=" in body:
            k, v = body.split("=", 1)
            try:
                overrides[k.upper()] = (kind, float(v))
            except ValueError:
                pass
    if ticker.upper() in overrides:
        kind, val = overrides[ticker.upper()]
        if kind == "iv":
            iv_val = val / 100.0 if val > 3 else val
            m_val = iv_val / rv_now if rv_now else float("nan")
        else:
            m_val, iv_val = val, val * rv_now
        chain = {"source": "screener", "expiry": "screener", "dte": 7, "spot_used": float(dfull["close"].iloc[-1]), "iv_atm": iv_val, "iv_25d_call": float("nan"),
                 "iv_10d_call": float("nan"), "short_pct_float": None, "short_ratio_days": None, "skew_s": None, "strike_step": float("nan"),
                 "rv_now": rv_now, "m_today": m_val, "yahoo_iv_atm": float("nan")}
        chain_note = f"IV from your screener: {iv_val:.0%} against realized {rv_now:.0%} -> m_today = {m_val:.2f}."
        print(f"  {chain_note}")
    if chain is None and args.iv_from_chain:
        if offline:
            chain_note = "Chain calibration skipped: offline CSV source."
        elif args.end:
            chain_note = "Chain calibration skipped: --end makes today's chain meaningless against a historical spot."
        else:
            chain = chain_calibration(ticker, float(dfull["close"].iloc[-1]), rv_now, args.rate)
            if chain is None:
                chain_note = ("Chain calibration was requested but found no usable quotes (pre-market, after hours, or no listed options). "
                              "Re-run between 9:30 and 16:00 New York time to get m_today and the skew.")
        if chain is None:
            print(f"  ({chain_note})")
    if wk.attrs.get("data_gaps"):
        print(f"  data gaps: {wk.attrs['data_gaps']} week(s) follow a hole of more than {MAX_WEEK_GAP_DAYS} days and are excluded")
    stats = weekly_stats(wk, daily)
    yearly = yearly_stats(wk, daily)
    asof = wk.index[-1]
    horizon, exceed = {}, {}
    for label, months in (("1y", 12), ("2y", 24), ("all", None)):
        cutoff = lookback_cutoff(asof, months)
        sub = week_window(wk, cutoff)
        if label in stats.index:
            horizon[label] = horizon_block(sub["logret"], h=h, b_iid=args.boot, b_block=args.block_boot, seed=1)
            exceed[label] = exceedance_table(sub["ret"], float(stats.loc[label, "ann_vol"]), rate=args.rate,
                                             iv_today=chain["iv_atm"] if chain else None)
    horizon_df = pd.DataFrame(horizon).T.astype(float) if horizon else pd.DataFrame()
    for c in ("weeks", "n_windows"):
        if c in horizon_df.columns:
            horizon_df[c] = horizon_df[c].round().astype("Int64")
    dep = dependence_block((week_window(wk, lookback_cutoff(asof, 24)) if len(wk) > 30 else wk)["logret"])
    cond = conditional_stats(wk, daily, years=args.cond_years)
    decomp, top_days = daily_decomposition(daily)
    rolling = rolling_regime(wk)
    reliable = stats[stats["direction_reliable"]] if not stats.empty else stats
    regime_flag = ""
    if len(reliable) >= 2:
        spread = (reliable["median_wk"].max() - reliable["median_wk"].min()) / reliable["sd_wk"].mean()
        regime_flag = f"Median-across-lookbacks spread = {spread:.2f} sigma -> {'REGIME-DEPENDENT' if spread > 0.5 else 'stable across lookbacks'}."
    verdict = verdict_lines(stats, horizon.get("1y", {}), "1y")

    print("\n" + "\n".join(verdict))
    print("\n-- Direction, by lookback (nested) --")
    print(fmt_table(stats, DIR_COLS))
    print("\n-- Shape and tails, by lookback --")
    print(fmt_table(stats, SHAPE_COLS))
    print("\n-- Calendar years --")
    print(fmt_table(yearly, DIR_COLS[:8] + ["ann_vol", "cum_ex_top5", "n_wk_gt_15", "max_dd"]))
    print(f"\n-- {h}-week horizon --")
    print(fmt_table(horizon_df, HORIZON_COLS))
    if "1y" in exceed:
        print("\n-- Exceedance vs implied, last 1y (per $1 of spot) --")
        print(fmt_table(exceed["1y"]))
    print(f"\n-- The week after ... (last {args.cond_years:g}y) --")
    print(fmt_table(cond))
    print(f"\n-- Dependence (last 2y): ACF1..4 {dep.get('acf1', float('nan')):+.2f} {dep.get('acf2', float('nan')):+.2f} {dep.get('acf3', float('nan')):+.2f} {dep.get('acf4', float('nan')):+.2f} (SE {dep.get('acf_se', float('nan')):.2f}) | "
          f"Ljung-Box p {dep.get('ljung_box_p', float('nan')):.2f} | VR2 {dep.get('vr2', float('nan')):.2f} VR3 {dep.get('vr3', float('nan')):.2f} VR4 {dep.get('vr4', float('nan')):.2f}")
    print("\n-- Daily: overnight gap vs intraday, gap events --")
    print(fmt_table(decomp))
    print(f"\n-- Rolling 26-week median/win rate: {rolling}  {regime_flag}")

    # ---- chain calibration + earnings
    if chain and chain.get("source") != "screener":
        print(f"\n-- Today's chain {chain['expiry']} ({chain['dte']}d): ATM IV {chain['iv_atm']:.0%} (Yahoo column said {chain['yahoo_iv_atm']:.0%}), 25d call {chain['iv_25d_call']:.0%}, "
              f"skew s {chain['skew_s']}, step {chain['strike_step']}, RV now {rv_now:.0%}  =>  m_today = {chain['m_today']:.2f}"
              + (f"  | short interest {chain['short_pct_float']*100:.0f}% of float, {chain['short_ratio_days']} days to cover" if chain.get("short_pct_float") else ""))
    iv_mults = sorted(set(list(args.iv_mult) + [1.0] + ([round(chain["m_today"], 2)] if chain and chain["m_today"] == chain["m_today"] else [])))
    deltas = list(args.deltas) if args.base_delta in args.deltas else list(args.deltas) + [args.base_delta]
    mgmts = list(args.mgmt) if "hold" in args.mgmt else ["hold"] + list(args.mgmt)
    if deltas != list(args.deltas) or mgmts != list(args.mgmt):
        print(f"  (added base delta {args.base_delta} and/or 'hold' to the sweep so the baseline exists)")
    vol_1y = float(stats.loc["1y", "ann_vol"]) if "1y" in stats.index else float(stats["ann_vol"].iloc[-1])
    skew_s = args.skew if args.skew is not None else (chain["skew_s"] if chain and chain.get("skew_s") is not None else (0.3 if vol_1y >= 0.35 else -0.2))
    strike_step = args.strike_step or (chain["strike_step"] if (chain and args.grid_from_chain and chain["strike_step"] == chain["strike_step"]) else None)
    earnings = set()
    if args.earnings != "include" or args.earnings_file:
        earnings = load_earnings(ticker, args.earnings_file, fetch=(args.earnings != "include") and not offline)
    if earnings:
        src = []
        if args.earnings_file:
            src.append("file")
        if args.earnings != "include" and not offline:
            src.append("yfinance")
        print(f"  earnings dates known: {len(earnings)} ({' + '.join(src) or 'file'}) -> mode '{args.earnings}'")
    elif not offline and args.earnings != "include":
        print("  earnings dates: none found (yfinance calendar is unreliable for small caps; pass --earnings-file to skip them)")
    si_hist, float_shares = None, None
    if not offline and not args.no_short_interest:
        try:
            si_hist = short_interest_history(ticker)
            if si_hist.empty:
                si_hist = None
        except Exception as exc:  # noqa: BLE001
            print(f"  (short interest unavailable: {exc})")
        if si_hist is not None:
            try:
                import yfinance as yf
                float_shares = float((yf.Ticker(ticker).info or {}).get("floatShares") or 0) or None
            except Exception:  # noqa: BLE001
                float_shares = None
            last = si_hist.iloc[-1]
            pct = f", {last['short_interest']/float_shares*100:.0f}% of float" if float_shares else ""
            print(f"  short interest (Nasdaq, settlement {last['settlement_date'].date()}): {last['short_interest']/1e6:.1f}M shares{pct}, {last['days_to_cover']:.1f} days to cover; {len(si_hist)} settlement dates")
    regimes = regime_frame(dfull, si_hist, float_shares)

    # ---- backtests
    def mk(**kw):
        base = dict(side="call", delta=args.base_delta, width_pct=args.widths[0], iv_mult=1.0, rv_mode=args.rv_mode, skew=skew_s,
                    r=args.rate, strike_step=strike_step, snap_mode=args.snap, slip_frac=args.slip, slip_floor_per_leg=args.slip_floor,
                    mark_iv_mult=args.mark_iv_mult, commission=args.commission,
                    mgmt="hold", earnings=args.earnings if earnings else "include", earnings_move=args.earnings_move,
                    risk_budget=args.risk_budget)
        base.update(kw)
        return SpreadConfig(**base)

    rows, trade_sets = [], {}
    for delta in deltas:
        for m in iv_mults:
            for mg in mgmts:
                for w in args.widths:
                    cfg = mk(delta=delta, iv_mult=m, mgmt=mg, width_pct=w)
                    tr = backtest_spreads(dfull, wk, cfg, earnings, regimes)
                    trade_sets[cfg.label()] = (cfg, tr)
                    for plabel, months in (("1y", 12), ("2y", 24), ("3y", 36), ("all", None)):
                        s = summarize_trades(slice_period(tr, months))
                        if s.get("trades", 0) == 0:
                            continue
                        rows.append({"config": cfg.label(), "period": plabel, "side": "call", "delta": delta, "iv_mult": m, "mgmt": mg, "width": w,
                                     "trades": s["trades"], "win_rate": s["win_rate"], "total_pnl": s["total_pnl"], "pnl_const_risk": s["pnl_const_risk"],
                                     "expectancy_pct_risk": s["expectancy_pct_risk"], "profit_factor": s["profit_factor"], "max_dd_$": s["max_drawdown_$"],
                                     "beyond_short": s["finished_beyond_short_rate"], "model_delta": s["avg_model_delta"], "stock_ret": s["stock_buy_hold"]})
    sweep = pd.DataFrame(rows)
    if not sweep.empty and not args.no_breakeven:
        be_rows = {}
        for delta in deltas:
            for w in args.widths:
                for plabel, months in (("1y", 12), ("all", None)):
                    be_rows[(delta, w, plabel)] = breakeven_mult(dfull, wk, mk(delta=delta, width_pct=w, mgmt="hold"), earnings, regimes, months)
        sweep["m_be"] = [be_rows.get((d, w, p), float("nan")) if mg == "hold" else float("nan")
                         for d, w, p, mg in zip(sweep["delta"], sweep["width"], sweep["period"], sweep["mgmt"])]

    base_cfg = mk()
    base_label = base_cfg.label() if base_cfg.label() in trade_sets else next(iter(trade_sets))
    base_cfg, base_trades = trade_sets[base_label]
    put_cfg = replace(base_cfg, side="put")
    put_trades = backtest_spreads(dfull, wk, put_cfg, earnings, regimes)
    asof_tr = base_trades["expiry_date"].max() if "expiry_date" in base_trades.columns and len(base_trades) else asof
    m_be = {}
    if not args.no_breakeven:
        for plabel, months in (("1y", 12), ("2y", 24), ("all", None)):
            m_be[plabel] = breakeven_mult(dfull, wk, base_cfg, earnings, regimes, months)
    mirror_rows = []
    for plabel, months in (("1y", 12), ("2y", 24), ("all", None)):
        sc, sp = summarize_trades(slice_period(base_trades, months, asof_tr)), summarize_trades(slice_period(put_trades, months, asof_tr))
        if sc.get("trades"):
            mirror_rows.append({"period": plabel, "call_trades": sc["trades"], "call_win": sc["win_rate"], "call_pnl": sc["total_pnl"], "call_exp_pct_risk": sc["expectancy_pct_risk"],
                                "put_trades": sp.get("trades", 0), "put_win": sp.get("win_rate", float("nan")), "put_pnl": sp.get("total_pnl", float("nan")),
                                "put_exp_pct_risk": sp.get("expectancy_pct_risk", float("nan")),
                                "call_minus_put_pct_risk": sc["expectancy_pct_risk"] - sp.get("expectancy_pct_risk", float("nan")),
                                "m_be_call": m_be.get(plabel, float("nan"))})
    mirror = pd.DataFrame(mirror_rows).set_index("period") if mirror_rows else pd.DataFrame()
    # pricing-mode sensitivity: the same baseline priced off three realized-vol estimators
    mode_rows = []
    for mode in ("blend", "rv20", "longrun"):
        tr_m = base_trades if mode == base_cfg.rv_mode else backtest_spreads(dfull, wk, replace(base_cfg, rv_mode=mode), earnings, regimes)
        for plabel, months in (("1y", 12), ("all", None)):
            s = summarize_trades(slice_period(tr_m, months, asof_tr))
            if s.get("trades"):
                mode_rows.append({"rv_mode": mode, "period": plabel, "trades": s["trades"], "win_rate": s["win_rate"], "total_pnl": s["total_pnl"],
                                  "expectancy_pct_risk": s["expectancy_pct_risk"], "full_loss_rate": s["full_loss_rate"], "avg_sigma_used": s["avg_sigma_used"]})
    pricing_modes = pd.DataFrame(mode_rows).set_index(["rv_mode", "period"]) if mode_rows else pd.DataFrame()
    n_variants = int(sweep["config"].nunique()) if not sweep.empty else 0

    base_summary = summarize_trades(slice_period(base_trades, 12, asof_tr))
    base_all = summarize_trades(base_trades)
    yr = per_year(base_trades)
    ok = completed(base_trades)
    if "rv20_pctile" in ok.columns:
        ok["rv_bucket"] = pd.cut(ok["rv20_pctile"], [-0.01, 0.33, 0.66, 1.01], labels=["low third", "mid third", "high third"]).astype(str)
        ok.loc[ok["rv20_pctile"].isna(), "rv_bucket"] = "warm-up (no trailing-year percentile)"
    base_trend = group_summary(ok, "trend")
    base_si = pd.DataFrame()
    if "si_days_to_cover" in ok.columns and ok["si_days_to_cover"].notna().sum() >= 10:
        med = ok["si_days_to_cover"].median()
        ok["si_bucket"] = np.where(ok["si_days_to_cover"].isna(), "unknown", np.where(ok["si_days_to_cover"] > med, f"days-to-cover above {med:.1f}", f"days-to-cover at or below {med:.1f}"))
        base_si = group_summary(ok, "si_bucket")
    base_rv = group_summary(ok, "rv_bucket") if "rv_bucket" in ok.columns else pd.DataFrame()
    ok["last_week"] = np.where(ok["prev_week_ret"] > 0.15, "after > +15%", np.where(ok["prev_week_ret"] > 0, "after up week", "after down week"))
    base_cond = group_summary(ok, "last_week")
    base_earn = group_summary(ok, "earnings_week") if ("earnings_week" in ok.columns and ok["earnings_week"].any()) else pd.DataFrame()
    regime_filtered = pd.DataFrame()
    if args.regime_filter:
        try:
            sub_f = ok.query(args.regime_filter)
            rows_f = []
            for name, frame in (("unfiltered", ok), (f"filter: {args.regime_filter}", sub_f)):
                for plabel, months in (("1y", 12), ("all", None)):
                    s_f = summarize_trades(slice_period(frame, months, asof_tr))
                    if s_f.get("trades"):
                        rows_f.append({"set": name, "period": plabel, "trades": s_f["trades"], "win_rate": s_f["win_rate"], "total_pnl": s_f["total_pnl"],
                                       "expectancy_pct_risk": s_f["expectancy_pct_risk"], "expectancy_se_pct_risk": s_f["expectancy_se_pct_risk"],
                                       "full_loss_rate": s_f["full_loss_rate"], "stock_ret": s_f["stock_buy_hold"]})
            regime_filtered = pd.DataFrame(rows_f).set_index(["set", "period"]) if rows_f else pd.DataFrame()
        except Exception as exc:  # noqa: BLE001
            print(f"  (regime filter {args.regime_filter!r} failed: {exc}; columns: trend, pct_from_high52, rv20_pctile, above_dma200, prev_week_ret, earnings_week)")
    bench = {"buy_hold_100": float((ok["ST"].iloc[-1] - ok["S0"].iloc[0]) * 100.0) if len(ok) else float("nan"),
             "leaky_short": leaky_short_benchmark(dfull, base_trades)}

    print_summary_dict(f"-- Baseline {base_label}: last 1 year --", base_summary)
    print_summary_dict(f"-- Baseline {base_label}: all data --", base_all)
    print(f"\n-- Break-even IV/RV multiplier m_be for the baseline: {m_be}" + (f"  (m_today from chain: {chain['m_today']:.2f})" if chain else ""))
    print("\n-- Call side vs put side (mirror test) --")
    print(fmt_table(mirror))
    print("\n-- Same baseline priced off three realized-vol estimators (an edge that exists in only one is a strike-placement artifact) --")
    print(fmt_table(pricing_modes))
    print("\n-- Baseline by calendar year --")
    print(fmt_table(yr))
    print("\n-- Baseline by trend state at entry --")
    print(fmt_table(base_trend))
    print("\n-- Baseline by realized-vol percentile at entry --")
    print(fmt_table(base_rv))
    print("\n-- Baseline conditional on last week --")
    print(fmt_table(base_cond))
    if not base_earn.empty:
        print("\n-- Baseline: earnings weeks vs the rest (earnings weeks are a different trade) --")
        print(fmt_table(base_earn))
    if not base_si.empty:
        print("\n-- Baseline by short interest at entry (Nasdaq bi-monthly, known with a 10-business-day lag) --")
        print(fmt_table(base_si))
    if not regime_filtered.empty:
        print("\n-- Baseline: regime filter vs unfiltered (a filter fixed before looking at P&L is the only kind that counts) --")
        print(fmt_table(regime_filtered))
    print(f"\n-- Benchmarks over the same weeks: buy&hold 100 sh ${bench['buy_hold_100']:,.0f} | leaky-cap short 100 sh ${bench['leaky_short']:,.0f} | call spreads ${base_all.get('total_pnl', 0):,.0f}")
    print(f"\n-- Sweep (last 1y): {n_variants} configurations evaluated; at zero true edge the best of 16 reads about +6%/wk, so rank by expectancy with its SE, never by win rate --")
    if not sweep.empty:
        print(fmt_table(sweep[sweep["period"] == "1y"].drop(columns=["period", "side"]).set_index("config")))

    # ---- brief card
    m_today_summ = {}
    if chain and chain.get("m_today") == chain.get("m_today"):
        lab = mk(iv_mult=round(chain["m_today"], 2)).label()
        if lab in trade_sets:
            tr_mt = trade_sets[lab][1]
            m_today_summ = {"1y": summarize_trades(slice_period(tr_mt, 12, asof_tr)), "all": summarize_trades(tr_mt)}
    skipped_note = (f"backtest skipped {base_all.get('skipped', 0)} weeks: {base_all.get('skip_reasons', 'none')}" if base_all.get("skipped") else "no weeks skipped") + f"; call-wing skew assumed {skew_s:+.1f}"
    si_now = None
    if si_hist is not None and len(si_hist):
        last = si_hist.iloc[-1]
        si_now = {"days_to_cover": float(last["days_to_cover"]), "pct_float": (float(last["short_interest"]) / float_shares) if float_shares else None, "date": str(last["settlement_date"].date())}
    brief = brief_fields(ticker, wk, stats, base_summary, base_all, m_be, mirror, cond, yr, sweep, chain, chain_note, m_today_summ, skipped_note, decomp, si_now)

    # ---- outputs
    out_dir = Path(args.out) / ticker.upper()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "brief.html").write_text(brief_html([brief]), encoding="utf-8")
    stats.drop(columns=["pooled"], errors="ignore").to_csv(out_dir / "weekly_stats.csv")
    yearly.drop(columns=["pooled"], errors="ignore").to_csv(out_dir / "yearly_stats.csv")
    horizon_df.to_csv(out_dir / "horizon.csv")
    for k, v in exceed.items():
        v.to_csv(out_dir / f"exceedance_{k}.csv")
    cond.to_csv(out_dir / "conditional.csv")
    decomp.to_csv(out_dir / "daily_decomposition.csv")
    wk.to_csv(out_dir / "weekly_bars.csv")
    sweep.to_csv(out_dir / "sweep.csv", index=False)
    base_trades.to_csv(out_dir / f"trades_{base_label}.csv", index=False)
    put_trades.to_csv(out_dir / f"trades_{put_cfg.label()}.csv", index=False)
    ctx = dict(ticker=ticker, source=source, wk=wk, stats=stats, yearly=yearly, dir_cols=DIR_COLS, shape_cols=SHAPE_COLS, horizon=horizon,
               horizon_df=horizon_df[[c for c in HORIZON_COLS if c in horizon_df.columns]] if not horizon_df.empty else horizon_df, h=h,
               exceed=exceed, cond=cond, dep=dep, decomp=decomp, top_days=top_days, rolling=rolling, regime_flag=regime_flag, verdict=verdict,
               chain=chain, base_trades=base_trades, put_trades=put_trades, base_cfg=base_cfg, base_summary=base_summary, base_all=base_all,
               m_be=m_be, mirror=mirror, base_year=yr, base_trend=base_trend, base_rv=base_rv, base_cond=base_cond, bench=bench, sweep=sweep,
               pricing_modes=pricing_modes, n_variants=n_variants, base_earn=base_earn, regime_filtered=regime_filtered, chain_note=chain_note)
    (out_dir / "report.html").write_text(html_report(ctx), encoding="utf-8")
    summary = {"ticker": ticker, "version": __version__, "source": source, "asof": str(wk["date"].iloc[-1].date()), "last_close": float(wk["close"].iloc[-1]),
               "rv_now": rv_now, "chain": chain, "verdict": verdict, "weekly_stats": json.loads(stats.to_json(orient="index")),
               "yearly_stats": json.loads(yearly.to_json(orient="index")), "horizon": horizon, "dependence": dep, "rolling": rolling,
               "exceedance_1y": json.loads(exceed["1y"].to_json(orient="index")) if "1y" in exceed else None,
               "n_variants": n_variants, "pricing_modes": json.loads(pricing_modes.reset_index().to_json(orient="records")) if not pricing_modes.empty else None,
               "baseline": {"config": asdict(base_cfg), "last_1y": base_summary, "all": base_all, "m_be": m_be,
                            "mirror": json.loads(mirror.to_json(orient="index")) if not mirror.empty else None}}
    (out_dir / "summary.json").write_text(json.dumps(_json_safe(summary), indent=1, default=str, allow_nan=False))
    print(f"\nWrote {out_dir}/report.html (+ csv, summary.json)")
    return {"ticker": ticker, "wk": wk, "daily": daily, "stats": stats, "base_trades": base_trades, "put_trades": put_trades, "base_cfg": base_cfg,
            "summary": summary, "exceed": exceed, "sweep": sweep, "brief": brief}


def pooled_report(results: list[dict], args) -> None:
    """Stack stock-weeks across the basket: the only level with statistical power for shape and exceedance."""
    print(f"\n{'='*110}\nPOOLED BASKET: {', '.join(r['ticker'] for r in results)}\n{'='*110}")
    asof = max(r["wk"].index[-1] for r in results)
    rows, pooled_exceed = [], {}
    for label, months in LOOKBACKS:
        cutoff = lookback_cutoff(asof, months)
        subs = [week_window(r["wk"], cutoff) for r in results]
        stacked = pd.concat(subs)
        m = window_metrics(stacked, None, pooled=True)
        if not m:
            continue
        vols = []
        for r in results:
            d = r["daily"] if cutoff is None else r["daily"][r["daily"].index > cutoff]
            if len(d) > 10:
                vols.append(float(np.log(d["close"]).diff().dropna().std(ddof=1) * math.sqrt(252)))
        if vols:
            m["ann_vol"] = float(np.mean(vols))
        rows.append({"window": label, **m})
        if label in ("1y", "all"):
            tbl_p = exceedance_table(stacked["ret"], m["ann_vol"], rate=args.rate)
            # a basket is heterogeneous: benchmark each ticker at its own vol and weight by its weeks
            parts = []
            for r_, v_ in zip(results, vols):
                sub_ = week_window(r_["wk"], cutoff)["ret"].dropna()
                if len(sub_):
                    parts.append((len(sub_), exceedance_table(sub_, v_, rate=args.rate)))
            if parts:
                wsum = sum(n_ for n_, _ in parts)
                for col in ("lognormal_p", "bs_credit_w5", "bs_credit_w10"):
                    tbl_p[col] = sum(n_ * t_[col] for n_, t_ in parts) / wsum
                for w_ in (5, 10):
                    tbl_p[f"credit_over_payout_w{w_}"] = tbl_p[f"bs_credit_w{w_}"] / tbl_p[f"payout_w{w_}"].where(tbl_p[f"payout_w{w_}"] > 0)
            pooled_exceed[label] = tbl_p
    pstats = pd.DataFrame(rows).set_index("window") if rows else pd.DataFrame()
    print("\n-- Pooled direction --")
    print(fmt_table(pstats, DIR_COLS))
    print("\n-- Pooled shape --")
    print(fmt_table(pstats, [c for c in SHAPE_COLS if c != "max_dd"]))
    if "1y" in pooled_exceed:
        print("\n-- Pooled exceedance vs implied, last 1y --")
        print(fmt_table(pooled_exceed["1y"]))
    all_trades = pd.concat([r["base_trades"].assign(ticker=r["ticker"]) for r in results], ignore_index=True)
    all_puts = pd.concat([r["put_trades"].assign(ticker=r["ticker"]) for r in results], ignore_index=True)
    print("\n-- Pooled baseline (sum across tickers): calls | puts --")
    for plabel, months in (("1y", 12), ("all", None)):
        sc, sp = summarize_trades(slice_period(all_trades, months)), summarize_trades(slice_period(all_puts, months))
        if sc.get("trades"):
            print(f"  {plabel:<4} calls: n={sc['trades']:<5} win {sc['win_rate']*100:5.1f}%  P&L ${sc['total_pnl']:>10,.0f}  exp {sc['expectancy_pct_risk']*100:+5.1f}% of risk   |   "
                  f"puts: n={sp.get('trades', 0):<5} win {sp.get('win_rate', float('nan'))*100:5.1f}%  P&L ${sp.get('total_pnl', 0):>10,.0f}  exp {sp.get('expectancy_pct_risk', float('nan'))*100:+5.1f}%")
    print("\n-- Pooled baseline by ticker (all data) --")
    print(fmt_table(group_summary(all_trades, "ticker")))
    print("\n-- Pooled baseline by trend state at entry (all data) --")
    print(fmt_table(group_summary(all_trades, "trend")))
    print("\n-- Pooled baseline by year --")
    print(fmt_table(per_year(all_trades)))
    out_dir = Path(args.out) / "POOLED"
    out_dir.mkdir(parents=True, exist_ok=True)
    pstats.drop(columns=["pooled"], errors="ignore").to_csv(out_dir / "pooled_weekly_stats.csv")
    for k, v in pooled_exceed.items():
        v.to_csv(out_dir / f"pooled_exceedance_{k}.csv")
    all_trades.to_csv(out_dir / "pooled_trades_call.csv", index=False)
    all_puts.to_csv(out_dir / "pooled_trades_put.csv", index=False)
    print(f"\nWrote {out_dir}/")


def run_ticker(ticker: str, args) -> dict:
    cache_dir = Path(os.path.expanduser(args.cache_dir))
    csv = args.csv if args.csv else (str(Path(args.csv_dir) / f"{ticker.upper()}.csv") if args.csv_dir else None)
    daily, source = load_prices(ticker, cache_dir, csv=csv, refresh=not args.no_refresh, source=args.source, force=args.refresh)
    if args.end:
        daily = daily[daily.index <= pd.Timestamp(args.end)]
    daily_full = daily
    if args.start:
        daily = daily[daily.index >= pd.Timestamp(args.start)]   # analysis window; daily_full keeps the warm-up for vol/regimes
    if len(daily) < 30:
        raise SystemExit(f"{ticker}: only {len(daily)} daily bars in the window; need at least 30")
    if args.brief:
        with contextlib.redirect_stdout(io.StringIO()):
            res = analyse_ticker(ticker, daily, source, args, daily_full=daily_full)
        print(brief_text(res["brief"]) + "\n")
        return res
    return analyse_ticker(ticker, daily, source, args, daily_full=daily_full)


# ======================================================================================================
# 9. Self-test on synthetic data
# ======================================================================================================

def _frame_from_close(close: np.ndarray, S0: float) -> pd.DataFrame:
    idx = pd.bdate_range(end=pd.Timestamp("2026-10-09"), periods=len(close))
    opn = np.concatenate([[S0], close[:-1]])
    return pd.DataFrame({"open": opn, "high": np.maximum(opn, close) * 1.01, "low": np.minimum(opn, close) * 0.99, "close": close}, index=idx)


def _gbm_path(n_days: int, mu: float, sigma: float, rng, S0: float = 50.0) -> pd.DataFrame:
    dt = 7.0 / 365.0 / 5.0  # five sessions carry one calendar week of variance, matching the backtest's T = days/365
    lr = rng.normal((mu - 0.5 * sigma * sigma) * dt, sigma * math.sqrt(dt), n_days)
    return _frame_from_close(S0 * np.exp(np.cumsum(lr)), S0)


def _mixture_path(n_weeks: int, rng, p_rip: float, grind: tuple, rip: tuple, S0: float = 50.0) -> pd.DataFrame:
    """Each week is a 'grind' or a 'rip' week; its 5 daily simple returns are i.i.d. N(mean/5, sd/sqrt(5))."""
    is_rip = rng.random(n_weeks) < p_rip
    daily = np.empty(n_weeks * 5)
    for w in range(n_weeks):
        m, sd = rip if is_rip[w] else grind
        daily[w * 5:(w + 1) * 5] = rng.normal(m / 5.0, sd / math.sqrt(5.0), 5)
    daily = np.maximum(daily, -0.6)
    return _frame_from_close(S0 * np.cumprod(1.0 + daily), S0)


def _run_paths(paths: list, cfg: SpreadConfig) -> tuple[pd.DataFrame, pd.Series]:
    trades, rets = [], []
    for d in paths:
        wk = weekly_series(d, drop_partial=False)
        trades.append(backtest_spreads(d, wk, cfg))
        rets.append(wk["ret"].dropna())
    return pd.concat(trades, ignore_index=True), pd.concat(rets, ignore_index=True)


def selftest() -> int:
    fails = 0
    def check(name, ok, detail=""):
        nonlocal fails
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
        if not ok:
            fails += 1

    print("Self-test 1: closed-form checks")
    check("BS call S=100 K=100 T=1 r=5% vol=20% = 10.4506", abs(bs_call(100, 100, 1, 0.05, 0.2) - 10.450584) < 1e-4, f"got {bs_call(100,100,1,0.05,0.2):.6f}")
    check("put-call parity", abs(bs_call(100, 95, 0.5, 0.04, 0.6) - bs_put(100, 95, 0.5, 0.04, 0.6) - (100 - 95 * math.exp(-0.02))) < 1e-9)
    check("nppf(0.975) = 1.959964", abs(nppf(0.975) - 1.959964) < 1e-6, f"got {nppf(0.975):.6f}")
    check("nppf(0.001) = -3.090232", abs(nppf(0.001) + 3.090232) < 1e-5, f"got {nppf(0.001):.6f}")
    check("chi2 sf df=4 at 9.488 = 0.05", abs(chi2_sf_df4(9.488) - 0.05) < 1e-3, f"got {chi2_sf_df4(9.488):.4f}")
    K = strike_for_call_delta(60, 5 / 252, 0.04, 0.9, 0.30)
    check("call delta->strike->delta round trip (30d)", abs(bs_call_delta(60, K, 5 / 252, 0.04, 0.9) - 0.30) < 1e-6, f"K={K:.3f} ({K/60-1:+.1%} OTM)")
    Kp = strike_for_put_delta(60, 5 / 252, 0.04, 0.9, 0.30)
    check("put delta->strike->delta round trip (30d)", abs((1 - bs_call_delta(60, Kp, 5 / 252, 0.04, 0.9)) - 0.30) < 1e-6, f"K={Kp:.3f}")
    iv = implied_vol(bs_call(60, 65, 5 / 252, 0.04, 0.9), 60, 65, 5 / 252, 0.04, True)
    check("implied vol inversion recovers 90%", abs(iv - 0.9) < 1e-6, f"got {iv:.6f}")
    check("lognormal exceedance P(week > +10%) at 100% vol = 22.5%", abs(lognormal_exceedance(0.10, 1.0 / math.sqrt(52)) - 0.225) < 0.005,
          f"got {lognormal_exceedance(0.10, 1.0/math.sqrt(52)):.3f}")
    lo, hi = wilson_ci(23, 52)
    check("Wilson CI 23/52 = [32%, 58%]", abs(lo - 0.32) < 0.01 and abs(hi - 0.58) < 0.01, f"got [{lo:.3f}, {hi:.3f}]")

    print("Self-test 1b: deterministic weekly statistics and clock helpers")
    # 12 alternating weeks +10% / -5% starting from 100: every statistic is known by hand
    closes = [100.0]
    for i in range(12):
        closes.append(closes[-1] * (1.10 if i % 2 == 0 else 0.95))
    idx = pd.bdate_range("2026-01-05", periods=13 * 5, freq="B")  # a Monday, so each 5-day block is one calendar week
    dd = pd.DataFrame({"close": np.repeat(closes, 5)}, index=idx)
    dd["open"] = dd["close"]; dd["high"] = dd["close"]; dd["low"] = dd["close"]
    wkd = weekly_series(dd, drop_partial=False)
    md = window_metrics(wkd, dd)
    check("alternating +10%/-5% weeks: mean 2.5%, median 2.5%, win 50%, payoff 2.0, cum +30.2%",
          abs(md["mean_wk"] - 0.025) < 1e-9 and abs(md["median_wk"] - 0.025) < 1e-9 and abs(md["win_rate"] - 0.5) < 1e-9
          and abs(md["payoff_ratio"] - 2.0) < 1e-9 and abs(md["cum_return"] - ((1.1 * 0.95) ** 6 - 1)) < 1e-9,
          f"mean {md['mean_wk']:.4f} median {md['median_wk']:.4f} win {md['win_rate']:.2f} payoff {md['payoff_ratio']:.2f} cum {md['cum_return']:.4f}")
    hol = pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0]}, index=pd.to_datetime(["2026-04-06", "2026-04-07", "2026-04-08", "2026-04-09"]))  # Good Friday week
    hol["open"] = hol["high"] = hol["low"] = hol["close"]
    wkh = weekly_series(hol, drop_partial=False)
    check("holiday-Friday week closes on Thursday", len(wkh) == 1 and wkh["date"].iloc[0] == pd.Timestamp("2026-04-09") and wkh.index[0] == pd.Timestamp("2026-04-10"))
    past_thu = weekly_series(hol, drop_partial=True, now=datetime(2026, 10, 13, 12, 0, tzinfo=NY))
    check("a past week ending on a Thursday is kept as complete", len(past_thu) == 1)
    cut_wed = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=pd.to_datetime(["2026-04-06", "2026-04-07", "2026-04-08"]))
    cut_wed["open"] = cut_wed["high"] = cut_wed["low"] = cut_wed["close"]
    check("a week ending on a Wednesday is dropped as partial", len(weekly_series(cut_wed, drop_partial=True, now=datetime(2026, 10, 13, 12, 0, tzinfo=NY))) == 0)
    cur = pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0]}, index=pd.to_datetime(["2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15"]))
    cur["open"] = cur["high"] = cur["low"] = cur["close"]
    check("the current week is dropped before Friday's close and kept after",
          len(weekly_series(cur, drop_partial=True, now=datetime(2026, 10, 16, 12, 0, tzinfo=NY))) == 0
          and len(weekly_series(cur, drop_partial=True, now=datetime(2026, 10, 16, 16, 30, tzinfo=NY))) == 1)
    gapd = pd.concat([pd.DataFrame({"close": [10.0] * 10}, index=pd.bdate_range("2026-01-05", periods=10)),
                      pd.DataFrame({"close": [20.0] * 10}, index=pd.bdate_range("2026-03-02", periods=10))])
    gapd["open"] = gapd["high"] = gapd["low"] = gapd["close"]
    wkg = weekly_series(gapd, drop_partial=False)
    check("a hole in the data does not become a +100% week", wkg["ret"].dropna().abs().max() < 1e-9 and wkg.attrs["data_gaps"] == 1)
    one_y = week_window(weekly_series(_gbm_path(600, 0.0, 0.5, np.random.default_rng(1)), drop_partial=False), pd.Timestamp("2026-10-09") - pd.DateOffset(months=12))
    check("a 1-year window holds 52 weekly returns, not 53", int(one_y["ret"].notna().sum()) == 52, f"got {int(one_y['ret'].notna().sum())}")
    lo_m, hi_m = median_ci(pd.Series(np.arange(1, 53, dtype=float)))
    check("median CI is symmetric in rank", abs((lo_m - 1) - (52 - hi_m)) <= 1, f"[{lo_m}, {hi_m}]")
    check("poisson tail P(X>=3 | 0.8) = 0.047", abs(poisson_tail(3, 0.8) - 0.0474) < 0.002, f"got {poisson_tail(3, 0.8):.4f}")
    tue_noon = datetime(2026, 10, 13, 12, 0, tzinfo=NY)
    check("expected_last_session: Tuesday noon -> Monday", expected_last_session(tue_noon) == date(2026, 10, 12))
    check("expected_last_session: Tuesday 16:30 -> Tuesday", expected_last_session(datetime(2026, 10, 13, 16, 30, tzinfo=NY)) == date(2026, 10, 13))
    check("expected_last_session: Sunday -> Friday", expected_last_session(datetime(2026, 10, 11, 10, 0, tzinfo=NY)) == date(2026, 10, 9))
    live = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.to_datetime(["2026-10-12", "2026-10-13"]))
    check("drop_live_bar removes today's bar before the close and keeps it after",
          len(drop_live_bar(live, tue_noon)) == 1 and len(drop_live_bar(live, datetime(2026, 10, 13, 16, 30, tzinfo=NY))) == 2)
    ym = pd.DataFrame({("Close", "ASTS"): [10.0, 11.0], ("Open", "ASTS"): [9.0, 10.5], ("High", "ASTS"): [10.5, 11.5], ("Low", "ASTS"): [8.9, 10.2], ("Volume", "ASTS"): [1, 2]},
                      index=pd.DatetimeIndex(pd.to_datetime(["2026-10-08", "2026-10-09"]).tz_localize("America/New_York")))
    ym.columns = pd.MultiIndex.from_tuples(ym.columns, names=["Price", "Ticker"])
    nd = _normalise_daily(ym)
    check("yfinance MultiIndex + tz-aware frame normalises to flat tz-naive OHLC", list(nd.columns[:4]) == ["open", "high", "low", "close"] and nd.index.tz is None and nd["close"].iloc[-1] == 11.0)
    ya = pd.DataFrame({"Open": [10.0], "High": [11.0], "Low": [9.0], "Close": [10.0], "Adj Close": [5.0], "Volume": [1]}, index=pd.to_datetime(["2026-10-09"]))
    check("Adj Close rescales OHLC", abs(_normalise_daily(ya)["open"].iloc[0] - 5.0) < 1e-12)

    blk = {"form": ["8-K", "10-Q", "8-K", "8-K/A"], "items": ["2.02,9.01", "", "1.01", "2.02"], "reportDate": ["2026-08-11", "2026-08-12", "2026-09-01", ""], "filingDate": ["2026-08-12", "2026-08-12", "2026-09-02", "2026-05-13"]}
    ed = harvest_edgar_8k(blk)
    check("EDGAR 8-K item 2.02 harvest picks event dates (and filing date when reportDate is blank)", ed == {pd.Timestamp("2026-08-11"), pd.Timestamp("2026-05-13")}, str(sorted(ed)))
    si = parse_nasdaq_short_interest({"data": {"shortInterestTable": {"rows": [{"settlementDate": "09/30/2026", "interest": "12,345,678", "avgDailyShareVolume": "4,000,000", "daysToCover": "3.09"},
                                                                                 {"settlementDate": "09/15/2026", "interest": "11,000,000", "avgDailyShareVolume": "5,500,000", "daysToCover": "2.00"}]}}})
    check("Nasdaq short-interest table parses and lags availability by 10 business days", len(si) == 2 and si["short_interest"].iloc[-1] == 12345678 and si["available_date"].iloc[-1] == pd.Timestamp("2026-10-14"), str(si["available_date"].tolist()))

    print("Self-test 2: one hand-checked trade")
    d = pd.DataFrame({"close": [100.0 * math.exp(0.05 * math.sin(i)) for i in range(30)] + [100.0, 105.0, 110.0, 118.0, 120.0]},
                     index=pd.bdate_range("2026-01-05", periods=35))
    d["open"] = d["close"]; d["high"] = d["close"] * 1.001; d["low"] = d["close"] * 0.999
    wk = weekly_series(d, drop_partial=False)
    cfg = SpreadConfig(delta=0.30, width_pct=0.10, iv_mult=1.0, rv_mode="rv20", slip_frac=0.0, slip_floor_per_leg=0.0, commission=0.0, mgmt="hold", skew=0.0)
    tr = backtest_spreads(d, wk, cfg)
    last = tr[tr["skipped"].isna()].iloc[-1]
    expected = (last["credit"] - min(last["ST"] - last["K_short"], last["width"])) * 100
    check("P&L = (credit - capped payoff) x 100 on a max-loss week", abs(last["pnl"] - expected) < 1e-9, f"pnl={last['pnl']:.2f} K={last['K_short']} w={last['width']}")
    check("max loss never exceeds (width - credit) x 100", (tr[tr["skipped"].isna()]["pnl"] >= -(tr[tr["skipped"].isna()]["max_risk"])).all())
    check("short strike snapped UP sits above the raw delta strike", last["K_short"] >= strike_for_call_delta(last["S0"], last["T_days"] / 365, 0.04, last["sigma"], 0.3) - 1e-9)
    check("slippage floor applies on tiny spreads", SpreadConfig().slip_floor_per_leg == 0.03 and max(0.02 * 0.5, 2 * 0.03) == 0.06)

    print("Self-test 3: fair-value sanity on a random walk (mu = r, IV = true vol, flat skew) -> expectancy ~ 0 before costs")
    rng = np.random.default_rng(7)
    paths = [_gbm_path(300, mu=0.04, sigma=0.90, rng=rng) for _ in range(150)]
    base = SpreadConfig(delta=0.5, width_pct=0.10, iv_mult=1.0, rv_mode="rv20", slip_frac=0.0, slip_floor_per_leg=0.0, commission=0.0, iv_floor=0.0, skew=0.0, snap_mode="nearest")
    _, wret = _run_paths(paths, base)
    mean_w, med_w, win_w = float(wret.mean()), float(wret.median()), float((wret > 0).mean())
    check("GBM shows the 'grind down / rip up' signature with ZERO edge: median < 0, win rate < 50%, mean > median",
          med_w < 0 and win_w < 0.5 and (mean_w - med_w) > 0.004, f"mean {mean_w*100:+.2f}%  median {med_w*100:+.2f}%  win {win_w*100:.1f}%  (n={len(wret)})")
    for delta in (0.5, 0.3, 0.16):
        for side in ("call", "put"):
            s = summarize_trades(_run_paths(paths, replace(base, delta=delta, side=side))[0])
            check(f"{side} d{delta}: |expectancy| < 2.5% of max risk over {s['trades']} weeks", abs(s["expectancy_pct_risk"]) < 0.025,
                  f"expectancy {s['expectancy_pct_risk']*100:+.2f}%  win {s['win_rate']*100:.1f}%  beyond short {s['finished_beyond_short_rate']*100:.1f}% vs model delta {s['avg_model_delta']*100:.1f}%")
    s = summarize_trades(_run_paths(paths, replace(base, delta=0.3, iv_mult=1.3))[0])
    check("charging 1.3x realized vol makes the seller's expectancy clearly positive", s["expectancy_pct_risk"] > 0.03, f"expectancy {s['expectancy_pct_risk']*100:+.2f}%")
    s = summarize_trades(_run_paths(paths, replace(base, delta=0.3, slip_frac=0.02, slip_floor_per_leg=0.03, commission=0.65))[0])
    check("default slippage + commissions alone cost the seller several % of risk per week", s["expectancy_pct_risk"] < -0.02, f"expectancy {s['expectancy_pct_risk']*100:+.2f}%")
    ex = exceedance_table(wret, 0.90)
    inside = all((ex.loc[k, "ci_lo"] - 0.01) <= ex.loc[k, "lognormal_p"] <= (ex.loc[k, "ci_hi"] + 0.01) for k in (0.05, 0.10, 0.20))
    check("exceedance table: lognormal benchmark inside the realized CI on GBM", inside,
          f"k=10%: realized {ex.loc[0.10,'realized_p']:.3f} [{ex.loc[0.10,'ci_lo']:.3f},{ex.loc[0.10,'ci_hi']:.3f}] vs lognormal {ex.loc[0.10,'lognormal_p']:.3f}")
    long_path = _gbm_path(252 * 6, mu=0.04, sigma=0.90, rng=np.random.default_rng(11))
    wk_long = weekly_series(long_path, drop_partial=False)
    mbe = breakeven_mult(long_path, wk_long, replace(base, delta=0.3), None, None, None)
    check("break-even multiplier m_be on a fair GBM path (mu = r) is within [0.75, 1.3]", 0.75 <= mbe <= 1.3, f"m_be = {mbe}")
    hb = horizon_block(wk_long["logret"], h=3, b_iid=4000, b_block=400)
    check("3-week block-bootstrap CI contains the point estimate", hb["p_up_block_ci_lo"] <= hb["p_up_overlap"] <= hb["p_up_block_ci_hi"],
          f"p_up {hb['p_up_overlap']:.3f} CI [{hb['p_up_block_ci_lo']:.3f}, {hb['p_up_block_ci_hi']:.3f}] iid {hb['p_up_iid_boot']:.3f} normal {hb['p_up_normal']:.3f}")

    print("Self-test 4 (informational): two 'grind down, rip up' regimes; expectancy as % of max risk, no costs, flat skew; calls | puts")
    regimes = {
        "RIP regime  (88% weeks N(-2%,7%), 12% weeks N(+22%,15%)) ~ ASTS 2024-25": (0.12, (-0.02, 0.07), (0.22, 0.15)),
        "GRIND regime (88% weeks N(-2.5%,6%), 12% weeks N(+12%,10%)) ~ ASTS Jun-Sep 2026": (0.12, (-0.025, 0.06), (0.12, 0.10)),
    }
    for name, (p_rip, grind, rip) in regimes.items():
        rng = np.random.default_rng(3)
        paths = [_mixture_path(52, rng, p_rip, grind, rip) for _ in range(120)]
        _, wret = _run_paths(paths, base)
        print(f"    {name}")
        print(f"      weekly mean {wret.mean()*100:+.2f}%  median {wret.median()*100:+.2f}%  win {(wret>0).mean()*100:.1f}%  skew {wret.skew():.2f}  ann vol {wret.std()*math.sqrt(52)*100:.0f}%  (n={len(wret)})")
        for m in (1.0, 1.25, 1.5):
            line = []
            for delta in (0.5, 0.3, 0.16, 0.1):
                sc = summarize_trades(_run_paths(paths, replace(base, delta=delta, iv_mult=m))[0])
                sp = summarize_trades(_run_paths(paths, replace(base, delta=delta, iv_mult=m, side="put"))[0])
                line.append(f"d{delta:<4}: {sc['expectancy_pct_risk']*100:+5.1f}% | {sp['expectancy_pct_risk']*100:+5.1f}%")
            print(f"      m={m:<4}: " + "   ".join(line))
    print("      (read: the same skew signature gives opposite results depending on the drift of the period; with IV = realized vol a rip-prone")
    print("       name is not a free short; and in a positively skewed name the ATM spread is the best short while far-OTM spreads are the worst,")
    print("       because the fat right tail is exactly where the OTM spreads lose their whole width.)")

    print(f"\nSelf-test {'PASSED' if fails == 0 else 'FAILED'} ({fails} failures)")
    return 1 if fails else 0


# ======================================================================================================
# 10. CLI
# ======================================================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tickers", nargs="*", help="e.g. ASTS RKLB HIMS (add SPY/QQQ as benchmarks)")
    p.add_argument("--csv", help="use this daily OHLCV CSV instead of downloading (one ticker only)")
    p.add_argument("--csv-dir", help="directory of TICKER.csv files to use instead of downloading")
    p.add_argument("--source", default="auto", choices=["auto", "yfinance", "stooq", "alphavantage"])
    p.add_argument("--cache-dir", default="~/.cache/avgweek")
    p.add_argument("--no-refresh", action="store_true", help="use cached data even if stale (offline)")
    p.add_argument("--refresh", action="store_true", help="force a full re-download even if the cache looks current")
    p.add_argument("--start", help="ignore data before this date (YYYY-MM-DD)")
    p.add_argument("--end", help="ignore data after this date (YYYY-MM-DD): 'what would this have said a year ago'")
    p.add_argument("--include-partial", action="store_true", help="keep the current, unfinished week")
    p.add_argument("--horizon", type=int, default=3, help="forward horizon in weeks (default 3)")
    p.add_argument("--cond-years", type=float, default=2.0, help="window for the week-after table")
    p.add_argument("--boot", type=int, default=10000, help="iid bootstrap draws")
    p.add_argument("--block-boot", type=int, default=2000, help="stationary block bootstrap replications")
    p.add_argument("--deltas", type=float, nargs="+", default=[0.5, 0.3, 0.16, 0.1], help="short-strike |deltas| to sweep")
    p.add_argument("--base-delta", type=float, default=0.3, help="delta for the detailed baseline tables")
    p.add_argument("--widths", type=float, nargs="+", default=[0.10], help="spread widths as a fraction of spot")
    p.add_argument("--iv-mult", type=float, nargs="+", default=[0.9, 1.0, 1.15, 1.3], help="IV/RV multipliers m to sweep; 1.1-1.25 is the normal range for single names, above 1.6 is rare (m_today from --iv or the chain is added)")
    p.add_argument("--iv-from-chain", action="store_true", help="read today's weekly chain from yfinance: m_today, skew, strike step")
    p.add_argument("--iv", nargs="*", default=[], help="implied vol from your screener per ticker, e.g. ASTS=105 RKLB=0.8 (percent or decimal); sets m_today = IV / realized vol")
    p.add_argument("--m", nargs="*", default=[], help="IV/RV multiplier per ticker directly, e.g. ASTS=1.2")
    p.add_argument("--no-short-interest", action="store_true", help="skip the Nasdaq short-interest fetch")
    p.add_argument("--grid-from-chain", action="store_true", help="use the chain's modal strike spacing as the grid for the whole backtest")
    p.add_argument("--rv-mode", default="blend", choices=["blend", "rv20", "longrun"], help="realized-vol estimator behind the IV proxy")
    p.add_argument("--skew", type=float, default=None, help="call skew s (default: from chain if available, else 0.3; 0 = flat)")
    p.add_argument("--rate", type=float, default=0.04)
    p.add_argument("--strike-step", type=float, default=None)
    p.add_argument("--snap", default="up", choices=["up", "nearest"], help="snap the short strike away from spot (conservative) or to nearest")
    p.add_argument("--slip", type=float, default=0.02, help="execution haircut per open/close as a fraction of width")
    p.add_argument("--commission", type=float, default=0.65, help="per contract per leg")
    p.add_argument("--mgmt", nargs="+", default=["hold", "tp50_sl2x"], choices=["hold", "tp50", "sl2x", "tp50_sl2x", "sl_touch", "close1dte"],
                   help="management variants (daily-close triggers, model marks at 1.25x vol beyond the short strike): approximate")
    p.add_argument("--mark-iv-mult", type=float, default=1.25, help="vol multiplier for management marks once spot is beyond the short strike")
    p.add_argument("--slip-floor", type=float, default=0.03, help="minimum slippage per leg in dollars")
    p.add_argument("--regime-filter", help="pandas query over trade rows, e.g. \"trend == 'down' and rv20_pctile > 0.5\"; shown next to unfiltered")
    p.add_argument("--universe", help="text file of tickers (one per line or whitespace separated) appended to the positional tickers")
    p.add_argument("--earnings", default="skip", choices=["skip", "include", "bump"], help="what to do with earnings weeks when dates are known")
    p.add_argument("--earnings-file", help="text file of earnings dates (YYYY-MM-DD) to use instead of / in addition to yfinance")
    p.add_argument("--earnings-move", type=float, default=0.12)
    p.add_argument("--risk-budget", type=float, default=2000.0, help="for the constant-risk sizing variant")
    p.add_argument("--no-breakeven", action="store_true", help="skip the break-even multiplier search (faster)")
    p.add_argument("--no-pool", action="store_true", help="skip the pooled-basket section when several tickers are given")
    p.add_argument("--out", default="avgweek_out")
    p.add_argument("--brief", action="store_true", help="print one short card per ticker (six lines that decide the thesis) and write BRIEF.html; nothing else")
    p.add_argument("--selftest", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        return selftest()
    if args.brief and args.mgmt == ["hold", "tp50_sl2x"]:
        args.mgmt = ["hold"]
    if args.universe:
        extra = []
        for line in Path(args.universe).read_text().splitlines():
            line = line.split("#", 1)[0]
            extra += [tok.strip(",;") for tok in line.split() if tok.strip(",;")]
        args.tickers = list(args.tickers) + extra
    if not args.tickers:
        build_parser().print_help()
        return 2
    if args.csv and len(args.tickers) != 1:
        raise SystemExit("--csv works with exactly one ticker; use --csv-dir for several")
    results, failed = [], []
    for t in args.tickers:
        try:
            results.append(run_ticker(t.upper(), args))
        except SystemExit as exc:
            failed.append((t, str(exc)))
            print(f"\n!! {t}: {exc}")
        except Exception as exc:  # noqa: BLE001 - one bad ticker must not sink the basket
            import traceback
            failed.append((t, repr(exc)))
            print(f"\n!! {t} failed: {exc!r}\n{traceback.format_exc(limit=3)}")
    if failed:
        print(f"\nFailed tickers: {', '.join(f'{t} ({msg[:80]})' for t, msg in failed)}")
    if not results:
        return 1
    if args.brief:
        cards = [r["brief"] for r in results]
        basket = ""
        if len(results) > 1:
            calls = pd.concat([r["base_trades"] for r in results], ignore_index=True)
            puts = pd.concat([r["put_trades"] for r in results], ignore_index=True)
            sc, sp = summarize_trades(calls), summarize_trades(puts)
            sc1, sp1 = summarize_trades(slice_period(calls, 12)), summarize_trades(slice_period(puts, 12))
            n_desc = sum(1 for c in cards if c["desc"] == "YES")
            n_trade = sum(1 for c in cards if c["trade"].startswith("YES"))
            basket = (f"Basket of {len(cards)}: grind-down-rip-up signature in {n_desc}; call spreads a YES in {n_trade}. Pooled at fair IV, all data: calls {_sgn(sc.get('expectancy_pct_risk'), 1)} vs puts {_sgn(sp.get('expectancy_pct_risk'), 1)} of risk/wk; "
                      f"last 1y: calls {_sgn(sc1.get('expectancy_pct_risk'), 1)} vs puts {_sgn(sp1.get('expectancy_pct_risk'), 1)}.")
            print(basket)
        out = Path(args.out) / "BRIEF.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(brief_html(cards, basket), encoding="utf-8")
        print(f"\nWrote {out}  (full tables per ticker are in {args.out}/<TICKER>/report.html)")
        return 0
    if len(results) > 1:
        print(f"\n{'='*110}\nCROSS-TICKER (baseline call spreads, last 1y)\n{'='*110}")
        for r in results:
            b = r["summary"]["baseline"]["last_1y"]
            st = r["stats"].loc["1y"] if "1y" in r["stats"].index else None
            med = f"median wk {st['median_wk']*100:+.1f}% [{st['median_ci_lo']*100:+.1f},{st['median_ci_hi']*100:+.1f}] win {st['win_rate']*100:.0f}%" if st is not None else ""
            print(f"  {r['ticker']:<6} {med:<58} spreads: n={b.get('trades',0):<3} win {b.get('win_rate',float('nan'))*100:5.1f}%  exp {b.get('expectancy_pct_risk',float('nan'))*100:+5.1f}% of risk  "
                  f"P&L ${b.get('total_pnl',0):>7,.0f}  m_be {_num(r['summary']['baseline']['m_be'].get('1y', float('nan')))}  stock {_pct(b.get('stock_buy_hold'), 0)}")
        if not args.no_pool:
            pooled_report(results, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
