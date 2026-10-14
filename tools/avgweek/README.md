# avgweek: the average week, and selling weekly call credit spreads

A standalone tool (separate from the barbell screener) that answers three questions for a ticker or a basket:

1. On the average week, does this stock go up or down, by how much, and how sure can we be?
2. What is the probability it is up (or up more than X%) three weeks out?
3. If I had sold a weekly call credit spread every week, what would have happened, and under what assumption about implied vol?

It runs on the laptop in Python 3.13 with numpy and pandas. yfinance is used for prices and today's option chain; Stooq, Alpha Vantage and plain CSV files are fallbacks. Nothing paid. Every statistic is printed to the terminal, written to CSV, and rendered in a self-contained `report.html`, and a `summary.json` is written for the screener to read later.

**This is not trading advice.** Read the "what it can and cannot claim" section before acting on any number it prints.

---

## Quick start (laptop)

```bash
cd tools/avgweek
python3 -m pip install -U yfinance requests        # numpy and pandas are already there
python3 avgweek.py --selftest                       # ~3 minutes; proves the math on synthetic data
python3 avgweek.py --universe universe_highbeta.txt --min-vol 0.6 --brief --out reports                   # one card per high-beta name + the pooled basket line
python3 avgweek.py --universe universe_highbeta.txt --min-vol 0.6 --brief --hold-weeks 2 --out reports2   # the same, two-week spreads entered every other week
python3 avgweek.py ASTS RKLB HIMS --iv-from-chain --out reports        # full report per name; the chain only answers between 9:30 and 16:00 New York
open reports/BRIEF.html reports/ASTS/report.html
```

`universe_highbeta.txt` holds 37 long-history, optionable high-beta names (ASTS, RKLB, HIMS, MSTR, SMR, the 2021 meme and SPAC cohort, the semis and crypto miners). It is a survivor list: names that went to zero are not in it, which flatters the stock and hurts the call seller. Add `--iv ASTS=105 MSTR=70 ...` with your screener's implied vols to get the program at today's pricing instead of at IV = realized vol.

Other ways to run it (a failing ticker is reported and skipped; the rest of a basket still runs):

```bash
python3 avgweek.py ASTS --end 2025-10-10              # what the tool would have said a year ago
python3 avgweek.py ASTS --csv asts_daily.csv          # offline, from a Yahoo or Stooq CSV download
python3 avgweek.py ASTS RKLB HIMS --csv-dir data/     # offline basket (data/ASTS.csv, data/RKLB.csv, ...)
python3 avgweek.py ASTS --deltas 0.5 0.3 0.1 --iv-mult 1.0 1.2 1.5 --mgmt hold --no-breakeven   # faster, narrower sweep
python3 avgweek.py ASTS --earnings-file asts_earnings.txt   # hand-maintained earnings dates (one YYYY-MM-DD per line)
python3 avgweek.py ASTS MSTR RKLB HIMS SMR --brief --iv ASTS=105 MSTR=70 --out reports   # cards for a hand-picked list; IV from your screener
python3 avgweek.py ASTS --regime-filter "trend == 'down' and rv20_pctile > 0.5"   # a pre-registered filter, shown next to unfiltered
python3 avgweek.py --universe screener_tickers.txt --out reports   # the whole screener basket + pooled view (one ticker per line, # comments allowed)
python3 avgweek.py ASTS SPY --brief                                # SPY, QQQ, IWM, DIA, VOO, VTI print a reference card only: never pooled, never counted as a candidate
python3 iv_snapshot.py ASTS RKLB HIMS --out iv_log.csv      # run daily near the close; builds a real IV history
```

Where the answer to "if I sold a call credit spread every week for the past year on ASTS" lives:

- Terminal: the block `-- Baseline call_d0.30_w10%_m1_hold: last 1 year --`, then `-- Break-even IV/RV multiplier m_be --`, then `-- Sweep (last 1y) --`.
- Files: `reports/ASTS/trades_call_d0.30_w10%_m1_hold.csv` is the week-by-week ledger; `sweep.csv` has every configuration; `report.html` has everything with charts.

The number to compare across configurations is `expectancy_pct_risk` (average P&L per week as a fraction of the maximum loss), with its standard error. Never rank by win rate: a 10-delta program wins 93% of weeks and loses money.

The arithmetic behind "short strike slightly above price, high win rate, every week or two": held to expiry, a credit spread breaks even at a win rate of `1 - credit / width`. Near-the-money weekly credits on these names run about 10 to 15% of width after costs, so the program has to win about 85 to 87% of its weeks; the realized rate at the 30-delta strike over these histories is 75 to 85%. The card's `Sell call spreads?` line is that comparison, expressed in dollars of risk per week (per two weeks with `--hold-weeks 2`), with the noise band one year of data carries (about +/-5 points).

---

## 1. Your framing questions, answered

### Is "the average week" the right unit?

Yes. A weekly call credit spread is a bet on one Friday-close-to-Friday-close return, so that return is the thing to measure. Weeks are built from daily bars with `resample("W-FRI").last()`, so a holiday Friday automatically closes on Thursday, exactly like the option. The current week is dropped until its Friday has closed (16:15 New York), a history that ends on a Monday to Wednesday is treated as a data cut, and a week that follows a hole of more than 10 days in the data is excluded rather than booked as a multi-week return. Daily data is still used where it is better: realized vol (precision improves with sampling frequency, direction does not), the overnight-vs-intraday decomposition, intraweek touches of the short strike, and regime tags.

Finer sampling does not help you learn direction. The standard error of the annual drift is `sigma / sqrt(years)` whatever the sampling frequency. At 100% vol, one year of data gives a drift error of 100% per year; five years still leaves 45%. So the lookback ladder (1m, 2m, 3m, 6m, 1y, 2y, 3y, 5y) mostly shows which regime each window happened to cover, not a stable property of the stock. The tool prints rows under 26 weeks greyed for direction (`direction_reliable = no`) and shows calendar years and a rolling 26-week median as non-nested views.

### "The average week" is three numbers, and they disagree for these names

| statistic | what it answers | ASTS-type behaviour |
|---|---|---|
| mean weekly return | what a constant-dollar long earns per week | small positive |
| mean log return (= geometric) | what buy-and-hold actually compounds at | mean minus the volatility drag `sigma^2/2` |
| median / win rate | the typical week, the sign frequency | negative, below 50% |

The gap is volatility drag. At 14% weekly standard deviation the drag is about 1% per week, or 50% per year. A pure random walk with zero excess drift (expected return equal to the risk-free rate, which is also the measure option prices live on) and 90% annual vol has a median weekly return of -0.7%, a win rate of 47.8%, and positive skew, by closed form. The self-test reproduces this on simulated paths: mean -0.14%, median -0.92%, win rate 46.8% over 8,850 weeks.

**So "grinds down most weeks but rips occasionally" is what every 80 to 120% vol stock looks like, edge or no edge.** It is not by itself a reason to sell calls. The verdict line at the top of every report separates the two things that matter: DIRECTION (median, win rate and t-statistic with intervals; for one ticker-year almost always "not distinguishable from zero") and SHAPE (skew against the lognormal null, count of +15% weeks against the lognormal expectation, tail ratio, and the year's return with its top five weeks removed).

### P(up in three weeks)

Four estimators are printed, because each has a different failure mode:

- overlapping 3-week windows (what the first draft used): 50 windows from 52 weeks, but each shares two thirds of its content with its neighbours, so the effective sample is about n/3;
- non-overlapping blocks: honest but only 17 observations per year;
- an iid bootstrap of three weekly log returns (10,000 draws): the point estimate and the quantiles of the 3-week return, P(> +10/20/30%), P(< -20%);
- a stationary block bootstrap (mean block four weeks, 2,000 replications): the confidence interval that survives weekly dependence;
- a normal approximation `Phi(sqrt(3) * mu / sigma)` as a labelled sanity row (biased upward under right skew).

Expect roughly 40 to 50% with a 95% interval of about 25 to 65% for any single high-vol name. Any realistic drift keeps the true number inside 40 to 60%, and the interval is wider than that whole band. The strikes you care about map to the exceedance table instead (section 3 of the report).

---

## 2. The thesis, tested on synthetic data before touching real prices

The self-test builds two "grind down, rip up" mixtures (88% grind weeks, 12% rip weeks) and sells spreads at IV = m x realized vol, no costs, flat skew. Expectancy per week as % of max risk, calls | puts:

```
RIP regime  (grind N(-2%,7%), rip N(+22%,15%))   mean +1.0%/wk  median -1.2%  win 44%  skew 2.2  ann vol 87%   ~ ASTS 2024-25
  m=1.0 : ATM  +2.3 | -4.1    30d -1.6 | -0.6    16d -4.0 | +0.2    10d -3.9 | +1.0
  m=1.25: ATM  +6.9 | +1.1    30d +2.3 | +6.1    16d -1.6 | +4.3    10d -2.4 | +3.1
  m=1.5 : ATM +10.5 | +4.8    30d +5.4 |+12.0    16d +0.3 | +7.6    10d -1.4 | +4.8
GRIND regime (grind N(-2.5%,6%), rip N(+12%,10%)) mean -0.7%/wk  median -1.8%  win 40%  skew 1.4  ann vol 59%   ~ ASTS Jun-Sep 2026
  m=1.0 : ATM  +5.0 |-10.2    30d +0.5 | -5.4    16d -1.7 | -2.6    10d -1.5 | -0.8
  m=1.25: ATM  +9.9 | -5.3    30d +4.3 | +0.5    16d +0.5 | +1.6    10d -0.0 | +1.9
  m=1.5 : ATM +13.9 | -1.4    30d +7.3 | +6.3    16d +2.4 | +5.0    10d +0.9 | +3.5
```

What this says, and what the red-team Monte Carlo (20,000 simulated years per regime) confirmed with sharper numbers:

1. **The drift of the period you trade decides the sign.** The same skew signature gives opposite results in the two regimes. In a melt-up like ASTS 2024 to 2025 every delta loses at m <= 1.0 and the program is destroyed at the regime turn (a seller carrying the old regime's vol estimate into a melt-up loses 7 to 9% of risk per week). In a grind like June to September 2026 everything works. The lookbacks you proposed cannot see the turn: the standard error of a 4-week mean is about 7% per week.
2. **The IV/RV multiplier m is the whole result.** Moving m from 0.9 to 1.5 moves the 30-delta expectancy by roughly 8 to 9 points of risk per week. Free data cannot tell you which m was true in any past week. The tool therefore solves for the break-even multiplier `m_be` per period and asks you to compare it with `m_today` from the live chain. A program that needs m_be = 1.3 needs a persistent 30% variance premium, which these names do not reliably offer outside earnings.
3. **Your "3 delta / 1 delta" instinct is backwards for positively skewed names.** The at-the-money spread is the best short and the far out-of-the-money spread is the worst, while the win rate rises from about 70% to 93% as delta falls. A breach of a far-OTM strike in these names is almost never partial: at 10 delta, about 70% of breaches take the full width, and lognormal pricing assumes mostly partial breaches. Near the money the spread harvests the negative median. Sell closer to the money, or do not sell.
4. **Width does not protect.** A +40% week blows through any 10%-wide spread. Width sets the credit-to-risk ratio, nothing else. The tool prints the full-width-loss rate (expect 3 to 8 full losses per year) and the worst week per year.
5. **The mirror test.** The identical program on the put side is run every time. If puts do as well as calls under the same m, the result is the assumed vol premium, not the directional thesis. Only a call-side advantage supports "grind down". One caveat the verifiers raised: with a single flat vol on both wings the comparison is itself biased, because the fair smile of a rip-prone name is lower on the put wing and higher on the call wing. Treat the mirror as decisive only once the two wings are priced from the chain (`--iv-from-chain` sets the skew; the IV log will eventually give both wings).
6. **Costs matter more than you think.** With the default slippage (2% of width per side, floor $0.03 per leg) and $0.65 per contract per leg, a 30-delta program at m = 1 loses about 4 to 5% of risk per week on a fair random walk. A 10-delta spread that collects 15 cents has no room.
7. **One year is one draw.** At zero true edge, one backtested year at 16 delta shows more than +2% of risk per week 36% of the time and a win rate above 90% 62% of the time. The best of 16 zero-edge variants on one year reads about +6% per week. The report stamps the number of configurations evaluated and prints the expectancy standard error; use them.
8. **Earnings weeks are 8% of weeks and 12 to 19% of losing P&L.** Skipping them is the one free filter. The tool skips them by default when it knows the dates (yfinance calendars are unreliable for small caps; keep a hand-maintained file per ticker).
9. **The trailing-vol proxy misprices at the worst time.** Trailing realized vol is low after calm weeks and high after wild ones, so the backtest books too little credit before the next rip and too much after it. With independent simulated returns this alone splits the 30-delta result into about -10% of risk per week in low-trailing-vol entry weeks and +10% in high ones. Any filter correlated with recent vol will inherit that artifact. The baseline is therefore priced three ways (blend, RV20, long-run) and shown side by side; a result that exists in only one mode is an artifact of the credit assumption, not a property of the stock.

---

## 3. What the report contains

`--brief` prints one card per ticker instead: average and typical week, where the move happens (overnight vs intraday), whether selling call spreads worked at fair pricing and at your screener's IV, direction vs premium, what happens after a +15% week, the sign by year, and the best delta. `BRIEF.html` collects the cards. The full report below is still written next to it.

| section | what it shows | how to read it |
|---|---|---|
| Verdict | DIRECTION / SHAPE / 3-WEEK lines for the last year | the only three sentences most people need |
| 1. Average week by lookback | weeks, cumulative return, mean, mean log, median with CI, win rate with Wilson CI, t-stat; then vol, drag, skew vs lognormal null, tail ratio, +15% week count vs expectation, return without top 5 weeks, max drawdown | rows under 26 weeks are noise for direction; calendar years and the rolling 26-week median are the non-nested views; the REGIME-DEPENDENT flag fires when the medians across rungs disagree by more than half a weekly sigma |
| 2. Three-week horizon | the four estimators and the quantiles | read the block-bootstrap interval |
| 3. Exceedance vs implied | for k = 0, 2.5, 5, 7.5, 10, 15, 20, 30% OTM: realized P(week > +k) with CI, the driftless-lognormal (price-martingale) benchmark at the window's realized vol, today's implied probability from the chain (if fetched), the average overshoot, the share of breaches that took the full width, the average weekly payout of a spread sold k% OTM for 5% and 10% widths, and the credit Black-Scholes would have charged at m = 1 | this is the model-free core; `credit_over_payout > 1` at m = 1 means the seller was paid more than it cost; a strike that only works above m = 1.3 relies on the variance premium |
| 4. The week after | next-week mean, median, win rate with CI, P(> +10%), worst, conditional on last week's sign, size, two-week streaks, 3-week moves, new 52-week high, new 26-week low, realized-vol tercile; ACF 1 to 4, Ljung-Box, variance ratios | compare each row with "all weeks", not with zero; conditioning on a +15% week selects lucky weeks that regress regardless; |rho| below about 0.27 is noise at n = 52 |
| 5. Daily decomposition | overnight gap vs intraday drift, overnight share of variance, count of sessions with |move| > 10% and > 20%, the ten largest sessions | the gap risk a weekly spread cannot manage; this is where your "intraday" idea belongs |
| 6. Baseline spread program | KPIs, cumulative P&L (1 contract and constant risk), call vs put mirror, break-even m, by year, by trend state at entry, by realized-vol percentile, by short interest at entry, conditional on last week, earnings weeks vs the rest, the regime filter if given, benchmarks (buy & hold 100 shares, and a weekly short of 100 shares covered at the first close at or above the long strike, labelled leaky because gaps jump through it) | regime panels are the finding; management variants (tp50, sl2x, close1dte) are approximate because they are marked on daily closes with flat intraweek IV |
| 7. Sweep | heatmap of expectancy by short delta x m with the break-even m column, full table for the last year | rank by expectancy with its standard error; count the variants |
| Footer | what it can and cannot claim | please read it once |

Pooled basket (several tickers): the same direction and shape tables on stacked stock-weeks, the pooled exceedance table, and the baseline program summed across tickers, by ticker, by trend state and by year. A five-point edge in exceedance needs about 260 ticker-weeks to resolve; the basket is the only level with statistical power. The basket line under the cards counts stock-weeks, the names with the grind-down-rip-up signature, and the names where call spreads were a YES, then pools every trade at fair IV (calls vs puts, all data and last year). Index ETFs (SPY, QQQ, IWM, DIA, VOO, VTI) print a reference card when passed, so you can see whether "grinding down" is name-specific or market-wide, but they are never pooled and never counted: an index week has positive drift and negative skew, the opposite of what a call seller wants, and a YES on SPY says nothing about ASTS.

---

## 4. The backtest, precisely

Per week, at the last close of the week (entry) to the last close of the next week (expiry):

- `sigma = max(m * RV, floor)`, where RV is the realized-vol estimate at entry (`blend` = max(0.25 RV10 + 0.5 RV20 + 0.25 RV60, 0.6 RV252); alternatives `rv20` and `longrun` = max(RV20, 0.75 RV252)).
- Short strike from the target delta by inverse Black-Scholes, snapped **away from spot** to the listed grid ($0.50 under $25, $1 under $100, $2.50 under $200, else $5; override with `--strike-step` or `--grid-from-chain`). Long strike = short + width (default 10% of spot, snapped). Time to expiry = calendar days / 365, the same clock the chain calibration uses for days to expiry.
- Each leg priced with its own vol: `IV(K) = sigma * (1 + s * (0.5 - |delta_K|))`, s = 0.3 by default or calibrated from the chain's 25-delta call. Set `--skew 0` for flat vol. Fat tails alone justify only a small wing premium (about 1.05x at 10 delta, 1.10x at 5 delta once the drift convention is matched); the rest of a real call wing is supply and demand, which only the chain can tell you.
- Credit = mid minus slippage (2% of width, floor $0.03 per leg); commissions $0.65 per contract per leg, charged on open and on any close (expiring worthless is free). Weeks whose credit is under 3% of width are skipped and counted.
- Hold to expiry: P&L = credit - max(0, min(S_T - K_short, width)), times 100. Max risk = (width - credit) x 100 plus a worst-case closing round of commissions and slippage, so P&L never falls below -max risk. Constant-risk sizing: contracts = floor(risk budget / max risk); weeks where one contract exceeds the budget are counted and shown.
- Management variants evaluate on daily closes: take profit at 50% of credit (`tp50`), stop at a loss of 2x credit (`sl2x`), close when the daily high crosses the short strike (`sl_touch`), close at 1 DTE if spot is within 1% of the short strike (`close1dte`). Marks use the entry vol, multiplied by 1.25 once spot is beyond the short strike (IV expands on rips; marks at entry vol are optimistic). Touch rate from daily highs is reported separately. All of these are approximate: no intraday prices, no real IV path.
- Earnings weeks: `skip` (default when dates are known), `include`, or `bump` (adds the implied move to that week's vol: `sqrt(IV^2 + (move/0.8)^2 / T)`).
- Regime tags per trade: trend (close vs 50-day average and its 20-day slope), % below 52-week high, realized-vol percentile over the trailing year, above/below the 200-day average, earnings week, previous week's return. `--regime-filter` takes a pandas query over these columns and shows the filtered program next to the unfiltered one.
- `m_be`: the multiplier at which the program's total P&L over the period is zero (bisection over 0.5 to 4, re-running the backtest), per delta and width. The summary also reports the expectancy standard error, the full-width-loss rate, expectancy as % of notional, and `years_to_2sigma = (2 sd / |mean|)^2 / 52`, the years of weekly data it would take to separate this configuration from zero.
- Put mirror: the same configuration with put spreads, short strike snapped away from spot downward.
- `--min-vol 0.6`: only enter when the trailing realized vol at the Friday close is at least 60%, across all years. This is the test of "high-beta, high-vol names" on the weeks that actually qualified, rather than on the calm early history of a name that became volatile later. Skipped entries are counted in the card header and the week counts shrink to match.
- `--hold-weeks 2`: sell a two-week spread every other week (entry Friday, expiry two Fridays later, T = 14/365, strike from the same target delta at that horizon). A hole in the data longer than 17 days skips the entry, as 10 days does for weekly spreads. Every per-period number on the card is then per two weeks and labelled `/2wk`; the per-year table and the basket line annualise by the trades' own cadence.

**Can claim**: how often and by how much weekly closes exceeded each strike distance (exact history); the hold-to-expiry P&L as a function of m, and the break-even m per period, ticker and configuration; whether calls beat puts under the same assumptions; the average-week statistics with honest intervals.

**Cannot claim**: the actual historical dollar result (entry credits depend on IV, skew and fills that are not in free data); that "the average week is down" is a property of the stock rather than of the window; realistic stop or target performance without intraday prices and real IV dynamics; anything about the next regime.

What the evidence says about m for these names: across S&P 500 single names the median IV/HV is about 1.12, high-beta snapshots run about 1.15 to 1.25, and readings above 1.6 are the top decile; index options carry a larger premium than single stocks; retail-heavy names show upside skew (out-of-the-money calls bid above at-the-money), which raises the credit on far strikes but also marks the tail the market expects. The default sweep is therefore 0.9, 1.0, 1.15, 1.3, and your screener's IV is the number to use when you have it.

Known bias: with a noisy trailing-vol proxy, the seller's P&L is concave in the vol error, so m = 1 is slightly pessimistic for the seller (about 1 to 2% of risk per week in the fair random-walk test). Right after a vol spike it flips to optimistic. The default m sweep is 0.9, 1.0, 1.15, 1.3 plus `m_today` from `--iv` or the chain; the detailed baseline tables are always at m = 1. Clock note: option time runs on calendar days (7/365 for a normal week, like the chain's days to expiry) while realized vol is annualised on 252 sessions, so "m = 1" is IV equal to RV only up to that convention (about 2% in sigma on a 5-session week, 10% on a 4-session holiday week).

---

## 5. How to read the real ASTS run, in order

1. Verdict lines. If DIRECTION says "not distinguishable from zero" (it will for one year), the thesis rests on SHAPE and on regime, not on the average week.
2. Calendar-year table and the rolling 26-week median. 2024 and 2025 were melt-ups, 2026 a grind. The "last 1 year" window straddles the all-time high; read per-year rows and run `--end 2025-10-10` to see what the tool would have said then.
3. Exceedance table, last year and all data: at 2.5%, 5% and 10% OTM, is `credit_over_payout_w10` above 1 at m = 1? Compare `realized_p` with `implied_p_today`.
4. Break-even m against `m_today` from `--iv-from-chain`. If m_be is above m_today, the market is not paying enough for the program as priced today.
5. Mirror test: `call_minus_put_pct_risk` by period. Positive in the grind slice and negative in the melt-up slice is what the thesis predicts.
6. Trend-state and vol-percentile tables: if the edge lives only in `trend = down`, the trade is a regime call, and you need a rule for recognising the regime that was fixed before you looked.
7. Heatmap: does the sign agree across neighbouring deltas and across m? Does it survive m = 1.0 and all three pricing modes?

Filter protocol, so you cannot fool yourself: pre-register at most four filters (previous week up, realized-vol or logged IV rank above its median, skip earnings, a fixed delta/width grid). Discover on ASTS, confirm out of sample on RKLB and HIMS, then on ten other screener names not chosen because they fit the story. Require at least 100 trades per cell, the same sign in every per-year panel and under all three pricing modes. Penalise the chosen variant's t-statistic by roughly `sqrt(2 ln N)` for N variants tried.

---

## 6. Data

- **Prices**: yfinance first (`period="max"`, adjusted close, the live intraday bar is dropped before 16:15 New York), then Stooq (same live-bar rule) (`https://stooq.com/q/d/l/?s=asts.us&i=d`), then Alpha Vantage (`ALPHAVANTAGE_API_KEY`, 25 requests a day, unadjusted), then the stale cache, then an error with a `--csv` hint. Cache at `~/.cache/avgweek/TICKER.csv`, re-downloaded in full when older than the last final session (never appended: a split rewrites adjusted history). `--no-refresh` for offline, `--refresh` to force.
- **yfinance in 2026**: it breaks for a few weeks whenever Yahoo changes the cookie/crumb flow (`pip install -U yfinance`), rate-limits shared IPs (`YFRateLimitError`: wait, or `--source stooq`), returns MultiIndex columns and tz-aware indexes (handled), and includes today's partial bar during the session (handled).
- **Implied vol from your screener**: `--iv ASTS=105 RKLB=0.8` (percent or decimal) sets `m_today = IV / realized vol` without touching Yahoo's chain; `--m ASTS=1.2` sets the multiplier directly. The brief card then reports the program at that multiplier.
- **Earnings dates**: SEC EDGAR 8-K filings with item 2.02 (Results of Operations), the event date of each filing, merged with Yahoo's calendar and any `--earnings-file`. Set `AVGWEEK_CONTACT="name email"` in your environment to satisfy the SEC's fair-access policy.
- **Short interest**: Nasdaq's per-symbol bi-monthly table (about one year), used with a 10-business-day publication lag; the baseline is split by days-to-cover above or below its median, and the latest reading sits in the card header. `--no-short-interest` skips it.
- **Today's option chain**: `--iv-from-chain` reads the weekly expiry nearest 7 days, recomputes IVs from bid/ask mids with our own Black-Scholes inversion (Yahoo's `impliedVolatility` column is often stale), and reports ATM IV, 25-delta and 10-delta call IV, the skew slope, the modal strike spacing, short interest, and `m_today = IV_atm / RV`. `m_today` is added to the sweep automatically.
- **Building a real IV history**: `iv_snapshot.py` appends one row per ticker per day (spot, expiry, ATM IV, RV20, m, skew, and strike/IV/mid/bid/ask plus the 10%-wide spread credit at the 50/30/20/10-delta calls) to `iv_log.csv`. Schedule it at 3:50pm ET on trading days (cron on macOS or Linux, Task Scheduler on Windows). After three to six months, the median m per name replaces the 1.0/1.25/1.5 guesses, and per-delta multipliers can be set from the skew column.
- **Historical option chains (second pass, one evening)**: the DoltHub `post-no-preference/options` database is the only broad free source (end-of-day chains with bid/ask/delta for US optionable names from about 2019; coverage of ASTS before it became liquid is uncertain). Existence check via the SQL API, then one request per Friday for the next-Friday expiry, cached per date, then a `--chain-source dolthub` mode that takes strikes and credits (sell at bid, buy at ask) from the files instead of Black-Scholes. OptionsDX free samples are large caps only; CBOE history is paid, but `https://cdn.cboe.com/api/global/delayed_quotes/options/ASTS.json` is a second no-auth live snapshot source. None of these hosts was reachable from the sandbox this was built in, so verify on the laptop.

---

## 7. Talking to the screener

`summary.json` per ticker holds: verdict lines, the full lookback and calendar-year tables, the horizon block, dependence statistics, the rolling-regime summary, the 1-year exceedance table, the chain calibration, the pricing-mode sensitivity, and the baseline program summary for the last year and all data with `m_be` and the mirror table. The screener can read it as-is.

Everything in sections 2 (Black-Scholes), 4 (statistics) and 5 (backtest) of `avgweek.py` is pure and importable:

```python
from avgweek import load_prices, weekly_series, weekly_stats, exceedance_table, SpreadConfig, backtest_spreads, summarize_trades
```

A browser version was deliberately not built yet. Yahoo and Stooq do not send CORS headers, so a GitHub Pages page cannot fetch prices on its own; the right architecture when wanted is for the Python tool to publish `data/TICKER.json` next to a page that reads it same-origin. The math (erf-based normal CDF, Acklam inverse CDF, bisection IV) was written without scipy so it ports to about 60 lines of JavaScript.

---

## 8. Files

| file | purpose |
|---|---|
| `avgweek.py` | the tool: data, statistics, backtest, report, self-test, CLI |
| `iv_snapshot.py` | daily chain logger |
| `universe_highbeta.txt` | 37 long-history high-beta optionable names for `--universe`; survivorship caveat in its comments |
| `README.md` | this document |
| `.gitignore` | keeps reports, caches and logs out of the repo |

Backlog, in order of value: slippage per leg from the logged bid/ask instead of a fraction of width; `--chain-source dolthub`; per-delta IV multipliers from the skew log; a touch-based stop using daily highs; Yang-Zhang OHLC realized vol as an alternative estimator; `--publish-json` for a browser page.
