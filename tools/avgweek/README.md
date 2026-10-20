# avgweek: the average week, and selling weekly call credit spreads

**What this is.** Three small Python scripts that read free daily price history and answer, honestly, three questions about volatile stocks: what the average week looks like (it is usually down, and the year is made in a handful of weeks); whether selling weekly call or put credit spreads on the name would have paid, at full cost and at an assumed option price; and where the name sits in its own history of grinds and run-ups. Everything is tested on synthetic data first and on a 37-name basket back to the 1980s, and every number is printed with its error bar.

**Disclaimer.** This is a personal research project, not investment advice, and nothing here is a recommendation to buy, sell or write any security or option. No historical option prices exist in free data, so every spread P&L is conditional on an assumed implied volatility; the universe is a list of names that survived, which flatters every long-biased result; one year of weekly data carries a noise band of about five points; free price feeds contain errors. Credit spreads can lose their full width in one week, and in these names they regularly do. Past results, real or simulated, do not predict future ones. Use at your own risk and do your own work.


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
python3 -m pip install -U yfinance requests
python3 avgweek.py --selftest                       # ~3 minutes; proves the math on synthetic data
python3 decide.py ASTS=105 RKLB=80                  # the one-screen answer: CCS, PCS or NEITHER for the week, the month, the year
python3 decide.py --test --universe universe_highbeta.txt   # walk-forward proof of that rule; read it before trusting a verdict
python3 avgweek.py --universe universe_highbeta.txt --min-vol 0.6 --brief --out reports                   # one card per high-beta name + the pooled basket line
python3 avgweek.py --universe universe_highbeta.txt --min-vol 0.6 --brief --hold-weeks 2 --out reports2   # the same, two-week spreads entered every other week
python3 avgweek.py ASTS RKLB HIMS --iv-from-chain --out reports        # full report per name; the chain only answers between 9:30 and 16:00 New York
open reports/BRIEF.html reports/ASTS/report.html
```

Two things about the shell. Paste one command at a time: zsh runs a pasted block line by line and feeds anything typed while a command runs into the next prompt. And do not paste comments: zsh passes a trailing `# like this` to the program as arguments (both tools now drop everything from a `#` token on, but `pip` and `open` do not). Do not run `pip install -U pandas numpy` for these tools: they work on pandas 2.x and 3.x as they are, and pandas 3 breaks Streamlit (your screener) until Streamlit catches up. If that upgrade already happened, `python3 -m pip install "pandas<3" "numpy<2.5"` puts it back.

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

## 0. decide.py: CCS, PCS or neither

`avgweek.py` is the research tool. `decide.py` is the one you use on a Friday: type a ticker, get three lines.

```
ASTS  $52.10 on 2026-10-09   IV 105% (screener), 1.14x realized   283 weeks since 2021
  week   PCS      sell 48p / buy 43p                     CCS  -1.2%  PCS  +3.4%  of risk per trade (hurdle +2.7%)
  month  NEITHER                                         CCS  -2.0%  PCS  +1.0%  of risk per trade (hurdle +3.3%)
  year   PCS      every week; this week sell 48p / buy 43p   CCS  -3.1%  PCS  +2.9%  of risk per week (hurdle +2.7%)
```

Run it with no arguments for a prompt: type a ticker and its IV from your screener, `ASTS 105` or `ASTS=105`, several on one line if you like, and `q` to finish. `--why` adds one line per row with the sample size, standard errors, win rates and credits. `--asof 2025-10-10` shows what it would have said a year ago. Index ETFs are refused as candidates.

Every run also writes a page: `decide.html` with one card per ticker (the verdict, the strikes, and the two sides' expectancies as bars against the hurdle tick; "numbers behind this card" unfolds the full text), or `decide_test.html` for `--test`. It follows your system's light or dark theme, needs no network, and in prompt mode grows as you type, so open it once and refresh. `--html path` renames it, `--no-html` skips it.

The determinant, the same for all three rows:

1. The stock's own weekly closes. For horizon h (1 week, 4 weeks) every past h-week return in the trailing window (3 years; all data for the year row) is rescaled from the realized vol at its own start to today's realized vol (filtered historical simulation), so the shape of the past (rips, grinds, gaps) is measured in today's units.
2. Today's 30-delta call spread and put spread, 10% of spot wide on the listed strike grid, with the wing skew, slippage and commissions of the backtest (`price_spread()` is shared), priced at today's IV: your screener's number, or `--m` times realized (default 1.15).
3. Expectancy of each side = average over the rescaled history of (net credit - capped payoff) / max risk.
4. Hurdle = the market's average return over the horizon plus 2.5 points, per unit of risk. That is your success criterion, "done weekly, beat the market by 2.5%", read per trade on the capital at risk: it is the only reading that is testable (per year it is undetectable; per week on the whole account it is impossible by construction). `--hurdle` and `--market-drift` change it. A side is a candidate when its expectancy clears the hurdle; the verdict is the better candidate, else NEITHER.

The year row answers "which side, every week, for the coming year": the weekly determinant on all available history.

`--test` is the proof. It runs the identical rule at every past Friday using only data up to that Friday, then scores what the chosen side actually did over the next period, pooled across the names you give it:

```
Walk-forward, 37 name(s), IV assumed 1.15x realized, hurdle = market + 2.5 points per unit of risk
  week   rule +0.4% of risk per trade (SE 0.3%, n 9,812 trades, ...)   traded 38% (CCS 7%, PCS 31%), chosen side minus the other +1.1% (SE 0.9%)   always CCS -3.0%   always PCS -1.6%   best in hindsight +9.8%
         SPY over the same span +0.2% per trade; rule minus SPY +0.2 points (target +2.5)
```

How to read it: `rule` is what following the verdicts would have earned per unit of risk, with weeks it stood aside counted as zero; compare it with `always CCS`, `always PCS` and with `best in hindsight` (the ceiling: the better side, or cash, chosen with perfect foresight). `chosen side minus the other` is the only clean measure of whether the side-picking adds anything; on a fair random walk it is zero within its standard error (self-test 5). `rule minus SPY` against the target is the pass/fail line for your criterion. `--m-grid` repeats the test at IV = 1.0, 1.15 and 1.3 times realized, because no free source has IV history and every P&L number is conditional on that assumption.

The week and month rows also print a state split: trades opened while the name was 40% or more below its 52-week high, against the rest. It answers whether put spreads did worse in a grind, which the verdicts themselves cannot see: they rescale past weeks to today's volatility whatever state the name was in. On fair random walks the two halves agree within their error bars, which the self-test checks; a gap of one standard error is noise.

Limits: the same week on several names is correlated, so each standard error clusters trades entered in the same week (for the month and year rows, trades whose holding periods overlap), and the naive figure beside it, which treats every stock-week as independent, shows how much that matters; the year row has a handful of observations per name; on stocks under about $25 the $0.50 strike grid and the slippage floor dominate the credit, and the tool will mostly say NEITHER for the right reason. When a side shows n/a, a line under the card says why, and the test counts such weeks as unsellable and leaves them out rather than scoring them as zero, so a name that spent years under $10 (ONDS, or ASTS and RKLB in 2023) contributes few trades; the IV you type applies to all three rows (term structure is ignored).

## 0b. cycle.py: where a name sits in its own history of run-ups

```
python3 cycle.py ASTS RKLB HIMS --universe universe_highbeta.txt
```

Per name:

- **State now**: distance below the 52-week and all-time highs, the current leg (down from a peak, or up from a trough) with its depth and age, and the share of the name's own 26-week windows that were lower than today's.
- **Run-up catalogue**: every trough-to-peak run-up of 2x or more (a 30% zigzag on weekly closes; `--run` and `--threshold` change it), each with the grind that preceded it, and the waits from a peak to the trough that started the next run-up, with today's wait against them.
- **The grind now**, for a name in a down leg: the leg's depth at its low and its average weekly pace against the name's past legs; among past legs, own and basket, that reached the same depth, how many more weeks and how much further they fell before the trough, and their final depth; and how long the legs that lasted as long ran in total. These lines start from the leg's low so far. When that low is weeks old, the line says so and how far today is above it. For a name in an up leg, the grind that preceded it.
- **Base rates**: from weeks at least as far below the 52-week high as today, how often the next 13, 26 and 52 weeks held a +50% move, a 2x, a -30% fall and a -50% fall, each against the same rate from any week; the quartiles of the 26-week return; and the three calendar years that supply most of those weeks.
- **Basket now**, with `--universe`, once at the end: how many names are in a down leg today, how old and deep those legs are, and how many are in an up leg.

With `--universe` the waits, grind lines and base rates are pooled across the other names, which is where the statistics live: a single name has three to six episodes. A base rate close to the any-week rate means the depth of the drawdown carries no timing information; on a random walk the two are equal, up and down, which the self-test checks. Each horizon counts only weeks with a full forward window, so the sample, printed per horizon, shrinks with the horizon. The year line shows how concentrated the evidence is: when one bear market supplies a third of the weeks, the basket rows are closer to one episode than to thousands of independent weeks. The grind lines answer "how did legs that were already this deep finish", which is not the same as "is this one near its end": a leg that reached today's depth and troughed there counts as zero more weeks and zero further fall. Only finished legs count, so legs still running today are left out and long legs are under-represented. The universe is today's survivors, so legs that ended in a delisting are missing too; both make the further-fall numbers understate the downside.


## 0c. coveredcall.py: own the shares, sell a call above the price once a year

```
python3 coveredcall.py
python3 coveredcall.py MSFT AAPL NVDA --target 20
python3 coveredcall.py --m-grid --detail
```

Each cycle holds the shares and sells one call expiring 52 weeks later, struck 12% above the price (`--target`). At expiry the position is worth the lower of the price and the strike, plus the premium and the interest it earned, and that amount rolls into the next cycle, so shares called away are bought back at the market. Buy and hold is the same shares with no call. With no tickers it runs the Mag 7, four highly traded mid caps (AAL, ETSY, ROKU, LYFT) and four highly traded high-beta names (PLTR, COIN, MSTR, RKLB) over the last five years, with SPY buy and hold as the reference line. `--years`, `--tenor` (weeks to expiry) and `--target` change the shape, so monthly calls 3% out of the money are `--tenor 4 --target 3`. `--delta 0.2` sets every strike at a 20-delta call instead, which puts it the same number of standard deviations above the price on a calm stock and a wild one.

Per name it prints the five-year total and the compounded yearly return for both, their difference, how many years the shares were called away, the average premium, the worst single year for each, and the share of all one-year windows in the span, starting any week, where the covered call beat holding. That last column guards against a lucky start date. The page `coveredcall.html` shows the same table.

No free source has option price history, so every call is priced with Black-Scholes at IV = m times the trailing one-year realized vol (m = 1; `--m-grid` adds 0.9 and 1.15), with the same wing skew as the other tools, approximate one-year Treasury yields by quarter (`--rate` fixes one), and 3% off each premium for the bid-ask spread (`--slip`). Real one-year quotes can differ by 10 to 20%. Prices are total-return, so dividends count for both; for dividend payers this slightly understates the covered call, which is negligible for the Mag 7. Early assignment and taxes are ignored. On a fair random walk the call changes nothing on average, which the self-test checks; the strategy wins when the stock goes sideways or down and loses when it rallies past the strike.

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
- **yfinance in 2026**: it breaks for a few weeks whenever Yahoo changes the cookie/crumb flow (`pip install -U yfinance`), rate-limits shared IPs (`YFRateLimitError`: wait, or `--source stooq`), returns MultiIndex columns and tz-aware indexes (handled), and includes today's partial bar during the session (handled). Under pandas 3 it returns a second-resolution index while CSVs and the short-interest table come back in microseconds; every date is normalised to nanoseconds on the way in, so the merges never see mixed units.
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
| `decide.py` | the one-screen verdict (CCS, PCS or NEITHER by week, month, year) with its walk-forward test and self-test |
| `cycle.py` | run-up catalogue, waits between run-ups, grind context, and conditional base rates up and down, per name and pooled |
| `coveredcall.py` | covered calls struck a set distance above the price, sold every year, against buy and hold |
| `iv_snapshot.py` | daily chain logger |
| `universe_highbeta.txt` | 37 long-history high-beta optionable names for `--universe`; survivorship caveat in its comments |
| `README.md` | this document |
| `.gitignore` | keeps reports, caches and logs out of the repo |

Backlog, in order of value: slippage per leg from the logged bid/ask instead of a fraction of width; `--chain-source dolthub`; per-delta IV multipliers from the skew log; a touch-based stop using daily highs; Yang-Zhang OHLC realized vol as an alternative estimator; `--publish-json` for a browser page.
