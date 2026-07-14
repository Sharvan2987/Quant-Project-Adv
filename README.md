# Quant-Project-Adv
# Regime-Shift: Macro-Aware Tactical Asset Allocation

A program that looks at market data, figures out whether the market is currently calm, falling, or in a full-blown crisis, and automatically reshuffles a portfolio (stocks, bonds, gold) to match.

Most simple portfolios pick fixed weights once (like 60% stocks / 40% bonds) and never change them. That works fine in normal times, but falls apart in a crash — a fixed split has no way of knowing the world has changed. This project detects when the market's "mood" shifts, and adjusts the portfolio in response.

---

## What it actually does

1. Pulls real daily price data for a stock (HDFC Bank), a gold ETF, and a government bond ETF, all from Yahoo Finance.
2. Builds two simple signals from that data: **momentum** (is it trending up or down?) and **volatility** (how much is it swinging?).
3. Feeds those signals into a **Hidden Markov Model (HMM)**, which sorts every day into one of three hidden "regimes" — Bull, Bear, or Crisis — without ever being told the answer.
4. Tests whether that regime detection would have actually worked in real time (not just in hindsight) using **walk-forward validation**.
5. Uses the detected regime to decide portfolio weights across stocks/bonds/gold using **convex optimization (cvxpy)**, with a different objective for each regime.
6. Backtests the whole thing — including trading costs — against two simple benchmarks: a static 60/40 portfolio and an equal-weight portfolio.

---

## Why these decisions

### Why 3 regimes (Bull / Bear / Crisis)?
Three is the smallest number that captures the distinction that actually matters for a portfolio: a calm uptrend, a calm-to-moderate downtrend, and a violent, high-volatility panic. Each of these genuinely calls for a different portfolio stages, which is the whole point of the project. More regimes (4, 5+) would let the model overfit and start finding "regimes" that are really just noise.

### Why momentum + volatility as features?
The HMM never sees a price chart — it only sees whatever numbers we hand it, so those numbers have to actually describe "what kind of day is this." Momentum (1-week / 1-month / 1-quarter rolling return) captures *direction*. Volatility (rolling standard deviation of returns) captures *turbulence*. Neither alone is enough: a stock can be volatile and rising (a wild bull run) or calm and falling (a slow bleed). Together, the combination is what actually separates "Bull" from "Bear" from "Crisis."

### Why HDFC Bank as the regime detector?
The project brief calls for NSE-based regime detection. HDFC Bank (`HDFCBANK.NS`) is a large, liquid NSE stock with a long, clean trading history, which makes it a reasonable proxy for "what mood is the Indian market in" without needing a broad index license.

### Why these three assets for the portfolio (stock / gold / bonds)?
- **HDFC Bank** (`HDFCBANK.NS`) — large and liquid.
- **GOLDBEES.NS** — an NSE-listed ETF tracking domestic gold prices.
- **LTGILTBEES.NS** — an NSE-listed long-duration (8–13yr) government bond ETF.

All three are pulled live via `yfinance`, so the code stays reproducible and current. One real constraint: the bond ETF only started trading in **June 2016**, so the final portfolio backtest (Phase 5) runs from mid-2016 onward, not from 2010.

### Why walk-forward validation?
This is the most important part of the whole project. If you fit one HMM on 15 years of data and then use it to label what regime 2016 was in, that model secretly "knows" about 2020's crash and later years while labeling 2016 — information nobody actually had at the time. Backtests with this bug almost always look great, because the model is quietly cheating.

Walk-forward validation fixes this by simulating time honestly: train only on data strictly *before* a given point, label the next chunk, slide the window forward, repeat. A leakage check explicitly verifies that no training window ever contains a date later than the days it's predicting.

### Why a "sticky" HMM initialization and causal smoothing?
Left alone, an HMM fit on noisy daily features can flip between regimes every few days, which doesn't look like a real market phase (regimes should last weeks to months). Two fixes were used:
1. The transition matrix is initialized with a strong self-transition bias before fitting, nudging the model toward persistent regimes rather than a degenerate rapidly-switching solution.
2. A causal smoothing filter only confirms a new regime once it's persisted for several consecutive days. It only ever looks backward in time, so it doesn't introduce any lookahead — it just adds a realistic "confirmation delay," the same way a real-time observer wouldn't declare a crisis after one noisy day.

### Why different objectives per regime?
- **Bull** → maximize risk-adjusted return (worth taking on more risk when conditions are favorable).
- **Bear** → minimize portfolio variance (capital preservation over chasing returns).
- **Crisis** → minimize variance *and* cap combined stock+gold exposure.
  
### Why transaction costs?
A regime-switching strategy that trades for "free" will always look better than it should — every time the model flips its opinion, real money would be spent crossing the bid-ask spread. A small cost (7.5 bps) is charged on portfolio turnover at every rebalance, so the backtest reflects a strategy that could actually survive frequent regime flips, not one that's secretly trading for free.

### Why is the regime signal shifted by one day before trading on it?
A day's momentum/volatility features are computed using that day's own closing price. In live trading you wouldn't know that day's regime until after the market closed, so trading on it during the same session is a subtle lookahead. The regime series is shifted forward by one day before being used to set portfolio weights, so today's trade is always based on yesterday's fully-known regime.

---

## How to run it

**Requirements:**
```
pandas
numpy
matplotlib
yfinance
scikit-learn
hmmlearn
cvxpy
```

**Run:**
1. Open `Quant_Project.ipynb` in Jupyter.
2. Restart the kernel and run all cells top to bottom, in order. Later phases depend on variables created in earlier ones (e.g. Phase 5 needs `df_hdfc_model` from Phase 3, and `wf_df` from Phase 4).
3. You'll need an internet connection — all price data is pulled live from Yahoo Finance via `yfinance`, nothing is read from a local file.

## Reproducing results

Because all data is pulled live, results will drift slightly every time you re-run the notebook as new trading days get added to the dataset — this is expected and not a bug. The overall regime patterns, walk-forward leakage check (should always report 0 violations).

---

## Results

| Metric | Dynamic (Regime-Based) | Static 60/40 | Equal-Weight |
|---|---|---|---|
| Annual Return | 9.43% | 9.07% | 14.35% |
| Annual Volatility | 12.12% | 16.74% | 11.67% |
| Sharpe | 0.803 | 0.602 | 1.207 |
| Sortino | 1.170 | 0.879 | 1.760 |
| Max Drawdown | -15.19% | -28.06% | -17.28% |
| Calmar | 0.620 | 0.323 | 0.830 |
| Avg. Annual Turnover | 8.56x | ~0 | ~0 |

**Reading these results honestly:** the Dynamic regime strategy clearly beats the classic Static 60/40 benchmark it was most directly built to improve on — higher Sharpe (0.803 vs 0.602), and a much shallower max drawdown (-15.19% vs -28.06%). That's the core claim of the project holding up: reacting to detected regimes protected capital better than a fixed split did, especially during drawdowns.
 
It does **not** beat the naive Equal-Weight portfolio, though, which posts a higher return, higher Sharpe/Sortino/Calmar, and even slightly lower volatility. Two honest reasons for this, worth stating rather than hiding:
1. **Turnover cost is real and large.** 8.56x average annual turnover at 7.5 bps per unit of turnover adds up to a meaningful, continuous drag that a never-rebalanced-away-from-target portfolio simply doesn't pay.
2. A 3-asset, 1-regime-detector setup has limited room to add value over equal-weight, which is itself a well-known hard-to-beat baseline in the asset allocation.
   
This is a legitimate finding, not a failure of the code — a regime-switching strategy that costs more to run than it earns in edge is exactly the kind of result the transaction-cost modeling was built to surface,.
 
**Transition probability matrix** (from the trained HMM, Phase 3, trained on 4,017 HDFC Bank data points):
 
|        | To Bear | To Bull | To Crisis |
|---|---|---|---|
| **From Bear**   | 0.955 | 0.033 | 0.012 |
| **From Bull**   | 0.023 | 0.971 | 0.005 |
| **From Crisis** | 0.012 | 0.012 | 0.976 |
 
Self-transition probabilities: **0.955 / 0.971 / 0.976** — all comfortably above the 0.9 threshold, confirming regimes are genuinely persistent rather than flickering day to day. Raw HMM output produced 122 regime switches over the full history; causal smoothing trimmed that to 119.
 
**Mean regime characteristics:**
 
| Regime | Avg. Return | Avg. Volatility |
|---|---|---|
| Bear | -0.1063 | 1.1816 |
| Bull | 0.1558 | 1.0170 |
| Crisis | 0.1165 | 2.0969 |
 
**Regime-overlaid price chart:** see the **Phase 3** output cell — HDFC Bank price with Bull/Bear/Crisis bands shaded on top.
 
**Walk-forward regime chart & leakage check:** see **Phase 4** output — should print `Leakage violations found: 0`.
 
**Equity curve comparison:** see the final **Phase 5** output cell — Dynamic vs. Static 60/40 vs. Equal-Weight, plus the allocation-over-time chart showing how the strategy shifts between assets as the detected regime changes.
 
