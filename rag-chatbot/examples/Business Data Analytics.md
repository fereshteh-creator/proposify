# Business Data Analytics: Beyond Alpha: Using Business KPIs to Drive Algorithmic Trading Decisions By Swarajkumar Gawali

## Hypotheses

**H1**  
Incorporating business KPIs into the evaluation of quantitative trading strategies improves the long-term Sharpe ratio compared to strategies evaluated solely on price and volume indicators.

**H2**  
KPI-aligned strategies demonstrate lower drawdowns and higher strategy persistence in dynamic market environments.

**H3**  
KPI-based scoring frameworks can detect business-model-related risk exposures earlier than conventional quantitative models.

---

## Design Plan

### Study type

Observational study – Data is collected from study subjects that are not randomly assigned to a treatment. This includes surveys, “natural experiments,” and regression discontinuity designs.

### Blinding

No blinding is involved in this study.

### Study design

This study is conceptual and framework-driven. It uses historical financial datasets and simulated KPI overlays to test various strategy scoring mechanisms. The evaluation framework includes:

- Backtesting baseline quant models (momentum, mean-reversion)
- Overlaying firm-level KPIs as additional filters
- Measuring performance impact: Sharpe ratio, drawdowns, strategy decay
- Benchmarking vs traditional factor models (Fama-French, etc.)

The study does not use primary surveys or human participants.

---

## Sampling Plan

### Existing data

Registration prior to accessing the data.

### Data collection procedures

- Publicly available financial databases (e.g., Yahoo Finance, Quandl)
- Simulated KPI datasets based on SaaS benchmarks (for testing)
- Secondary research from firm-level financial statements

All data used is either publicly available or synthetic and non-sensitive.

### Sample size

This study does not involve human participants or biological sampling. The “sampling plan” refers to the simulated selection of data points for backtesting purposes.

- Simulated datasets include financial time series (stock prices, volume) and synthetic business KPIs (e.g., CAC, churn rate).
- A purposive sampling logic is applied to select companies from specific sectors (e.g., SaaS, Fintech) with KPI relevance.
- Time periods are selected based on market regimes: pre-2020 (stable), 2020–21 (volatile), post-2022 (adaptive phase).
- For robustness, random sampling of date ranges and cross-validation subsets is applied to test strategy generalizability.

This is a theoretical modeling study. No participants or human subjects are sampled.

---

## Variables

### Manipulated variables

The study is based on simulated strategy modeling and does not involve traditional experimental manipulation of human subjects. However, the following variables are systematically manipulated to study their effect on strategy performance:

1. **KPI score weights (\(w_1, w_2, w_3\))**

   - The composite KPI score is calculated using weighted KPIs (e.g., NRR, churn, CAC).
   - Different weight combinations are tested to observe the sensitivity of strategy ranking.

2. **Signal lag window**

   - KPI signals are intentionally lagged by varying periods (e.g., 0, 30, 60, 90 days) to study the timing impact on strategy effectiveness.

3. **Drawdown penalty in composite score**

   - The penalty applied to drawdown in the composite strategy score formula is modified to observe how risk adjustment affects final strategy rankings.

4. **Persistence thresholds**
   - Definitions of “persistent strategy” are altered (e.g., top 25% for 2 vs. 3 periods) to examine model robustness.

### Measured variables

- Approximately 10–15 quantitative strategies (e.g., momentum, mean reversion, factor models) are tested.
- Each strategy is evaluated across ~5 years of daily historical price data (~1,250 trading days).
- Simulated business KPIs (such as CAC, churn, NRR) are generated for 50 synthetic firms.
- The total dataset includes ~50,000+ data points combining price signals and KPI overlays.

The sample size is chosen to balance simulation complexity with interpretability. No formal power analysis is required, as the study is exploratory and simulation-driven.

### Indices

The following market indices are used for benchmarking:

- **S&P 500 Index** – market return proxy
- **Nasdaq Composite** – technology-focused firms
- **Russell 2000** – small-cap benchmark
- **Equal-weighted strategy portfolio** – custom performance baseline

---

## Analysis Plan

### Statistical models

The analysis evaluates quantitative trading strategies using both financial metrics and business KPI overlays.

1. **Strategy backtesting**

   - Historical price data is used to backtest 10–15 quant strategies (e.g., momentum, mean reversion).
   - Each strategy is executed across ~1,250 trading days.
   - Performance metrics: Sharpe ratio, max drawdown, volatility, alpha decay.

2. **KPI overlay and scoring**

   - Simulated KPI datasets are overlaid on strategy results.
   - A composite KPI score is calculated using weighted averages of NRR, CAC, and churn.
   - Strategies are re-ranked using a scoring formula:  
     \[
     \text{Composite Score} = w_1 \cdot \text{Sharpe} + w_2 \cdot \text{KPI Score} - w_3 \cdot \text{Drawdown}
     \]

3. **Statistical analysis**

   - Correlation analysis between KPI scores and Sharpe ratio to test H1.
   - Regression analysis to estimate impact of KPI alignment on risk-adjusted returns.
   - Principal Component Analysis (PCA) to reduce multi-collinearity between KPIs.

4. **Strategy stability tests**

   - Alpha decay tracking across rebalance periods to test H2.
   - Strategy persistence measured using quantile tracking over time.

5. **Tools used**
   - Python libraries: `pandas`, `numpy`, `backtrader`, `statsmodels`
   - Monte Carlo simulation for drawdown and risk forecasting
   - Visualizations via Matplotlib/Seaborn for result interpretation

The analysis is exploratory and aims to identify patterns in KPI-aligned strategy behavior.

### Transformations

Several data transformations are applied to standardize and prepare financial and KPI variables:

1. **Log returns**  
   \[
   r*t = \ln\left(\frac{P_t}{P*{t-1}}\right)
   \]

2. **Z-score standardization of KPIs**  
   \[
   Z = \frac{X - \mu}{\sigma}
   \]

3. **Composite KPI score construction**  
   \[
   \text{KPI Score} = w_1 \cdot \text{NRR} - w_2 \cdot \text{Churn} - w_3 \cdot \text{CAC}
   \]

4. **Rank normalization**

   - Strategy performance metrics are rank-normalized across all strategies per month/quarter.

5. **Volatility scaling**

   - Return series are scaled by rolling standard deviation to account for regime shifts.

6. **Lag transformations**  
   \[
   \text{Lagged KPI}_t = \text{KPI}_{t-n}
   \]

These transformations align different data types and improve comparability in scoring and regression analysis.

### Inference criteria

Inference is based on both statistical significance and performance thresholds:

1. **P-value thresholds** – For regression analyses, \(p < 0.05\) is considered statistically significant.
2. **Sharpe ratio benchmarks** – Post-KPI Sharpe ratio increase ≥ 0.25 over baseline → “KPI-enhanced”.
3. **Max drawdown threshold** – Drawdown reduction ≥ 20% relative to baseline → effective risk filtering.
4. **Alpha decay rate comparison** – Decay rate improvement ≥ 10% across rebalance periods → increased robustness.
5. **Persistence quantile** – Strategies in the top 25% quantile across ≥ 3 consecutive evaluation periods → “persistent”.
6. **Composite score threshold** – Minimum composite score improvement of 15% after KPI overlay → strategy labeled as improved.

Where applicable, confidence intervals are reported to assess robustness.

### Exploratory analysis

Exploratory analyses include:

1. **Market regime analysis** – Strategies examined across different market periods (pre-2020, pandemic, post-2022) to observe KPI sensitivity under varying macro conditions.
2. **Unplanned KPI combinations** – Additional KPI pairs (e.g., CAC vs NRR, churn vs ARPU) tested for correlation and effect on strategy ranking.
3. **Heatmaps of strategy score volatility** – Seaborn heatmaps used to explore how KPI score volatility influences performance fluctuations.
4. **Lag sensitivity grids** – KPI signal delay tested at various intervals (15, 30, 60, 90 days) to identify optimal timing windows.
5. **PCA component impact** – PCA used to reduce KPI dimensions and examine component weights for interpretability.

These exploratory insights refine the framework but are not part of formal hypothesis testing.

---

## Other

This work was published by Swarajkumar Gawali, a student at Manipal Academy of Higher Education in the Business Analytics department, holding CFA and Six Sigma Green Belt certifications.
