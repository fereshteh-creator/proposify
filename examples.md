# Applied Artificial Intelligence: Effects of Artificial Intelligence Tools on Reading Comprehension

## Hypotheses

- For poor performing participants (determined based on test accuracy for the control passage [no AI tool access]), AI tools will improve reading comprehension as measured by accuracy.
- For better performing participants on the control passage, these tools will worsen reading comprehension as measured by accuracy.
- The best tools for poor performing participants will be the AI Q&A tutor and AI-generated outline with respect to improving accuracy, and use of the summary tool will not impact accuracy but will reduce time on passage.

---

## Design Plan

### Study type

Experiment – A researcher randomly assigns treatments to study subjects, including field or lab experiments. This is also known as an intervention experiment and includes randomized controlled trials.

### Blinding

No blinding is involved in this study.

Personnel who interact directly with the study subjects (either human or non-human subjects) will not be aware of the assigned treatments (commonly known as “double blind”).

### Study design

This is a within-subject cross-over study. Participants will each receive five randomly chosen reading passages drawn from the American College Testing (ACT) test and presented in a random order. Each of the four AI tools and one control (i.e., no AI tool) conditions will be randomly assigned to each of the passages, resulting in a unique tool–passage pair for that participant.

Participants are given instructions on how to use the tools when not already apparent. They will also initially be given a shorter tutorial passage to introduce them to the mechanics of the test and portal. After reading each passage and using the relevant AI tool, they will take the multiple-choice questions for that passage as drawn from the ACT. After completing these questions, they will be asked to rate the tool they used in terms of effectiveness and enjoyment. Reloading or exiting out of the page will result in them getting a new passage when they return.

### Randomization

Each participant will get every condition. The order of these conditions is randomized for each participant.

---

## Sampling Plan

### Existing data

Registration prior to creation of data.

### Data collection procedures

Data will be collected through an online test portal. Participants will be fluent in English, currently living in the US, and between the ages of 18 (or the age of majority in their respective state, if different) and 22 years old. Study candidates are identified through the Prolific platform and will be compensated at a rate of \$10 an hour with a \$6 bonus for top performing participants (with participants aware of this bonus opportunity). Participants who don't meet the quality control metrics will not be compensated. Given the online recruitment approach, data collection is expected to conclude within 1–2 months of study initiation.

### Sample size

Our target sample size is 200 participants, with replacement based on participants failing to meet quality control metrics to result in a final sample of approximately 200 people.

### Sample size rationale

As our hypotheses, which are based on a 16-person pilot study we conducted, involve dividing the sample at the median of control passage performance, our total _N_ was derived from power calculations on each of the subgroups (i.e., poor vs better performing groups). A power calculation with a 2-sided alpha of 0.05 and 80% power for a Cohen’s _d_ = 0.3 showed that approximately 90 participants will be required.

Given that dividing at the pre-specified median cutoff may favor one subgroup over the other in terms of size (due to the limited levels of accuracy possible for the 10 multiple choice questions on the control passage), a total sample size of 200 was deemed appropriate. The _d_ = 0.3 effect size was chosen as anything below that would not have practical significance in terms of demonstration that AI tools improved or worsened performance.

---

## Variables

### Manipulated variables

The primary study manipulation is the use of an AI tool, compared to a control condition where no tools are used (which corresponds to the typical way ACT passages are read and tested on). All AI tools are built upon GPT-4. The four tools are:

- a Q&A tutor chatbot
- a Socratic method discussion chatbot
- an AI-generated summary of the passage
- an AI-generated collapsible/expandable outline for the passage

### Measured variables

- Test accuracy for each comprehension test taken for each passage.
- Time spent on the passage until participants decide to click the **“Take Quiz”** button (indicating they feel they understand the passage).
- Perceived effectiveness and enjoyment of the tools, measured via ratings after each individual passage test.

Additional information collected about participants includes age, education, demographics, SAT/ACT score (if remembered), income, and other factors.

---

## Analysis Plan

### Statistical models

The comparison condition for all AI tools is the control condition where participants read a passage and take a comprehension test about that passage without any AI tools. To test the hypotheses, conditions will be compared with the control using pairwise _t_-tests.

The conditions analyzed are the test scores in the following cases:

- control with no AI tools
- AI-generated summary
- Socratic method discussion chatbot
- Q&A tutor
- AI-generated passage outline

Analyses will be done separately for the poor and better performing groups (based on the control passage performance). This split will be done at the study median for the control passage, attempting to keep the group sizes as similar as possible.

The primary analysis will use accuracy as the dependent variable, with secondary analyses using time spent on the passage as well as user ratings. The null hypothesis is that all group means are equal.

### Transformations

Distributions will be examined for each dependent variable to ensure appropriateness for parametric statistical testing. If needed, variables will be transformed and/or non-parametric tests will be used.

### Inference criteria

A threshold of _p_ < 0.05 will be used to establish statistical significance.

### Data exclusion

The complete data for any participant who didn't pass the quality control metrics will be excluded (participants using the 1.5 IQR rule for time spent on any passage or those who scored less than 30% on two or more passages, denoting below-chance performance). No awareness checks will be used.

### Missing data

There will be no missing data, as participants must provide all data for every case before completing the test. Participants who withdraw from the task before completion will have their data completely excluded.

### Exploratory analysis

Exploratory analyses may examine how participant demographic and socioeconomic background, SAT/ACT score, previous AI experience, and education level impact how the AI tools affect them and their perceived efficacy. Additionally, analyses will investigate how the different AI tools affect time taken with respect to test accuracy.

---

## Other

We developed these AI tools using the OpenAI GPT LLM. We used the GPT-4 Assistants API for the Socratic Method and Q&A Tutor chatbots. We used the GPT-4 Chat Completions API for the generated outline and the generated summary. We sourced the passages and questions from validated, official ACT practice tests from multiple years.

# Innovation and Entrepreneurship: Thriving among Entrepreneurs: Utilization Hypotheses

## Hypotheses

Our primary hypotheses focus on the overall use of any mental health service. For supplemental analyses, we will provide detail on the specific services used and how those correspond to the E10 indices.

**H5-1**  
Controlling for E10 total scores at baseline, entrepreneurs who request (and are given access to) mental health resources by selecting 1 or 2 on item 11 will have greater likelihood of mental health services utilization during the 6-month follow-up period compared to those who do not use that link.

**H5-2a**  
Controlling for E10 baseline total scores, entrepreneurs who seek help by selecting 1 or 2 on item 11 of the E10 at baseline will have better mental health at follow-up. Four parallel regression models will be computed to assess outcomes (the E10 total score, the E10 wellbeing and emotional distress dimensions, and the DSM-5 Cross-cutting Symptom Measure).

**H5-2b**  
Controlling for baseline business success or failure and baseline E10 scores, entrepreneurs who seek help (using item 11 of the E10) will have better business outcomes (aggregate success/failure score) during the 6-month follow-up period, as compared with those who do not seek help.

In parallel, those who report utilization of mental health services during the follow-up period (aside from the E10) will report better mental health and business outcomes, controlling for baseline. To assess this, analyses will conjointly consider the E10 item 11 and endorsement of other treatment utilization in modeling mental health and business outcomes.

**H5-3**  
These effects of the E10 scores on mental health and business outcomes will be particularly strong among entrepreneurs at higher mental health (E10) risk, as assessed using the interaction of E10 scores and help-seeking. Help-seeking will be defined as accessing mental health resources by selecting 1 or 2 on E10 item 11 at baseline or by reporting use of mental health services in the prior six months on the follow-up study.

To test this, we will add variables to the regression models noted in H5-2, constructing four parallel hierarchical linear models. These will regress the effects of E10 baseline scores, mental health care utilization, and the interaction of E10 scores × health care utilization on mental health (the E10 total score, E10 wellbeing and emotional distress dimensions, and the DSM-5 Cross-cutting Symptom Measure) and business outcomes (aggregate success/failure index, controlling for baseline business success/failure index) at follow-up.

We anticipate that among those with high E10 scores, help-seeking will not have robust predictive power. Among those with low E10 scores, those who seek help will have more positive mental health and business outcomes than those who do not seek help.

**H5-4**  
E10 scores at baseline will inversely correlate significantly with utilization of mental health services during the 6-month follow-up, such that individuals with lower E10 scores at baseline will be more likely to endorse utilizing mental health services during the 6-month follow-up period than individuals with higher E10 scores at baseline.

**H5-5**  
The relationship between perceived need/interest as indexed by E10 item 11 and service utilization in the six-month follow-up period will be moderated by the degree of access barriers (GUPI), as reflected in an interaction term of E10 baseline scores and GUPI barrier scores. We expect that the positive relationship between E10 baseline scores and service utilization in the 6-month follow-up period will be diminished among individuals who experience barriers to care.

**H5-6**  
Scores at or above the clinical threshold (4+) on the PC-PTSD at baseline will be associated with (1) interest in mental health resources (E10 question 11) and (2) lower likelihood of accessing mental health services during the 6-month follow-up period compared to individuals who screen negative.

---

## Design Plan

### Study type

Observational study – Data is collected from study subjects that are not randomly assigned to a treatment. This includes surveys, “natural experiments,” and regression discontinuity designs.

### Blinding

No blinding is involved in this study.

### Study design

This is an observational, longitudinal study. Participants will complete measures of business success/failure, mental health, and interest in treatment (the E10) at baseline. At a 6-month follow-up, we will examine treatment utilization and will repeat key indices of business success/failure and mental health.

---

## Sampling Plan

### Existing data

Registration prior to any human observation of the data.

### Explanation of existing data

We have released our survey for participants to take. We have not yet examined the distribution of any of these key variables.

### Data collection procedures

As with the parent study, VCs and others invested in the mental health of entrepreneurs will be asked to distribute survey invitations to entrepreneurs. Entrepreneurs will be given a link to take the baseline and follow-up surveys online.

### Sample size

We will recruit as many entrepreneurs as we are able to reach and can pay for their participation.

### Sample size rationale

Sample size will be limited by our ability to identify participants and by available funding to reimburse participants for their participation.

---

## Variables

### Manipulated variables

Participants who request information about mental health services will be provided with links for how to identify practitioners, and for online materials and apps, and self-help reading.

### Measured variables

We have detailed indices of the E10, GUPI, and business success/failure in the parent pre-registration.

---

## Analysis Plan

### Statistical models

We will use correlations (for bivariate hypotheses) and multiple regression models (for multivariable and interaction term hypotheses). Variables will be standardized before testing interaction terms.

### Transformations

We will examine distributions for normality. Outlier scores will be considered for validity and potentially removed. Distributions will be transformed if kurtosis is a concern.

### Inference criteria

_p_ < .05, two-tailed tests.

### Data exclusion

Participants who fail attention check items will be excluded from analyses.

### Missing data

Multiple imputation will be conducted if data assumptions of missing at random are met.

### Exploratory analysis

We will conduct supplemental analyses to examine more specific forms of treatment utilization. We will consider gender, age, and country of origin as possible moderators/confounds in effects.

---

## Other

Data cleaning will be conducted in full accordance with the parent study (details also pre-registered here).

# Software Architecture: A Comparative Study of Model Training Time and Cost on AWS, Azure, and GCP using DevOps Automation

## Hypotheses

**H1 (Directional)**  
AWS and GCP will demonstrate faster model training time compared to Azure when using equivalent virtual machine specifications (4 vCPUs, 16 GB RAM) and identical DevOps pipelines.  
→ _Predicted effect:_ AWS and GCP will complete model training in less time than Azure.

**H2 (Directional)**  
The cost of model training on Azure will be higher than on AWS and GCP when measured over the same training duration and instance configuration.  
→ _Predicted effect:_ Azure incurs greater cloud compute cost for equivalent workloads.

---

## Design Plan

### Study type

Observational study – Data is collected from study subjects that are not randomly assigned to a treatment. This includes surveys, “natural experiments,” and regression discontinuity designs.

### Blinding

No blinding is involved in this study.

### Study design

This study employs a within-subject (paired) experimental design to compare model training time, cost, and performance across three major cloud platforms: AWS, Azure, and Google Cloud Platform (GCP). The same Convolutional Neural Network (CNN) model is trained on an identical dataset (CIFAR-10) using consistent DevOps automation pipelines configured for each platform.

**Key design features:**

- **Within-subject design:** The same model training task is executed on each cloud platform. This paired approach controls for model complexity and dataset variability, allowing direct comparison of platform-specific performance metrics.
- **Repeated measures:** Training runs are performed for 30 minutes on each platform to capture variability and ensure statistical reliability of results. Each run is scheduled and executed with consistent resource specifications (e.g., VM instance types: `t3.xlarge` on AWS, `Standard_D4s_v3` on Azure, and `e2-standard-4` on GCP).
- **Counterbalancing:** To mitigate systematic effects from temporal or environmental factors (e.g., time-of-day variations, cloud service load), the order of platform usage is rotated across runs (e.g., AWS → Azure → GCP, Azure → GCP → AWS, GCP → AWS → Azure).
- **Randomization:** The timing and order of training runs are randomized within constraints to avoid bias due to external factors such as network latency or cloud resource availability fluctuations.
- **Outcome measures:** Primary outcomes include total model training time, monetary cost of cloud resource usage during training, and performance metrics such as model accuracy and convergence rate.
- **Automation and reproducibility:** All model training workflows are automated using CI/CD pipelines, ensuring consistent deployment and execution on each platform and supporting reproducibility.

---

## Sampling Plan

### Existing data

Registration prior to creation of data.

### Explanation of existing data

The study does not rely on any previously collected or externally sourced datasets. All data used in the analysis (training time, cost, and accuracy) will be generated in real time during the course of this study through automated training runs on each cloud platform.

To ensure objectivity, no prior analysis has been conducted on the forthcoming data. No summary statistics or performance outcomes have been observed or reviewed in advance. The infrastructure and code for data collection were set up before any training jobs were executed, and logging was configured to begin only when the benchmarking runs started. Analyses will be conducted solely on data generated after the study design was finalized.

### Data collection procedures

Training jobs will be deployed programmatically on AWS, Azure, and GCP using automated DevOps pipelines. Each job will train the same CNN model on the CIFAR-10 dataset using identical training parameters. Jobs will be run on comparable virtual machine types (`t3.xlarge` on AWS, `Standard_D4s_v3` on Azure, and `e2-standard-4` on GCP). Each training run will be logged for training time, resource usage, monetary cost, and final model accuracy.

Each cloud platform will receive at least 10 repeated training runs. The order in which platforms are tested will be randomized and rotated to control for time-based variability. Data collection is expected to take approximately one week, depending on job completion speed and cloud availability.

Only completed training jobs that meet the expected configuration and do not encounter errors (e.g., interruption, timeout, misconfiguration) will be included in the analysis. Failed or incomplete jobs will be excluded. All raw logs and results will be stored securely for analysis.

### Sample size

This study does not involve human or biological participants. Instead, the “sample units” refer to model training sessions conducted on three different cloud platforms: AWS, Azure, and GCP. Each session involves training a CNN model on the CIFAR-10 dataset using PyTorch, with automation handled via DevOps pipelines.

A total of 90 training runs are conducted (30 per cloud provider) across standardized virtual machine instances:

- **AWS:** `t3.xlarge`
- **Azure:** `Standard_D4s_v3`
- **GCP:** `e2-standard-4`

Each training run lasts 30 minutes and is repeated to account for performance variability, ensuring statistical validity in the cost and performance comparisons.

Thus, the effective sample size is 90 independent training sessions, distributed evenly across the three platforms.

---

## Variables

### Manipulated variables

The manipulated variable is the cloud platform used to train the model. This categorical variable has three levels:

- Amazon Web Services (AWS)
- Microsoft Azure
- Google Cloud Platform (GCP)

Each level corresponds to a separate cloud environment where the same CNN model is trained using identical parameters and resource specifications.

### Measured variables

Variables measured fall into three categories:

**Outcome variables (dependent)**

1. **Training time (seconds):** Total duration required to complete model training for each run on a given cloud provider.
2. **Training cost (USD):** Calculated cost per training run based on provider-specific instance pricing and runtime duration.
3. **Performance-to-cost ratio:** Derived metric computed by dividing model accuracy by training cost to assess efficiency.

**Predictor variables (independent)**

- **Cloud platform:** Categorical variable (AWS, Azure, GCP).
- **Instance type:** Specific virtual machine type used (e.g., `t3.xlarge`, `Standard_D4s_v3`, `e2-standard-4`).
- **DevOps pipeline configuration:** Standardized CI/CD pipeline configuration, identical across providers.

**Control variables**

1. **Model architecture:** Fixed CNN architecture trained on CIFAR-10 using PyTorch.
2. **Training duration:** Fixed to 30 minutes per run to ensure uniformity for time-based cost comparison.
3. **Dataset:** CIFAR-10 dataset used consistently for all runs.

---

## Analysis Plan

### Statistical models

The main goal of the analysis is to compare the training time, cost, and model accuracy across the three cloud platforms: AWS, Azure, and GCP. A repeated measures ANOVA will be used, as the same training task is performed multiple times on each platform, allowing paired comparisons.

- **Dependent variables:** Training time, training cost, and model accuracy.
- **Factor:** Cloud platform (three levels: AWS, Azure, GCP).
- **Repeated measure:** Each training run.

If the overall test shows a significant difference, pairwise comparisons between platforms (AWS vs Azure, AWS vs GCP, and Azure vs GCP) will be run with Bonferroni correction to adjust for multiple comparisons.

Before running the tests, assumptions of ANOVA (including sphericity and normality) will be checked. If sphericity is violated, a correction such as Greenhouse–Geisser will be applied.

In addition to the main tests, the relationship between training time and cost on each platform will be explored using correlation analysis. The potential impact of run order on the results will also be examined.

All analyses will be conducted in R or Python, with code and data retained for reproducibility and transparency.

### Transformations

No data transformations are planned for the primary analysis. The dependent variables—training time (seconds), cost (USD), and accuracy (percentage)—will be analyzed in their raw continuous form.

The categorical variable _cloud platform_ (AWS, Azure, GCP) will be dummy coded for any regression-based exploratory analyses, with AWS used as the reference category.

If violations of normality are detected in residuals during model checks, appropriate transformations (e.g., log transformation for skewed time or cost data) may be applied and documented as exploratory.

### Inference criteria

The standard criterion of _p_ < .05 will be used to determine statistical significance for the repeated measures ANOVA and follow-up tests. All tests will be two-tailed.

For pairwise comparisons following a significant ANOVA result, Bonferroni correction will be applied to adjust for multiple comparisons and control the family-wise error rate. If assumptions for ANOVA are violated and alternative models are used, the same _p_ < .05 threshold will apply.

Effect sizes (e.g., partial eta squared) may also be reported alongside _p_-values to provide additional context for practical significance.

### Data exclusion

Each training run will be checked to ensure it completes successfully and that all expected metrics (training time, cost, and accuracy) are recorded. Runs that fail due to infrastructure issues (e.g., instance interruption, misconfiguration, or timeouts) or are missing key data will be excluded from the analysis.

Outliers in training time or cost will initially be retained but flagged and reviewed to determine whether they result from technical anomalies rather than natural variation.

### Missing data

If a training run does not produce complete data for all three key metrics—training time, cost, and accuracy—it will be excluded from the analysis. Only fully completed runs with all required outputs will be included to ensure consistency and comparability across cloud platforms.

### Exploratory analysis

Exploratory analyses will examine relationships between system-level metrics (such as CPU utilization, memory usage, and disk I/O) and the primary outcome measures (training time, cost, and accuracy). Correlations between training time and cost within each platform will also be explored to identify potential efficiency patterns. These analyses will be reported explicitly as exploratory.

---

## Other

This study is part of a larger technical publication titled **“A Comparative Study of Model Training Time and Cost on AWS, Azure, and GCP using DevOps Automation.”** The goal is to evaluate performance and cost-efficiency of cloud platforms when training the same machine learning model using reproducible DevOps pipelines.

All training jobs are executed using fully automated CI/CD workflows to ensure consistency across platforms. Infrastructure-as-Code tools (e.g., Terraform) are used to provision identical environments, and logging is standardized across platforms to capture comparable metrics.

There is no prior publication using this specific dataset. The study design and automation pipeline are inspired by best practices in MLOps and cloud benchmarking. All code, configurations, and data will be made publicly available for transparency and replication.

No conflicts of interest or external funding are associated with this study.

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

# Project management and agility: Development and Validation of a Measurement Scale for the Agility of a Software Development Project.

## Hypotheses

The theoretically defined structure of the agility construct fits the data well.

---

## Design Plan

### Study type

Other.

### Blinding

No blinding is involved in this study.

### Study design

The study examines whether the theoretically derived measurement model of software project agility can be confirmed based on collected data (see _Agile_Score.png_).

The key question is:

> Can the theoretically derived measurement model of software project agility be confirmed based on collected data?

---

## Sampling Plan

### Existing data

Registration prior to creation of data.

### Data collection procedures

Snowball sampling will be used for data collection.

To motivate participation, the survey is designed so that participants receive an estimated agility level of their projects at the end of the survey.

### Sample size

There is no general and established procedure for determining the minimum required sample size to conduct a confirmatory factor analysis (Wolf et al., 2013). Instead, there are varying _a priori_ recommendations in the literature (Hinderks & Thomaschewski, 2018).

For this study, a **minimum sample size of 200** was set. This is based on the recommendations of Kline (1998), Loehlin (1998), and Boomsma and Hoogland (2001).

However:

- Klopp (2010) additionally recommends a consistent commonality of \(h^2 > 0.50\) for each item at a sample size of 200.
- Guadagnoli & Velicer (1988) consider a sample size of 150 sufficient for a factor loading of at least 0.60 with at least four items per factor.
- With fewer than four factors and smaller factor loadings, a sample size of 300 would be required.

In view of the fact that 6 of the 30 agile practices are operationalized with only 3 items each, a **sample size of 300** is therefore aimed at.

---

## Variables

### Manipulated variables

None (no experimental manipulation).

### Measured variables

See _Agile_Items.pdf_ for the detailed list of items measuring the 30 agile practices.

---

## Analysis Plan

### Statistical models

The overall study model can be divided into two models based on the specification type of the indicators:

1. **First-order measurement model (reflective model)**

   - The 30 agile practices with their respective items.
   - This model will be examined using **confirmatory factor analysis (CFA)**.
   - Goal: to examine how well the formulated items fit their respective factors (agile practices).

2. **Second-order measurement model (formative model)**
   - The agile construct represented by the 30 agile practices as formative indicators.
   - This model will be analyzed using **partial least squares (PLS) regression analysis**.
   - Goal: to gain insight into whether, and how well, the selected practices actually fit and form the overall agile construct.

The analysis of the overall model will be conducted sequentially: first the reflective CFA, then the formative PLS model.

---

## Other

None specified.

# Sustainable Business: The Role of Information in Promoting Environmental Concern and Attitudes, Pro-environmental Behavior and Consciousness of Sustainable Consumption in Higher Education

## Hypotheses

**H1**  
The provision of information on sustainability positively influences students' environmental concern and attitudes.

**H2**  
The provision of information on sustainability positively influences students' pro-environmental behavior.

**H3**  
The provision of information on sustainability positively influences students' consciousness of sustainable consumption.

**H4a**  
Students' actual knowledge gain realized through the provision of information (i.e., the master's course) positively moderates the influence of the provision of information on students' environmental concern and attitudes; that is, the effect of the provision of information on environmental concern and attitudes is stronger for students with higher realized knowledge gain.

**H4b**  
Students' actual knowledge gain realized through the provision of information (i.e., the master's course) positively moderates the influence of the provision of information on students' pro-environmental behavior; that is, the effect of the provision of information on pro-environmental behavior is stronger for students with higher realized knowledge gain.

**H4c**  
Students' actual knowledge gain realized through the provision of information (i.e., the master's course) positively moderates the influence of the provision of information on students' consciousness of sustainable consumption; that is, the effect of the provision of information on consciousness of sustainable consumption is stronger for students with higher realized knowledge gain.

**H5a**  
Students' commitment (i.e., average weekly time spent studying the content of the master's lecture) positively moderates the influence of the provision of information on students' environmental concern and attitudes; that is, the effect of the provision of information on environmental concern and attitudes is stronger for students with higher commitment.

**H5b**  
Students' commitment (i.e., average weekly time spent studying the content of the master's lecture) positively moderates the influence of the provision of information on students' pro-environmental behavior; that is, the effect of the provision of information on pro-environmental behavior is stronger for students with higher commitment.

**H5c**  
Students' commitment (i.e., average weekly time spent studying the content of the master's lecture) positively moderates the influence of the provision of information on students' consciousness of sustainable consumption; that is, the effect of the provision of information on consciousness of sustainable consumption is stronger for students with higher commitment.

---

## Design Plan

### Study type

Observational study – Data is collected from study subjects that are not randomly assigned to a treatment. This includes surveys, “natural experiments,” and regression discontinuity designs.

### Blinding

No blinding is involved in this study.

### Study design

The study is based on a one-group pretest (wave I; T1)–posttest (wave II; T2) experimental design, with data collected via two survey waves.

- **T1 (wave 1):** Marks the beginning of the master's lecture _Sustainable Management_ at the University of Osnabrück during the summer semester 2022, when students are presumed to possess little or no prior knowledge about sustainability.
- **T2 (wave 2):** Marks the end of the master's course _Sustainable Management_, where students have attended the course for almost a whole academic semester (i.e., 8–12 weeks).

The actual time interval depends on the time between participation in the two survey waves (T1, T2), which may differ between students. However, a minimum time interval of 8 weeks (i.e., 75% of the academic semester) between wave 1 and wave 2 is guaranteed.

---

## Sampling Plan

### Existing data

Registration prior to creation of data.

### Data collection procedures

Participants will be recruited through advertisements in the master's lecture _Sustainable Management_ at the German University of Osnabrück during the summer semester 2022.

- Participants will be compensated with 4 bonus points (in addition to 120 regularly achievable points), which will be credited if they pass the exam (min. GPA 1.0).
- Participation in both survey waves (T1, T2) is required to receive bonus points.
- Participants must be enrolled in the master's course _Sustainable Management_ in the summer semester 2022 at the University of Osnabrück.

Data collection will be conducted via the online survey system _LimeSurvey_:

- **T1:** At the beginning of lectures (28 March–17 April 2022).
- **T2:** At the end of lectures (13–30 June 2022).

### Sample size

Based on experience from comparable lectures and the same lecture in the winter semester 2020/21, approximately 100–120 students are expected to attend the course in the summer semester 2022.

- Target sample size: **80 participants**.
- The aim is to recruit at least 100 participants, assuming potential invalid responses and incomplete questionnaires.

### Stopping rule

Data collection will only take place during:

- **T1:** 28 March–17 April 2022
- **T2:** 13–30 June 2022

No interim stopping rule is applied.

---

## Variables

### Manipulated variables

No variables will be experimentally manipulated in this study.

### Measured variables

Unless otherwise stated (\*), all variables are collected in both wave 1 (T1) and wave 2 (T2).

1. **Actual knowledge gain (objective and subjective)**

   - **Objective knowledge:**  
     Geiger et al.’s (2019) Environmental Knowledge Test (EKT), comprising 33 single-choice questions covering:
     - Basic Ecology
     - Climate
     - Resources
     - Consumption Behavior
     - Society/Politics
     - Economy
     - Environmental Contamination
   - **Subjective knowledge:**  
     One self-report item:  
     _“How great would you rate your knowledge in the area of sustainability at the present time?”_  
     (5-point Likert scale: _very low_ to _very high_).
   - Knowledge gain is operationalized as the difference between T1 and T2 (delta knowledge) in both objective and subjective measures.

2. **Environmental consciousness**

   - Schleyer-Lindenmann et al.’s (2018) German validated version of the New Ecological Paradigm (NEP) scale by Dunlap et al. (2000).
   - 15 items covering:
     - Limits of Growth
     - Antianthropocentrism
     - Fragility of Balance
     - Rejection of Exemptionalism
     - Ecocrisis
   - Response scale: 1–6 (1 = _I totally disagree_, 6 = _I totally agree_).

3. **Pro-environmental behavior**

   - Geiger et al.’s (2019) Short Impact-Based Pro-environmental Behavior Scale (SIBS), 18 items.
   - Response scale: 0–4 (0 = _never_, 4 = _always_).
   - Two dichotomous electricity items are combined into one 5-point scale:
     - neither item affirmed = 0
     - renewable energy provider affirmed = 2
     - own solar panel = 3
     - both items affirmed = 4
   - Car use is measured with three questions (ownership, gasoline usage/fuel type, annual driving distance in km) and combined into a 5-point index of car use.

4. **Consciousness of sustainable consumption**

   - Short version of Balderjahn et al.’s (2013) Consciousness for Sustainable Consumption (CSC) scale, 9 items.
   - Response scale: 1–5 (1 = _I totally disagree_, 5 = _I totally agree_).
   - Captures:
     - Ecologically and socially sustainable consumption
     - Voluntary simplicity

5. **Demographic and situational variables\***  
   Collected to control for relevant factors associated with environmental concern, pro-environmental behavior, and sustainable consumption:

   - (a) Sex
   - (b) Age
   - (c) Monthly disposable income (7 categories: _< 500 €_ to _> 3000 €_)
   - (d) Interest in the topic of sustainability (1–5; 1 = _very low_, 5 = _very high_)
   - (e) Students' commitment – average weekly time spent studying the content of the course in addition to attendance (numeric)
   - (f) Number of household members (numeric)

   \* Demographic and situational variables are collected in T1 only, with the exception of (e) commitment, which can only be meaningfully answered retrospectively.

---

## Analysis Plan

### Statistical models

Given the one-group pretest–posttest design, the study analyzes the causal effects of the provision of information (the course) on:

- (a) Environmental consciousness
- (b) Pro-environmental behavior
- (c) Consciousness for sustainable consumption

For hypothesis testing:

- A **structural equation model (SEM)** will be specified, regressing the latent constructs (a)–(c) on the intervention of providing information on sustainability, operationalized as a dummy variable:

  - wave I (T1) = 0
  - wave II (T2) = 1

- Moderators:

  - Students’ realized knowledge gain (H4a–H4c).
  - Students’ commitment (average weekly study time; H5a–H5c).

  Both moderators will be specified as predictors of the latent constructs (a)–(c).

- Control variables:
  - Age
  - Gender
  - Monthly disposable income
  - Interest in the topic of sustainability
  - Number of household members

These variables will be included in the SEM as additional predictors of the three latent constructs (a)–(c).

---

## Data Exclusion

**Inclusion criteria**

- Participants must be attending the master's course _Sustainable Management_ during the summer semester 2022 at the German University of Osnabrück.

**Exclusion criteria**

- Participants showing no variance in response patterns (SD = 0).
- Participants who fail the attention check item:  
  _“This is an attention check. Please click on ‘4’ here.”_
- Participants who do not meet a reasonable minimum participation time (3 minutes).

---

## Missing Data

- Within-questionnaire missing data is minimized by requiring responses to proceed.
- Participants who complete only T1 but not T2 will be included in cross-sectional analyses only.

---

## Exploratory Analysis

None specified.

---

## Other

None specified.

# Wirtschafts- und Digitalrecht (Business & Digital Law): Digital and Physical Surveillance in Public Perception

## Hypotheses

We will test the following confirmatory hypotheses:

**H1**  
Participants will perceive differences in how much information digital and physical surveillance can gather.

**H2**  
Participants will rate the invasiveness of digital and physical surveillance differently.

**H3**  
Acceptability differs between overt (informed) and covert (uninformed) surveillance.

**H4**  
The perceived invasiveness of a surveillance type is associated with its acceptability.

**H5**  
The perceived effectiveness of a surveillance method is associated with its acceptability.

**H6–H8**  
Trust in government, law enforcement, and companies is positively associated with acceptance of surveillance measures.

**H9–H11**  
Trust in government, law enforcement, and companies is negatively associated with concern about surveillance misuse.

**H12**  
Perceived levels of local crime and organized crime are associated with surveillance acceptance.

---

## Design Plan

### Study type

Observational Study – Data is collected from study subjects that are not randomly assigned to a treatment. This includes surveys, “natural experiments,” and regression discontinuity designs.

### Blinding

No blinding is involved in this study.

### Study design

This is a cross-sectional, between-subjects survey study conducted using Qualtrics as the survey platform and Prolific for recruiting a representative sample of 1,000 U.S. adults.

Participants are randomly assigned to one of two groups that differ in the order of question blocks:

- One group is first presented with descriptions of surveillance methods, then evaluates their acceptability.
- The other group is first asked about the acceptability of surveillance methods, then sees the descriptions.

Some follow-up questions are conditionally shown (e.g., questions about minority status or justifications for choosing a more invasive surveillance method).

The study includes bot detection and at least one attention check to ensure data quality. Sensitive questions include a “Prefer not to answer” option to protect participant privacy.

---

## Randomization

Randomization is implemented at multiple levels to reduce order effects and formulation bias:

1. **Group assignment (between-subjects)**  
   Participants are randomly assigned (simple randomization) to one of two groups that differ in the order of question blocks. These options are presented with equal frequency.

2. **Question order randomization (within-subjects)**  
   For categorical response options, answer choices are randomized unless there is a natural order (e.g., Likert scales from “strongly agree” to “strongly disagree”).

3. **Wording direction (formulation bias control)**  
   Directionally sensitive comparative questions (e.g., “Digital surveillance is more/equally/less invasive than physical surveillance”) are presented in either direction at random, using simple randomization at the question level.

All randomizations are implemented through Qualtrics’ built-in randomization tools, which use simple (non-stratified) random assignment without replacement.

---

## Sampling Plan

### Existing data

Registration prior to creation of data.

### Data collection procedures

Data will be collected through an online survey hosted on Qualtrics, with participants recruited via Prolific.

- Target sample: 1,000 U.S.-based adults.
- Sampling: Representative of the U.S. general population in terms of age, gender, and ethnicity, using Prolific’s representative sampling tools.

**Inclusion criteria**

- Age 18 or older
- Consent given
- Currently residing in the United States
- Fluent in English
- Passed Prolific’s pre-screening for representative eligibility

**Exclusion criteria**

- Participants who fail the attention check
- Participants flagged as bots or providing incomplete responses
- Participants who complete the survey in less than 1/3 of the median completion time

The survey takes approximately 10 minutes to complete. Each participant is compensated at a fair hourly rate, in line with or exceeding Prolific’s minimum pay guidelines.

Recruitment and data collection are expected to be completed in less than a week, depending on participant flow and quality screening. All data will be reviewed after collection to exclude invalid responses before analysis.

### Sample size

The study analyzes data from 1,000 individual participants, all adults residing in the United States. Each participant completes the survey once, providing a single data point per variable measured.

### Sample size rationale

This sample size was determined using G\*Power to provide sufficient power (0.80) to detect small to medium effect sizes (e.g., r = 0.1, r = 0.3) in key relationships (such as between trust in institutions and acceptance of surveillance methods) at the standard α = .05. A margin was added to account for answers that may be excluded from analysis.

---

## Variables

### Manipulated variables

This is an observational study with no experimentally manipulated variables.

### Measured variables

- **Perceived Information Gathering Capabilities**  
  Categorical variable with 3 levels:

  - Digital more
  - Physical more
  - Both equally

- **Perceived Invasiveness**  
  Categorical variable collapsed into 3 levels:

  - Digital more invasive
  - Physical more invasive
  - Equal

- **Acceptability Ratings**  
  4 items on a 5-point Likert scale:

  - Covert Digital
  - Overt Digital
  - Covert Physical
  - Overt Physical

  These are used to compute mean scores for:

  - Digital surveillance acceptance
  - Physical surveillance acceptance
  - Overt surveillance acceptance
  - Covert surveillance acceptance
  - Overall surveillance acceptance

- **Effectiveness Ratings**  
  4 items (matching the four surveillance types above) on a 5-point Likert scale, measuring perceived effectiveness.

- **Trust Variables**

  - Trust in government
  - Trust in law enforcement
  - Trust in companies  
    Each rated on a 5-point Likert scale.

- **Concern About Misuse**  
  Average of perceived likelihood of abuse of digital and physical surveillance, each on a 5-point Likert scale.

- **Perceived Crime and Organized Crime**  
  Two separate ordinal 5-point items measuring perceived levels of local crime and organized crime.

---

## Analysis Plan

### Statistical models

**H1**  
Chi-square goodness-of-fit test to assess whether the distribution of responses for Perceived Information Gathering Capabilities deviates significantly from a uniform distribution.

**H2**  
Chi-square goodness-of-fit test to determine whether responses on the Perceived Invasiveness variable are evenly distributed across the three categories.

**H3**  
Compute mean Acceptability Ratings for overt and covert surveillance (averaging the two items in each category) and compare them using a paired t-test.

**H4**  
For each participant, calculate a difference score between their mean Acceptability Ratings for digital vs. physical surveillance. Use a one-way ANOVA (or Kruskal–Wallis test, if assumptions are violated) to test whether this difference score varies across the three groups defined by Perceived Invasiveness.

**H5**  
For each of the four surveillance types, calculate a Spearman correlation between the corresponding Effectiveness Rating and Acceptability Rating. Apply Holm–Bonferroni correction across these 4 correlations to control for multiple comparisons.

**H6–H8**  
Compute Spearman correlations between each of the three Trust Variables (government, law enforcement, companies) and the mean of the four Acceptability Ratings. Apply Holm–Bonferroni correction across these 3 correlations.

**H9–H11**  
Compute Spearman correlations between each of the three Trust Variables and the Concern About Misuse variable. Apply Holm–Bonferroni correction across these 3 correlations.

**H12**  
Compute two Spearman correlations:

- Perceived Crime vs. mean Acceptability Ratings
- Perceived Organized Crime vs. mean Acceptability Ratings

Apply Holm–Bonferroni correction across these 2 correlations.

---

## Transformations

None specified.

---

## Inference Criteria

None specified beyond the use of standard inferential tests described above (chi-square tests, t-tests, ANOVA/Kruskal–Wallis, Spearman correlations with Holm–Bonferroni corrections).

---

## Data Exclusion

See exclusion criteria under **Data collection procedures** (failed attention checks, bots, incomplete responses, and too-fast completion times).

---

## Missing Data

In the survey, answers to all questions are mandatory. The survey cannot be completed unless all required data are provided.

---

## Exploratory Analysis

In addition to the preregistered confirmatory hypotheses, the following exploratory (hypothesis-guided) questions may be examined. Any results will be clearly labeled as exploratory:

- **EH1**: Participants will perceive differences in how common digital and physical surveillance are.
- **EH2**: Preferences for surveillance method (digital vs. physical) are associated with perceived invasiveness.
- **EH3**: The perceived likelihood of being surveilled is associated with acceptance of surveillance practices.
- **EH4**: Belief that surveillance has increased, decreased, or stayed the same is associated with the acceptance of surveillance.
- **EH5**: Participants who both believe surveillance has increased and say they are worried about this will differ in their acceptance score compared to other combinations of these two variables.
- **EH6**: Belief that surveillance has increased is associated with a different likelihood of saying the evolution of surveillance in the past five years is worrisome than believing it has stayed the same or decreased.
- **EH7**: Reporting that one feels deliberately disadvantaged by the government or police is associated with trust in … (text truncated as in original preregistration).
- **EH8**: Among those who perceive both crime and corruption as high, support for surveillance will differ depending on whether corruption or crime is perceived as more concerning.
- **EH9**: Perceived levels of corruption are associated with trust in authorities.
- **EH10**: Among participants who perceive high corruption and report low acceptance of surveillance, the option of independent monitoring is associated with differences in support for surveillance compared to those who reported high acceptance.
- **EH11**: Some participants will express general disagreement with a surveillance type (e.g., covert digital) while supporting most corresponding specific methods, and vice versa (general agreement but rejecting specific methods).
- **EH12**: Some participants will express a preference for being surveilled through the method (digital or physical) that they also believe would reveal more information about them.
- **EH13**: Age, gender, and education level are associated with differences in acceptance, trust, and perceived invasiveness of surveillance methods.
- **EH14**: Participants with higher income or higher education will differ in their acceptance of digital surveillance methods.
- **EH15**: Participants whose stated political affiliation aligns with the party currently in power (Republican) will differ in their trust in government, acceptance of surveillance, and concerns about surveillance abuse compared to participants whose affiliation does not align with the governing party.
- **EH16**: Among participants who indicate their responses would change if the Democratic Party were currently in power, the type and direction of reported change (e.g., trust, support, fear of abuse) will differ based on their stated political affiliation.

---

## Other

None specified.

# Digital Government: Analyzing micro and small organizations (MSEs) understanding and readiness for digital transformation towards the solutions offered by government institutions

## Research Aims

The aim of this study is to investigate the perception of micro and small organizations in the tourism sector regarding the cost–benefit ratio of government-subsidized digitization solutions for their digital transformation.

If experts are available and time permits, I will also try to interview them to find out which approaches they use to help micro and small organizations.

**Aim type:** Exploring

---

## Research Question(s)

**Main research question**

How do tourism MSE (micro and small enterprise) organizations perceive government efforts to promote digitalization, and what factors influence their adoption or rejection of digitalization solutions?

**Initial assumptions / expectations**

These assumptions will be used if relevant scientific literature can be found:

A. There is a relationship between the cost–benefit analysis from SEBRAE and SMEs’ digitalization adoption, which can be explained by a mismatch between the two.

B. The government’s digitalization plans are not tailored to the specific understanding, needs, and capabilities of MSE organizations, creating a gap that hinders adoption.

C. The lack of effective communication channels between the government and MSE organizations is the main reason for the gap between government digitalization plans and MSE organizations’ awareness, despite the government’s preparedness and understanding.

If one of these assumptions is supported by scientific literature, the approach will be more **deductive**; otherwise, the study will follow an **inductive** approach.

---

## Anticipated Duration

June 2023

---

# Design Plan

## Study Design

The study is based on:

- **Semi-structured interviews** with approximately 15–25 people in charge of micro or small organizations in the tourism sector.
- **One-on-one expert interviews** with representatives of the government of Ceará and SEBRAE related to the research topic.
- **User testing / usability testing**, to understand the process a person would follow when approaching SEBRAE for help in obtaining information about their products.

Interviews will be conducted primarily online via Microsoft Teams; in some cases, WhatsApp calls or chat may be used. All interviews will be transcribed.

---

## Sampling and Case Selection Strategy

- One-on-one expert interviews helped to sharpen the focus of the topic and identify pain points on which the study can be based.
- The target is **25 micro or small business managers/owners** to perform in-depth exploration of potential causal relationships and improve robustness of conclusions.
- The sample will include managers/owners of:

  - _Posadas_ and hotels
  - Restaurants
  - Kitesurfing schools

- Many interviewees are people I already know or who lived in the same place as I did during my time in Brazil (Jericoacoara, Cumbuco, etc.).

**Survey / interview logistics**

- Each interview will last approximately **10–30 minutes**, depending on the interviewee’s experience and answers.
- Some interviewees may have no experience or knowledge of government or SEBRAE offers; they may only be able to answer the general and institutional blocks.
- In the first step, **50%** of interviewees will receive the survey in advance.
  - This is to reduce the risk that reading the survey beforehand makes them cancel the interview.
- The other **50%** will **not** receive the survey in advance but will instead be sent the _SEBRATech_ link before or during the interview.
- All interviews will be conducted online via Microsoft Teams where possible.

**Stopping criteria**

- Data collection will stop once **15 interviews** have been completed, unless the collected data is insufficient to reach proper conclusions. In that case, more interviews may be conducted.

---

# Data Collection

## Data Sources and Types

**Primary data**

- Managers or owners of:
  - Hotels / _posadas_
  - Restaurants
  - Kitesurf schools

**Secondary data**

- Work as business analyst in two tourist locations
- Personal contacts
- Instagram accounts
- Contact with politicians and personalities at local and state level
- Literature analysis
- User testing
- Document analysis

Some governmental contacts cannot be named, but examples of organizations/companies that may be interviewed include:

- Pousada Aki Jeri (Jericoacoara)
- Sunset Jeri Flat (Jericoacoara)
- Pousada Beleleu (Jericoacoara)
- Restaurant Pimienta Verde (Jericoacoara)
- Restaurant Nox (Jericoacoara)
- Vila Calango Surf School (Jericoacoara)
- Pousada Tucano (Cumbuco)
- Hostel/Kite School Indiana (Cumbuco)
- Restaurant Isabella (Cumbuco)
- Restaurant Chez Marc (Cumbuco)
- Restaurant Francesinha (Cumbuco)
- Pousada/Kite School Meeting Point (Cumbuco)
- Pousada/Kite School Katavento (Cumbuco)
- Etc.

---

## Data Collection Methods

- **Semi-structured interviews** with managers/owners of micro and small organizations in the tourism sector, focusing on:

  - Perceptions of government digitization offers (e.g., SEBRAE)
  - Perceived costs, benefits, barriers, and needs

- **One-to-one expert interviews** with:

  - Government representatives
  - SEBRAE representatives

  These help refine the focus and better understand the policy and support context.

- **User testing / usability testing**:
  - Observing or reconstructing what happens when someone approaches SEBRAE (online, via WhatsApp, or in person in Fortaleza) to request help and information about digital products and services.

Interviews may be conducted:

- Via Microsoft Teams (preferred)
- Via WhatsApp calls or chat
- Occasionally in person

All interviews will be transcribed and then translated (from Portuguese to English) using tools such as Google Translate or DeepL, followed by manual checking.

---

## Data Collection Tools / Instruments

- Semi-structured interview guides (for both SME managers/owners and experts).
- Microsoft Teams / WhatsApp for conducting and recording interviews.
- Transcription and translation tools (e.g., Google Translate, DeepL).
- User testing protocols for interactions with SEBRAE (via email, WhatsApp, or in-person office visits in Fortaleza).

---

# Analysis Plan

## Data Analysis Approach

The study will use **qualitative content analysis** with **inductive coding** to understand why micro and small enterprises are not more actively adopting digital technologies provided by SEBRAE and related government initiatives.

Key steps:

1. **Transcription** of all interviews.
2. **Inductive coding** of transcripts using content analysis:
   - Identify repeated words, phrases, and ideas related to:
     - Cost–benefit perceptions
     - Barriers to adoption
     - Awareness and understanding of SEBRAE offers
     - Trust, usability, and relevance of digital solutions
   - Generate initial codes and categories directly from the data.
3. **Categorization and theme development**:
   - Group codes into broader themes (e.g., accessibility, understanding, prioritization, validation of utility).
   - Explore relationships between themes and the perceived cost–benefit of digital transformation.
4. **Pattern identification**:
   - Look for cross-case patterns and recurring issues.
   - Use triangulation with user testing, expert interviews, and document analysis to refine interpretations.
5. **Unexpected findings**:
   - Pay particular attention to unexpected or emergent themes, especially given the semi-structured nature of interviews, where interviewees can freely express fears, concerns, and experiences.

The overall aim is to show how SMEs perceive the cost–benefit ratio of digital transformation and how this links to the design and communication of SEBRAE’s digitalization offers.

---

## Data Analysis Process

- Coding and analysis will be conducted **manually** using **MAXQDA** for inductive coding and qualitative content analysis.
- All analysis will be performed by **Pablo Fernandez Macho**.
- The coding scheme (codebook) will be made available in the data shared on the Open Science Framework (OSF).

---

# Credibility Strategies

Selected strategies:

- **Bringing in different perspectives**
- **Dialogues with subjects**
- **Data triangulation**
- **User experience / user testing**

### Rationale

- **Cultural context & access challenges**  
  Due to cultural differences and practical constraints, it may be difficult to reach some participants multiple times. Owners and managers of SMEs often work very hard and may only be available once. Therefore, interviews will be designed to gather as much relevant information as possible in a single session.

- **Focus on SEBRAE’s offer**  
  The research will focus on the offer of SEBRAE and whether it meets SME needs, particularly in four areas:

  1. Accessibility of content
  2. Understanding of content
  3. Prioritization of content
  4. Validation of utility

- **Data triangulation**  
  Combining:

  - Semi-structured interviews with SME managers/owners
  - One-on-one expert interviews (government, SEBRAE)
  - Document analysis and literature review
  - User testing and lived experience of the digitalization process

  This allows comparison and contrast of different stakeholder perspectives and helps identify common themes and robust patterns.

- **User testing / UX perspective**  
  User testing provides real-life experiences of the digitalization support process (e.g., contacting SEBRAE), making the findings more credible and grounded in practice.

---

# Miscellaneous

## Reflection on Positionality

The author has worked since 2014 in a platform that helps SMEs with digitization, digitalization, and digital transformation.

The intention is to reveal that current strategies may need to be rethought to become more user-oriented, applying understanding and empathy. The study aims to uncover possible “silo thinking” on both sides (government/SEBRAE and SMEs) and to highlight how this affects the perceived cost–benefit of digital transformation solutions.
