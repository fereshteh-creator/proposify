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
