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
