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
