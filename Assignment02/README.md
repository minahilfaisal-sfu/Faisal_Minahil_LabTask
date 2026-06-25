# IAT 461 / 882 — Assignment 2: Modeling Phigma User Behavior
Deadline: June 1st - 23:55

## Overview

In this assignment you take on the role of a data scientist at **Phigma** — a design SaaS platform with a freemium whiteboarding tool. You have access to data from two separate research campaigns run by the Phigma product team.

Your job is not just to fit models. It is to **interpret them like a researcher**: understand what the numbers mean, think carefully about which variables belong in your model and why, and communicate findings in plain language.

This assignment gives you practice in:

- **Feature engineering** from raw behavioral and datetime data
- **Linear regression** as a statistical analysis tool — interpreting coefficients, p-values, and effect sizes
- **Logistic regression** as a classification model — interpreting odds ratios and evaluating predictions
- **Applying a second classifier** of your choice and comparing it critically to logistic regression

### Learning goals

- Translate raw data into model-ready features
- Understand the difference between statistical inference (explaining) and predictive modeling (predicting)
- Interpret regression coefficients and odds ratios as measures of feature importance
- Evaluate classifier performance using appropriate metrics
- Think critically about which variables should and should not be included in a model

---

## The Phigma Context

Phigma is a browser-based design and whiteboarding tool with a freemium model. Free-tier users have access to core whiteboarding features; a paid subscription unlocks AI tools, export options, and team collaboration at scale.

The Phigma data science team ran **two separate studies**:

**Study 1 — AI Assistant A/B Experiment** (`phigma_ab_study.csv`)
The team randomly assigned free-tier users to one of two conditions during a 14-day observation window: access to a new AI design assistant (treatment) or no AI access (control). The question: *does the AI assistant increase how long users spend in whiteboard sessions?*

**Study 2 — Subscription Conversion** (`phigma_subscription.csv`)
A separate cohort of free-tier users was observed over 30 days. The question: *which behavioral and demographic factors predict whether a user subscribes to a paid plan?*

These are two different research questions, two different datasets, and two different modeling approaches. But they tell a connected story about the same product.

---

## Data and Tools

### Datasets

Both datasets are **synthetic** — generated for this course using realistic distributional properties. They are available on Canvas under **Data and Code**.

**`phigma_ab_study.csv`** — 2,000 rows, one per free-tier user (Study 1)

| Column | Description |
|---|---|
| `user_id` | Unique user identifier |
| `ab_group` | Experimental condition: `control` or `treatment` (AI assistant access) |
| `device_type` | Primary device: `desktop`, `tablet`, or `mobile` |
| `account_age_days` | Days since account creation at experiment start |
| `prior_whiteboard_sessions` | Number of whiteboard sessions in the 14 days *before* the experiment |
| `plan_region` | User's geographic region: `NA`, `EU`, `APAC`, `LATAM` |
| `num_collaborators` | Number of collaborators on the user's account before the experiment |
| `feature_tour_completed` | Whether the user completed Phigma's onboarding feature tour (0/1) |
| `templates_used` | Number of templates used *during* the experiment window |
| `ai_suggestions_used` | Number of AI suggestions accepted *during* the experiment window |
| `whiteboard_actions_per_session` | Average number of canvas actions per session during the experiment |
| `avg_whiteboard_session_duration_min` | **Outcome** — average whiteboard session duration in minutes |

**`phigma_subscription.csv`** — 3,000 rows, one per free-tier user (Study 2)

| Column | Description |
|---|---|
| `user_id` | Unique user identifier |
| `signup_date` | Date the user created their free account (YYYY-MM-DD) |
| `first_session_date` | Date of the user's first whiteboard session (YYYY-MM-DD) |
| `device_type` | Primary device: `desktop`, `tablet`, or `mobile` |
| `country_tier` | Market tier based on country: `tier_1`, `tier_2`, `tier_3` |
| `referral_source` | How the user found Phigma: `organic`, `social`, `paid_ad`, `referral` |
| `num_sessions_30d` | Total whiteboard sessions in first 30 days |
| `total_time_min_30d` | Total minutes spent in the app in first 30 days |
| `projects_created_30d` | Number of new projects created in first 30 days |
| `exports_30d` | Number of file exports (PNG, PDF, SVG) in first 30 days |
| `collaborators_invited_30d` | Number of collaborators invited in first 30 days |
| `support_tickets_30d` | Number of support tickets submitted in first 30 days |
| `used_premium_feature_preview` | Whether the user clicked into a paywalled feature (0/1) |
| `ai_assistant_sessions_30d` | Number of sessions where the user interacted with the AI assistant |
| `subscribed` | **Outcome** — whether the user subscribed within 30 days (0/1) |

### Environment

- Analysis in **Python** using **Jupyter Notebooks**
- Use the scaffold notebook provided on Canvas: `IAT461_A2_Scaffold.ipynb`

### Required libraries

- `pandas`, `numpy` — data handling and feature engineering
- `matplotlib`, `seaborn` — visualization
- `scipy.stats` — normality checks
- `statsmodels` — OLS regression and logistic regression with p-values
- `sklearn` — train/test split, classification metrics, and your chosen model in Part C

### Optional (for hyperparameter tuning bonus)

- `sklearn.model_selection.GridSearchCV` or `RandomizedSearchCV`
- `xgboost` or `lightgbm` if you choose those models (install separately with `pip install xgboost` or `pip install lightgbm`)

---

## Assignment Structure

This assignment has **three parts** plus an **optional bonus**.

---

### Part A — Linear Regression: Did the AI Assistant Change User Behavior?

**Dataset:** `phigma_ab_study.csv`
**Goal:** Fit an OLS regression model to explain average whiteboard session duration. Treat this as a statistical analysis, not a prediction task — you are trying to understand *what explains* session duration, not forecast it. There is no train/test split in this part.

#### A1 — Exploratory Data Analysis

Before modeling, understand your data.

- Load the dataset and inspect shape, types, and summary statistics
- Plot the distribution of `avg_whiteboard_session_duration_min`
  - Is it normally distributed? Use a histogram and a Q-Q plot to assess
  - Compute skewness
- Apply a **log transformation** to the outcome and re-examine the distribution
  - Does it improve? Justify your choice of whether to model the raw or log-transformed outcome
- For each numeric predictor, examine its distribution and its relationship with the outcome (correlation, scatterplot, or boxplot as appropriate)
- Examine `ai_suggestions_used` and `whiteboard_actions_per_session` closely
  - What do you notice about these variables? Why might including them in a regression model be problematic?

> **Markdown required:** Summarize your EDA findings in 3–5 sentences. Identify at least two features you will *not* include in your model and explain why.

#### A2 — Feature Engineering

Prepare your features for OLS.

- Encode `ab_group` as a binary dummy variable (0 = control, 1 = treatment)
- One-hot encode `device_type` (drop one level to avoid the dummy variable trap)
- One-hot encode `plan_region` (drop one level)
- Apply any transformations to numeric predictors you think are justified (e.g. log-transforming right-skewed predictors). Explain each choice.
- Confirm your final feature matrix contains no missing values

> **Markdown required:** List your final feature set and briefly justify each inclusion and any transformations applied.

#### A3 — OLS Regression

Fit a linear regression model using `statsmodels.OLS`.

- Use the log-transformed outcome if your EDA supports it
- Print the full `model.summary()` output
- Interpret the following:
  - **R²** — how much variance does the model explain?
  - **Coefficient for `ab_group`** — what is the estimated effect of AI assistant access on session duration? If you used a log outcome, back-transform to minutes for interpretation.
  - **p-values** — which predictors are statistically significant at α = 0.05?
  - **Effect sizes** — for significant predictors, is the effect *practically* meaningful, not just statistically significant?
- Check regression assumptions with diagnostic plots:
  - Residuals vs. fitted (linearity and homoscedasticity)
  - Q-Q plot of residuals (normality of errors)

> **Markdown required:** Write a short "results" paragraph as if reporting findings in a research paper. Include the treatment effect estimate, its significance, and at least two other notable predictors. Discuss whether the AI assistant had a meaningful practical effect.

#### A4 — Exploratory Interaction Analysis

The average treatment effect tells you the AI assistant's impact *across all users*. But experimental effects are rarely uniform — some users may benefit more than others.

- Choose **one theoretically motivated interaction** to test. Good candidates include:
  - `ab_group × device_type` — does the AI help more on desktop where there is more screen space?
  - `ab_group × prior_whiteboard_sessions` — does the AI help novices more, or power users?
  - `ab_group × num_collaborators` — does the AI matter more for solo users who have no team to collaborate with?
- State your hypothesis *before* running the model
- Create the interaction term and add it to your model
- Interpret the interaction coefficient: for which subgroup is the AI assistant more or less effective?
- Visualize the interaction (e.g. separate regression lines per group, or a grouped bar chart)

> **Markdown required:** State your hypothesis before fitting. After fitting, describe whether the data supported it and what it implies for how Phigma might target the AI assistant rollout.

---

### Part B — Logistic Regression: Who Converts?

**Dataset:** `phigma_subscription.csv`
**Goal:** Build a logistic regression model to predict subscription. Use `statsmodels` for interpretation and `sklearn` for predictive evaluation.

#### B1 — Feature Engineering

This dataset contains raw columns that must be transformed before modeling. Derive the following features:

| Feature to create | How |
|---|---|
| `days_to_first_session` | Difference in days between `first_session_date` and `signup_date` |
| `avg_session_duration_min` | `total_time_min_30d` / `num_sessions_30d` |
| `is_weekend_signup` | 1 if `signup_date` falls on a Saturday or Sunday, else 0 |
| One-hot encoding of `device_type` | Drop one level |
| One-hot encoding of `referral_source` | Drop one level |
| Encoding of `country_tier` | Ordinal (1/2/3) or one-hot — justify your choice |

**Edge case:** What happens when `num_sessions_30d` is 0 for a user? Handle this explicitly and describe your decision.

After engineering, confirm:
- No missing or infinite values in your feature matrix
- All features are numeric

> **Markdown required:** For each engineered feature, explain in one sentence why it might be predictive of subscription.

#### B2 — EDA for Subscription

Before modeling, explore the relationship between your features and the outcome.

- What is the overall subscription rate? Is the dataset imbalanced?
- For at least **three features**, visualize how subscription rates differ across groups or values (e.g. subscription rate by `referral_source`, distribution of `exports_30d` by subscription status)
- Compute a correlation matrix for numeric features — note any strongly correlated pairs

> **Markdown required:** Identify the two or three features that appear most strongly associated with subscription in your EDA. Do any associations surprise you?

#### B3 — Logistic Regression with statsmodels (Inferential)

Fit a logistic regression using `statsmodels.Logit` on the **full dataset** (no train/test split — this is the inferential, interpretive model).

- Print the full `model.summary()`
- For each significant predictor (p < 0.05), compute the **odds ratio** by exponentiating the coefficient
- Interpret at least **three odds ratios** in plain language (e.g. "Users who previewed a premium feature were X times more likely to subscribe, holding other factors constant")
- Identify any predictor whose sign or magnitude surprises you and offer an interpretation

> **Markdown required:** Write a short "results" paragraph reporting which behavioral factors most strongly predict subscription and in which direction. Include at least one nuanced or counterintuitive finding.

#### B4 — Logistic Regression with sklearn (Predictive)

Now treat this as a prediction task.

- Split the data 80/20 using `train_test_split` with `stratify=y` and `random_state=42`
- Fit `LogisticRegression` on the training set
- Evaluate on the test set:
  - Confusion matrix (visualize as a heatmap)
  - Classification report (precision, recall, F1 per class)
  - ROC-AUC score
  - ROC curve plot
- Given the class imbalance (~20% positive rate), which metric is most informative here? Justify your answer.

> **Markdown required:** Interpret your confusion matrix. How many false negatives does the model produce and why does that matter from Phigma's business perspective?

---

### Part C — Your Model: Choose, Learn, Apply, Compare

**Dataset:** `phigma_subscription.csv` — use the **same 80/20 train/test split** from B4  
**Goal:** Choose a second classifier, understand how it works, apply it, and compare it honestly to logistic regression.

#### C1 — Choose Your Model

Select **one** model from the list below:

| Model | Family |
|---|---|
| Decision Tree | Tree-based |
| Random Forest | Ensemble (bagging) |
| Gradient Boosting | Ensemble (boosting) |
| XGBoost | Ensemble (boosting) |
| LightGBM | Ensemble (boosting) |
| K-Nearest Neighbors | Instance-based |
| Naive Bayes | Probabilistic |
| Support Vector Machine | Margin-based |
| Linear Discriminant Analysis | Statistical |
| AdaBoost | Ensemble (boosting) |

> **Markdown required (before any code):** In 3–5 sentences, explain how your chosen model works conceptually. Answer: How does it learn from training data? What does it optimize? Why might it perform differently from logistic regression on this particular dataset?

#### C2 — Fit and Evaluate

- Fit your chosen model using **default hyperparameters** on the training set from B4
- Evaluate on the same test set:
  - Confusion matrix
  - Classification report (precision, recall, F1)
  - ROC-AUC score
  - ROC curve

#### C3 — Model Comparison

Compare your model to logistic regression.

- Present a **summary table** with F1, ROC-AUC, precision, and recall for both models side by side
- Plot both ROC curves on the same axes with a legend
- Which model performs better? By how much?
- Why do you think your model performed the way it did, given the nature of the data and the mechanics of the model?

> **Markdown required:** Write a comparison paragraph. Do not just report numbers — explain *why* the performance difference occurred. If your model did worse than logistic regression, that is a valid result worth explaining.

#### C4 — Bonus: Hyperparameter Tuning (+5 points)

- Define a parameter grid appropriate for your model (minimum 3 parameters or combinations)
- Run `GridSearchCV` or `RandomizedSearchCV` with 5-fold cross-validation on the training set
- Report the best parameters found
- Evaluate the tuned model on the test set
- One sentence: did tuning meaningfully change performance?

---

## Deliverables and Submission

Upload **two files** to Canvas:

1. **Completed notebook:** `iat461_a2_[YourName]_[SFUid].ipynb`
2. **Notebook exported as PDF:** `iat461_a2_[YourName]_[SFUid].pdf`

All code must run top-to-bottom without errors. Markdown cells must be completed — blank or placeholder cells receive zero for that section.

---

## Marking Rubric (100 points)

| Section | What we are looking for | Points |
|---|---|---|
| **A1 — EDA** | Distribution check, log-transform justification, confounder identification with explanation | 8 |
| **A2 — Feature Engineering** | Correct encoding, transformation choices explained | 7 |
| **A3 — OLS Regression** | Correct model, coefficient + R² interpretation, effect size discussion, assumption checks | 15 |
| **A4 — Interaction Analysis** | Hypothesis stated before fitting, correct interaction term, interpretation and visualization | 10 |
| **B1 — Feature Engineering** | All derived features correct, edge case handled, justification for each feature | 10 |
| **B2 — EDA** | Subscription rate noted, at least three meaningful visualizations, correlation awareness | 7 |
| **B3 — Logistic (Inferential)** | Correct model, odds ratios computed and interpreted, nuanced written discussion | 13 |
| **B4 — Logistic (Predictive)** | Correct split, all metrics reported, metric choice justified, business interpretation of confusion matrix | 10 |
| **C1 — Model Choice** | Clear conceptual explanation written before any code | 5 |
| **C2 — Fit and Evaluate** | Correct implementation, all metrics reported | 5 |
| **C3 — Comparison** | Honest comparison table + dual ROC plot, mechanistic explanation of performance difference | 10 |
| **C4 — Tuning (Bonus)** | Parameter grid defined, cross-validation run correctly, tuned results compared to default | +5 |

### Grading philosophy

- **Interpretation matters as much as code.** A model run without a Markdown explanation earns at most half the section marks.
- **Correctness over complexity.** A well-explained simple model beats a complicated one with no interpretation.
- **Honest analysis is rewarded.** If your chosen model underperforms logistic regression, explaining why thoughtfully earns full marks for C3.

---

## Academic Integrity and AI Use

This is an **individual** assignment. You may discuss general concepts with peers; all code and written interpretation must be your own.

### AI tools

AI coding tools (ChatGPT, Copilot, etc.) are permitted for specific tasks with full disclosure.

**Permitted:**
- Generating code for a specific, narrow step (e.g. "how do I exponentiate statsmodels coefficients to get odds ratios")
- Debugging syntax errors
- Understanding what a library function does

**Not permitted:**
- Asking AI to interpret your results or write your Markdown cells
- Generating entire task sections and submitting the output
- Using AI to write comparison or results paragraphs

### Disclosure format

```python
#BEGIN[Tool name][URL]["exact prompt used"]
# ... your code ...
#END[Tool name]
```

Prompts may be tested. AI-written interpretations will receive zero for that section.

---

*Datasets: `phigma_ab_study.csv` and `phigma_subscription.csv` — synthetic data generated for IAT 461/882, Summer 2026. Available on Canvas under Data and Code.*
