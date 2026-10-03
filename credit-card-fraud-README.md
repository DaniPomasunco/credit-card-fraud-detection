# Credit Card Fraud Detection

**Can we flag fraudulent card transactions before they cost the business money?**
Analysis of 671,028 card transactions (2019–2020), a machine learning model that ranks transactions by fraud risk, and an interactive dashboard to explore the results.

🔗 **[Live dashboard](https://credit-card-fraud-group11-gepvmu5uxdjqf3kn2k4bqm.streamlit.app/)** · 📓 **[Full analysis notebook](Credit%20Card%20Fraud%20Analysis_V4.ipynb)**

> The dashboard runs on Streamlit's free tier. If it shows a "wake up" screen, click the button and give it a few seconds.

---

## The problem

Fraud is rare but expensive. In this dataset only **0.59% of transactions are fraudulent**, yet they account for **4.2–4.8% of total transaction value**, about **$2.1M** across the two years. A model that simply predicts "legitimate" every time would be 99.4% accurate and completely useless, so the work focuses on catching the rare cases.

## What the data shows

| Finding | Evidence |
| --- | --- |
| Fraudulent transactions are much larger | Average **$527** vs **$68** for legitimate ones (median $369 vs $47) |
| Night-time and amount drive the risk | Night-time flag and transaction amount are the two strongest predictors, far ahead of the rest |
| Some channels are riskier | Online shopping and miscellaneous online categories carry higher fraud rates |
| Fraud grew as a share of activity | Fraud rate rose from 0.57% (2019) to 0.63% (2020) |

## Approach

1. **Data exploration with SQL (DuckDB)**: data quality checks, fraud by year, amount, category, time of day, customer age and job.
2. **Feature engineering**: transaction amount (log), customer age, distance between customer and merchant, hour, weekday, night and weekend flags, merchant category.
3. **Model comparison** with 5-fold stratified cross-validation, all models weighted for class imbalance:

| Model | CV ROC-AUC | CV F1 (fraud) |
| --- | --- | --- |
| Logistic Regression (baseline) | 0.891 | 0.04 |
| Random Forest | 0.988 | 0.78 |
| **Histogram Gradient Boosting** | **0.997** | 0.45 |

4. **Tuning** with randomized search (30 iterations), then a **decision threshold chosen to maximise F1**, since the threshold is a business lever, not a fixed 0.5.
5. **Explainability** with permutation importance and SHAP, so every flag can be explained to a fraud analyst.

## Results on unseen data

The tuned Gradient Boosting model was evaluated once on a held-out 20% test set (134,206 transactions):

| Metric | Value |
| --- | --- |
| ROC-AUC | **0.92** |
| Precision (fraud) | 0.55: more than half of flagged transactions are real fraud |
| Recall (fraud) | 0.36 at the chosen threshold |

## Business recommendations

- **Step-up authentication** above a transaction amount threshold.
- **Lower overnight limits** unless the customer has pre-authorised spending.
- **Enhanced monitoring** of online shopping and miscellaneous online merchants.
- **Keep geo and velocity rules** running alongside the model.

## Limitations and next steps

- Cross-validation AUC (0.997) is higher than test AUC (0.92), so the next step is to investigate that gap, for example with a time-based split.
- Recall of 36% means many frauds still pass at this threshold. Lowering it catches more fraud at the cost of more false alarms; the right point depends on the cost of each.

## Built with AI

The interactive dashboard was built with **Claude Code**, turning the notebook analysis into a deployed Streamlit app.

## Context

Final project for **E628 Data Science for Business**, London Business School.
<!-- Add your role here, e.g. "Team project; I led the exploratory analysis, modelling and the dashboard." -->

## Tech stack

Python · pandas · DuckDB (SQL) · scikit-learn · SHAP · matplotlib · seaborn · Streamlit · Claude Code

## Run it yourself

```bash
pip install pandas numpy duckdb scikit-learn shap matplotlib seaborn
jupyter notebook "Credit Card Fraud Analysis_V4.ipynb"
```

The notebook downloads the dataset automatically from the course repository.
