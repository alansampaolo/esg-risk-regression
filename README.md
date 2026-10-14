# ESG Risk Score Regression

Machine learning project · Python · PyTorch · SHAP · LLM API

## Objective

The aim was to **identify the main factors associated with and predictive of ESG risk scores**, and to **understand which variables were most relevant for the model**. Those predictive relationships could also **motivate future causal research**, although the project itself was not designed to estimate causal effects.

## What was compared

| | |
|---|---|
| **Data** | 378 S&P 500 companies: financial data (SimFin) merged with ESG risk scores and controversy data |
| **Target** | Total ESG Risk Score (higher = greater exposure to ESG-related risks) |
| **Models** | Linear, Ridge, Lasso, polynomial, GAM, decision tree, bagging, random forest, gradient boosting, SVR (RBF), MLP, PyTorch neural network |
| **Validation** | 5-fold cross-validation on the training set, preprocessing inside pipelines, test set used once for final evaluation |

ESG sub-scores too close to the target were removed to avoid data leakage.

## Key results

- The **PyTorch neural network** performs best on the test set (**R² 0.47**, RMSE 4.92), with SVR (0.44) and GAM (0.43) as the closest alternatives; polynomial regression severely overfits.
- **Most relevant variables:** sector (energy firms show higher ESG risk, real estate and technology lower), **controversy score** and **company size** (total assets, employees), consistently across standardised coefficients and SHAP values.
- The moderate R² suggests that quantitative variables capture relevant information but do not fully explain ESG risk.
- **LLM application:** the model's prediction is passed to an LLM via API to generate a readable ESG risk report for each company (e.g. Comcast: predicted 23.05 vs. actual 23.10).

## Repository

- `esg_risk_regression.ipynb` — full Python workflow
- `esg_risk_regression_report.pdf` — project presentation
- `esg_financials_merged_simfin_2023.csv` — dataset

**Tools:** Python · pandas · scikit-learn · PyTorch · SHAP · pyGAM · Groq API
