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
| **Validation** | Train/test split first, preprocessing inside pipelines, 5-fold cross-validation for tuning, early stopping on a validation split taken from the training set, test set used once for final evaluation |

E, S and G sub-scores were removed because they are components of the target (data leakage).

## Key results

- The **PyTorch neural network** achieves the highest test R² (**0.47**, RMSE 4.92), with SVR (0.44) and GAM (0.43) close behind; polynomial regression severely overfits. With about 76 test companies, differences of a few points of R² should not be over-interpreted.
- **Most relevant variables:** sector (relative to the baseline sector, energy firms show higher ESG risk, though only borderline significant, while real estate and technology show markedly lower risk), **controversy score** and **company size** (total assets, employees), consistently across standardised coefficients and SHAP values.
- The moderate R² suggests that quantitative variables capture relevant information but do not fully explain ESG risk.
- **LLM application:** the model's prediction for a company is passed to an LLM (GPT-OSS 120B via Groq API) together with its data, to generate a readable ESG risk report. Example: Comcast, the first company in the test set, predicted 23.05 vs. actual 23.10; the average test error is about 4 points (MAE 3.97).

## Limitations

- Small sample, so model rankings are not robust; the PyTorch network is evaluated on a single validation split and seed, while the other models are tuned with cross-validation.
- Results are predictive associations, not causal effects.

## Repository

- `esg_risk_regression.ipynb` — full Python workflow
- `esg_risk_regression_presentation.pdf` — project presentation
- `esg_financials_merged_simfin_2023.csv` — dataset

**Tools:** Python · pandas · scikit-learn · PyTorch · SHAP · pyGAM · Groq API
