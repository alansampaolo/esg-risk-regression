# ESG Risk Regression

Machine learning project for predicting corporate ESG Risk Scores using financial, sector and controversy related data.

## Project Overview

The goal of this project is to predict the Total ESG Risk Score of companies using financial and company related variables, sector information, and controversy related features.

The dataset was constructed by merging ESG data with financial data over the same reference period. The merged dataset initially contained 441 observations, with 378 companies retained in the final modelling sample after preprocessing.

## Data Sources

The dataset used in this project was constructed by merging information from two external sources:

- **SimFin** for company-level financial data
- **Kaggle** for ESG-related data on S&P 500 companies

The two datasets were aligned to the same reference period and merged using companies available in both sources.

Because the financial dataset did not contain all S&P 500 companies, the resulting merged sample was smaller than the full index universe.

The merged dataset initially contained 441 observations, while the final modelling sample contained 378 companies after preprocessing.

## Methodology

The analysis follows a supervised machine learning workflow:

1. Data cleaning and preprocessing
2. Removal of potentially redundant or leakage-prone variables
3. Train/test split with 5-fold cross-validation performed on the training set
4. Model comparison across linear, nonlinear and machine learning approaches
5. Final evaluation on the held-out test set
6. Model interpretation using standardized coefficients and SHAP values

## Models

The following models were compared:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Generalized Additive Model (GAM)
- Polynomial Regression
- Decision Tree
- Random Forest
- Bagging
- Gradient Boosting
- Support Vector Regression (RBF kernel)
- MLPRegressor
- PyTorch Neural Network

## Results

The PyTorch Neural Network achieved the best predictive performance on the held-out test set.

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| PyTorch Neural Network | 4.915 | 3.968 | 0.470 |
| SVR (RBF) | 5.056 | 4.071 | 0.440 |
| GAM | 5.122 | 4.266 | 0.425 |
| MLPRegressor | 5.161 | 4.231 | 0.416 |
| Linear Regression | 5.258 | 4.404 | 0.394 |
| Gradient Boosting | 5.263 | 4.184 | 0.393 |
| Ridge Regression | 5.277 | 4.412 | 0.390 |
| Lasso Regression | 5.287 | 4.422 | 0.387 |
| Bagging | 5.411 | 4.229 | 0.358 |
| Random Forest | 5.481 | 4.287 | 0.341 |
| Decision Tree | 5.631 | 4.407 | 0.305 |

The best test R² is moderate, indicating that the available quantitative variables capture relevant information about ESG Risk Scores but do not fully explain them.

## Interpretation and Limitations

Model interpretation was performed using two complementary approaches:

- standardized coefficients for the linear regression model;
- SHAP values for the PyTorch Neural Network.

The interpretation is predictive rather than causal. In other words, the coefficients and SHAP values identify variables that are associated with the model predictions, but they do not establish causal effects on ESG Risk Scores.

This distinction is important because the project was designed for prediction, not for causal inference or policy evaluation.

The main limitations are:

- the final sample is smaller than the full S&P 500 universe because only companies available in both original data sources could be merged;
- the final modelling sample contains 378 companies;
- the best test R² is moderate;
- the analysis does not provide causal evidence and should therefore not be interpreted as a direct basis for policy recommendations.

## LLM Application

As an additional application, the output of the best-performing PyTorch model was connected to a Large Language Model API to generate a short, readable ESG risk report.

The workflow is:

`Company data → PyTorch prediction → LLM-generated report`

The purpose of this step is not to improve the prediction itself, but to translate the model output into a more accessible narrative format.

## Repository Structure

```text
esg-risk-prediction/
│
├── README.md
├── esg_risk_prediction.ipynb
├── esg_financials_merged_simfin_2023.csv
└── esg_risk_prediction_report.pdf
```

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- PyTorch
- SHAP
- statsmodels
- pyGAM
- Matplotlib
- Groq API
- Jupyter Notebook
  
## Files

- `esg_risk_prediction.ipynb` — main analysis and machine learning workflow
- `esg_financials_merged_simfin_2023.csv` — merged ESG and financial dataset
- `esg_risk_prediction_report.pdf` — project presentation
