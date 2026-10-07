# Explainable Machine Learning

This project compares global and local model-interpretation methods for linear regression and gradient boosting on the 2021 World Happiness Report data.

## Questions explored

- How does feature scaling affect coefficient-based interpretation?
- How do linear and tree-based models behave outside the observed feature range?
- When do ICE, PDP, and ALE tell different stories?
- How stable are permutation and occlusion-based importance estimates?
- How do global and local SHAP explanations differ?
- How sensitive is LIME to neighbourhood size and kernel width?

## Methods

- linear regression, Lasso, and gradient boosting;
- Individual Conditional Expectation (ICE);
- Partial Dependence Plots (PDP);
- Accumulated Local Effects (ALE);
- permutation and occlusion importance;
- SHAP;
- LIME and a compact custom LIME implementation.

## Main observations

- Linear regression extrapolates its fitted trend beyond the training range, while tree-based boosting becomes nearly constant outside its learned split thresholds.
- Global importance methods broadly agree on the most influential variables, although magnitudes and rankings vary by model and method.
- Local explanations depend on the selected observation and neighbourhood definition.
- Increasing the number of LIME perturbations improves coefficient stability, while kernel width controls the trade-off between locality and smoothness.

## Important limitation

`Dystopia + residual` participates in the construction and decomposition of the reported happiness score. Using it as an ordinary predictor introduces target leakage and explains the unusually small error of the linear model. The numerical performance should therefore not be interpreted as real-world predictive power.

The dataset is retained because it provides a compact demonstration of interpretation techniques. A deployment-oriented study should remove target-derived variables and define a prospective prediction target.

## Files

- `xai_analysis.ipynb` — complete analysis with saved figures and results.

## Running the notebook

Install the dependencies from the repository-level `requirements.txt`, then open the notebook from the repository root. An internet connection is required to retrieve the dataset.
