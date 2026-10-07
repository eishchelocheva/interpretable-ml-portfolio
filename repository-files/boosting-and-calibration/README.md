# Gradient Boosting and Probability Calibration

This project implements a binary gradient-boosting classifier and compares it with established boosting libraries on a temporal train/validation/test split.

## Topics covered

- gradient boosting implemented from scratch;
- early stopping and validation-based model selection;
- categorical-feature handling;
- Bernoulli and Bayesian bootstrap strategies;
- feature subsampling and GOSS;
- feature quantization;
- Optuna hyperparameter optimization;
- XGBoost, LightGBM, and CatBoost comparisons;
- feature importance;
- probability calibration with isotonic regression;
- ROC AUC, log loss, Brier score, and calibration curves.

## Main observations

- Boosting outperforms the tested random-forest configuration on this dataset, although the result is dataset- and protocol-dependent.
- LightGBM provides a strong speed–quality trade-off for the tabular experiments.
- Ranking quality and probability quality are separate objectives: isotonic calibration changes ROC AUC very little but improves log loss and Brier score.
- Model comparison is performed on a temporal split rather than a random split to better expose distribution change over time.

## Files

- `boosting.py` — custom `BoostingClassifier` implementation;
- `boosting_calibration.ipynb` — experiments, library comparisons, tuning, and calibration analysis.

## Data availability

The notebook expects the source dataset at:

```text
data/hw6_dataset.pq
```

The dataset was distributed as part of a course assignment and is not included in this repository. Saved notebook outputs are retained so that the methodology and results can be reviewed without access to the original file.

## Running the notebook

Install the dependencies from the repository-level `requirements.txt`, place an authorized copy of the dataset at the path above, and open the notebook from the repository root.
