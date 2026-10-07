# Interpretable ML Portfolio

Selected machine-learning projects focused on model interpretation, gradient boosting, probability calibration, and neural-network fundamentals.

The repository documents not only final metrics but also the reasoning behind model selection, evaluation protocols, preprocessing choices, and limitations. The projects originated from coursework at the HSE Faculty of Computer Science and were reorganized into standalone portfolio studies.

## Projects

| Project | Topics | Highlights |
|---|---|---|
| [Explainable Machine Learning](explainable-ml/) | ICE, PDP, ALE, permutation importance, occlusion, SHAP, LIME | Comparison of global and local explanations for linear regression and gradient boosting; custom compact LIME implementation; stability analysis |
| [Gradient Boosting and Calibration](boosting-and-calibration/) | Custom boosting, categorical features, sampling, quantization, Optuna, XGBoost, LightGBM, CatBoost | From-scratch binary gradient boosting and comparison of discrimination and probability calibration |
| [Neural Network from Scratch](neural-network-from-scratch/) | Backpropagation, layers, activations, losses, optimizers, regularization | NumPy framework with manual backward passes; final test MSE of **74.59**, compared with **89.75** for Ridge regression |

## Technical stack

- Python, NumPy, pandas, SciPy;
- scikit-learn;
- SHAP, LIME, PyALE;
- XGBoost, LightGBM, CatBoost;
- Optuna;
- Matplotlib and seaborn.

## Repository structure

```text
interpretable-ml-portfolio/
├── explainable-ml/
├── boosting-and-calibration/
├── neural-network-from-scratch/
├── requirements.txt
└── README.md
```

Each project directory contains its own description and one or more notebooks with saved results. Large datasets and generated model artifacts are intentionally excluded.

## Installation

Clone the repository and install the shared dependencies:

```bash
git clone https://github.com/eishchelocheva/interpretable-ml-portfolio.git
cd interpretable-ml-portfolio
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Open the notebooks from the repository root so that relative imports and data paths resolve consistently.

## Reproducibility and data

- The explainability notebook downloads the World Happiness Report data from the URL recorded in the notebook.
- The boosting notebook expects `data/hw6_dataset.pq`. The dataset is not redistributed because it was supplied for a course assignment; saved outputs are retained for review.
- The neural-network regression notebook downloads the public YearPredictionMSD dataset from the UCI Machine Learning Repository on first use.

Random seeds are fixed where applicable. Some experiments are computationally intensive, and saved notebook outputs are included to make the results inspectable without rerunning every model.

## Notes on interpretation

Model explanations describe the behaviour of a fitted model, not causal relationships. The explainability project also discusses target leakage associated with the `Dystopia + residual` feature and treats the dataset as a methodological demonstration rather than evidence of real-world predictive performance.

## Author

**Ekaterina Shchelocheva**  
BSc student in Applied Mathematics and Computer Science  
Faculty of Computer Science, HSE University
