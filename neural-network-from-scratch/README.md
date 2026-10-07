# Neural Network from Scratch

A compact neural-network framework implemented with NumPy, followed by an end-to-end regression study on the Million Song Dataset Year Prediction task.

## Highlights

- manual forward and backward propagation;
- affine, low-rank linear, Batch Normalization, and Dropout layers;
- ReLU, Sigmoid, GELU, Softmax, and LogSoftmax activations;
- MSE and cross-entropy losses;
- SGD with momentum/Nesterov updates and Adam;
- mini-batch data loading and a Sequential container;
- controlled experiments on scaling, optimizers, model capacity, regularization, and learning rate.

## Results

| Model | Test MSE |
|---|---:|
| Constant predictor | 117.63 |
| Ridge regression | 89.75 |
| NumPy neural network | **74.59** |

## Files

- `01_framework_implementation.ipynb` — framework overview and synthetic end-to-end check;
- `02_regression_experiments.ipynb` — model-development experiments and final evaluation;
- `modules/` — the NumPy framework implementation.

## Running the notebooks

Install the dependencies from the repository-level `requirements.txt`, open the notebooks from this directory, and run them in filename order. The second notebook downloads the public YearPredictionMSD dataset from the UCI Machine Learning Repository on first use; the dataset itself is not committed.

## Acknowledgement

This project originated as coursework for the HSE Faculty of Computer Science. The framework design was inspired by a related YSDA Deep Vision and Graphics assignment, as documented in `modules/base.py`. The implementation, experiments, and portfolio presentation are the author's own work.
