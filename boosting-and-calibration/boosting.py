from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score
from typing import Iterable
from tqdm.auto import tqdm

from sklearn.base import BaseEstimator, ClassifierMixin

class Quantizer:
    def __init__(self, quantization_type=None, nbins=255):
        self.quantization_type = quantization_type
        self.nbins = nbins

    def fit(self, X: np.ndarray):
        if self.quantization_type is None:
            self._thresholds = None
            return self
        self._thresholds = []
        for j in range(X.shape[1]):
            col = X[:, j].astype(float)
            if self.quantization_type == "uniform":
                edges = np.linspace(col.min(), col.max(), self.nbins + 1)
            elif self.quantization_type == "quantile":
                edges = np.quantile(col, np.linspace(0, 1, self.nbins + 1))
            self._thresholds.append(edges[1:-1])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.quantization_type is None or self._thresholds is None:
            return X
        X = X.copy().astype(float)
        for j, thresh in enumerate(self._thresholds):
            X[:, j] = np.digitize(X[:, j], thresh)
        return X
      
class BoostingClassifier(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        base_model_class=DecisionTreeRegressor,
        base_model_params: dict | None = None,
        n_estimators: int = 20,
        learning_rate: float = 0.05,
        random_state: int | None = None,
        verbose: bool = True,
        early_stopping_rounds: int | None = None,
        eval_metric: str = "",
        cat_features: Iterable | None = None,
        bootstrap_type: str | None = "Bernoulli",
        subsample: float = 0.7,
        bagging_temperature: float = 1.0,
        rsm: float = 1.0,
        goss: bool = False,
        goss_k: float = 0.2,
        quantization_type: str | None = None,
        nbins: int = 255,
    ):
        super().__init__()

        self.base_model_class = base_model_class
        self.base_model_params = {} if base_model_params is None else base_model_params

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        self.models = [0] * (n_estimators)
        self.gammas = [0] * (n_estimators)

        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)
        self.verbose = verbose

        self.history = defaultdict(list)  # {"train_roc_auc": [], "train_loss": [], ...}

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.grad_fn = lambda y, z: y * (
            1 - self.sigmoid(y * z)
        )
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.cat_features = cat_features
        self.bootstrap_type = bootstrap_type
        self.subsample = subsample
        self.bagging_temperature = bagging_temperature
        self.rsm = rsm
        self.feature_masks = [None] * n_estimators
        self.goss = goss
        self.goss_k = goss_k
        self.quantization_type = quantization_type
        self.nbins = nbins
        self.quantizer = Quantizer(quantization_type, nbins)
        self.n_features_=0

    def partial_fit(
        self, X: np.ndarray, y: np.ndarray, train_predictions, i, mask=None, sample_weight=None
    ) -> None:
        n_feat = X.shape[1]
        feat_mask = (
            np.sort(self._rng.choice(n_feat, max(1, int(n_feat * self.rsm)), replace=False))
            if self.rsm < 1.0
            else None
        )
        self.feature_masks[i] = feat_mask
        X_fit = X[:, feat_mask] if feat_mask is not None else X
        tree = self.base_model_class(**self.base_model_params, random_state=self.random_state)
        if self.goss:
            anti = self.grad_fn(y, train_predictions)
            n = len(y)
            n_large = max(1, int(n * self.goss_k))
            order = np.argsort(-np.abs(anti))
            large_idx = order[:n_large]
            small_idx = self._rng.choice(
                order[n_large:], max(1, int((n - n_large) * self.subsample)), replace=False
            )
            factor = (1 - self.goss_k) / self.subsample
            anti_fit = np.concatenate([anti[large_idx], anti[small_idx] * factor])
            X_goss = np.vstack([X_fit[large_idx], X_fit[small_idx]])
            tree.fit(X_goss, anti_fit)
        elif mask is not None:
            anti = self.grad_fn(y[mask], train_predictions[mask])
            tree.fit(X_fit[mask], anti)
        elif sample_weight is not None:
            anti = self.grad_fn(y, train_predictions)
            tree.fit(X_fit, anti, sample_weight=sample_weight)
        else:
            anti = self.grad_fn(y, train_predictions)
            tree.fit(X_fit, anti)

        new_prediction = tree.predict(X_fit)
        gamma = self._find_optimal_gamma(y, train_predictions, new_prediction)
        self.models[i] = tree
        self.gammas[i] = gamma
        train_predictions += self.learning_rate * gamma * new_prediction

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        eval_set: tuple | None = None,
        use_best_model: bool = False,
    ) -> None:
        self.history = defaultdict(list)
        if self.cat_features is not None:
            self._cat_fit(X_train, y_train)
            X_train = self._cat_transform(X_train.copy()).astype(float)
            if eval_set is not None:
                X_valid_enc = self._cat_transform(eval_set[0].copy()).astype(float)
                eval_set = (X_valid_enc, eval_set[1])
        
        self.quantizer.fit(X_train)
        X_train = self.quantizer.transform(X_train)
        if eval_set is not None:
            eval_set = (self.quantizer.transform(eval_set[0].copy()), eval_set[1])
        self.n_features_ = X_train.shape[1]

        train_predictions = np.zeros(X_train.shape[0])
        self.classes_ = np.unique(y_train)  # не рекомендуется убирать, нужно для калибровки
        estimator_range = range(self.n_estimators)
        if self.verbose:
            estimator_range = tqdm(estimator_range)
        if self.eval_metric:
            metric_key = self.eval_metric
        elif eval_set is not None:
            metric_key = "val_roc_auc"
        else:
            metric_key = "train_roc_auc"
        lower_is_better = "loss" in metric_key

        best = np.inf if lower_is_better else -np.inf
        patience = self.early_stopping_rounds
        best_iteration = 0
        last_i = 0

        val_predictions = np.zeros(eval_set[1].shape[0]) if eval_set is not None else None
        for i in estimator_range:
            last_i = i
            if self.goss:
                self.partial_fit(X_train, y_train, train_predictions, i)
            elif self.bootstrap_type == "Bernoulli":
                mask = self._rng.random(len(y_train)) < self.subsample
                self.partial_fit(X_train, y_train, train_predictions, i, mask=mask)
            elif self.bootstrap_type == "Bayesian":
                U = self._rng.uniform(0, 1, size=len(y_train))
                w = (-np.log(U + 1e-10)) ** self.bagging_temperature
                self.partial_fit(X_train, y_train, train_predictions, i, sample_weight=w)
            else:
                self.partial_fit(X_train, y_train, train_predictions, i)

            self.history["train_roc_auc"].append(roc_auc_score(y_train, train_predictions))
            self.history["train_loss"].append(self.loss_fn(y_train, train_predictions))

            if eval_set is not None:
                X_valid, y_valid = eval_set
                mask_i = self.feature_masks[i]
                X_val_pred = X_valid[:, mask_i] if mask_i is not None else X_valid
                val_predictions += (
                    self.learning_rate * self.gammas[i] * self.models[i].predict(X_val_pred)
                )
                self.history["val_roc_auc"].append(roc_auc_score(y_valid, val_predictions))
                self.history["val_loss"].append(self.loss_fn(y_valid, val_predictions))

            if self.early_stopping_rounds:
                current = self.history[metric_key][-1]
                improved = (current < best) if lower_is_better else (current > best)
                if improved:
                    best = current
                    best_iteration = i
                    patience = self.early_stopping_rounds
                else:
                    patience -= 1
                    if patience == 0:
                        break
        self.models = self.models[: last_i + 1]
        self.gammas = self.gammas[: last_i + 1]
        self.feature_masks = self.feature_masks[: last_i + 1]
        if use_best_model and self.early_stopping_rounds:
            self.models = self.models[: best_iteration + 1]
            self.gammas = self.gammas[: best_iteration + 1]
            self.feature_masks = self.feature_masks[: best_iteration + 1]

        # чтобы было удобнее смотреть
        for key in self.history:
            self.history[key] = np.array(self.history[key])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.cat_features is not None:
            X = self._cat_transform(X.copy()).astype(float)
        
        X = self.quantizer.transform(X)
        predictions = np.zeros(X.shape[0])
        for model, gamma, mask in zip(self.models, self.gammas, self.feature_masks):
            X_pred = X[:, mask] if mask is not None else X
            predictions += self.learning_rate * gamma * model.predict(X_pred)
        proba = self.sigmoid(predictions)
        return np.column_stack([1 - proba, proba])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return np.where(proba >= 0.5, 1, -1)

    def plot_history(self, keys):
        keys = [keys] if isinstance(keys, str) else list(keys)
        plt.figure(figsize=(8, 5))
        for key in keys:
            plt.plot(self.history[key], label=key)
        plt.xlabel("iteration")
        plt.legend()
        plt.grid(True)
        plt.show()

    def _find_optimal_gamma(
        self, y: np.ndarray, old_predictions: np.ndarray, new_predictions: np.ndarray
    ) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return roc_auc_score(y == 1, self.predict_proba(X)[:, 1])

    def _cat_fit(self, X: np.array, y: np.ndarray):
        if self.cat_features is None:
            return self
        pos = (y == 1).astype(float)
        self._cat_maps = {}
        self._cat_defaults = {}
        for j in self.cat_features:
            cats, inv = np.unique(X[:, j], return_inverse=True)
            sums = np.bincount(inv, weights=pos, minlength=len(cats))
            cnts = np.bincount(inv, minlength=len(cats))
            self._cat_maps[j] = dict(zip(cats, sums / cnts))
            self._cat_defaults[j] = pos.mean()
        return self

    def _cat_transform(self, X: np.ndarray):
        if self.cat_features is None:
            return X
        for j in self.cat_features:
            mapping = self._cat_maps[j]
            default = self._cat_defaults[j]
            X[:, j] = np.array([mapping.get(v, default) for v in X[:, j]])
        return X
    def get_feature_importance(self, X: np.ndarray | None = None, y: np.ndarray | None = None, type="split"):
        importances = np.zeros(self.n_features_)
        for model, gamma, feat_mask in zip(self.models, self.gammas, self.feature_masks):
            fi = model.feature_importances_
            if feat_mask is not None:
                full_fi = np.zeros(self.n_features_)
                full_fi[feat_mask] = fi
            else:
                full_fi = fi
            importances += gamma * full_fi
        importances /= importances.sum()
        return importances
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        if self.cat_features is not None:
            X = self._cat_transform(X).astype(float)
        X = self.quantizer.transform(X)
        return X