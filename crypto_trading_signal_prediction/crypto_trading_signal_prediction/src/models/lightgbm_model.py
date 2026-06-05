"""
LightGBM model with Optuna hyperparameter optimization.

Trains LightGBM on triple-barrier labels with proper validation, sample
weighting, and — critically — label encoding. The labeller emits labels in
{-1, 0, 1}, but LightGBM's `multiclass` objective requires contiguous integer
classes {0, ..., k-1}. This module encodes on fit and decodes on predict, and
derives `num_class` from the data so it also works when only two classes are
present (e.g. vertical_label='sign').

Author: Research Team
Date: 2024
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Any, List
import logging
import lightgbm as lgb
import optuna
from optuna.pruners import MedianPruner
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score,
)
import joblib
from pathlib import Path

from src.validation.purged_kfold import PurgedKFold
from config.model_config import MODEL_CONFIG

logger = logging.getLogger(__name__)


class LightGBMClassifier:
    """LightGBM classifier with label encoding and financial-ML defaults."""

    def __init__(
        self,
        params: Optional[Dict] = None,
        use_sample_weights: bool = True,
        random_state: int = 42,
    ):
        self.params = dict(params or MODEL_CONFIG.lightgbm.to_dict())
        self.use_sample_weights = use_sample_weights
        self.random_state = random_state
        self.model: Optional[lgb.Booster] = None
        self.feature_importance_: Optional[pd.DataFrame] = None
        self.training_history_: Dict = {}
        self.classes_: Optional[np.ndarray] = None      # original labels, sorted
        self._to_internal: Dict = {}                    # original -> 0..k-1
        self._to_original: Dict = {}                    # 0..k-1 -> original

    def _encode(self, y: pd.Series) -> np.ndarray:
        return y.map(self._to_internal).to_numpy()

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[pd.Series] = None,
        eval_set: Optional[List[Tuple]] = None,
        early_stopping_rounds: int = 50,
        verbose: int = 100,
    ) -> 'LightGBMClassifier':
        logger.info("Training LightGBM on %d samples, %d features", len(X), X.shape[1])

        # --- Label encoding: original {-1,0,1,...} -> internal {0..k-1} ------
        self.classes_ = np.sort(y.dropna().unique())
        k = len(self.classes_)
        if k < 2:
            raise ValueError(f"Need >=2 classes to train, found {k}: {self.classes_}")
        self._to_internal = {c: i for i, c in enumerate(self.classes_)}
        self._to_original = {i: c for c, i in self._to_internal.items()}
        y_enc = self._encode(y)

        # Align objective / num_class to the data actually present.
        params = dict(self.params)
        if k == 2:
            params['objective'] = 'binary'
            params.pop('num_class', None)
            params['metric'] = 'binary_logloss'   # overwrite any multiclass metric
        else:
            params['objective'] = 'multiclass'
            params['num_class'] = k
            params['metric'] = 'multi_logloss'

        # is_unbalance and explicit sample weights both correct for imbalance;
        # using both double-counts. Prefer the weights the caller supplies.
        if self.use_sample_weights and sample_weight is not None:
            params.pop('is_unbalance', None)
            params.pop('class_weight', None)

        train_data = lgb.Dataset(
            X, label=y_enc,
            weight=sample_weight.to_numpy() if (self.use_sample_weights and sample_weight is not None) else None,
            free_raw_data=False,
        )

        valid_sets, valid_names = [train_data], ['train']
        if eval_set:
            for i, item in enumerate(eval_set):
                X_val, y_val = item[0], item[1]
                w_val = item[2] if len(item) > 2 else None
                val_data = lgb.Dataset(
                    X_val, label=self._encode(y_val),
                    weight=(w_val.to_numpy() if (self.use_sample_weights and w_val is not None) else None),
                    reference=train_data, free_raw_data=False,
                )
                valid_sets.append(val_data)
                valid_names.append(f'valid_{i}')

        callbacks = [lgb.record_evaluation(self.training_history_)]
        if verbose and verbose > 0:
            callbacks.append(lgb.log_evaluation(period=verbose))
        else:
            callbacks.append(lgb.log_evaluation(period=0))
        if early_stopping_rounds and early_stopping_rounds > 0:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))

        self.model = lgb.train(
            params=params, train_set=train_data,
            valid_sets=valid_sets, valid_names=valid_names, callbacks=callbacks,
        )

        self.feature_importance_ = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importance(importance_type='gain'),
            'split_importance': self.model.feature_importance(importance_type='split'),
        }).sort_values('importance', ascending=False)

        logger.info("Training complete. Best iteration: %s", self.model.best_iteration)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained yet")
        proba = self.model.predict(X)
        if proba.ndim == 1:  # binary objective returns P(class 1)
            proba = np.column_stack([1 - proba, proba])
        return proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict original-space labels ({-1, 0, 1, ...})."""
        internal = np.argmax(self.predict_proba(X), axis=1)
        return np.vectorize(self._to_original.get)(internal)

    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        if self.feature_importance_ is None:
            raise ValueError("Model not trained yet")
        return self.feature_importance_.head(top_n) if top_n else self.feature_importance_

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        joblib.dump({
            'params': self.params,
            'feature_importance': self.feature_importance_,
            'training_history': self.training_history_,
            'classes_': self.classes_,
            'to_internal': self._to_internal,
        }, str(path.with_suffix('.metadata.pkl')))
        logger.info("Model saved to %s", path)

    def load(self, path: Path):
        path = Path(path)
        self.model = lgb.Booster(model_file=str(path))
        meta = joblib.load(str(path.with_suffix('.metadata.pkl')))
        self.params = meta['params']
        self.feature_importance_ = meta['feature_importance']
        self.training_history_ = meta['training_history']
        self.classes_ = meta['classes_']
        self._to_internal = meta['to_internal']
        self._to_original = {i: c for c, i in self._to_internal.items()}
        logger.info("Model loaded from %s", path)


class LightGBMOptimizer:
    """Optuna hyperparameter search over a purged CV splitter."""

    def __init__(
        self,
        cv_splitter,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        metric: str = 'f1_weighted',
        direction: str = 'maximize',
        random_state: int = 42,
    ):
        self.cv_splitter = cv_splitter
        self.n_trials = n_trials
        self.timeout = timeout
        self.metric = metric
        self.direction = direction
        self.random_state = random_state
        self.study: Optional[optuna.Study] = None
        self.best_params_: Optional[Dict] = None

    def optimize(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[pd.Series] = None,
    ) -> Dict:
        logger.info("Optuna: %d trials, metric=%s", self.n_trials, self.metric)
        self.study = optuna.create_study(
            direction=self.direction,
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )

        def objective(trial: optuna.Trial) -> float:
            params = self._suggest_params(trial)
            scores = []
            for fold, (tr, va) in enumerate(self.cv_splitter.split(X)):
                X_tr, X_va = X.iloc[tr], X.iloc[va]
                y_tr, y_va = y.iloc[tr], y.iloc[va]
                w_tr = sample_weight.iloc[tr] if sample_weight is not None else None
                w_va = sample_weight.iloc[va] if sample_weight is not None else None

                # A fold missing a class would corrupt the encoding; skip it.
                if y_tr.dropna().nunique() < 2:
                    continue

                model = LightGBMClassifier(params=params)
                model.fit(X_tr, y_tr, sample_weight=w_tr,
                          eval_set=[(X_va, y_va, w_va)],
                          early_stopping_rounds=50, verbose=0)

                y_pred = model.predict(X_va)
                if self.metric == 'f1_weighted':
                    score = f1_score(y_va, y_pred, average='weighted')
                elif self.metric == 'accuracy':
                    score = accuracy_score(y_va, y_pred)
                elif self.metric == 'roc_auc':
                    proba = model.predict_proba(X_va)
                    score = roc_auc_score(y_va, proba, multi_class='ovr',
                                          average='weighted', labels=model.classes_)
                else:
                    raise ValueError(f"Unknown metric: {self.metric}")
                scores.append(score)

                trial.report(float(np.mean(scores)), fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return float(np.mean(scores)) if scores else float('-inf')

        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        self.best_params_ = self.study.best_params
        logger.info("Best %s = %.4f", self.metric, self.study.best_value)
        return self.best_params_

    def _suggest_params(self, trial: optuna.Trial) -> Dict:
        # objective / num_class are set inside fit() from the data.
        return {
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'subsample_freq': 1,
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'n_estimators': 1000,
            'random_state': self.random_state,
            'verbose': -1,
        }

    def get_optimization_history(self) -> pd.DataFrame:
        if self.study is None:
            raise ValueError("No optimization run yet")
        df = self.study.trials_dataframe()
        return df.sort_values('value', ascending=(self.direction == 'minimize'))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.random.seed(42)
    n, f = 4000, 30
    dates = pd.date_range('2023-01-01', periods=n, freq='1h')
    X = pd.DataFrame(np.random.randn(n, f),
                     columns=[f'feature_{i}' for i in range(f)], index=dates)
    # Labels in {-1, 0, 1} — the labeller's native space.
    y = pd.Series(np.random.choice([-1, 0, 1], size=n, p=[0.3, 0.4, 0.3]),
                  index=dates, name='label')
    X['event_end_time'] = dates + pd.to_timedelta(np.random.randint(1, 24, n), unit='h')
    w = pd.Series(np.random.uniform(0.5, 2.0, n), index=dates)

    cut = int(0.8 * n)
    feats = [c for c in X.columns if c != 'event_end_time']
    model = LightGBMClassifier()
    model.fit(X[feats].iloc[:cut], y.iloc[:cut], sample_weight=w.iloc[:cut],
              eval_set=[(X[feats].iloc[cut:], y.iloc[cut:], w.iloc[cut:])],
              early_stopping_rounds=30, verbose=0)
    y_pred = model.predict(X[feats].iloc[cut:])
    print(f"classes={model.classes_}  acc={accuracy_score(y.iloc[cut:], y_pred):.3f}  "
          f"f1={f1_score(y.iloc[cut:], y_pred, average='weighted'):.3f}")