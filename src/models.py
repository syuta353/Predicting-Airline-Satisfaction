"""モデルの学習（共通の CV 分割で OOF と test 予測を作る）。"""

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.common import N_SPLITS, TARGET, get_folds


@dataclass
class CVResult:
    oof: np.ndarray
    test_pred: np.ndarray
    fold_scores: list[float]
    best_iters: list[int]
    importance: pd.Series = field(default_factory=pd.Series)


def run_lgbm_cv(train: pd.DataFrame, test: pd.DataFrame, features: list[str], params: dict) -> CVResult:
    """LightGBM を 5分割 CV で学習し、OOF・test 予測・各 fold の AUC を返す。"""
    X, y = train[features], train[TARGET]
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    fold_scores, best_iters = [], []
    importance = pd.Series(0.0, index=features)
    for fold, (tr_idx, va_idx) in enumerate(get_folds(y)):
        dtrain = lgb.Dataset(X.iloc[tr_idx], y.iloc[tr_idx])
        dvalid = lgb.Dataset(X.iloc[va_idx], y.iloc[va_idx])
        model = lgb.train(
            params, dtrain, num_boost_round=10_000, valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)],
        )
        oof[va_idx] = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration)
        test_pred += model.predict(test[features], num_iteration=model.best_iteration) / N_SPLITS
        importance += model.feature_importance("gain") / N_SPLITS
        score = roc_auc_score(y.iloc[va_idx], oof[va_idx])
        fold_scores.append(score)
        best_iters.append(model.best_iteration)
        print(f"fold {fold + 1}: AUC={score:.5f}  best_iter={model.best_iteration}", flush=True)

    print(f"CV mean={np.mean(fold_scores):.5f}  std={np.std(fold_scores):.5f}  OOF AUC={roc_auc_score(y, oof):.5f}")
    return CVResult(oof, test_pred, fold_scores, best_iters, importance.sort_values(ascending=False))
