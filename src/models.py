"""モデルの学習（共通の CV 分割で OOF と test 予測を作る）。"""

from dataclasses import dataclass, field
from typing import Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

from src.common import CAT_COLS, N_SPLITS, OUTPUT_DIR, TARGET, get_folds, save_submission

EARLY_STOPPING = 200
MAX_ROUNDS = 10_000


@dataclass
class CVResult:
    oof: np.ndarray
    test_pred: np.ndarray
    fold_scores: list[float]
    best_iters: list[int]
    importance: pd.Series = field(default_factory=pd.Series)


# 1 fold 分の学習関数：(X_tr, y_tr, X_va, y_va, X_test, params) -> (va_pred, test_pred, best_iter, importance)
FitFn = Callable[..., tuple[np.ndarray, np.ndarray, int, pd.Series]]


def _run_cv(train: pd.DataFrame, test: pd.DataFrame, features: list[str], params: dict, fit_fn: FitFn) -> CVResult:
    X, y = train[features], train[TARGET]
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    fold_scores, best_iters = [], []
    importance = pd.Series(0.0, index=features)
    for fold, (tr_idx, va_idx) in enumerate(get_folds(y)):
        va_pred, te_pred, best_iter, imp = fit_fn(
            X.iloc[tr_idx], y.iloc[tr_idx], X.iloc[va_idx], y.iloc[va_idx], test[features], params
        )
        oof[va_idx] = va_pred
        test_pred += te_pred / N_SPLITS
        importance = importance.add(imp.reindex(features).fillna(0) / N_SPLITS)
        score = roc_auc_score(y.iloc[va_idx], va_pred)
        fold_scores.append(score)
        best_iters.append(best_iter)
        print(f"fold {fold + 1}: AUC={score:.5f}  best_iter={best_iter}", flush=True)

    print(f"CV mean={np.mean(fold_scores):.5f}  std={np.std(fold_scores):.5f}  OOF AUC={roc_auc_score(y, oof):.5f}")
    return CVResult(oof, test_pred, fold_scores, best_iters, importance.sort_values(ascending=False))


def _fit_lgbm(X_tr, y_tr, X_va, y_va, X_te, params):
    model = lgb.train(
        params, lgb.Dataset(X_tr, y_tr), num_boost_round=MAX_ROUNDS, valid_sets=[lgb.Dataset(X_va, y_va)],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False), lgb.log_evaluation(0)],
    )
    it = model.best_iteration
    imp = pd.Series(model.feature_importance("gain"), index=X_tr.columns)
    return model.predict(X_va, num_iteration=it), model.predict(X_te, num_iteration=it), it, imp


def _fit_xgb(X_tr, y_tr, X_va, y_va, X_te, params):
    dtr = xgb.DMatrix(X_tr, y_tr, enable_categorical=True)
    dva = xgb.DMatrix(X_va, y_va, enable_categorical=True)
    model = xgb.train(
        params, dtr, num_boost_round=MAX_ROUNDS, evals=[(dva, "valid")],
        early_stopping_rounds=EARLY_STOPPING, verbose_eval=False,
    )
    rng = (0, model.best_iteration + 1)
    imp = pd.Series(model.get_score(importance_type="total_gain"))
    dte = xgb.DMatrix(X_te, enable_categorical=True)
    return model.predict(dva, iteration_range=rng), model.predict(dte, iteration_range=rng), model.best_iteration + 1, imp


def _fit_cat(X_tr, y_tr, X_va, y_va, X_te, params):
    # CatBoost はカテゴリ列を文字列で受け取る
    cat = [c for c in CAT_COLS if c in X_tr.columns]
    to_str = lambda X: X.astype({c: str for c in cat})
    model = CatBoostClassifier(**params)
    model.fit(
        Pool(to_str(X_tr), y_tr, cat_features=cat), eval_set=Pool(to_str(X_va), y_va, cat_features=cat),
        early_stopping_rounds=EARLY_STOPPING, use_best_model=True, verbose=False,
    )
    imp = pd.Series(model.get_feature_importance(), index=X_tr.columns)
    va_pred = model.predict_proba(Pool(to_str(X_va), cat_features=cat))[:, 1]
    te_pred = model.predict_proba(Pool(to_str(X_te), cat_features=cat))[:, 1]
    return va_pred, te_pred, model.get_best_iteration() + 1, imp


def save_cv_result(exp_id: str, test: pd.DataFrame, res: CVResult) -> None:
    """OOF・test 予測・重要度・提出ファイルを outputs/<exp_id>/ に保存する。"""
    out_dir = OUTPUT_DIR / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "oof.npy", res.oof)
    np.save(out_dir / "test_pred.npy", res.test_pred)
    res.importance.to_csv(out_dir / "importance.csv", header=["importance"])
    print(f"saved: {save_submission(exp_id, test, res.test_pred)}")


def run_lgbm_cv(train: pd.DataFrame, test: pd.DataFrame, features: list[str], params: dict) -> CVResult:
    """LightGBM を 5分割 CV で学習する。重要度は gain。"""
    return _run_cv(train, test, features, params, _fit_lgbm)


def run_xgb_cv(train: pd.DataFrame, test: pd.DataFrame, features: list[str], params: dict) -> CVResult:
    """XGBoost を 5分割 CV で学習する（category 型の列はそのまま渡す）。重要度は total_gain。"""
    return _run_cv(train, test, features, params, _fit_xgb)


def run_cat_cv(train: pd.DataFrame, test: pd.DataFrame, features: list[str], params: dict) -> CVResult:
    """CatBoost を 5分割 CV で学習する（CAT_COLS をカテゴリ列として渡す）。params には iterations を含める。"""
    return _run_cv(train, test, features, params, _fit_cat)
