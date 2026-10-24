"""OOF 予測を使ったアンサンブル（順位平均の重みを決める）。"""

import itertools

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from src.common import OUTPUT_DIR


def load_preds(exp_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """各実験の OOF と test 予測を、0〜1 に正規化した順位に変換して (行数, 実験数) の配列で返す。"""
    to_rank = lambda p: rankdata(p) / len(p)
    oof = np.column_stack([to_rank(np.load(OUTPUT_DIR / e / "oof.npy")) for e in exp_ids])
    test = np.column_stack([to_rank(np.load(OUTPUT_DIR / e / "test_pred.npy")) for e in exp_ids])
    return oof, test


def weight_grid(n_models: int, step: float = 0.05) -> np.ndarray:
    """合計 1・非負の重みの組み合わせを step 刻みですべて返す。"""
    k = round(1 / step)
    grid = [c for c in itertools.product(range(k + 1), repeat=n_models) if sum(c) == k]
    return np.array(grid) / k


def best_weights(oof: np.ndarray, y: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, float]:
    """grid の中で OOF AUC が最大になる重みを返す。"""
    scores = [roc_auc_score(y, oof @ w) for w in grid]
    i = int(np.argmax(scores))
    return grid[i], scores[i]


def nested_cv(oof: np.ndarray, y: np.ndarray, folds: list, grid: np.ndarray) -> pd.DataFrame:
    """重みを 4 fold で決めて残り 1 fold で評価し、重みの決め方の過学習を確認する。"""
    rows = []
    for k, (_, va_idx) in enumerate(folds):
        fit_idx = np.setdiff1d(np.arange(len(y)), va_idx)
        w, _ = best_weights(oof[fit_idx], y[fit_idx], grid)
        rows.append({"fold": k + 1, "weights": w, "auc": roc_auc_score(y[va_idx], oof[va_idx] @ w)})
    return pd.DataFrame(rows)
