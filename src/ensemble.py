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


def run_rank_ensemble(exp_id: str, base_exps: list[str], model_label: str, note: str = "") -> None:
    """base_exps の OOF を順位平均し、重みを総当たりで決めて、提出ファイル作成と実験記録まで行う。"""
    from src.common import DATA_DIR, TARGET, get_folds, log_experiment, save_submission

    y = pd.read_csv(DATA_DIR / "train.csv", usecols=[TARGET])[TARGET].astype(int).values
    test = pd.read_csv(DATA_DIR / "test.csv", usecols=["id"])
    folds = get_folds(pd.Series(y))
    oof, test_rank = load_preds(base_exps)
    grid = weight_grid(len(base_exps))
    fold_aucs = lambda p: [roc_auc_score(y[va], p[va]) for _, va in folds]

    print("OOF の相関:")
    print(pd.DataFrame(oof, columns=base_exps).corr().round(5).to_string())
    for e, col in zip(base_exps, oof.T):
        print(f"{e} 単体     CV={np.mean(fold_aucs(col)):.5f}")
    equal = np.full(len(base_exps), 1 / len(base_exps))
    print(f"均等重み          CV={np.mean(fold_aucs(oof @ equal)):.5f}")
    w, _ = best_weights(oof, y, grid)
    scores = fold_aucs(oof @ w)
    print(f"最適重み {dict(zip(base_exps, w.round(2)))}  CV={np.mean(scores):.5f}")
    nested = nested_cv(oof, y, folds, grid)
    print(nested.assign(weights=nested["weights"].map(lambda v: v.round(2).tolist())).to_string(index=False))
    print(f"ネスト CV 平均={nested['auc'].mean():.5f}")

    out_dir = OUTPUT_DIR / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "oof.npy", oof @ w)
    np.save(out_dir / "test_pred.npy", test_rank @ w)
    print(f"saved: {save_submission(exp_id, test, test_rank @ w)}")
    weights = ", ".join(f"{e}={x:.2f}" for e, x in zip(base_exps, w))
    log_experiment(
        exp_id, "順位平均アンサンブル", f"{model_label} の順位平均。重み {weights}（0.05 刻みで OOF AUC 最大）",
        scores, note=f"{note}。ネスト CV {nested['auc'].mean():.5f}",
    )


def nested_cv(oof: np.ndarray, y: np.ndarray, folds: list, grid: np.ndarray) -> pd.DataFrame:
    """重みを 4 fold で決めて残り 1 fold で評価し、重みの決め方の過学習を確認する。"""
    rows = []
    for k, (_, va_idx) in enumerate(folds):
        fit_idx = np.setdiff1d(np.arange(len(y)), va_idx)
        w, _ = best_weights(oof[fit_idx], y[fit_idx], grid)
        rows.append({"fold": k + 1, "weights": w, "auc": roc_auc_score(y[va_idx], oof[va_idx] @ w)})
    return pd.DataFrame(rows)
