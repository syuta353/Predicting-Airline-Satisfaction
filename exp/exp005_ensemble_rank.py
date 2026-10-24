"""exp005: LightGBM / XGBoost / CatBoost の順位平均アンサンブル（Issue #3）。

各実験の OOF を順位に変換し、合計 1 の重みを 0.05 刻みで総当たりして OOF AUC が最大になるものを選ぶ。
重みの決め方の過学習は、4 fold で重みを決めて残り 1 fold で評価するネスト CV で確認する。

使い方:
    python exp/exp005_ensemble_rank.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.common import DATA_DIR, OUTPUT_DIR, TARGET, get_folds, log_experiment, save_submission
from src.ensemble import best_weights, load_preds, nested_cv, weight_grid

EXP_ID = "exp005"
BASE_EXPS = ["exp001", "exp003", "exp004"]  # LightGBM / XGBoost / CatBoost


def fold_aucs(y, pred, folds):
    return [roc_auc_score(y[va], pred[va]) for _, va in folds]


def main() -> None:
    y = pd.read_csv(DATA_DIR / "train.csv", usecols=[TARGET])[TARGET].astype(int).values
    test = pd.read_csv(DATA_DIR / "test.csv", usecols=["id"])
    folds = get_folds(pd.Series(y))
    oof, test_rank = load_preds(BASE_EXPS)
    grid = weight_grid(len(BASE_EXPS))

    equal = np.full(len(BASE_EXPS), 1 / len(BASE_EXPS))
    print(f"均等重み          CV={np.mean(fold_aucs(y, oof @ equal, folds)):.5f}")

    w, _ = best_weights(oof, y, grid)
    scores = fold_aucs(y, oof @ w, folds)
    print(f"最適重み {dict(zip(BASE_EXPS, w.round(2)))}  CV={np.mean(scores):.5f}")

    nested = nested_cv(oof, y, folds, grid)
    print("ネスト CV（重みを 4 fold で決めて残り 1 fold で評価）:")
    print(nested.assign(weights=nested["weights"].map(lambda v: v.round(2).tolist())).to_string(index=False))
    print(f"ネスト CV 平均={nested['auc'].mean():.5f}")

    out_dir = OUTPUT_DIR / EXP_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "oof.npy", oof @ w)
    np.save(out_dir / "test_pred.npy", test_rank @ w)
    print(f"saved: {save_submission(EXP_ID, test, test_rank @ w)}")
    weights = ", ".join(f"{e}={x:.2f}" for e, x in zip(BASE_EXPS, w))
    log_experiment(
        EXP_ID, "順位平均アンサンブル",
        f"LGB(exp001) / XGB(exp003) / Cat(exp004) の順位平均。重み {weights}（0.05 刻みで OOF AUC 最大）",
        scores, note=f"Issue #3。ネスト CV {nested['auc'].mean():.5f}",
    )


if __name__ == "__main__":
    main()
