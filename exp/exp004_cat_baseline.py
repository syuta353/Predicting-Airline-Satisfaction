"""exp004: CatBoost ベースライン（元の特徴量21列、カテゴリ列は CatBoost のカテゴリ処理に任せる）（Issue #2）。

使い方:
    python exp/exp004_cat_baseline.py --sample   # 5万行で動作確認（記録・提出ファイルは作らない）
    python exp/exp004_cat_baseline.py            # 全量で実行
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.common import SEED, TARGET, load_data, log_experiment
from src.models import MAX_ROUNDS, run_cat_cv, save_cv_result

EXP_ID = "exp004"
PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "iterations": MAX_ROUNDS,
    "learning_rate": 0.08,
    "depth": 7,
    "l2_leaf_reg": 3.0,
    "random_seed": SEED,
    "thread_count": -1,
}


def main(sample: bool) -> None:
    train, test = load_data(nrows=50_000 if sample else None)
    features = [c for c in train.columns if c not in ("id", TARGET)]

    res = run_cat_cv(train, test, features, PARAMS)
    print("重要度（PredictionValuesChange）上位 10:")
    print(res.importance.head(10).round(2).to_string())
    if sample:
        return

    save_cv_result(EXP_ID, test, res)
    log_experiment(
        EXP_ID, "CatBoost",
        f"元特徴量21列 / lr=0.08, depth=7 / best_iter 平均 {int(np.mean(res.best_iters))}",
        res.fold_scores, note="Issue #2",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true")
    main(parser.parse_args().sample)
