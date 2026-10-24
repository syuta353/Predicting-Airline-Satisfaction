"""exp003: XGBoost ベースライン（元の特徴量21列、カテゴリ列は category 型のまま）（Issue #2）。

使い方:
    python exp/exp003_xgb_baseline.py --sample   # 5万行で動作確認（記録・提出ファイルは作らない）
    python exp/exp003_xgb_baseline.py            # 全量で実行
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.common import SEED, TARGET, load_data, log_experiment
from src.models import run_xgb_cv, save_cv_result

EXP_ID = "exp003"
PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "learning_rate": 0.05,
    "max_depth": 8,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "max_cat_to_onehot": 1,
    "seed": SEED,
    "nthread": -1,
}


def main(sample: bool) -> None:
    train, test = load_data(nrows=50_000 if sample else None)
    features = [c for c in train.columns if c not in ("id", TARGET)]

    res = run_xgb_cv(train, test, features, PARAMS)
    print("重要度（total_gain）上位 10:")
    print(res.importance.head(10).round(0).to_string())
    if sample:
        return

    save_cv_result(EXP_ID, test, res)
    log_experiment(
        EXP_ID, "XGBoost",
        f"元特徴量21列 / lr=0.05, max_depth=8 / best_iter 平均 {int(np.mean(res.best_iters))}",
        res.fold_scores, note="Issue #2",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true")
    main(parser.parse_args().sample)
