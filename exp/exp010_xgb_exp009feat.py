"""exp010: XGBoost + exp009 の特徴量（Issue #8）。

特徴量は exp009 と同じ（src/features.py の build_exp009_features）。パラメータは exp003 と同じ。

使い方:
    python exp/exp010_xgb_exp009feat.py --sample   # 5万行で動作確認（記録・提出ファイルは作らない）
    python exp/exp010_xgb_exp009feat.py            # 全量で実行
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.common import SEED, load_data, load_original, log_experiment
from src.features import build_exp009_features
from src.models import run_xgb_cv, save_cv_result

EXP_ID = "exp010"
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
    train, test, features, fold_fn = build_exp009_features(train, test, load_original())
    print(f"特徴量 {len(features)} 列 + fold 内 TE 1 列")

    res = run_xgb_cv(train, test, features, PARAMS, fold_fn=fold_fn)
    if sample:
        return

    save_cv_result(EXP_ID, test, res)
    log_experiment(
        EXP_ID, "XGBoost",
        f"exp009 の特徴量（{len(features) + 1} 列）/ パラメータは exp003 と同じ / best_iter 平均 {int(np.mean(res.best_iters))}",
        res.fold_scores, note="Issue #8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true")
    main(parser.parse_args().sample)
