"""exp009: exp008b + 元データで学習した教師モデルの予測（Issue #10）。

元データだけで学習した LightGBM の予測 logit（orig_teacher_logit）を特徴量に足す。
それ以外の特徴量とパラメータは exp008b（元データ TE + 路線の件数・fold 内 TE・路線プロファイル）と同じ。

使い方:
    python exp/exp009_lgbm_teacher.py --sample   # 5万行で動作確認（記録・提出ファイルは作らない）
    python exp/exp009_lgbm_teacher.py            # 全量で実行
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.common import SEED, TARGET, load_data, load_original, log_experiment
from src.features import (
    ROUTE_KEY, add_original_te, add_original_teacher, add_route_count, add_route_profile, make_fold_target_encoder,
)
from src.models import run_lgbm_cv, save_cv_result

EXP_ID = "exp009"
PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": SEED,
    "verbosity": -1,
    "n_jobs": -1,
}


def main(sample: bool) -> None:
    train, test = load_data(nrows=50_000 if sample else None)
    orig = load_original()
    base = [c for c in train.columns if c not in ("id", TARGET)]
    train, test = add_original_teacher(train, test, orig, base)
    train, test = add_original_te(train, orig, base), add_original_te(test, orig, base)
    train, test = add_route_count(train, test)
    train, test = add_route_profile(train, test)
    features = [c for c in train.columns if c not in ("id", TARGET)]
    print(f"特徴量 {len(features)} 列 + fold 内 TE 1 列")

    res = run_lgbm_cv(train, test, features, PARAMS, fold_fn=make_fold_target_encoder([ROUTE_KEY]))
    print("重要度（gain）上位 15:")
    print(res.importance.head(15).round(0).to_string())
    if sample:
        return

    save_cv_result(EXP_ID, test, res)
    log_experiment(
        EXP_ID, "LightGBM",
        f"exp008b + 元データの教師モデルの予測 logit。計 {len(features) + 1} 列 / パラメータは exp008b と同じ "
        f"/ best_iter 平均 {int(np.mean(res.best_iters))}",
        res.fold_scores, note="Issue #10",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true")
    main(parser.parse_args().sample)
