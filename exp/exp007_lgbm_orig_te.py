"""exp007: LightGBM + 元データでのターゲットエンコーディング特徴量（Issue #5 の案 B）。

各列の値ごとに、元データでの満足率（`<col>_orig_te`）と件数（`<col>_orig_cnt`）を特徴量に足す。
元データはコンペのデータと別なので、fold をまたいだリークは起きない。パラメータは exp001 と同じ。

使い方:
    python exp/exp007_lgbm_orig_te.py --sample   # 5万行で動作確認（記録・提出ファイルは作らない）
    python exp/exp007_lgbm_orig_te.py            # 全量で実行
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.common import SEED, TARGET, load_data, load_original, log_experiment
from src.features import add_original_te
from src.models import run_lgbm_cv, save_cv_result

EXP_ID = "exp007"
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
    train, test = add_original_te(train, orig, base), add_original_te(test, orig, base)
    features = [c for c in train.columns if c not in ("id", TARGET)]
    print(f"特徴量 {len(features)} 列")

    res = run_lgbm_cv(train, test, features, PARAMS)
    print("重要度（gain）上位 15:")
    print(res.importance.head(15).round(0).to_string())
    if sample:
        return

    save_cv_result(EXP_ID, test, res)
    log_experiment(
        EXP_ID, "LightGBM",
        f"exp001 + 全21列の元データ TE（満足率・件数）{len(features)}列 / パラメータは exp001 と同じ "
        f"/ best_iter 平均 {int(np.mean(res.best_iters))}",
        res.fold_scores, note="Issue #5 案B",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true")
    main(parser.parse_args().sample)
