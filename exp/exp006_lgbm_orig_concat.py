"""exp006: LightGBM + 元データを学習データに追加（Issue #5 の案 A）。

元データ（約13万行）を各 fold の学習側にだけ足す。検証はコンペのデータのみで行うので、exp001 と CV を比較できる。
パラメータと特徴量は exp001 と同じ。

使い方:
    python exp/exp006_lgbm_orig_concat.py --sample   # 5万行で動作確認（記録・提出ファイルは作らない）
    python exp/exp006_lgbm_orig_concat.py            # 全量で実行
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.common import CAT_COLS, SEED, TARGET, load_data, load_original, log_experiment
from src.models import run_lgbm_cv, save_cv_result

EXP_ID = "exp006"
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
    for col in CAT_COLS:  # カテゴリの並びを揃えないと concat で category 型が崩れる
        assert set(orig[col].cat.categories) == set(train[col].cat.categories), col
        orig[col] = orig[col].cat.set_categories(train[col].cat.categories)
    features = [c for c in train.columns if c not in ("id", TARGET)]
    print(f"コンペ {len(train)} 行 + 元データ {len(orig)} 行")

    res = run_lgbm_cv(train, test, features, PARAMS, extra_train=orig)
    if sample:
        return

    save_cv_result(EXP_ID, test, res)
    log_experiment(
        EXP_ID, "LightGBM",
        f"exp001 + 元データ {len(orig)} 行を各 fold の学習側に追加 / パラメータは exp001 と同じ "
        f"/ best_iter 平均 {int(np.mean(res.best_iters))}",
        res.fold_scores, note="Issue #5 案A",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true")
    main(parser.parse_args().sample)
