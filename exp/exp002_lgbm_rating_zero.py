"""exp002: LightGBM + 評価項目の 0（該当なし）を扱う特徴量（Issue #1）。

exp001 と同じパラメータで、src/features.py の add_rating_zero_features を追加した。

使い方:
    python exp/exp002_lgbm_rating_zero.py --sample   # 5万行で動作確認（記録・提出ファイルは作らない）
    python exp/exp002_lgbm_rating_zero.py            # 全量で実行
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.common import OUTPUT_DIR, SEED, TARGET, load_data, log_experiment, save_submission
from src.features import add_rating_zero_features
from src.models import run_lgbm_cv

EXP_ID = "exp002"
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
    train, test = add_rating_zero_features(train), add_rating_zero_features(test)
    features = [c for c in train.columns if c not in ("id", TARGET)]
    print(f"特徴量 {len(features)} 列")

    res = run_lgbm_cv(train, test, features, PARAMS)
    print("重要度（gain）上位 15:")
    print(res.importance.head(15).round(0).to_string())
    if sample:
        return

    out_dir = OUTPUT_DIR / EXP_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "oof.npy", res.oof)
    np.save(out_dir / "test_pred.npy", res.test_pred)
    res.importance.to_csv(out_dir / "importance.csv", header=["gain"])
    path = save_submission(EXP_ID, test, res.test_pred)
    print(f"saved: {path}")
    log_experiment(
        EXP_ID, "LightGBM",
        f"exp001 + 評価0の特徴量（is0 フラグ・0 の個数・0 を除いた平均/最小/最大/標準偏差・低/高評価の個数）{len(features)}列 "
        f"/ パラメータは exp001 と同じ / best_iter 平均 {int(np.mean(res.best_iters))}",
        res.fold_scores, note="Issue #1",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true")
    main(parser.parse_args().sample)
