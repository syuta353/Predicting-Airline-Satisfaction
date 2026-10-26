"""exp008: exp007 + Flight Distance を「路線 ID」として扱う特徴量（Issue #9）。

ブロックごとの効果を見るため、3 つの変形を順に足していく：
    a: 路線の件数（fd_count）+ fold 内ターゲットエンコーディング（flight_distance_te）
    b: a + 路線プロファイル（路線ごとの他の列の平均 fd_mean_*。ラベル不使用）
    c: b + 残差（その行の値 − 路線平均 fd_resid_*）
パラメータは exp007 と同じ。

使い方:
    python exp/exp008_lgbm_route.py --variant a --sample   # 5万行で動作確認（記録・提出ファイルは作らない）
    python exp/exp008_lgbm_route.py --variant a            # 全量で実行（ID は exp008a）
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.common import SEED, TARGET, load_data, load_original, log_experiment
from src.features import ROUTE_KEY, add_original_te, add_route_count, add_route_profile, make_fold_target_encoder
from src.models import run_lgbm_cv, save_cv_result

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
DESCRIPTIONS = {
    "a": "路線の件数 + fold 内 TE",
    "b": "路線の件数 + fold 内 TE + 路線プロファイル（平均 18 列）",
    "c": "路線の件数 + fold 内 TE + 路線プロファイル + 残差 14 列",
}


def main(variant: str, sample: bool) -> None:
    exp_id = f"exp008{variant}"
    train, test = load_data(nrows=50_000 if sample else None)
    orig = load_original()
    base = [c for c in train.columns if c not in ("id", TARGET)]
    train, test = add_original_te(train, orig, base), add_original_te(test, orig, base)  # exp007 と同じ
    train, test = add_route_count(train, test)
    if variant in ("b", "c"):
        train, test = add_route_profile(train, test, residual=(variant == "c"))
    features = [c for c in train.columns if c not in ("id", TARGET)]
    print(f"{exp_id}: 特徴量 {len(features)} 列 + fold 内 TE 1 列")

    res = run_lgbm_cv(train, test, features, PARAMS, fold_fn=make_fold_target_encoder([ROUTE_KEY]))
    print("重要度（gain）上位 15:")
    print(res.importance.head(15).round(0).to_string())
    if sample:
        return

    save_cv_result(exp_id, test, res)
    log_experiment(
        exp_id, "LightGBM",
        f"exp007 + {DESCRIPTIONS[variant]}。計 {len(features) + 1} 列 / パラメータは exp007 と同じ "
        f"/ best_iter 平均 {int(np.mean(res.best_iters))}",
        res.fold_scores, note=f"Issue #9 変形{variant}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["a", "b", "c"], required=True)
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()
    main(args.variant, args.sample)
