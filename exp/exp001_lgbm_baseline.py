"""exp001: LightGBM ベースライン（元の特徴量そのまま、カテゴリ列は category 型）。

使い方:
    python exp/exp001_lgbm_baseline.py --sample   # 5万行で動作確認（記録・提出ファイルは作らない）
    python exp/exp001_lgbm_baseline.py            # 全量で実行
"""

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.common import N_SPLITS, OUTPUT_DIR, SEED, TARGET, get_folds, load_data, log_experiment, save_submission

EXP_ID = "exp001"
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
    features = [c for c in train.columns if c not in ("id", TARGET)]
    X, y = train[features], train[TARGET]

    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    fold_scores, best_iters = [], []
    for fold, (tr_idx, va_idx) in enumerate(get_folds(y)):
        dtrain = lgb.Dataset(X.iloc[tr_idx], y.iloc[tr_idx])
        dvalid = lgb.Dataset(X.iloc[va_idx], y.iloc[va_idx])
        model = lgb.train(
            PARAMS, dtrain, num_boost_round=10_000, valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)],
        )
        oof[va_idx] = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration)
        test_pred += model.predict(test[features], num_iteration=model.best_iteration) / N_SPLITS
        score = roc_auc_score(y.iloc[va_idx], oof[va_idx])
        fold_scores.append(score)
        best_iters.append(model.best_iteration)
        print(f"fold {fold + 1}: AUC={score:.5f}  best_iter={model.best_iteration}")

    print(f"CV mean={np.mean(fold_scores):.5f}  std={np.std(fold_scores):.5f}  OOF AUC={roc_auc_score(y, oof):.5f}")
    if sample:
        return

    out_dir = OUTPUT_DIR / EXP_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "oof.npy", oof)
    np.save(out_dir / "test_pred.npy", test_pred)
    path = save_submission(EXP_ID, test, test_pred)
    print(f"saved: {path}")
    log_experiment(
        EXP_ID, "LightGBM",
        f"元特徴量21列 / lr=0.05, leaves=63 / best_iter 平均 {int(np.mean(best_iters))}",
        fold_scores, note="ベースライン",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true")
    main(parser.parse_args().sample)
