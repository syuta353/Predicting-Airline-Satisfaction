"""exp013〜015: exp009 / exp010 / exp011 を 10分割で学習し直す（Issue #11）。

特徴量とパラメータは元の実験と同じで、分割数だけ 5 → 10 に変える。各 fold の学習データが 80% → 90% に増える。
10分割の OOF は「90% で学習したモデルの予測」なので、test 予測（10 モデルの平均）と同じ条件で評価できる。

記録表の CV は、他の実験と比べられるように、10分割の OOF をいつもの 5分割の区切りで評価した値にする。

使い方:
    python exp/exp013_10fold.py --model lgbm --sample   # 5万行で動作確認（記録・提出ファイルは作らない）
    python exp/exp013_10fold.py --model lgbm            # 全量（exp013）。xgb → exp014、cat → exp015
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))
from exp009_lgbm_teacher import PARAMS as LGBM_PARAMS
from exp010_xgb_exp009feat import PARAMS as XGB_PARAMS
from exp011_cat_exp009feat import PARAMS as CAT_PARAMS
from src.common import TARGET, get_folds, load_data, load_original, log_experiment
from src.features import build_exp009_features
from src.models import run_cat_cv, run_lgbm_cv, run_xgb_cv, save_cv_result

N_SPLITS_FINAL = 10
MODELS = {
    # model: (実験 ID, 元の実験, モデル名, 学習関数, パラメータ)
    "lgbm": ("exp013", "exp009", "LightGBM", run_lgbm_cv, LGBM_PARAMS),
    "xgb": ("exp014", "exp010", "XGBoost", run_xgb_cv, XGB_PARAMS),
    "cat": ("exp015", "exp011", "CatBoost", run_cat_cv, CAT_PARAMS),
}


def main(model: str, sample: bool) -> None:
    exp_id, base_exp, label, run_cv, params = MODELS[model]
    train, test = load_data(nrows=50_000 if sample else None)
    train, test, features, fold_fn = build_exp009_features(train, test, load_original())

    res = run_cv(train, test, features, params, fold_fn=fold_fn, n_splits=N_SPLITS_FINAL)
    y = train[TARGET].values
    scores5 = [roc_auc_score(y[va], res.oof[va]) for _, va in get_folds(train[TARGET])]
    print(f"{exp_id}: 10分割の OOF を 5分割の区切りで評価  CV={np.mean(scores5):.5f}  {[round(s, 5) for s in scores5]}")
    if sample:
        return

    save_cv_result(exp_id, test, res)
    log_experiment(
        exp_id, label,
        f"{base_exp} を 10分割で学習（特徴量・パラメータは同じ）/ best_iter 平均 {int(np.mean(res.best_iters))} "
        f"/ 10分割の fold 平均 {np.mean(res.fold_scores):.5f}",
        scores5, note=f"Issue #11。Fold 列は 10分割の OOF を 5分割の区切りで評価した値（{base_exp} と比較用）",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()
    main(args.model, args.sample)
