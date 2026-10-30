"""exp017: CatBoost + exp009 の特徴量 + 評価 × 文脈の交差 39 列を 10分割で学習（Issue #6 の 2本目の候補）。

交差は評価 13 列 × {Class, Type of Travel, Customer Type}（src/features.py の add_rating_context_cross）。
Discussion では CatBoost で +0.0003、路線・教師特徴量の上でも +0.00013 と報告されている（LightGBM では悪化）。
パラメータは exp011 / exp015 と同じ。比較対象は exp015（同じ設定で交差なし）。

使い方:
    python exp/exp017_cat_cross.py --sample   # 5万行で動作確認（記録・提出ファイルは作らない）
    python exp/exp017_cat_cross.py            # 全量で実行
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))
from exp011_cat_exp009feat import PARAMS
from src.common import OUTPUT_DIR, TARGET, get_folds, load_data, load_original, log_experiment
from src.features import add_rating_context_cross, build_exp009_features
from src.models import run_cat_cv, save_cv_result

EXP_ID = "exp017"
N_SPLITS_FINAL = 10
# カテゴリが 43 列になると、CatBoost が自動で作るカテゴリの組み合わせで極端に遅くなる（5万行でも 30 分以上）。
# 交差は自分で作って渡しているので、自動の組み合わせは止める
PARAMS = {**PARAMS, "max_ctr_complexity": 1}


def main(sample: bool) -> None:
    start = time.time()
    train, test = load_data(nrows=50_000 if sample else None)
    train, test, features, fold_fn = build_exp009_features(train, test, load_original())
    train, test = add_rating_context_cross(train), add_rating_context_cross(test)
    features = [c for c in train.columns if c not in ("id", TARGET)]
    print(f"特徴量 {len(features)} 列 + fold 内 TE 1 列")

    # 1 fold 約 22 分・全体で約 4 時間かかり、途中でプロセスが止まることがあるので fold ごとに保存して続きから再開できるようにする
    ckpt_dir = None if sample else OUTPUT_DIR / EXP_ID / "folds"
    res = run_cat_cv(train, test, features, PARAMS, fold_fn=fold_fn, n_splits=N_SPLITS_FINAL, checkpoint_dir=ckpt_dir)
    y = train[TARGET].values
    scores5 = [roc_auc_score(y[va], res.oof[va]) for _, va in get_folds(train[TARGET])]
    print(f"{EXP_ID}: 10分割の OOF を 5分割の区切りで評価  CV={np.mean(scores5):.5f}  {[round(s, 5) for s in scores5]}")
    print(f"所要時間 {(time.time() - start) / 60:.1f} 分")
    if sample:
        return

    save_cv_result(EXP_ID, test, res)
    log_experiment(
        EXP_ID, "CatBoost",
        f"exp015 + 評価 × {{Class, Type of Travel, Customer Type}} の交差 39 列（カテゴリ）。計 {len(features) + 1} 列 "
        f"/ 10分割 / パラメータは exp011 と同じ / best_iter 平均 {int(np.mean(res.best_iters))} "
        f"/ 10分割の fold 平均 {np.mean(res.fold_scores):.5f}",
        scores5, note="Issue #6。Fold 列は 10分割の OOF を 5分割の区切りで評価した値（exp015 と比較用）",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true")
    main(parser.parse_args().sample)
