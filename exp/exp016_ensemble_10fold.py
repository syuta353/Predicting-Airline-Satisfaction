"""exp016: 10分割で学習した LightGBM / XGBoost / CatBoost（exp013〜015）の順位平均アンサンブル（Issue #11）。

方法は exp012 と同じ（重みを 0.05 刻みで総当たりし、ネスト CV で過学習を確認）。
outputs/ の OOF を使うので、exp013・exp014・exp015 を先に実行しておく。

使い方:
    python exp/exp016_ensemble_10fold.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.ensemble import run_rank_ensemble

if __name__ == "__main__":
    run_rank_ensemble(
        "exp016", ["exp013", "exp014", "exp015"],
        model_label="10分割の LGB(exp013) / XGB(exp014) / Cat(exp015)", note="Issue #11",
    )
