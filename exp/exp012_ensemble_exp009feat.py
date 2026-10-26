"""exp012: exp009 の特徴量で学習した LightGBM / XGBoost / CatBoost の順位平均アンサンブル（Issue #8）。

方法は exp005 と同じ（重みを 0.05 刻みで総当たりし、ネスト CV で過学習を確認）。
outputs/ の OOF を使うので、exp009・exp010・exp011 を先に実行しておく。

使い方:
    python exp/exp012_ensemble_exp009feat.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.ensemble import run_rank_ensemble

if __name__ == "__main__":
    run_rank_ensemble(
        "exp012", ["exp009", "exp010", "exp011"],
        model_label="LGB(exp009) / XGB(exp010) / Cat(exp011)", note="Issue #8",
    )
