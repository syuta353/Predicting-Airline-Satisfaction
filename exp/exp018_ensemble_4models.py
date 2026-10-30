"""exp018: exp016 の 3 モデルに、交差特徴量の CatBoost（exp017）を足した 4 モデルの順位平均（Issue #6）。

exp017 は単体では exp015 とほぼ同じ CV だが、exp015 との OOF 相関が 0.9886 と低めで、混ぜると伸びる。
方法は exp016 と同じ。4 モデルは重みの組み合わせが多く、ネスト CV まで含めて 30 分ほどかかる。

使い方:
    python exp/exp018_ensemble_4models.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.ensemble import run_rank_ensemble

if __name__ == "__main__":
    run_rank_ensemble(
        "exp018", ["exp013", "exp014", "exp015", "exp017"],
        model_label="10分割の LGB(exp013) / XGB(exp014) / Cat(exp015) / Cat + 交差(exp017)", note="Issue #6",
    )
