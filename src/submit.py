"""outputs/<ID>/submission.csv を形式チェックしてから Kaggle に提出し、Public LB の結果を待って表示する。

使い方:
    python src/submit.py exp001 "メッセージ"
"""

import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.common import DATA_DIR, OUTPUT_DIR, check_submission

COMPETITION = "playground-series-s6e10"


def main(exp_id: str, message: str) -> None:
    path = OUTPUT_DIR / exp_id / "submission.csv"
    test = pd.read_csv(DATA_DIR / "test.csv", usecols=["id"])
    check_submission(pd.read_csv(path), test)
    print(f"形式チェック OK: {path}")

    subprocess.run(
        ["kaggle", "competitions", "submit", "-c", COMPETITION, "-f", str(path), "-m", f"{exp_id}: {message}"],
        check=True,
    )
    # スコアが出るまで待つ（最大 5 分）
    for _ in range(30):
        time.sleep(10)
        out = subprocess.run(
            ["kaggle", "competitions", "submissions", "-c", COMPETITION, "--csv"],
            capture_output=True, text=True, check=True,
        ).stdout
        # 今回の提出行（メッセージが `<ID>:` で始まる最新の行）を探す
        latest = next((line for line in out.splitlines() if f",{exp_id}:" in line), "")
        status = latest.upper()
        if "COMPLETE" in status or "ERROR" in status:
            print("ref,fileName,date,description,status,publicScore,privateScore")
            print(latest)
            return
    print("5 分以内にスコアが出なかった。後で `kaggle competitions submissions -c playground-series-s6e10` で確認する")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
