"""実験で共通して使う処理（データ読み込み・CV分割・提出チェック・実験記録）。"""

import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "DataSet"
OUTPUT_DIR = ROOT / "outputs"
RECORD_FILE = ROOT / "records" / "experiments.md"

TARGET = "satisfaction"
N_SPLITS = 5
SEED = 42
N_TEST_ROWS = 299_844

CAT_COLS = ["gender", "customer_type", "type_of_travel", "class"]
RATING_COLS = [
    "inflight_wifi_service", "departure_arrival_time_convenient", "ease_of_online_booking",
    "gate_location", "food_and_drink", "online_boarding", "seat_comfort",
    "inflight_entertainment", "on_board_service", "leg_room_service",
    "baggage_handling", "checkin_service", "cleanliness",
]
NUM_COLS = ["age", "flight_distance", "departure_delay_in_minutes", "arrival_delay_in_minutes"]


def to_snake(name: str) -> str:
    """列名を snake_case にする（空白・`/`・`-` を含む列名を LightGBM などで扱えるように）。"""
    return re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_")


def load_data(nrows: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """train / test を読み込み、列名を整えてカテゴリ列を category 型にする。

    train の `satisfaction` は True=1 / False=0 の int に変換する。
    """
    train = pd.read_csv(DATA_DIR / "train.csv", nrows=nrows)
    test = pd.read_csv(DATA_DIR / "test.csv", nrows=nrows)
    train.columns = [to_snake(c) for c in train.columns]
    test.columns = [to_snake(c) for c in test.columns]
    train[TARGET] = train[TARGET].astype(int)

    for col in CAT_COLS:
        cats = sorted(set(train[col].dropna()) | set(test[col].dropna()))
        train[col] = pd.Categorical(train[col], categories=cats)
        test[col] = pd.Categorical(test[col], categories=cats)
    return train, test


def get_folds(y: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    """全実験で共通の StratifiedKFold 分割を返す。"""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    return list(skf.split(np.zeros(len(y)), y))


def check_submission(sub: pd.DataFrame, test: pd.DataFrame) -> None:
    """提出ファイルの形式を確認する。問題があれば AssertionError を出す。"""
    assert list(sub.columns) == ["id", TARGET], f"列名が不正: {list(sub.columns)}"
    assert len(sub) == N_TEST_ROWS, f"行数が不正: {len(sub)}"
    assert sub["id"].equals(test["id"].reset_index(drop=True)), "id が test と一致しない"
    assert sub[TARGET].notna().all(), "欠損あり"
    assert sub[TARGET].between(0, 1).all(), "0〜1 の範囲外の値あり"


def save_submission(exp_id: str, test: pd.DataFrame, pred: np.ndarray) -> Path:
    """形式チェックをしてから提出ファイルを書き出す。"""
    sub = pd.DataFrame({"id": test["id"].values, TARGET: pred})
    check_submission(sub, test)
    path = OUTPUT_DIR / exp_id / "submission.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    return path


def log_experiment(exp_id: str, model: str, settings: str, fold_scores: list[float], note: str = "") -> None:
    """records/experiments.md の表に1行追記する。LB は提出後に記入する。"""
    folds = " / ".join(f"{s:.5f}" for s in fold_scores)
    row = (
        f"| {exp_id} | {date.today():%Y-%m-%d} | {model} | {settings} | {folds} "
        f"| **{np.mean(fold_scores):.5f}** | {np.std(fold_scores):.5f} | - | {note} |\n"
    )
    with open(RECORD_FILE, "a", encoding="utf-8") as f:
        f.write(row)
