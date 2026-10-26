"""特徴量の作成。"""

import numpy as np
import pandas as pd

from src.common import RATING_COLS, TARGET


def add_original_te(df: pd.DataFrame, orig: pd.DataFrame, cols: list[str], min_count: int = 20) -> pd.DataFrame:
    """各列の値ごとに、元データでの満足率と件数を特徴量として追加する（`<col>_orig_te` / `<col>_orig_cnt`）。

    元データはコンペのデータと別なので、CV の fold をまたいだリークは起きない。
    件数が min_count 未満の値と、元データにない値は NaN にする。
    """
    df = df.copy()
    for col in cols:
        stats = orig.groupby(orig[col].astype(str), observed=True)[TARGET].agg(["mean", "size"])
        key = df[col].astype(str)
        df[f"{col}_orig_te"] = key.map(stats["mean"].where(stats["size"] >= min_count)).astype(float)
        df[f"{col}_orig_cnt"] = key.map(stats["size"]).fillna(0).astype(float)
    return df


def add_rating_zero_features(df: pd.DataFrame) -> pd.DataFrame:
    """評価項目の 0（該当なし）を扱う特徴量を追加する。

    EDA で、0 は 1〜5 と同じ順序の値として扱えないことがわかったため、
    0 のフラグと、0 を欠損とみなした評価の集計値を作る。
    """
    df = df.copy()
    ratings = df[RATING_COLS]
    is_zero = ratings.eq(0)
    nz = ratings.where(~is_zero)  # 0 を NaN にした評価

    for col in RATING_COLS:
        if is_zero[col].any():
            df[f"{col}_is0"] = is_zero[col].astype(np.int8)
    df["rating_n_zero"] = is_zero.sum(axis=1)
    df["rating_mean_nz"] = nz.mean(axis=1)
    df["rating_min_nz"] = nz.min(axis=1)
    df["rating_max_nz"] = nz.max(axis=1)
    df["rating_std_nz"] = nz.std(axis=1)
    df["rating_n_low"] = nz.le(2).sum(axis=1)   # 1〜2 の個数
    df["rating_n_high"] = nz.ge(4).sum(axis=1)  # 4〜5 の個数
    return df
