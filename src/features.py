"""特徴量の作成。"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.common import RATING_COLS, SEED, TARGET

ROUTE_KEY = "flight_distance"  # 約 3,500 種類の値が繰り返し出てくるので「路線 ID」として扱う


CROSS_CONTEXT_COLS = ["class", "type_of_travel", "customer_type"]


def add_rating_context_cross(df: pd.DataFrame) -> pd.DataFrame:
    """評価 13 列 × {Class, Type of Travel, Customer Type} の交差 39 列を category 型で追加する（CatBoost 向け）。

    列名は `<評価>_x_<文脈>`、値は「評価値|カテゴリ」。Discussion で CatBoost にだけ効くと報告されている（LightGBM では悪化）。
    """
    crosses = {
        f"{r}_x_{c}": (df[r].astype(str) + "|" + df[c].astype(str)).astype("category")
        for r in RATING_COLS for c in CROSS_CONTEXT_COLS
    }
    return df.assign(**crosses)


def add_route_count(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """同じ Flight Distance の行数（train + test で数える。ラベル不使用）を `fd_count` として追加する。"""
    counts = pd.concat([train[ROUTE_KEY], test[ROUTE_KEY]]).value_counts()
    return (
        train.assign(fd_count=train[ROUTE_KEY].map(counts).astype(float)),
        test.assign(fd_count=test[ROUTE_KEY].map(counts).astype(float)),
    )


def add_route_profile(
    train: pd.DataFrame, test: pd.DataFrame, residual: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """路線（Flight Distance）ごとの他の列の平均を `fd_mean_<col>` として追加する（train + test で計算。ラベル不使用）。

    residual=True のときは、その行の値と路線平均との差 `fd_resid_<col>` も追加する（評価 13 列と年齢）。
    """
    both = pd.concat([train, test], ignore_index=True)
    src = both[RATING_COLS + ["age", "departure_delay_in_minutes"]].astype(float)
    src["is_business_class"] = (both["class"] == "Business").astype(float)
    src["is_eco_class"] = (both["class"] == "Eco").astype(float)
    src["is_business_travel"] = (both["type_of_travel"] == "Business travel").astype(float)
    src["is_loyal"] = (both["customer_type"] == "Loyal Customer").astype(float)
    src["is_female"] = (both["gender"] == "Female").astype(float)
    means = src.groupby(both[ROUTE_KEY]).mean().add_prefix("fd_mean_")

    out = []
    for df in (train, test):
        prof = means.reindex(df[ROUTE_KEY]).set_index(df.index)
        df = pd.concat([df, prof], axis=1)
        if residual:
            for col in RATING_COLS + ["age"]:
                df[f"fd_resid_{col}"] = df[col] - df[f"fd_mean_{col}"]
        out.append(df)
    return out[0], out[1]


def _smoothed_mean(key: pd.Series, y: pd.Series, prior: float, m: float) -> pd.Series:
    stats = y.groupby(key.values, observed=True).agg(["sum", "count"])
    return (stats["sum"] + m * prior) / (stats["count"] + m)


def make_fold_target_encoder(cols: list[str], m: float = 10.0, n_inner: int = 5):
    """fold 内ターゲットエンコーディングを作る関数（models の fold_fn に渡す）を返す。

    - 検証側・test 側：その fold の学習側全体での平滑化した満足率
    - 学習側：学習側をさらに n_inner 分割し、自分を含まない部分の満足率（自分のラベルが入らないようにする）
    平滑化は (合計 + m × 全体平均) / (件数 + m)。列名は `<col>_te`。
    """
    def fold_fn(X_tr, y_tr, X_va, X_te):
        X_tr, X_va, X_te = X_tr.copy(), X_va.copy(), X_te.copy()
        prior = float(y_tr.mean())
        inner = KFold(n_splits=n_inner, shuffle=True, random_state=SEED)
        for col in cols:
            name = f"{col}_te"
            full = _smoothed_mean(X_tr[col], y_tr, prior, m)
            X_va[name] = X_va[col].map(full).fillna(prior).astype(float).values
            X_te[name] = X_te[col].map(full).fillna(prior).astype(float).values
            te_tr = np.empty(len(X_tr))
            for fit_idx, enc_idx in inner.split(X_tr):
                part = _smoothed_mean(X_tr[col].iloc[fit_idx], y_tr.iloc[fit_idx], prior, m)
                te_tr[enc_idx] = X_tr[col].iloc[enc_idx].map(part).fillna(prior).astype(float).values
            X_tr[name] = te_tr
        return X_tr, X_va, X_te

    return fold_fn


def build_exp009_features(train: pd.DataFrame, test: pd.DataFrame, orig: pd.DataFrame):
    """exp009 の特徴量（2026-10-26 時点の最良）を作る。モデルの種類を変えて同じ特徴量を使うときに呼ぶ。

    元データの教師モデル + 元データ TE + 路線の件数・プロファイル。fold 内 TE は返り値の fold_fn で作る。
    戻り値：(train, test, features, fold_fn)
    """
    base = [c for c in train.columns if c not in ("id", TARGET)]
    train, test = add_original_teacher(train, test, orig, base)
    train, test = add_original_te(train, orig, base), add_original_te(test, orig, base)
    train, test = add_route_count(train, test)
    train, test = add_route_profile(train, test)
    features = [c for c in train.columns if c not in ("id", TARGET)]
    return train, test, features, make_fold_target_encoder([ROUTE_KEY])


def add_original_teacher(
    train: pd.DataFrame, test: pd.DataFrame, orig: pd.DataFrame, cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """元データだけで学習した LightGBM（教師モデル）の予測 logit を `orig_teacher_logit` として追加する。

    元データはコンペのデータと別なので、コンペの train 全体に教師の予測を付けてもリークは起きない。
    学習回数は元データ上の 5分割 CV（early stopping）で決め、元データ全体で学習し直す。
    """
    import lightgbm as lgb

    from src.common import CAT_COLS

    orig = orig.copy()
    for col in CAT_COLS:  # カテゴリの並びをコンペのデータに揃える
        orig[col] = orig[col].cat.set_categories(train[col].cat.categories)
    params = {
        "objective": "binary", "metric": "auc", "learning_rate": 0.05, "num_leaves": 63,
        "min_child_samples": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
        "seed": SEED, "verbosity": -1, "n_jobs": -1,
    }
    dtrain = lgb.Dataset(orig[cols], orig[TARGET])
    cv = lgb.cv(params, dtrain, num_boost_round=5_000, nfold=5, stratified=True, seed=SEED,
                callbacks=[lgb.early_stopping(100, verbose=False)])
    n_rounds = len(cv["valid auc-mean"])
    print(f"教師モデル：元データの CV AUC={cv['valid auc-mean'][-1]:.5f}  rounds={n_rounds}")
    model = lgb.train(params, dtrain, num_boost_round=n_rounds)
    return (
        train.assign(orig_teacher_logit=model.predict(train[cols], raw_score=True)),
        test.assign(orig_teacher_logit=model.predict(test[cols], raw_score=True)),
    )


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
