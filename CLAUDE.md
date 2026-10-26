# CLAUDE.md

Kaggle Playground Series S6E10「Predicting Airline Satisfaction」の作業リポジトリ。

## 最初に読むもの

- `Description/description.md` の「分析用サマリー」：タスク、データの列構成、提出形式、ルールがまとまっている
- 原文（Overview / Data / Rules）は同じファイルの後半にある。ルールの細部を確認するときだけ読めばよい
- `records/experiments.md`：これまでの実験の CV スコア（各 fold・平均）と Public LB。次の実験の比較や最終提出の選択で、必要に応じて確認する
- `knowledge/discussion_summary.md`：Kaggle Discussion 全件の要約。効いた施策・効かなかった施策とその効果量、CV / LB の関係、最終提出の選び方。新しい施策を考えるときに確認する
- **GitHub の Issues**（`gh issue list`）：次のアクションなどのタスクを管理している。作業を始める前に確認する

## コンペの要点

- 二値分類。`satisfaction` の確率（0〜1）を予測する
- 評価指標は **ROC AUC**。確率の大小の順序だけが効くので、確率の較正は不要
- `satisfaction` は bool 型。**`True` を陽性クラス（1）** とする
- 提出は 1日10回まで。最終提出は 2つまで選択
- 締切：**2026年10月31日 23:59 UTC**（日本時間 11月1日 8:59）

## フォルダ構成

```
Predicting Airline Satisfaction/
├── CLAUDE.md
├── .gitignore
├── DataSet/                     # コンペデータ（Git 管理外）
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   └── original/                # 元データ（Kaggle Dataset teejmahal20/airline-passenger-satisfaction）
│       ├── train.csv            # train + test を合わせて 129,880 行。load_original() で読み込む
│       └── test.csv
├── Description/
│   └── description.md           # コンペ説明（分析用サマリー + 原文）
├── src/                         # 共通処理
│   ├── __init__.py
│   ├── common.py                # データ読み込み（load_data / load_original）・CV 分割・提出チェック・実験記録への追記
│   ├── ensemble.py              # アンサンブル（OOF の順位化・重みの総当たり・ネスト CV）
│   ├── features.py              # 特徴量の作成（評価0の特徴量、元データでのターゲットエンコーディング）
│   ├── models.py                # モデルの CV 学習（run_lgbm_cv / run_xgb_cv / run_cat_cv）と結果の保存（save_cv_result）
│   │                            # run_lgbm_cv の extra_train に渡したデータは各 fold の学習側にだけ足す
│   └── submit.py                # 提出ファイルをチェックして Kaggle に提出し、LB を表示
├── exp/                         # 実験スクリプト（1実験 1ファイル、expXXX_<内容>.py）
│   ├── exp001_lgbm_baseline.py  # 学習処理を内部に持つ（src/models.py 作成前のため）
│   ├── exp002_lgbm_rating_zero.py
│   ├── exp003_xgb_baseline.py
│   ├── exp004_cat_baseline.py   # CatBoost は遅い（サンプル5万行で約3分、全量で約47分）
│   ├── exp005_ensemble_rank.py  # exp001/003/004 の順位平均。outputs/ の OOF を使うので、元の実験を先に実行しておく
│   ├── exp006_lgbm_orig_concat.py  # 元データを学習データに追加
│   └── exp007_lgbm_orig_te.py   # 元データでのターゲットエンコーディング
├── knowledge/
│   └── discussion_summary.md    # Kaggle Discussion の要約（2026-10-26 時点、全 42 Topic）
├── notebooks/
│   └── eda.ipynb                # EDA（先頭にまとめあり）
├── records/
│   └── experiments.md           # 実験記録（CV / LB）
└── outputs/                     # OOF・test 予測・提出ファイル（Git 管理外）
    └── expXXX/
        ├── oof.npy
        ├── test_pred.npy
        ├── importance.csv       # 特徴量重要度（exp002 以降。指標はモデルごとに異なる）
        └── submission.csv
```

- 作業フォルダはローカル（`C:\Users\PC\OneDrive\job\kaggle_compete\2026\Predicting Airline Satisfaction`）。以前の Google Drive 上のフォルダは参照しない
- データファイルは Git にコミットしない。ルール上、コンペデータの再配布は禁止されている
- `load_data()` は列名を snake_case に変換する（例：`Departure/Arrival time convenient` → `departure_arrival_time_convenient`）

## 作業の進め方

- 実験：`python exp/expXXX_*.py --sample` で動作確認してから全量で実行する。全量で実行すると `records/experiments.md` に1行追記される
- 提出：`python src/submit.py expXXX "メッセージ"`。LB が出たら `records/experiments.md` の Public LB 列に記入する
- タスク管理：次のアクションや検討事項は GitHub の Issues で管理し、必要に応じて追加・更新・削除する
  - 追加：新しいアクションや検討事項が出てきたとき（実験結果や EDA から得た次の打ち手など）
  - 更新：進捗に合わせてチェックリストを更新し、結果（CV / LB、わかったこと）をコメントで残す。方針が変わったら本文を直す
  - クローズ：完了したとき。不要になったときは、理由をコメントしてから `not planned` でクローズする（履歴を残すため、削除はしない）
- **CLAUDE.md の更新**：次のことが起きたら、このファイルに反映する
  - EDA で新しくわかったことがある → 「EDA でわかったこと」に追記する
  - ファイルやフォルダを追加・削除した → 「フォルダ構成」のツリーを更新する

## 分析の方針

- **手元の交差検証（CV）スコアを主な指標にする**。Public LB は test の一部でしか評価されないため、LB に合わせて調整しすぎない
- 交差検証は `StratifiedKFold`（5分割、`shuffle=True`、`random_state=42`）を基本とし、実験間で同じ分割を使う
- 実験ごとに CV スコア（各 fold と平均）・主な設定・LB スコアを記録し、最終提出 2つを選ぶ判断材料にする
- 提出前に、行数が 299,844、`id` が test と一致、値が 0〜1、欠損なし、列名が `id,satisfaction` であることを確認する（`src/submit.py` で自動チェック）

## EDA でわかったこと

詳細は `notebooks/eda.ipynb`。

- **train / test の分布はほぼ同じ**（カテゴリ比率・数値の平均とも差は小さい）。exp001 でも CV 0.95888 / LB 0.95849 と近い
- **評価項目（0〜5）の 0 は順序値として扱えない**：`inflight_wifi_service` は 0 の満足率 0.887（1〜3 は 0.27〜0.39）、`online_boarding` は 0 で 0.620（1〜3 は 0.09〜0.13）。0 の割合は多くて 3.7%（`departure_arrival_time_convenient`）。`baggage_handling` には 0 がない
  - ただし木モデルでは元の列だけで 0 を分岐できるため、0 フラグや 0 を除いた集計特徴量を足しても CV は改善しなかった（exp002：0.95888 → 0.95871）
- **満足率の差が大きい列**：`class`（Business 0.73 / Eco 0.17 / Eco Plus 0.24）、`type_of_travel`（Business 0.58 / Personal 0.10）、`customer_type`（Loyal 0.50 / disloyal 0.20）。評価項目では `online_boarding`、`inflight_wifi_service`、`inflight_entertainment` など
- **遅延**：出発・到着遅延の相関 0.84、両方 0 が 87%。平均は約 1 分で、元データ（約 15 分）より大幅に小さい
- **元データとの比較**（`DataSet/original/`、129,880 行）：
  - 列は同じ。ただし元データには `Inflight service` が1列多い（コンペにはない）
  - **遅延の分布が大きく違う**：出発遅延が 0 の割合はコンペ 91% / 元データ 56%、平均は約 1 分 / 約 15 分、90%点は 0 分 / 44 分
  - それ以外（カテゴリ比率、評価の平均、満足率 44.4% / 43.5%）は近い。`flight_distance` はコンペのほうがやや長い（平均 1,354 / 1,190）
  - 評価の 0 の傾向も同じ（元データでは `inflight_wifi_service` が 0 の満足率 0.997）
  - 元データと完全一致する行は test 4行、train 17行のみ
  - **元データは「特徴量」として使うと効く**：各列の値ごとの元データでの満足率（ターゲットエンコーディング）を足すと CV +0.00072 / LB +0.00069（exp007）。学習データに行を足すだけだと逆に下がった（exp006：-0.00020）
- `arrival_delay_in_minutes` の欠損はごく少ない（train 292 / test 130）
- train 内の重複行、train と完全一致する test 行はない

## 環境

- Windows 11。Python は Anaconda（Python 3.11）
- 主なライブラリ：lightgbm **4.6.0**（4.7.0 はこの環境で学習時にクラッシュするため上げない）、xgboost、catboost、optuna、kaggle CLI
- Kaggle API の認証ファイル（`~/.kaggle/kaggle.json`）は設定済み
- データが約70万行あるため、重い処理は先にサンプルで動作確認してから全量で回す
