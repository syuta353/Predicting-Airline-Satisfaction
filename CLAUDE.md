# CLAUDE.md

Kaggle Playground Series S6E10「Predicting Airline Satisfaction」の作業リポジトリ。

## 最初に読むもの

- `Description/description.md` の「分析用サマリー」：タスク、データの列構成、提出形式、ルールがまとまっている
- 原文（Overview / Data / Rules）は同じファイルの後半にある。ルールの細部を確認するときだけ読めばよい

## コンペの要点

- 二値分類。`satisfaction` の確率（0〜1）を予測する
- 評価指標は **ROC AUC**。確率の大小の順序だけが効くので、確率の較正は不要
- `satisfaction` は bool 型。**`True` を陽性クラス（1）** とする
- 提出は 1日10回まで。最終提出は 2つまで選択
- 締切：**2026年10月31日 23:59 UTC**（日本時間 11月1日 8:59）

## フォルダ構成

| パス | 内容 | Git |
|---|---|---|
| `DataSet/` | `train.csv` / `test.csv` / `sample_submission.csv` | 管理しない（`.gitignore` 済み）|
| `Description/` | コンペ説明 | 管理する |

- 作業フォルダはローカル（`C:\Users\PC\OneDrive\job\kaggle_compete\2026\Predicting Airline Satisfaction`）。以前の Google Drive 上のフォルダは参照しない
- データファイルは Git にコミットしない。ルール上、コンペデータの再配布は禁止されている

## 分析の方針

- **手元の交差検証（CV）スコアを主な指標にする**。Public LB は test の一部でしか評価されないため、LB に合わせて調整しすぎない
- 交差検証は `StratifiedKFold`（5分割、`shuffle=True`、`random_state=42`）を基本とし、実験間で同じ分割を使う
- 実験ごとに CV スコア（各 fold と平均）・主な設定・LB スコアを記録し、最終提出 2つを選ぶ判断材料にする
- 提出前に、行数が 299,844、`id` が test と一致、値が 0〜1、欠損なし、列名が `id,satisfaction` であることを確認する

## 環境

- Windows 11。Python は Anaconda（Python 3.11）
- Kaggle API の認証ファイル（`~/.kaggle/kaggle.json`）は設定済み
- データが約70万行あるため、重い処理は先にサンプルで動作確認してから全量で回す
