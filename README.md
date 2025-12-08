# multi-label-calc

マルチラベルの一致率を計算するツールです。

## 概要

与えられたCSVファイルに含まれるマルチラベルデータの一致率（Jaccard類似度）を計算します。
CSVファイルでは、各カテゴリに該当する場合は "o" でマークされています。

## 使い方

```bash
python multi_label_calc.py <csv_file>
```

### 例

```bash
python multi_label_calc.py sample_data.csv
```

## CSVフォーマット

CSVファイルはタブ区切り（TSV）形式で、以下のような構造である必要があります：

- メタデータ列: `message`, `pr_id`, `repo_url`, `commit_url`, `Memo`
- ラベル列: 分類カテゴリ（該当する場合は "o" とマーク）

### サンプルCSV

```
message	pr_id	repo_url	commit_url	Category1	Category2	Category3	Memo
Example message	123	https://...	https://...	o		o	
Another message	124	https://...	https://...		o	o	
```

## 出力

スクリプトは以下の情報を出力します：

- 分析された行数
- ラベルカテゴリの総数
- ペアワイズ比較の総数
- 平均一致率（Jaccard類似度）
- 平均一致率（パーセンテージ）
- 各ラベルの出現頻度

## 一致率の計算方法

このツールは、すべての行のペアワイズ比較を行い、Jaccard類似度を使用して一致率を計算します：

```
Jaccard類似度 = |共通ラベル| / |全ラベルの和集合|
```

## 要件

- Python 3.6以上
- 標準ライブラリのみ使用（外部依存なし）

## ライセンス

MIT License