# multi-label-calc

A tool to calculate agreement rates for multi-label data.

## Overview

This tool calculates the agreement rate (Jaccard similarity) for multi-label data contained in a provided CSV file. In the CSV file, applicable categories are marked with "o".

## Requirements

- Python 3.8 or higher

## Usage
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
python inter_rater_agreement.py <csv_file>
```

### Example
```bash
python inter_rater_agreement.py sample_data.csv sample_data2.csv
```

## CSV Format

The CSV file must be in tab-separated values (TSV) format and adhere to the following structure:

- **Metadata columns**: message, pr_id, repo_url, commit_url, Memo
- **Label columns**: Classification categories (marked with "o" if applicable)
```
message	pr_id	repo_url	commit_url	Category1	Category2	Category3	Memo
Example message	123	https://...	https://...	o		o	
Another message	124	https://...	https://...		o	o	
```

## Output

The script generates a comprehensive report on inter-rater reliability, categorized into the following sections:

### Summary Statistics

- Number of common items analyzed (matched by SHA)
- Number of label categories

### Label-wise Agreement Metrics

- **Macro-Averaged Cohen's Kappa**: Average of Kappa scores across all labels
- **Micro-Averaged Cohen's Kappa**: Kappa calculated on pooled data
- **Krippendorff's Alpha**: Reliability coefficient suitable for nominal data

### Set-based Metrics (Label Combination Agreement)

- **Mean Jaccard Coefficient**: Average intersection-over-union of assigned labels
- **Exact Match Ratio**: Percentage of items where raters agreed on all labels perfectly
- **Hamming Score**: Accuracy of label assignment per category

### Detailed Per-Label Analysis

- **Per-Label Kappa**: Cohen's Kappa score for each specific category
- **Positive/Negative Agreement**: Probability of agreement given that at least one rater marked (or didn't mark) the label
- **Confusion Matrix**: Raw counts for:
  - **Both+**: Both raters marked the label
  - **Both-**: Neither rater marked the label
  - **Only1/Only2**: Disagreements where only one rater marked the label

### Annotator Bias

Comparison of label frequencies between the two raters to identify if one rater is systematically more "generous" or "strict" than the other.

### Disagreement Export

If the `--export-disagreements` flag is used, the script creates a CSV file (e.g., `disagreements_rater1_vs_rater2.csv`) containing items with a Jaccard score below 0.5.

## Calculation Methods

- **Cohen's Kappa**: Calculated using `sklearn.metrics.cohen_kappa_score`
- **Krippendorff's Alpha**: Calculated using the `krippendorff` library (nominal metric)
- **Jaccard Similarity**: |Intersection| / |Union| of label sets per item
- **Hamming Score**: Correct Labels / Total Labels averaged across items

## License

MIT License
