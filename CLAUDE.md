# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Python tool for calculating multi-label match rates using Jaccard similarity. It processes tab-separated CSV files where labels are marked with "o" and computes pairwise similarity metrics.

## Commands

### Running the Tool
```bash
python multi_label_calc.py <csv_file>
```

Example:
```bash
python multi_label_calc.py sample_data.csv
```

### Testing
No automated test suite exists. Manual testing can be done with:
```bash
python multi_label_calc.py sample_data.csv
```

## Architecture

### Single-Module Design
The entire functionality is contained in `multi_label_calc.py` (193 lines). The code follows a functional design with clear separation:

1. **Data Reading** (`read_csv_file`): Parses tab-delimited CSV files
2. **Column Filtering** (`get_label_columns`): Separates metadata columns from label columns
3. **Label Extraction** (`get_labels_for_row`): Identifies which labels are marked "o" for each row
4. **Match Rate Calculation** (`calculate_match_rate`): Computes pairwise Jaccard similarity across all rows
5. **Output Formatting** (`print_results`): Displays statistics and label frequencies

### Data Format Expectations

**Metadata columns** (excluded from analysis):
- `message`, `pr_id`, `repo_url`, `commit_url`, `Memo`

**Label columns**: Any column not in the metadata set

**Label marking**: Cells containing "o" (case-insensitive, whitespace-trimmed)

### Jaccard Similarity Implementation

The match rate calculation (lines 84-101) performs all-pairs comparison:
- For each pair of rows, computes: `|intersection| / |union|`
- Special case: Two empty label sets = 1.0 (perfect match)
- Averages across all pairs to produce final match rate

### No External Dependencies
Uses only Python standard library (csv, sys, pathlib, typing). The requirements.txt exists but notes no external dependencies are needed.
