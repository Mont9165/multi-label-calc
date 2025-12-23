#!/usr/bin/env python3
"""
Inter-Rater Agreement Calculator

This script calculates inter-rater agreement metrics for multi-label classifications:
- Macro-averaged Cohen's Kappa
- Krippendorff's Alpha
- Set-based Metrics (Jaccard, Exact Match Ratio)
- Confusion Matrix Analysis (Positive/Negative Agreement)
- Annotator Bias Analysis
- Disagreement Analysis with CSV export

The script expects two CSV files from different raters annotating the same items.
Uses scikit-learn for Cohen's Kappa and krippendorff library for Krippendorff's Alpha.
"""

import csv
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Set
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
import krippendorff


def read_csv_file(file_path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Read CSV file and return headers and data rows.

    Args:
        file_path: Path to the CSV file

    Returns:
        Tuple of (headers, data_rows)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=',')
        headers = list(reader.fieldnames)
        data = list(reader)
    return headers, data


def get_label_columns(headers: List[str]) -> List[str]:
    """
    Get the label columns from headers.
    Excludes metadata columns.

    Args:
        headers: List of column headers

    Returns:
        List of label column names
    """
    metadata_columns = {'sha', 'message', 'pr_id', 'repo_url', 'commit_url', 'Memo'}
    return [col for col in headers if col not in metadata_columns]


def align_data_by_sha(data1: List[Dict[str, str]], data2: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Align two datasets by SHA to ensure we're comparing the same items.

    Args:
        data1: First rater's data
        data2: Second rater's data

    Returns:
        Tuple of aligned datasets
    """
    # Create dictionaries keyed by SHA
    dict1 = {row['sha']: row for row in data1 if 'sha' in row and row['sha']}
    dict2 = {row['sha']: row for row in data2 if 'sha' in row and row['sha']}

    # Find common SHAs
    common_shas = sorted(set(dict1.keys()) & set(dict2.keys()))

    # Return aligned data
    aligned1 = [dict1[sha] for sha in common_shas]
    aligned2 = [dict2[sha] for sha in common_shas]

    return aligned1, aligned2


def get_binary_label(row: Dict[str, str], label: str) -> int:
    """
    Get binary label (1 if marked with 'o', 0 otherwise).

    Args:
        row: Dictionary representing a row from the CSV
        label: Label column name

    Returns:
        1 if label is marked, 0 otherwise
    """
    return 1 if row.get(label, '').strip().lower() == 'o' else 0


def get_label_set(row: Dict[str, str], label_columns: List[str]) -> Set[str]:
    """
    Get the set of labels marked with 'o' for a given row.

    Args:
        row: Dictionary representing a row from the CSV
        label_columns: List of label column names

    Returns:
        Set of label names that are marked with 'o'
    """
    return {col for col in label_columns if get_binary_label(row, col) == 1}


def calculate_macro_averaged_kappa(data1: List[Dict[str, str]],
                                   data2: List[Dict[str, str]],
                                   label_columns: List[str]) -> Dict:
    """
    Calculate macro-averaged Cohen's Kappa across all labels using scikit-learn.

    Args:
        data1: First rater's data
        data2: Second rater's data
        label_columns: List of label column names

    Returns:
        Dictionary containing Kappa statistics
    """
    kappa_scores = {}

    for label in label_columns:
        rater1 = np.array([get_binary_label(row, label) for row in data1])
        rater2 = np.array([get_binary_label(row, label) for row in data2])

        # Use scikit-learn's cohen_kappa_score
        kappa = cohen_kappa_score(rater1, rater2)
        kappa_scores[label] = kappa

    # Calculate macro average
    macro_kappa = np.mean(list(kappa_scores.values())) if kappa_scores else 0.0

    return {
        'macro_averaged_kappa': macro_kappa,
        'per_label_kappa': kappa_scores
    }


def calculate_micro_averaged_kappa(data1: List[Dict[str, str]],
                                   data2: List[Dict[str, str]],
                                   label_columns: List[str]) -> float:
    """
    Calculate micro-averaged Cohen's Kappa by pooling all label decisions.

    Args:
        data1: First rater's data
        data2: Second rater's data
        label_columns: List of label column names

    Returns:
        Micro-averaged Kappa coefficient
    """
    all_rater1 = []
    all_rater2 = []

    for label in label_columns:
        rater1 = [get_binary_label(row, label) for row in data1]
        rater2 = [get_binary_label(row, label) for row in data2]
        all_rater1.extend(rater1)
        all_rater2.extend(rater2)

    # Calculate Kappa on pooled data
    micro_kappa = cohen_kappa_score(all_rater1, all_rater2)

    return micro_kappa


def calculate_krippendorff_alpha(data1: List[Dict[str, str]],
                                 data2: List[Dict[str, str]],
                                 label_columns: List[str]) -> float:
    """
    Calculate Krippendorff's Alpha for nominal data using krippendorff library.

    Args:
        data1: First rater's data
        data2: Second rater's data
        label_columns: List of label column names

    Returns:
        Krippendorff's Alpha coefficient
    """
    # Build reliability data matrix
    # Shape: (n_raters, n_items * n_labels)
    # Each label for each item is treated as a separate judgment

    rater1_values = []
    rater2_values = []

    for label in label_columns:
        for row1, row2 in zip(data1, data2):
            val1 = get_binary_label(row1, label)
            val2 = get_binary_label(row2, label)
            rater1_values.append(val1)
            rater2_values.append(val2)

    # Create reliability data matrix (2 raters x n_judgments)
    reliability_data = np.array([
        rater1_values,
        rater2_values
    ])

    # Calculate Krippendorff's Alpha using the library
    # level_of_measurement='nominal' for binary/categorical data
    alpha = krippendorff.alpha(reliability_data, level_of_measurement='nominal')

    return alpha


def calculate_set_based_metrics(data1: List[Dict[str, str]],
                                 data2: List[Dict[str, str]],
                                 label_columns: List[str]) -> Dict:
    """
    Calculate set-based metrics (Jaccard, Exact Match Ratio).

    Args:
        data1: First rater's data
        data2: Second rater's data
        label_columns: List of label column names

    Returns:
        Dictionary containing set-based metrics
    """
    jaccard_scores = []
    exact_matches = 0
    n_items = len(data1)

    for row1, row2 in zip(data1, data2):
        set1 = get_label_set(row1, label_columns)
        set2 = get_label_set(row2, label_columns)

        # Calculate Jaccard coefficient for this item
        if len(set1) == 0 and len(set2) == 0:
            jaccard = 1.0  # Both empty = perfect match
        elif len(set1.union(set2)) == 0:
            jaccard = 0.0
        else:
            jaccard = len(set1.intersection(set2)) / len(set1.union(set2))

        jaccard_scores.append(jaccard)

        # Check for exact match
        if set1 == set2:
            exact_matches += 1

    mean_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0.0
    exact_match_ratio = exact_matches / n_items if n_items > 0 else 0.0

    return {
        'mean_jaccard': mean_jaccard,
        'exact_match_ratio': exact_match_ratio,
        'exact_matches': exact_matches,
        'total_items': n_items,
        'jaccard_scores': jaccard_scores  # Keep for disagreement analysis
    }


def calculate_hamming_score(data1: List[Dict[str, str]],
                            data2: List[Dict[str, str]],
                            label_columns: List[str]) -> Dict:
    """
    Calculate Hamming Score (accuracy per label position) for multi-label classification.

    Hamming Score = (# of correctly predicted labels) / (total # of labels)
    averaged across all items.

    Args:
        data1: First rater's data
        data2: Second rater's data
        label_columns: List of label column names

    Returns:
        Dictionary containing Hamming metrics
    """
    hamming_scores = []
    n_items = len(data1)
    n_labels = len(label_columns)

    for row1, row2 in zip(data1, data2):
        correct = 0
        for label in label_columns:
            if get_binary_label(row1, label) == get_binary_label(row2, label):
                correct += 1

        # Hamming score for this item
        hamming_score = correct / n_labels if n_labels > 0 else 0.0
        hamming_scores.append(hamming_score)

    mean_hamming = np.mean(hamming_scores) if hamming_scores else 0.0

    # Also calculate Hamming Loss (1 - Hamming Score)
    mean_hamming_loss = 1.0 - mean_hamming

    return {
        'hamming_score': mean_hamming,
        'hamming_loss': mean_hamming_loss,
        'per_item_scores': hamming_scores
    }


def calculate_confusion_matrix_metrics(data1: List[Dict[str, str]],
                                       data2: List[Dict[str, str]],
                                       label_columns: List[str]) -> Dict:
    """
    Calculate confusion matrix based metrics (Positive/Negative Agreement).

    Args:
        data1: First rater's data
        data2: Second rater's data
        label_columns: List of label column names

    Returns:
        Dictionary containing confusion matrix metrics
    """
    per_label_metrics = {}

    for label in label_columns:
        rater1 = np.array([get_binary_label(row, label) for row in data1])
        rater2 = np.array([get_binary_label(row, label) for row in data2])

        # Calculate confusion matrix elements
        both_positive = np.sum((rater1 == 1) & (rater2 == 1))
        both_negative = np.sum((rater1 == 0) & (rater2 == 0))
        rater1_only = np.sum((rater1 == 1) & (rater2 == 0))
        rater2_only = np.sum((rater1 == 0) & (rater2 == 1))

        # Positive Agreement: P(both say yes | at least one says yes)
        positive_total = both_positive + rater1_only + rater2_only
        positive_agreement = both_positive / positive_total if positive_total > 0 else 0.0

        # Negative Agreement: P(both say no | at least one says no)
        negative_total = both_negative + rater1_only + rater2_only
        negative_agreement = both_negative / negative_total if negative_total > 0 else 0.0

        per_label_metrics[label] = {
            'positive_agreement': positive_agreement,
            'negative_agreement': negative_agreement,
            'both_positive': both_positive,
            'both_negative': both_negative,
            'rater1_only': rater1_only,
            'rater2_only': rater2_only
        }

    return per_label_metrics


def calculate_annotator_bias(data1: List[Dict[str, str]],
                             data2: List[Dict[str, str]],
                             label_columns: List[str],
                             rater1_name: str,
                             rater2_name: str) -> Dict:
    """
    Calculate annotator bias (label frequency comparison).

    Args:
        data1: First rater's data
        data2: Second rater's data
        label_columns: List of label column names
        rater1_name: Name of first rater
        rater2_name: Name of second rater

    Returns:
        Dictionary containing bias metrics
    """
    n_items = len(data1)
    bias_metrics = {}

    for label in label_columns:
        rater1_count = sum(get_binary_label(row, label) for row in data1)
        rater2_count = sum(get_binary_label(row, label) for row in data2)

        rater1_freq = rater1_count / n_items if n_items > 0 else 0.0
        rater2_freq = rater2_count / n_items if n_items > 0 else 0.0

        bias_metrics[label] = {
            f'{rater1_name}_count': rater1_count,
            f'{rater2_name}_count': rater2_count,
            f'{rater1_name}_freq': rater1_freq,
            f'{rater2_name}_freq': rater2_freq,
            'difference': rater1_count - rater2_count,
            'freq_difference': rater1_freq - rater2_freq
        }

    return bias_metrics


def extract_disagreement_items(data1: List[Dict[str, str]],
                               data2: List[Dict[str, str]],
                               label_columns: List[str],
                               jaccard_scores: List[float],
                               threshold: float = 0.5) -> List[Dict]:
    """
    Extract items with low agreement (Jaccard < threshold).

    Args:
        data1: First rater's data
        data2: Second rater's data
        label_columns: List of label column names
        jaccard_scores: List of Jaccard scores for each item
        threshold: Jaccard threshold for disagreement

    Returns:
        List of disagreement items with details
    """
    disagreement_items = []

    for idx, (row1, row2, jaccard) in enumerate(zip(data1, data2, jaccard_scores)):
        if jaccard < threshold:
            set1 = get_label_set(row1, label_columns)
            set2 = get_label_set(row2, label_columns)

            disagreement_items.append({
                'sha': row1.get('sha', ''),
                'message': row1.get('message', ''),
                'jaccard_score': jaccard,
                'rater1_labels': ', '.join(sorted(set1)),
                'rater2_labels': ', '.join(sorted(set2)),
                'only_rater1': ', '.join(sorted(set1 - set2)),
                'only_rater2': ', '.join(sorted(set2 - set1)),
                'common_labels': ', '.join(sorted(set1 & set2))
            })

    return disagreement_items


def export_disagreement_items(disagreement_items: List[Dict], output_file: str):
    """
    Export disagreement items to CSV.

    Args:
        disagreement_items: List of disagreement items
        output_file: Output CSV file path
    """
    if not disagreement_items:
        print(f"No disagreement items to export.")
        return

    fieldnames = ['sha', 'message', 'jaccard_score', 'rater1_labels', 'rater2_labels',
                  'only_rater1', 'only_rater2', 'common_labels']

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(disagreement_items)

    print(f"\n✓ Disagreement items exported to: {output_file}")


def print_results(n_items: int, n_labels: int, kappa_stats: Dict, micro_kappa: float,
                 alpha: float, set_metrics: Dict, hamming_metrics: Dict,
                 confusion_metrics: Dict, bias_metrics: Dict,
                 rater1_name: str, rater2_name: str) -> None:
    """
    Print the inter-rater agreement statistics.

    Args:
        n_items: Number of items compared
        n_labels: Number of label categories
        kappa_stats: Dictionary containing Macro-averaged Kappa statistics
        micro_kappa: Micro-averaged Kappa value
        alpha: Krippendorff's Alpha value
        set_metrics: Dictionary containing set-based metrics
        hamming_metrics: Dictionary containing Hamming metrics
        confusion_metrics: Dictionary containing confusion matrix metrics
        bias_metrics: Dictionary containing bias metrics
        rater1_name: Name of first rater
        rater2_name: Name of second rater
    """
    print("=" * 70)
    print("Inter-Rater Agreement Results")
    print("=" * 70)
    print(f"Number of common items analyzed: {n_items}")
    print(f"Number of label categories: {n_labels}")

    # 1. Label-wise Metrics (Kappa, Alpha)
    print("\n" + "=" * 70)
    print("1. Label-wise Agreement Metrics")
    print("=" * 70)
    print(f"Macro-Averaged Cohen's Kappa: {kappa_stats['macro_averaged_kappa']:.4f}")
    print(f"Micro-Averaged Cohen's Kappa: {micro_kappa:.4f}")
    print(f"Krippendorff's Alpha: {alpha:.4f}")

    print("\nNote:")
    print("  - Macro-average: Each label weighted equally")
    print("  - Micro-average: Each decision weighted equally (reflects label frequency)")

    print("\nInterpretation:")
    kappa = kappa_stats['macro_averaged_kappa']
    if kappa < 0:
        print("  Less than chance agreement")
    elif kappa < 0.20:
        print("  Slight agreement")
    elif kappa < 0.40:
        print("  Fair agreement")
    elif kappa < 0.60:
        print("  Moderate agreement")
    elif kappa < 0.80:
        print("  Substantial agreement")
    else:
        print("  Almost perfect agreement")

    # 2. Set-based Metrics
    print("\n" + "=" * 70)
    print("2. Set-based Metrics (Label Combination Agreement)")
    print("=" * 70)
    print(f"Mean Jaccard Coefficient: {set_metrics['mean_jaccard']:.4f}")
    print(f"Exact Match Ratio: {set_metrics['exact_match_ratio']:.4f} ({set_metrics['exact_matches']}/{set_metrics['total_items']})")
    print(f"Hamming Score: {hamming_metrics['hamming_score']:.4f}")
    print(f"Hamming Loss: {hamming_metrics['hamming_loss']:.4f}")
    print("\nInterpretation:")
    print(f"  - On average, {set_metrics['mean_jaccard']*100:.1f}% of labels overlap between raters (Jaccard)")
    print(f"  - {set_metrics['exact_match_ratio']*100:.1f}% of items have complete agreement (all labels match)")
    print(f"  - {hamming_metrics['hamming_score']*100:.1f}% of label decisions match across all items (Hamming)")

    # 3. Per-Label Kappa
    print("\n" + "=" * 70)
    print("3. Per-Label Cohen's Kappa")
    print("=" * 70)
    sorted_labels = sorted(kappa_stats['per_label_kappa'].items(),
                          key=lambda x: x[1], reverse=True)
    for label, kappa_val in sorted_labels:
        print(f"{label}: {kappa_val:.4f}")

    # 4. Positive/Negative Agreement
    print("\n" + "=" * 70)
    print("4. Positive/Negative Agreement (Confusion Matrix)")
    print("=" * 70)
    print("Label | Pos.Agr | Neg.Agr | Both+ | Both- | Only1 | Only2")
    print("-" * 70)
    for label in label_columns:
        metrics = confusion_metrics[label]
        print(f"{label[:30]:30} | {metrics['positive_agreement']:.3f} | "
              f"{metrics['negative_agreement']:.3f} | "
              f"{metrics['both_positive']:5d} | {metrics['both_negative']:5d} | "
              f"{metrics['rater1_only']:5d} | {metrics['rater2_only']:5d}")

    print("\nNote:")
    print("  - Pos.Agr: Agreement when at least one rater marks the label")
    print("  - Neg.Agr: Agreement when at least one rater doesn't mark it")
    print("  - Both+: Both raters marked the label")
    print("  - Both-: Neither rater marked the label")
    print("  - Only1/Only2: Only one rater marked the label")

    # 5. Annotator Bias
    print("\n" + "=" * 70)
    print("5. Annotator Bias (Label Frequency Comparison)")
    print("=" * 70)
    print(f"Label | {rater1_name:>15} | {rater2_name:>15} | Difference")
    print("-" * 70)

    total_r1 = 0
    total_r2 = 0

    for label in label_columns:
        metrics = bias_metrics[label]
        r1_count = metrics[f'{rater1_name}_count']
        r2_count = metrics[f'{rater2_name}_count']
        r1_freq = metrics[f'{rater1_name}_freq']
        r2_freq = metrics[f'{rater2_name}_freq']
        diff = metrics['difference']

        total_r1 += r1_count
        total_r2 += r2_count

        print(f"{label[:30]:30} | {r1_count:4d} ({r1_freq:.1%}) | "
              f"{r2_count:4d} ({r2_freq:.1%}) | {diff:+4d}")

    print("-" * 70)
    print(f"{'TOTAL':30} | {total_r1:4d} | {total_r2:4d} | {total_r1-total_r2:+4d}")

    print("\nInterpretation:")
    if abs(total_r1 - total_r2) > n_items * 0.1:  # More than 10% difference
        print(f"  ⚠ Large bias detected: {rater1_name} tends to mark more labels than {rater2_name}")
        print("    This suggests different interpretation thresholds between raters.")
    else:
        print(f"  ✓ Similar overall labeling frequency between raters")

    # Summary
    print("\n" + "=" * 70)
    print("Metrics Range Reference:")
    print("=" * 70)
    print("  1 = Perfect agreement")
    print("  0 = Agreement expected by chance")
    print(" <0 = Less agreement than expected by chance")
    print("=" * 70)


def main():
    """Main function to run the inter-rater agreement calculator."""
    if len(sys.argv) < 3:
        print("Usage: python inter_rater_agreement.py <rater1_csv> <rater2_csv> [--export-disagreements]")
        print("\nExample:")
        print("  python inter_rater_agreement.py kyogo_inspection.csv kosei_inspection.csv")
        print("  python inter_rater_agreement.py kyogo_inspection.csv kosei_inspection.csv --export-disagreements")
        sys.exit(1)

    csv_file1 = sys.argv[1]
    csv_file2 = sys.argv[2]
    export_disagreements = '--export-disagreements' in sys.argv

    # Extract rater names from filenames
    rater1_name = Path(csv_file1).stem.replace('_inspection', '')
    rater2_name = Path(csv_file2).stem.replace('_inspection', '')

    # Check if files exist
    if not Path(csv_file1).exists():
        print(f"Error: File '{csv_file1}' not found.")
        sys.exit(1)
    if not Path(csv_file2).exists():
        print(f"Error: File '{csv_file2}' not found.")
        sys.exit(1)

    try:
        # Read both CSV files
        headers1, data1 = read_csv_file(csv_file1)
        headers2, data2 = read_csv_file(csv_file2)

        if not data1 or not data2:
            print("Error: One or both CSV files are empty.")
            sys.exit(1)

        # Get label columns (assuming both files have the same labels)
        global label_columns
        label_columns = get_label_columns(headers1)

        if not label_columns:
            print("Error: No label columns found in CSV.")
            sys.exit(1)

        # Align data by SHA
        aligned1, aligned2 = align_data_by_sha(data1, data2)

        if not aligned1:
            print("Error: No common items (SHA) found between the two files.")
            sys.exit(1)

        print(f"Found {len(aligned1)} common items between the two raters.")

        # Calculate all metrics
        print("\nCalculating metrics...")

        # 1. Macro-averaged Kappa
        kappa_stats = calculate_macro_averaged_kappa(aligned1, aligned2, label_columns)

        # 2. Micro-averaged Kappa
        micro_kappa = calculate_micro_averaged_kappa(aligned1, aligned2, label_columns)

        # 3. Krippendorff's Alpha
        alpha = calculate_krippendorff_alpha(aligned1, aligned2, label_columns)

        # 4. Set-based Metrics
        set_metrics = calculate_set_based_metrics(aligned1, aligned2, label_columns)

        # 5. Hamming Score
        hamming_metrics = calculate_hamming_score(aligned1, aligned2, label_columns)

        # 6. Confusion Matrix Metrics
        confusion_metrics = calculate_confusion_matrix_metrics(aligned1, aligned2, label_columns)

        # 7. Annotator Bias
        bias_metrics = calculate_annotator_bias(aligned1, aligned2, label_columns, rater1_name, rater2_name)

        # Print results
        print_results(len(aligned1), len(label_columns), kappa_stats, micro_kappa, alpha,
                     set_metrics, hamming_metrics, confusion_metrics, bias_metrics,
                     rater1_name, rater2_name)

        # 6. Disagreement Analysis
        if export_disagreements:
            disagreement_items = extract_disagreement_items(
                aligned1, aligned2, label_columns,
                set_metrics['jaccard_scores'],
                threshold=0.5
            )

            if disagreement_items:
                output_file = f"disagreements_{rater1_name}_vs_{rater2_name}.csv"
                export_disagreement_items(disagreement_items, output_file)
                print(f"Exported {len(disagreement_items)} items with Jaccard < 0.5")
            else:
                print("\nNo disagreement items found (all Jaccard scores >= 0.5)")

    except Exception as e:
        print(f"Error processing CSV files: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
