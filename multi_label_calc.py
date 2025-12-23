#!/usr/bin/env python3
"""
Multi-Label Match Rate Calculator

This script calculates the match rate (agreement rate) for multi-label classifications
in a CSV file. The CSV should contain columns where "o" marks applicable categories.
"""

import csv
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple


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
        headers = reader.fieldnames
        data = list(reader)
    return headers, data


def get_label_columns(headers: List[str]) -> List[str]:
    """
    Get the label columns from headers.
    Excludes metadata columns like message, pr_id, repo_url, commit_url, Memo.
    
    Args:
        headers: List of column headers
        
    Returns:
        List of label column names
    """
    metadata_columns = {'sha', 'message', 'pr_id', 'repo_url', 'commit_url', 'Memo'}
    return [col for col in headers if col not in metadata_columns]


def get_labels_for_row(row: Dict[str, str], label_columns: List[str]) -> Set[str]:
    """
    Get the set of labels marked with "o" for a given row.
    
    Args:
        row: Dictionary representing a row from the CSV
        label_columns: List of label column names
        
    Returns:
        Set of label names that are marked with "o"
    """
    return {col for col in label_columns if row.get(col, '').strip().lower() == 'o'}


def calculate_match_rate(data: List[Dict[str, str]], label_columns: List[str]) -> Dict:
    """
    Calculate match rate statistics for multi-label data.
    
    For pair-wise comparison, this calculates the Jaccard similarity
    (intersection over union) between label sets.
    
    Args:
        data: List of data rows
        label_columns: List of label column names
        
    Returns:
        Dictionary containing match rate statistics
    """
    if len(data) < 2:
        return {
            'total_rows': len(data),
            'error': 'Need at least 2 rows to calculate match rate'
        }
    
    # Get label sets for all rows
    label_sets = [get_labels_for_row(row, label_columns) for row in data]
    
    # Calculate pairwise match rates (Jaccard similarity)
    total_pairs = 0
    total_similarity = 0.0
    
    for i in range(len(label_sets)):
        for j in range(i + 1, len(label_sets)):
            set1 = label_sets[i]
            set2 = label_sets[j]
            
            # Jaccard similarity: |intersection| / |union|
            if len(set1) == 0 and len(set2) == 0:
                # Both empty - perfect match
                similarity = 1.0
            else:
                similarity = len(set1.intersection(set2)) / len(set1.union(set2))
            
            total_similarity += similarity
            total_pairs += 1
    
    average_match_rate = total_similarity / total_pairs if total_pairs > 0 else 0.0
    
    # Calculate label frequency
    label_counts = {}
    for label_set in label_sets:
        for label in label_set:
            label_counts[label] = label_counts.get(label, 0) + 1
    
    return {
        'total_rows': len(data),
        'total_labels': len(label_columns),
        'total_pairs': total_pairs,
        'average_match_rate': average_match_rate,
        'average_match_percentage': average_match_rate * 100,
        'label_frequency': label_counts
    }


def print_results(stats: Dict) -> None:
    """
    Print the match rate statistics in a readable format.
    
    Args:
        stats: Dictionary containing match rate statistics
    """
    if 'error' in stats:
        print(f"Error: {stats['error']}")
        print(f"Total rows: {stats['total_rows']}")
        return
    
    print("=" * 60)
    print("Multi-Label Match Rate Results")
    print("=" * 60)
    print(f"Total rows analyzed: {stats['total_rows']}")
    print(f"Total label categories: {stats['total_labels']}")
    print(f"Total pairwise comparisons: {stats['total_pairs']}")
    print(f"\nAverage Match Rate (Jaccard Similarity): {stats['average_match_rate']:.4f}")
    print(f"Average Match Percentage: {stats['average_match_percentage']:.2f}%")
    
    if stats.get('label_frequency'):
        print("\n" + "=" * 60)
        print("Label Frequency")
        print("=" * 60)
        sorted_labels = sorted(stats['label_frequency'].items(), 
                              key=lambda x: x[1], reverse=True)
        for label, count in sorted_labels:
            percentage = (count / stats['total_rows']) * 100
            print(f"{label}: {count} ({percentage:.1f}%)")


def main():
    """Main function to run the multi-label match rate calculator."""
    if len(sys.argv) < 2:
        print("Usage: python multi_label_calc.py <csv_file>")
        print("\nExample:")
        print("  python multi_label_calc.py data.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    if not Path(csv_file).exists():
        print(f"Error: File '{csv_file}' not found.")
        sys.exit(1)
    
    try:
        # Read CSV file
        headers, data = read_csv_file(csv_file)
        
        if not data:
            print("Error: CSV file is empty.")
            sys.exit(1)
        
        # Get label columns
        label_columns = get_label_columns(headers)
        
        if not label_columns:
            print("Error: No label columns found in CSV.")
            sys.exit(1)
        
        # Calculate match rate
        stats = calculate_match_rate(data, label_columns)
        
        # Print results
        print_results(stats)
        
    except Exception as e:
        print(f"Error processing CSV file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()