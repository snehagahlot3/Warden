#!/usr/bin/env python3
"""
Dataset Inspection Script for WARDEN-T

Reports:
- number of rows
- number of columns
- column names
- data types
- missing values
- infinite values
- label distribution
- timestamp range
- unique attack labels
- unique protocols
- approximate file size

Usage:
    python inspect_dataset.py --data-dir <path> [--label-col <col>] [--timestamp-col <col>]

Example:
    python inspect_dataset.py --data-dir data/raw/cic-ids2017/csv/
    python inspect_dataset.py --data-dir data/raw/unsw-nb15/csv/ --label-col Label --timestamp-col Stime
    python inspect_dataset.py --data-dir data/raw/ciciot2023/ --label-col label
"""

import argparse
import os
import sys
import math
from pathlib import Path

import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect network traffic datasets for WARDEN-T")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to the directory containing CSV files")
    parser.add_argument("--label-col", type=str, default=None,
                        help="Name of the label column (auto-detected if not specified)")
    parser.add_argument("--timestamp-col", type=str, default=None,
                        help="Name of the timestamp column (auto-detected if not specified)")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional output file path for the report")
    return parser.parse_args()


def find_csv_files(data_dir):
    """Find all CSV files in the directory."""
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"ERROR: Directory {data_dir} does not exist.")
        sys.exit(1)
    csv_files = sorted(data_path.glob("*.csv"))
    csv_files += sorted(data_path.glob("*.CSV"))
    return csv_files


def load_all_csvs(csv_files):
    """Load all CSV files and concatenate them."""
    if not csv_files:
        print(f"ERROR: No CSV files found in {csv_files[0].parent}")
        sys.exit(1)

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            dfs.append(df)
        except Exception as e:
            print(f"WARNING: Could not read {f}: {e}")

    if not dfs:
        print("ERROR: Could not load any CSV files.")
        sys.exit(1)

    combined = pd.concat(dfs, ignore_index=True)
    return combined


def count_infinite(df):
    """Count infinite values across all columns."""
    inf_count = 0
    inf_cols = {}
    for col in df.columns:
        if df[col].dtype in [np.float64, np.float32, np.float16]:
            n_inf = np.isinf(df[col].values).sum()
            if n_inf > 0:
                inf_count += n_inf
                inf_cols[col] = n_inf
    return inf_count, inf_cols


def detect_label_column(df):
    """Attempt to auto-detect the label column."""
    candidates = ["Label", "label", "attack_cat", "attack_category", "Attack",
                  "class", "target", "Category", "category", "Label_"]
    for col in candidates:
        if col in df.columns:
            return col
    # Check for columns with mostly binary or few unique values that aren't IDs
    for col in df.columns:
        n_unique = df[col].nunique()
        if n_unique <= 20 and df[col].dtype in [object, int, np.int64]:
            if col.lower() not in ["srcip", "dstip", "service", "proto", "state"]:
                return col
    return None


def detect_timestamp_column(df):
    """Attempt to auto-detect the timestamp column."""
    candidates = ["Timestamp", "timestamp", "Stime", "Ltime", "ts", "time",
                  "StartTime", "StartTimeStamp", "record_start_time"]
    for col in candidates:
        if col in df.columns:
            return col
    # Look for datetime-like columns
    for col in df.columns:
        if "time" in col.lower() or "date" in col.lower():
            if col not in ["Label", "label", "attack_cat"]:
                return col
    return None


def format_size(bytes_count):
    """Format bytes into human-readable size."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_count < 1024.0:
            return f"{bytes_count:.2f} {unit}"
        bytes_count /= 1024.0
    return f"{bytes_count:.2f} PB"


def inspect_dataset(data_dir, label_col=None, timestamp_col=None, output=None):
    """Main inspection logic."""
    csv_files = find_csv_files(data_dir)
    print(f"\n{'='*70}")
    print(f"WARDEN-T DATASET INSPECTION REPORT")
    print(f"{'='*70}")
    print(f"Data Directory: {os.path.abspath(data_dir)}")
    print(f"CSV Files Found: {len(csv_files)}")
    for f in csv_files:
        print(f"  - {f.name} ({format_size(f.stat().st_size)})")

    # Load data
    print(f"\n--- Loading Data ---")
    df = load_all_csvs(csv_files)
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")

    # File size of the directory
    dir_size = sum(f.stat().st_size for f in csv_files)
    print(f"Approximate CSV data size: {format_size(dir_size)}")

    # Column names and data types
    print(f"\n--- Column Names and Data Types ({len(df.columns)} columns) ---")
    print(f"{'Column Name':<40} {'Dtype':<15}")
    print(f"{'-'*40} {'-'*15}")
    for col in df.columns:
        print(f"{col:<40} {str(df[col].dtype):<15}")

    # Missing values
    print(f"\n--- Missing Values ---")
    missing = df.isnull().sum()
    missing_total = missing.sum()
    missing_cols = missing[missing > 0].sort_values(ascending=False)
    print(f"Total missing values: {missing_total:,}")
    if len(missing_cols) > 0:
        print(f"Columns with missing values ({len(missing_cols)}):")
        for col, count in missing_cols.items():
            pct = (count / len(df)) * 100
            print(f"  {col}: {count:,} ({pct:.2f}%)")
    else:
        print("No missing values found.")

    # Infinite values
    print(f"\n--- Infinite Values ---")
    inf_count, inf_cols = count_infinite(df)
    print(f"Total infinite values: {inf_count:,}")
    if inf_cols:
        print(f"Columns with infinite values ({len(inf_cols)}):")
        for col, count in sorted(inf_cols.items(), key=lambda x: -x[1]):
            print(f"  {col}: {count:,}")
    else:
        print("No infinite values found.")

    # Label distribution
    if label_col is None:
        label_col = detect_label_column(df)
    if label_col:
        print(f"\n--- Label Distribution (column: '{label_col}') ---")
        label_dist = df[label_col].value_counts()
        total = len(df)
        for label, count in label_dist.items():
            pct = (count / total) * 100
            print(f"  {label}: {count:,} ({pct:.2f}%)")
        print(f"Unique labels: {df[label_col].nunique()}")
    else:
        print("\n--- Label Distribution ---")
        print("Could not auto-detect label column. Consider specifying --label-col")

    # Timestamp range
    if timestamp_col is None:
        timestamp_col = detect_timestamp_column(df)
    if timestamp_col and timestamp_col in df.columns:
        print(f"\n--- Timestamp Range (column: '{timestamp_col}') ---")
        ts_col = df[timestamp_col]
        # Try to convert to datetime if not already
        try:
            ts_series = pd.to_datetime(ts_col, errors='coerce')
            valid_ts = ts_series.dropna()
            if len(valid_ts) > 0:
                print(f"  Min timestamp: {valid_ts.min()}")
                print(f"  Max timestamp: {valid_ts.max()}")
                print(f"  Time span: {valid_ts.max() - valid_ts.min()}")
                print(f"  Total records with valid timestamps: {len(valid_ts):,}")
            else:
                print(f"  Could not parse timestamps as datetime.")
                print(f"  Min raw value: {ts_col.min()}")
                print(f"  Max raw value: {ts_col.max()}")
        except Exception as e:
            print(f"  Error parsing timestamps: {e}")
            print(f"  Min raw value: {ts_col.min()}")
            print(f"  Max raw value: {ts_col.max()}")
    else:
        print(f"\n--- Timestamp Range ---")
        if timestamp_col:
            print(f"  Column '{timestamp_col}' found but could not parse.")
        else:
            print("Could not auto-detect timestamp column. Consider specifying --timestamp-col")

    # Unique attack labels
    if label_col and label_col in df.columns:
        print(f"\n--- Unique Attack Labels ---")
        unique_labels = sorted(df[label_col].unique())
        for i, label in enumerate(unique_labels):
            count = (df[label_col] == label).sum()
            print(f"  {i+1}. {label} ({count:,} records)")

    # Unique protocols
    proto_cols = [c for c in df.columns if 'proto' in c.lower() and c.lower() != 'proto' or c == 'proto']
    if proto_cols:
        print(f"\n--- Unique Protocols ---")
        for pc in proto_cols:
            unique_protos = df[pc].unique()
            print(f"  Column '{pc}': {sorted(unique_protos.tolist())[:20]}...")
            if len(unique_protos) <= 20:
                for p in sorted(unique_protos.tolist()):
                    count = (df[pc] == p).sum()
                    print(f"    {p}: {count:,}")
            else:
                print(f"    (and {len(unique_protos)-20} more)")
    else:
        # Check for protocol-like columns
        proto_like = [c for c in df.columns if 'protocol' in c.lower() or 'Protocol' in c]
        if proto_like:
            print(f"\n--- Unique Protocols ---")
            for pc in proto_like:
                unique_vals = df[pc].unique()
                print(f"  Column '{pc}': {sorted(unique_vals.tolist())[:20]}")

    # Summary statistics for numeric columns
    print(f"\n--- Numeric Summary (first 10 columns) ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:10]
    if len(numeric_cols) > 0:
        desc = df[numeric_cols].describe()
        print(desc.to_string())

    # Unique counts for key columns
    print(f"\n--- Key Column Unique Counts ---")
    key_cols = ["Timestamp", "timestamp", "Source IP", "sourceip", "SourceIP",
                "Destination IP", "dstip", "DestinationIP", "Source Port",
                "sport", "Destination Port", "dsport", "Protocol", "proto"]
    for col in key_cols:
        if col in df.columns:
            print(f"  {col}: {df[col].nunique():,} unique values")

    # Attack category columns if present
    attack_cols = [c for c in df.columns if 'attack' in c.lower() or 'Attack' in c]
    if attack_cols:
        print(f"\n--- Attack-related Columns ---")
        for ac in attack_cols:
            print(f"  {ac}: {df[ac].nunique()} unique values")

    # Total file count
    print(f"\n--- Summary ---")
    print(f"Total CSV files processed: {len(csv_files)}")
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Missing values: {missing_total:,}")
    print(f"Infinite values: {inf_count:,}")
    if label_col:
        print(f"Label column: {label_col} ({df[label_col].nunique()} classes)")
    if timestamp_col:
        print(f"Timestamp column: {timestamp_col}")
    print(f"{'='*70}")

    # Save to output file if specified
    if output:
        with open(output, 'w') as f:
            # Redirect stdout temporarily
            import io
            old_stdout = sys.stdout
            sys.stdout = f
            inspect_dataset(data_dir, label_col, timestamp_col)
            sys.stdout = old_stdout
        print(f"\nReport saved to: {output}")


if __name__ == "__main__":
    args = parse_args()
    inspect_dataset(args.data_dir, args.label_col, args.timestamp_col, args.output)
