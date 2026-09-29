# -*- coding: utf-8 -*-
"""
Phase 1-2: Download, save, and inspect CIC-IDS2017 data.
"""
import os
import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np

# Add src to path
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "raw" / "CIC-IDS2017"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("PHASE 1-2: ACQUIRE AND INSPECT CIC-IDS2017")
print("=" * 70)

# Load from Hugging Face
print("\n[1] Loading CIC-IDS2017 from Hugging Face...")
from datasets import load_dataset
ds = load_dataset('bvk/CICIDS-2017', split='train')
df = ds.to_pandas()
print(f"    Loaded {len(df):,} rows, {len(df.columns)} columns")

# Save to CSV
csv_path = DATA_DIR / "cic_ids2017_full.csv"
df.to_csv(csv_path, index=False)
file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
print(f"    Saved to {csv_path}")
print(f"    File size: {file_size_mb:.2f} MB")

# === DETAILED INSPECTION ===
print("\n" + "=" * 70)
print("DETAILED CSV INSPECTION")
print("=" * 70)

# File info
print(f"\n[FILENAME] {csv_path.name}")
print(f"[FILE SIZE] {file_size_mb:.2f} MB ({os.path.getsize(csv_path):,} bytes)")
print(f"[ROWS] {len(df):,}")
print(f"[COLUMNS] {len(df.columns)}")

# Exact column names
print(f"\n[COLUMN NAMES] ({len(df.columns)} columns):")
for i, col in enumerate(df.columns):
    print(f"  {i+1:3d}. {col} ({df[col].dtype})")

# Timestamp column analysis
print(f"\n[TIMESTAMP COLUMN]")
print(f"  Column name: 'Timestamp'")
print(f"  Dtype: {df['Timestamp'].dtype}")
print(f"  Sample values (first 10): {df['Timestamp'].head(10).tolist()}")
print(f"  Sample values (last 10): {df['Timestamp'].tail(10).tolist()}")

# Try to parse timestamp
print(f"\n  Attempting to parse timestamps...")
ts_series = pd.to_datetime(df['Timestamp'], errors='coerce', format='%M:%S.%f')
valid_ts = ts_series.dropna()
print(f"  Valid parsed timestamps: {len(valid_ts):,} out of {len(df):,}")
if len(valid_ts) > 0:
    print(f"  Min: {valid_ts.min()}")
    print(f"  Max: {valid_ts.max()}")
    print(f"  Span: {valid_ts.max() - valid_ts.min()}")

# Label column
print(f"\n[LABEL COLUMN] 'Label'")
label_counts = df['Label'].value_counts()
print(f"  Unique labels: {sorted(df['Label'].unique())}")
print(f"  Distribution:")
for label, count in label_counts.items():
    pct = (count / len(df)) * 100
    print(f"    {label}: {count:,} ({pct:.2f}%)")

# Attempted Category
print(f"\n[ATTEMPTED CATEGORY COLUMN]")
print(f"  Unique values: {df['Attempted Category'].unique()[:20]}")
print(f"  NaN count: {df['Attempted Category'].isna().sum():,}")

# Protocol column
print(f"\n[PROTOCOL COLUMN] 'Protocol'")
proto_counts = df['Protocol'].value_counts()
print(f"  Unique protocols: {sorted(df['Protocol'].unique())}")
for p, c in proto_counts.items():
    print(f"    {p}: {c:,}")

# Source/Destination columns
print(f"\n[ADDRESS COLUMNS]")
print(f"  Src IP dec: {df['Src IP dec'].nunique():,} unique, dtype={df['Src IP dec'].dtype}")
print(f"  Dst IP dec: {df['Dst IP dec'].nunique():,} unique, dtype={df['Dst IP dec'].dtype}")
print(f"  Src Port: min={df['Src Port'].min()}, max={df['Src Port'].max()}, nunique={df['Src Port'].nunique()}")
print(f"  Dst Port: min={df['Dst Port'].min()}, max={df['Dst Port'].max()}, nunique={df['Dst Port'].nunique()}")

# TCP Flags
print(f"\n[TCP FLAG COLUMNS]")
flag_cols = ['FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
             'ACK Flag Count', 'URG Flag Count', 'CWR Flag Count', 'ECE Flag Count']
for col in flag_cols:
    print(f"  {col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}, total_unique={df[col].nunique()}")

# Flow volume features
print(f"\n[FLOW VOLUME FEATURES]")
print(f"  Flow Duration: min={df['Flow Duration'].min()}, max={df['Flow Duration'].max()}, mean={df['Flow Duration'].mean():.0f}")
print(f"  Total Fwd Packet: min={df['Total Fwd Packet'].min()}, max={df['Total Fwd Packet'].max()}, mean={df['Total Fwd Packet'].mean():.1f}")
print(f"  Total Bwd packets: min={df['Total Bwd packets'].min()}, max={df['Total Bwd packets'].max()}, mean={df['Total Bwd packets'].mean():.1f}")
print(f"  Total Length of Fwd Packet: min={df['Total Length of Fwd Packet'].min()}, max={df['Total Length of Fwd Packet'].max()}, mean={df['Total Length of Fwd Packet'].mean():.0f}")
print(f"  Total Length of Bwd Packet: min={df['Total Length of Bwd Packet'].min()}, max={df['Total Length of Bwd Packet'].max()}, mean={df['Total Length of Bwd Packet'].mean():.0f}")

# Timing features
print(f"\n[TIMING / IAT FEATURES]")
print(f"  Flow IAT Mean: min={df['Flow IAT Mean'].min()}, max={df['Flow IAT Mean'].max()}, mean={df['Flow IAT Mean'].mean():.0f}")
print(f"  Fwd IAT Mean: min={df['Fwd IAT Mean'].min()}, max={df['Fwd IAT Mean'].max()}, mean={df['Fwd IAT Mean'].mean():.0f}")
print(f"  Bwd IAT Mean: min={df['Bwd IAT Mean'].min()}, max={df['Bwd IAT Mean'].max()}, mean={df['Bwd IAT Mean'].mean():.0f}")

# Packet features
print(f"\n[PACKET FEATURES]")
print(f"  Packet Length Mean: min={df['Packet Length Mean'].min()}, max={df['Packet Length Mean'].max()}, mean={df['Packet Length Mean'].mean():.1f}")
print(f"  Packet Length Std: min={df['Packet Length Std'].min()}, max={df['Packet Length Std'].max()}, mean={df['Packet Length Std'].mean():.1f}")
print(f"  Average Packet Size: min={df['Average Packet Size'].min()}, max={df['Average Packet Size'].max()}, mean={df['Average Packet Size'].mean():.1f}")

# Missing values
print(f"\n[MISSING VALUES]")
missing = df.isnull().sum()
missing_total = missing.sum()
missing_cols = missing[missing > 0].sort_values(ascending=False)
print(f"  Total missing: {missing_total:,}")
if len(missing_cols) > 0:
    for col, count in missing_cols.items():
        print(f"    {col}: {count:,} ({count/len(df)*100:.2f}%)")

# Infinite values
print(f"\n[INFINITE VALUES]")
inf_count = 0
for col in df.select_dtypes(include=[np.number]).columns:
    n_inf = np.isinf(df[col].values).sum()
    if n_inf > 0:
        inf_count += n_inf
        print(f"    {col}: {n_inf:,}")
print(f"  Total infinite: {inf_count:,}")

# Duplicate rows
print(f"\n[DUPLICATE ROWS]")
n_dups = df.duplicated().sum()
print(f"  Full duplicate rows: {n_dups:,}")

# Attack vs benign distribution by day if we can determine time
print(f"\n[TEMPORAL DISTRIBUTION]")
# Check if we can determine which attack is in which time period
# Based on the Hugging Face data, let's look at timestamp patterns
print(f"  Timestamp type: {type(df['Timestamp'].iloc[0])}")
print(f"  Timestamp sample range: {df['Timestamp'].iloc[0]} to {df['Timestamp'].iloc[-1]}")

# Save inspection summary
summary = {
    "filename": "cic_ids2017_full.csv",
    "file_size_mb": round(file_size_mb, 2),
    "total_rows": len(df),
    "total_columns": len(df.columns),
    "missing_total": int(missing_total),
    "infinite_total": int(inf_count),
    "duplicate_rows": int(n_dups),
    "columns": list(df.columns),
    "dtypes": {col: str(df[col].dtype) for col in df.columns},
    "labels": {label: int(count) for label, count in df['Label'].value_counts().items()},
    "protocols": {int(p): int(c) for p, c in proto_counts.items()},
    "timestamp_sample_first": str(df['Timestamp'].iloc[0]),
    "timestamp_sample_last": str(df['Timestamp'].iloc[-1]),
}

with open(DATA_DIR / "inspection_summary.json", 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\n[✓] Inspection summary saved to {DATA_DIR / 'inspection_summary.json'}")
print(f"\n{'=' * 70}")
print("PHASE 1-2 COMPLETE")
print(f"{'=' * 70}")
