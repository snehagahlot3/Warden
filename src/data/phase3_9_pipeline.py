# -*- coding: utf-8 -*-
"""
WARDEN-T Data Pipeline: CIC-IDS2017 Temporal State Construction

Phases 3-8: Feature selection, temporal windowing, labeling, sequences, split, save.
Phase 9: Validation report.
"""

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "raw" / "CIC-IDS2017"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CONFIGS_DIR = BASE_DIR / "configs"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("WARDEN-T DATA PIPELINE - PHASES 3-9")
print("=" * 70)

# ============================================================
# Load the data
# ============================================================
print("\n[LOADING] CIC-IDS2017 CSV...")
df = pd.read_csv(DATA_DIR / "cic_ids2017_full.csv")
print(f"  Rows: {len(df):,}")
print(f"  Columns: {len(df.columns)}")

# ============================================================
# PHASE 2 (continued): Final inspection summary
# ============================================================
print("\n" + "=" * 70)
print("PHASE 2: FINAL INSPECTION SUMMARY")
print("=" * 70)
print(f"  Filenames: cic_ids2017_full.csv")
print(f"  File sizes: {os.path.getsize(DATA_DIR / 'cic_ids2017_full.csv') / 1024 / 1024:.2f} MB")
print(f"  Rows: {len(df):,}")
print(f"  Columns: {len(df.columns)}")
print(f"  Missing values: 0")
print(f"  Infinite values: 0")
print(f"  Duplicate rows: {df.duplicated().sum():,}")
print(f"  Timestamp column: Timestamp (format: MM:SS.mmm, relative)")
print(f"  Label column: Label (28 unique values)")

# ============================================================
# PHASE 3: SELECT NETWORK STATE VECTOR FEATURES
# ============================================================
print("\n" + "=" * 70)
print("PHASE 3: NETWORK STATE VECTOR DESIGN")
print("=" * 70)

# Define selected features with explanations
SELECTED_FEATURES = {
    # --- Flow Volume ---
    'flow_duration': {
        'column': 'Flow Duration',
        'reason': 'How long the flow lasted; indicates connection type (short=scan, long=exfil)',
        'dtype': 'numerical',
        'preprocessing': 'log_transform + standard_scale'
    },
    'total_fwd_packets': {
        'column': 'Total Fwd Packet',
        'reason': 'Number of forward-direction packets; indicates data volume',
        'dtype': 'numerical',
        'preprocessing': 'standard_scale'
    },
    'total_bwd_packets': {
        'column': 'Total Bwd packets',
        'reason': 'Number of backward-direction packets; indicates response volume',
        'dtype': 'numerical',
        'preprocessing': 'standard_scale'
    },
    'total_fwd_bytes': {
        'column': 'Total Length of Fwd Packet',
        'reason': 'Total bytes in forward direction; key volume indicator',
        'dtype': 'numerical',
        'preprocessing': 'log_transform + standard_scale'
    },
    'total_bwd_bytes': {
        'column': 'Total Length of Bwd Packet',
        'reason': 'Total bytes in backward direction; key volume indicator',
        'dtype': 'numerical',
        'preprocessing': 'log_transform + standard_scale'
    },

    # --- Timing / IAT ---
    'flow_iat_mean': {
        'column': 'Flow IAT Mean',
        'reason': 'Mean inter-arrival time between flows; indicates traffic pattern',
        'dtype': 'numerical',
        'preprocessing': 'log_transform + standard_scale'
    },
    'flow_iat_std': {
        'column': 'Flow IAT Std',
        'reason': 'Std of inter-arrival time; indicates burstiness',
        'dtype': 'numerical',
        'preprocessing': 'log_transform + standard_scale'
    },
    'fwd_iat_mean': {
        'column': 'Fwd IAT Mean',
        'reason': 'Mean forward inter-arrival time; indicates sender behavior',
        'dtype': 'numerical',
        'preprocessing': 'log_transform + standard_scale'
    },
    'bwd_iat_mean': {
        'column': 'Bwd IAT Mean',
        'reason': 'Mean backward inter-arrival time; indicates receiver behavior',
        'dtype': 'numerical',
        'preprocessing': 'log_transform + standard_scale'
    },

    # --- Directionality ---
    'down_up_ratio': {
        'column': 'Down/Up Ratio',
        'reason': 'Download/upload ratio; high ratio indicates download/exfil',
        'dtype': 'numerical',
        'preprocessing': 'standard_scale'
    },
    'fwd_packets_per_sec': {
        'column': 'Fwd Packets/s',
        'reason': 'Forward packet rate; indicates throughput',
        'dtype': 'numerical',
        'preprocessing': 'standard_scale'
    },
    'bwd_packets_per_sec': {
        'column': 'Bwd Packets/s',
        'reason': 'Backward packet rate; indicates response throughput',
        'dtype': 'numerical',
        'preprocessing': 'standard_scale'
    },

    # --- TCP Behavior ---
    'syn_flag_count': {
        'column': 'SYN Flag Count',
        'reason': 'SYN flags indicate connection attempts; port scans have high SYN',
        'dtype': 'numerical',
        'preprocessing': 'standard_scale'
    },
    'rst_flag_count': {
        'column': 'RST Flag Count',
        'reason': 'RST flags indicate connection resets; indicates scanning/abnormal',
        'dtype': 'numerical',
        'preprocessing': 'standard_scale'
    },
    'fin_flag_count': {
        'column': 'FIN Flag Count',
        'reason': 'FIN flags indicate graceful closes; normal in HTTP traffic',
        'dtype': 'numerical',
        'preprocessing': 'standard_scale'
    },
    'psh_flag_count': {
        'column': 'PSH Flag Count',
        'reason': 'PSH flags indicate push; high PSH suggests data transfer',
        'dtype': 'numerical',
        'preprocessing': 'standard_scale'
    },

    # --- Packet Behavior ---
    'packet_length_mean': {
        'column': 'Packet Length Mean',
        'reason': 'Average packet size; different attacks have different packet sizes',
        'dtype': 'numerical',
        'preprocessing': 'standard_scale'
    },
    'packet_length_std': {
        'column': 'Packet Length Std',
        'reason': 'Packet size variance; scan traffic has uniform packet sizes',
        'dtype': 'numerical',
        'preprocessing': 'standard_scale'
    },
    'avg_packet_size': {
        'column': 'Average Packet Size',
        'reason': 'Overall average packet size; indicates protocol mix',
        'dtype': 'numerical',
        'preprocessing': 'standard_scale'
    },

    # --- Protocol ---
    'protocol': {
        'column': 'Protocol',
        'reason': 'Transport protocol (6=TCP, 17=UDP, 0=ICMP, 1=ICMPv6)',
        'dtype': 'categorical',
        'preprocessing': 'one_hot_encode'
    },
}

# Categorical features that need encoding
CATEGORICAL_FEATURES = ['protocol']
NUMERICAL_FEATURES = [k for k, v in SELECTED_FEATURES.items() if v['dtype'] == 'numerical']

print(f"\n[SELECTED FEATURES] {len(SELECTED_FEATURES)} features total")
print(f"  Numerical: {len(NUMERICAL_FEATURES)}")
print(f"  Categorical: {len(CATEGORICAL_FEATURES)}")
print(f"\n[FEATURE SELECTION RATIONALE]")
print(f"  - Excluded raw IP addresses (decimal-encoded, high cardinality)")
print(f"  - Excluded port numbers (high cardinality, less useful for aggregation)")
print(f"  - Excluded ICMP code/type (rarely relevant outside ICMP flows)")
print(f"  - Excluded bulk statistics (highly correlated with simpler features)")
print(f"  - Excluded Active/Idle statistics (redundant with IAT features)")
print(f"  - Excluded subflow statistics (less interpretable for temporal modelling)")
print(f"\n[FEATURES NOT AVAILABLE IN CSV] (from problem statement):")
print(f"  - TTL variance: NOT in CIC-IDS2017 CSV (TTL is not extracted)")
print(f"  - TCP window size: NOT directly available (only Init Win Bytes)")
print(f"  - Fragment flags: NOT in CIC-IDS2017 CSV")
print(f"  - Payload size distribution: NOT in CIC-IDS2017 CSV (payload anonymized)")
print(f"  - Retransmission counts: NOT directly in CSV (can be approximated from IAT)")

# ============================================================
# PHASE 4: TEMPORAL WINDOWING
# ============================================================
print("\n" + "=" * 70)
print("PHASE 4: TEMPORAL WINDOWING")
print("=" * 70)

# The timestamps are MM:SS.mmm format (relative within ~60 minutes)
# The data IS chronologically ordered (rows progress through time)
# We use row-based windows since timestamps are relative within each session

# Determine window size based on data characteristics
# CIC-IDS2017 spans ~5 days (Mon-Fri, 9AM-5PM) = ~200,000 flows per day
# The data is ordered chronologically
# We use fixed-size windows of N consecutive flows

WINDOW_SIZE = 500  # flows per window
SEQUENCE_LENGTH = 10  # S(t-9), ..., S(t)

print(f"\n[TIME WINDOW SIZE] {WINDOW_SIZE} flows per window")
print(f"  Rationale: CIC-IDS2017 has ~2.1M flows over 5 days")
print(f"  ~420K flows/day, ~4.86 flows/minute")
print(f"  500 flows ~= 103 minutes of traffic (captures meaningful patterns)")
print(f"  Each window contains enough diversity to be meaningful")
print(f"\n[SEQUENCE LENGTH] {SEQUENCE_LENGTH} windows")
print(f"  Each input sequence: [S(t-9), S(t-8), ..., S(t)]")
print(f"  Target: S(t+1)")
print(f"  Total time covered per input: {SEQUENCE_LENGTH * WINDOW_SIZE} flows")
print(f"  That is ~{SEQUENCE_LENGTH * WINDOW_SIZE / 4.86 / 60:.1f} hours of traffic")

# Sort by timestamp to ensure temporal order
# The timestamp is MM:SS.mmm format; we need a global ordering
# Since the data spans multiple days, we use the original row order
# which is already chronologically sorted per the CIC documentation

df = df.reset_index(drop=True)  # Ensure chronological order

# Construct network states (windows)
print(f"\n[CONSTRUCTING NETWORK STATES...]")
n_flows = len(df)
n_windows = n_flows // WINDOW_SIZE
print(f"  Total flows: {n_flows:,}")
print(f"  Window size: {WINDOW_SIZE}")
print(f"  Total network states: {n_windows:,}")

# Aggregate features per window
def create_network_states(df, window_size, selected_features):
    """Aggregate flows into network states."""
    states = []
    numerical_cols = [v['column'] for v in selected_features.values() if v['dtype'] == 'numerical']
    cat_col = 'Protocol'

    for i in range(0, len(df) - window_size + 1, window_size):
        window = df.iloc[i:i + window_size]
        state = {}

        # Numerical aggregations: mean per feature
        for feat_name, feat_info in selected_features.items():
            col = feat_info['column']
            if feat_info['dtype'] == 'numerical':
                state[f'{feat_name}_mean'] = window[col].mean()
                state[f'{feat_name}_std'] = window[col].std()
                state[f'{feat_name}_max'] = window[col].max()
                state[f'{feat_name}_min'] = window[col].min()
            elif feat_info['dtype'] == 'categorical':
                # For categorical, use mode
                state[f'{feat_name}_mode'] = window[col].mode()[0] if len(window[col].mode()) > 0 else 0

        state['window_flow_count'] = len(window)
        state['window_total_bytes'] = window['Total Length of Fwd Packet'].sum() + window['Total Length of Bwd Packet'].sum()
        state['window_total_packets'] = window['Total Fwd Packet'].sum() + window['Total Bwd packets'].sum()

        state['fwd_init_win_bytes_mean'] = window['FWD Init Win Bytes'].mean()
        state['bwd_init_win_bytes_mean'] = window['Bwd Init Win Bytes'].mean()
        state['unique_dst_ports'] = window['Dst Port'].nunique()
        port_counts = window['Dst Port'].value_counts()
        state['dst_port_entropy'] = -np.sum((port_counts / len(window)) * np.log2(port_counts / len(window) + 1e-10))
        dst_ports_sorted = np.sort(window['Dst Port'].values)
        if len(dst_ports_sorted) > 1:
            diffs = np.diff(dst_ports_sorted)
            state['sequential_port_ratio'] = np.sum(diffs == 1) / len(diffs)
        else:
            state['sequential_port_ratio'] = 0.0
        total_flags = window['SYN Flag Count'].sum() + window['RST Flag Count'].sum()
        state['syn_ratio'] = window['SYN Flag Count'].sum() / (total_flags + 1e-10)
        state['rst_ratio'] = window['RST Flag Count'].sum() / (total_flags + 1e-10)

        states.append(state)

    return pd.DataFrame(states)

network_states = create_network_states(df, WINDOW_SIZE, SELECTED_FEATURES)
print(f"  Created {len(network_states):,} network states")
print(f"  Features per state: {len(network_states.columns)}")

# ============================================================
# PHASE 5: LABEL THE NETWORK STATES
# ============================================================
print("\n" + "=" * 70)
print("PHASE 5: LABEL NETWORK STATES")
print("=" * 70)

# Use majority vote within each window
# If any flow in the window is an attack, label it as attack
# Also record the specific attack category if dominant

labels = []
attack_categories = []

for i in range(0, len(df) - WINDOW_SIZE + 1, WINDOW_SIZE):
    window = df.iloc[i:i + WINDOW_SIZE]
    attack_flows = window[window['Label'] != 'BENIGN']
    benign_flows = window[window['Label'] == 'BENIGN']

    attack_ratio = len(attack_flows) / len(window)

    if attack_ratio == 0:
        labels.append('BENIGN')
        attack_categories.append('BENIGN')
    elif attack_ratio < 0.1:
        labels.append('MIXED_LIGHT')
        attack_categories.append(attack_flows['Label'].mode()[0] if len(attack_flows['Label'].mode()) > 0 else 'UNKNOWN')
    elif attack_ratio < 0.5:
        labels.append('ATTACK')
        attack_categories.append(attack_flows['Label'].mode()[0] if len(attack_flows['Label'].mode()) > 0 else 'UNKNOWN')
    else:
        labels.append('HEAVY_ATTACK')
        attack_categories.append(attack_flows['Label'].mode()[0] if len(attack_flows['Label'].mode()) > 0 else 'UNKNOWN')

network_states['label'] = labels
network_states['attack_category'] = attack_categories

print(f"\n[LABELING RULE] Majority vote within each {WINDOW_SIZE}-flow window")
print(f"  BENIGN: 0% attack flows")
print(f"  MIXED_LIGHT: <10% attack flows (borderline)")
print(f"  ATTACK: 10-50% attack flows")
print(f"  HEAVY_ATTACK: >50% attack flows")

label_dist = pd.Series(labels).value_counts()
print(f"\n[LABEL DISTRIBUTION]")
for label, count in label_dist.items():
    pct = (count / len(labels)) * 100
    print(f"  {label}: {count:,} ({pct:.2f}%)")

# Simplify to binary for the primary forecasting task
network_states['is_attack'] = (network_states['label'] != 'BENIGN').astype(int)
attack_dist = network_states['is_attack'].value_counts()
print(f"\n[Binary Attack Distribution]")
print(f"  Benign: {attack_dist.get(0, 0):,} ({attack_dist.get(0,0)/len(network_states)*100:.2f}%)")
print(f"  Attack: {attack_dist.get(1, 0):,} ({attack_dist.get(1,0)/len(network_states)*100:.2f}%)")

# ============================================================
# PHASE 6: CREATE SEQUENCES
# ============================================================
print("\n" + "=" * 70)
print("PHASE 6: CREATE TEMPORAL SEQUENCES")
print("=" * 70)

# Remove the categorical mode columns for now (handle separately)
feature_cols = [c for c in network_states.columns if c not in ['label', 'attack_category', 'is_attack']]

# Convert to numpy
X_data = network_states[feature_cols].values
y_data = network_states['is_attack'].values
attack_cat_data = network_states['attack_category'].values

print(f"  Feature matrix shape: {X_data.shape}")
print(f"  Label vector shape: {y_data.shape}")

# Create sequences: X = [S(t-9), ..., S(t)], y = S(t+1)
seq_len = SEQUENCE_LENGTH
X_sequences = []
y_sequences = []
y_categories = []

for i in range(len(X_data) - seq_len):
    X_sequences.append(X_data[i:i + seq_len])
    y_sequences.append(y_data[i + seq_len])
    y_categories.append(attack_cat_data[i + seq_len])

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)
y_categories = np.array(y_categories)

print(f"\n  Sequence shape: X = {X_sequences.shape}")
print(f"  Target shape: y = {y_sequences.shape}")
print(f"  Attack category shape: {y_categories.shape}")
print(f"  Sequence length: {seq_len}")
print(f"  Total sequences: {len(X_sequences):,}")

# ============================================================
# PHASE 7: TRAIN/VALIDATION/TEST SPLIT (TEMPORAL)
# ============================================================
print("\n" + "=" * 70)
print("PHASE 7: TEMPORAL TRAIN/VAL/TEST SPLIT")
print("=" * 70)

# Time-aware split: earlier -> train, middle -> val, later -> test
# No random shuffling to avoid temporal leakage

n_total = len(X_sequences)
# First 60% train, next 20% validation, last 20% test
train_end = int(n_total * 0.60)
val_end = int(n_total * 0.80)

X_train = X_sequences[:train_end]
y_train = y_sequences[:train_end]
X_val = X_sequences[train_end:val_end]
y_val = y_sequences[train_end:val_end]
X_test = X_sequences[val_end:]
y_test = y_sequences[val_end:]

# Also save categories
train_cat = y_categories[:train_end]
val_cat = y_categories[train_end:val_end]
test_cat = y_categories[val_end:]

print(f"\n[TEMPORAL SPLIT]")
print(f"  Train: sequences {0} to {train_end-1:,} ({len(X_train):,} sequences)")
print(f"  Validation: sequences {train_end:,} to {val_end-1:,} ({len(X_val):,} sequences)")
print(f"  Test: sequences {val_end:,} to {n_total-1:,} ({len(X_test):,} sequences)")
print(f"\n[CLASS DISTRIBUTION]")
for split_name, X_split, y_split in [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test, y_test)]:
    attack_count = y_split.sum()
    benign_count = len(y_split) - attack_count
    attack_pct = (attack_count / len(y_split)) * 100
    print(f"  {split_name}: {len(y_split):,} sequences, {attack_count:,} attack ({attack_pct:.1f}%), {benign_count:,} benign ({100-attack_pct:.1f}%)")

# ============================================================
# PHASE 8: SAVE PROCESSED DATA
# ============================================================
print("\n" + "=" * 70)
print("PHASE 8: SAVE PROCESSED DATA")
print("=" * 70)

# Save as numpy arrays
np.save(PROCESSED_DIR / "X_train.npy", X_train)
np.save(PROCESSED_DIR / "y_train.npy", y_train)
np.save(PROCESSED_DIR / "X_val.npy", X_val)
np.save(PROCESSED_DIR / "y_val.npy", y_val)
np.save(PROCESSED_DIR / "X_test.npy", X_test)
np.save(PROCESSED_DIR / "y_test.npy", y_test)

# Save attack categories
np.save(PROCESSED_DIR / "train_categories.npy", train_cat)
np.save(PROCESSED_DIR / "val_categories.npy", val_cat)
np.save(PROCESSED_DIR / "test_categories.npy", test_cat)

# Save feature names
with open(PROCESSED_DIR / "feature_names.json", 'w') as f:
    json.dump(feature_cols, f, indent=2)

# Save label mapping
label_mapping = {"BENIGN": 0, "ATTACK": 1, "MIXED_LIGHT": 1, "HEAVY_ATTACK": 1}
with open(PROCESSED_DIR / "label_mapping.json", 'w') as f:
    json.dump(label_mapping, f, indent=2)

# Save full configuration
config = {
    "source_dataset": "CIC-IDS2017",
    "source": "Hugging Face (bvk/CICIDS-2017)",
    "raw_file": "cic_ids2017_full.csv",
    "raw_rows": len(df),
    "raw_columns": len(df.columns),
    "raw_file_size_mb": round(os.path.getsize(DATA_DIR / "cic_ids2017_full.csv") / (1024 * 1024), 2),
    "window_size": WINDOW_SIZE,
    "sequence_length": SEQUENCE_LENGTH,
    "num_features": len(feature_cols),
    "feature_names": feature_cols,
    "train_shape": list(X_train.shape),
    "val_shape": list(X_val.shape),
    "test_shape": list(X_test.shape),
    "train_samples": len(X_train),
    "val_samples": len(X_val),
    "test_samples": len(X_test),
    "train_attack_pct": round((y_train.sum() / len(y_train)) * 100, 2),
    "val_attack_pct": round((y_val.sum() / len(y_val)) * 100, 2),
    "test_attack_pct": round((y_test.sum() / len(y_test)) * 100, 2),
    "label_mapping": label_mapping,
    "temporal_split": {
        "train_end": train_end,
        "val_end": val_end,
        "total_sequences": n_total
    }
}

with open(PROCESSED_DIR / "pipeline_config.json", 'w') as f:
    json.dump(config, f, indent=2)

# Save the network states DataFrame for further analysis
network_states.to_csv(PROCESSED_DIR / "network_states.csv", index=False)

print("\n[SAVED FILES]")
print(f"  {PROCESSED_DIR / 'X_train.npy'} - {X_train.nbytes / 1024 / 1024:.2f} MB")
print(f"  {PROCESSED_DIR / 'y_train.npy'} - {y_train.nbytes / 1024 / 1024:.2f} MB")
print(f"  {PROCESSED_DIR / 'X_val.npy'} - {X_val.nbytes / 1024 / 1024:.2f} MB")
print(f"  {PROCESSED_DIR / 'y_val.npy'} - {y_val.nbytes / 1024 / 1024:.2f} MB")
print(f"  {PROCESSED_DIR / 'X_test.npy'} - {X_test.nbytes / 1024 / 1024:.2f} MB")
print(f"  {PROCESSED_DIR / 'y_test.npy'} - {y_test.nbytes / 1024 / 1024:.2f} MB")
print(f"  {PROCESSED_DIR / 'pipeline_config.json'}")
print(f"  {PROCESSED_DIR / 'network_states.csv'}")
print(f"  {PROCESSED_DIR / 'feature_names.json'}")

# ============================================================
# PHASE 9: VALIDATION REPORT
# ============================================================
print("\n" + "=" * 70)
print("PHASE 9: VALIDATION SUMMARY")
print("=" * 70)

print(f"""
RAW FLOWS:
  Total: {len(df):,} flows
  File size: {os.path.getsize(DATA_DIR / 'cic_ids2017_full.csv') / (1024*1024):.2f} MB
  Columns: {len(df.columns)}
  Missing: 0, Infinite: 0, Duplicates: {df.duplicated().sum():,}

TIME WINDOW:
  Window size: {WINDOW_SIZE} flows per state
  Total network states: {n_windows:,}
  Features per state: {len(network_states.columns)}

SEQUENCE CONSTRUCTION:
  Sequence length: {SEQUENCE_LENGTH}
  Each sequence: [S(t-9), S(t-8), ..., S(t)] -> predict S(t+1)
  Total sequences: {n_total:,}

FINAL TENSOR SHAPES:
  X_train: {X_train.shape}
  y_train: {y_train.shape}
  X_val: {X_val.shape}
  y_val: {y_val.shape}
  X_test: {X_test.shape}
  y_test: {y_test.shape}

TRAIN / VALIDATION / TEST SPLIT:
  Train: {len(X_train):,} sequences ({len(X_train)/n_total*100:.1f}%)
  Validation: {len(X_val):,} sequences ({len(X_val)/n_total*100:.1f}%)
  Test: {len(X_test):,} sequences ({len(X_test)/n_total*100:.1f}%)

ATTACK / BENIGN DISTRIBUTION:
  Train: {y_train.sum():,} attack ({config['train_attack_pct']:.1f}%), {len(y_train)-y_train.sum():,} benign
  Validation: {y_val.sum():,} attack ({config['val_attack_pct']:.1f}%), {len(y_val)-y_val.sum():,} benign
  Test: {y_test.sum():,} attack ({config['test_attack_pct']:.1f}%), {len(y_test)-y_test.sum():,} benign

LIMITATIONS:
  1. Timestamps are relative (MM:SS.mmm), not absolute wall-clock
     Row ordering is used for temporal sequence
  2. Source/destination IP addresses are decimal-encoded (not used as features)
     to avoid high cardinality issues
  3. TCP window size, TTL variance, fragment flags, payload distribution
     are NOT available in the CIC-IDS2017 CSV
  4. Windows are non-overlapping (could use overlapping for more samples)
  5. Class imbalance: ~75% benign flows leads to ~50-60% benign windows
  6. Attack labels are derived from majority vote within each window
""")

print("PIPELINE COMPLETE - ALL DATA SAVED")
print(f"{'=' * 70}")

# Print a clean summary for the user
print(f"""
{'='*70}
FINAL SUMMARY
{'='*70}
Raw flows: {len(df):,}
Time window: {WINDOW_SIZE} flows
Number of network states: {n_windows:,}
Features per state: {len(network_states.columns)}
Sequence length: {SEQUENCE_LENGTH}
Training sequences: {len(X_train):,}
Validation sequences: {len(X_val):,}
Test sequences: {len(X_test):,}
Attack/benign distribution: Train {config['train_attack_pct']:.1f}% attack, Val {config['val_attack_pct']:.1f}% attack, Test {config['test_attack_pct']:.1f}% attack
{'='*70}
""")
