# -*- coding: utf-8 -*-
"""
Reusable network state construction for WARDEN-T.

Extracted verbatim from src/data/phase3_9_pipeline.py (Phase 4) so that both
the training pipeline and the CSV inference path use the identical aggregation
logic. DO NOT modify this without updating both consumers.
"""

import numpy as np
import pandas as pd

WINDOW_SIZE = 500
SEQUENCE_LENGTH = 10

SELECTED_FEATURES = {
    'flow_duration': {'column': 'Flow Duration', 'dtype': 'numerical'},
    'total_fwd_packets': {'column': 'Total Fwd Packet', 'dtype': 'numerical'},
    'total_bwd_packets': {'column': 'Total Bwd packets', 'dtype': 'numerical'},
    'total_fwd_bytes': {'column': 'Total Length of Fwd Packet', 'dtype': 'numerical'},
    'total_bwd_bytes': {'column': 'Total Length of Bwd Packet', 'dtype': 'numerical'},
    'flow_iat_mean': {'column': 'Flow IAT Mean', 'dtype': 'numerical'},
    'flow_iat_std': {'column': 'Flow IAT Std', 'dtype': 'numerical'},
    'fwd_iat_mean': {'column': 'Fwd IAT Mean', 'dtype': 'numerical'},
    'bwd_iat_mean': {'column': 'Bwd IAT Mean', 'dtype': 'numerical'},
    'down_up_ratio': {'column': 'Down/Up Ratio', 'dtype': 'numerical'},
    'fwd_packets_per_sec': {'column': 'Fwd Packets/s', 'dtype': 'numerical'},
    'bwd_packets_per_sec': {'column': 'Bwd Packets/s', 'dtype': 'numerical'},
    'syn_flag_count': {'column': 'SYN Flag Count', 'dtype': 'numerical'},
    'rst_flag_count': {'column': 'RST Flag Count', 'dtype': 'numerical'},
    'fin_flag_count': {'column': 'FIN Flag Count', 'dtype': 'numerical'},
    'psh_flag_count': {'column': 'PSH Flag Count', 'dtype': 'numerical'},
    'packet_length_mean': {'column': 'Packet Length Mean', 'dtype': 'numerical'},
    'packet_length_std': {'column': 'Packet Length Std', 'dtype': 'numerical'},
    'avg_packet_size': {'column': 'Average Packet Size', 'dtype': 'numerical'},
    'protocol': {'column': 'Protocol', 'dtype': 'categorical'},
}

REQUIRED_COLUMNS = [
    'Flow Duration', 'Total Fwd Packet', 'Total Bwd packets',
    'Total Length of Fwd Packet', 'Total Length of Bwd Packet',
    'Flow IAT Mean', 'Flow IAT Std', 'Fwd IAT Mean', 'Bwd IAT Mean',
    'Down/Up Ratio', 'Fwd Packets/s', 'Bwd Packets/s',
    'SYN Flag Count', 'RST Flag Count', 'FIN Flag Count', 'PSH Flag Count',
    'Packet Length Mean', 'Packet Length Std', 'Average Packet Size',
    'Protocol', 'Label',
    'FWD Init Win Bytes', 'Bwd Init Win Bytes',
    'Dst Port', 'Src Port',
]


def parse_timestamp(ts_str):
    """Parse MM:SS.mmm format to seconds."""
    try:
        parts = str(ts_str).split(':')
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    except (ValueError, IndexError):
        return 0.0


def create_network_states(df, window_size=WINDOW_SIZE, selected_features=SELECTED_FEATURES, time_based=False, window_seconds=30.0):
    """Aggregate flows into network states."""
    states = []
    cat_col = 'Protocol'

    if time_based:
        timestamps = df['Timestamp'].apply(parse_timestamp).values
        global_ts = np.zeros(len(timestamps))
        offset = 0.0
        prev_ts = 0.0
        for i in range(len(timestamps)):
            if timestamps[i] < prev_ts:
                offset += prev_ts
            global_ts[i] = timestamps[i] + offset
            prev_ts = timestamps[i]

        n_windows = int(global_ts[-1] / window_seconds) + 1
        for w in range(n_windows):
            start_ts = w * window_seconds
            end_ts = start_ts + window_seconds
            mask = (global_ts >= start_ts) & (global_ts < end_ts)
            window = df[mask]
            if len(window) < 10:
                continue
            state = _compute_state(window, selected_features)
            states.append(state)
    else:
        for i in range(0, len(df) - window_size + 1, window_size):
            window = df.iloc[i:i + window_size]
            state = _compute_state(window, selected_features)
            states.append(state)

    return pd.DataFrame(states)


def _compute_state(window, selected_features):
    """Compute a single network state from a window of flows."""
    state = {}

    for feat_name, feat_info in selected_features.items():
        col = feat_info['column']
        if feat_info['dtype'] == 'numerical':
            state[f'{feat_name}_mean'] = window[col].mean()
            state[f'{feat_name}_std'] = window[col].std()
            state[f'{feat_name}_max'] = window[col].max()
            state[f'{feat_name}_min'] = window[col].min()
        elif feat_info['dtype'] == 'categorical':
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

    return state


def validate_csv_columns(df):
    """Check that the uploaded CSV has all required columns. Returns list of missing columns."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    return missing


def build_sequences(states_df, feature_names, sequence_length=SEQUENCE_LENGTH):
    """Build [10, 81] sequences from network states (identical to training Phase 6)."""
    X_data = states_df[feature_names].values
    sequences = []
    for i in range(len(X_data) - sequence_length):
        sequences.append(X_data[i:i + sequence_length])
    return np.array(sequences)
