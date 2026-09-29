# -*- coding: utf-8 -*-

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .state_construction import create_network_states

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def main():
    parser = argparse.ArgumentParser(description="Time-based window construction")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--window-seconds", type=float, default=30.0, help="Window size in seconds")
    parser.add_argument("--output-dir", default=str(PROCESSED_DIR), help="Output directory")
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")

    print(f"\nConstructing time-based windows ({args.window_seconds}s)...")
    states_df = create_network_states(df, time_based=True, window_seconds=args.window_seconds)
    print(f"  Network states: {len(states_df):,}")
    print(f"  Features per state: {len(states_df.columns)}")

    feature_cols = [c for c in states_df.columns if c not in ['label', 'attack_category', 'is_attack']]
    print(f"  Feature columns: {len(feature_cols)}")

    X_data = states_df[feature_cols].values
    sequences = []
    for i in range(len(X_data) - 10):
        sequences.append(X_data[i:i + 10])
    sequences = np.array(sequences)
    print(f"  Sequences: {sequences.shape}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "X_time_based.npy", sequences)
    states_df.to_csv(output_dir / "network_states_time_based.csv", index=False)

    with open(output_dir / "time_based_config.json", "w") as f:
        json.dump({
            "window_seconds": args.window_seconds,
            "n_states": len(states_df),
            "n_sequences": len(sequences),
            "n_features": len(feature_cols),
            "feature_names": feature_cols,
        }, f, indent=2)

    print(f"\nSaved to {output_dir}")
    print(f"  X_time_based.npy: {sequences.shape}")
    print(f"  network_states_time_based.csv")
    print(f"  time_based_config.json")


if __name__ == "__main__":
    main()
