# -*- coding: utf-8 -*-

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def attribute_hosts(states_csv, flagged_window_indices, feature_names, top_k=5):
    df = pd.read_csv(states_csv)

    if 'src_ip' not in df.columns:
        return {"error": "No source IP column in states file"}

    benign_df = df[df['is_attack'] == 0]
    if len(benign_df) == 0:
        return {"error": "No benign windows to compute median"}

    benign_median = benign_df[feature_names].median()

    results = {}
    for widx in flagged_window_indices:
        if widx >= len(df):
            continue
        window = df.iloc[widx]
        deviations = (window[feature_names] - benign_median).abs()
        top_features = deviations.nlargest(top_k).index.tolist()

        src_ip = window.get('src_ip', 'unknown')
        score = float(deviations[top_features].sum())

        results[widx] = {
            'src_ip': src_ip,
            'deviation_score': score,
            'top_features': top_features,
        }

    return results


def main():
    states_csv = PROCESSED_DIR / "network_states.csv"
    if not states_csv.exists():
        print(f"Error: {states_csv} not found")
        sys.exit(1)

    with open(PROCESSED_DIR / "feature_names.json") as f:
        feature_names = json.load(f)

    flagged = [100, 500, 1000, 1500, 2000]
    results = attribute_hosts(states_csv, flagged, feature_names)

    print("Per-host attribution results:")
    print(json.dumps(results, indent=2))

    with open(BASE_DIR / "models" / "per_host_attribution.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
