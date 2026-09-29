# -*- coding: utf-8 -*-
"""
WARDEN-T CSV inference pipeline.

Takes a flow-level CIC-IDS2017-style CSV and produces a multi-step attack
forecast using the existing trained model and fitted scaler.

Pipeline:
  CSV -> validate columns -> create_network_states() -> build_sequences()
  -> scaler.transform() -> recursive_rollout() -> forecast + evidence

All preprocessing is identical to training. No new scaler is fitted.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.state_construction import (
    create_network_states,
    validate_csv_columns,
    build_sequences,
    WINDOW_SIZE,
    SEQUENCE_LENGTH,
)
from ..explainability.attention import (
    load_artifacts,
    extract_attention,
    perturbation_importance,
)
from .rollout import recursive_rollout

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def process_csv(csv_file, experiment="original", horizon=5):
    """
    Process an uploaded flow-level CSV and return forecast results.

    Parameters
    ----------
    csv_file : file-like object (Streamlit UploadedFile or path)
    experiment : "original" or "attack_day"
    horizon : int, number of recursive forecast steps

    Returns
    -------
    dict with keys:
      success, error, filename, n_flows, n_states, n_sequences,
      missing_columns, forecast, attention, importance,
      sequence_raw, sequence_norm, states_df, feature_names
    """
    result = {
        'success': False,
        'error': None,
        'filename': None,
        'n_flows': 0,
        'n_states': 0,
        'n_sequences': 0,
        'missing_columns': [],
        'forecast': None,
        'attention': None,
        'importance': None,
        'sequence_raw': None,
        'sequence_norm': None,
        'states_df': None,
        'feature_names': None,
    }

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        result['error'] = f"Failed to read CSV: {e}"
        return result

    result['filename'] = getattr(csv_file, 'name', 'uploaded.csv')
    result['n_flows'] = len(df)

    missing = validate_csv_columns(df)
    result['missing_columns'] = missing
    if missing:
        result['error'] = f"Missing required columns: {', '.join(missing)}"
        return result

    min_flows = WINDOW_SIZE * SEQUENCE_LENGTH
    if len(df) < min_flows:
        result['error'] = (
            f"Insufficient flows: {len(df):,} flows provided, "
            f"but the model requires at least {min_flows:,} flows "
            f"({SEQUENCE_LENGTH} temporal states x {WINDOW_SIZE} flows each) "
            f"for one forecast sequence."
        )
        return result

    states_df = create_network_states(df)
    result['n_states'] = len(states_df)
    result['states_df'] = states_df

    with open(PROCESSED_DIR / "feature_names.json") as f:
        feature_names = json.load(f)
    result['feature_names'] = feature_names

    states_df = states_df[feature_names]

    sequences = build_sequences(states_df, feature_names)
    result['n_sequences'] = len(sequences)

    if len(sequences) == 0:
        result['error'] = "No sequences could be constructed from the uploaded data."
        return result

    sequence_raw = sequences[-1]
    result['sequence_raw'] = sequence_raw

    artifacts = load_artifacts(experiment=experiment)
    scaler = artifacts['scaler']

    sequence_norm = scaler.transform(sequence_raw)
    result['sequence_norm'] = sequence_norm

    forecast = recursive_rollout(
        artifacts['model'],
        sequence_norm,
        horizon=horizon,
        feature_mean=scaler.mean,
        feature_std=scaler.std,
    )
    result['forecast'] = forecast

    attn = extract_attention(artifacts['model'], sequence_norm, device=artifacts['device'])
    result['attention'] = attn

    importance = perturbation_importance(
        artifacts['model'],
        sequence_norm,
        artifacts['feature_names'],
        scaler=scaler,
        device=artifacts['device'],
    )
    result['importance'] = importance

    result['success'] = True
    return result
