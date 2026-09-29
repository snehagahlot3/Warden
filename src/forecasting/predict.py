# -*- coding: utf-8 -*-
"""
WARDEN-T inference entry point.

    python -m src.forecasting.predict
    python -m src.forecasting.predict --index 0 --horizon 5
    python -m src.forecasting.predict --file my_sequence.csv

Runs a real K-step RECURSIVE forecast on one processed sequence and prints the
current observation, the per-horizon forecast, and the evidence (top
contributing features).

RECURSIVE FORECASTING NOTICE
============================
Horizons t+2..t+K are produced by feeding the model's own prediction back into
the model. The model does not observe future traffic. Read the printed horizons
as extrapolation under the learned dynamics, not as measurements. t+1 is the
only horizon grounded in real observed data.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from ..explainability.attention import explain_sample
from ..features.attack_mapping import describe_mapping
from ..models.dataset import MODELS_DIR, load_raw_splits
from .rollout import recursive_rollout

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"


def load_sequence_from_file(path, sequence_length, n_features):
    """Load a [10, 81] sequence from a .npy file or a headerless CSV."""
    path = Path(path)
    if path.suffix == ".npy":
        seq = np.load(path)
    else:
        seq = np.loadtxt(path, delimiter=",", skiprows=1)
    seq = np.asarray(seq, dtype=np.float64)
    if seq.shape != (sequence_length, n_features):
        raise ValueError(
            f"expected sequence shape {(sequence_length, n_features)}, got {seq.shape}"
        )
    return seq


def main():
    parser = argparse.ArgumentParser(
        description="WARDEN-T recursive multi-step attack forecast"
    )
    parser.add_argument(
        "--index", type=int, default=0, help="index into the test split (default 0)"
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--horizon", type=int, default=None, help="K, default from config")
    parser.add_argument(
        "--file", default=None, help="path to a [10,81] .npy/.csv sequence"
    )
    parser.add_argument(
        "--no-evidence", action="store_true", help="skip the perturbation analysis"
    )
    parser.add_argument(
        "--experiment",
        default="original",
        choices=["original", "attack_day"],
        help="which trained run to load (default: original)",
    )
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = {}
        for doc in yaml.safe_load_all(f):
            if isinstance(doc, dict):
                config.update(doc)
    horizon = args.horizon or config["forecasting"]["horizon"]
    n_features = config["model"]["input_dim"]
    sequence_length = config["model"]["sequence_length"]

    from ..explainability.attention import explain_sample, load_artifacts

    artifacts = load_artifacts(experiment=args.experiment)
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    feature_names = artifacts["feature_names"]
    index_to_category = artifacts["index_to_category"]
    device = artifacts["device"]

    if args.experiment == "attack_day":
        # Sequences come from the attack-day-aware split, not the original one.
        import pandas as pd

        from ..models.attack_day_split import (
            CANDIDATE_A,
            build_sequence_indices,
            load_states_and_targets,
            make_split_arrays,
            state_blocks,
            verify_segments,
        )
        from ..models.dataset import PROCESSED_DIR

        verify_segments(verbose=False)
        n_states = len(
            pd.read_csv(PROCESSED_DIR / "network_states.csv", usecols=["is_attack"])
        )
        blocks = state_blocks(n_states)
        seq_idx = build_sequence_indices(blocks, CANDIDATE_A)
        X_all, y_all, cat_all = load_states_and_targets(feature_names)
        X_split, y_split, cat_split, _ = make_split_arrays(
            seq_idx[args.split], X_all, y_all, cat_all, feature_names
        )
    else:
        raw = load_raw_splits()
        X_split = raw[f"X_{args.split}"]
        cat_split = raw[f"cat_{args.split}"]

    if args.file:
        sequence_raw = load_sequence_from_file(args.file, sequence_length, n_features)
        source = f"file: {args.file}"
    else:
        sequence_raw = X_split[args.index]
        source = f"{args.split} split, index {args.index}"

    # If the sequence came from a prepared split, its true future is known and
    # is reported for reference. It is NOT used by the rollout.
    true_future = None
    if (
        args.file is None
        and args.split in ("validation", "test")
        and args.index < len(X_split) - sequence_length
    ):
        true_future = cat_split[args.index + sequence_length]

    sequence_norm = scaler.transform(sequence_raw)

    # ------------------------------------------------------------- header
    print("=" * 70)
    print("WARDEN-T - MULTI-STEP RECURSIVE ATTACK FORECAST")
    print("=" * 70)
    print(f"  source              : {source}")
    print(f"  experiment          : {args.experiment}")
    print(f"  model checkpoint    : best epoch {artifacts['best_epoch']}")
    print(f"  device              : {device}")

    print("\nCurrent observation")
    print("-" * 70)
    print(f"  Input sequence: {sequence_length} windows")
    print(f"  Features per window : {n_features}")
    print(f"  Covers states       : S(t-{sequence_length - 1}) .. S(t)")

    # ------------------------------------------------------------ forecast
    result = recursive_rollout(
        model,
        sequence_norm,
        horizon=horizon,
        feature_mean=scaler.mean,
        feature_std=scaler.std,
    )

    print("\nForecast (RECURSIVE - model never observes future traffic)")
    print("-" * 70)
    print(
        f"  {'Horizon':<9}{'AttackProb':>13}{'Predicted category':>32}"
        f"{'Confidence':>12}"
    )
    for step in result["steps"]:
        cat_name = index_to_category[step["category_index"]]
        print(
            f"  t+{step['horizon']:<7}"
            f"{step['attack_probability']:>13.4f}"
            f"{cat_name:>32}"
            f"{step['category_confidence']:>12.4f}"
        )

    print("\n  ATT&CK interpretation layer (optional, separate from the prediction)")
    print("-" * 70)
    for step in result["steps"]:
        cat_name = index_to_category[step["category_index"]]
        print(f"  t+{step['horizon']}: {describe_mapping(cat_name)}")

    if true_future is not None:
        print(f"\n  Reference only (real observed S(t+{sequence_length}), not used in rollout):")
        print(f"    true category at t+{sequence_length} : {true_future}")

    print("\n  Predicted network state S(t+1), top features by z-score deviation")
    print("-" * 70)
    s_next = result["steps"][0]["next_state_raw"]
    order = np.argsort(-np.abs(s_next))[:5]
    for rank, fi in enumerate(order, 1):
        print(f"    {rank}. {feature_names[fi]:<38s} {s_next[fi]:>14.4f}")

    # ------------------------------------------------------------ evidence
    if not args.no_evidence:
        print("\nEvidence")
        print("-" * 70)
        print(f"  Method: {args.split}[{args.index}] - model feature importance /")
        print("  perturbation analysis (NOT causal, NOT SHAP)")
        print()
        importance = explain_sample(
            index=args.index,
            split=args.split,
            top_k=5,
            verbose=False,
            experiment=args.experiment,
        )[3]
        print(f"  Attack probability for this sample: "
              f"{importance['base_attack_probability']:.4f}")
        print("  Top contributing features:")
        for rank, rec in enumerate(importance["importances"][:5], 1):
            print(
                f"  {rank}. {rec['feature']:<38s} "
                f"importance={rec['importance_score']:+.4f}"
            )

    print("\n" + "=" * 70)
    print("NOTE: t+2 .. t+K are recursive extrapolations. Error compounds with")
    print("      each step because each predicted state becomes the next input.")
    print("      Only t+1 is grounded in observed data.")
    print("=" * 70)


if __name__ == "__main__":
    sys.exit(main())
