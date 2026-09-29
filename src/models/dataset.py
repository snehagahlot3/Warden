# -*- coding: utf-8 -*-
"""
WARDEN-T dataset layer.

IMPORTANT - EXACT SEMANTICS OF THE PRE-GENERATED ARRAYS
========================================================
The data pipeline (src/data/phase3_9_pipeline.py) built a matrix of network
states `states` with shape (4199, 81) and then slid a window over it:

    for j in range(len(states) - 10):
        X[j] = states[j : j+10]                    # S(t-9) ... S(t)
        y[j] = is_attack[ j+10 ]                   # attack flag of S(t+1)
        cat[j] = attack_category[ j+10 ]           # dominant label of S(t+10)

This was VERIFIED against data/processed/network_states.csv (see
`verify_alignment()`), and all three checks returned True:
    X  == stack(states[j:j+10])
    y  == is_attack[j+10]
    cat== attack_category[j+10]

Consequences used by the model layer:
  * The binary target `y` is the BENIGN / non-BENIGN flag of the NEXT state.
    NOTE: `label_mapping.json` in data/processed is the *window severity* map
    (BENIGN / MIXED_LIGHT / ATTACK / HEAVY_ATTACK -> 0/1), NOT a category map.
    It is therefore not usable as the attack-category index space.
  * The attack-category targets are plain strings stored in
    `*_categories.npy`. No category index map exists in the pipeline, so one is
    derived HERE strictly from the observed data (see `build_category_vocab`).
    No label is invented.
  * The regression target S(t+1) is the *same 81-dimensional feature space* as
    the input, so it is normalised with the identical train-derived scaler and
    can be fed straight back into the model for recursive forecasting.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

SEQUENCE_LENGTH = 10

# Category used for targets that never appear in the train/validation portion.
# The temporal split crosses the five CIC-IDS2017 capture days, and each day
# carries a different attack family, so some test categories are genuinely
# out-of-vocabulary. They are reported honestly rather than silently dropped.
OOV_CATEGORY = "UNSEEN_OUT_OF_VOCABULARY"


def load_metadata():
    """Load the feature-name list and pipeline config written by the pipeline."""
    with open(PROCESSED_DIR / "feature_names.json") as f:
        feature_names = json.load(f)
    with open(PROCESSED_DIR / "pipeline_config.json") as f:
        pipeline_config = json.load(f)
    return feature_names, pipeline_config


def build_category_vocab(categories_by_split):
    """
    Derive the attack-category index space from observed data only.

    The vocabulary is built from the TRAIN and VALIDATION categories. Test
    categories are deliberately NOT inspected when sizing the model head, so
    the architecture cannot gain information from the test labels. Any test
    category outside this vocabulary is mapped to OOV_CATEGORY and counted.
    """
    observed = set()
    for split in ("train", "val"):
        observed.update(np.unique(categories_by_split[split]).tolist())
    classes = sorted(observed) + [OOV_CATEGORY]
    return {name: i for i, name in enumerate(classes)}


def load_states():
    """Return the full (4199, 81) network-state matrix written by the pipeline."""
    feature_names, _ = load_metadata()
    states_df = pd.read_csv(PROCESSED_DIR / "network_states.csv")
    return states_df[feature_names].values.astype(np.float64), feature_names


def load_raw_splits():
    """Load the untouched pipeline outputs for the three splits."""
    data = {
        "X_train": np.load(PROCESSED_DIR / "X_train.npy"),
        "y_train": np.load(PROCESSED_DIR / "y_train.npy"),
        "X_val": np.load(PROCESSED_DIR / "X_val.npy"),
        "y_val": np.load(PROCESSED_DIR / "y_val.npy"),
        "X_test": np.load(PROCESSED_DIR / "X_test.npy"),
        "y_test": np.load(PROCESSED_DIR / "y_test.npy"),
    }
    for split in ("train", "val", "test"):
        data[f"cat_{split}"] = np.load(
            PROCESSED_DIR / f"{split}_categories.npy", allow_pickle=True
        )
    return data


def verify_alignment(verbose=True):
    """
    Assert the documented relationship between the saved arrays and the
    underlying network-state table. Run before training so a silent upstream
    change can never quietly corrupt the model targets.
    """
    states, _ = load_states()
    raw = load_raw_splits()
    X = np.concatenate([raw["X_train"], raw["X_val"], raw["X_test"]])
    y = np.concatenate([raw["y_train"], raw["y_val"], raw["y_test"]])
    cat = np.concatenate(
        [raw["cat_train"], raw["cat_val"], raw["cat_test"]]
    )
    n = len(X)

    expected_X = np.stack([states[i : i + SEQUENCE_LENGTH] for i in range(n)])
    x_ok = np.allclose(X, expected_X)

    # Re-read is_attack / attack_category directly from the state table.
    states_df = pd.read_csv(PROCESSED_DIR / "network_states.csv")
    is_attack = states_df["is_attack"].values
    attack_category = states_df["attack_category"].values
    y_ok = np.array_equal(y, is_attack[SEQUENCE_LENGTH : SEQUENCE_LENGTH + n])
    cat_ok = np.array_equal(
        cat, attack_category[SEQUENCE_LENGTH : SEQUENCE_LENGTH + n]
    )

    if verbose:
        print("[alignment] X  == stack(states[j:j+10]) :", x_ok)
        print("[alignment] y  == is_attack[j+10]       :", y_ok)
        print("[alignment] cat== attack_category[j+10] :", cat_ok)
    assert x_ok and y_ok and cat_ok, "pipeline array alignment check failed"
    return True


def next_state_targets(split_sizes, states, scaler=None):
    """
    Build the S(t+1) regression targets.

    Global sequence j uses states[j+10] as its next state. The split offsets
    are 0, train, train+val (a pure temporal split), so the target for a local
    index i of a split is simply states[offset + i + 10].

    When `scaler` is given the targets are returned in the same normalised
    space as the model input, which is required for the state loss to be
    commensurate with the other two tasks.
    """
    targets, offsets = [], {}
    offset = 0
    for split, size in zip(("train", "val", "test"), split_sizes):
        offsets[split] = offset
        block = states[offset + SEQUENCE_LENGTH : offset + size + SEQUENCE_LENGTH]
        targets.append(scaler.transform(block) if scaler is not None else block)
        offset += size
    return targets, offsets


class FeatureScaler:
    """
    Robust feature scaler fitted on the training sequences only.

    WHY ROBUST (median / IQR) AND NOT PLAIN z-SCORING
    ------------------------------------------------
    The upstream pipeline aggregates raw CICFlowMeter statistics and emits
    untransformed values; the "log_transform + standard_scale" steps described
    in its feature table were never actually applied. The resulting 81
    features have wildly different and heavy-tailed distributions - the
    per-feature training standard deviation ranges from ~0.02 to ~6e7.

    Plain z-scoring on that data is unstable: measured on the real splits, the
    training period has both near-constant features (fin_flag_count_min, and
    *_min columns that are usually 0) and extremely heavy-tailed ones
    (window_total_bytes). A near-constant training column gets a tiny std, so
    a val value only slightly above the training mass maps to a z-score in the
    hundreds. Measured worst case: 2234 sigma on the validation split, which
    made the state-regression loss diverge to NaN.

    Median/IQR scaling is robust to that: it is driven by the bulk of the
    distribution rather than by extreme tails, so a heavy-tailed column and a
    near-constant column are both rescaled sensibly. `clip` then bounds the
    result so no single cell can dominate the loss.

    Statistics come from the training split only; val/test are never used to
    fit anything. The same scaler is applied to the S(t+1) regression targets
    because they live in the identical feature space, which is what makes
    recursive forecasting well defined.
    """

    def __init__(self, center, scale, clip):
        self.mean = np.asarray(center, dtype=np.float64)   # median
        self.std = np.asarray(scale, dtype=np.float64)     # IQR-based spread
        self.clip = float(clip)

    @classmethod
    def fit(cls, X_train, clip=10.0):
        flat = X_train.reshape(-1, X_train.shape[-1])
        q25 = np.percentile(flat, 25, axis=0)
        q75 = np.percentile(flat, 75, axis=0)
        center = np.median(flat, axis=0)
        # 1.349 * IQR estimates the standard deviation for normal data, but is
        # itself outlier-resistant. Fall back to std where the IQR degenerates.
        scale = (q75 - q25) / 1.349
        std = flat.std(axis=0)
        scale = np.where(scale < 1e-6, std, scale)
        scale[scale < 1e-8] = 1.0
        return cls(center, scale, clip)

    def transform(self, arr):
        z = (arr - self.mean) / self.std
        return np.clip(z, -self.clip, self.clip) if self.clip else z

    def inverse_transform(self, arr):
        return arr * self.std + self.mean

    def save(self, path):
        np.savez(path, mean=self.mean, std=self.std, clip=np.array(self.clip))

    @classmethod
    def load(cls, path):
        data = np.load(path)
        return cls(data["mean"], data["std"], float(data["clip"]))


class WardenDataset(Dataset):
    """
    Yields (X, y_risk, y_category, S_next).

    X        : (10, 81) float32  normalised input sequence
    y_risk   : (1,)     float32   0/1 attack flag of S(t+1)
    y_cat    : ()       int64     category index of S(t+1)
    S_next   : (81,)    float32   normalised S(t+1)
    """

    def __init__(self, X, y, categories, next_state, category_to_index):
        assert len(X) == len(y) == len(categories) == len(next_state)
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        self.next_state = np.ascontiguousarray(next_state, dtype=np.float32)
        self.category_to_index = category_to_index
        self.y_cat = np.asarray(
            [
                category_to_index.get(str(c), category_to_index[OOV_CATEGORY])
                for c in categories
            ],
            dtype=np.int64,
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.X[i]),
            torch.from_numpy(self.y[i]),
            torch.tensor(self.y_cat[i], dtype=torch.long),
            torch.from_numpy(self.next_state[i]),
        )


def describe_targets(verbose=True):
    """
    Print the real shape / dtype / value / distribution of every target array
    so the heads and losses are chosen from observed data, not assumptions.
    """
    raw = load_raw_splits()
    report = {}
    for split in ("train", "val", "test"):
        y = raw[f"y_{split}"]
        cat = raw[f"cat_{split}"]
        values, counts = np.unique(y, return_counts=True)
        report[split] = {
            "n": int(len(y)),
            "y_shape": list(y.shape),
            "y_dtype": str(y.dtype),
            "y_unique": [int(v) for v in values],
            "y_distribution": {int(v): int(c) for v, c in zip(values, counts)},
            "category_shape": list(cat.shape),
            "category_dtype": str(cat.dtype),
            "category_counts": {
                str(k): int(v) for k, v in zip(*np.unique(cat, return_counts=True))
            },
        }
    if verbose:
        print("=" * 70)
        print("TARGET INSPECTION (real values read from disk)")
        print("=" * 70)
        for split, rep in report.items():
            print(f"\n[{split}]  n={rep['n']}")
            print(f"  y  shape={rep['y_shape']} dtype={rep['y_dtype']} uniq={rep['y_unique']}")
            print(f"  y  distribution={rep['y_distribution']}")
            print(f"  cat shape={rep['category_shape']} dtype={rep['category_dtype']}")
            for name, count in sorted(
                rep["category_counts"].items(), key=lambda kv: -kv[1]
            ):
                print(f"     {name:<40s} {count:5d}  {count / rep['n'] * 100:5.2f}%")
    return report
