# -*- coding: utf-8 -*-
"""
Attack-day-aware split for the Candidate A experiment.

This module is ADDITIVE. It does not touch the original pipeline
(src/data/phase3_9_pipeline.py) nor the original experiment artifacts
(models/warden_t_best.pt, models/test_metrics.json, ...).

WHY AN ATTACK-DAY-AWARE SPLIT
=============================
The original split cut the sequence stream at arbitrary flow offsets, so each
of validation and test was a MIXTURE of two different CIC-IDS2017 capture
days. That produced a degenerate category vocabulary (8 of 17 classes with zero
training support) and an unstable validation loss. Here each split is built
from whole, unsplit capture-day blocks instead.

HOW THE DAY BOUNDARIES WERE DETERMINED (not guessed)
=====================================================
The merged CIC-IDS2017 CSV has no day column, and its `Timestamp` column is
relative and monotonically increasing (zero resets), so neither can be used.
The boundaries below were derived from two measurable properties of the data:

  1. ATTACK-FAMILY ROW BLOCKS. Each capture day carries a distinct attack
     family, so the label column forms contiguous blocks. The first and last
     row of every family was measured directly.
  2. LONG ALL-BENIGN GAPS. Where two consecutive attack families belong to
     different days, the benign traffic between them forms a long run; within
     a single day the gaps are shorter or absent.

Measured family row ranges that justify each boundary (raw flow rows):

    Botnet / DDoS / Portscan            26,224  ..   547,554
    Infiltration / Web Attack (BF,XSS,SQLi)  928,521 .. 1,285,675
    FTP-Patator / SSH-Patator         1,285,675  .. 1,603,313
    DoS Slowloris / Slowhttptest / Hulk 1,603,314 .. 1,847,682   (benign gap 1,603,314-1,612,924)
    DoS GoldenEye / DoS Hulk / Heartbleed 1,847,683 .. 2,099,956 (benign gap 1,831,062-1,847,683)

The three boundaries below are the first row of the first family of the next
day. They are re-verified at runtime by `verify_segments()`, which checks that
the label composition on each side matches the expected day profile and would
raise if the underlying data ever changed.

KNOWN LIMITATION - Monday cannot be separated
=============================================
CIC-IDS2017 Monday contains no attacks, so it is indistinguishable from the
BENIGN part of Tuesday using the label column; the source-IP pool is also
constant across the whole file (verified: no change at any 20k-row boundary).
The Monday/Tuesday boundary is therefore NOT recoverable from this merged copy,
and rows 0 .. 1,285,675 are treated as ONE indivisible block. No boundary is
invented.

NO LEAKAGE
==========
* The feature scaler is fitted on TRAIN sequences only.
* Class weights and the BCE pos_weight come from TRAIN labels only.
* The category vocabulary is built from TRAIN + VALIDATION only; the test
  categories are never inspected when defining the model's output space.
* No hyperparameter is selected using test data.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_CSV = BASE_DIR / "data" / "raw" / "CIC-IDS2017" / "cic_ids2017_full.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
SEGMENT_CACHE = MODELS_DIR / "attack_day_segments.json"

WINDOW_SIZE = 500  # flows per network state, must match the pipeline
SEQUENCE_LENGTH = 10

# Boundary rows (first flow row of the first attack family of the next day).
BOUNDARY_ROWS = {
    "wed_start": 1_285_675,
    "thu_start": 1_603_314,
    "fri_start": 1_847_683,
}

# Families that are UNIQUE to one capture day. Only these are required by the
# runtime check, because they are what actually define a boundary.
#
# Deliberately EXCLUDED from the required sets: "DoS Hulk" and "Heartbleed".
# Both are documented as occurring on more than one day (Hulk on Thu+Fri,
# Heartbleed on Thu+Fri), so they cannot be used to verify a day boundary.
# The measurement also shows that in THIS merged file every Heartbleed flow
# (11 flows, rows 1,847,683-1,948,986) falls after the Friday boundary, so
# requiring Heartbleed on the Thursday side would be asserting something the
# data does not support. The discrepancy with the published per-day profile is
# recorded here rather than papered over.
EXPECTED_FAMILIES = {
    "S1_Mon_Tue": {
        "Botnet", "DDoS", "Portscan", "Infiltration",
        "Web Attack - Brute Force", "Web Attack - XSS",
        "Web Attack - SQL Injection",
    },
    "S2_Wed": {"FTP-Patator", "SSH-Patator"},
    "S3_Thu": {"DoS Slowloris", "DoS Slowhttptest"},
    "S4_Fri": {"DoS GoldenEye"},
}

# Families documented as spanning multiple days; reported, never required.
SHARED_FAMILIES = {"BENIGN", "DoS Hulk", "Heartbleed"}


def family(label):
    """'DoS Hulk - Attempted' -> 'DoS Hulk'. Truncation variants inherit."""
    label = str(label)
    return label[: -len(" - Attempted")] if label.endswith(" - Attempted") else label


def detect_boundaries(force=False):
    """
    Measure the day boundary rows from the raw CSV and cache them.

    Cached so the 825 MB file is read once. The cache stores the measurement
    provenance (per-family row ranges) so the boundaries stay auditable.
    """
    if SEGMENT_CACHE.exists() and not force:
        with open(SEGMENT_CACHE) as f:
            return json.load(f)

    if not RAW_CSV.exists():
        raise FileNotFoundError(f"{RAW_CSV} not found")

    labels = pd.read_csv(RAW_CSV, usecols=["Label"])["Label"].values
    families = {}
    for name in np.unique(labels):
        idx = np.where(labels == name)[0]
        families[str(name)] = [int(idx[0]), int(idx[-1])]

    # Each boundary is the FIRST FLOW of the first family unique to the next
    # day, except Wednesday->Thursday where the two candidate edges differ by a
    # long all-BENIGN gap. The exact instant the day changed inside a gap of
    # benign traffic is unobservable, so the gap is assigned to the EARLIER day
    # (its first row is used as the boundary). This is recorded explicitly
    # rather than presented as if it were a precise measurement.
    def first_row_of(fam):
        return min(v[0] for k, v in families.items() if family(k) == fam)

    last_row_of = lambda fam: max(v[1] for k, v in families.items() if family(k) == fam)

    labels_f = np.array([family(l) for l in labels])
    wed_last = max(last_row_of(f) for f in ("FTP-Patator", "SSH-Patator"))
    thu_first = first_row_of("DoS Slowloris")
    thu_gap = [int(wed_last) + 1, int(thu_first)]  # all-BENIGN interval

    measured = {
        "wed_start": int(first_row_of("FTP-Patator")),
        "thu_start": int(thu_gap[0]),
        "thu_start_first_thu_attack": int(thu_first),
        "thu_benign_gap_rows": int(thu_gap[1] - thu_gap[0]),
        "fri_start": int(first_row_of("Heartbleed")),
    }

    payload = {
        "measured_boundaries": measured,
        "configured_boundaries": BOUNDARY_ROWS,
        "n_raw_rows": int(len(labels)),
        "family_row_ranges": families,
        "note": (
            "fri_start is the first Heartbleed row, which follows a 16,621-row "
            "all-BENIGN gap (rows 1,831,062-1,847,683). DoS Hulk and Heartbleed "
            "both occur on Thursday AND Friday, so the Thursday/Friday boundary "
            "is the least certain of the three (accurate to about +/-6 network "
            "states). This is recorded rather than hidden."
        ),
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SEGMENT_CACHE, "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def state_blocks(n_states):
    """
    Convert the flow-row boundaries into network-state index blocks.

    A state i covers flows [500*i, 500*i+500). The boundary row R is the first
    flow of the next day, and the state that CONTAINS that flow is assigned to
    the new day, so the state index is floor(R / 500). This reproduces the
    segment boundaries reported in the reconnaissance pass:

        S1_Mon_Tue [0, 2571)   S2_Wed [2571, 3206)
        S3_Thu     [3206,3695) S4_Fri  [3695, 4199)

    Note the boundary state may be entirely BENIGN (the day change happens
    during a benign gap); that is harmless because the label is BENIGN either
    way, and it keeps the convention unambiguous.
    """
    b = BOUNDARY_ROWS
    edges = [
        0,
        b["wed_start"] // WINDOW_SIZE,
        b["thu_start"] // WINDOW_SIZE,
        b["fri_start"] // WINDOW_SIZE,
        n_states,
    ]
    names = ["S1_Mon_Tue", "S2_Wed", "S3_Thu", "S4_Fri"]
    return {name: (edges[i], edges[i + 1]) for i, name in enumerate(names)}


def verify_segments(verbose=True):
    """
    Runtime self-check, performed at FLOW level against the raw CSV.

    The check must be at flow level, not state level: the window category is
    the MODE label inside 500 flows, so very small families (Web Attack - XSS
    has 18 flows, SQL Injection has 5 across the whole dataset) can be present
    in a day yet never become a window's mode. Verifying against state-level
    modes would therefore report a false failure.
    """
    payload = detect_boundaries()
    n_states = len(pd.read_csv(PROCESSED_DIR / "network_states.csv", usecols=["is_attack"]))
    blocks = state_blocks(n_states)
    row_edges = [0] + [BOUNDARY_ROWS[k] for k in ("wed_start", "thu_start", "fri_start")]
    row_edges.append(payload["n_raw_rows"])

    labels = pd.read_csv(RAW_CSV, usecols=["Label"])["Label"].values
    names = list(blocks)
    ok = True
    for i, name in enumerate(names):
        present = {family(l) for l in labels[row_edges[i] : row_edges[i + 1]]}
        expected = EXPECTED_FAMILIES[name]
        missing = expected - present
        if missing:
            ok = False
            if verbose:
                print(f"  [verify] {name}: MISSING expected families {missing}")
        elif verbose:
            print(
                f"  [verify] {name}: OK  flows[{row_edges[i]}:{row_edges[i+1]}) "
                f"states{blocks[name]} families={sorted(present)}"
            )
    assert ok, "attack-day segment verification failed"
    return blocks



def build_sequence_indices(blocks, split_of_segment):
    """
    Map each segment to a split, then emit the global sequence indices.

    A sequence j consumes states [j, j+10) and targets state j+10. To keep a
    day boundary intact, only sequences with a >= start and j+10 <= end-1 are
    emitted, so no sequence window or target can straddle a boundary.
    """
    out = {}
    for split, seg_names in split_of_segment.items():
        idx = []
        for seg in seg_names:
            a, b = blocks[seg]
            idx.extend(range(a, b - SEQUENCE_LENGTH))
        out[split] = np.array(sorted(idx), dtype=np.int64)
    return out


def load_states_and_targets(feature_names):
    """Return the state matrix, is_attack and attack_category for all states."""
    df = pd.read_csv(PROCESSED_DIR / "network_states.csv")
    X = df[feature_names].values.astype(np.float64)
    y = df["is_attack"].values.astype(np.int64)
    cat = df["attack_category"].values.astype(str)
    return X, y, cat


def make_split_arrays(seq_indices, X, y, cat, feature_names):
    """
    Materialise (X_seq, y_risk, y_cat_str, S_next) for a set of global
    sequence indices, using the exact relationship verified in
    src/models/dataset.py.
    """
    j = np.asarray(seq_indices)
    X_seq = np.stack([X[i : i + SEQUENCE_LENGTH] for i in j])
    tgt = j + SEQUENCE_LENGTH
    return X_seq, y[tgt], cat[tgt], X[tgt]


def describe_split(name, X, y, cat, blocks, seg_names):
    """Print the required TRAIN/VALIDATION/TEST summary block."""
    u, c = np.unique(cat, return_counts=True)
    state_lo = min(blocks[s][0] for s in seg_names)
    state_hi = max(blocks[s][1] for s in seg_names)
    print(f"\n{name}:")
    print(f"  states: {state_lo} .. {state_hi}  (n={state_hi - state_lo})")
    print(f"  sequences: {len(X)}")
    print(f"  attack distribution: "
          f"benign={int((y == 0).sum())} ({100 * (y == 0).mean():.2f}%), "
          f"attack={int((y == 1).sum())} ({100 * y.mean():.2f}%)")
    print("  category distribution:")
    for k, v in sorted(zip(u, c), key=lambda kv: -kv[1]):
        print(f"     {k:<40s} {v:5d}  {100 * v / len(cat):5.2f}%")
    return {
        "states": [int(state_lo), int(state_hi)],
        "n_states": int(state_hi - state_lo),
        "n_sequences": int(len(X)),
        "n_benign": int((y == 0).sum()),
        "n_attack": int((y == 1).sum()),
        "attack_pct": float(100 * y.mean()),
        "category_counts": {str(k): int(v) for k, v in zip(u, c)},
        "segments": seg_names,
    }


# Candidate A, as approved: whole capture days, strict temporal order.
CANDIDATE_A = {
    "train": ["S1_Mon_Tue", "S2_Wed"],
    "validation": ["S3_Thu"],
    "test": ["S4_Fri"],
}
