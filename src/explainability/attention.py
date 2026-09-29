# -*- coding: utf-8 -*-
"""
WARDEN-T explainability layer.

TWO MECHANISMS, BOTH CLEARLY LABELLED
=====================================

1. TEMPORAL ATTENTION
   Real attention probabilities taken directly from the first encoder layer of
   the trained transformer (`need_weights=True`). Shows which of the 10 input
   windows the model attended to. This is a description of the model's internal
   weighting, not proof of causality.

2. MODEL FEATURE IMPORTANCE / PERTURBATION ANALYSIS
   For every one of the 81 input features, that feature is replaced by its
   training mean across ALL 10 timesteps of the sequence, the model is re-run,
   and the change in the predicted attack probability is recorded.

       importance_j = P(attack | original) - P(attack | feature j masked)

   A large positive value means masking that feature lowered the model's
   attack probability, i.e. the model relied on it to reach its answer.

   THIS IS NOT A CAUSAL EXPLANATION. It measures model sensitivity to
   perturbation under this specific masking scheme. It is not a statement about
   what is true in the network, and it is not SHAP. It can miss correlated
   features (masking one of two near-duplicates barely changes the output) and
   it can assign importance to a feature that is only acting as a proxy.

Label used throughout: "Model feature importance / perturbation analysis".
"""

import numpy as np
import torch

from ..models.dataset import MODELS_DIR, FeatureScaler


@torch.no_grad()
def extract_attention(model, sequence, device=None):
    """
    Return real attention weights for a single sequence.

    Parameters
    ----------
    model    : WardenTransformer
    sequence : [T, F] array or [1, T, F] tensor, in normalised space

    Returns
    -------
    dict
      per_layer : list of [T, T] averaged over heads, one array per layer
      layer0_head_averaged : convenience view of the first layer
      input_importance : [T] attention mass each timestep received in layer 0
                         (mean over query positions, i.e. column sums)
    """
    device = device or next(model.parameters()).device
    x = torch.as_tensor(np.asarray(sequence, dtype=np.float32)) if isinstance(
        sequence, np.ndarray
    ) else sequence.float()
    if x.dim() == 2:
        x = x.unsqueeze(0)
    x = x.to(device)

    model.eval()
    out = model(x, need_weights=True)
    per_layer = [a[0].cpu().numpy() for a in out["attention"]]

    # Column sums of layer-0 attention = how much each input timestep
    # contributed to the queries that produced the pooled representation.
    input_importance = per_layer[0].mean(axis=0)

    return {
        "per_layer": per_layer,
        "layer0_head_averaged": per_layer[0],
        "input_importance": input_importance,
        "sequence_length": x.size(1),
    }


@torch.no_grad()
def perturbation_importance(
    model,
    sequence,
    feature_names,
    scaler=None,
    device=None,
    mask_value=0.0,
):
    """
    Model feature importance / perturbation analysis.

    Each input feature is masked (set to `mask_value`, which is 0.0 in
    z-scored space, i.e. the training mean) across the whole temporal sequence,
    and the drop in predicted attack probability is recorded.

    Returns a list of dicts sorted by importance, descending.
    """
    device = device or next(model.parameters()).device
    x = (
        torch.as_tensor(np.asarray(sequence, dtype=np.float32))
        if isinstance(sequence, np.ndarray)
        else sequence.float().clone()
    )
    if x.dim() == 2:
        x = x.unsqueeze(0)
    x = x.to(device)
    model.eval()

    base = model(x)
    base_prob = float(torch.sigmoid(base["risk_logit"]).item())
    base_cat = base["category_logits"].argmax(-1).item()
    base_cat_prob = float(torch.softmax(base["category_logits"], -1).max().item())

    n_features = x.size(-1)
    records = []

    for j in range(n_features):
        x_pert = x.clone()
        x_pert[:, :, j] = mask_value
        out = model(x_pert)
        p = float(torch.sigmoid(out["risk_logit"]).item())
        records.append(
            {
                "feature_index": j,
                "feature": feature_names[j],
                "attack_probability_original": base_prob,
                "attack_probability_masked": p,
                "importance_score": base_prob - p,
                "abs_importance": abs(base_prob - p),
                "category_changed": bool(out["category_logits"].argmax(-1).item() != base_cat),
            }
        )

    records.sort(key=lambda r: r["abs_importance"], reverse=True)
    return {
        "base_attack_probability": base_prob,
        "base_category_index": base_cat,
        "base_category_confidence": base_cat_prob,
        "mask_value_zspace": mask_value,
        "method": "Model feature importance / perturbation analysis",
        "is_causal_explanation": False,
        "caveat": (
            "Perturbation-based model sensitivity only. Not causal, not SHAP. "
            "Correlated features can mask each other's importance."
        ),
        "importances": records,
    }


EXPERIMENTS = {
    "original": {
        "checkpoint": "warden_t_best.pt",
        "scaler": "scaler.npz",
        "vocabulary": "category_vocabulary.json",
        "history": "training_history.json",
    },
    "attack_day": {
        "checkpoint": "attack_day_best.pt",
        "scaler": "attack_day_scaler.npz",
        "vocabulary": "attack_day_vocabulary.json",
        "history": "attack_day_training_history.json",
    },
}


def load_artifacts(device=None, experiment="original"):
    """
    Load a trained checkpoint, feature names and scaler from models/.

    `experiment` selects which run to load, so the original model and the
    attack-day experiment coexist without either overwriting the other.
    """
    import json

    import torch

    from ..models.warden_transformer import build_model

    if experiment not in EXPERIMENTS:
        raise ValueError(
            f"unknown experiment {experiment!r}; choose from {list(EXPERIMENTS)}"
        )
    files = EXPERIMENTS[experiment]
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = MODELS_DIR / files["checkpoint"]
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"{ckpt_path} not found - run "
            f"`python -m src.models.train{'_attack_day' if experiment == 'attack_day' else ''}` first."
        )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(
        ckpt["model_config"],
        ckpt["num_categories"],
        ckpt.get("unlearnable_categories"),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with open(MODELS_DIR / files["vocabulary"]) as f:
        vocab = json.load(f)
    with open(MODELS_DIR / files["history"]) as f:
        history = json.load(f)

    scaler = FeatureScaler.load(MODELS_DIR / files["scaler"])
    return {
        "model": model,
        "scaler": scaler,
        "feature_names": ckpt["feature_names"],
        "index_to_category": ckpt["index_to_category"],
        "category_to_index": vocab["category_to_index"],
        "best_epoch": history.get("best_epoch"),
        "device": device,
        "experiment": experiment,
    }


def explain_sample(index=0, split="test", top_k=5, verbose=True, experiment="original"):
    """
    Run attention + perturbation explainability on one real sample.

    Returns (sequence, category_name, attention, importance) where
    `sequence` is the raw (unscaled) test sequence.
    """
    from ..models.dataset import load_raw_splits

    artifacts = load_artifacts(experiment=experiment)
    model = artifacts["model"]
    scaler = artifacts["scaler"]

    if experiment == "attack_day":
        from ..models.attack_day_split import (
            CANDIDATE_A,
            build_sequence_indices,
            load_states_and_targets,
            make_split_arrays,
            state_blocks,
            verify_segments,
        )
        from ..models.dataset import PROCESSED_DIR
        import pandas as pd

        verify_segments(verbose=False)
        n_states = len(
            pd.read_csv(PROCESSED_DIR / "network_states.csv", usecols=["is_attack"])
        )
        blocks = state_blocks(n_states)
        idx = build_sequence_indices(blocks, CANDIDATE_A)[split]
        X_all, y_all, cat_all = load_states_and_targets(artifacts["feature_names"])
        X, _, categories, _ = make_split_arrays(
            idx, X_all, y_all, cat_all, artifacts["feature_names"]
        )
    else:
        raw = load_raw_splits()
        X = raw[f"X_{split}"]
        categories = raw[f"cat_{split}"]

    sequence_raw = X[index]
    sequence_norm = scaler.transform(sequence_raw)

    attn = extract_attention(model, sequence_norm, device=artifacts["device"])
    importance = perturbation_importance(
        model,
        sequence_norm,
        artifacts["feature_names"],
        scaler=scaler,
        device=artifacts["device"],
    )

    if verbose:
        print("=" * 70)
        print("EXPLAINABILITY - single real test sample")
        print("=" * 70)
        print(f"  split / index     : {split}[{index}]")
        print(f"  true category     : {categories[index]}")
        print(f"  predicted attack  : {importance['base_attack_probability']:.4f}")
        print(
            f"  predicted category: "
            f"{artifacts['index_to_category'][importance['base_category_index']]} "
            f"(p={importance['base_category_confidence']:.4f})"
        )

        print("\n  Temporal attention (layer 0, head-averaged, row=query):")
        print("        " + "".join(f"  t-{9 - i:<2d}" for i in range(10)))
        for qi, row in enumerate(attn["layer0_head_averaged"]):
            label = f"  t-{9 - qi:<2d} |"
            print(label + "".join(f"{v:5.2f}" for v in row))

        print("\n  Input-step importance (attention mass received, layer 0):")
        for i, v in enumerate(attn["input_importance"]):
            bar = "#" * int(v * 40)
            print(f"    t-{9 - i:<2d} {v:5.3f} {bar}")

        print(
            f"\n  Top {top_k} features by "
            f"{importance['method']}:"
        )
        for rank, rec in enumerate(importance["importances"][:top_k], 1):
            print(
                f"    {rank}. {rec['feature']:<38s} "
                f"importance={rec['importance_score']:+.4f}  "
                f"(p {rec['attack_probability_original']:.4f} -> "
                f"{rec['attack_probability_masked']:.4f})"
            )
        print(f"\n  {importance['caveat']}")

    return sequence_raw, categories[index], attn, importance
