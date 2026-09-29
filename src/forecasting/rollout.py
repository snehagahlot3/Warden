# -*- coding: utf-8 -*-
"""
WARDEN-T recursive multi-step forecasting.

WHAT THIS IS (AND WHAT IT IS NOT)
=================================
This module performs *recursive* (autoregressive) rollout:

    step 1:  [S(t-9), ..., S(t)]            -> S_hat(t+1)
    step 2:  [S(t-8), ..., S(t), S_hat(t+1)] -> S_hat(t+2)
    step 3:  [S(t-7), ..., S_hat(t+2)]      -> S_hat(t+3)
    ...

At step k+1 the model is conditioned on the state it predicted at step k. The
model NEVER observes real future traffic during the rollout.

Consequence that must be stated plainly: errors compound. Each step feeds a
prediction back as input, so rollout error grows with the horizon. Predictions
beyond t+1 are extrapolation of the model's own behaviour, not measurements of
what actually happened. The first horizon (t+1) is the only one backed by real
data; t+2..t+K are model-generated.

All arrays handled here are in the *normalised* feature space (z-scored with
the train-derived scaler), which is the space the network operates in. Use
`inverse_transform` from the FeatureScaler to return to raw feature units.
"""

import numpy as np
import torch


@torch.no_grad()
def recursive_rollout(model, sequence, horizon=5, feature_mean=None, feature_std=None):
    """
    Run a K-step recursive forecast from a single observed sequence.

    Parameters
    ----------
    model    : WardenTransformer (in eval mode)
    sequence : array-like [T, F] or tensor [1, T, F], normalised space
    horizon  : number of recursive steps K
    feature_mean, feature_std : optional, used to also return raw-unit states

    Returns
    -------
    dict with
      steps        : list of dicts, one per horizon t+1 .. t+K
      final_state  : normalised S_hat(t+K)
    """
    model.eval()
    device = next(model.parameters()).device

    if isinstance(sequence, np.ndarray):
        x = torch.from_numpy(np.ascontiguousarray(sequence, dtype=np.float32))
    else:
        x = sequence.detach().clone().float()
    if x.dim() == 2:
        x = x.unsqueeze(0)
    x = x.to(device)

    seq_len = x.size(1)
    window = x.clone()  # rolling buffer, most recent state last

    steps = []
    for k in range(1, horizon + 1):
        out = model(window)
        next_state = out["next_state"]          # [1, F]
        risk_prob = torch.sigmoid(out["risk_logit"]).squeeze(-1)
        cat_prob = torch.softmax(out["category_logits"], dim=-1)
        cat_conf, cat_idx = cat_prob.max(dim=-1)

        state_np = next_state.squeeze(0).cpu().numpy()
        if feature_mean is not None and feature_std is not None:
            raw_state = state_np * np.asarray(feature_std) + np.asarray(feature_mean)
        else:
            raw_state = None

        steps.append(
            {
                "horizon": k,
                "next_state": state_np,
                "next_state_raw": raw_state,
                "attack_probability": float(risk_prob.item()),
                "category_index": int(cat_idx.item()),
                "category_confidence": float(cat_conf.item()),
                "category_probabilities": cat_prob.squeeze(0).cpu().numpy(),
            }
        )

        # Roll the window forward: drop the oldest state, append the prediction.
        window = torch.cat([window[:, 1:, :], next_state.unsqueeze(1)], dim=1)

    return {"steps": steps, "final_state": steps[-1]["next_state"]}


@torch.no_grad()
def batched_rollout(model, sequences, horizon=5, feature_mean=None, feature_std=None):
    """
    Vectorised recursive rollout over a batch.

    Mathematically identical to `recursive_rollout` but applies every step to
    the whole batch at once (much faster for many sequences).
    """
    model.eval()
    device = next(model.parameters()).device

    if isinstance(sequences, np.ndarray):
        x = torch.from_numpy(np.ascontiguousarray(sequences, dtype=np.float32))
    else:
        x = sequences.detach().clone().float()
    if x.dim() == 2:
        x = x.unsqueeze(0)
    x = x.to(device)

    window = x.clone()
    all_steps = []

    for k in range(1, horizon + 1):
        out = model(window)
        next_state = out["next_state"]
        risk_prob = torch.sigmoid(out["risk_logit"]).squeeze(-1)
        cat_prob = torch.softmax(out["category_logits"], dim=-1)
        cat_conf, cat_idx = cat_prob.max(dim=-1)

        raw_state = None
        if feature_mean is not None and feature_std is not None:
            raw_state = (
                next_state.cpu().numpy() * np.asarray(feature_std)
                + np.asarray(feature_mean)
            )

        all_steps.append(
            {
                "horizon": k,
                "next_state": next_state.cpu().numpy(),
                "next_state_raw": raw_state,
                "attack_probability": risk_prob.cpu().numpy(),
                "category_index": cat_idx.cpu().numpy(),
                "category_confidence": cat_conf.cpu().numpy(),
                "category_probabilities": cat_prob.cpu().numpy(),
            }
        )

        window = torch.cat([window[:, 1:, :], next_state.unsqueeze(1)], dim=1)

    return {"steps": all_steps, "final_state": all_steps[-1]["next_state"]}
