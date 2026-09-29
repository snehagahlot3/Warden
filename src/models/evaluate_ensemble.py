# -*- coding: utf-8 -*-

import json
import sys
from pathlib import Path

import numpy as np
import torch

from .dataset import MODELS_DIR, FeatureScaler, load_raw_splits, load_states, next_state_targets
from .warden_transformer import build_model
from .train import load_config, set_seed, get_device

BASE_DIR = Path(__file__).resolve().parents[2]


def train_single_model(seed, device):
    config = load_config()
    model_cfg = config["model"]
    train_cfg = config["training"]

    set_seed(seed)
    raw = load_raw_splits()
    states, feature_names = load_states()

    scaler = FeatureScaler.fit(raw["X_train"], clip=train_cfg.get("feature_clip", 10.0))
    X_train = scaler.transform(raw["X_train"])
    X_val = scaler.transform(raw["X_val"])
    X_test = scaler.transform(raw["X_test"])

    split_sizes = (len(raw["X_train"]), len(raw["X_val"]), len(raw["X_test"]))
    nxt, _ = next_state_targets(split_sizes, states, scaler=scaler)

    model = build_model(model_cfg, num_categories=2).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"])

    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(raw["y_train"]).float().unsqueeze(1),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(15):
        model.train()
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out["risk_logit"], y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test).float().to(device)
        out = model(X_test_t)
        risk_prob = torch.sigmoid(out["risk_logit"]).squeeze(-1).cpu().numpy()

    return risk_prob


def main():
    device = get_device()
    n_seeds = 3
    all_probs = []

    for seed in [42, 123, 456][:n_seeds]:
        print(f"Training model with seed {seed}...")
        probs = train_single_model(seed, device)
        all_probs.append(probs)

    all_probs = np.array(all_probs)
    mean_prob = all_probs.mean(axis=0)
    std_prob = all_probs.std(axis=0)

    raw = load_raw_splits()
    y_test = raw["y_test"]

    from sklearn.metrics import f1_score, roc_auc_score

    results = {
        "n_seeds": n_seeds,
        "seeds": [42, 123, 456][:n_seeds],
        "mean_risk_prob": mean_prob.tolist(),
        "std_risk_prob": std_prob.tolist(),
        "y_test": y_test.tolist(),
        "f1_mean": float(f1_score(y_test, (mean_prob > 0.5).astype(int), zero_division=0)),
        "roc_auc_mean": float(roc_auc_score(y_test, mean_prob)) if len(np.unique(y_test)) > 1 else None,
        "mean_uncertainty": float(std_prob.mean()),
        "max_uncertainty": float(std_prob.max()),
    }

    with open(MODELS_DIR / "ensemble_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nEnsemble F1: {results['f1_mean']:.4f}")
    print(f"Ensemble ROC-AUC: {results['roc_auc_mean']:.4f}")
    print(f"Mean uncertainty: {results['mean_uncertainty']:.4f}")
    print(f"Results saved to {MODELS_DIR / 'ensemble_results.json'}")


if __name__ == "__main__":
    sys.exit(main())
