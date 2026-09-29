# -*- coding: utf-8 -*-

import json
import sys
from pathlib import Path

import numpy as np
import torch

from .dataset import (
    MODELS_DIR,
    FeatureScaler,
    load_raw_splits,
    load_states,
    next_state_targets,
)
from .warden_transformer import build_model
from .train import load_config, set_seed, get_device

BASE_DIR = Path(__file__).resolve().parents[2]


def evaluate_loo_family(family_to_holdout, device):
    config = load_config()
    model_cfg = config["model"]
    train_cfg = config["training"]

    set_seed(42)
    raw = load_raw_splits()
    states, feature_names = load_states()

    train_mask = np.array([str(c) != family_to_holdout for c in raw["cat_train"]])
    test_mask = np.array([str(c) == family_to_holdout for c in raw["cat_test"]])

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        return None

    X_train = raw["X_train"][train_mask]
    y_train = raw["y_train"][train_mask]
    X_test = raw["X_test"][test_mask]
    y_test = raw["y_test"][test_mask]

    scaler = FeatureScaler.fit(X_train, clip=train_cfg.get("feature_clip", 10.0))
    X_train_norm = scaler.transform(X_train)
    X_test_norm = scaler.transform(X_test)

    split_sizes = (len(X_train), 0, len(X_test))
    nxt, _ = next_state_targets(split_sizes, states, scaler=scaler)

    model = build_model(model_cfg, num_categories=2).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"])

    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train_norm).float(),
        torch.from_numpy(y_train).float().unsqueeze(1),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(20):
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
        X_test_t = torch.from_numpy(X_test_norm).float().to(device)
        out = model(X_test_t)
        risk_prob = torch.sigmoid(out["risk_logit"]).squeeze(-1).cpu().numpy()

    return {
        "family": family_to_holdout,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "precision": float(precision_score(y_test, (risk_prob > 0.5).astype(int), zero_division=0)),
        "recall": float(recall_score(y_test, (risk_prob > 0.5).astype(int), zero_division=0)),
        "f1": float(f1_score(y_test, (risk_prob > 0.5).astype(int), zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, risk_prob)) if len(np.unique(y_test)) > 1 else None,
    }


def main():
    device = get_device()
    families = ["PortScan", "DDoS", "DoS Hulk", "Bot", "FTP-Patator"]

    results = {}
    for family in families:
        print(f"\n{'='*60}")
        print(f"Leave-one-family-out: {family}")
        print(f"{'='*60}")
        result = evaluate_loo_family(family, device)
        if result:
            results[family] = result
            print(f"  F1: {result['f1']:.4f}")
            print(f"  ROC-AUC: {result['roc_auc']:.4f}" if result['roc_auc'] else "  ROC-AUC: N/A")

    with open(MODELS_DIR / "loo_family_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {MODELS_DIR / 'loo_family_results.json'}")


if __name__ == "__main__":
    sys.exit(main())
