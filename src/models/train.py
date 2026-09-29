# -*- coding: utf-8 -*-
"""
WARDEN-T training entry point.

    python -m src.models.train

Trains the multi-task temporal transformer on the processed CIC-IDS2017
temporal sequences, with early stopping on the validation total loss, and
saves the best checkpoint plus the full training history.

Artifacts written to models/:
    scaler.npz                train-only feature scaler
    category_vocabulary.json  data-derived attack-category index space
    warden_t_best.pt          best-validation checkpoint
    training_history.json     per-epoch losses and validation metrics
    category_class_weights.json  data-derived class weights actually used
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from .dataset import (
    MODELS_DIR,
    OOV_CATEGORY,
    FeatureScaler,
    WardenDataset,
    build_category_vocab,
    describe_targets,
    load_metadata,
    load_raw_splits,
    load_states,
    next_state_targets,
    verify_alignment,
)
from .losses import WardenMultiTaskLoss, compute_class_weights, compute_pos_weight
from .warden_transformer import build_model

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"


def load_config():
    """
    Load configs/config.yaml.

    The file is a multi-document YAML (one document per dataset/config block),
    so every document is parsed and merged into a single mapping. Later
    documents override earlier ones, which is what puts the `model: / training:
    / forecasting:` blocks at the end of the file in charge.
    """
    with open(CONFIG_PATH) as f:
        merged = {}
        for doc in yaml.safe_load_all(f):
            if isinstance(doc, dict):
                merged.update(doc)
    return merged


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    """Use CUDA when available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def run_validation(model, loader, criterion, device, category_names):
    """Compute average total loss and risk/category metrics on a split."""
    model.eval()
    totals = {"total": 0.0, "state": 0.0, "risk": 0.0, "category": 0.0}
    risk_true, risk_pred, risk_prob = [], [], []
    cat_true, cat_pred = [], []

    for X, y, cat, nxt in loader:
        X, y, cat, nxt = X.to(device), y.to(device), cat.to(device), nxt.to(device)
        outputs = model(X)
        loss, parts = criterion(outputs, nxt, y, cat)

        totals["total"] += loss.item()
        for k, v in parts.items():
            totals[k] += v.item()

        risk_true.extend(y.squeeze(-1).cpu().numpy().tolist())
        risk_pred.extend(
            (outputs["risk_logit"] > 0).long().squeeze(-1).cpu().numpy().tolist()
        )
        risk_prob.extend(
            torch.sigmoid(outputs["risk_logit"]).squeeze(-1).cpu().numpy().tolist()
        )
        cat_true.extend(cat.cpu().numpy().tolist())
        cat_pred.extend(outputs["category_logits"].argmax(-1).cpu().numpy().tolist())

    n = len(loader)
    metrics = {k: v / n for k, v in totals.items()}
    # Validation ROC-AUC is computed for model selection (it is threshold-free
    # and therefore comparable across decision thresholds). Selection is done
    # on validation only; test metrics are produced once, by evaluate.py.
    try:
        val_auc = (
            float(roc_auc_score(risk_true, risk_prob))
            if len(set(risk_true)) > 1
            else float("nan")
        )
    except ValueError:
        val_auc = float("nan")
    metrics.update(
        {
            "risk_roc_auc": val_auc,
            "risk_f1": f1_score(risk_true, risk_pred, zero_division=0),
            "risk_precision": precision_score(risk_true, risk_pred, zero_division=0),
            "risk_recall": recall_score(risk_true, risk_pred, zero_division=0),
            "risk_accuracy": float(np.mean(np.array(risk_true) == np.array(risk_pred))),
            "category_accuracy": float(
                np.mean(np.array(cat_true) == np.array(cat_pred))
            ),
            "category_macro_f1": f1_score(
                cat_true, cat_pred, average="macro", zero_division=0
            ),
        }
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train WARDEN-T")
    parser.add_argument("--epochs", type=int, default=None, help="override config epochs")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--risk-pos-weight",
        dest="risk_pos_weight",
        action="store_true",
        default=None,
        help="apply neg/pos BCE reweighting derived from the training labels",
    )
    parser.add_argument(
        "--no-risk-pos-weight",
        dest="risk_pos_weight",
        action="store_false",
        help="plain BCE (default: training split is attack-majority)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run a single epoch on a small subset to validate the code path",
    )
    args = parser.parse_args()

    config = load_config()
    model_cfg = config["model"]
    train_cfg = config["training"]
    epochs = args.epochs or train_cfg["epochs"]
    batch_size = args.batch_size or train_cfg["batch_size"]
    lr = args.lr if args.lr is not None else train_cfg["learning_rate"]
    patience = args.patience or train_cfg["patience"]
    if args.risk_pos_weight is not None:
        train_cfg["risk_pos_weight"] = args.risk_pos_weight

    set_seed(args.seed)
    device = get_device()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("WARDEN-T TRAINING")
    print("=" * 70)
    print(f"  device      : {device}")
    if device.type == "cpu":
        print("  CUDA not available -> training on CPU")
    print(f"  epochs      : {epochs}   batch_size: {batch_size}")
    print(f"  lr          : {lr}   weight_decay: {train_cfg['weight_decay']}")
    print(f"  patience    : {patience}")

    # ---------------------------------------------------------------- data
    print("\n[1/6] Verifying pipeline array alignment ...")
    verify_alignment()

    print("\n[2/6] Inspecting real target arrays ...")
    describe_targets()

    raw = load_raw_splits()
    states, feature_names = load_states()

    # Scaler fitted on train only, applied to BOTH the sequences and the
    # S(t+1) regression targets (identical 81-dim feature space).
    scaler = FeatureScaler.fit(
        raw["X_train"], clip=train_cfg.get("feature_clip", 10.0)
    )
    scaler.save(MODELS_DIR / "scaler.npz")

    split_sizes = (len(raw["X_train"]), len(raw["X_val"]), len(raw["X_test"]))
    nxt, offsets = next_state_targets(split_sizes, states, scaler=scaler)
    assert len(nxt[0]) == split_sizes[0] and len(nxt[2]) == split_sizes[2]

    # Category vocabulary from TRAIN + VAL only (never from test).
    category_to_index = build_category_vocab(
        {"train": raw["cat_train"], "val": raw["cat_val"]}
    )
    index_to_category = {v: k for k, v in category_to_index.items()}
    num_categories = len(category_to_index)

    train_cats = set(np.unique(raw["cat_train"]).tolist())
    val_cats = set(np.unique(raw["cat_val"]).tolist())
    test_cats = set(np.unique(raw["cat_test"]).tolist())
    oov_test = sorted(c for c in test_cats if c not in category_to_index)
    print(f"\n  category vocabulary size : {num_categories} (train+val only)")
    print(f"  categories in vocab      : {sorted(category_to_index)[:4]} ...")
    print(f"  test categories         : {len(test_cats)}")
    print(f"  test cats NOT in vocab  : {len(oov_test)} -> {oov_test}")
    if oov_test:
        print(
            "  NOTE: the temporal split crosses the five CIC-IDS2017 capture days,\n"
            "        each carrying a different attack family. Test categories the\n"
            "        model never saw cannot be learned; they are reported, not hidden."
        )

    # Scaler fitted on train only (already done above, before target building).
    Xs = {
        "train": scaler.transform(raw["X_train"]),
        "val": scaler.transform(raw["X_val"]),
        "test": scaler.transform(raw["X_test"]),
    }
    ys = {"train": raw["y_train"], "val": raw["y_val"], "test": raw["y_test"]}
    cats = {
        "train": raw["cat_train"],
        "val": raw["cat_val"],
        "test": raw["cat_test"],
    }

    datasets = {
        split: WardenDataset(
            Xs[split], ys[split], cats[split], nxt[i], category_to_index
        )
        for i, split in enumerate(("train", "val", "test"))
    }
    train_loader = DataLoader(
        datasets["train"], batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(datasets["val"], batch_size=batch_size, shuffle=False)

    if args.smoke:
        print("\n  [SMOKE TEST] 1 epoch on 256 train / 128 val samples")
        full_train_dataset = datasets["train"]
        full_val_dataset = datasets["val"]
        datasets["train"] = torch.utils.data.Subset(full_train_dataset, range(256))
        datasets["val"] = torch.utils.data.Subset(full_val_dataset, range(128))
        train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(datasets["val"], batch_size=batch_size, shuffle=False)
        epochs = 1
    else:
        # Referenced by the class-weight computation below.
        full_train_dataset = datasets["train"]

    # --------------------------------------------------------------- model
    # Data-derived class weights (never fabricated). Computed first because the
    # unlearnable-class mask is needed to build the model.
    train_cat_idx = torch.from_numpy(full_train_dataset.y_cat)
    class_weights, class_counts = compute_class_weights(
        train_cat_idx, num_categories, mode=train_cfg["class_weight_mode"]
    )
    unlearnable = class_counts <= 0
    n_unlearnable = int(unlearnable.sum())

    print("\n[3/6] Building model ...")
    model = build_model(model_cfg, num_categories, unlearnable).to(device)
    print(f"  parameters  : {model.num_parameters():,}")
    print(
        f"  d_model={model_cfg['d_model']} nhead={model_cfg['nhead']} "
        f"layers={model_cfg['num_layers']} ffn={model_cfg['dim_feedforward']} "
        f"dropout={model_cfg['dropout']} categories={num_categories}"
    )

    pos_weight, pos_weight_ratio = compute_pos_weight(
        torch.from_numpy(raw["y_train"].astype(np.float32)),
        enabled=train_cfg.get("risk_pos_weight", False),
    )
    print(
        f"  risk pos_weight : {pos_weight.item():.4f}  "
        f"(train neg/pos ratio = {pos_weight_ratio:.4f}; "
        f"train is {raw['y_train'].mean() * 100:.1f}% attack, so no "
        "minority reweighting is applied by default)"
    )
    print("  category class weights (from train counts):")
    for i, w in enumerate(class_weights.tolist()):
        flag = "  <- 0 train samples, masked out of softmax" if class_counts[i] == 0 else ""
        print(
            f"     [{i:2d}] {index_to_category[i][:38]:<40s} "
            f"count={int(class_counts[i]):5d}  weight={w:.4f}{flag}"
        )
    print(
        f"  {n_unlearnable} of {num_categories} categories have no training "
        "support and are therefore unpredictable."
    )

    criterion = WardenMultiTaskLoss(
        num_categories=num_categories,
        category_class_weights=class_weights.to(device),
        risk_pos_weight=pos_weight.to(device),
        unlearnable_classes=unlearnable,
        lambda_state=train_cfg["lambda_state"],
        lambda_risk=train_cfg["lambda_risk"],
        lambda_category=train_cfg["lambda_category"],
        huber_beta=train_cfg["huber_beta"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=train_cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, patience // 2)
    )

    with open(MODELS_DIR / "category_vocabulary.json", "w") as f:
        json.dump(
            {
                "category_to_index": category_to_index,
                "index_to_category": index_to_category,
                "derived_from": "train + val categories only",
                "out_of_vocabulary_label": OOV_CATEGORY,
                "test_categories_not_in_vocab": oov_test,
            },
            f,
            indent=2,
        )
    with open(MODELS_DIR / "category_class_weights.json", "w") as f:
        json.dump(
            {
                str(index_to_category[i]): {
                    "train_count": int(class_counts[i]),
                    "weight": float(class_weights[i]),
                }
                for i in range(num_categories)
            },
            f,
            indent=2,
        )

    # ------------------------------------------------------------ training
    print("\n[4/6] Training ...")
    print("-" * 70)
    header = (
        f"{'Epoch':>5}  {'TrainLoss':>10}  {'ValLoss':>10}  "
        f"{'RiskF1':>8}  {'RiskPrec':>9}  {'RiskRec':>8}  {'CatAcc':>7}  {'LR':>10}"
    )
    print(header)
    print("-" * 70)

    history = []
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    checkpoint_path = MODELS_DIR / "warden_t_best.pt"
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running = {"total": 0.0, "state": 0.0, "risk": 0.0, "category": 0.0}
        n_batches = 0

        for X, y, cat, nxt in train_loader:
            X, y, cat, nxt = X.to(device), y.to(device), cat.to(device), nxt.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss, parts = criterion(outputs, nxt, y, cat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
            optimizer.step()

            running["total"] += loss.item()
            for k, v in parts.items():
                running[k] += v.item()
            n_batches += 1

        train_metrics = {k: v / n_batches for k, v in running.items()}
        val_metrics = run_validation(model, val_loader, criterion, device, index_to_category)
        scheduler.step(val_metrics["total"])

        print(
            f"{epoch:>5}  {train_metrics['total']:>10.4f}  {val_metrics['total']:>10.4f}  "
            f"{val_metrics['risk_f1']:>8.4f}  {val_metrics['risk_precision']:>9.4f}  "
            f"{val_metrics['risk_recall']:>8.4f}  {val_metrics['category_accuracy']:>7.4f}  "
            f"{optimizer.param_groups[0]['lr']:>10.2e}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_total_loss": train_metrics["total"],
                "train_state_loss": train_metrics["state"],
                "train_risk_loss": train_metrics["risk"],
                "train_category_loss": train_metrics["category"],
                "val_total_loss": val_metrics["total"],
                "val_state_loss": val_metrics["state"],
                "val_risk_loss": val_metrics["risk"],
                "val_category_loss": val_metrics["category"],
                "val_risk_f1": val_metrics["risk_f1"],
                "val_risk_precision": val_metrics["risk_precision"],
                "val_risk_recall": val_metrics["risk_recall"],
                "val_risk_accuracy": val_metrics["risk_accuracy"],
                "val_category_accuracy": val_metrics["category_accuracy"],
                "val_category_macro_f1": val_metrics["category_macro_f1"],
                "val_risk_roc_auc": val_metrics["risk_roc_auc"],
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        if not np.isfinite(val_metrics["total"]):
            print(
                f"\n  Non-finite validation loss at epoch {epoch} - "
                "divergence detected, stopping."
            )
            if best_epoch == -1:
                print("  No usable checkpoint exists yet; run training again.")
            break

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_total_loss": best_val_loss,
                    "model_config": model_cfg,
                    "num_categories": num_categories,
                    "unlearnable_categories": unlearnable.tolist(),
                    "index_to_category": index_to_category,
                    "category_to_index": category_to_index,
                    "feature_names": feature_names,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"\n  Early stopping: no validation improvement for "
                    f"{patience} epochs (best epoch {best_epoch})."
                )
                break

    elapsed = time.time() - start_time

    with open(MODELS_DIR / "training_history.json", "w") as f:
        json.dump(
            {
                "config": {
                    "model": model_cfg,
                    "training": train_cfg,
                    "epochs_run": len(history),
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "patience": patience,
                    "seed": args.seed,
                    "device": str(device),
                },
                "best_epoch": best_epoch,
                "best_val_total_loss": best_val_loss,
                "wall_clock_seconds": elapsed,
                "parameters": model.num_parameters(),
                "num_categories": num_categories,
                "history": history,
            },
            f,
            indent=2,
        )

    print("-" * 70)
    print(f"\n[5/6] Training complete in {elapsed / 60:.1f} min")
    print(f"  best epoch      : {best_epoch}")
    print(f"  best val loss   : {best_val_loss:.4f}")
    print(f"  checkpoint      : {checkpoint_path}")

    print("\n[6/6] Saved artifacts:")
    for name in (
        "scaler.npz",
        "category_vocabulary.json",
        "category_class_weights.json",
        "warden_t_best.pt",
        "training_history.json",
    ):
        p = MODELS_DIR / name
        print(f"  {p}  ({p.stat().st_size / 1024:.1f} KB)")

    print("\nMetrics in this output are TRAINING/VALIDATION only.")
    print("Test metrics are produced by: python -m src.models.evaluate")


if __name__ == "__main__":
    sys.exit(main())
