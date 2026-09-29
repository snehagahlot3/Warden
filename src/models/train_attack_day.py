# -*- coding: utf-8 -*-
"""
Candidate A: attack-day-aware training. SAME WARDEN-T architecture.

    python -m src.models.train_attack_day

TRAIN      = S1 Mon+Tue  +  S2 Wed   (whole capture days)
VALIDATION = S3 Thu                  (whole capture day)
TEST       = S4 Fri                  (whole capture day)

LEAKAGE CONTROLS
================
* Feature scaler  : fitted on TRAIN sequences only.
* Class weights   : from TRAIN labels only.
* BCE pos_weight  : from TRAIN labels only.
* Category vocab  : TRAIN + VALIDATION only. The test categories are never
                    inspected when sizing the model head.
* Early stopping  : on validation total loss. The test split is not touched
                    during training, and no hyperparameter is chosen using it.
* Day boundaries  : intact; no 10-step sequence straddles a boundary.

This script never writes to models/warden_t_best.pt or models/test_metrics.json.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from .attack_day_split import (
    CANDIDATE_A,
    MODELS_DIR,
    build_sequence_indices,
    describe_split,
    detect_boundaries,
    load_states_and_targets,
    make_split_arrays,
    verify_segments,
)
from .dataset import (
    OOV_CATEGORY,
    FeatureScaler,
    WardenDataset,
    load_metadata,
)
from .losses import WardenMultiTaskLoss, compute_class_weights, compute_pos_weight
from .train import get_device, run_validation, set_seed
from .warden_transformer import build_model

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"
EXPERIMENT_DIR = MODELS_DIR


def load_config():
    with open(CONFIG_PATH) as f:
        merged = {}
        for doc in yaml.safe_load_all(f):
            if isinstance(doc, dict):
                merged.update(doc)
    return merged


def main():
    config = load_config()
    model_cfg, train_cfg = config["model"], config["training"]

    set_seed(train_cfg.get("seed", 42))
    device = get_device()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("CANDIDATE A - ATTACK-DAY-AWARE SPLIT (same WARDEN-T architecture)")
    print("=" * 74)
    print(f"  device {device} | epochs {train_cfg['epochs']} | "
          f"batch {train_cfg['batch_size']} | lr {train_cfg['learning_rate']}")

    # ---------------------------------------------------- boundary provenance
    print("\n[1/6] Deriving capture-day boundaries from the raw CSV ...")
    payload = detect_boundaries()
    m = payload["measured_boundaries"]
    print(f"  measured: wed_start={m['wed_start']} "
          f"thu_start={m['thu_start']} (first Thu attack at "
          f"{m['thu_start_first_thu_attack']}, benign gap of "
          f"{m['thu_benign_gap_rows']:,} rows) fri_start={m['fri_start']}")
    print("  verifying day composition on each side ...")
    blocks = verify_segments()

    # ------------------------------------------------------------ build split
    feature_names, _ = load_metadata()
    X_all, y_all, cat_all = load_states_and_targets(feature_names)
    seq_idx = build_sequence_indices(blocks, CANDIDATE_A)

    split_data, split_report = {}, {}
    print("\n[2/6] Attack-day-aware split")
    for name, key in (("TRAIN", "train"), ("VALIDATION", "validation"), ("TEST", "test")):
        X, y, cat, nxt = make_split_arrays(seq_idx[key], X_all, y_all, cat_all, feature_names)
        split_data[key] = (X, y, cat, nxt)
        segs = CANDIDATE_A[key]
        rngs = ", ".join(f"[{blocks[s][0]},{blocks[s][1]})" for s in segs)
        print(f"  {key} day blocks: {', '.join(segs)}  states {rngs}")
        split_report[key] = describe_split(
            name, X, y, cat, blocks, segs
        )
        split_report[key]["sequence_index_ranges"] = [
            [int(blocks[s][0]), int(blocks[s][1] - 10)] for s in segs
        ]

    # Explicit statement of the boundary-crossing sequences that were dropped.
    print("\n  Day-boundary integrity check:")
    for key, segs in CANDIDATE_A.items():
        lo = min(blocks[s][0] for s in segs)
        hi = max(blocks[s][1] for s in segs)
        n_in = (hi - lo) - 10  # usable if the block were contiguous
        n_out = len(seq_idx[key])
        print(f"    {key:<11} states[{lo}:{hi}) -> {n_out} sequences"
              f"  (a contiguous block would give {n_in}; "
              f"{n_in - n_out} boundary-straddling sequences dropped)")

    # --------------------------------------------------------- leakage safety
    print("\n[3/6] Fitting scaler on TRAIN ONLY and deriving TRAIN-only weights")
    Xtr, ytr, cattr, ntr = split_data["train"]
    scaler = FeatureScaler.fit(Xtr, clip=train_cfg.get("feature_clip", 10.0))
    scaler.save(EXPERIMENT_DIR / "attack_day_scaler.npz")

    # Vocabulary from TRAIN + VALIDATION only (test never inspected).
    vocab_source = list(np.unique(cattr)) + list(np.unique(split_data["validation"][2]))
    category_to_index = {str(c): i for i, c in enumerate(sorted(set(vocab_source)))}
    category_to_index[OOV_CATEGORY] = len(category_to_index)
    index_to_category = {v: k for k, v in category_to_index.items()}
    num_categories = len(category_to_index)

    test_cats = set(np.unique(split_data["test"][2]).tolist())
    oov_test = sorted(c for c in test_cats if c not in category_to_index)
    print(f"  categories in vocabulary : {num_categories} (train+val only)")
    print(f"  test categories          : {len(test_cats)}")
    print(f"  test categories NOT in vocabulary : {len(oov_test)} -> {oov_test}")
    print("  NOTE: the vocabulary is defined WITHOUT looking at test, so no test")
    print("        label information influences the model architecture.")

    datasets, loaders = {}, {}
    for i, key in enumerate(("train", "validation", "test")):
        X, y, cat, nxt = split_data[key]
        ds = WardenDataset(
            scaler.transform(X), y, cat, scaler.transform(nxt), category_to_index
        )
        datasets[key] = ds
        loaders[key] = DataLoader(
            ds, batch_size=train_cfg["batch_size"], shuffle=(key == "train")
        )

    # ----------------------------------------------------------- model + loss
    print("\n[4/6] Building the SAME WARDEN-T architecture")
    train_cat_idx = torch.from_numpy(datasets["train"].y_cat)
    class_weights, class_counts = compute_class_weights(
        train_cat_idx, num_categories, mode=train_cfg["class_weight_mode"]
    )
    unlearnable = class_counts <= 0
    print("  category class weights (TRAIN counts only):")
    for i, w in enumerate(class_weights.tolist()):
        flag = "   <- 0 train samples, masked from softmax" if class_counts[i] == 0 else ""
        print(f"     [{i:2d}] {index_to_category[i][:38]:<40s} "
              f"count={int(class_counts[i]):5d}  weight={w:.4f}{flag}")
    print(f"  {int(unlearnable.sum())} of {num_categories} categories have zero "
          "TRAIN support and are masked out of the softmax.")

    model = build_model(model_cfg, num_categories, unlearnable).to(device)
    print(f"  parameters: {model.num_parameters():,}  "
          f"d_model={model_cfg['d_model']} nhead={model_cfg['nhead']} "
          f"layers={model_cfg['num_layers']} ffn={model_cfg['dim_feedforward']} "
          f"dropout={model_cfg['dropout']}")

    pos_weight, pos_ratio = compute_pos_weight(
        torch.from_numpy(ytr.astype(np.float32)),
        enabled=train_cfg.get("risk_pos_weight", True),
    )
    print(f"  risk pos_weight (TRAIN neg/pos = {pos_ratio:.4f}) -> "
          f"{pos_weight.item():.4f}")

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
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, train_cfg["patience"] // 2)
    )

    # --------------------------------------------------------------- training
    print("\n[5/6] Training (early stopping on VALIDATION total loss)")
    print("-" * 74)
    print(f"{'Epoch':>5}  {'TrainLoss':>10}  {'ValLoss':>10}  {'RiskF1':>8}  "
          f"{'RiskPrec':>9}  {'RiskRec':>8}  {'ValAUC':>8}  {'CatAcc':>7}  {'LR':>10}")
    print("-" * 74)

    history, best_val, best_epoch, stale = [], float("inf"), -1, 0
    checkpoint = EXPERIMENT_DIR / "attack_day_best.pt"
    t0 = time.time()

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        agg = {"total": 0.0, "state": 0.0, "risk": 0.0, "category": 0.0}
        nb = 0
        for Xb, yb, cb, nb_ in loaders["train"]:
            Xb, yb, cb, nb_ = Xb.to(device), yb.to(device), cb.to(device), nb_.to(device)
            optimizer.zero_grad()
            out = model(Xb)
            loss, parts = criterion(out, nb_, yb, cb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_cfg["grad_clip"]
            )
            optimizer.step()
            agg["total"] += loss.item()
            for k, v in parts.items():
                agg[k] += v.item()
            nb += 1
        tr = {k: v / nb for k, v in agg.items()}

        va = run_validation(model, loaders["validation"], criterion, device,
                            index_to_category)
        scheduler.step(va["total"])
        auc = va["risk_roc_auc"]
        print(f"{epoch:>5}  {tr['total']:>10.4f}  {va['total']:>10.4f}  "
              f"{va['risk_f1']:>8.4f}  {va['risk_precision']:>9.4f}  "
              f"{va['risk_recall']:>8.4f}  {auc:>8.4f}  "
              f"{va['category_accuracy']:>7.4f}  "
              f"{optimizer.param_groups[0]['lr']:>10.2e}")

        history.append({
            "epoch": epoch,
            "train_total_loss": tr["total"], "train_state_loss": tr["state"],
            "train_risk_loss": tr["risk"], "train_category_loss": tr["category"],
            "val_total_loss": va["total"], "val_state_loss": va["state"],
            "val_risk_loss": va["risk"], "val_category_loss": va["category"],
            "val_risk_f1": va["risk_f1"], "val_risk_precision": va["risk_precision"],
            "val_risk_recall": va["risk_recall"], "val_risk_accuracy": va["risk_accuracy"],
            "val_risk_roc_auc": auc, "val_category_accuracy": va["category_accuracy"],
            "val_category_macro_f1": va["category_macro_f1"],
            "lr": optimizer.param_groups[0]["lr"],
        })

        if not np.isfinite(va["total"]):
            print("\n  Non-finite validation loss - divergence, stopping.")
            break
        if va["total"] < best_val:
            best_val, best_epoch, stale = va["total"], epoch, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch, "val_total_loss": best_val,
                "model_config": model_cfg, "num_categories": num_categories,
                "unlearnable_categories": unlearnable.tolist(),
                "index_to_category": index_to_category,
                "category_to_index": category_to_index,
                "feature_names": feature_names,
                "experiment": "candidate_a_attack_day",
            }, checkpoint)
        else:
            stale += 1
            if stale >= train_cfg["patience"]:
                print(f"\n  Early stopping: no validation improvement for "
                      f"{train_cfg['patience']} epochs (best epoch {best_epoch}).")
                break

    elapsed = time.time() - t0
    early_stopped = len(history) < train_cfg["epochs"]

    with open(EXPERIMENT_DIR / "attack_day_training_history.json", "w") as f:
        json.dump({
            "experiment": "candidate_a_attack_day",
            "config": {"model": model_cfg, "training": train_cfg,
                       "batch_size": train_cfg["batch_size"],
                       "learning_rate": train_cfg["learning_rate"],
                       "epochs_configured": train_cfg["epochs"],
                       "patience": train_cfg["patience"],
                       "device": str(device)},
            "split": split_report,
            "best_epoch": best_epoch,
            "best_val_total_loss": best_val,
            "epochs_trained": len(history),
            "early_stopped": early_stopped,
            "wall_clock_seconds": elapsed,
            "parameters": model.num_parameters(),
            "num_categories": num_categories,
            "test_categories_out_of_vocabulary": oov_test,
            "history": history,
        }, f, indent=2)

    with open(EXPERIMENT_DIR / "attack_day_vocabulary.json", "w") as f:
        json.dump({
            "experiment": "candidate_a_attack_day",
            "category_to_index": category_to_index,
            "index_to_category": index_to_category,
            "derived_from": "train + validation categories only",
            "out_of_vocabulary_label": OOV_CATEGORY,
            "test_categories_not_in_vocab": oov_test,
        }, f, indent=2)

    print("-" * 74)
    print(f"\n[6/6] Done in {elapsed / 60:.1f} min | epochs trained {len(history)} "
          f"(configured {train_cfg['epochs']}) | early stopping fired: {early_stopped}")
    print(f"  best epoch    : {best_epoch}")
    print(f"  best val loss : {best_val:.4f}")
    print(f"  checkpoint    : {checkpoint}  (warden_t_best.pt untouched)")
    print("\nTest metrics: python -m src.models.evaluate_attack_day")


if __name__ == "__main__":
    sys.exit(main())
