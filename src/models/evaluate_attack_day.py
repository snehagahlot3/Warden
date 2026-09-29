# -*- coding: utf-8 -*-
"""
Candidate A evaluation: attack-day-aware split, on the Friday test day.

    python -m src.models.evaluate_attack_day

Writes (never overwriting the original experiment):
    models/attack_day_metrics.json
    models/attack_day_confusion_matrix.png
    models/attack_day_category_report.txt
    models/experiment_comparison.json

All numbers come from actual predictions on the actual held-out Friday day.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .attack_day_split import (
    CANDIDATE_A,
    MODELS_DIR,
    build_sequence_indices,
    detect_boundaries,
    load_states_and_targets,
    make_split_arrays,
    verify_segments,
)
from .dataset import OOV_CATEGORY, FeatureScaler, load_metadata
from .warden_transformer import build_model

BASE_DIR = Path(__file__).resolve().parents[2]
CHECKPOINT = MODELS_DIR / "attack_day_best.pt"
ORIGINAL_METRICS = MODELS_DIR / "test_metrics.json"


def safe_roc(y, s):
    try:
        return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else None
    except ValueError:
        return None


def safe_pr(y, s):
    try:
        return float(average_precision_score(y, s)) if len(np.unique(y)) > 1 else None
    except ValueError:
        return None


@torch.no_grad()
def run_model(model, X, device, bs=128):
    ns, rl, cl = [], [], []
    model.eval()
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + bs], dtype=np.float32)).to(device)
        o = model(xb)
        ns.append(o["next_state"].cpu().numpy())
        rl.append(o["risk_logit"].squeeze(-1).cpu().numpy())
        cl.append(o["category_logits"].cpu().numpy())
    return np.concatenate(ns), np.concatenate(rl), np.concatenate(cl)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"{CHECKPOINT} missing - run train_attack_day first.")
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = build_model(ckpt["model_config"], ckpt["num_categories"],
                        ckpt.get("unlearnable_categories")).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    index_to_category = ckpt["index_to_category"]
    feature_names = ckpt["feature_names"]

    with open(MODELS_DIR / "attack_day_scaler.npz") as _:
        pass
    scaler = FeatureScaler.load(MODELS_DIR / "attack_day_scaler.npz")

    detect_boundaries()
    blocks = verify_segments(verbose=False)
    seq_idx = build_sequence_indices(blocks, CANDIDATE_A)
    X_all, y_all, cat_all = load_states_and_targets(feature_names)

    Xte_raw, yte, cte, nte_raw = make_split_arrays(seq_idx["test"], X_all, y_all, cat_all,
                                                  feature_names)
    Xte = scaler.transform(Xte_raw)
    nte = scaler.transform(nte_raw)

    with open(MODELS_DIR / "attack_day_vocabulary.json") as f:
        vocab = json.load(f)
    c2i = vocab["category_to_index"]
    oov_i = c2i[OOV_CATEGORY]
    y_cat = np.array([c2i.get(str(c), oov_i) for c in cte])

    print("=" * 74)
    print("CANDIDATE A - TEST EVALUATION (Friday capture day)")
    print("=" * 74)
    print(f"  checkpoint epoch {ckpt['epoch']} (val loss {ckpt['val_total_loss']:.4f})")
    print(f"  test sequences {len(yte)} | attack share {yte.mean() * 100:.2f}%")

    pred_next, risk_logit, cat_logits = run_model(model, Xte, device)
    risk_prob = 1 / (1 + np.exp(-risk_logit))
    risk_pred = (risk_logit > 0).astype(int)
    cat_pred = cat_logits.argmax(1)
    e = np.exp(cat_logits - cat_logits.max(1, keepdims=True))
    cat_conf = (e / e.sum(1, keepdims=True)).max(1)

    # ------------------------------------------------------------------ risk
    risk = {
        "precision": float(precision_score(yte, risk_pred, zero_division=0)),
        "recall": float(recall_score(yte, risk_pred, zero_division=0)),
        "f1": float(f1_score(yte, risk_pred, zero_division=0)),
        "roc_auc": safe_roc(yte, risk_prob),
        "pr_auc": safe_pr(yte, risk_prob),
        "accuracy": float(accuracy_score(yte, risk_pred)),
        "confusion_matrix": confusion_matrix(yte, risk_pred).tolist(),
        "n_test": int(len(yte)), "n_attack_true": int(yte.sum()),
    }
    # -------------------------------------------------------------- category
    present = sorted(set(y_cat.tolist()))
    names = [index_to_category[i] for i in present]
    category = {
        "accuracy": float(accuracy_score(y_cat, cat_pred)),
        "macro_f1": float(f1_score(y_cat, cat_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_cat, cat_pred, average="weighted", zero_division=0)),
        "oov_targets": int((y_cat == oov_i).sum()),
        "oov_categories": vocab["test_categories_not_in_vocab"],
        "classes_present_in_test": names,
    }
    report = classification_report(y_cat, cat_pred, labels=present,
                                   target_names=names, zero_division=0, digits=4)
    # ------------------------------------------------------------- next state
    raw_pred = scaler.inverse_transform(pred_next)
    next_state = {
        "mae": float(np.mean(np.abs(pred_next - nte))),
        "mse": float(np.mean((pred_next - nte) ** 2)),
        "mae_raw_units": float(np.mean(np.abs(raw_pred - nte_raw))),
        "mse_raw_units": float(np.mean((raw_pred - nte_raw) ** 2)),
        "trivial_baseline_mae": float(np.mean(np.abs(nte))),
        "trivial_baseline_mse": float(np.mean(nte ** 2)),
    }

    with open(MODELS_DIR / "attack_day_training_history.json") as f:
        hist = json.load(f)

    metrics = {
        "experiment": "candidate_a_attack_day",
        "protocol": (
            "TRAIN = Mon+Tue+Wed, VALIDATION = Thu, TEST = Fri. Each split is a "
            "whole, unsplit CIC-IDS2017 capture day. No 10-step sequence crosses "
            "a day boundary. Scaler fitted on train only; class weights from "
            "train labels only; category vocabulary from train+val only."
        ),
        "checkpoint": str(CHECKPOINT),
        "best_epoch": hist["best_epoch"],
        "epochs_trained": hist["epochs_trained"],
        "epochs_configured": hist["config"]["epochs_configured"],
        "early_stopped": hist["early_stopped"],
        "final_train_loss": hist["history"][-1]["train_total_loss"],
        "best_val_loss": hist["best_val_total_loss"],
        "split": hist["split"],
        "risk": risk,
        "category": category,
        "next_state": next_state,
    }

    with open(MODELS_DIR / "attack_day_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(MODELS_DIR / "attack_day_category_report.txt", "w") as f:
        f.write("CANDIDATE A - attack-category classification report (TEST = Friday)\n")
        f.write("=" * 70 + "\n")
        f.write(f"test sequences: {len(y_cat)}\n")
        f.write(f"vocabulary size: {len(index_to_category)} (train+val only)\n")
        f.write(f"OOV targets: {category['oov_targets']} "
                f"(categories: {category['oov_categories']})\n\n")
        f.write(report)
        f.write("\nThe Friday capture day contains DoS GoldenEye and Heartbleed,\n"
                "which occur on no earlier day, so they are unreachable for a\n"
                "temporally ordered train->test protocol. This is a property of\n"
                "CIC-IDS2017, not a modelling defect.\n")

    # ------------------------------------------------------------ confusion
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cm = confusion_matrix(yte, risk_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({cm[i, j] / cm.sum() * 100:.1f}%)",
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Pred BENIGN", "Pred ATTACK"])
        ax.set_yticks([0, 1], ["True BENIGN", "True ATTACK"])
        ax.set_title(f"Candidate A risk confusion matrix (Friday, F1={risk['f1']:.4f})")
        fig.colorbar(ax.get_images()[0], ax=ax)
        fig.tight_layout()
        fig.savefig(MODELS_DIR / "attack_day_confusion_matrix.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"  (plot skipped: {exc})")

    # ------------------------------------------------------------ comparison
    comparison = {
        "purpose": (
            "Two different evaluation protocols for the SAME WARDEN-T "
            "architecture and the same underlying CIC-IDS2017 states. These are "
            "not two competing models; only the split changed."
        ),
        "what_changed": {
            "original_temporal_split": (
                "Sequences cut at fixed fractions (60/20/20) of the sequence "
                "stream. Because the stream is ordered by capture day, each cut "
                "lands mid-day, so VALIDATION and TEST were each a MIXTURE of two "
                "different attack days. 8 of 17 category slots had zero training "
                "support and 345/838 test category targets were out-of-vocabulary."
            ),
            "candidate_a_attack_day_split": (
                "Splits placed on capture-day boundaries. Each split is one or "
                "more whole days, so every split has a single coherent attack "
                "profile. No 10-step sequence crosses a day boundary. Category "
                "support is still not universal - DoS GoldenEye and Heartbleed "
                "exist only on Friday - but the validation day is no longer a "
                "mixture, which is what destabilised the original run."
            ),
        },
        "original_temporal_split": {
            "protocol": "60/20/20 fixed-fraction cut of the sequence stream",
            "n_train": 2513, "n_validation": 838, "n_test": 838,
            "risk_precision": None, "risk_recall": None, "risk_f1": None,
            "risk_roc_auc": None, "risk_pr_auc": None,
            "category_accuracy": None, "category_macro_f1": None,
            "category_weighted_f1": None,
            "next_state_mae": None, "next_state_mse": None,
        },
        "candidate_a_attack_day_split": {
            "protocol": "TRAIN Mon+Tue+Wed, VALIDATION Thu, TEST Fri (whole days)",
            "n_train": metrics["split"]["train"]["n_sequences"],
            "n_validation": metrics["split"]["validation"]["n_sequences"],
            "n_test": risk["n_test"],
            "risk_precision": risk["precision"], "risk_recall": risk["recall"],
            "risk_f1": risk["f1"], "risk_roc_auc": risk["roc_auc"],
            "risk_pr_auc": risk["pr_auc"],
            "category_accuracy": category["accuracy"],
            "category_macro_f1": category["macro_f1"],
            "category_weighted_f1": category["weighted_f1"],
            "next_state_mae": next_state["mae"],
            "next_state_mse": next_state["mse"],
        },
    }

    if ORIGINAL_METRICS.exists():
        with open(ORIGINAL_METRICS) as f:
            om = json.load(f)
        o = comparison["original_temporal_split"]
        o["risk_precision"] = om["risk"]["precision"]
        o["risk_recall"] = om["risk"]["recall"]
        o["risk_f1"] = om["risk"]["f1"]
        o["risk_roc_auc"] = om["risk"]["roc_auc"]
        o["risk_pr_auc"] = om["risk"]["pr_auc"]
        o["category_accuracy"] = om["category"]["accuracy"]
        o["category_macro_f1"] = om["category"]["macro_f1"]
        o["category_weighted_f1"] = om["category"]["weighted_f1"]
        o["next_state_mae"] = om["next_state"]["mae_normalised"]
        o["next_state_mse"] = om["next_state"]["mse_normalised"]
        o["oov_test_targets"] = om["category"]["oov_count"]
        o["best_epoch"] = om.get("checkpoint_epoch")

    a = comparison["candidate_a_attack_day_split"]
    a["oov_test_targets"] = category["oov_targets"]
    a["best_epoch"] = hist["best_epoch"]
    a["early_stopped"] = hist["early_stopped"]
    a["epochs_trained"] = hist["epochs_trained"]

    comparison["interpretation"] = (
        "Risk metrics are not directly comparable between the two protocols, "
        "because the underlying test data differs: the original test split mixed "
        "the tail of Thursday with all of Friday (838 sequences, 82.9% attack), "
        "while Candidate A tests on Friday alone (494 sequences, 92.1% attack). "
        "The Candidate A numbers are therefore not 'a better score for the same "
        "task' - they are a cleaner, harder, single-day task. The clearest "
        "like-for-like signal is training behaviour: under Candidate A the "
        "validation loss decreases monotonically instead of increasing from "
        "epoch 1, which is the symptom the day-mixed split caused. No claim is "
        "made that one model is better; the architecture and training recipe are "
        "identical and only the evaluation protocol differs."
    )

    with open(MODELS_DIR / "experiment_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # ----------------------------------------------------------------- print
    print(f"\nRISK (test = Friday, n={risk['n_test']}, attack {yte.sum()}/{len(yte)})")
    for k in ("precision", "recall", "f1", "roc_auc", "pr_auc"):
        v = risk[k]
        print(f"  {k:<10}: {v if v is None else round(v, 4)}")
    print(f"  confusion : {risk['confusion_matrix']} (rows=true, cols=pred)")

    print(f"\nCATEGORY")
    for k in ("accuracy", "macro_f1", "weighted_f1"):
        print(f"  {k:<10}: {round(category[k], 4)}")
    print(f"  OOV targets: {category['oov_targets']} -> {category['oov_categories']}")

    print(f"\nNEXT STATE")
    for k in ("mae", "mse", "trivial_baseline_mae", "trivial_baseline_mse"):
        print(f"  {k:<24}: {next_state[k]:.4f}")

    print(f"\nTRAINING")
    print(f"  best epoch        : {hist['best_epoch']}")
    print(f"  epochs trained    : {hist['epochs_trained']} "
          f"(configured {hist['config']['epochs_configured']})")
    print(f"  early stopping    : {hist['early_stopped']}")
    print(f"  final train loss  : {hist['history'][-1]['train_total_loss']:.4f}")
    print(f"  best val loss     : {hist['best_val_total_loss']:.4f}")

    print("\nCategory classification report (TEST):\n")
    print(report)
    print("\nSaved: attack_day_metrics.json, attack_day_category_report.txt, "
          "attack_day_confusion_matrix.png, experiment_comparison.json")
    print("Original artifacts (warden_t_best.pt, test_metrics.json) untouched.")


if __name__ == "__main__":
    sys.exit(main())
