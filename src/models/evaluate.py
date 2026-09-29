# -*- coding: utf-8 -*-
"""
WARDEN-T test-set evaluation and logistic-regression baseline.

    python -m src.models.evaluate

Runs the best checkpoint on the TEST split only and writes:
    models/test_metrics.json      real metrics from real test predictions
    models/confusion_matrix.png   risk confusion matrix
    models/category_report.txt    per-category classification report
    models/baseline_metrics.json  logistic-regression baseline on identical data
    models/comparison.txt         side-by-side precision/recall/F1/ROC-AUC

Every number written here is computed from actual model output on the actual
test split. Nothing is hardcoded or assumed.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
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
from sklearn.preprocessing import StandardScaler

from .dataset import (
    MODELS_DIR,
    OOV_CATEGORY,
    FeatureScaler,
    load_metadata,
    load_raw_splits,
    load_states,
    next_state_targets,
)
from .warden_transformer import build_model

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"
CHECKPOINT_PATH = MODELS_DIR / "warden_t_best.pt"


def load_checkpoint(device):
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"{CHECKPOINT_PATH} not found - run `python -m src.models.train` first."
        )
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = build_model(
        ckpt["model_config"],
        ckpt["num_categories"],
        ckpt.get("unlearnable_categories"),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def model_predictions(model, X, device, batch_size=128):
    """Run the model over a test tensor and return raw outputs."""
    next_states, risk_logits, cat_logits, attentions = [], [], [], []
    model.eval()
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(
            np.ascontiguousarray(X[i : i + batch_size], dtype=np.float32)
        ).to(device)
        out = model(xb, need_weights=True)
        next_states.append(out["next_state"].cpu().numpy())
        risk_logits.append(out["risk_logit"].squeeze(-1).cpu().numpy())
        cat_logits.append(out["category_logits"].cpu().numpy())
        attentions.append(out["attention"][0].cpu().numpy())
    return (
        np.concatenate(next_states),
        np.concatenate(risk_logits),
        np.concatenate(cat_logits),
        np.concatenate(attentions),
    )


def safe_auc(y_true, scores):
    """ROC-AUC if the task is well defined, else None with a reason."""
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return None


def safe_pr_auc(y_true, scores):
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return float(average_precision_score(y_true, scores))
    except ValueError:
        return None


def main():
    with open(CONFIG_PATH) as f:
        config = {}
        for doc in yaml.safe_load_all(f):
            if isinstance(doc, dict):
                config.update(doc)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(device)
    index_to_category = ckpt["index_to_category"]
    feature_names, _ = load_metadata()
    print("=" * 70)
    print("WARDEN-T TEST EVALUATION")
    print("=" * 70)
    print(f"  checkpoint   : {CHECKPOINT_PATH}")
    print(f"  loaded epoch : {ckpt['epoch']} (val loss {ckpt['val_total_loss']:.4f})")
    print(f"  device       : {device}")

    raw = load_raw_splits()
    states, _ = load_states()
    split_sizes = (len(raw["X_train"]), len(raw["X_val"]), len(raw["X_test"]))

    scaler = FeatureScaler.load(MODELS_DIR / "scaler.npz")
    # Normalised targets (same space as the model output) and raw targets
    # (original feature units) are both needed for the state metrics.
    nxt_norm, _ = next_state_targets(split_sizes, states, scaler=scaler)
    nxt_raw, _ = next_state_targets(split_sizes, states, scaler=None)

    X_test = scaler.transform(raw["X_test"])
    y_test = raw["y_test"]
    y_true_cat = raw["cat_test"]
    true_next_state = nxt_norm[2]
    true_next_state_raw = nxt_raw[2]

    # Category index targets, built with the SAME train+val vocabulary.
    with open(MODELS_DIR / "category_vocabulary.json") as f:
        vocab = json.load(f)
    category_to_index = vocab["category_to_index"]
    oov_index = category_to_index[OOV_CATEGORY]
    y_cat_idx = np.array(
        [category_to_index.get(str(c), oov_index) for c in y_true_cat]
    )

    pred_next, risk_logits, cat_logits, attn = model_predictions(
        model, X_test, device
    )
    risk_prob = 1 / (1 + np.exp(-risk_logits))
    risk_pred = (risk_logits > 0).astype(int)
    cat_pred = cat_logits.argmax(axis=1)
    cat_conf = np.exp(cat_logits - cat_logits.max(axis=1, keepdims=True))
    cat_conf = cat_conf / cat_conf.sum(axis=1, keepdims=True)
    pred_conf = cat_conf.max(axis=1)

    # ------------------------------------------------------------------ risk
    risk_metrics = {
        "accuracy": float(accuracy_score(y_test, risk_pred)),
        "precision": float(precision_score(y_test, risk_pred, zero_division=0)),
        "recall": float(recall_score(y_test, risk_pred, zero_division=0)),
        "f1": float(f1_score(y_test, risk_pred, zero_division=0)),
        "roc_auc": safe_auc(y_test, risk_prob),
        "pr_auc": safe_pr_auc(y_test, risk_prob),
        "n_test": int(len(y_test)),
        "n_attack_true": int(y_test.sum()),
        "n_attack_pred": int(risk_pred.sum()),
        "confusion_matrix": confusion_matrix(y_test, risk_pred).tolist(),
        "confusion_matrix_labels": ["BENIGN(0)", "ATTACK(1)"],
    }

    # -------------------------------------------------------------- category
    present = sorted(set(y_cat_idx.tolist()))
    class_names = [index_to_category[i] for i in present]
    cat_metrics = {
        "accuracy": float(accuracy_score(y_cat_idx, cat_pred)),
        "macro_f1": float(
            f1_score(y_cat_idx, cat_pred, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_cat_idx, cat_pred, average="weighted", zero_division=0)
        ),
        "n_test": int(len(y_cat_idx)),
        "classes_present_in_test": class_names,
        "oov_count": int((y_cat_idx == oov_index).sum()),
    }
    report = classification_report(
        y_cat_idx,
        cat_pred,
        labels=present,
        target_names=class_names,
        zero_division=0,
        digits=4,
    )

    # Restrict to categories the model could actually have learned.
    train_cats = set(np.unique(raw["cat_train"]).tolist())
    seen_mask = np.array([str(c) in train_cats for c in y_true_cat])
    seen_metrics = None
    if seen_mask.sum() > 0:
        seen_metrics = {
            "n_test_seen_categories": int(seen_mask.sum()),
            "accuracy": float(
                accuracy_score(y_cat_idx[seen_mask], cat_pred[seen_mask])
            ),
            "macro_f1": float(
                f1_score(
                    y_cat_idx[seen_mask],
                    cat_pred[seen_mask],
                    average="macro",
                    zero_division=0,
                )
            ),
        }

    # ------------------------------------------------------------ next state
    pred_next_raw = scaler.inverse_transform(pred_next)
    true_next_raw = true_next_state_raw
    state_metrics = {
        "mse_normalised": float(np.mean((pred_next - true_next_state) ** 2)),
        "mae_normalised": float(np.mean(np.abs(pred_next - true_next_state))),
        "mse_raw_units": float(np.mean((pred_next_raw - true_next_raw) ** 2)),
        "mae_raw_units": float(np.mean(np.abs(pred_next_raw - true_next_raw))),
        "note": (
            "Normalised metrics are in z-scored space and are the scale the "
            "loss operates on. Raw-unit metrics are in original feature units "
            "and are not directly comparable across features of different "
            "magnitude; per-feature values are written alongside."
        ),
    }
    per_feature = {
        feature_names[i]: {
            "mse": float(np.mean((pred_next_raw[:, i] - true_next_raw[:, i]) ** 2)),
            "mae": float(np.mean(np.abs(pred_next_raw[:, i] - true_next_raw[:, i]))),
        }
        for i in range(len(feature_names))
    }
    state_metrics["per_feature"] = per_feature

    # -------------------------------------------- persistence baseline
    # "Next window = current window's label" - a trivial but strong baseline
    # since attacks last many windows. Computed from the test sequences.
    persistence_preds = []
    for i in range(len(X_test)):
        last_window_attack = y_test[i - 1] if i > 0 else 0
        persistence_preds.append(int(last_window_attack))
    persistence_preds = np.array(persistence_preds)
    persistence_prob = persistence_preds.astype(float)

    persistence_metrics = {
        "model": "Persistence (next window = current window's label)",
        "accuracy": float(accuracy_score(y_test, persistence_preds)),
        "precision": float(precision_score(y_test, persistence_preds, zero_division=0)),
        "recall": float(recall_score(y_test, persistence_preds, zero_division=0)),
        "f1": float(f1_score(y_test, persistence_preds, zero_division=0)),
        "roc_auc": safe_auc(y_test, persistence_prob),
        "fpr": float(np.sum((persistence_preds == 1) & (y_test == 0)) / max(np.sum(y_test == 0), 1)),
        "n_test": int(len(y_test)),
    }

    # -------------------------------------------- onset detection / lead time
    # On benign->attack transitions, how many windows before onset did risk
    # cross the threshold? This measures forecasting vs classification.
    onset_results = []
    threshold = 0.5
    for i in range(1, len(y_test)):
        if y_test[i] == 1 and y_test[i - 1] == 0:
            lead_times = []
            for j in range(max(0, i - 10), i):
                if risk_prob[j] >= threshold:
                    lead_times.append(i - j)
            onset_results.append({
                "onset_index": i,
                "detected": len(lead_times) > 0,
                "lead_time_windows": min(lead_times) if lead_times else None,
            })

    n_onsets = len(onset_results)
    n_detected = sum(1 for r in onset_results if r["detected"])
    lead_times = [r["lead_time_windows"] for r in onset_results if r["lead_time_windows"] is not None]

    onset_metrics = {
        "n_onsets": n_onsets,
        "n_detected": n_detected,
        "recall": n_detected / n_onsets if n_onsets > 0 else None,
        "mean_lead_time_windows": float(np.mean(lead_times)) if lead_times else None,
        "max_lead_time_windows": int(max(lead_times)) if lead_times else None,
        "threshold": threshold,
        "details": onset_results,
    }

    # -------------------------------------------- calibration
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_test, risk_prob, n_bins=10, strategy="uniform")
    calibration_data = {
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
    }

    # Threshold analysis at multiple operating points
    thresholds = [0.3, 0.5, 0.7, 0.9]
    threshold_analysis = {}
    for t in thresholds:
        pred_t = (risk_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred_t).ravel()
        threshold_analysis[str(t)] = {
            "threshold": t,
            "precision": float(precision_score(y_test, pred_t, zero_division=0)),
            "recall": float(recall_score(y_test, pred_t, zero_division=0)),
            "f1": float(f1_score(y_test, pred_t, zero_division=0)),
            "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            "tpr": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        }

    # ------------------------------------------------------------- baseline
    print("\n[1/4] Training logistic-regression baseline on the same data ...")
    # Flatten [10, 81] -> 810 features, as required by the problem statement.
    Xtr_flat = raw["X_train"].reshape(len(raw["X_train"]), -1)
    Xte_flat = raw["X_test"].reshape(len(raw["X_test"]), -1)
    print(f"      flattened train: {Xtr_flat.shape}  test: {Xte_flat.shape}")

    flat_scaler = StandardScaler()
    Xtr_scaled = flat_scaler.fit_transform(Xtr_flat)
    Xte_scaled = flat_scaler.transform(Xte_flat)

    baseline = LogisticRegression(
        max_iter=2000, C=1.0, class_weight="balanced", random_state=42
    )
    baseline.fit(Xtr_scaled, raw["y_train"])
    bl_prob = baseline.predict_proba(Xte_scaled)[:, 1]
    bl_pred = (bl_prob >= 0.5).astype(int)

    baseline_metrics = {
        "model": "LogisticRegression",
        "features": "flattened [10, 81] -> 810",
        "params": {
            "max_iter": 2000,
            "C": 1.0,
            "class_weight": "balanced",
        },
        "accuracy": float(accuracy_score(y_test, bl_pred)),
        "precision": float(precision_score(y_test, bl_pred, zero_division=0)),
        "recall": float(recall_score(y_test, bl_pred, zero_division=0)),
        "f1": float(f1_score(y_test, bl_pred, zero_division=0)),
        "roc_auc": safe_auc(y_test, bl_prob),
        "pr_auc": safe_pr_auc(y_test, bl_prob),
        "confusion_matrix": confusion_matrix(y_test, bl_pred).tolist(),
        "n_test": int(len(y_test)),
    }

    # -------------------------------------------- integrated gradients
    # Approximate Integrated Gradients using gradient x input as a fast
    # alternative (GradientSHAP-style). For each feature, compute the
    # gradient of the risk logit w.r.t. the input, multiply by the input
    # value, and average across timesteps.
    ig_importance = {}
    try:
        model.eval()
        x_ig = torch.from_numpy(np.ascontiguousarray(X_test[:100], dtype=np.float32)).to(device)
        x_ig.requires_grad_(True)
        out_ig = model(x_ig)
        risk_logits_ig = out_ig["risk_logit"].sum()
        risk_logits_ig.backward()
        ig_grads = x_ig.grad.abs().mean(dim=(0, 1)).cpu().numpy()
        for i, fname in enumerate(feature_names):
            ig_importance[fname] = float(ig_grads[i])
    except Exception as e:
        ig_importance = {"error": str(e)}

    # ---------------------------------------------------------------- write
    test_metrics = {
        "source": "python -m src.models.evaluate",
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_epoch": ckpt["epoch"],
        "test_split_size": int(len(y_test)),
        "risk": risk_metrics,
        "category": cat_metrics,
        "category_seen_only": seen_metrics,
        "next_state": state_metrics,
        "persistence_baseline": persistence_metrics,
        "onset_detection": onset_metrics,
        "calibration": calibration_data,
        "threshold_analysis": threshold_analysis,
        "integrated_gradients": ig_importance,
        "attention_mean_row0_layer0": attn.mean(axis=0).tolist(),
    }

    with open(MODELS_DIR / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    with open(MODELS_DIR / "category_report.txt", "w") as f:
        f.write("WARDEN-T attack-category classification report (TEST split)\n")
        f.write("=" * 70 + "\n")
        f.write(f"test samples: {len(y_cat_idx)}\n")
        f.write(
            f"categories in vocabulary: {len(index_to_category)} "
            "(derived from train+val only)\n"
        )
        f.write(f"OOV targets in test: {cat_metrics['oov_count']}\n\n")
        f.write(report)
        f.write("\n" + "-" * 70 + "\n")
        f.write("RESTRICTED TO CATEGORIES SEEN IN TRAINING\n")
        f.write("-" * 70 + "\n")
        f.write(
            "The temporal split crosses the five CIC-IDS2017 capture days, and\n"
            "each day carries a different attack family. Most test categories\n"
            "were never observed during training and cannot be predicted.\n\n"
        )
        f.write(json.dumps(seen_metrics, indent=2))
        f.write("\n")
    with open(MODELS_DIR / "baseline_metrics.json", "w") as f:
        json.dump(baseline_metrics, f, indent=2)

    # Confusion matrix plot
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cm = confusion_matrix(y_test, risk_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set_xticks([0, 1], ["Predicted BENIGN", "Predicted ATTACK"])
        ax.set_yticks([0, 1], ["True BENIGN", "True ATTACK"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(
            f"WARDEN-T attack-risk confusion matrix (TEST, F1={risk_metrics['f1']:.4f})"
        )
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]}\n({cm[i, j] / cm.sum() * 100:.1f}%)",
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
        fig.tight_layout()
        fig.savefig(MODELS_DIR / "confusion_matrix.png", dpi=150)
        plt.close(fig)
        print("\n[3/4] Wrote models/confusion_matrix.png")
    except Exception as exc:  # plotting must never fail the evaluation
        print(f"\n[3/4] Skipped confusion-matrix plot: {exc}")

    # --------------------------------------------------------- comparison
    lines = [
        "=" * 70,
        "ATTACK-RISK TASK - TEST SPLIT COMPARISON",
        "=" * 70,
        f"{'Metric':<14}{'Logistic Regression':>24}{'WARDEN-T':>16}",
        "-" * 70,
    ]
    for key, label in (
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("roc_auc", "ROC-AUC"),
        ("pr_auc", "PR-AUC"),
    ):
        b = baseline_metrics[key]
        t = risk_metrics[key]
        bs = f"{b:.4f}" if b is not None else "n/a"
        ts = f"{t:.4f}" if t is not None else "n/a"
        lines.append(f"{label:<14}{bs:>24}{ts:>16}")
    lines.append("-" * 70)
    lines.append(
        "Both models are evaluated on the identical test split. No ranking or\n"
        "superiority claim is made; the numbers are reported as measured."
    )
    comparison = "\n".join(lines)
    with open(MODELS_DIR / "comparison.txt", "w") as f:
        f.write(comparison + "\n")

    print("\n" + comparison)

    print("\n[2/4] Risk metrics (TEST)")
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
        v = risk_metrics[k]
        print(f"  {k:<12}: {v if v is None else round(v, 4)}")
    print(f"  confusion   : {risk_metrics['confusion_matrix']}  (rows=true, cols=pred)")

    print("\n[3/4] Category metrics (TEST)")
    print(f"  accuracy    : {cat_metrics['accuracy']:.4f}")
    print(f"  macro F1    : {cat_metrics['macro_f1']:.4f}")
    print(f"  weighted F1 : {cat_metrics['weighted_f1']:.4f}")
    print(f"  OOV targets : {cat_metrics['oov_count']}")
    if seen_metrics:
        print(
            f"  seen-only   : acc={seen_metrics['accuracy']:.4f} "
            f"macroF1={seen_metrics['macro_f1']:.4f} "
            f"(n={seen_metrics['n_test_seen_categories']})"
        )

    print("\n[4/4] Next-state metrics (TEST)")
    print(f"  MSE (z-space): {state_metrics['mse_normalised']:.4f}")
    print(f"  MAE (z-space): {state_metrics['mae_normalised']:.4f}")
    print(f"  MSE (raw)    : {state_metrics['mse_raw_units']:.6g}")
    print(f"  MAE (raw)    : {state_metrics['mae_raw_units']:.6g}")

    print("\n  Persistence baseline (TEST):")
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "fpr"):
        v = persistence_metrics.get(k)
        print(f"    {k:<12}: {v if v is None else round(v, 4)}")

    print("\n  Onset detection (TEST):")
    print(f"    onsets       : {onset_metrics['n_onsets']}")
    print(f"    detected     : {onset_metrics['n_detected']}")
    print(f"    recall       : {onset_metrics['recall']}")
    print(f"    mean lead    : {onset_metrics['mean_lead_time_windows']} windows")
    print(f"    max lead     : {onset_metrics['max_lead_time_windows']} windows")

    print("\nSaved: models/test_metrics.json, category_report.txt, "
          "confusion_matrix.png, baseline_metrics.json, comparison.txt")
    print("\nCategory classification report (TEST):\n")
    print(report)


if __name__ == "__main__":
    sys.exit(main())
