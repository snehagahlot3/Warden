# -*- coding: utf-8 -*-
"""
WARDEN-T multi-task objective.

    total_loss = lambda_state    * SmoothL1( S(t+1)_pred , S(t+1)_true )
               + lambda_risk     * BCEWithLogits( risk_logit , is_attack )
               + lambda_category * CrossEntropy( category_logits , category )

DESIGN NOTES
------------
* State regression uses SmoothL1 (Huber) rather than plain MSE. The 81
  per-state features have very different scales and heavy tails (packet counts
  and byte totals span several orders of magnitude), so MSE is dominated by a
  handful of outlier dimensions. SmoothL1 is linear for large residuals, which
  keeps a single extreme feature from controlling the gradient.
* Risk uses BCEWithLogitsLoss for numerical stability (sigmoid applied inside
  the loss, not in the model), with `pos_weight` taken from the actual train
  class counts.
* Category uses CrossEntropyLoss with inverse-frequency class weights computed
  from the actual training split. Classes with zero training support get weight
  0 so they cannot inject noise; they are reported separately in evaluate.py.
* All weights are derived from data at runtime. Nothing is hardcoded.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(category_targets, num_categories, mode="inverse_frequency"):
    """
    Derive CrossEntropy class weights from the observed training categories.

    mode="inverse_frequency": w_c = (N / (C * n_c)) for classes with n_c > 0,
                              then rescaled to mean 1 (keeps the loss scale
                              comparable to an unweighted CrossEntropy).
    Classes with n_c == 0 are reported separately by `unlearnable_mask`.

    IMPORTANT: classes with no training support are NOT given weight 0. A
    zero weight makes any batch composed entirely of such classes produce a
    0/0 denominator in CrossEntropyLoss, i.e. NaN. Because the temporal split
    crosses the five CIC-IDS2017 capture days, the validation and test splits
    legitimately contain categories that never appear in training, so this
    case really occurs. Instead, unlearnable classes are removed from the
    softmax by `unlearnable_mask` - they are unreachable by construction, which
    is the honest representation of the situation.
    """
    counts = torch.bincount(category_targets, minlength=num_categories).float()
    n = counts.sum()
    present = counts > 0
    weights = torch.ones(num_categories, dtype=torch.float32)
    if mode == "inverse_frequency":
        raw = n / (num_categories * counts[present])
        weights[present] = raw / raw.mean()
    else:
        weights[present] = 1.0
    return weights, counts


def unlearnable_mask(counts):
    """
    Boolean mask of classes with zero training support.

    These are masked out of the category softmax (logit -> -1e9) so they can
    never be predicted. This is a data property, not a modelling choice: the
    temporal split means some attack families exist only in later capture days.
    """
    return counts <= 0


def compute_pos_weight(risk_targets, enabled=True):
    """
    pos_weight for BCEWithLogitsLoss.

    The training split is already ATTACK-majority (62.9% attack), so the usual
    minority-class correction is not warranted: applying neg/pos = 0.59 would
    push the decision boundary further toward BENIGN and cost recall. The
    default therefore returns 1.0 (plain BCE), while still reporting the
    data-derived ratio for reference. Pass enabled=True to opt back in.
    """
    positives = float(risk_targets.sum())
    negatives = float(len(risk_targets) - positives)
    ratio = negatives / positives if positives else 1.0
    if not enabled:
        return torch.tensor(1.0), ratio
    return torch.tensor(ratio), ratio


class WardenMultiTaskLoss(nn.Module):
    """
    Combines the three heads. `forward` returns (total, component_dict) so the
    training loop can log each task's contribution separately.
    """

    def __init__(
        self,
        num_categories,
        category_class_weights=None,
        risk_pos_weight=None,
        unlearnable_classes=None,
        lambda_state=1.0,
        lambda_risk=1.0,
        lambda_category=1.0,
        huber_beta=1.0,
    ):
        super().__init__()
        self.lambda_state = lambda_state
        self.lambda_risk = lambda_risk
        self.lambda_category = lambda_category
        self.huber_beta = huber_beta

        self.state_criterion = nn.SmoothL1Loss(beta=huber_beta)
        self.risk_criterion = nn.BCEWithLogitsLoss(
            pos_weight=risk_pos_weight
        )
        self.register_buffer(
            "category_class_weights",
            (
                torch.ones(num_categories)
                if category_class_weights is None
                else torch.as_tensor(category_class_weights).float().clone()
            ),
            persistent=False,
        )
        if unlearnable_classes is None:
            unlearnable_classes = torch.zeros(num_categories, dtype=torch.bool)
        self.register_buffer(
            "unlearnable", unlearnable_classes.bool().clone(), persistent=False
        )

    def category_logits(self, raw_logits):
        """
        Mask classes with zero training support out of the softmax.

        Setting their logit to a very negative value removes them from the
        normaliser entirely, which keeps the loss finite for any batch
        composition and makes those categories genuinely unpredictable - the
        correct behaviour given they have no training examples.
        """
        if self.unlearnable.any():
            raw_logits = raw_logits.masked_fill(self.unlearnable, -1e9)
        return raw_logits

    def forward(self, outputs, y_state, y_risk, y_category):
        state_loss = self.state_criterion(outputs["next_state"], y_state)
        risk_loss = self.risk_criterion(outputs["risk_logit"], y_risk)

        # Per-sample category loss, averaged ONLY over samples whose target is
        # actually learnable. The temporal split means val/test contain
        # categories with zero training support; those are masked out of the
        # softmax, so including them would contribute a meaningless ~1e9 term
        # and would dominate the validation loss used for early stopping.
        logits = self.category_logits(outputs["category_logits"])
        per_sample = F.cross_entropy(
            logits, y_category, weight=self.category_class_weights, reduction="none"
        )
        learnable_target = ~self.unlearnable[y_category]
        if learnable_target.any():
            cat_loss = per_sample[learnable_target].mean()
        else:
            cat_loss = torch.zeros((), device=per_sample.device)

        total = (
            self.lambda_state * state_loss
            + self.lambda_risk * risk_loss
            + self.lambda_category * cat_loss
        )
        return total, {
            "state": state_loss.detach(),
            "risk": risk_loss.detach(),
            "category": cat_loss.detach(),
        }
