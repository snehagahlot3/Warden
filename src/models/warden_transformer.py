# -*- coding: utf-8 -*-
"""
WARDEN-T: Temporal Transformer encoder with three prediction heads.

ARCHITECTURE
============
    S(t-9) ... S(t)                      input   [batch, 10, 81]
        |
        v
    Feature Projection  Linear(81 -> 64)          [batch, 10, 64]
        |
        v
    Positional Encoding  (sinusoidal, fixed buffer)
        |
        v
    Temporal Transformer Encoder
        2 layers x 4 heads, feedforward 128, dropout 0.1, pre-norm FFN
        |
        v
    Temporal Representation  z = enc[:, -1]        [batch, 64]
        |
        +--> HEAD 1  Next-state regression   Linear(64 -> 81)   -> S(t+1)
        +--> HEAD 2  Attack risk             Linear(64 -> 1)    -> logit P(attack)
        +--> HEAD 3  Attack category         Linear(64 -> C)    -> C category logits

NOTES ON IMPLEMENTATION CHOICES
-------------------------------
* `nn.TransformerEncoder` does not expose per-layer attention weights, so the
  encoder stack is written out explicitly using `nn.MultiheadAttention`. This
  keeps `src/explainability/attention.py` able to return real attention maps
  instead of approximating them.
* The temporal representation is the encoding of the most recent state
  (position t). A mean-pool variant is available via `pooling="mean"` for
  ablation, but "last" is the default because the forecasting objective is
  conditioned on the current state.
* Every head reads the same representation, so the three tasks share one
  temporal encoder (multi-task learning).
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.

    The temporal order of the 10 windows carries the entire signal for a
    forecasting task, so position must be injected explicitly: attention alone
    is permutation invariant over the sequence axis.
    """

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class TemporalEncoderLayer(nn.Module):
    """
    Pre-norm transformer encoder block.

    self-attention over the temporal axis -> position-wise feedforward,
    each with a residual connection. Kept explicit (rather than using
    nn.TransformerEncoderLayer) so that the attention probabilities are
    available as a return value for explainability.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x, need_weights=False):
        attn_out, attn_weights = self.self_attn(
            x, x, x, need_weights=need_weights, average_attn_weights=True
        )
        x = self.norm1(x + self.dropout1(attn_out))
        ff = self.linear2(self.dropout1(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout2(ff))
        if need_weights:
            return x, attn_weights
        return x, None


class WardenTransformer(nn.Module):
    """
    WARDEN-T - Temporal Transformer for multi-step network attack forecasting.

    Parameters
    ----------
    input_dim      : 81   features per network state S_t
    output_dim     : 81   features predicted for S(t+1)
    d_model        : 64   transformer width
    nhead          : 4    attention heads
    num_layers     : 2    encoder layers
    dim_feedforward: 128   position-wise FFN width
    dropout        : 0.1
    num_categories : C     attack-category count from the data-derived vocab
    max_len        : 64   maximum supported sequence length (rollout safety)
    """

    def __init__(
        self,
        input_dim=81,
        output_dim=81,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        num_categories=18,
        max_len=64,
        pooling="last",
        unlearnable_categories=None,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.num_categories = num_categories
        self.max_len = max_len
        self.pooling = pooling

        # ---- Feature projection 81 -> 64 -------------------------------
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.input_norm = nn.LayerNorm(d_model)

        # ---- Temporal transformer encoder ------------------------------
        self.layers = nn.ModuleList(
            [
                TemporalEncoderLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )

        # ---- Heads -----------------------------------------------------
        # HEAD 1: next network state (regression, no activation - the target
        #         is a z-scored feature vector)
        self.state_head = nn.Linear(d_model, output_dim)
        # HEAD 2: attack risk (single logit, sigmoid applied outside the loss
        #         via BCEWithLogitsLoss for numerical stability)
        self.risk_head = nn.Linear(d_model, 1)
        # HEAD 3: attack category (softmax applied outside via CrossEntropyLoss)
        self.category_head = nn.Linear(d_model, num_categories)
        # HEAD 4: multi-horizon risk (K logits for t+1..t+K)
        self.multi_horizon_risk_head = nn.Linear(d_model, 5)

        # Categories with zero training support are masked out of the softmax.
        # They are unreachable by construction (see src/models/losses.py), and
        # masking them in the model keeps training and inference consistent.
        if unlearnable_categories is None:
            unlearnable = torch.zeros(num_categories, dtype=torch.bool)
        else:
            unlearnable = torch.as_tensor(
                unlearnable_categories, dtype=torch.bool
            ).clone()
        self.register_buffer("unlearnable", unlearnable, persistent=True)

    def masked_category_logits(self, logits):
        """Remove unlearnable categories from the softmax."""
        if bool(self.unlearnable.any()):
            logits = logits.masked_fill(self.unlearnable, -1e9)
        return logits

    # ------------------------------------------------------------------
    def encode(self, x, need_weights=False):
        """
        Run the feature projection + positional encoding + transformer stack.

        Returns
        -------
        encoded : [batch, T, d_model]
        attn    : list of [batch, T, T] per layer, or None
        """
        if x.dim() != 3:
            raise ValueError(f"expected [batch, T, F], got {tuple(x.shape)}")
        if x.size(1) > self.max_len:
            raise ValueError(
                f"sequence length {x.size(1)} exceeds max_len {self.max_len}"
            )

        h = self.input_projection(x)          # 81 -> 64
        h = self.input_norm(h)
        h = self.positional_encoding(h)

        attn_list = []
        for layer in self.layers:
            h, attn = layer(h, need_weights=need_weights)
            if need_weights:
                attn_list.append(attn)
        return h, (attn_list if need_weights else None)

    def pool(self, encoded):
        """
        Reduce the [batch, T, d_model] sequence to [batch, d_model].

        "last" takes the representation of the most recent observed state
        S(t); "mean" averages over the whole observation window.
        """
        if self.pooling == "last":
            return encoded[:, -1]
        if self.pooling == "mean":
            return encoded.mean(dim=1)
        raise ValueError(f"unknown pooling mode: {self.pooling}")

    # ------------------------------------------------------------------
    def forward(self, x, need_weights=False):
        """
        Returns a dict with
          next_state : [batch, 88]
          risk_logit : [batch, 1]  (kept 2-D to match the [batch, 1] targets
                                   consumed by BCEWithLogitsLoss)
          category_logits : [batch, C]
          multi_horizon_risk_logits : [batch, 5]  (t+1..t+5)
          representation : [batch, 64]
          attention : list of [batch, T, T] or None
        """
        encoded, attn = self.encode(x, need_weights=need_weights)
        z = self.pool(encoded)
        return {
            "next_state": self.state_head(z),
            "risk_logit": self.risk_head(z),
            "category_logits": self.masked_category_logits(self.category_head(z)),
            "multi_horizon_risk_logits": self.multi_horizon_risk_head(z),
            "representation": z,
            "attention": attn,
        }

    @torch.no_grad()
    def predict_step(self, x):
        """Convenience wrapper returning sigmoid risk + softmax category probs."""
        out = self.forward(x)
        return {
            "next_state": out["next_state"],
            "risk_prob": torch.sigmoid(out["risk_logit"]),
            "category_prob": torch.softmax(out["category_logits"], dim=-1),
        }

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg, num_categories, unlearnable_categories=None):
    """Instantiate WARDEN-T from the `model` section of configs/config.yaml."""
    return WardenTransformer(
        input_dim=cfg["input_dim"],
        output_dim=cfg.get("output_dim", cfg["input_dim"]),
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
        num_categories=num_categories,
        max_len=cfg.get("max_len", 64),
        pooling=cfg.get("pooling", "last"),
        unlearnable_categories=unlearnable_categories,
    )
