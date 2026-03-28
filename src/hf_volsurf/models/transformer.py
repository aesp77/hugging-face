"""Transformer model for vol surface prediction.

Extracted from notebooks/04_transformer_vol.ipynb.
Predicts next day's vol surface from a lookback window.
"""

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset

logger = logging.getLogger(__name__)

SURFACE_DIM = 8 * 13  # 104


@dataclass
class TransformerConfig:
    """Configuration for VolSurfaceTransformer."""

    d_model: int = SURFACE_DIM  # 104
    nhead: int = 4
    num_layers: int = 2
    dim_ff: int = 256
    lookback: int = 20
    dropout: float = 0.1


class VolSurfaceTransformer(nn.Module):
    """Transformer encoder for vol surface prediction.

    Input: (batch, lookback, 104) — sequence of flattened surfaces.
    Output: (batch, 104) — predicted next surface.
    """

    def __init__(self, config: TransformerConfig | None = None):
        super().__init__()
        cfg = config or TransformerConfig()
        self.config = cfg
        self.pos_embedding = nn.Parameter(
            torch.randn(1, cfg.lookback, cfg.d_model) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_ff,
            batch_first=True,
            dropout=cfg.dropout,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.num_layers
        )
        self.head = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, input_ids=None, labels=None, **kwargs):
        x = input_ids + self.pos_embedding
        x = self.encoder(x)
        pred = self.head(x[:, -1, :])

        loss = None
        if labels is not None:
            loss = nn.functional.mse_loss(pred, labels)
        return {"loss": loss, "logits": pred}


class VolSurfaceWindowDataset(TorchDataset):
    """Windowed dataset for vol surface prediction.

    Each sample: input = LOOKBACK consecutive surfaces, target = next surface.
    """

    def __init__(self, surfaces: np.ndarray, lookback: int = 20):
        self.surfaces = surfaces
        self.lookback = lookback
        self.n_samples = len(surfaces) - lookback

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        window = self.surfaces[idx : idx + self.lookback]
        target = self.surfaces[idx + self.lookback]
        return {
            "input_ids": torch.tensor(
                window.reshape(self.lookback, SURFACE_DIM), dtype=torch.float32
            ),
            "labels": torch.tensor(
                target.reshape(SURFACE_DIM), dtype=torch.float32
            ),
        }
