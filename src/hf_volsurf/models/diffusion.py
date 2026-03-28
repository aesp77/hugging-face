"""DDPM generative model for vol surfaces.

Extracted from notebooks/05_diffusion_vol.ipynb.
Generates realistic implied volatility surfaces from noise.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from diffusers import DDPMScheduler, UNet2DModel

logger = logging.getLogger(__name__)


@dataclass
class DDPMConfig:
    """Configuration for VolSurfaceDDPM."""

    block_out_channels: tuple[int, ...] = (32, 64)
    layers_per_block: int = 1
    num_train_timesteps: int = 1000
    beta_schedule: str = "squaredcos_cap_v2"
    lr: float = 1e-3
    epochs: int = 30
    batch_size: int = 64


class VolSurfaceDDPM:
    """DDPM wrapper for vol surface generation.

    Handles padding (13→16 strikes), normalisation, and the
    full denoising loop.
    """

    def __init__(
        self,
        config: DDPMConfig | None = None,
        device: str | None = None,
    ):
        cfg = config or DDPMConfig()
        self.config = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = UNet2DModel(
            sample_size=(8, 16),
            in_channels=1,
            out_channels=1,
            layers_per_block=cfg.layers_per_block,
            block_out_channels=cfg.block_out_channels,
            down_block_types=("DownBlock2D",) * len(cfg.block_out_channels),
            up_block_types=("UpBlock2D",) * len(cfg.block_out_channels),
        ).to(self.device)

        self.scheduler = DDPMScheduler(
            num_train_timesteps=cfg.num_train_timesteps,
            beta_schedule=cfg.beta_schedule,
        )

        self.grid_min: float = 0.0
        self.grid_max: float = 1.0

    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def load_weights(self, path: Path) -> None:
        """Load trained model weights."""
        self.model.load_state_dict(
            torch.load(str(path), map_location=self.device, weights_only=True)
        )
        self.model.eval()
        logger.info("Loaded DDPM weights from %s", path)

    def save_weights(self, path: Path) -> None:
        """Save model weights."""
        torch.save(self.model.state_dict(), str(path))
        logger.info("Saved DDPM weights to %s", path)

    def set_normalization(self, grid_min: float, grid_max: float) -> None:
        """Set min-max normalization range (from training data)."""
        self.grid_min = grid_min
        self.grid_max = grid_max

    def prepare_training_data(
        self, grids: np.ndarray
    ) -> torch.Tensor:
        """Normalise and pad grids for training.

        Args:
            grids: (N, 8, 13) array of IV values.

        Returns:
            Tensor of shape (N, 1, 8, 16).
        """
        self.grid_min = float(grids.min())
        self.grid_max = float(grids.max())
        normed = (grids - self.grid_min) / (self.grid_max - self.grid_min)
        padded = np.pad(normed, ((0, 0), (0, 0), (0, 3)), mode="edge")
        return torch.tensor(padded, dtype=torch.float32).unsqueeze(1)

    @torch.no_grad()
    def generate(self, n_samples: int = 10) -> np.ndarray:
        """Generate vol surfaces from noise.

        Returns:
            (n_samples, 8, 13) array of IV values (denormalised).
        """
        self.model.eval()
        sample = torch.randn(n_samples, 1, 8, 16).to(self.device)
        self.scheduler.set_timesteps(self.config.num_train_timesteps)

        for t in self.scheduler.timesteps:
            pred = self.model(sample, t).sample
            sample = self.scheduler.step(pred, t, sample).prev_sample

        out = sample.cpu().numpy()[:, 0, :, :13]  # strip padding
        return out * (self.grid_max - self.grid_min) + self.grid_min
