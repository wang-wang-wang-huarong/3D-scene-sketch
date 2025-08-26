# models/udf.py
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class UDF(nn.Module):
    """Simple unsigned distance field represented by an MLP."""

    def __init__(self, hidden_dim: int = 256, num_layers: int = 8):
        super().__init__()
        layers = []
        in_dim = 3
        for i in range(num_layers):
            out_dim = hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the unsigned distance field at coordinates ``x``.

        Args:
            x: Tensor of shape ``(..., 3)`` with xyz coordinates.
        Returns:
            Tensor with the same batch shape as ``x`` but with one channel
            containing the unsigned distance to the surface.
        """
        return self.mlp(x).squeeze(-1)


class FeatureHead(nn.Module):
    """Small MLP that predicts a feature vector for a 3D point."""

    def __init__(self, feature_dim: int = 32, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        layers = []
        in_dim = 3
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, feature_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def udf_to_density(udf: torch.Tensor, beta: float = 100.0) -> torch.Tensor:
    """Convert unsigned distance values to a soft density field.

    A simple exponential falloff is used which works well for sparse geometry.

    Args:
        udf: Unsigned distance values ``(N, S)``.
        beta: Controls the sharpness of the surface. Higher beta makes the
            surface harder.
    Returns:
        Density tensor with the same shape as ``udf``.
    """
    return torch.exp(-beta * udf.clamp(min=0.0))


def volume_render_rays(
    udf: UDF,
    feat_head: Optional[FeatureHead],
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    num_samples: int = 64,
    beta: float = 100.0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Perform simple volume rendering along rays.

    Args:
        udf: The unsigned distance field model.
        feat_head: Optional feature head used for REPA-style alignment.
        rays_o: Ray origins ``(N, 3)``.
        rays_d: Ray directions ``(N, 3)`` (should be normalised).
        near: Near plane distance.
        far: Far plane distance.
        num_samples: Number of samples along each ray.
        beta: Conversion parameter from distance to density.

    Returns:
        depth: Predicted depth along each ray ``(N,)``.
        features: Rendered feature vector for each ray ``(N, F)`` if
            ``feat_head`` is not ``None`` otherwise ``None``.
    """
    device = rays_o.device
    t_vals = torch.linspace(near, far, num_samples, device=device)
    # (N, S, 3)
    points = rays_o[:, None, :] + rays_d[:, None, :] * t_vals[None, :, None]
    points_flat = points.reshape(-1, 3)
    udf_vals = udf(points_flat).view(points.shape[:-1])
    density = udf_to_density(udf_vals, beta=beta)

    deltas = t_vals[1:] - t_vals[:-1]
    deltas = torch.cat([deltas, deltas[-1:]], dim=0)
    deltas = deltas[None, :].expand_as(density)
    alpha = 1.0 - torch.exp(-density * deltas)
    # Accumulated transmittance
    T = torch.cumprod(
        torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1e-10], dim=1),
        dim=1
    )[:, :-1]
    weights = alpha * T

    depth = torch.sum(weights * t_vals[None, :], dim=-1)

    feat = None
    if feat_head is not None:
        feat_vals = feat_head(points_flat).view(points.shape[0], points.shape[1], -1)
        feat = torch.sum(weights[..., None] * feat_vals, dim=1)
    return depth, feat