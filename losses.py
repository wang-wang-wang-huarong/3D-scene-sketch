# losses.py
from typing import Optional

import torch
import torch.nn.functional as F


def depth_loss(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    confidence: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """L1 depth supervision weighted by confidence."""
    if confidence is None:
        confidence = torch.ones_like(gt_depth)
    return (confidence * (pred_depth - gt_depth).abs()).mean()


def chamfer_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Symmetric Chamfer distance between two point clouds.
    Args:
        pred: (N, 3)
        target: (M, 3)
    """
    dist = torch.cdist(pred, target)
    return dist.min(dim=1)[0].mean() + dist.min(dim=2)[0].mean()


def track_loss(udf_vals: torch.Tensor, delta: float = 0.0) -> torch.Tensor:
    """Penalise UDF values for tracked points towards zero.
    Args:
        udf_vals: UDF evaluated at tracked 3D points
        delta: target small distance (default 0.0)
    """
    return F.huber_loss(udf_vals, udf_vals.new_full(udf_vals.shape, delta))


def feature_alignment_loss(
    pred_feat: torch.Tensor,
    target_feat: torch.Tensor,
    confidence: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """REPA-style feature alignment loss."""
    if confidence is None:
        confidence = torch.ones(pred_feat.shape[0], device=pred_feat.device)
    # reshape confidence for broadcasting
    confidence = confidence[..., None]
    return (confidence * (pred_feat - target_feat) ** 2).mean()