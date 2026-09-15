"""Differentiable topology terms: soft skeleton, clDice, soft Dice and soft vessel membership."""
from __future__ import annotations

import torch
import torch.nn.functional as F


CLDICE_ITERS = 3           # soft-skeletonisation iterations (Shit et al. 2021)


def soft_erode(x: torch.Tensor) -> torch.Tensor:
    """Greyscale erosion with a separable 3-D cross, as in the reference implementation."""
    p1 = -F.max_pool3d(-x, (3, 1, 1), (1, 1, 1), (1, 0, 0))
    p2 = -F.max_pool3d(-x, (1, 3, 1), (1, 1, 1), (0, 1, 0))
    p3 = -F.max_pool3d(-x, (1, 1, 3), (1, 1, 1), (0, 0, 1))
    return torch.min(torch.min(p1, p2), p3)


def soft_dilate(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool3d(x, (3, 3, 3), (1, 1, 1), (1, 1, 1))


def soft_open(x: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(x))


def soft_skeletonize(x: torch.Tensor, iterations: int = CLDICE_ITERS) -> torch.Tensor:
    opened = soft_open(x)
    skeleton = F.relu(x - opened)
    for _ in range(iterations):
        x = soft_erode(x)
        opened = soft_open(x)
        delta = F.relu(x - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """1 - clDice. Harmonic mean of topology precision and sensitivity, no smoothing of
    the inputs: the skeleton is the only place structure is extracted."""
    skeleton_pred = soft_skeletonize(pred)
    skeleton_true = soft_skeletonize(target)
    precision = ((skeleton_pred * target).sum() + smooth) / (skeleton_pred.sum() + smooth)
    sensitivity = ((skeleton_true * pred).sum() + smooth) / (skeleton_true.sum() + smooth)
    return 1.0 - (2.0 * precision * sensitivity) / (precision + sensitivity + 1e-8)


def vessel_probability_percentile(volume: torch.Tensor, p99: torch.Tensor, p999: torch.Tensor) -> torch.Tensor:
    """Soft vessel membership on the per-volume band -- the same ramp `vessel_weight_map`
    uses, and the same band `vessel_scores` thresholds, so the loss and the metric agree."""
    lo = p99.float().view(-1, 1, 1, 1, 1)
    hi = torch.clamp(p999.float().view(-1, 1, 1, 1, 1), min=lo + 1e-4)
    return ((volume.float() - lo) / (hi - lo)).clamp(0.0, 1.0)


def vessel_probability(volume: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Soft vessel membership from the training-set intensity band. Fixed thresholds, not
    per-case percentiles, because the decoded crop is too small for a stable p99."""
    return ((volume.float() - lo) / max(hi - lo, 1e-4)).clamp(0.0, 1.0)


def vessel_probability_banded(volume: torch.Tensor, band: torch.Tensor) -> torch.Tensor:
    """Soft vessel membership on a PER-SAMPLE band. `band` is (B, 2) holding (lo, hi) --
    the case's own whole-volume (p98, p99.5), the support vessel_scores thresholds on."""
    lo = band[:, 0].view(-1, 1, 1, 1, 1).float()
    hi = torch.clamp(band[:, 1].view(-1, 1, 1, 1, 1).float(), min=lo + 1e-4)
    return ((volume.float() - lo) / (hi - lo)).clamp(0.0, 1.0)


def soft_dice_loss(pred: torch.Tensor, target: torch.Tensor,
                   smooth: float = 1.0) -> torch.Tensor:
    """1 - soft Dice on the vessel probability maps. This is the term that constrains
    EXTENT: pred.sum() sits in the denominator, so a prediction that is too thick is
    penalised directly. clDice cannot do this -- see the module docstring."""
    num = 2.0 * (pred * target).sum() + smooth
    den = pred.sum() + target.sum() + smooth
    return 1.0 - num / den
