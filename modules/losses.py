"""Vessel-weighted reconstruction losses in latent space (Frangi weights)."""
from __future__ import annotations

import torch


def frangi_weighted_l1(pred, target, vessel, alpha: float):
    return ((1.0 + alpha * vessel) * (pred - target).abs()).mean()


def frangi_weighted_mip_l1(pred, target, vessel, alpha: float):
    """MIP along each axis, weighted by the vesselness projected the same way. Continuity
    is judged on MIPs, so this is where a broken vessel actually shows."""
    weight = 1.0 + alpha * vessel
    return torch.stack([(weight.amax(dim=axis) *
                         (pred.amax(dim=axis) - target.amax(dim=axis)).abs()).mean()
                        for axis in (-1, -2, -3)]).mean()


def vessel_weighted_gradient_l1(pred, target, vessel, alpha: float):
    weight = 1.0 + alpha * vessel
    total = pred.new_tensor(0.0)
    for axis in (-3, -2, -1):
        if pred.shape[axis] < 2:
            continue
        d_pred = torch.diff(pred, dim=axis)
        d_target = torch.diff(target, dim=axis)
        w = weight.narrow(axis, 1, weight.shape[axis] - 1)
        total = total + (w * (d_pred - d_target).abs()).mean()
    return total
