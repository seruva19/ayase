"""Minimal tensor helpers used by the official FVMD PIPs++ runtime.

Modified for Ayase from upstream ``fvmd/utils/basic.py``: unrelated training,
visualization, and file helpers were omitted, and device selection is inferred
from the caller instead of defaulting to CUDA.
"""

import torch

EPS = 1e-6


def reduce_masked_mean(x, mask, dim=None, keepdim=False):
    """Return the upstream masked mean used by the PIPs++ training path."""
    for a, b in zip(x.size(), mask.size()):
        assert a == b
    prod = x * mask
    if dim is None:
        numer = torch.sum(prod)
        denom = EPS + torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = EPS + torch.sum(mask, dim=dim, keepdim=keepdim)
    return numer / denom


def meshgrid2d(batch, height, width, stack=False, norm=False, device=None, on_chans=False):
    """Create a batched 2-D coordinate grid in the upstream coordinate order."""
    grid_y = torch.linspace(0.0, height - 1, height, device=device).reshape(1, height, 1)
    grid_y = grid_y.repeat(batch, 1, width)
    grid_x = torch.linspace(0.0, width - 1, width, device=device).reshape(1, 1, width)
    grid_x = grid_x.repeat(batch, height, 1)

    if norm:
        grid_y = 2.0 * (grid_y / float(height - 1)) - 1.0
        grid_x = 2.0 * (grid_x / float(width - 1)) - 1.0
        grid_y = torch.clamp(grid_y, min=-2.0, max=2.0)
        grid_x = torch.clamp(grid_x, min=-2.0, max=2.0)

    if stack:
        dim = 1 if on_chans else -1
        return torch.stack([grid_x, grid_y], dim=dim)
    return grid_y, grid_x
