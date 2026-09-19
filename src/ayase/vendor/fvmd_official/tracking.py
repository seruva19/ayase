"""PIPs++ tracking and motion-field extraction for official FVMD.

Modified for Ayase from upstream ``fvmd/keypoint_tracking.py``: command-line,
TensorBoard, Fire, Loguru, DataLoader, visualization, and cache-writing paths
were removed. Model loading and tensor inference remain available as library
functions with package-relative imports.
"""

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch

from .nets.pips2 import Pips
from .utils.basic import meshgrid2d

PIPS_WEIGHTS = (
    "https://github.com/ljh0v0/FVMD-frechet-video-motion-distance/"
    "releases/download/pips2_weights/pips2_weights.pth"
)


def load_pips(
    weights: Union[str, Path] = PIPS_WEIGHTS,
    device: Union[str, torch.device] = "cpu",
    stride: int = 8,
    progress: bool = True,
    trusted_checkpoint: bool = False,
) -> Pips:
    """Load PIPs++; unsafe pickle is opt-in after caller-side hash verification."""
    device = torch.device(device)
    weight_location = str(weights)
    if weight_location.startswith(("https://", "http://")):
        checkpoint = torch.hub.load_state_dict_from_url(
            weight_location, map_location=device, progress=progress
        )
    else:
        try:
            checkpoint = torch.load(
                weight_location,
                map_location=device,
                weights_only=not trusted_checkpoint,
            )
        except TypeError:  # PyTorch before the weights_only argument.
            if not trusted_checkpoint:
                raise RuntimeError(
                    "old PyTorch cannot safely load this checkpoint without explicit trust"
                )
            checkpoint = torch.load(weight_location, map_location=device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model = Pips(stride=stride).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def calc_velocity(trajectories: torch.Tensor) -> torch.Tensor:
    """Compute Eq. (1): first-order positions with one leading zero field."""
    batch, frames, points, dims = trajectories.shape
    if dims != 2:
        raise ValueError("trajectories must have shape (B, S, N, 2)")
    delta = trajectories[:, 1:] - trajectories[:, :-1]
    zero = torch.zeros(batch, 1, points, 2, device=trajectories.device, dtype=trajectories.dtype)
    return torch.cat([zero, delta], dim=1)


def calc_acceleration(velocity: torch.Tensor) -> torch.Tensor:
    """Reproduce the released FVMD 1.0.0 acceleration helper exactly.

    Despite the parameter name, the released evaluator calls this helper with
    trajectories, takes ``input[:, 2:] - input[:, 1:-1]``, and prepends two
    zero fields. This differs from paper Eq. (2), but is retained so scores are
    numerically compatible with the official evaluator.
    """
    batch, frames, points, dims = velocity.shape
    if dims != 2:
        raise ValueError("velocity must have shape (B, S, N, 2)")
    delta = velocity[:, 2:] - velocity[:, 1:-1]
    zeros = torch.zeros(batch, 2, points, 2, device=velocity.device, dtype=velocity.dtype)
    return torch.cat([zeros, delta], dim=1)


def run_tracking(model: Pips, rgbs: torch.Tensor, N: int = 64, iters: int = 16):
    """Track a square grid of points through one RGB clip.

    ``rgbs`` must have shape ``(1, S, 3, H, W)`` and values in ``[0, 255]``.
    """
    rgbs = rgbs.float()
    batch, _frames, _channels, height, width = rgbs.shape
    if batch != 1:
        raise ValueError("official FVMD tracking supports batch size 1")
    side = int(np.sqrt(N).round())
    if side < 2:
        raise ValueError("N must produce a grid with at least two points per side")

    grid_y, grid_x = meshgrid2d(batch, side, side, device=rgbs.device)
    grid_y = 8 + grid_y.reshape(batch, -1) / float(side - 1) * (height - 16)
    grid_x = 8 + grid_x.reshape(batch, -1) / float(side - 1) * (width - 16)
    xy0 = torch.stack([grid_x, grid_y], dim=-1)
    trajectories = xy0.unsqueeze(1).repeat(1, rgbs.shape[1], 1, 1)
    predictions, _animation, _features, _loss = model(
        trajectories, rgbs, iters=iters, feat_init=None, beautify=True
    )
    return predictions[-1]


def tracking_fullseq(
    model: Pips,
    rgbs: torch.Tensor,
    sw=None,
    N: int = 400,
    iters: int = 8,
    S_max: int = 16,
    name: str = "sample",
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Track consecutive upstream-style windows and return trajectory fields.

    ``sw`` and ``name`` are accepted for source compatibility but visualization
    is intentionally omitted. Windows advance by ``S_max - 1``, matching the
    released runtime.
    Acceleration intentionally preserves the released evaluator's mismatch
    with paper Eq. (2), including its trajectory input and two-zero padding.
    """
    del sw, name
    rgbs = rgbs.float()
    if rgbs.ndim != 5 or rgbs.shape[0] != 1:
        raise ValueError("rgbs must have shape (1, S, C, H, W)")
    if S_max < 2:
        raise ValueError("S_max must be at least 2")

    frame_count = rgbs.shape[1]
    trajectories: List[torch.Tensor] = []
    velocities: List[torch.Tensor] = []
    accelerations: List[torch.Tensor] = []
    current = 0
    while current + S_max <= frame_count:
        clip = rgbs[:, current : current + S_max]
        tracked = run_tracking(model, clip, N=N, iters=iters)
        velocity = calc_velocity(tracked)
        acceleration = calc_acceleration(tracked)
        trajectories.append(tracked)
        velocities.append(velocity)
        accelerations.append(acceleration)
        current += S_max - 1
    return trajectories, velocities, accelerations
