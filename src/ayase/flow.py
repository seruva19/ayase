"""Dense RAFT optical-flow FIELD primitive.

The flow-metric modules (:mod:`ayase.modules.advanced_flow`,
:mod:`ayase.modules.raft_motion`) compute a dense RAFT displacement field
internally and then reduce it to a scalar (``flow_score`` / ``raft_motion_score``).
Consumers that need the full field -- camera-motion compensation, subject-region
residual flow, warping -- have no way to obtain it from those modules.

This module exposes the dense field ``(H, W, 2)`` for a single frame pair, using the
same RAFT-Large weights (``C_T_SKHT_V2``) and preprocessing as ``advanced_flow`` so
that field-based metrics stay numerically consistent with ``flow_score``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# torchvision mirror for the RAFT-Large weights (shared with ``advanced_flow``).
_RAFT_LARGE_MIRROR = (
    "raft_large_C_T_SKHT_V2-ff5fadd5.pth",
    "https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/"
    "advanced_flow/raft_large_C_T_SKHT_V2-ff5fadd5.pth",
)

# Cache: (variant, device) -> (model, transforms). RAFT is heavy; build once per worker.
_MODELS: Dict[Tuple[str, str], Tuple[Any, Any]] = {}


def load_raft_flow_model(device: str = "auto", models_dir: str = "models") -> Tuple[Any, Any, str]:
    """Load and cache RAFT-Large plus its preprocessing transforms.

    Uses the Ayase weight mirror and device resolution, matching ``advanced_flow``.
    The model and transforms are cached per resolved device so repeated calls (one
    per frame pair) do not rebuild the network.

    Returns:
        ``(model, transforms, device_str)``.
    """
    import os

    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    from ayase.config import download_torch_hub_checkpoint
    from ayase.runtime import resolve_torch_device

    device_str = resolve_torch_device(device)
    cached = _MODELS.get(("raft_large", device_str))
    if cached is not None:
        return cached[0], cached[1], device_str

    # Redirect torch.hub's checkpoint cache to models_dir, then prefetch from the
    # Ayase-controlled mirror (best effort; torchvision falls back to its own hub).
    os.environ.setdefault("TORCH_HOME", str(models_dir))
    filename, url = _RAFT_LARGE_MIRROR
    try:
        download_torch_hub_checkpoint(filename, url, models_dir)
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("RAFT mirror prefetch skipped (%s); using torchvision hub", exc)

    weights = Raft_Large_Weights.DEFAULT  # == Raft_Large_Weights.C_T_SKHT_V2
    model = raft_large(weights=weights, progress=False).to(device_str).eval()
    transforms = weights.transforms()
    _MODELS[("raft_large", device_str)] = (model, transforms)
    logger.info("RAFT flow-field model ready on %s", device_str)
    return model, transforms, device_str


def _crop_to_multiple_of_8(frame: np.ndarray) -> np.ndarray:
    """Centre-anchored crop so both spatial dims are divisible by 8 (RAFT requires it)."""
    h8 = (frame.shape[0] // 8) * 8
    w8 = (frame.shape[1] // 8) * 8
    return frame[:h8, :w8]


def raft_flow_field(
    prev_rgb: np.ndarray,
    cur_rgb: np.ndarray,
    *,
    device: str = "auto",
    models_dir: str = "models",
) -> np.ndarray:
    """Dense RAFT optical flow ``(H, W, 2)`` between two RGB ``uint8`` frames.

    RAFT requires spatial dims divisible by 8; both frames are cropped to the largest
    such size before inference. The RAFT-Large ``weights.transforms()`` preprocessing
    scales ``uint8`` [0, 255] to [-1, 1] -- identical to a manual ``2*x - 1`` on the
    [0, 1] frame -- so the field matches what ``advanced_flow`` sees internally.

    Args:
        prev_rgb: first frame, ``(H, W, 3)`` RGB ``uint8``.
        cur_rgb: second frame, same shape.
        device: torch device spec ("auto", "cpu", "cuda", ...).
        models_dir: where RAFT weights are cached.

    Returns:
        ``(H, W, 2)`` ``float32`` per-pixel displacement in pixels, in the cropped
        frame's coordinates. ``H``/``W`` are the inputs floored to multiples of 8.

    Raises:
        ValueError: if either frame is smaller than 8 px on a side after cropping.
    """
    import torch

    model, transforms, device_str = load_raft_flow_model(device, models_dir)

    prev_c = _crop_to_multiple_of_8(prev_rgb)
    cur_c = _crop_to_multiple_of_8(cur_rgb)
    if prev_c.shape[0] < 8 or prev_c.shape[1] < 8:
        raise ValueError(f"frames too small for RAFT after /8 crop: {prev_rgb.shape}")

    def _to_tensor(frame: np.ndarray) -> Any:
        # uint8 (H, W, 3) -> (1, 3, H, W); transforms() handles [0,255] -> [-1,1].
        return torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).unsqueeze(0)

    img1, img2 = transforms(_to_tensor(prev_c), _to_tensor(cur_c))
    img1, img2 = img1.to(device_str), img2.to(device_str)
    with torch.no_grad():
        flow = model(img1, img2)[-1][0]  # (2, H, W)
    return flow.permute(1, 2, 0).cpu().numpy().astype(np.float32)  # (H, W, 2)
