# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Vendored, inference-only CoTracker3 (offline) point tracker.
# Upstream: facebookresearch/co-tracker. Only the offline model and its
# predictor are vendored; datasets, training, evaluation and visualization
# code are intentionally excluded. Pure PyTorch, no custom CUDA ops.

from __future__ import annotations

import os
from typing import Optional, Tuple, Union

import numpy as np
import torch

from ayase.vendor.cotracker.predictor import CoTrackerPredictor

__all__ = ["load_cotracker", "CoTrackerWrapper", "CoTrackerPredictor"]

# HuggingFace repo hosting the offline CoTracker3 checkpoint used by VMBench.
_HF_REPO_ID = "GD-ML/VMBench"
_HF_FILENAME = "scaled_offline.pth"
_HF_REVISION = "437cf3b7c667cd23e3f1e24e19e0af7868088907"


def _resolve_checkpoint(models_dir: str = "models") -> str:
    """Download the offline CoTracker3 checkpoint from HuggingFace.

    Returns a local filesystem path to ``scaled_offline.pth``.
    """
    from huggingface_hub import hf_hub_download

    os.makedirs(models_dir, exist_ok=True)
    return hf_hub_download(
        repo_id=_HF_REPO_ID,
        filename=_HF_FILENAME,
        revision=_HF_REVISION,
        cache_dir=models_dir,
    )


def load_cotracker(
    models_dir: str = "models",
    device: str = "cuda",
    window_len: int = 60,
) -> "CoTrackerWrapper":
    """Load the offline CoTracker3 model and return a stateful wrapper.

    The checkpoint is fetched once from HuggingFace (``GD-ML/VMBench`` /
    ``scaled_offline.pth``) and cached under ``models_dir``.

    Args:
        models_dir: directory used as the HuggingFace download cache.
        device: torch device string (e.g. ``"cuda"`` or ``"cpu"``).
        window_len: temporal window length; 60 matches the VMBench setup.
    """
    checkpoint = _resolve_checkpoint(models_dir)
    predictor = CoTrackerPredictor(
        checkpoint=checkpoint,
        v2=False,
        offline=True,
        window_len=window_len,
    ).to(device)
    predictor.eval()
    return CoTrackerWrapper(predictor, device=device)


def _to_video_tensor(
    video: Union[torch.Tensor, np.ndarray, list],
    device: str,
) -> torch.Tensor:
    """Normalize input into a float ``[1, T, 3, H, W]`` tensor in range 0-255."""
    if isinstance(video, torch.Tensor):
        vid = video
    else:
        vid = torch.as_tensor(np.asarray(video))

    if vid.ndim == 4:
        # [T, H, W, 3] (channels-last) -> [1, T, 3, H, W]
        if vid.shape[-1] == 3:
            vid = vid.permute(0, 3, 1, 2)  # [T, 3, H, W]
        vid = vid.unsqueeze(0)  # [1, T, 3, H, W]
    elif vid.ndim == 5:
        # already [1, T, 3, H, W]
        pass
    else:
        raise ValueError(
            f"Unsupported video shape {tuple(vid.shape)}; expected "
            "[1, T, 3, H, W] or [T, H, W, 3]."
        )

    return vid.float().to(device)


def _to_mask_tensor(
    segm_mask: Optional[Union[torch.Tensor, np.ndarray, list]],
    device: str,
) -> Optional[torch.Tensor]:
    """Normalize a mask into a float ``[1, 1, H, W]`` tensor (values >0 = kept)."""
    if segm_mask is None:
        return None
    if isinstance(segm_mask, torch.Tensor):
        mask = segm_mask
    else:
        mask = torch.as_tensor(np.asarray(segm_mask))

    if mask.ndim == 2:
        mask = mask[None, None]  # [1, 1, H, W]
    elif mask.ndim == 3:
        mask = mask[None]  # [1, 1, H, W] from [1, H, W]
    elif mask.ndim != 4:
        raise ValueError(
            f"Unsupported mask shape {tuple(mask.shape)}; expected "
            "[H, W] or [1, 1, H, W]."
        )
    return mask.float().to(device)


class CoTrackerWrapper:
    """Stateful, inference-only wrapper around :class:`CoTrackerPredictor`.

    The underlying predictor is loaded once and reused across calls. The raw
    predictor remains accessible via :attr:`predictor`.
    """

    def __init__(self, predictor: CoTrackerPredictor, device: str = "cuda"):
        self.predictor = predictor
        self.device = device

    @torch.no_grad()
    def track(
        self,
        video: Union[torch.Tensor, np.ndarray, list],
        grid_size: int = 30,
        grid_query_frame: int = 0,
        segm_mask: Optional[Union[torch.Tensor, np.ndarray, list]] = None,
        backward_tracking: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Track a grid of points through the video.

        Args:
            video: ``[1, T, 3, H, W]`` float tensor in pixel range 0-255, or a
                ``[T, H, W, 3]`` uint8 numpy array / list (converted internally).
            grid_size: side length of the query point grid (``grid_size**2``
                candidate points before masking).
            grid_query_frame: frame index the grid is queried from.
            segm_mask: ``[H, W]`` or ``[1, 1, H, W]`` mask (bool or 0/255); grid
                points are kept only where the mask is > 0. ``None`` keeps all.
            backward_tracking: also track backwards from the query frame.

        Returns:
            ``(pred_tracks, pred_visibility)`` where ``pred_tracks`` is
            ``[1, T, N, 2]`` pixel xy and ``pred_visibility`` is ``[1, T, N]``
            boolean.
        """
        video_t = _to_video_tensor(video, self.device)
        mask_t = _to_mask_tensor(segm_mask, self.device)

        pred_tracks, pred_visibility = self.predictor(
            video_t,
            grid_size=grid_size,
            grid_query_frame=grid_query_frame,
            backward_tracking=backward_tracking,
            segm_mask=mask_t,
        )
        return pred_tracks, pred_visibility
