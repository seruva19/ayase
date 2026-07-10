"""Vendored VideoMAEv2 (ViT-giant) commonsense-adherence classifier backend.

This package provides a small, self-contained wrapper around the VideoMAEv2
finetune Vision Transformer used as the classifier for VMBench's Commonsense
Adherence Score (CAS). The fine-tuned checkpoint is fetched from the Hugging
Face Hub at load time. Only the inference path (model definition + minimal
test-time preprocessing) is vendored; all training/distributed utilities of the
upstream repository are intentionally omitted.

The checkpoint has an ordinal commonsense head (not a Kinetics-400 head); the
number of classes is read from the checkpoint's head weight at load time.

Public API:
    load_videomae(models_dir, device) -> VideoMaeWrapper
    VideoMaeWrapper.logits(frames_rgb_list) -> torch.Tensor        # [num_classes]
    VideoMaeWrapper.commonsense_adherence_score(frames_rgb_list) -> float

Commonsense Adherence Score formula
-----------------------------------
Discovered from the VMBench repository (github.com/AMAP-ML/VMBench),
``bench_utils/cas_utils.py``. The classifier is a 5-way *ordinal* head. Per
clip:

    probabilities = softmax(logits)                      # cas_utils.final_test
    prob_weights  = [0.0, 0.25, 0.5, 0.75, 1.0]          # cas_utils.final_merge
    CAS           = sum(probabilities * prob_weights)    # in [0, 1]

i.e. the expected value of the ordinal rating (rating k in {0..4} mapped to
k/4). Upstream averages the softmax vector over 10 temporal segments x 3 spatial
crops before the weighted sum; this wrapper computes a single-view estimate
(one center crop of a 16-frame clip). The reported dimension score multiplies
CAS by 100 (``bench_utils/calculate_score.py``).
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch

from .modeling_finetune import vit_giant_patch14_224
from .preprocess import preprocess_frames

__all__ = ["load_videomae", "VideoMaeWrapper", "PROB_WEIGHTS"]

# Hugging Face repository hosting the VMBench-finetuned checkpoint.
_HF_REPO_ID = "GD-ML/VMBench"
_CHECKPOINT_FILENAME = "vit_g_vmbench.pt"

# Ordinal rating weights used by VMBench to collapse the per-class softmax into
# a single 0-1 commonsense-adherence score (bench_utils/cas_utils.py).
PROB_WEIGHTS = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)


def _extract_state_dict(ckpt) -> dict:
    """Unwrap the checkpoint to a bare parameter ``state_dict``."""
    if isinstance(ckpt, dict):
        for key in ("module", "model", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def _find_head_weight_shape(state_dict: dict):
    """Return the classification-head weight shape ``[num_classes, embed_dim]``."""
    for key in ("head.weight", "module.head.weight", "model.head.weight"):
        if key in state_dict:
            return tuple(state_dict[key].shape)
    # Fall back: any key ending in "head.weight".
    for key, val in state_dict.items():
        if key.endswith("head.weight"):
            return tuple(val.shape)
    raise KeyError("No classification-head weight found in checkpoint.")


def load_videomae(models_dir: str = "models",
                  device: str = "cuda") -> "VideoMaeWrapper":
    """Build the VideoMAEv2 giant classifier and return a ``VideoMaeWrapper``.

    The checkpoint is downloaded from the Hugging Face Hub (repo
    ``GD-ML/VMBench``, file ``vit_g_vmbench.pt``) into ``models_dir`` on first
    use and cached there. ``num_classes`` is determined from the checkpoint's
    head weight so the architecture always matches the weights.

    Arguments:
        models_dir: Local directory to download/cache the checkpoint into.
        device: Torch device string (e.g. ``"cuda"`` or ``"cpu"``).

    Returns:
        A ``VideoMaeWrapper`` with the model loaded once, in eval mode.
    """
    from huggingface_hub import hf_hub_download

    checkpoint_path = hf_hub_download(
        repo_id=_HF_REPO_ID,
        filename=_CHECKPOINT_FILENAME,
        local_dir=models_dir,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(ckpt)
    num_classes = _find_head_weight_shape(state_dict)[0]

    model = vit_giant_patch14_224(
        img_size=224,
        num_classes=num_classes,
        all_frames=16,
        tubelet_size=2,
        use_mean_pooling=True,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    return VideoMaeWrapper(
        model,
        device=device,
        num_classes=num_classes,
        missing_keys=list(missing),
        unexpected_keys=list(unexpected),
    )


class VideoMaeWrapper:
    """Stateful wrapper running the VideoMAEv2 CAS classifier on RGB clips."""

    def __init__(self,
                 model,
                 device: str = "cuda",
                 num_classes: int = 5,
                 missing_keys: List[str] | None = None,
                 unexpected_keys: List[str] | None = None) -> None:
        self._model = model
        self._device = device
        self.num_classes = num_classes
        self.missing_keys = missing_keys or []
        self.unexpected_keys = unexpected_keys or []

    @property
    def device(self) -> str:
        return self._device

    @torch.no_grad()
    def logits(self, frames_rgb_list: Sequence[np.ndarray]) -> "torch.Tensor":
        """Return the raw class logits for one clip as a ``[num_classes]`` tensor.

        Arguments:
            frames_rgb_list: Sequence of HxWxC uint8 RGB frames (a decoded clip).

        Returns:
            A 1-D ``torch.Tensor`` of length ``num_classes`` on CPU (float32).
        """
        clip = preprocess_frames(frames_rgb_list).to(self._device)
        out = self._model(clip)  # [1, num_classes]
        return out.float().squeeze(0).cpu()

    def probabilities(self, frames_rgb_list: Sequence[np.ndarray]) -> "torch.Tensor":
        """Return softmax probabilities for one clip (``[num_classes]``)."""
        logits = self.logits(frames_rgb_list)
        return torch.softmax(logits, dim=0)

    def commonsense_adherence_score(self,
                                    frames_rgb_list: Sequence[np.ndarray]) -> float:
        """Return the VMBench Commonsense Adherence Score in ``[0, 1]``.

        Applies the exact VMBench CAS formula (bench_utils/cas_utils.py):
        softmax the logits, then take the weighted sum with ordinal weights
        ``[0.0, 0.25, 0.5, 0.75, 1.0]``. This is a single-view estimate.

        Raises:
            ValueError: if the head is not the 5-way ordinal commonsense head
                the CAS weights are defined for.
        """
        probs = self.probabilities(frames_rgb_list).numpy().astype(np.float64)
        if probs.shape[0] != PROB_WEIGHTS.shape[0]:
            raise ValueError(
                f"CAS weights expect {PROB_WEIGHTS.shape[0]} ordinal classes "
                f"but the head produced {probs.shape[0]}.")
        return float(np.sum(probs * PROB_WEIGHTS))
