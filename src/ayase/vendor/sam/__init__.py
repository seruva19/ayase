"""Vendored Segment Anything (SAM) box-prompted masking backend.

This package provides a small, self-contained wrapper around the SAM
predictor for box-prompted subject masking. The model weights are fetched
from the Hugging Face Hub at load time. Only the box-prompted prediction
path is used; the automatic mask generator and ONNX export are intentionally
omitted.

Public API:
    load_sam(models_dir, device, model_type) -> SamWrapper
    SamWrapper.subject_mask(image, boxes_xyxy) -> np.ndarray  # HxW bool
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .build_sam import sam_model_registry
from .predictor import SamPredictor

__all__ = ["load_sam", "SamWrapper"]

# Hugging Face repository that hosts the SAM checkpoints used here.
_HF_REPO_ID = "GD-ML/VMBench"
_CHECKPOINT_FILENAMES = {
    "vit_h": "sam_vit_h_4b8939.pth",
}


def load_sam(models_dir: str = "models", device: str = "cuda", model_type: str = "vit_h") -> "SamWrapper":
    """Build a SAM predictor and return a stateful ``SamWrapper``.

    The checkpoint is downloaded from the Hugging Face Hub (repo
    ``GD-ML/VMBench``) into ``models_dir`` on first use and cached there.

    Arguments:
        models_dir: Local directory to download/cache the checkpoint into.
        device: Torch device string (e.g. ``"cuda"`` or ``"cpu"``).
        model_type: SAM variant to build. Only ``"vit_h"`` is exercised here.

    Returns:
        A ``SamWrapper`` with the model loaded once and ready for repeated
        per-frame masking.
    """
    from huggingface_hub import hf_hub_download

    if model_type not in _CHECKPOINT_FILENAMES:
        raise ValueError(
            f"Unsupported model_type {model_type!r}; expected one of "
            f"{sorted(_CHECKPOINT_FILENAMES)}."
        )

    checkpoint = hf_hub_download(
        repo_id=_HF_REPO_ID,
        filename=_CHECKPOINT_FILENAMES[model_type],
        local_dir=models_dir,
    )
    sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
    sam.eval()
    return SamWrapper(SamPredictor(sam))


class SamWrapper:
    """Stateful wrapper that runs box-prompted SAM masking per frame.

    The SAM model is loaded once. For each frame, the image embedding is
    computed internally via ``SamPredictor.set_image`` and the union of the
    per-box masks is returned as a single HxW boolean array.
    """

    def __init__(self, predictor: SamPredictor) -> None:
        self._predictor = predictor

    @property
    def device(self) -> torch.device:
        return self._predictor.device

    def subject_mask(
        self,
        image: np.ndarray,
        boxes_xyxy,
        image_format: str = "RGB",
    ) -> np.ndarray:
        """Return the union of per-box SAM masks for one frame.

        Arguments:
            image: A single frame as an HxWxC uint8 numpy array. Its color
                order is declared by ``image_format`` (``"RGB"`` or ``"BGR"``);
                it is converted internally as needed.
            boxes_xyxy: K subject boxes in pixel ``xyxy`` format, as a numpy
                array or torch tensor of shape ``[K, 4]`` (a single ``[4]``
                box is also accepted). If ``None`` or empty, an all-``False``
                mask is returned.
            image_format: Color order of ``image``; one of ``"RGB"``/``"BGR"``.

        Returns:
            An HxW boolean numpy array: the logical union of the per-box
            masks (``multimask_output=False``).
        """
        h, w = image.shape[:2]

        if boxes_xyxy is None:
            return np.zeros((h, w), dtype=bool)

        if isinstance(boxes_xyxy, torch.Tensor):
            boxes = boxes_xyxy.detach().to(device=self.device, dtype=torch.float)
        else:
            boxes = torch.as_tensor(
                np.asarray(boxes_xyxy), dtype=torch.float, device=self.device
            )
        if boxes.ndim == 1:
            boxes = boxes.unsqueeze(0)
        if boxes.numel() == 0:
            return np.zeros((h, w), dtype=bool)

        self._predictor.set_image(image, image_format=image_format)
        transformed_boxes = self._predictor.transform.apply_boxes_torch(
            boxes, image.shape[:2]
        )
        masks, _, _ = self._predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        # masks: [K, 1, H, W] boolean -> union over the K boxes.
        union = masks[:, 0].any(dim=0)
        return union.detach().cpu().numpy().astype(bool)
