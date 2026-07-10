# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Public, self-contained inference API for the vendored GroundingDINO detector.

Exposes :func:`load_grounding_dino`, which builds the SwinB open-vocabulary
detector and returns a stateful :class:`GroundingDinoWrapper`. The wrapper's
:meth:`GroundingDinoWrapper.ground` method takes a single video frame plus a
subject phrase and returns the matching boxes in ``xyxy`` pixel coordinates.

Weights are fetched from the Hugging Face Hub at runtime (the SwinB checkpoint
from ``GD-ML/VMBench`` and the standard ``bert-base-uncased`` text encoder); no
weight path is hard-coded. Multi-scale deformable attention uses the pure
PyTorch implementation, so no compiled CUDA extension is required.
"""

import os
from typing import List, NamedTuple, Union

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

import cv2

from . import datasets  # noqa: F401  (ensures subpackage import path is valid)
from .datasets import transforms as T
from .models import build_model
from .util.misc import clean_state_dict
from .util.slconfig import SLConfig
from .util.utils import get_phrases_from_posmap

# Hugging Face location of the SwinB checkpoint used by VMBench's PAS metric.
_HF_REPO_ID = "GD-ML/VMBench"
_HF_CHECKPOINT_FILENAME = "groundingdino_swinb_cogcoor.pth"
_CONFIG_FILENAME = "GroundingDINO_SwinB_cfg.py"


class GroundingResult(NamedTuple):
    """Result of a single grounding call.

    Attributes:
        boxes: ``torch.Tensor`` of shape ``[K, 4]`` in ``xyxy`` pixel
            coordinates for the source frame (empty ``[0, 4]`` if nothing
            was grounded).
        scores: ``torch.Tensor`` of shape ``[K]`` with the per-box confidence
            (max token logit after sigmoid).
        phrases: list of ``K`` decoded text phrases, one per box.
    """

    boxes: torch.Tensor
    scores: torch.Tensor
    phrases: List[str]


def _preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."


def _build_transform():
    return T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def load_grounding_dino(models_dir: str = "models", device: str = "cuda") -> "GroundingDinoWrapper":
    """Build the GroundingDINO SwinB detector and return a stateful wrapper.

    Args:
        models_dir: optional directory used as the Hugging Face download cache
            for the checkpoint. If falsy, the default Hugging Face cache is used.
        device: torch device string (e.g. ``"cuda"`` or ``"cpu"``).

    Returns:
        A :class:`GroundingDinoWrapper` with the model + tokenizer loaded once.
    """
    config_path = os.path.join(os.path.dirname(__file__), "config", _CONFIG_FILENAME)

    cache_dir = models_dir if models_dir else None
    checkpoint_path = hf_hub_download(
        repo_id=_HF_REPO_ID,
        filename=_HF_CHECKPOINT_FILENAME,
        cache_dir=cache_dir,
    )

    args = SLConfig.fromfile(config_path)
    args.device = device
    model = build_model(args)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    # Any residual keys here are non-inference buffers only; surface them for
    # visibility without failing (mirrors the reference ``strict=False`` load).
    _ = load_res

    model.eval()
    model = model.to(device)

    return GroundingDinoWrapper(model=model, device=device)


class GroundingDinoWrapper:
    """Stateful GroundingDINO grounder (model + tokenizer loaded once)."""

    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.device = device
        self.tokenizer = model.tokenizer
        self._transform = _build_transform()

    @torch.no_grad()
    def ground(
        self,
        image: Union[np.ndarray, "Image.Image"],
        caption: str,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ) -> GroundingResult:
        """Ground a subject phrase in a single frame.

        Args:
            image: one frame, either a BGR ``np.ndarray`` (OpenCV convention)
                or a PIL image.
            caption: subject noun/phrase, e.g. ``"person"``. A trailing period
                is added automatically as GroundingDINO expects.
            box_threshold: keep boxes whose max token score exceeds this.
            text_threshold: token-selection threshold used to decode phrases.

        Returns:
            A :class:`GroundingResult` with ``xyxy`` pixel boxes, scores and
            phrases. ``boxes`` is an empty ``[0, 4]`` tensor if nothing matched.
        """
        pil_image, height, width = self._to_pil_rgb(image)
        caption = _preprocess_caption(caption)

        image_tensor, _ = self._transform(pil_image, None)
        image_tensor = image_tensor.to(self.device)

        outputs = self.model(image_tensor[None], captions=[caption])

        # (num_queries, 256) token logits; (num_queries, 4) cxcywh normalized
        logits = outputs["pred_logits"].sigmoid()[0].cpu()
        boxes = outputs["pred_boxes"][0].cpu()

        keep = logits.max(dim=1)[0] > box_threshold
        logits = logits[keep]
        boxes = boxes[keep]
        scores = logits.max(dim=1)[0] if logits.numel() else logits.new_zeros((0,))

        tokenized = self.tokenizer(caption)
        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, self.tokenizer).replace(
                ".", ""
            )
            for logit in logits
        ]

        boxes_xyxy = self._cxcywh_norm_to_xyxy_pixels(boxes, height, width)
        return GroundingResult(boxes=boxes_xyxy, scores=scores, phrases=phrases)

    @staticmethod
    def _to_pil_rgb(image):
        if isinstance(image, np.ndarray):
            # OpenCV frames are BGR; convert to RGB for the model transform.
            height, width = image.shape[:2]
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb), height, width
        if isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
            width, height = pil_image.size
            return pil_image, height, width
        raise TypeError(
            "image must be a BGR numpy.ndarray or a PIL.Image, got {}".format(type(image))
        )

    @staticmethod
    def _cxcywh_norm_to_xyxy_pixels(boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes.new_zeros((0, 4))
        scale = torch.tensor([width, height, width, height], dtype=boxes.dtype)
        boxes = boxes * scale
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return torch.stack([x1, y1, x2, y2], dim=-1)
