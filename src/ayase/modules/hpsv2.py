"""HPSv2 prompt-conditioned human preference scoring.

Scores image/prompt pairs with raw HPSv2 preference scores. Videos are scored
by uniformly sampling frames and averaging frame scores.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
from PIL import Image

from ayase.image import arrays_to_pil, sample_frames
from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

from ._reward_utils import get_prompt, load_rgb_image

logger = logging.getLogger(__name__)


class HPSv2Module(PipelineModule):
    name = "hpsv2"
    description = "HPSv2 prompt-image human preference scoring"
    default_config = {
        "backend": "auto",  # auto | hpsv2 | diffsynth
        "num_frames": 5,
        "device": "auto",
        "dtype": None,
        "warning_threshold": None,
        "max_image_size": 1024,
        "resize_to_square": False,
        "vram_limit": None,
    }
    models = [
        {
            "id": "hpsv2",
            "type": "pip_package",
            "install": "pip install hpsv2",
            "task": "HPSv2 prompt-conditioned reward model",
        },
        {
            "id": "DiffSynth-Studio/ImageMetrics:HPSv2",
            "type": "other",
            "url": "https://modelscope.cn/models/DiffSynth-Studio/ImageMetrics",
            "task": "DiffSynth HPSv2 metric weights",
            "notes": "Used when the optional diffsynth backend is available.",
        },
    ]
    metric_info = {
        "hpsv2_score": "HPSv2 prompt-image preference score (higher=better)",
    }
    metric_groups = {
        "hpsv2_score": "alignment",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.backend_pref = str(self.config.get("backend", "auto")).lower()
        self.num_frames = int(self.config.get("num_frames", 5))
        self.device_config = self.config.get("device", "auto")
        self.dtype_config = self.config.get("dtype")
        self.warning_threshold = self.config.get("warning_threshold")
        self.max_image_size = int(self.config.get("max_image_size", 1024))
        self.resize_to_square = bool(self.config.get("resize_to_square", False))
        self._backend = None
        self._model = None
        self._ml_available = False

    def setup(self) -> None:
        if self.test_mode:
            return

        if self.backend_pref in ("auto", "hpsv2") and self._try_hpsv2_package():
            return
        if self.backend_pref in ("auto", "diffsynth") and self._try_diffsynth():
            return

        if self.backend_pref not in ("auto", "hpsv2", "diffsynth"):
            logger.warning("Unsupported HPSv2 backend: %s", self.backend_pref)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        prompt = get_prompt(sample, self.config)
        if not prompt:
            return sample

        try:
            frames = self._load_frames(sample)
            if not frames:
                return sample

            score = self._score_frames(frames, prompt)
            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.hpsv2_score = score

            if self.warning_threshold is not None and score < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low HPSv2 score: {score:.3f}",
                        details={"hpsv2_score": score},
                    )
                )
        except Exception as e:
            logger.warning("HPSv2 inference failed for %s: %s", sample.path, e)

        return sample

    def _try_hpsv2_package(self) -> bool:
        try:
            import hpsv2

            self._model = hpsv2
            self._backend = "hpsv2"
            self._ml_available = True
            logger.info("HPSv2 package backend initialized")
            return True
        except Exception as e:
            logger.debug("HPSv2 package backend unavailable: %s", e)
            return False

    def _try_diffsynth(self) -> bool:
        try:
            import torch
            from diffsynth.metrics import HPSv2Metric

            from ayase.runtime import resolve_torch_device, resolve_torch_dtype

            device = resolve_torch_device(self.device_config)
            dtype = resolve_torch_dtype(device, self.dtype_config)
            kwargs = {
                "device": torch.device(device),
                "vram_limit": self.config.get("vram_limit"),
            }
            if dtype is not None:
                kwargs["torch_dtype"] = dtype

            self._model = HPSv2Metric.from_pretrained(**kwargs)
            self._backend = "diffsynth"
            self._ml_available = True
            logger.info("DiffSynth HPSv2 backend initialized on %s", device)
            return True
        except Exception as e:
            logger.debug("DiffSynth HPSv2 backend unavailable: %s", e)
            return False

    def _score_frames(self, frames: List[Image.Image], prompt: str) -> Optional[float]:
        scores = []
        if self._backend == "diffsynth":
            try:
                result = self._model.compute(prompt, frames)
                scores.extend(_as_floats(result))
            except Exception as e:
                logger.debug("DiffSynth HPSv2 scoring failed: %s", e)
        else:
            for frame in frames:
                try:
                    result = self._model.score(frame, prompt)
                    scores.extend(_as_floats(result))
                except Exception as e:
                    logger.debug("HPSv2 frame scoring failed: %s", e)

        return float(np.mean(scores)) if scores else None

    def _load_frames(self, sample: Sample) -> List[Image.Image]:
        if not sample.is_video:
            image = load_rgb_image(
                sample.path,
                max_image_size=self.max_image_size,
                resize_to_square=self.resize_to_square,
            )
            return [image] if image is not None else []

        try:
            arrays = sample_frames(sample.path, max_frames=self.num_frames, color="rgb")
            return arrays_to_pil(arrays)
        except Exception as e:
            logger.debug("HPSv2 frame loading failed for %s: %s", sample.path, e)
            return []


def _as_floats(value) -> List[float]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        values = []
        for item in value:
            values.extend(_as_floats(item))
        return values
    try:
        return [float(value)]
    except (TypeError, ValueError):
        return []
