"""AHIQ (Attention-based Hybrid Image Quality) module."""

import logging
from typing import List, Optional

import numpy as np

from ayase.image import sample_frames
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class AHIQModule(PipelineModule):
    name = "ahiq"
    description = "Attention-based Hybrid IQA full-reference (higher=better)"
    default_config = {"subsample": 8}
    metric_groups = {
        "ahiq": "fr_quality",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._model = None
        self._device = None
        self._backend = "unavailable"

    def setup(self) -> None:
        try:
            import pyiqa
            import torch
            from ayase.runtime import resolve_torch_device

            device = resolve_torch_device(self.config.get("device", "auto"))
            self._model = pyiqa.create_metric("ahiq", device=device)
            try:
                self._device = next(self._model.parameters()).device
            except StopIteration:
                self._device = torch.device(device)
            self._ml_available = True
            self._backend = "pyiqa"
            logger.info("AHIQ model loaded on %s", device)
        except ImportError:
            logger.warning("AHIQ unavailable: pyiqa is not installed (pip install pyiqa)")
        except Exception as e:
            logger.warning("AHIQ unavailable: %s", e)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available:
            return sample
        reference_path = getattr(sample, "reference_path", None)
        if reference_path is None:
            return sample
        try:
            import cv2
            import torch

            ref_frames = self._load_frames(reference_path)
            dist_frames = self._load_frames(sample.path)
            if not ref_frames or not dist_frames:
                return sample

            n = min(len(ref_frames), len(dist_frames))
            scores = []
            device = self._device
            for i in range(n):
                # frames are RGB read-only views; cv2.resize returns a fresh
                # writable array so torch.from_numpy is safe afterwards.
                ref_rgb = ref_frames[i]
                dist_rgb = dist_frames[i]
                h = min(ref_rgb.shape[0], dist_rgb.shape[0])
                w = min(ref_rgb.shape[1], dist_rgb.shape[1])
                ref_rgb = cv2.resize(np.ascontiguousarray(ref_rgb), (w, h))
                dist_rgb = cv2.resize(np.ascontiguousarray(dist_rgb), (w, h))
                ref_t = torch.from_numpy(np.ascontiguousarray(ref_rgb)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                dist_t = torch.from_numpy(np.ascontiguousarray(dist_rgb)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                with torch.no_grad():
                    score = self._model(dist_t.to(device), ref_t.to(device)).item()
                scores.append(score)

            if scores:
                sample.quality_metrics.ahiq = float(np.mean(scores))
        except Exception as e:
            logger.warning("AHIQ processing failed: %s", e)
        return sample

    def _load_frames(self, path) -> List[np.ndarray]:
        subsample = self.config.get("subsample", 8)
        try:
            return sample_frames(path, max_frames=subsample, color="rgb")
        except Exception as e:
            logger.debug("AHIQ frame load failed for %s: %s", path, e)
            return []
