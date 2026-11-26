"""Estimate aesthetic and technical image quality with NIMA.

Uses the PyIQA NIMA model on images or sampled video frames and averages frame predictions;
it does not assess temporal quality. Scores use the model's 1-10 scale, and higher is better.
Basis: https://github.com/chaofengc/IQA-PyTorch
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class NIMAModule(PipelineModule):
    name = "nima"
    implementation_status = "implemented"
    implementation_notes = "Real trained pyiqa NIMA on RGB float tensors and frame means; image method preserved with disclosed temporal adaptation."
    implementation_review = "Source inspection; confidence high; pretrained inference not validated."
    provenance = 'adapted'
    deviations = {'*': 'The cited score is defined per image. Ayase also writes this field for video by selecting decoded frames and aggregating their image scores; frame selection and pooling follow this module, not a published native video protocol. Image inputs use the image backend.'}
    sources = {
        "nima_score": "NIMA (Talebi & Milanfar, TIP 2018) via pyiqa — https://github.com/chaofengc/IQA-PyTorch",
    }
    description = "NIMA aesthetic and technical image quality (1-10 scale)"
    default_config = {"subsample": 8, "batch_frames": False}
    metric_groups = {
        "nima_score": "aesthetic",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._model = None
        self._device = "cpu"
        self._backend = None

    def setup(self) -> None:
        try:
            import pyiqa
            import torch

            from ayase.runtime import resolve_torch_device

            device = resolve_torch_device(self.config.get("device", "auto"))
            self._model = pyiqa.create_metric("nima", device=device)
            try:
                self._device = next(self._model.parameters()).device
            except StopIteration:
                self._device = torch.device(device)
            self._ml_available = True
            self._backend = "pyiqa"
            logger.info("NIMA model loaded on %s", device)
        except ImportError as e:
            self._backend = "unavailable"
            logger.warning("NIMA unavailable: %s", e)
        except Exception as e:
            self._backend = "unavailable"
            logger.warning("NIMA unavailable: %s", e)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available:
            return sample
        try:
            frames = self._load_frames(sample)
            if frames:
                sample.quality_metrics.nima_score = self._score_frames(frames)
        except Exception as e:
            logger.warning("NIMA processing failed: %s", e)
        return sample

    def _score_frames(self, frames: list) -> float:
        import cv2
        import torch
        from ayase.runtime import torch_inference_context

        def tensor(frame):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        with torch_inference_context(str(self._device), "fp32", False):
            if self.config.get("batch_frames", False) and len(frames) > 1:
                try:
                    scores = self._model(
                        torch.stack([tensor(frame) for frame in frames]).to(self._device)
                    )
                    if scores.ndim == 0 or len(scores) != len(frames) or scores.numel() != len(frames):
                        raise ValueError("NIMA did not return one score per frame")
                    return float(np.mean(scores.flatten().cpu().tolist()))
                except Exception as exc:
                    logger.debug("NIMA batch failed; retrying individual frames: %s", exc)

            scores = [
                self._model(tensor(frame).unsqueeze(0).to(self._device)).reshape(())
                for frame in frames
            ]
            return float(np.mean(torch.stack(scores).cpu().tolist()))

    def _load_frames(self, sample: Sample) -> list:
        import cv2

        subsample = self.config.get("subsample", 8)
        frames = []
        if sample.is_video:
            from ayase.image import sample_frames

            return sample_frames(sample.path, max_frames=subsample, color="bgr", floor_stride=True)
        else:
            frame = cv2.imread(str(sample.path))
            if frame is not None:
                frames.append(frame)
        return frames
