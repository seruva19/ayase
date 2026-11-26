"""Estimate no-reference perceptual image quality with MANIQA.

Uses the PyIQA MANIQA model on images or uniformly sampled video frames and averages frame
scores; it does not assess temporal quality. Higher is better, with no fixed range assumed.
Basis: https://github.com/chaofengc/IQA-PyTorch
"""

import logging
from typing import Optional

import numpy as np

from ayase.image import sample_frames
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class MANIQAModule(PipelineModule):
    name = "maniqa"
    implementation_status = "implemented"
    implementation_notes = "Trained pyiqa MANIQA runs on RGB image/frame tensors; frame averaging is explicitly disclosed adaptation."
    implementation_review = "Source inspection; confidence high; pretrained inference not validated."
    provenance = 'adapted'
    deviations = {'*': 'The cited score is defined per image. Ayase also writes this field for video by selecting decoded frames and aggregating their image scores; frame selection and pooling follow this module, not a published native video protocol. Image inputs use the image backend.'}
    sources = {
        "maniqa_score": "MANIQA (Yang et al., CVPRW 2022) via pyiqa — https://github.com/chaofengc/IQA-PyTorch",
    }
    description = "MANIQA multi-dimension attention no-reference IQA"
    default_config = {"subsample": 8, "batch_frames": False}
    metric_groups = {
        "maniqa_score": "nr_quality",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self.batch_frames = self.config.get("batch_frames", False)
        self._ml_available = False
        self._model = None
        self._device = "cpu"
        self._backend = None

    def setup(self) -> None:
        try:
            import pyiqa
            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            self._model = pyiqa.create_metric("maniqa", device=self._device)
            self._ml_available = True
            self._backend = "pyiqa"
            logger.info("MANIQA model loaded on %s", self._device)
        except ImportError:
            self._backend = "unavailable"
            logger.warning("pyiqa not installed. Install with: pip install pyiqa")
        except Exception as e:
            self._backend = "unavailable"
            logger.warning("MANIQA unavailable: %s", e)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available:
            return sample
        try:
            frames = self._load_frames(sample)
            if not frames:
                return sample

            sample.quality_metrics.maniqa_score = self._score_frames(frames)
        except Exception as e:
            logger.warning("MANIQA processing failed: %s", e)
        return sample

    def _score_frames(self, frames: list) -> float:
        import torch
        from ayase.runtime import torch_inference_context

        def to_tensor(frame):
            array = np.ascontiguousarray(frame, dtype=np.float32)
            return torch.from_numpy(array).permute(2, 0, 1).div(255.0)

        with torch_inference_context(str(self._device), "fp32", False):
            if self.batch_frames and len(frames) > 1:
                try:
                    scores = self._model(
                        torch.stack([to_tensor(frame) for frame in frames]).to(self._device)
                    )
                    if scores.ndim == 0 or scores.shape[0] != len(frames):
                        raise ValueError("MANIQA backend did not return a score per frame")
                    if scores.numel() != len(frames):
                        raise ValueError("MANIQA backend returned multiple scores per frame")
                    return float(np.mean(scores.reshape(-1).cpu().tolist()))
                except Exception as exc:
                    logger.debug("MANIQA frame batch failed; retrying one frame at a time: %s", exc)

            scores = [
                self._model(to_tensor(frame).unsqueeze(0).to(self._device)).reshape(())
                for frame in frames
            ]
            return float(np.mean(torch.stack(scores).cpu().tolist()))

    def _load_frames(self, sample: Sample) -> list:
        subsample = self.config.get("subsample", 8)
        return list(sample_frames(sample.path, max_frames=subsample, color="rgb"))
