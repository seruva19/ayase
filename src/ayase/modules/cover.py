"""Run the native COVER semantic, aesthetic, and technical quality evaluator.

Each image is passed as one frame; each video contributes up to eight uniformly
sampled RGB frames. The native model's raw ``technical``, ``aesthetic``,
``semantic``, and ``overall`` outputs populate the corresponding ``cover_*``
fields, without normalization or an asserted range; larger ``cover_score`` is
treated as better by the configurable low-quality threshold. This is a
per-sample, no-reference evaluator and does not consume captions or generation
prompts. Scores are produced only by the native pretrained COVER package, whose
training-domain and scale limitations apply; no proxy is used when unavailable.
Model and method source: https://github.com/vztu/COVER
"""

import logging
from typing import List, Optional

import numpy as np

from ayase.image import load_representative_frame, sample_frames
from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class COVERModule(PipelineModule):
    name = "cover"
    description = "COVER 3-branch comprehensive video quality (semantic + aesthetic + technical)"
    default_config = {
        "subsample": 8,
        "quality_threshold": 30.0,
    }
    metric_groups = {
        "cover_aesthetic": "aesthetic",
        "cover_score": "nr_quality",
        "cover_semantic": "aesthetic",
        "cover_technical": "nr_quality",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._model = None
        self._device = "cpu"
        self._backend = "unavailable"

    def setup(self) -> None:
        try:
            import torch
            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            try:
                from cover import COVER as COVERModel
            except ImportError:
                logger.warning(
                    "COVER unavailable: the native COVER package is not installed "
                    "(github.com/vztu/COVER); cover_* metrics skipped."
                )
                return

            self._model = COVERModel(pretrained=True)
            self._model.eval()
            self._model = self._model.to(self._device)
            self._ml_available = True
            self._backend = "cover"
            logger.info("COVER model loaded on %s", self._device)
        except Exception as e:
            logger.warning("COVER unavailable: %s", e)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available:
            return sample

        try:
            self._process_cover(sample)

            threshold = self.config.get("quality_threshold", 30.0)
            if (
                sample.quality_metrics.cover_score is not None
                and sample.quality_metrics.cover_score < threshold
            ):
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low COVER quality score: {sample.quality_metrics.cover_score:.1f}",
                        recommendation="Review video for quality issues",
                    )
                )
        except Exception as e:
            logger.warning("COVER processing failed: %s", e)
        return sample

    def _process_cover(self, sample: Sample) -> None:
        """Process with native COVER model."""
        import torch

        frames = self._load_frames(sample)
        if not frames:
            return

        tensors = []
        for frame in frames:
            arr = np.ascontiguousarray(frame)
            t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
            tensors.append(t)

        video_tensor = torch.stack(tensors).unsqueeze(0).to(self._device)

        with torch.no_grad():
            result = self._model(video_tensor)

        if isinstance(result, dict):
            sample.quality_metrics.cover_technical = float(result.get("technical", 0))
            sample.quality_metrics.cover_aesthetic = float(result.get("aesthetic", 0))
            sample.quality_metrics.cover_semantic = float(result.get("semantic", 0))
            sample.quality_metrics.cover_score = float(result.get("overall", 0))
        else:
            sample.quality_metrics.cover_score = float(result)

    def _load_frames(self, sample: Sample) -> List[np.ndarray]:
        subsample = self.config.get("subsample", 8)
        if sample.is_video:
            return sample_frames(sample.path, max_frames=subsample, color="rgb")
        frame = load_representative_frame(sample.path, color="rgb")
        return [frame] if frame is not None else []
