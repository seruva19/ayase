"""COVER (Comprehensive Video Quality Evaluator) module.

3-branch architecture: semantic + aesthetic + technical.
Winner of AIS 2024 VQA Challenge at CVPR 2024.
"""

import logging
from typing import Optional

import numpy as np

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

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._model = None

    def setup(self) -> None:
        try:
            # COVER uses a custom model; try loading via its package
            # Fallback: use DOVER (which COVER extends) as approximation
            import torch

            try:
                from cover import COVER as COVERModel

                self._model = COVERModel(pretrained=True)
                self._model.eval()
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._model = self._model.to(device)
                self._ml_available = True
                self._backend = "cover"
                logger.info("COVER model loaded on %s", device)
            except ImportError:
                # Fallback to DOVER which has similar 3-branch design
                import pyiqa

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._model = pyiqa.create_metric("dover", device=device)
                self._ml_available = True
                self._backend = "dover"
                logger.info("COVER fallback: using DOVER on %s", device)
        except (ImportError, Exception) as e:
            logger.warning("COVER unavailable: %s", e)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available:
            return sample

        try:
            if self._backend == "dover":
                self._process_dover_fallback(sample)
            else:
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
        import cv2
        import torch

        frames = self._load_frames(sample)
        if not frames:
            return

        device = next(self._model.parameters()).device
        tensors = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            tensors.append(t)

        video_tensor = torch.stack(tensors).unsqueeze(0).to(device)

        with torch.no_grad():
            result = self._model(video_tensor)

        if isinstance(result, dict):
            sample.quality_metrics.cover_technical = float(result.get("technical", 0))
            sample.quality_metrics.cover_aesthetic = float(result.get("aesthetic", 0))
            sample.quality_metrics.cover_semantic = float(result.get("semantic", 0))
            sample.quality_metrics.cover_score = float(result.get("overall", 0))
        else:
            sample.quality_metrics.cover_score = float(result)

    def _process_dover_fallback(self, sample: Sample) -> None:
        """Process with DOVER as approximation for COVER."""
        import cv2
        import torch

        frames = self._load_frames(sample)
        if not frames:
            return

        scores = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = (
                torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            )
            device = next(self._model.parameters()).device
            tensor = tensor.to(device)
            with torch.no_grad():
                score = self._model(tensor).item()
            scores.append(score)

        overall = float(np.mean(scores))
        sample.quality_metrics.cover_score = overall
        # DOVER doesn't separate branches, but we approximate
        sample.quality_metrics.cover_technical = overall
        sample.quality_metrics.cover_aesthetic = overall

    def _load_frames(self, sample: Sample) -> list:
        import cv2

        subsample = self.config.get("subsample", 8)
        frames = []
        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = list(range(0, total, max(1, total // subsample)))[:subsample]
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            cap.release()
        else:
            frame = cv2.imread(str(sample.path))
            if frame is not None:
                frames.append(frame)
        return frames
