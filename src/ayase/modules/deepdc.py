"""DeepDC — Deep Distribution Conformance NR-IQA.

2024 — measures how well local deep features conform to a natural image
distribution. Uses pyiqa backend when available, otherwise falls back
to heuristic NSS conformance estimation.

deepdc_score — LOWER = better quality (distance from natural distribution)
"""

import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class DeepDCModule(PipelineModule):
    name = "deepdc"
    description = "DeepDC distribution conformance NR-IQA via pyiqa (2024, lower=better)"
    default_config = {
        "subsample": 8,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._model = None
        self._backend = "heuristic"

    def setup(self) -> None:
        # Tier 1: Try pyiqa DeepDC
        try:
            import pyiqa
            self._model = pyiqa.create_metric("deepdc", device="cpu")
            self._backend = "pyiqa"
            logger.info("DeepDC (pyiqa) initialised")
            return
        except (ImportError, Exception):
            pass

        # Tier 2: Heuristic fallback
        self._backend = "heuristic"
        logger.info("DeepDC (heuristic) initialised — install pyiqa for full model")

    def process(self, sample: Sample) -> Sample:
        try:
            if self._backend == "pyiqa":
                score = self._process_pyiqa(sample)
            else:
                score = self._process_heuristic(sample)

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.deepdc_score = score

        except Exception as e:
            logger.warning(f"DeepDC failed for {sample.path}: {e}")

        return sample

    def _process_pyiqa(self, sample: Sample) -> Optional[float]:
        """Process via pyiqa DeepDC metric."""
        import torch

        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            try:
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total <= 0:
                    return None
                indices = np.linspace(0, total - 1, min(self.subsample, total), dtype=int)
                scores = []
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    tensor = (
                        torch.from_numpy(frame_rgb)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .float()
                        / 255.0
                    )
                    with torch.no_grad():
                        result = self._model(tensor)
                    val = float(result.item()) if hasattr(result, "item") else float(result)
                    scores.append(val)
            finally:
                cap.release()
            return float(np.mean(scores)) if scores else None
        else:
            img = cv2.imread(str(sample.path))
            if img is None:
                return None
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = (
                torch.from_numpy(img_rgb)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                / 255.0
            )
            with torch.no_grad():
                result = self._model(tensor)
            return float(result.item()) if hasattr(result, "item") else float(result)

    def _process_heuristic(self, sample: Sample) -> Optional[float]:
        """Heuristic: NSS conformance as distribution distance proxy (lower=better)."""
        frames_gray = []

        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            try:
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total <= 0:
                    return None
                indices = np.linspace(0, total - 1, min(self.subsample, total), dtype=int)
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frames_gray.append(
                            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
                        )
            finally:
                cap.release()
        else:
            img = cv2.imread(str(sample.path))
            if img is None:
                return None
            frames_gray.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64))

        if not frames_gray:
            return None

        deviation_scores = []
        for gray in frames_gray:
            # MSCN coefficients
            mu = cv2.GaussianBlur(gray, (7, 7), 7 / 6)
            sigma = np.sqrt(
                np.abs(cv2.GaussianBlur(gray * gray, (7, 7), 7 / 6) - mu * mu)
            )
            sigma = np.maximum(sigma, 1e-7)
            mscn = (gray - mu) / sigma

            # Natural images: MSCN follows GGD with shape ~0.3-0.5
            # Deviation from expected GGD shape = higher distortion
            std_mscn = np.std(mscn)
            mean_abs = np.mean(np.abs(mscn))
            kurt = float(np.mean(mscn ** 4)) / (std_mscn ** 4 + 1e-7)

            # Expected values for pristine content
            shape_dev = abs(mean_abs - 0.798)  # expected for GGD
            kurt_dev = abs(kurt - 2.4) / 3.0   # expected kurtosis ~2.4
            var_dev = abs(std_mscn - 1.0)       # normalized, expected ~1.0

            deviation = 0.4 * shape_dev + 0.35 * kurt_dev + 0.25 * var_dev
            deviation_scores.append(deviation)

        score = float(np.mean(deviation_scores))
        return float(np.clip(score, 0.0, 1.0))
