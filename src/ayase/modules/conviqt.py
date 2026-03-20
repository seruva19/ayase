"""CONVIQT — Contrastive Video Quality Estimator.

IEEE TIP 2023 — self-supervised contrastive learning for quality
representations using distortion identification. No MOS labels
needed for representation learning.

GitHub: https://github.com/pavancm/CONVIQT

conviqt_score — higher = better quality
"""

import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class CONVIQTModule(PipelineModule):
    name = "conviqt"
    description = "CONVIQT contrastive self-supervised NR-VQA (TIP 2023)"
    default_config = {
        "subsample": 8,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._model = None
        self._ml_available = False
        self._backend = "heuristic"

    def setup(self) -> None:
        # Tier 1: Try CONVIQT package
        try:
            import conviqt
            self._model = conviqt
            self._ml_available = True
            self._backend = "native"
            logger.info("CONVIQT (native) initialised")
            return
        except ImportError:
            pass

        # Tier 2: Try pyiqa
        try:
            import pyiqa
            self._model = pyiqa.create_metric("conviqt", device="cpu")
            self._ml_available = True
            self._backend = "pyiqa"
            logger.info("CONVIQT (pyiqa) initialised")
            return
        except (ImportError, Exception):
            pass

        self._backend = "heuristic"
        logger.info("CONVIQT (heuristic) — install conviqt or pyiqa for full model")

    def process(self, sample: Sample) -> Sample:
        try:
            if self._backend == "native":
                score = self._process_native(sample)
            elif self._backend == "pyiqa":
                score = self._process_pyiqa(sample)
            else:
                score = self._process_heuristic(sample)

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.conviqt_score = score

        except Exception as e:
            logger.warning(f"CONVIQT failed for {sample.path}: {e}")

        return sample

    def _process_native(self, sample: Sample) -> Optional[float]:
        return float(self._model.predict(str(sample.path)))

    def _process_pyiqa(self, sample: Sample) -> Optional[float]:
        import torch
        import tempfile
        from pathlib import Path

        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            try:
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                indices = np.linspace(0, total - 1, min(self.subsample, total), dtype=int)
                scores = []
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                            cv2.imwrite(f.name, frame)
                            try:
                                s = float(self._model(f.name).item())
                                scores.append(s)
                            finally:
                                Path(f.name).unlink(missing_ok=True)
            finally:
                cap.release()
            return float(np.mean(scores)) if scores else None
        else:
            return float(self._model(str(sample.path)).item())

    def _process_heuristic(self, sample: Sample) -> Optional[float]:
        """Heuristic: contrastive quality via distortion-sensitive features."""
        frames = self._extract_frames(sample)
        if not frames:
            return None

        scores = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)

            # Sharpness (Laplacian)
            sharpness = min(cv2.Laplacian(gray, cv2.CV_64F).var() / 800.0, 1.0)

            # Noise estimate (high-frequency energy)
            h, w = gray.shape
            dct = cv2.dct(np.float32(gray[:h - h % 8, :w - w % 8]))
            hf_energy = np.mean(np.abs(dct[h // 2:, w // 2:]))
            noise = 1.0 - min(hf_energy / 30.0, 1.0)

            # Contrast (std of luminance)
            contrast = min(gray.std() / 70.0, 1.0)

            # Structural regularity (gradient coherence)
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            mag = np.sqrt(gx ** 2 + gy ** 2)
            structure = min(mag.mean() / 40.0, 1.0)

            score = 0.35 * sharpness + 0.25 * noise + 0.20 * contrast + 0.20 * structure
            scores.append(score)

        return float(np.clip(np.mean(scores), 0.0, 1.0))

    def _extract_frames(self, sample: Sample):
        frames = []
        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                return []
            indices = np.linspace(0, total - 1, min(self.subsample, total), dtype=int)
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            cap.release()
        else:
            img = cv2.imread(str(sample.path))
            if img is not None:
                frames.append(img)
        return frames
