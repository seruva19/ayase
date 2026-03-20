"""Unified-VQA — Unified Video Quality Assessment (FR+NR Multi-task).

2025 — combines spatial sharpness, temporal consistency, and
reference similarity (when available) into a unified quality score.
Operates in NR mode when no reference is provided, FR mode otherwise.

Stores result in dover_score as proxy (shared NR quality field).
"""

import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class UnifiedVQAModule(PipelineModule):
    name = "unified_vqa"
    description = "Unified-VQA FR+NR multi-task quality assessment (2025)"
    default_config = {
        "subsample": 8,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._model = None
        self._backend = "heuristic"

    def setup(self) -> None:
        # Tier 1: Try native unified_vqa
        try:
            import unified_vqa
            self._model = unified_vqa
            self._backend = "native"
            logger.info("Unified-VQA (native) initialised")
            return
        except ImportError:
            pass

        # Tier 2: Heuristic fallback
        self._backend = "heuristic"
        logger.info("Unified-VQA (heuristic) initialised — install unified_vqa for full model")

    def process(self, sample: Sample) -> Sample:
        try:
            if self._backend == "native":
                score = float(self._model.predict(str(sample.path)))
            else:
                score = self._process_heuristic(sample)

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                # Store in dover_score as proxy (shared NR quality field)
                # Only set if not already set by DOVER module
                if sample.quality_metrics.dover_score is None:
                    sample.quality_metrics.dover_score = score

        except Exception as e:
            logger.warning(f"Unified-VQA failed for {sample.path}: {e}")

        return sample

    def _process_heuristic(self, sample: Sample) -> Optional[float]:
        """Heuristic: spatial sharpness + temporal consistency + reference similarity."""
        frames = []
        reference_path = getattr(sample, "reference_path", None)

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
                        frames.append(frame)
            finally:
                cap.release()
        else:
            img = cv2.imread(str(sample.path))
            if img is not None:
                frames.append(img)

        if not frames:
            return None

        # Component 1: Spatial sharpness
        sharpness_scores = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_scores.append(min(lap_var / 800.0, 1.0))
        spatial_sharpness = float(np.mean(sharpness_scores))

        # Component 2: Spatial quality (contrast + noise)
        spatial_quality_scores = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
            contrast = min(gray.std() / 65.0, 1.0)
            # Noise estimation via high-frequency energy
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1)
            noise_est = sobel.var()
            cleanliness = 1.0 / (1.0 + noise_est * 0.0001)
            spatial_quality_scores.append(0.5 * contrast + 0.5 * cleanliness)
        spatial_quality = float(np.mean(spatial_quality_scores))

        # Component 3: Temporal consistency
        if len(frames) > 1:
            grays = [
                cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float64) for f in frames
            ]
            diffs = []
            for i in range(len(grays) - 1):
                diffs.append(np.mean(np.abs(grays[i + 1] - grays[i])))
            diffs = np.array(diffs)
            # Low variance in diffs = smooth temporal transitions
            temporal_consistency = 1.0 / (1.0 + np.var(diffs) * 0.01)
        else:
            temporal_consistency = 1.0

        # Component 4: Reference similarity (FR mode if reference available)
        ref_similarity = None
        if reference_path is not None:
            try:
                from pathlib import Path
                ref_path = Path(reference_path)
                if ref_path.exists():
                    ref_img = cv2.imread(str(ref_path))
                    if ref_img is not None:
                        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY).astype(np.float64)
                        dist_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).astype(np.float64)

                        if dist_gray.shape != ref_gray.shape:
                            dist_gray = cv2.resize(
                                dist_gray, (ref_gray.shape[1], ref_gray.shape[0])
                            )

                        # SSIM-like comparison
                        c1 = (0.01 * 255) ** 2
                        c2 = (0.03 * 255) ** 2
                        mu_d = cv2.GaussianBlur(dist_gray, (11, 11), 1.5)
                        mu_r = cv2.GaussianBlur(ref_gray, (11, 11), 1.5)
                        sigma_d2 = cv2.GaussianBlur(dist_gray ** 2, (11, 11), 1.5) - mu_d ** 2
                        sigma_r2 = cv2.GaussianBlur(ref_gray ** 2, (11, 11), 1.5) - mu_r ** 2
                        sigma_dr = (
                            cv2.GaussianBlur(dist_gray * ref_gray, (11, 11), 1.5)
                            - mu_d * mu_r
                        )
                        ssim_map = ((2 * mu_d * mu_r + c1) * (2 * sigma_dr + c2)) / (
                            (mu_d ** 2 + mu_r ** 2 + c1) * (sigma_d2 + sigma_r2 + c2)
                        )
                        ref_similarity = float(np.mean(ssim_map))
            except Exception:
                pass

        # Combine components
        if ref_similarity is not None:
            # FR mode: include reference similarity
            score = (
                0.25 * spatial_sharpness
                + 0.20 * spatial_quality
                + 0.20 * temporal_consistency
                + 0.35 * ref_similarity
            )
        else:
            # NR mode: rely on spatial and temporal features
            score = (
                0.35 * spatial_sharpness
                + 0.30 * spatial_quality
                + 0.35 * temporal_consistency
            )

        return float(np.clip(score, 0.0, 1.0))
