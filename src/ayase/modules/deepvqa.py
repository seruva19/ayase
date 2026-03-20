"""DeepVQA — Deep Video Quality Assessor with Spatiotemporal Masking.

Kim et al. ECCV 2018 — full-reference VQA using convolutional neural
networks with spatiotemporal masking. Models human visual sensitivity
to spatial and temporal distortions.

deepvqa_score — higher = better quality
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)


class DeepVQAModule(ReferenceBasedModule):
    name = "deepvqa"
    description = "DeepVQA spatiotemporal masking FR-VQA (ECCV 2018)"
    metric_field = "deepvqa_score"
    default_config = {
        "subsample": 8,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._model = None
        self._backend = "heuristic"

    def setup(self) -> None:
        # Tier 1: Try native DeepVQA
        try:
            import deepvqa
            self._model = deepvqa
            self._backend = "native"
            logger.info("DeepVQA (native) initialised")
            return
        except ImportError:
            pass

        # Tier 2: Heuristic fallback
        self._backend = "heuristic"
        logger.info("DeepVQA (heuristic) initialised — install deepvqa for full model")

    def compute_reference_score(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        if self._backend == "native":
            return float(self._model.predict(str(sample_path), str(reference_path)))
        return self._compute_heuristic(sample_path, reference_path)

    def _read_frames(self, path: Path) -> list:
        """Read frames from video or image."""
        frames = []
        is_video = path.suffix.lower() in {
            ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv",
        }

        if is_video:
            cap = cv2.VideoCapture(str(path))
            try:
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total <= 0:
                    return frames
                indices = np.linspace(0, total - 1, min(self.subsample, total), dtype=int)
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frames.append(frame)
            finally:
                cap.release()
        else:
            img = cv2.imread(str(path))
            if img is not None:
                frames.append(img)

        return frames

    def _compute_heuristic(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        """Heuristic: spatiotemporal masking + error aggregation."""
        dist_frames = self._read_frames(sample_path)
        ref_frames = self._read_frames(reference_path)

        if not dist_frames or not ref_frames:
            return None

        # Match frame counts
        n_frames = min(len(dist_frames), len(ref_frames))
        dist_frames = dist_frames[:n_frames]
        ref_frames = ref_frames[:n_frames]

        frame_scores = []
        prev_dist_gray = None
        prev_ref_gray = None

        for i in range(n_frames):
            dist_gray = cv2.cvtColor(dist_frames[i], cv2.COLOR_BGR2GRAY).astype(np.float64)
            ref_gray = cv2.cvtColor(ref_frames[i], cv2.COLOR_BGR2GRAY).astype(np.float64)

            # Resize if needed
            if dist_gray.shape != ref_gray.shape:
                dist_gray = cv2.resize(dist_gray, (ref_gray.shape[1], ref_gray.shape[0]))

            # Spatial error
            spatial_err = np.mean((dist_gray - ref_gray) ** 2)

            # Spatial masking: weight errors by reference texture complexity
            ref_lap = cv2.Laplacian(ref_gray, cv2.CV_64F)
            spatial_mask = np.abs(ref_lap)
            # Normalize masking (areas with more texture tolerate more error)
            mask_weight = 1.0 / (1.0 + spatial_mask * 0.01)
            masked_err = np.mean(mask_weight * (dist_gray - ref_gray) ** 2)

            # Temporal masking
            temporal_mask_weight = 1.0
            if prev_dist_gray is not None and prev_ref_gray is not None:
                ref_motion = np.mean(np.abs(ref_gray - prev_ref_gray))
                # High motion areas tolerate more error
                temporal_mask_weight = 1.0 / (1.0 + ref_motion * 0.02)

            weighted_err = masked_err * temporal_mask_weight

            # Convert error to quality (PSNR-like mapping)
            if weighted_err > 0:
                quality = min(10.0 * np.log10(255.0 ** 2 / (weighted_err + 1e-7)) / 50.0, 1.0)
            else:
                quality = 1.0

            frame_scores.append(quality)
            prev_dist_gray = dist_gray
            prev_ref_gray = ref_gray

        score = float(np.mean(frame_scores))
        return float(np.clip(score, 0.0, 1.0))
