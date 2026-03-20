"""StableVQA — Video Stability Quality Assessment.

ACM MM 2023 — first model specifically targeting video stability
perception via optical flow + semantic + blur features.

GitHub: https://github.com/QMME/StableVQA

stablevqa_score — higher = better (more stable)
"""

import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class StableVQAModule(PipelineModule):
    name = "stablevqa"
    description = "StableVQA video stability quality assessment (ACM MM 2023)"
    default_config = {
        "step": 2,
        "max_frames": 120,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.step = self.config.get("step", 2)
        self.max_frames = self.config.get("max_frames", 120)
        self._model = None
        self._backend = "heuristic"

    def setup(self) -> None:
        try:
            import stablevqa
            self._model = stablevqa
            self._backend = "native"
            logger.info("StableVQA (native) initialised")
            return
        except ImportError:
            pass

        self._backend = "heuristic"
        logger.info("StableVQA (heuristic) — install stablevqa for full model")

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
            return sample

        try:
            score = (
                self._process_native(sample)
                if self._backend == "native"
                else self._process_heuristic(sample)
            )

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.stablevqa_score = score

        except Exception as e:
            logger.warning(f"StableVQA failed for {sample.path}: {e}")

        return sample

    def _process_native(self, sample: Sample) -> Optional[float]:
        return float(self._model.predict(str(sample.path)))

    def _process_heuristic(self, sample: Sample) -> Optional[float]:
        """Heuristic: optical flow variance + blur stability + frame consistency."""
        cap = cv2.VideoCapture(str(sample.path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 1:
            cap.release()
            return None

        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            return None

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.resize(prev_gray, (320, 240))

        flow_variances = []
        blur_scores = []
        frame_diffs = []
        frame_count = 0

        while frame_count < self.max_frames:
            for _ in range(self.step):
                ret, frame = cap.read()
                if not ret:
                    break
            if not ret:
                break

            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.resize(curr_gray, (320, 240))

            # Optical flow variance (instability indicator)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            flow_variances.append(float(np.var(mag)))

            # Blur score (Laplacian variance)
            blur = cv2.Laplacian(curr_gray, cv2.CV_64F).var()
            blur_scores.append(blur)

            # Frame difference
            diff = np.mean(np.abs(curr_gray.astype(float) - prev_gray.astype(float)))
            frame_diffs.append(diff)

            prev_gray = curr_gray
            frame_count += 1

        cap.release()

        if not flow_variances:
            return None

        # Stability score components
        # Low flow variance = stable
        mean_flow_var = np.mean(flow_variances)
        flow_stability = 1.0 / (1.0 + mean_flow_var * 0.05)

        # Consistent blur = stable (low variance of blur scores)
        blur_consistency = 1.0 / (1.0 + np.var(blur_scores) * 0.0001)

        # Smooth frame differences = stable
        diff_arr = np.array(frame_diffs)
        diff_smoothness = 1.0 / (1.0 + np.var(diff_arr) * 0.01)

        score = 0.50 * flow_stability + 0.25 * blur_consistency + 0.25 * diff_smoothness

        return float(np.clip(score, 0.0, 1.0))
