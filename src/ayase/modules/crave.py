"""CRAVE — Content-Rich AIGC Video Evaluator (2025).

Designed for Sora-era videos with multi-granularity text-temporal
fusion and hybrid motion-fidelity modeling.

GitHub: https://github.com/littlespray/CRAVE

crave_score — higher = better quality
"""

import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class CRAVEModule(PipelineModule):
    name = "crave"
    description = "CRAVE content-rich AIGC video evaluator (2025)"
    default_config = {"subsample": 12}

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 12)
        self._backend = "heuristic"

    def setup(self) -> None:
        try:
            import crave
            self._model = crave
            self._backend = "native"
            logger.info("CRAVE (native) initialised")
            return
        except ImportError:
            pass
        self._backend = "heuristic"
        logger.info("CRAVE (heuristic)")

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
            return sample
        try:
            score = (
                float(self._model.predict(str(sample.path)))
                if self._backend == "native"
                else self._process_heuristic(sample)
            )
            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.crave_score = score
        except Exception as e:
            logger.warning(f"CRAVE failed for {sample.path}: {e}")
        return sample

    def _process_heuristic(self, sample: Sample) -> Optional[float]:
        cap = cv2.VideoCapture(str(sample.path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 1:
            cap.release()
            return None
        indices = np.linspace(0, total - 1, min(self.subsample, total), dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()
        if len(frames) < 2:
            return None

        # Motion fidelity: optical flow consistency
        flow_vars = []
        for i in range(len(frames) - 1):
            g1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
            g1 = cv2.resize(g1, (320, 240))
            g2 = cv2.resize(g2, (320, 240))
            flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            flow_vars.append(np.var(mag))
        motion_fidelity = 1.0 / (1.0 + np.mean(flow_vars) * 0.02)

        # Temporal coherence
        diffs = []
        for i in range(len(frames) - 1):
            g1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(float)
            g2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY).astype(float)
            diffs.append(np.mean(np.abs(cv2.resize(g1, (160, 120)) - cv2.resize(g2, (160, 120)))))
        temporal_coherence = 1.0 / (1.0 + np.var(diffs) * 0.01)

        # Visual quality
        qualities = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
            qualities.append(min(cv2.Laplacian(gray, cv2.CV_64F).var() / 500.0, 1.0))
        visual = float(np.mean(qualities))

        score = 0.35 * visual + 0.35 * motion_fidelity + 0.30 * temporal_coherence
        return float(np.clip(score, 0.0, 1.0))
