"""Optical flow coherence module.

From CogVideoX pipeline. Measures bidirectional optical flow
consistency — forward flow composed with backward flow should
return to the origin. High coherence = physically plausible motion.
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class FlowCoherenceModule(PipelineModule):
    name = "flow_coherence"
    description = "Bidirectional optical flow consistency (0-1, higher=coherent)"
    default_config = {"subsample": 8}

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not sample.is_video:
            return sample

        try:
            import cv2

            subsample = self.config.get("subsample", 8)
            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = list(range(0, total, max(1, total // subsample)))[:subsample]

            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            cap.release()

            if len(frames) < 2:
                return sample

            coherences = []
            for i in range(len(frames) - 1):
                f1, f2 = frames[i], frames[i + 1]

                # Forward flow: f1 -> f2
                fwd = cv2.calcOpticalFlowFarneback(
                    f1, f2, None, 0.5, 3, 15, 3, 5, 1.2, 0,
                )
                # Backward flow: f2 -> f1
                bwd = cv2.calcOpticalFlowFarneback(
                    f2, f1, None, 0.5, 3, 15, 3, 5, 1.2, 0,
                )

                # Cycle consistency: fwd + bwd should be ~zero
                h, w = f1.shape
                # Sample grid points — vectorised
                ys = np.arange(0, h, 8)
                xs = np.arange(0, w, 8)
                grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
                grid_y = grid_y.ravel()
                grid_x = grid_x.ravel()

                # Forward flow at grid points
                dx_f = fwd[grid_y, grid_x, 0]
                dy_f = fwd[grid_y, grid_x, 1]

                # Destination in frame 2
                x2 = np.rint(grid_x + dx_f).astype(int)
                y2 = np.rint(grid_y + dy_f).astype(int)

                # Keep only points that land inside frame 2
                valid = (x2 >= 0) & (x2 < w) & (y2 >= 0) & (y2 < h)

                if valid.any():
                    x2v = x2[valid]
                    y2v = y2[valid]
                    dx_fv = dx_f[valid]
                    dy_fv = dy_f[valid]

                    dx_b = bwd[y2v, x2v, 0]
                    dy_b = bwd[y2v, x2v, 1]

                    err = np.sqrt((dx_fv + dx_b) ** 2 + (dy_fv + dy_b) ** 2)
                    mean_err = float(err.mean())
                    coherences.append(1.0 / (1.0 + mean_err))

            if coherences:
                sample.quality_metrics.flow_coherence = float(np.mean(coherences))
        except Exception as e:
            logger.warning("Flow coherence computation failed: %s", e)
        return sample
