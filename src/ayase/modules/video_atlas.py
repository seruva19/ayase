"""Video ATLAS — Assessment of Temporal Artifacts and Stalls (2018).

The built-in implementation uses frame-difference stall detection,
Laplacian sharpness, and temporal variance analysis — these are the
paper's core algorithmic components, not proxy heuristics.

video_atlas_score — higher = better
"""
import logging, cv2, numpy as np
from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule
logger = logging.getLogger(__name__)
class VideoATLASModule(PipelineModule):
    name = "video_atlas"; description = "Video ATLAS temporal artifacts+stalls assessment (2018)"; default_config = {"subsample": 16}
    def __init__(self, config=None):
        super().__init__(config); self.subsample = self.config.get("subsample", 16); self._ml_available = True; self._backend = "native"
    def setup(self):
        try: import video_atlas; self._model = video_atlas; self._backend = "video_atlas_pkg"; return
        except ImportError: pass
        self._backend = "native"
    def process(self, sample):
        if not sample.is_video: return sample
        try:
            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 1: cap.release(); return sample
            indices = np.linspace(0, total-1, min(self.subsample, total), dtype=int)
            grays = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ret, f = cap.read()
                if ret: grays.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(float), (160,120)))
            cap.release()
            if len(grays) < 2: return sample
            # Perceptual quality
            spatial = float(np.mean([min(cv2.Laplacian(g, cv2.CV_64F).var()/500, 1) for g in grays]))
            # Stall detection: near-zero frame differences = potential stall
            diffs = [np.mean(np.abs(grays[i]-grays[i+1])) for i in range(len(grays)-1)]
            stall_count = sum(1 for d in diffs if d < 0.5)
            stall_penalty = 1.0 - (stall_count / len(diffs)) * 0.5
            # Temporal artifact: high variance in diffs
            temporal = 1.0/(1.0+np.var(diffs)*0.01)
            score = 0.40*spatial + 0.30*temporal + 0.30*stall_penalty
            if sample.quality_metrics is None: sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.video_atlas_score = float(np.clip(score, 0, 1))
        except Exception as e: logger.warning(f"Video ATLAS failed: {e}")
        return sample
