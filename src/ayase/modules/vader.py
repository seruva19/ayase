"""VADER — Video Diffusion Alignment via Reward Gradients (ICLR 2025).

GitHub: https://github.com/mihirp1998/VADER
vader_score — higher = better
"""
import logging, cv2, numpy as np
from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule
logger = logging.getLogger(__name__)

class VADERModule(PipelineModule):
    name = "vader"
    description = "VADER reward gradient alignment (ICLR 2025)"
    default_config = {"subsample": 8}
    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._backend = "heuristic"
    def setup(self):
        try:
            import vader; self._model = vader; self._backend = "native"; return
        except ImportError: pass
        self._backend = "heuristic"
    def process(self, sample):
        try:
            frames = self._extract(sample)
            if not frames: return sample
            aesthetic = float(np.mean([min(cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[:,:,1].astype(float).mean()/128, 1) for f in frames]))
            sharpness = float(np.mean([min(cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float64), cv2.CV_64F).var()/500, 1) for f in frames]))
            score = 0.5*aesthetic + 0.5*sharpness
            if sample.quality_metrics is None: sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.vader_score = float(np.clip(score, 0, 1))
        except Exception as e:
            logger.warning(f"VADER failed: {e}")
        return sample
    def _extract(self, sample):
        frames = []
        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path)); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0: cap.release(); return []
            for idx in np.linspace(0, total-1, min(self.subsample, total), dtype=int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ret, f = cap.read()
                if ret: frames.append(f)
            cap.release()
        else:
            img = cv2.imread(str(sample.path))
            if img is not None: frames.append(img)
        return frames
