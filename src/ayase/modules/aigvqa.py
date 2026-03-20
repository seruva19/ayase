"""AIGVQA — Multi-Dimensional AI-Generated VQA (ICCVW 2025).

GitHub: https://github.com/IntMeGroup/AIGVQA
aigvqa_score — higher = better
"""
import logging, cv2, numpy as np
from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule
logger = logging.getLogger(__name__)

class AIGVQAModule(PipelineModule):
    name = "aigvqa"
    description = "AIGVQA multi-dimensional AIGC VQA (ICCVW 2025)"
    default_config = {"subsample": 8}
    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._backend = "heuristic"
    def setup(self):
        try:
            import aigvqa; self._model = aigvqa; self._backend = "native"; return
        except ImportError: pass
        self._backend = "heuristic"
    def process(self, sample):
        try:
            frames = self._extract(sample)
            if not frames: return sample
            spatials, temporals, aesthetics = [], [], []
            for f in frames:
                g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float64)
                spatials.append(min(cv2.Laplacian(g, cv2.CV_64F).var()/500, 1))
                hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
                aesthetics.append(min(hsv[:,:,1].astype(float).mean()/128, 1))
            if len(frames) > 1:
                for i in range(len(frames)-1):
                    g1 = cv2.resize(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(float),(160,120))
                    g2 = cv2.resize(cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY).astype(float),(160,120))
                    temporals.append(np.mean(np.abs(g1-g2)))
                temp_score = 1.0/(1.0+np.var(temporals)*0.01)
            else: temp_score = 1.0
            score = 0.4*np.mean(spatials) + 0.3*temp_score + 0.3*np.mean(aesthetics)
            if sample.quality_metrics is None: sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.aigvqa_score = float(np.clip(score, 0, 1))
        except Exception as e:
            logger.warning(f"AIGVQA failed: {e}")
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
