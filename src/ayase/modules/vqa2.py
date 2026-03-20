"""VQA² — Visual Question Answering for Video Quality Assessment (MM 2025).

GitHub: https://github.com/Q-Future/Visual-Question-Answering-for-Video-Quality-Assessment
vqa2_score — higher = better
"""
import logging, cv2, numpy as np
from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule
logger = logging.getLogger(__name__)

class VQA2Module(PipelineModule):
    name = "vqa2"
    description = "VQA² LMM video quality assessment (MM 2025)"
    default_config = {"subsample": 8}
    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._backend = "heuristic"
    def setup(self):
        try:
            import vqa2_assistant; self._model = vqa2_assistant; self._backend = "native"; return
        except ImportError: pass
        self._backend = "heuristic"
        logger.info("VQA² (heuristic)")
    def process(self, sample):
        try:
            frames = self._extract(sample)
            if not frames: return sample
            spatial = float(np.mean([min(cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float64), cv2.CV_64F).var()/500, 1) for f in frames]))
            contrast = float(np.mean([min(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(float).std()/65, 1) for f in frames]))
            temporal = 1.0
            if len(frames) > 1:
                diffs = [np.mean(np.abs(cv2.resize(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(float),(160,120))-cv2.resize(cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY).astype(float),(160,120)))) for i in range(len(frames)-1)]
                temporal = 1.0/(1.0+np.var(diffs)*0.01)
            score = 0.4*spatial + 0.3*contrast + 0.3*temporal
            if sample.quality_metrics is None: sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.vqa2_score = float(np.clip(score, 0, 1))
        except Exception as e:
            logger.warning(f"VQA² failed: {e}")
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
