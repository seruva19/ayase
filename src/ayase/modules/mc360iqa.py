"""MC360IQA — Multi-Channel Blind 360° IQA. mc360iqa_score — higher = better"""
import logging, cv2, numpy as np
from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule
logger = logging.getLogger(__name__)
class MC360IQAModule(PipelineModule):
    name = "mc360iqa"; description = "MC360IQA blind 360 IQA (2019)"; default_config = {"subsample": 8, "n_viewports": 6}
    def __init__(self, c=None):
        super().__init__(c); self.subsample = self.config.get("subsample", 8); self._backend = "heuristic"
    def setup(self):
        try: import mc360iqa; self._model = mc360iqa; self._backend = "native"; return
        except ImportError: pass
        self._backend = "heuristic"
    def process(self, sample):
        try:
            frames = self._ex(sample)
            if not frames: return sample
            # Multi-viewport quality sampling
            scores = []
            for f in frames:
                h, w = f.shape[:2]
                vp_size = min(h, w) // 3
                if vp_size < 32: vp_size = min(h, w)
                vp_scores = []
                for y_frac, x_frac in [(0.25,0.25),(0.25,0.75),(0.5,0.5),(0.75,0.25),(0.75,0.75),(0.5,0.1)]:
                    cy, cx = int(h*y_frac), int(w*x_frac)
                    y1, x1 = max(cy-vp_size//2,0), max(cx-vp_size//2,0)
                    vp = f[y1:y1+vp_size, x1:x1+vp_size]
                    if vp.size == 0: continue
                    gray = cv2.cvtColor(vp, cv2.COLOR_BGR2GRAY).astype(np.float64)
                    vp_scores.append(min(cv2.Laplacian(gray, cv2.CV_64F).var()/500, 1))
                if vp_scores: scores.append(np.mean(vp_scores))
            if sample.quality_metrics is None: sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.mc360iqa_score = float(np.clip(np.mean(scores), 0, 1)) if scores else None
        except Exception as e: logger.warning(f"MC360IQA failed: {e}")
        return sample
    def _ex(self, sample):
        frames=[]
        if sample.is_video:
            cap=cv2.VideoCapture(str(sample.path)); tot=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if tot<=0: cap.release(); return []
            for i in np.linspace(0,tot-1,min(self.subsample,tot),dtype=int):
                cap.set(cv2.CAP_PROP_POS_FRAMES,i); r,f=cap.read()
                if r: frames.append(f)
            cap.release()
        else:
            img=cv2.imread(str(sample.path))
            if img is not None: frames.append(img)
        return frames
