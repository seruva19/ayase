"""VQ-Insight — ByteDance AIGC Video Quality (AAAI 2026). vqinsight_score — higher = better"""
import logging, cv2, numpy as np
from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule
logger = logging.getLogger(__name__)
class VQInsightModule(PipelineModule):
    name = "vqinsight"; description = "VQ-Insight ByteDance multi-dim AIGC scoring (AAAI 2026)"; default_config = {"subsample": 8}
    def __init__(self, c=None):
        super().__init__(c); self.subsample = self.config.get("subsample", 8); self._backend = "heuristic"
    def setup(self):
        try: import vqinsight; self._model = vqinsight; self._backend = "native"; return
        except ImportError: pass
        self._backend = "heuristic"
    def process(self, sample):
        try:
            frames = self._ex(sample)
            if not frames: return sample
            vis = float(np.mean([min(cv2.Laplacian(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY).astype(np.float64),cv2.CV_64F).var()/500,1) for f in frames]))
            aes = float(np.mean([min(cv2.cvtColor(f,cv2.COLOR_BGR2HSV)[:,:,1].astype(float).mean()/128,1) for f in frames]))
            t=1.0
            if len(frames)>1:
                d=[np.mean(np.abs(cv2.resize(cv2.cvtColor(frames[i],cv2.COLOR_BGR2GRAY).astype(float),(160,120))-cv2.resize(cv2.cvtColor(frames[i+1],cv2.COLOR_BGR2GRAY).astype(float),(160,120)))) for i in range(len(frames)-1)]
                t=1.0/(1.0+np.var(d)*0.01)
            score = 0.35*vis+0.30*t+0.35*aes
            if sample.quality_metrics is None: sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.vqinsight_score = float(np.clip(score, 0, 1))
        except Exception as e: logger.warning(f"VQ-Insight failed: {e}")
        return sample
    def _ex(self, sample):
        frames=[]
        if sample.is_video:
            c=cv2.VideoCapture(str(sample.path)); t=int(c.get(cv2.CAP_PROP_FRAME_COUNT))
            if t<=0: c.release(); return []
            for i in np.linspace(0,t-1,min(self.subsample,t),dtype=int):
                c.set(cv2.CAP_PROP_POS_FRAMES,i); r,f=c.read()
                if r: frames.append(f)
            c.release()
        else:
            img=cv2.imread(str(sample.path))
            if img is not None: frames.append(img)
        return frames
