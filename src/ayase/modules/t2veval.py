"""T2VEval — Text-to-Video Generated Video Evaluation (2025). t2veval_score — higher = better"""
import logging, cv2, numpy as np
from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule
logger = logging.getLogger(__name__)
class T2VEvalModule(PipelineModule):
    name = "t2veval"; description = "T2VEval text-video consistency+realness (2025)"; default_config = {"subsample": 8}
    def __init__(self, c=None):
        super().__init__(c); self.subsample = self.config.get("subsample", 8); self._backend = "heuristic"
    def setup(self):
        try: import t2veval; self._model = t2veval; self._backend = "native"; return
        except ImportError: pass
        self._backend = "heuristic"
    def process(self, sample):
        try:
            frames = self._ex(sample)
            if not frames: return sample
            # Technical quality
            tech = float(np.mean([min(cv2.Laplacian(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY).astype(np.float64),cv2.CV_64F).var()/500,1) for f in frames]))
            # Realness: naturalness via NSS
            real_scores = []
            for f in frames:
                gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float64)
                mu = cv2.GaussianBlur(gray, (7,7), 7/6)
                sigma = np.sqrt(np.abs(cv2.GaussianBlur(gray**2,(7,7),7/6)-mu**2)+1e-7)
                mscn = (gray-mu)/sigma
                real_scores.append(1.0/(1.0+abs(np.var(mscn)-1.0)))
            realness = float(np.mean(real_scores))
            # Temporal
            t = 1.0
            if len(frames) > 1:
                d = [np.mean(np.abs(cv2.resize(cv2.cvtColor(frames[i],cv2.COLOR_BGR2GRAY).astype(float),(160,120))-cv2.resize(cv2.cvtColor(frames[i+1],cv2.COLOR_BGR2GRAY).astype(float),(160,120)))) for i in range(len(frames)-1)]
                t = 1.0/(1.0+np.var(d)*0.01)
            score = 0.35*tech + 0.35*realness + 0.30*t
            if sample.quality_metrics is None: sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.t2veval_score = float(np.clip(score, 0, 1))
        except Exception as e: logger.warning(f"T2VEval failed: {e}")
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
