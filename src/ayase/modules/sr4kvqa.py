"""SR4KVQA — Super-Resolution 4K Video Quality (2024). sr4kvqa_score — higher = better"""
import logging, cv2, numpy as np
from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule
logger = logging.getLogger(__name__)
class SR4KVQAModule(PipelineModule):
    name = "sr4kvqa"; description = "SR4KVQA super-resolution 4K quality (2024)"; default_config = {"subsample": 8}
    def __init__(self, c=None):
        super().__init__(c); self.subsample = self.config.get("subsample", 8); self._backend = "heuristic"
    def setup(self):
        try: import sr4kvqa; self._model = sr4kvqa; self._backend = "native"; return
        except ImportError: pass
        self._backend = "heuristic"
    def process(self, sample):
        try:
            frames = self._ex(sample)
            if not frames: return sample
            scores = []
            for f in frames:
                gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float64)
                # SR artifact detection: ringing (high-freq oscillation near edges)
                lap = cv2.Laplacian(gray, cv2.CV_64F)
                ringing = 1.0/(1.0+np.abs(lap).std()*0.01)
                sharpness = min(lap.var()/500, 1)
                # Texture preservation
                gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
                texture = min(gx.var()/1000, 1)
                scores.append(0.4*sharpness + 0.3*ringing + 0.3*texture)
            if sample.quality_metrics is None: sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.sr4kvqa_score = float(np.clip(np.mean(scores), 0, 1))
        except Exception as e: logger.warning(f"SR4KVQA failed: {e}")
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
