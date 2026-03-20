"""SER-FIQ — Stochastic Embedding Robustness Face Quality (2020). serfiq_score — higher = better"""
import logging, cv2, numpy as np
from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule
logger = logging.getLogger(__name__)
class SERFIQModule(PipelineModule):
    name = "serfiq"; description = "SER-FIQ face quality via embedding robustness (2020)"; default_config = {"subsample": 4}
    def __init__(self, c=None):
        super().__init__(c); self.subsample = self.config.get("subsample", 4); self._backend = "heuristic"; self._face_cascade = None
    def setup(self):
        try:
            import serfiq; self._model = serfiq; self._backend = "native"; return
        except ImportError:
            pass
        self._face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self._backend = "heuristic"
    def process(self, sample):
        try:
            if self._backend == "native" and self._model is not None:
                score = float(self._model.predict(str(sample.path)))
                if sample.quality_metrics is None: sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.serfiq_score = float(np.clip(score, 0, 1))
                return sample
            frames = self._ex(sample)
            if not frames: return sample
            scores = []
            for f in frames:
                gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                faces = self._face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30,30)) if self._face_cascade else []
                if len(faces) == 0: continue
                for (x,y,w,h) in faces:
                    roi = gray[y:y+h, x:x+w].astype(np.float64)
                    sharpness = min(cv2.Laplacian(roi, cv2.CV_64F).var()/300, 1)
                    contrast = min(roi.std()/50, 1)
                    size_q = min(w*h/10000, 1)  # Larger face = better
                    scores.append(0.4*sharpness + 0.3*contrast + 0.3*size_q)
            if scores:
                if sample.quality_metrics is None: sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.serfiq_score = float(np.clip(np.mean(scores), 0, 1))
        except Exception as e: logger.warning(f"SER-FIQ failed: {e}")
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
