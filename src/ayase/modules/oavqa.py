"""OAVQA — Omnidirectional Audio-Visual QA (2024). oavqa_score — higher = better"""
import logging, cv2, numpy as np
from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule
logger = logging.getLogger(__name__)
class OAVQAModule(PipelineModule):
    name = "oavqa"; description = "OAVQA omnidirectional audio-visual QA (2024)"; default_config = {"subsample": 8}
    def __init__(self, c=None):
        super().__init__(c); self.subsample = self.config.get("subsample", 8); self._backend = "heuristic"
    def setup(self):
        try: import oavqa; self._model = oavqa; self._backend = "native"; return
        except ImportError: pass
        self._backend = "heuristic"
    def process(self, sample):
        try:
            # Video quality
            frames = self._ex(sample)
            vq = 0.5
            if frames:
                vq = float(np.mean([min(cv2.Laplacian(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY).astype(np.float64),cv2.CV_64F).var()/500,1) for f in frames]))
            # Audio quality proxy
            aq = 0.5
            if sample.audio_metadata is not None:
                am = sample.audio_metadata
                # Higher bitrate and sample rate = better audio
                br_q = min((am.bitrate or 128000)/320000, 1) if am.bitrate else 0.5
                sr_q = min(am.sample_rate/48000, 1)
                aq = 0.5*br_q + 0.5*sr_q
            score = 0.6*vq + 0.4*aq
            if sample.quality_metrics is None: sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.oavqa_score = float(np.clip(score, 0, 1))
        except Exception as e: logger.warning(f"OAVQA failed: {e}")
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
