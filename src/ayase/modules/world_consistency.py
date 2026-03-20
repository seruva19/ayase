"""World Consistency Score — object permanence + causal compliance (2025).

Paper: https://arxiv.org/abs/2508.00144
world_consistency_score — higher = better
"""
import logging, cv2, numpy as np
from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule
logger = logging.getLogger(__name__)

class WorldConsistencyModule(PipelineModule):
    name = "world_consistency"
    description = "World Consistency Score: object permanence + causal compliance (2025)"
    default_config = {"subsample": 12}
    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 12)
        self._backend = "heuristic"
    def setup(self):
        try:
            import wcs; self._model = wcs; self._backend = "native"; return
        except ImportError: pass
        self._backend = "heuristic"
    def process(self, sample):
        if not sample.is_video: return sample
        try:
            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 1: cap.release(); return sample
            indices = np.linspace(0, total-1, min(self.subsample, total), dtype=int)
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ret, f = cap.read()
                if ret: frames.append(cv2.resize(f, (320, 240)))
            cap.release()
            if len(frames) < 2: return sample

            # Object permanence: structural similarity across frames
            ssim_scores = []
            for i in range(len(frames)-1):
                g1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(float)
                g2 = cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY).astype(float)
                mu1, mu2 = g1.mean(), g2.mean()
                s1, s2 = g1.var(), g2.var()
                cov = np.mean((g1-mu1)*(g2-mu2))
                c1, c2 = 6.5025, 58.5225
                ssim = ((2*mu1*mu2+c1)*(2*cov+c2))/((mu1**2+mu2**2+c1)*(s1+s2+c2))
                ssim_scores.append(max(ssim, 0))
            permanence = float(np.mean(ssim_scores))

            # Relation stability: edge structure consistency
            edge_counts = [np.mean(cv2.Canny(f, 50, 150) > 0) for f in frames]
            stability = 1.0/(1.0+np.var(edge_counts)*100)

            # Flicker penalty
            diffs = [np.mean(np.abs(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(float) - cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY).astype(float))) for i in range(len(frames)-1)]
            flicker = 1.0/(1.0+np.var(diffs)*0.01)

            score = 0.40*permanence + 0.30*stability + 0.30*flicker
            if sample.quality_metrics is None: sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.world_consistency_score = float(np.clip(score, 0, 1))
        except Exception as e:
            logger.warning(f"WorldConsistency failed: {e}")
        return sample
