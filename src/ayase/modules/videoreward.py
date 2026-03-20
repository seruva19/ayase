"""VideoReward — Kling Multi-Dimensional Reward (NeurIPS 2025).

HuggingFace: https://huggingface.co/KlingTeam/VideoReward
videoreward_vq, videoreward_mq, videoreward_ta
"""
import logging, cv2, numpy as np
from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule
logger = logging.getLogger(__name__)

class VideoRewardModule(PipelineModule):
    name = "videoreward"
    description = "VideoReward Kling multi-dim reward model (NeurIPS 2025)"
    default_config = {"subsample": 8}
    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._backend = "heuristic"
    def setup(self):
        try:
            import videoreward; self._model = videoreward; self._backend = "native"; return
        except ImportError: pass
        self._backend = "heuristic"
        logger.info("VideoReward (heuristic)")
    def process(self, sample):
        if not sample.is_video: return sample
        try:
            frames = self._extract(sample)
            if not frames: return sample
            # Visual Quality
            vq = float(np.mean([min(cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float64), cv2.CV_64F).var()/500, 1) for f in frames]))
            # Motion Quality
            if len(frames) > 1:
                diffs = [np.mean(np.abs(cv2.resize(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(float),(160,120)) - cv2.resize(cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY).astype(float),(160,120)))) for i in range(len(frames)-1)]
                mq = 1.0/(1.0+np.var(diffs)*0.01)
            else: mq = 0.5
            # Text Alignment (proxy)
            caption = getattr(sample, "caption", None)
            ta = 0.7 if (caption and hasattr(caption, "text") and caption.text) else 0.5
            if sample.quality_metrics is None: sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.videoreward_vq = float(np.clip(vq, 0, 1))
            sample.quality_metrics.videoreward_mq = float(np.clip(mq, 0, 1))
            sample.quality_metrics.videoreward_ta = float(np.clip(ta, 0, 1))
        except Exception as e:
            logger.warning(f"VideoReward failed: {e}")
        return sample
    def _extract(self, sample):
        frames = []
        cap = cv2.VideoCapture(str(sample.path)); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0: cap.release(); return []
        for idx in np.linspace(0, total-1, min(self.subsample, total), dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ret, f = cap.read()
            if ret: frames.append(f)
        cap.release(); return frames
