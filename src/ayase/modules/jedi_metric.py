"""JEDi (JEPA Embedding Distance) distribution metric.

ICLR 2025. Replaces FVD with V-JEPA features + kernel-based MMD.
Requires 16% of FVD's samples, 34% better human alignment.
Batch/distribution metric.
"""

import logging
from typing import Optional, List

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class JEDiModule(PipelineModule):
    name = "jedi"
    description = "JEDi distribution metric (V-JEPA + MMD, ICLR 2025)"
    default_config = {
        "num_frames": 16,
        "batch_size": 8,
        "trust_remote_code": True,
        "model_revision": None,
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._model = None
        self._device = None
        self._feature_cache: List[np.ndarray] = []

    def setup(self) -> None:
        try:
            import torch
            from transformers import AutoModel

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # V-JEPA model for video feature extraction
            model_name = "facebook/vjepa-giant"
            trc = self.config.get("trust_remote_code", True)
            rev = self.config.get("model_revision", None)
            self._model = AutoModel.from_pretrained(
                model_name, trust_remote_code=trc, revision=rev
            ).to(device)
            self._model.eval()
            self._device = device
            self._ml_available = True
            logger.info("V-JEPA model loaded for JEDi on %s", device)
        except (ImportError, Exception) as e:
            logger.warning("JEDi unavailable (V-JEPA not loaded): %s", e)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available:
            return sample
        if not sample.is_video:
            return sample

        try:
            features = self._extract_features(sample)
            if features is not None:
                self._feature_cache.append(features)
        except Exception as e:
            logger.warning("JEDi feature extraction failed: %s", e)
        return sample

    def _extract_features(self, sample: Sample) -> Optional[np.ndarray]:
        import cv2
        import torch

        num_frames = self.config.get("num_frames", 16)
        cap = cv2.VideoCapture(str(sample.path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = list(range(0, total, max(1, total // num_frames)))[:num_frames]

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (224, 224))
                frames.append(rgb)
        cap.release()

        if len(frames) < 2:
            return None

        # Pad to num_frames if needed
        while len(frames) < num_frames:
            frames.append(frames[-1])

        video_tensor = (
            torch.from_numpy(np.array(frames))
            .permute(0, 3, 1, 2)
            .float()
            / 255.0
        )
        video_tensor = video_tensor.unsqueeze(0).to(self._device)  # (1, T, C, H, W)

        with torch.no_grad():
            outputs = self._model(video_tensor)
            if hasattr(outputs, "last_hidden_state"):
                features = outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze()
            elif hasattr(outputs, "pooler_output"):
                features = outputs.pooler_output.cpu().numpy().squeeze()
            else:
                features = outputs[0].mean(dim=1).cpu().numpy().squeeze()

        return features

    def on_dispose(self) -> None:
        """Compute JEDi score from accumulated features."""
        if len(self._feature_cache) < 10:
            logger.info("JEDi: Not enough samples (%d) for distribution metric", len(self._feature_cache))
            self._feature_cache = []
            return

        try:
            features = np.array(self._feature_cache)
            # Compute MMD self-consistency (lower = more consistent distribution)
            n = len(features)
            mid = n // 2
            set1 = features[:mid]
            set2 = features[mid:]
            mmd = self._compute_mmd(set1, set2)
            logger.info("JEDi MMD score: %.4f (computed from %d samples)", mmd, n)
            if hasattr(self, "pipeline") and self.pipeline:
                if hasattr(self.pipeline, "add_dataset_metric"):
                    self.pipeline.add_dataset_metric("jedi", mmd)
        except Exception as e:
            logger.warning("JEDi computation failed: %s", e)
        finally:
            self._feature_cache = []

    @staticmethod
    def _compute_mmd(x: np.ndarray, y: np.ndarray) -> float:
        """Compute Maximum Mean Discrepancy with Gaussian kernel."""
        from scipy.spatial.distance import cdist

        sigma = np.median(cdist(x, y, "sqeuclidean"))
        if sigma == 0:
            sigma = 1.0

        xx = np.exp(-cdist(x, x, "sqeuclidean") / (2 * sigma))
        yy = np.exp(-cdist(y, y, "sqeuclidean") / (2 * sigma))
        xy = np.exp(-cdist(x, y, "sqeuclidean") / (2 * sigma))

        mmd = xx.mean() + yy.mean() - 2 * xy.mean()
        return float(max(0, mmd))


class JEDiCompatModule(JEDiModule):
    """Compatibility alias matching filename-based discovery."""

    name = "jedi_metric"
