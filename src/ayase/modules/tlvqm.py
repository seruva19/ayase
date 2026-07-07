"""TLVQM (Two-Level Video Quality Model) module.

Two-level NR-VQA: Level 1 extracts per-frame spatial features, Level 2 extracts
temporal features across frames; a trained SVR regressor maps the features to a
subjective MOS.

This module implements the CNN-TLVQM variant, which requires BOTH trained
artefacts from the CNN-TLVQM repository
(https://github.com/jarikorhonen/cnn-tlvqm) placed under
``<models_dir>/tlvqm/``:

  * ``cnn_tlvqm.pth`` — CNN feature-extractor weights
  * ``tlvqm_svr.pkl`` — trained SVR regressor

Without the trained SVR the model cannot predict a calibrated MOS. Earlier
revisions fell back to ImageNet ResNet-18 features / handcrafted features fed
through an *uncalibrated* heuristic mapping (``feat_norm / 50``); those proxies
did not reproduce TLVQM and have been removed. When the trained artefacts are
absent the metric is reported as unavailable.

Backend: **cnn_svr** — CNN feature extractor + trained SVR regressor.
"""

import logging
from typing import Optional

import numpy as np

from ayase.image import sample_frames
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class TLVQMModule(PipelineModule):
    name = "tlvqm"
    description = "Two-level video quality model (CNN-TLVQM, trained CNN+SVR)"
    default_config = {"subsample": 8}
    metric_groups = {
        "tlvqm_score": "nr_quality",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._cnn_model = None
        self._svr_model = None
        self._ml_available = False
        self._backend = None
        self._device = "cpu"
        self.mos_min = self.config.get("mos_min", 1.0)
        self.mos_max = self.config.get("mos_max", 5.0)

    def setup(self) -> None:
        if getattr(self, "test_mode", False):
            return
        try:
            import torch
            from pathlib import Path
            from ayase.runtime import resolve_torch_device

            models_dir = Path(self.config.get("models_dir", "models")) / "tlvqm"
            cnn_path = models_dir / "cnn_tlvqm.pth"
            svr_path = models_dir / "tlvqm_svr.pkl"

            if not (cnn_path.exists() and svr_path.exists()):
                self._backend = "unavailable"
                logger.info(
                    "TLVQM: trained artefacts not found (need '%s' and '%s' from "
                    "the CNN-TLVQM repository). Metric unavailable.",
                    cnn_path,
                    svr_path,
                )
                return

            import joblib
            from torchvision.models import resnet18

            self._device = resolve_torch_device(self.config.get("device", "auto"))

            cnn = resnet18(weights=None)
            cnn.fc = torch.nn.Identity()
            cnn.load_state_dict(
                torch.load(cnn_path, map_location=self._device, weights_only=True),
                strict=False,
            )
            self._cnn_model = cnn.to(self._device).eval()
            self._svr_model = joblib.load(svr_path)
            self._backend = "cnn_svr"
            self._ml_available = True
            logger.info(
                "TLVQM loaded CNN feature extractor + trained SVR on %s",
                self._device,
            )
        except ImportError:
            self._backend = "unavailable"
            logger.info("TLVQM: torch/torchvision/joblib not installed; metric unavailable.")
        except Exception as e:
            self._backend = "unavailable"
            logger.warning("TLVQM CNN-TLVQM loading failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        try:
            frames = self._load_frames(sample)
            if not frames:
                return sample

            score = self._process_cnn(frames)
            if score is not None:
                sample.quality_metrics.tlvqm_score = float(np.clip(score, 0.0, 1.0))
        except Exception as e:
            logger.warning("TLVQM failed: %s", e)
        return sample

    def _process_cnn(self, frames) -> Optional[float]:
        """Extract CNN features per frame and map to MOS via the trained SVR."""
        import torch

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        frame_features = []
        for frame in frames:
            import cv2

            rgb = cv2.resize(np.ascontiguousarray(frame), (224, 224))
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            tensor = (tensor - mean) / std
            tensor = tensor.to(self._device)
            with torch.no_grad():
                features = self._cnn_model(tensor)
            frame_features.append(features.cpu().numpy().flatten())

        avg_features = np.mean(frame_features, axis=0)

        if len(frame_features) >= 2:
            feat_diffs = [
                np.linalg.norm(frame_features[i + 1] - frame_features[i])
                for i in range(len(frame_features) - 1)
            ]
            temporal_stats = np.array(
                [np.mean(feat_diffs), np.std(feat_diffs), np.min(feat_diffs), np.max(feat_diffs)]
            )
        else:
            temporal_stats = np.zeros(4)

        combined = np.concatenate([avg_features, temporal_stats])
        prediction = self._svr_model.predict(combined.reshape(1, -1))[0]
        return float((prediction - self.mos_min) / (self.mos_max - self.mos_min))

    def _load_frames(self, sample: Sample) -> list:
        subsample = self.config.get("subsample", 8)
        try:
            return sample_frames(sample.path, max_frames=subsample, color="rgb")
        except Exception as e:
            logger.debug("TLVQM frame loading failed: %s", e)
            return []
