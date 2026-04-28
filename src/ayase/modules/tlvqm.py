"""TLVQM (Two-Level Video Quality Model) module.

Two-level NR-VQA: Level 1 extracts per-frame spatial features,
Level 2 extracts temporal features across frames.

Backend tiers:
  1. **CNN-TLVQM** — CNN feature extractor + SVR head
     (from ``github.com/jarikorhonen/cnn-tlvqm``)
  2. **Handcrafted** — traditional spatial+temporal features with
     heuristic linear weights
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class TLVQMModule(PipelineModule):
    name = "tlvqm"
    description = "Two-level video quality model (CNN-TLVQM or handcrafted fallback)"
    default_config = {"subsample": 8}

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._cnn_model = None
        self._svr_model = None
        self._backend = "handcrafted"
        self._device = "cpu"
        self.mos_min = self.config.get("mos_min", 1.0)
        self.mos_max = self.config.get("mos_max", 5.0)

    def setup(self) -> None:
        # Tier 1: CNN-TLVQM model (custom fine-tuned weights)
        try:
            import torch
            from pathlib import Path

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            models_dir = Path(self.config.get("models_dir", "models")) / "tlvqm"
            cnn_path = models_dir / "cnn_tlvqm.pth"
            svr_path = models_dir / "tlvqm_svr.pkl"

            if cnn_path.exists():
                # Load pre-trained CNN feature extractor
                from torchvision.models import resnet18
                cnn = resnet18(weights=None)
                # Remove classification head, use as feature extractor
                cnn.fc = torch.nn.Identity()
                cnn.load_state_dict(
                    torch.load(cnn_path, map_location=device, weights_only=True),
                    strict=False,
                )
                cnn = cnn.to(device).eval()
                self._cnn_model = cnn
                self._device = device
                self._backend = "cnn"
                logger.info("TLVQM loaded CNN feature extractor on %s", device)

                if svr_path.exists():
                    import joblib
                    self._svr_model = joblib.load(svr_path)
                    self._backend = "cnn_svr"
                    logger.info("TLVQM loaded SVR regressor from %s", svr_path)
                else:
                    logger.warning(
                        "TLVQM SVR regressor not found at '%s'. "
                        "CNN features will be mapped with a heuristic head. "
                        "To enable the full CNN-SVR pipeline, obtain "
                        "'tlvqm_svr.pkl' from the CNN-TLVQM repository "
                        "(https://github.com/jarikorhonen/cnn-tlvqm) and "
                        "place it in '%s'.",
                        svr_path,
                        models_dir,
                    )
                return
            else:
                logger.info(
                    "TLVQM custom CNN weights not found at '%s'. "
                    "To enable the CNN-TLVQM tier, obtain 'cnn_tlvqm.pth' "
                    "(and optionally 'tlvqm_svr.pkl') from the CNN-TLVQM "
                    "repository (https://github.com/jarikorhonen/cnn-tlvqm) "
                    "and place them in '%s'. Trying pretrained ResNet18 instead.",
                    cnn_path,
                    models_dir,
                )
        except ImportError:
            logger.info(
                "PyTorch or torchvision not installed; CNN-TLVQM tier unavailable."
            )
        except Exception as e:
            logger.warning("CNN-TLVQM loading failed: %s", e)

        # Tier 1b: Try ResNet18 pretrained features + heuristic
        try:
            import torch
            from torchvision.models import resnet18, ResNet18_Weights

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cnn = resnet18(weights=ResNet18_Weights.DEFAULT)
            cnn.fc = torch.nn.Identity()
            cnn = cnn.to(device).eval()
            self._cnn_model = cnn
            self._device = device
            self._backend = "cnn_pretrained"
            logger.info("TLVQM using pretrained ResNet18 features on %s", device)
            return
        except ImportError:
            logger.info(
                "torchvision not available for pretrained ResNet18; "
                "falling back to handcrafted features."
            )
        except Exception as e:
            logger.warning("ResNet18 pretrained loading failed: %s", e)

        # Tier 2: Handcrafted
        self._backend = "handcrafted"
        logger.info("TLVQM using handcrafted features")

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        try:
            import cv2

            frames = self._load_frames(sample)
            if not frames:
                return sample

            if self._backend in ("cnn", "cnn_svr", "cnn_pretrained"):
                score = self._process_cnn(frames, sample)
            else:
                score = self._process_handcrafted(frames, sample)

            if score is not None:
                sample.quality_metrics.tlvqm_score = float(np.clip(score, 0.0, 1.0))
        except Exception as e:
            logger.warning("TLVQM failed: %s", e)
        return sample

    def _process_cnn(self, frames, sample: Sample) -> Optional[float]:
        """Process using CNN feature extraction."""
        import torch
        import cv2

        # Extract CNN features per frame
        frame_features = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (224, 224))
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            # Normalize with ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            tensor = (tensor - mean) / std
            tensor = tensor.to(self._device)

            with torch.no_grad():
                features = self._cnn_model(tensor)
                frame_features.append(features.cpu().numpy().flatten())

        # Average features across frames
        avg_features = np.mean(frame_features, axis=0)

        # Temporal features from CNN
        if len(frame_features) >= 2:
            feat_diffs = []
            for i in range(len(frame_features) - 1):
                diff = np.linalg.norm(frame_features[i + 1] - frame_features[i])
                feat_diffs.append(diff)
            temporal_stats = np.array([
                np.mean(feat_diffs), np.std(feat_diffs),
                np.min(feat_diffs), np.max(feat_diffs)
            ])
        else:
            temporal_stats = np.zeros(4)

        combined = np.concatenate([avg_features, temporal_stats])

        if self._backend == "cnn_svr" and self._svr_model is not None:
            prediction = self._svr_model.predict(combined.reshape(1, -1))[0]
            return float((prediction - self.mos_min) / (self.mos_max - self.mos_min))

        # Heuristic proxy: without the trained SVR regressor, we map the
        # L2 norm of CNN features to [0,1].  This is *not* calibrated to
        # subjective MOS — install the SVR model for accurate predictions.
        feat_norm = np.linalg.norm(avg_features)
        spatial_quality = min(1.0, feat_norm / 50.0)

        if len(temporal_stats) >= 2 and temporal_stats[0] > 0:
            consistency = 1.0 / (1.0 + temporal_stats[1] / (temporal_stats[0] + 1e-8))
        else:
            consistency = 0.8

        return float(0.6 * spatial_quality + 0.4 * consistency)

    def _process_handcrafted(self, frames, sample: Sample) -> Optional[float]:
        """Process using handcrafted features."""
        import cv2

        # Level 1: Spatial features per frame
        level1_scores = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)

            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness = min(1.0, lap_var / 500.0)

            hp = gray - cv2.GaussianBlur(gray, (7, 7), 1.5)
            noise_est = np.median(np.abs(hp)) * 1.4826
            noise_free = max(0.0, 1.0 - noise_est / 25.0)

            contrast = min(1.0, gray.std() / 64.0)

            mu = cv2.GaussianBlur(gray, (7, 7), 7 / 6)
            sigma = np.sqrt(cv2.GaussianBlur((gray - mu) ** 2, (7, 7), 7 / 6) + 1e-7)
            mscn = (gray - mu) / (sigma + 1.0)
            mscn_kurtosis = float(np.mean(mscn ** 4) / (np.mean(mscn ** 2) ** 2 + 1e-8))
            naturalness = max(0.0, 1.0 - abs(mscn_kurtosis - 3.0) / 6.0)

            l1_score = 0.3 * sharpness + 0.25 * noise_free + 0.25 * contrast + 0.2 * naturalness
            level1_scores.append(l1_score)

        # Level 2: Temporal features
        level2_score = 0.8
        if len(frames) >= 2 and sample.is_video:
            flow_magnitudes = []
            flow_stds = []
            frame_diffs = []
            flicker_vals = []

            prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
            for i in range(1, len(frames)):
                curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                flow_magnitudes.append(np.mean(mag))
                flow_stds.append(np.std(mag))

                diff = np.mean(np.abs(prev_gray.astype(float) - curr_gray.astype(float)))
                frame_diffs.append(diff)

                if i >= 2:
                    prev_prev = cv2.cvtColor(frames[i - 2], cv2.COLOR_BGR2GRAY).astype(float)
                    d1 = prev_gray.astype(float) - prev_prev
                    d2 = curr_gray.astype(float) - prev_gray.astype(float)
                    flicker = np.mean(np.abs(d2 - d1))
                    flicker_vals.append(flicker)

                prev_gray = curr_gray

            mean_flow = np.mean(flow_magnitudes)
            motion_natural = float(np.exp(-0.5 * ((mean_flow - 5.0) / 8.0) ** 2))

            flow_var = np.std(flow_magnitudes) / (np.mean(flow_magnitudes) + 1e-8)
            smoothness = max(0.0, 1.0 - flow_var)

            flicker_score = max(0.0, 1.0 - np.mean(flicker_vals) / 20.0) if flicker_vals else 1.0

            diff_consistency = max(0.0, 1.0 - np.std(frame_diffs) / (np.mean(frame_diffs) + 1e-8))

            level2_score = (0.30 * motion_natural + 0.30 * smoothness +
                            0.20 * flicker_score + 0.20 * diff_consistency)

        spatial_score = float(np.mean(level1_scores))
        return float(0.5 * spatial_score + 0.5 * level2_score)

    def _load_frames(self, sample: Sample) -> list:
        import cv2

        subsample = self.config.get("subsample", 8)
        frames = []
        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = list(range(0, total, max(1, total // subsample)))[:subsample]
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            cap.release()
        else:
            frame = cv2.imread(str(sample.path))
            if frame is not None:
                frames.append(frame)
        return frames
