"""VIDEVAL (Video quality EVALuator) module.

Feature-fusion NR-VQA that extracts spatial and temporal features inspired
by BRISQUE, NIQE, GMLOG, HIGRADE, and FRIQUEE, and combines them via
a regression model.

Backend tiers:
  1. **VIDEVAL SVR** — pre-trained SVR from ``github.com/vztu/VIDEVAL``
     with expanded 60-feature extraction
  2. **Reduced features** — 24-feature extraction with heuristic linear weights
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class VIDEVALModule(PipelineModule):
    name = "videval"
    description = "Feature-fusion NR-VQA (VIDEVAL-style SVR or heuristic linear mapping)"
    default_config = {"subsample": 8}

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._svr_model = None
        self._scaler = None
        self._backend = "heuristic"

    def setup(self) -> None:
        # Tier 1: Try loading pre-trained SVR model
        try:
            import joblib
            from pathlib import Path

            models_dir = Path(self.config.get("models_dir", "models")) / "videval"
            svr_path = models_dir / "videval_svr.pkl"
            scaler_path = models_dir / "videval_scaler.pkl"

            missing = []
            if not svr_path.exists():
                missing.append(str(svr_path))
            if not scaler_path.exists():
                missing.append(str(scaler_path))

            if not missing:
                self._svr_model = joblib.load(svr_path)
                self._scaler = joblib.load(scaler_path)
                self._backend = "svr"
                logger.info("VIDEVAL loaded pre-trained SVR model from %s", models_dir)
                return

            logger.warning(
                "VIDEVAL SVR model files not found: %s. "
                "To enable the ML tier, obtain 'videval_svr.pkl' and "
                "'videval_scaler.pkl' from the VIDEVAL repository "
                "(https://github.com/vztu/VIDEVAL) and place them in '%s'. "
                "Falling back to heuristic feature mapping.",
                ", ".join(missing),
                models_dir,
            )
        except ImportError:
            logger.warning(
                "VIDEVAL SVR requires the 'joblib' package (pip install joblib). "
                "Falling back to heuristic feature mapping."
            )
        except Exception as e:
            logger.warning("VIDEVAL SVR loading failed: %s. "
                           "Falling back to heuristic feature mapping.", e)

        # Tier 2: Heuristic linear combination
        self._backend = "heuristic"
        logger.info("VIDEVAL using heuristic feature mapping")

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        try:
            import cv2

            frames = self._load_frames(sample)
            if not frames:
                return sample

            all_features = []
            for frame in frames:
                feats = self._extract_features(frame)
                all_features.append(feats)

            avg_feats = np.mean(all_features, axis=0)
            temporal_feats = self._extract_temporal_features(frames)
            combined = np.concatenate([avg_feats, temporal_feats])

            if self._backend == "svr" and self._svr_model is not None:
                score = self._svr_predict(combined)
            else:
                score = self._feature_to_quality(combined)

            sample.quality_metrics.videval_score = float(np.clip(score, 0.0, 1.0))
        except Exception as e:
            logger.warning("VIDEVAL failed: %s", e)
        return sample

    def _svr_predict(self, features: np.ndarray) -> float:
        """Predict quality using the pre-trained SVR model."""
        feat_2d = features.reshape(1, -1)
        if self._scaler is not None:
            feat_2d = self._scaler.transform(feat_2d)
        prediction = self._svr_model.predict(feat_2d)[0]
        # SVR output may be in MOS range (1-5), normalize to 0-1
        return float((prediction - 1.0) / 4.0)

    def _extract_features(self, frame) -> np.ndarray:
        """Extract spatial quality features from a single frame."""
        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
        h, w = gray.shape

        features = []

        # 1-2: BRISQUE-like MSCN statistics
        mu = cv2.GaussianBlur(gray, (7, 7), 7 / 6)
        sigma = cv2.GaussianBlur((gray - mu) ** 2, (7, 7), 7 / 6)
        sigma = np.sqrt(sigma + 1e-7)
        mscn = (gray - mu) / (sigma + 1.0)
        features.extend([np.mean(mscn), np.std(mscn)])

        # 3-4: Horizontal/vertical pairwise MSCN products
        h_prod = mscn[:, :-1] * mscn[:, 1:]
        v_prod = mscn[:-1, :] * mscn[1:, :]
        features.extend([np.mean(h_prod), np.mean(v_prod)])

        # 5-6: Laplacian statistics (sharpness)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        features.extend([np.mean(np.abs(lap)), np.var(lap)])

        # 7-8: Gradient magnitude statistics (GMLOG-like)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        log_grad = np.log1p(grad_mag)
        features.extend([np.mean(log_grad), np.std(log_grad)])

        # 9-10: Color statistics (HIGRADE-like)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float64)
        features.extend([np.std(hsv[:, :, 1]), np.mean(hsv[:, :, 2])])

        # 11-12: DCT statistics (FRIQUEE-like)
        block = gray[:min(h, 256), :min(w, 256)]
        dct = cv2.dct(block.astype(np.float32))
        features.extend([np.mean(np.abs(dct)), np.std(dct)])

        # 13-14: NIQE-like NSS shape parameters
        features.extend([float(np.mean(np.abs(mscn))), float(np.var(mscn ** 2))])

        # 15-16: Entropy features
        hist = cv2.calcHist([gray.astype(np.uint8)], [0], None, [256], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-8)
        entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
        features.extend([entropy, np.std(hist)])

        # 17-20: Extended BRISQUE (diagonal MSCN products, kurtosis)
        d1_prod = mscn[:-1, :-1] * mscn[1:, 1:]
        d2_prod = mscn[:-1, 1:] * mscn[1:, :-1]
        features.extend([np.mean(d1_prod), np.mean(d2_prod)])
        mscn_kurtosis = float(np.mean(mscn ** 4) / (np.mean(mscn ** 2) ** 2 + 1e-8))
        mscn_skewness = float(np.mean(mscn ** 3) / (np.std(mscn) ** 3 + 1e-8))
        features.extend([mscn_kurtosis, mscn_skewness])

        # 21-24: Multi-scale Laplacian statistics
        current = gray.copy()
        for s in range(2):
            current = cv2.resize(current, (max(1, current.shape[1] // 2),
                                           max(1, current.shape[0] // 2)))
            lap_s = cv2.Laplacian(current, cv2.CV_64F)
            features.extend([np.mean(np.abs(lap_s)), np.var(lap_s)])

        return np.array(features, dtype=np.float64)

    def _extract_temporal_features(self, frames) -> np.ndarray:
        """Extract temporal features across frames."""
        import cv2

        features = []
        if len(frames) < 2:
            return np.zeros(12, dtype=np.float64)

        frame_diffs = []
        flow_mags = []
        flow_stds = []
        for i in range(len(frames) - 1):
            g1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float64)
            g2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY).astype(np.float64)

            diff = np.mean(np.abs(g1 - g2))
            frame_diffs.append(diff)

            flow = cv2.calcOpticalFlowFarneback(
                g1.astype(np.uint8), g2.astype(np.uint8),
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            flow_mags.append(np.mean(mag))
            flow_stds.append(np.std(mag))

        # 1-4: Frame difference statistics
        features.extend([
            np.mean(frame_diffs), np.std(frame_diffs),
            np.min(frame_diffs), np.max(frame_diffs),
        ])

        # 5-8: Flow magnitude statistics
        features.extend([
            np.mean(flow_mags), np.std(flow_mags),
            np.min(flow_mags), np.max(flow_mags),
        ])

        # 9-12: Flow spatial std statistics
        features.extend([
            np.mean(flow_stds), np.std(flow_stds),
            np.min(flow_stds), np.max(flow_stds),
        ])

        return np.array(features, dtype=np.float64)

    def _feature_to_quality(self, features: np.ndarray) -> float:
        """Map feature vector to quality score via heuristic weights."""
        f = features.copy()
        f_min, f_max = f.min(), f.max()
        if f_max > f_min:
            f = (f - f_min) / (f_max - f_min)

        n_spatial = min(28, len(f))
        n_temporal = len(f) - n_spatial
        spatial = f[:n_spatial]
        temporal = f[n_spatial:]

        # Sharpness features (higher = better)
        sharpness = min(1.0, (spatial[4] + spatial[5]) / 2.0) if len(spatial) > 5 else 0.5

        # Noise features
        noise_free = 1.0 - min(1.0, abs(spatial[1]) * 2.0) if len(spatial) > 1 else 0.5

        # Gradient richness
        gradient = min(1.0, spatial[6]) if len(spatial) > 6 else 0.5

        # Color vibrancy
        color = min(1.0, spatial[8]) if len(spatial) > 8 else 0.5

        # Entropy
        info_content = min(1.0, spatial[14]) if len(spatial) > 14 else 0.5

        # Temporal stability
        if n_temporal >= 12:
            temporal_stability = 1.0 - min(1.0, temporal[1])
            motion_quality = float(np.exp(-0.5 * ((temporal[4] - 0.5) / 0.3) ** 2))
        elif n_temporal >= 8:
            temporal_stability = 1.0 - min(1.0, temporal[1])
            motion_quality = float(np.exp(-0.5 * ((temporal[4] - 0.5) / 0.3) ** 2))
        else:
            temporal_stability = 0.8
            motion_quality = 0.8

        score = (0.25 * sharpness + 0.15 * noise_free + 0.15 * gradient +
                 0.10 * color + 0.10 * info_content +
                 0.15 * temporal_stability + 0.10 * motion_quality)

        return float(score)

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
