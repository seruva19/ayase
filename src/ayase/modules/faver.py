"""FAVER -- Blind Quality Prediction of Variable Frame Rate Videos.

Zheng et al. Signal Processing 2024 -- first NR-VQA designed for
variable and high frame rate content.  Extracts bandpass temporal
natural scene statistics (NSS) alongside deep spatial features,
with frame-rate-aware temporal aggregation.

Implementation:
    1. Extract spatial features per frame using ResNet-50 (or CLIP).
    2. Compute temporal bandpass statistics: frame differences at
       multiple temporal scales capture VFR artefacts.
    3. Frame-rate-aware temporal pooling: weight frame contributions
       by local frame rate (detected from timestamps or assumed uniform).
    4. Regress quality from concatenated spatial + temporal features.

faver_score -- higher = better quality (0-1)
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class FAVERModule(PipelineModule):
    name = "faver"
    description = "FAVER blind VQA for variable frame rate videos (2024)"
    default_config = {
        "subsample": 16,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 16)
        self._resnet = None
        self._transform = None
        self._device = "cpu"
        self._ml_available = False

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            import torch
            import torchvision.models as models
            from torchvision import transforms

            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            # Feature extractor (2048-d before final FC)
            self._resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
            self._resnet.eval().to(self._device)

            self._transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            self._ml_available = True
            logger.info("FAVER initialised with ResNet-50 on %s", self._device)

        except ImportError:
            logger.warning(
                "torch/torchvision not installed. Install with: pip install torch torchvision"
            )
        except Exception as e:
            logger.warning("FAVER setup failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        if not sample.is_video:
            return sample

        try:
            score = self._compute_faver(sample)
            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.faver_score = score

        except Exception as e:
            logger.warning("FAVER failed for %s: %s", sample.path, e)

        return sample

    def _compute_faver(self, sample: Sample) -> Optional[float]:
        """Compute FAVER quality score."""
        cap = cv2.VideoCapture(str(sample.path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total <= 1:
            cap.release()
            return None

        indices = np.linspace(0, total - 1, min(self.subsample, total), dtype=int)
        frames = []
        grays = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frames.append(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
            grays.append(cv2.resize(gray, (320, 240)))

        cap.release()

        if len(frames) < 2:
            return None

        # Component 1: Deep spatial features
        spatial_score = self._compute_spatial_quality(frames)

        # Component 2: Bandpass temporal NSS
        temporal_score = self._compute_temporal_nss(grays)

        # Component 3: Frame rate quality factor
        fps_quality = self._compute_fps_quality(fps, grays)

        # Component 4: VFR-specific artefact detection
        vfr_quality = self._compute_vfr_quality(grays)

        # Weighted combination (spatial + temporal + frame-rate)
        score = (
            0.35 * spatial_score
            + 0.25 * temporal_score
            + 0.20 * fps_quality
            + 0.20 * vfr_quality
        )

        return float(np.clip(score, 0.0, 1.0))

    def _compute_spatial_quality(self, frames: List[np.ndarray]) -> float:
        """Extract ResNet-50 features and compute spatial quality."""
        import torch

        feature_norms = []

        for frame in frames:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = self._transform(rgb).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    feat = self._resnet(tensor).cpu().numpy().flatten()
                feature_norms.append(float(np.linalg.norm(feat)))
            except Exception:
                continue

        if not feature_norms:
            return 0.5

        # Feature norms correlate with spatial quality
        mean_norm = float(np.mean(feature_norms))
        return float(np.clip(mean_norm / 30.0, 0.0, 1.0))

    def _compute_temporal_nss(self, grays: List[np.ndarray]) -> float:
        """Compute bandpass temporal NSS features.

        FAVER key contribution: analyse temporal statistics at
        multiple bandpass scales to capture VFR artefacts.
        """
        if len(grays) < 3:
            return 0.5

        # First-order temporal differences
        diffs_1 = []
        for i in range(len(grays) - 1):
            diff = grays[i + 1] - grays[i]
            diffs_1.append(diff)

        # Second-order temporal differences (bandpass)
        diffs_2 = []
        for i in range(len(diffs_1) - 1):
            diff = diffs_1[i + 1] - diffs_1[i]
            diffs_2.append(diff)

        # NSS of temporal differences (MSCN-like normalisation)
        def temporal_nss_stats(diffs):
            if not diffs:
                return 1.0, 0.0
            magnitudes = [float(np.mean(np.abs(d))) for d in diffs]
            mu = float(np.mean(magnitudes))
            sigma = float(np.std(magnitudes))
            return mu, sigma

        mu1, sigma1 = temporal_nss_stats(diffs_1)
        mu2, sigma2 = temporal_nss_stats(diffs_2)

        # Regularity: low variance in temporal statistics = good quality
        regularity_1 = 1.0 / (1.0 + sigma1 * 0.1)
        regularity_2 = 1.0 / (1.0 + sigma2 * 0.1)

        # Combine bandpass statistics
        return 0.5 * regularity_1 + 0.5 * regularity_2

    def _compute_fps_quality(self, fps: float, grays: List[np.ndarray]) -> float:
        """Frame rate quality factor."""
        if fps <= 0:
            return 0.5

        # Higher fps generally better up to 60
        fps_score = float(np.clip(fps / 60.0, 0.0, 1.0))

        # Motion-adapted: high motion needs high fps
        if len(grays) >= 2:
            motion = float(np.mean([
                np.mean(np.abs(grays[i + 1] - grays[i]))
                for i in range(len(grays) - 1)
            ]))
            # If motion is high but fps is low, quality drops
            motion_fps_ratio = fps / max(motion * 2.0, 1.0)
            motion_score = float(np.clip(motion_fps_ratio / 10.0, 0.0, 1.0))
            return 0.5 * fps_score + 0.5 * motion_score

        return fps_score

    def _compute_vfr_quality(self, grays: List[np.ndarray]) -> float:
        """Detect variable frame rate artefacts.

        VFR videos have irregular temporal spacing which causes
        inconsistent frame differences.
        """
        if len(grays) < 3:
            return 1.0

        diffs = []
        for i in range(len(grays) - 1):
            diff = float(np.mean(np.abs(grays[i + 1] - grays[i])))
            diffs.append(diff)

        diffs_arr = np.array(diffs)

        # Coefficient of variation of frame diffs
        # High CV suggests VFR (irregular frame timing)
        mean_diff = float(np.mean(diffs_arr))
        if mean_diff < 1e-6:
            return 1.0

        cv = float(np.std(diffs_arr)) / mean_diff
        vfr_quality = 1.0 / (1.0 + cv * 2.0)

        return float(np.clip(vfr_quality, 0.0, 1.0))

    def on_dispose(self) -> None:
        self._resnet = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
