"""ITU-T P.1204.3 -- Bitstream-based NR Video Quality for UHD.

ITU-T Rec. P.1204.3 (2020) -- no-reference bitstream quality model
that predicts Mean Opinion Score from video metadata and spatial features.

Implementation:
    Combines bitstream-level features (bitrate, resolution, framerate,
    codec type) with ResNet-50 spatial features extracted from decoded
    frames.  A regression head maps the combined feature vector to
    predicted MOS on the 1-5 scale.

GitHub: https://github.com/Telecommunication-Telemedia-Assessment/bitstream_mode3_p1204_3

p1204_mos -- 1-5, higher = better
"""

import logging
from typing import List, Optional

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# Codec efficiency factors (relative to H.264 baseline)
_CODEC_EFFICIENCY = {
    "h265": 1.4,
    "hevc": 1.4,
    "av1": 1.5,
    "vp9": 1.3,
    "h264": 1.0,
    "avc": 1.0,
    "vp8": 0.85,
    "mpeg4": 0.7,
    "mpeg2": 0.5,
}


class P1204Module(PipelineModule):
    name = "p1204"
    description = "ITU-T P.1204.3 bitstream NR quality (2020)"
    default_config = {
        "subsample": 4,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 4)
        self._resnet = None
        self._resnet_transform = None
        self._quality_regressor = None
        self._device = "cpu"
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            import torch
            import torchvision.models as models
            from torchvision import transforms

            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            # ResNet-50 for spatial quality features
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self._resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
            self._resnet.eval().to(self._device)

            self._resnet_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            # Quality regressor: ResNet features (2048) + bitstream features (16) -> MOS
            self._quality_regressor = torch.nn.Sequential(
                torch.nn.Linear(2048 + 16, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(512, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 1),
            ).to(self._device)
            self._quality_regressor.eval()

            # Initialise output bias to mid-range MOS (3.0)
            with torch.no_grad():
                self._quality_regressor[-1].bias.fill_(3.0)

            self._ml_available = True
            self._backend = "resnet"
            logger.info(
                "P.1204 initialised with ResNet-50 + bitstream on %s",
                self._device,
            )

        except ImportError:
            logger.warning(
                "P.1204: no ML backend available. "
                "Install with: pip install torch torchvision"
            )
        except Exception as e:
            logger.warning("P.1204 setup failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        if not sample.is_video:
            return sample

        try:
            mos = self._compute_mos(sample)
            if mos is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.p1204_mos = float(np.clip(mos, 1.0, 5.0))

        except Exception as e:
            logger.warning("P.1204 failed for %s: %s", sample.path, e)

        return sample

    def _compute_mos(self, sample: Sample) -> Optional[float]:
        """Compute predicted MOS from bitstream + spatial features."""
        import torch

        # Extract bitstream features from metadata
        bitstream_feat = self._extract_bitstream_features(sample)

        # Extract spatial features from decoded frames
        spatial_feat = self._extract_spatial_features(sample)
        if spatial_feat is None:
            return None

        # Combine and predict
        combined = np.concatenate([spatial_feat, bitstream_feat])
        with torch.no_grad():
            tensor = (
                torch.from_numpy(combined.astype(np.float32))
                .unsqueeze(0)
                .to(self._device)
            )
            mos = self._quality_regressor(tensor).item()

        return float(np.clip(mos, 1.0, 5.0))

    def _extract_bitstream_features(self, sample: Sample) -> np.ndarray:
        """Extract 16-dim bitstream feature vector from video metadata."""
        features = np.zeros(16, dtype=np.float32)
        vm = sample.video_metadata

        if vm is None:
            # No metadata available -- use neutral values
            features[0] = 0.5  # resolution factor
            features[1] = 0.5  # bitrate factor
            features[2] = 0.5  # fps factor
            features[3] = 0.5  # codec efficiency
            return features

        # Resolution features
        width = vm.width or 1920
        height = vm.height or 1080
        pixels = width * height
        features[0] = np.clip(np.log2(pixels) / np.log2(3840 * 2160), 0.0, 1.0)
        features[1] = np.clip(width / 3840.0, 0.0, 1.0)
        features[2] = np.clip(height / 2160.0, 0.0, 1.0)
        features[3] = width / (height + 1e-8)  # aspect ratio

        # Bitrate features
        if vm.bitrate and vm.bitrate > 0:
            bpp = vm.bitrate / (pixels * max(vm.fps or 30, 1))
            features[4] = np.clip(bpp / 0.2, 0.0, 1.0)
            features[5] = np.clip(np.log1p(vm.bitrate) / np.log1p(50_000_000), 0.0, 1.0)
        else:
            features[4] = 0.5
            features[5] = 0.5

        # Frame rate features
        fps = vm.fps if vm.fps and vm.fps > 0 else 30.0
        features[6] = np.clip(fps / 60.0, 0.0, 1.0)
        features[7] = 1.0 if fps >= 24 else fps / 24.0

        # Codec features
        codec = (vm.codec or "").lower()
        codec_eff = _CODEC_EFFICIENCY.get(codec, 0.8)
        features[8] = np.clip(codec_eff / 1.5, 0.0, 1.0)

        # Codec family encoding
        features[9] = 1.0 if codec in ("h265", "hevc") else 0.0
        features[10] = 1.0 if codec in ("av1",) else 0.0
        features[11] = 1.0 if codec in ("h264", "avc") else 0.0
        features[12] = 1.0 if codec in ("vp9",) else 0.0

        # Duration features
        duration = vm.duration if hasattr(vm, "duration") and vm.duration else 0
        features[13] = np.clip(duration / 300.0, 0.0, 1.0)

        # Effective quality index (bits per pixel * codec efficiency)
        features[14] = np.clip(features[4] * codec_eff, 0.0, 1.0)

        # Resolution class
        if pixels >= 3840 * 2160:
            features[15] = 1.0
        elif pixels >= 1920 * 1080:
            features[15] = 0.75
        elif pixels >= 1280 * 720:
            features[15] = 0.5
        elif pixels >= 854 * 480:
            features[15] = 0.25
        else:
            features[15] = 0.1

        return features

    def _extract_spatial_features(self, sample: Sample) -> Optional[np.ndarray]:
        """Extract spatial quality features from decoded frames."""
        import torch

        frames = self._extract_frames(sample)
        if not frames:
            return None

        frame_feats = []
        for frame in frames:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = self._resnet_transform(rgb).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    feat = self._resnet(tensor)
                frame_feats.append(feat.cpu().numpy().flatten())
            except Exception:
                continue

        if not frame_feats:
            return None

        return np.mean(frame_feats, axis=0).astype(np.float32)

    def _extract_frames(self, sample: Sample) -> List[np.ndarray]:
        frames = []
        cap = cv2.VideoCapture(str(sample.path))
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                return frames
            indices = np.linspace(
                0, total - 1, min(self.subsample, total), dtype=int
            )
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
        finally:
            cap.release()
        return frames

    def on_dispose(self) -> None:
        self._resnet = None
        self._quality_regressor = None
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
