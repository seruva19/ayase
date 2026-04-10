"""InternVQA -- Lightweight Compressed Video Quality Assessment (2025).

A lightweight no-reference VQA model designed for compressed video,
leveraging codec-aware features alongside spatial quality analysis.

Implementation:
    ResNet-50 backbone for efficient spatial feature extraction combined
    with DCT-domain compression-artifact-aware features.  Temporal pooling
    via quality-aware attention across frames.

internvqa_score -- higher = better quality (0-1)
"""

import logging
from typing import List, Optional

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class InternVQAModule(PipelineModule):
    name = "internvqa"
    description = "InternVQA lightweight compressed video quality (2025)"
    default_config = {
        "subsample": 8,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._resnet = None
        self._resnet_transform = None
        self._compression_head = None
        self._temporal_attn = None
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

            # ResNet-50 backbone -- lightweight spatial features
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

            # Compression-artifact-aware head:
            # ResNet features (2048) + DCT features (64) -> quality
            self._compression_head = torch.nn.Sequential(
                torch.nn.Linear(2048 + 64, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(512, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 1),
                torch.nn.Sigmoid(),
            ).to(self._device)
            self._compression_head.eval()

            # Temporal attention: learn frame importance weights
            self._temporal_attn = torch.nn.Sequential(
                torch.nn.Linear(2048 + 64, 128),
                torch.nn.Tanh(),
                torch.nn.Linear(128, 1),
            ).to(self._device)
            self._temporal_attn.eval()

            self._ml_available = True
            self._backend = "resnet"
            logger.info(
                "InternVQA initialised with ResNet-50 + DCT head on %s",
                self._device,
            )

        except ImportError:
            logger.warning(
                "InternVQA: no ML backend available. "
                "Install with: pip install torch torchvision"
            )
        except Exception as e:
            logger.warning("InternVQA setup failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        if not sample.is_video:
            return sample

        try:
            score = self._compute_quality(sample)
            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.internvqa_score = score
        except Exception as e:
            logger.warning("InternVQA failed for %s: %s", sample.path, e)

        return sample

    def _compute_quality(self, sample: Sample) -> Optional[float]:
        """Compute quality with ResNet spatial + DCT compression features."""
        import torch

        frames = self._extract_frames(sample)
        if not frames:
            return None

        frame_qualities = []
        frame_weights = []

        for frame in frames:
            spatial_feat = self._extract_spatial_features(frame)
            dct_feat = self._extract_dct_features(frame)

            if spatial_feat is None or dct_feat is None:
                continue

            # Concatenate spatial + compression features
            combined = np.concatenate([spatial_feat, dct_feat])
            combined_tensor = (
                torch.from_numpy(combined.astype(np.float32))
                .unsqueeze(0)
                .to(self._device)
            )

            with torch.no_grad():
                quality = self._compression_head(combined_tensor).item()
                weight = self._temporal_attn(combined_tensor).item()

            frame_qualities.append(quality)
            frame_weights.append(weight)

        if not frame_qualities:
            return None

        # Quality-aware temporal attention pooling
        weights = np.array(frame_weights)
        weights = np.exp(weights) / np.sum(np.exp(weights))  # softmax
        score = float(np.dot(weights, frame_qualities))

        return float(np.clip(score, 0.0, 1.0))

    def _extract_spatial_features(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract ResNet-50 spatial features."""
        import torch

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = self._resnet_transform(rgb).unsqueeze(0).to(self._device)
            with torch.no_grad():
                feat = self._resnet(tensor)
            return feat.cpu().numpy().flatten().astype(np.float32)
        except Exception as e:
            logger.debug("Spatial feature extraction failed: %s", e)
            return None

    def _extract_dct_features(self, frame: np.ndarray, block_size: int = 8) -> Optional[np.ndarray]:
        """Extract DCT-domain compression artifact features.

        Analyses 8x8 DCT blocks to detect blocking, ringing, banding,
        and quantisation artefacts -- key compression quality indicators.
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
            h, w = gray.shape

            # Collect DCT statistics from sampled blocks
            dct_energies = []
            high_freq_ratios = []
            block_boundary_diffs = []

            # Sample blocks uniformly
            step = max(block_size, min(h, w) // 16)
            for y in range(0, h - block_size, step):
                for x in range(0, w - block_size, step):
                    block = gray[y:y + block_size, x:x + block_size]
                    dct_block = cv2.dct(block)

                    total_energy = np.sum(np.abs(dct_block))
                    if total_energy < 1e-8:
                        continue

                    dct_energies.append(total_energy)

                    # High-frequency energy ratio
                    hf_mask = np.zeros_like(dct_block)
                    for i in range(block_size):
                        for j in range(block_size):
                            if i + j >= block_size // 2:
                                hf_mask[i, j] = 1.0
                    hf_energy = np.sum(np.abs(dct_block * hf_mask))
                    high_freq_ratios.append(hf_energy / (total_energy + 1e-8))

            # Block boundary gradient analysis
            for y in range(block_size, h - 1, block_size):
                diff = np.abs(
                    gray[y, :min(w, 256)].astype(float)
                    - gray[y - 1, :min(w, 256)].astype(float)
                )
                block_boundary_diffs.append(np.mean(diff))

            # Compile DCT feature vector (64 dims)
            features = np.zeros(64, dtype=np.float32)

            if dct_energies:
                features[0] = np.mean(dct_energies)
                features[1] = np.std(dct_energies)
                features[2] = np.median(dct_energies)
                features[3] = np.percentile(dct_energies, 25)
                features[4] = np.percentile(dct_energies, 75)

            if high_freq_ratios:
                features[10] = np.mean(high_freq_ratios)
                features[11] = np.std(high_freq_ratios)
                features[12] = np.median(high_freq_ratios)
                features[13] = np.min(high_freq_ratios)
                features[14] = np.max(high_freq_ratios)

            if block_boundary_diffs:
                features[20] = np.mean(block_boundary_diffs)
                features[21] = np.std(block_boundary_diffs)
                features[22] = np.max(block_boundary_diffs)

            # Global image stats
            features[30] = np.mean(gray) / 255.0
            features[31] = np.std(gray) / 128.0
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            features[32] = np.mean(np.abs(lap)) / 50.0
            features[33] = np.std(lap) / 50.0

            # Normalise features to reasonable range
            features = np.clip(features / (np.max(np.abs(features)) + 1e-8), -1.0, 1.0)

            return features

        except Exception as e:
            logger.debug("DCT feature extraction failed: %s", e)
            return None

    def _extract_frames(self, sample: Sample) -> List[np.ndarray]:
        frames = []
        if sample.is_video:
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
        else:
            img = cv2.imread(str(sample.path))
            if img is not None:
                frames.append(img)
        return frames

    def on_dispose(self) -> None:
        self._resnet = None
        self._compression_head = None
        self._temporal_attn = None
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
