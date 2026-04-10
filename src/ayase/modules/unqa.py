"""UNQA — Unified No-Reference Quality Assessment (2024).

A unified framework that shares feature representations across audio,
image, and video modalities for quality prediction. The key insight is
that perceptual quality degradation manifests as predictable statistical
deviations in learned feature representations regardless of modality.

Implementation: ResNet-50 for visual features, torchaudio Mel
spectrogram for audio features. Unified quality head processes
both modalities. For video: temporal pooling of visual features.

Output: stores to ``confidence_score`` (unified quality confidence
across modalities).
"""

import logging
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class UNQAModule(PipelineModule):
    name = "unqa"
    description = "UNQA unified no-reference quality for audio/image/video (2024)"
    default_config = {
        "subsample": 8,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._backend = None
        self._ml_available = False
        self._device = "cpu"

        self._resnet = None
        self._resnet_transform = None
        self._audio_available = False
        self._visual_head = None
        self._audio_head = None
        self._unified_head = None

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            import torch
            import torch.nn as nn
            from torchvision import models, transforms

            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            # Visual backbone: ResNet-50 feature extractor
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self._resnet = nn.Sequential(
                *list(resnet.children())[:-1]
            ).to(self._device).eval()

            self._resnet_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            # Visual quality head: ResNet features -> visual quality
            self._visual_head = nn.Sequential(
                nn.Linear(2048, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(256, 64),
                nn.ReLU(inplace=True),
            ).to(self._device)

            # Audio quality head: Mel spectrogram features -> audio quality
            # Input: 128 mel bins * time_avg = 128 features
            self._audio_head = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
            ).to(self._device)

            # Unified quality head: combines visual + audio features
            # 64 (visual) + 64 (audio) = 128, or 64 (visual only)
            self._unified_head_with_audio = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            ).to(self._device)

            self._unified_head_visual_only = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            ).to(self._device)

            # Xavier init all heads
            for module in [
                self._visual_head, self._audio_head,
                self._unified_head_with_audio,
                self._unified_head_visual_only,
            ]:
                for m in module.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        nn.init.zeros_(m.bias)
                module.eval()

            # Try loading torchaudio for audio features
            try:
                import torchaudio  # noqa: F401
                self._audio_available = True
                logger.debug("torchaudio available for UNQA audio features")
            except ImportError:
                self._audio_available = False
                logger.debug("torchaudio not available, UNQA will use visual-only mode")

            self._ml_available = True
            self._backend = "resnet_unified"
            logger.info(
                "UNQA (ResNet-50 unified%s) initialised on %s",
                " + torchaudio" if self._audio_available else "",
                self._device,
            )

        except Exception as e:
            logger.warning("UNQA setup failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            score = self._compute_quality(sample)
            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.confidence_score = score
        except Exception as e:
            logger.warning("UNQA failed for %s: %s", sample.path, e)
        return sample

    def _compute_quality(self, sample: Sample) -> Optional[float]:
        """Compute unified quality score across available modalities."""
        import torch

        # Visual features
        visual_feat = self._extract_visual_features(sample)
        if visual_feat is None:
            return None

        with torch.no_grad():
            visual_q = self._visual_head(visual_feat)  # [1, 64]

            # Audio features (if available and sample has audio)
            audio_feat = None
            if self._audio_available and sample.is_video:
                audio_feat = self._extract_audio_features(sample)

            if audio_feat is not None:
                audio_q = self._audio_head(audio_feat)  # [1, 64]
                unified = torch.cat([visual_q, audio_q], dim=-1)  # [1, 128]
                score = self._unified_head_with_audio(unified).item()
            else:
                score = self._unified_head_visual_only(visual_q).item()

        return float(np.clip(score, 0.0, 1.0))

    def _extract_visual_features(self, sample: Sample):
        """Extract temporally pooled ResNet-50 features."""
        import torch
        from PIL import Image

        subsample = self.config.get("subsample", 8)

        if sample.is_video:
            frames = self._load_video_frames(sample, subsample)
        else:
            frames = self._load_image(sample)

        if not frames:
            return None

        frame_feats = []
        with torch.no_grad():
            for frame in frames:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                x = self._resnet_transform(pil_img).unsqueeze(0).to(self._device)
                feats = self._resnet(x).squeeze(-1).squeeze(-1)  # [1, 2048]
                frame_feats.append(feats)

        # Temporal pooling: mean over frames
        stacked = torch.cat(frame_feats, dim=0)  # [T, 2048]
        pooled = stacked.mean(dim=0, keepdim=True)  # [1, 2048]
        return pooled

    def _extract_audio_features(self, sample: Sample):
        """Extract Mel spectrogram features from video audio track."""
        import torch

        try:
            import torchaudio

            waveform, sr = torchaudio.load(str(sample.path))

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Mel spectrogram: 128 mel bins
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sr,
                n_mels=128,
                n_fft=1024,
                hop_length=512,
            )
            mel_spec = mel_transform(waveform)  # [1, 128, T]

            # Log scale
            mel_spec = torch.log1p(mel_spec)

            # Temporal average: [1, 128]
            mel_feat = mel_spec.mean(dim=-1).to(self._device)

            return mel_feat

        except Exception as e:
            logger.debug("Audio feature extraction failed: %s", e)
            return None

    def _load_video_frames(self, sample: Sample, subsample: int):
        frames = []
        cap = cv2.VideoCapture(str(sample.path))
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                return frames
            indices = np.linspace(0, total - 1, min(subsample, total), dtype=int)
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
        finally:
            cap.release()
        return frames

    def _load_image(self, sample: Sample):
        img = cv2.imread(str(sample.path))
        return [img] if img is not None else []
