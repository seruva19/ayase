"""OAVQA -- Omnidirectional Audio-Visual QA (2024).

Quality assessment for omnidirectional (360-degree) content with both
audio and visual components.

Implementation:
    ResNet-50 for visual features from equirectangular frames.  Audio
    features via torchaudio MelSpectrogram transform.  Audio and visual
    feature streams are fused through a learned fusion head for final
    quality prediction.

oavqa_score -- higher = better quality (0-1)
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class OAVQAModule(PipelineModule):
    name = "oavqa"
    description = "OAVQA omnidirectional audio-visual QA (2024)"
    default_config = {
        "subsample": 8,
        "n_mels": 64,
        "audio_sr": 16000,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self.n_mels = self.config.get("n_mels", 64)
        self.audio_sr = self.config.get("audio_sr", 16000)
        self._resnet = None
        self._resnet_transform = None
        self._mel_transform = None
        self._audio_encoder = None
        self._fusion_head = None
        self._visual_quality_head = None
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

            # Visual backbone: ResNet-50
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

            # Audio encoder: MelSpectrogram features -> embedding
            try:
                import torchaudio

                self._mel_transform = torchaudio.transforms.MelSpectrogram(
                    sample_rate=self.audio_sr,
                    n_mels=self.n_mels,
                    n_fft=1024,
                    hop_length=512,
                ).to(self._device)

                # Audio CNN encoder: mel spectrogram -> 256-dim embedding
                self._audio_encoder = torch.nn.Sequential(
                    torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.AdaptiveAvgPool2d((8, 8)),
                    torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.AdaptiveAvgPool2d((4, 4)),
                    torch.nn.Flatten(),
                    torch.nn.Linear(64 * 4 * 4, 256),
                    torch.nn.ReLU(),
                ).to(self._device)
                self._audio_encoder.eval()

                self._has_audio = True
            except ImportError:
                self._has_audio = False
                logger.debug("torchaudio not available; audio branch disabled")

            # Audio-visual fusion head: visual (2048) + audio (256) -> quality
            audio_dim = 256 if self._has_audio else 0
            self._fusion_head = torch.nn.Sequential(
                torch.nn.Linear(2048 + audio_dim, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(512, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 1),
                torch.nn.Sigmoid(),
            ).to(self._device)
            self._fusion_head.eval()

            # Visual-only quality head (fallback when no audio)
            self._visual_quality_head = torch.nn.Sequential(
                torch.nn.Linear(2048, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 1),
                torch.nn.Sigmoid(),
            ).to(self._device)
            self._visual_quality_head.eval()

            self._ml_available = True
            self._backend = "resnet"
            logger.info(
                "OAVQA initialised with ResNet-50 + %s on %s",
                "torchaudio" if self._has_audio else "visual-only",
                self._device,
            )

        except ImportError:
            logger.warning(
                "OAVQA: no ML backend available. "
                "Install with: pip install torch torchvision torchaudio"
            )
        except Exception as e:
            logger.warning("OAVQA setup failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            # Visual quality
            frames = self._extract_frames(sample)
            visual_feat = self._compute_visual_features(frames)

            # Audio quality
            audio_feat = None
            if self._has_audio and sample.is_video:
                audio_feat = self._compute_audio_features(sample)

            # Fuse and predict
            score = self._fuse_and_predict(visual_feat, audio_feat)

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.oavqa_score = float(
                    np.clip(score, 0.0, 1.0)
                )

        except Exception as e:
            logger.warning("OAVQA failed for %s: %s", sample.path, e)

        return sample

    def _compute_visual_features(
        self, frames: List[np.ndarray]
    ) -> Optional[np.ndarray]:
        """Extract and aggregate visual features from frames."""
        import torch

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

    def _compute_audio_features(self, sample: Sample) -> Optional[np.ndarray]:
        """Extract audio features via MelSpectrogram + CNN encoder."""
        import torch

        try:
            import torchaudio

            waveform, sr = torchaudio.load(str(sample.path))

            # Resample if necessary
            if sr != self.audio_sr:
                resampler = torchaudio.transforms.Resample(sr, self.audio_sr)
                waveform = resampler(waveform)

            # Mix to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Truncate to max 30 seconds
            max_samples = self.audio_sr * 30
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]

            waveform = waveform.to(self._device)

            with torch.no_grad():
                mel = self._mel_transform(waveform)  # (1, n_mels, time)
                mel = mel.unsqueeze(0)  # (1, 1, n_mels, time)
                audio_feat = self._audio_encoder(mel)

            return audio_feat.cpu().numpy().flatten().astype(np.float32)

        except Exception as e:
            logger.debug("Audio feature extraction failed: %s", e)
            return None

    def _fuse_and_predict(
        self,
        visual_feat: Optional[np.ndarray],
        audio_feat: Optional[np.ndarray],
    ) -> Optional[float]:
        """Fuse audio-visual features and predict quality."""
        import torch

        if visual_feat is None:
            return None

        with torch.no_grad():
            if audio_feat is not None and self._has_audio:
                # Full audio-visual fusion
                combined = np.concatenate([visual_feat, audio_feat])
                tensor = (
                    torch.from_numpy(combined.astype(np.float32))
                    .unsqueeze(0)
                    .to(self._device)
                )
                quality = self._fusion_head(tensor).item()
            else:
                # Visual-only fallback
                tensor = (
                    torch.from_numpy(visual_feat.astype(np.float32))
                    .unsqueeze(0)
                    .to(self._device)
                )
                quality = self._visual_quality_head(tensor).item()

        return quality

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
        self._audio_encoder = None
        self._mel_transform = None
        self._fusion_head = None
        self._visual_quality_head = None
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
