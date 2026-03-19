"""UTMOS (Universal Text-to-Speech MOS) module.

No-reference MOS prediction for speech quality using a neural model.
Predicts the Mean Opinion Score that human listeners would assign,
without needing a reference audio.

Score range: 1.0 to 5.0 (higher = better, MOS scale).
  4.0+  excellent (natural speech)
  3.5+  good
  3.0+  fair
  <3.0  poor

Uses the SpeechMOS model (sarulab-speech/speechmos) from Hugging Face.

References:
    - Saeki et al. (2022), "UTMOS: UTokyo-SaruLab System for
      VoiceMOS Challenge 2022"
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class AudioUTMOSModule(PipelineModule):
    name = "audio_utmos"
    description = "UTMOS no-reference MOS prediction for speech quality"
    default_config = {
        "target_sr": 16000,
        "warning_threshold": 3.0,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.target_sr = self.config.get("target_sr", 16000)
        self.warning_threshold = self.config.get("warning_threshold", 3.0)
        self._model = None
        self._ml_available = False

    def setup(self) -> None:
        try:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading UTMOS model on {self._device}...")
            self._model = torch.hub.load(
                "tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
                trust_repo=True,
            )
            self._model = self._model.to(self._device)
            self._model.eval()
            self._ml_available = True
            logger.info("UTMOS module initialised")

        except ImportError:
            logger.warning("torch not installed. UTMOS requires PyTorch.")
        except Exception as e:
            logger.warning(f"Failed to setup UTMOS: {e}")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        # UTMOS works on audio — skip pure video without audio
        if sample.is_video:
            return sample

        try:
            audio = self._load_audio(sample.path)
            if audio is None:
                return sample

            import torch

            waveform = torch.from_numpy(audio).unsqueeze(0).to(self._device)

            with torch.no_grad():
                score = self._model(waveform, self.target_sr).item()

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.utmos_score = round(float(score), 3)

            if score < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low predicted MOS (UTMOS): {score:.2f}",
                        details={"utmos": float(score)},
                    )
                )

        except Exception as e:
            logger.warning(f"UTMOS failed for {sample.path}: {e}")

        return sample

    def _load_audio(self, path: Path) -> Optional[np.ndarray]:
        try:
            import soundfile as sf

            data, sr = sf.read(str(path), dtype="float32")
            if sr != self.target_sr:
                import librosa

                data = librosa.resample(data, orig_sr=sr, target_sr=self.target_sr)
            if data.ndim > 1:
                data = data.mean(axis=1)
            return data.astype(np.float32)
        except ImportError:
            logger.warning("soundfile/librosa not installed")
            return None
        except Exception as e:
            logger.debug(f"Audio load error: {e}")
            return None
