"""Compact reference-free speech MOS prediction with Microsoft Distill-MOS.

Distill-MOS predicts one overall perceived-speech-quality mean opinion score on
the 1-5 ACR scale (higher is better).  It is a 4.3M-parameter convolutional
transformer distilled from an XLS-R teacher and is complementary to DNSMOS
(signal/background diagnostics), NISQA (multidimensional diagnostics), SQUIM
(estimated intrusive metrics), and Audiobox Aesthetics (general-audio axes).

The released v7 model is intended for short, primarily English speech affected
by VoIP, coding, packet loss, noise, or speech enhancement.  Music, ambience,
sound effects, spatial audio, unsupported languages, and unseen degradations are
outside its supported domain.  The paper reports particularly weak correlation
on out-of-domain Blizzard 2023 French TTS and speaker-adaptation data, so this
score must not be treated as a general TTS-naturalness or generic-audio metric.

Ayase follows the official preprocessing: select the first channel, resample to
16 kHz, right-pad short inputs to 7.68 seconds, and average predictions from
overlapping 7.68-second windows.  Official inference uses at most a one-second
gap between window starts and materializes every window.  To keep long media
bounded, Ayase deterministically samples at most ``max_windows`` uniformly
spaced windows while always including the tail (and the beginning when the cap
permits multiple windows).  This is exactly equivalent to upstream inference
while the official window count is within the cap, and a documented
approximation for longer recordings.  Peak normalization is performed inside
the official model independently for each window.

Primary sources:
    - Stahl and Gamper, "Distillation and Pruning for Scalable Self-Supervised
      Representation-Based Speech Quality Assessment," ICASSP 2025,
      https://arxiv.org/abs/2502.05356
    - https://github.com/microsoft/Distill-MOS

Official release artifact:
    - distillmos 0.9.1, tag commit
      b8d46ee2748176155619cda5315ab4d5ef6af28d
    - https://raw.githubusercontent.com/microsoft/Distill-MOS/
      b8d46ee2748176155619cda5315ab4d5ef6af28d/distillmos/weights/
      distill_mos_v7.pt
    - 16,907,522 bytes; SHA-256
      b18b3ac60227267cfb91e5d00ce22cc7b73716fd92f269484da1446e27031a40
    - MIT license for the released code and model
"""

import contextlib
import io
import logging
from typing import List, Optional

import numpy as np

from ayase.audio import load_audio
from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000
_SEGMENT_SAMPLES = 122880
_MAX_HOP_SAMPLES = 16000
_CHECKPOINT_COMMIT = "b8d46ee2748176155619cda5315ab4d5ef6af28d"
_CHECKPOINT_SHA256 = "b18b3ac60227267cfb91e5d00ce22cc7b73716fd92f269484da1446e27031a40"
_CHECKPOINT_URL = (
    "https://raw.githubusercontent.com/microsoft/Distill-MOS/"
    + _CHECKPOINT_COMMIT
    + "/distillmos/weights/distill_mos_v7.pt"
)
_IMAGE_SUFFIXES = {
    ".avif",
    ".bmp",
    ".gif",
    ".heic",
    ".heif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


class AudioDistillMOSModule(PipelineModule):
    """Predict overall speech MOS with the official Distill-MOS v7 model."""

    name = "audio_distill_mos"
    description = "Microsoft Distill-MOS compact reference-free speech quality (1-5 MOS)"
    required_packages = ["distillmos"]
    default_config = {
        "device": "auto",
        "max_windows": 12,
        "min_duration_seconds": 1.0,
        "silence_rms_threshold": 1e-5,
        "warning_threshold": None,
    }
    models = [
        {
            "id": "microsoft/Distill-MOS:distill_mos_v7.pt",
            "type": "pip_package",
            "url": _CHECKPOINT_URL,
            "install": "pip install distillmos==0.9.1",
            "task": "Reference-free overall speech quality MOS prediction",
            "size": "16,907,522 bytes (16.124269 MiB)",
            "auto_download": True,
            "license": "MIT",
            "notes": (
                "Bundled in distillmos 0.9.1; release commit "
                + _CHECKPOINT_COMMIT
                + "; SHA-256 "
                + _CHECKPOINT_SHA256
            ),
        },
    ]
    metric_info = {
        "distill_mos_score": "Distill-MOS overall speech quality MOS (1-5, higher=better)",
    }
    metric_groups = {
        "distill_mos_score": "audio",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.max_windows = max(1, int(self.config.get("max_windows", 12)))
        self.min_duration_seconds = max(
            0.0, float(self.config.get("min_duration_seconds", 1.0))
        )
        self.silence_rms_threshold = max(
            0.0, float(self.config.get("silence_rms_threshold", 1e-5))
        )
        threshold = self.config.get("warning_threshold")
        self.warning_threshold = None if threshold is None else float(threshold)
        self._model = None
        self._device = "cpu"
        self._backend = "unavailable"

    def setup(self) -> None:
        try:
            import distillmos

            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            # The upstream constructor prints its checkpoint path.  Keep Ayase
            # library output on logging while preserving the official loader.
            constructor_output = io.StringIO()
            with contextlib.redirect_stdout(constructor_output):
                model = distillmos.ConvTransformerSQAModel(
                    segmenting_in_forward=False
                )
            self._model = model.to(self._device).eval()
            self._backend = "distillmos:0.9.1/v7"
            message = constructor_output.getvalue().strip()
            if message:
                logger.debug("Distill-MOS loader: %s", message)
            logger.info("Distill-MOS initialized on %s", self._device)
        except Exception as e:
            self._model = None
            self._backend = "unavailable"
            logger.warning("Distill-MOS unavailable: %s", e)

    def process(self, sample: Sample) -> Sample:
        if self._model is None or sample.path.suffix.lower() in _IMAGE_SUFFIXES:
            return sample

        try:
            audio = load_audio(sample.path, target_sr=_SAMPLE_RATE, mono=False)
            waveform = self._first_channel(audio)
            windows = self._select_windows(waveform)
            if not windows:
                return sample

            score = self._score_windows(windows)
            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.distill_mos_score = round(score, 4)

            if self.warning_threshold is not None and score < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low predicted MOS (Distill-MOS): {score:.2f}",
                        details={"distill_mos_score": score},
                    )
                )
        except Exception as e:
            logger.warning("Distill-MOS failed for %s: %s", sample.path, e)

        return sample

    @staticmethod
    def _first_channel(audio: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Return upstream-compatible first-channel float32 audio."""
        if audio is None:
            return None
        waveform = np.asarray(audio, dtype=np.float32)
        if waveform.ndim == 0:
            return None
        if waveform.ndim > 1:
            waveform = waveform[:, 0]
        return np.ascontiguousarray(waveform.reshape(-1), dtype=np.float32)

    def _select_windows(self, audio: Optional[np.ndarray]) -> List[np.ndarray]:
        """Build official 7.68-second windows with a deterministic safety cap."""
        if audio is None:
            return []

        waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
        min_samples = int(round(self.min_duration_seconds * _SAMPLE_RATE))
        if waveform.size < max(1, min_samples):
            return []

        rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
        if not np.isfinite(rms) or rms <= self.silence_rms_threshold:
            return []

        if waveform.size < _SEGMENT_SAMPLES:
            return [
                np.pad(waveform, (0, _SEGMENT_SAMPLES - waveform.size)).astype(
                    np.float32, copy=False
                )
            ]
        if waveform.size == _SEGMENT_SAMPLES:
            return [np.ascontiguousarray(waveform)]

        overlength = waveform.size - _SEGMENT_SAMPLES
        official_count = int(np.ceil(overlength / _MAX_HOP_SAMPLES)) + 1
        count = min(self.max_windows, official_count)
        if count == 1:
            starts = np.asarray([overlength], dtype=np.int64)
        else:
            starts = np.linspace(0, overlength, num=count, dtype=np.int64)
        return [
            np.ascontiguousarray(
                waveform[int(start) : int(start) + _SEGMENT_SAMPLES]
            )
            for start in starts
        ]

    def _score_windows(self, windows: List[np.ndarray]) -> Optional[float]:
        """Return the official arithmetic mean MOS for prepared windows."""
        import torch

        batch = np.stack(windows, axis=0).astype(np.float32, copy=False)
        tensor = torch.from_numpy(batch).to(self._device)
        with torch.inference_mode():
            output = self._model(tensor)
        values = output.detach().cpu().reshape(-1).numpy().astype(np.float64)

        if values.size != len(windows) or not np.all(np.isfinite(values)):
            logger.warning("Distill-MOS returned invalid output shape or non-finite values")
            return None
        if np.any(values < 1.0) or np.any(values > 5.0):
            logger.warning("Distill-MOS returned scores outside the nominal 1-5 range")
            return None

        score = float(np.mean(values, dtype=np.float64))
        return score if np.isfinite(score) and 1.0 <= score <= 5.0 else None
