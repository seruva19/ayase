"""DNSMOS (Deep Noise Suppression MOS) module.

DNSMOS is Microsoft's non-intrusive speech quality metric that predicts
MOS scores without needing a reference signal. It outputs three scores:
  SIG  — signal quality (1-5)
  BAK  — background noise quality (1-5)
  OVRL — overall quality (1-5)

Based on ITU-T P.835 framework.

Requires ``torchmetrics[audio]`` or the Microsoft DNS-Challenge package.
"""

import logging
from pathlib import Path
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class DNSMOSModule(PipelineModule):
    name = "dnsmos"
    description = "DNSMOS non-intrusive audio quality (Microsoft, 1-5 MOS)"
    default_config = {}

    def __init__(self, config=None):
        super().__init__(config)
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        # Try torchmetrics DNSMOS
        try:
            from torchmetrics.audio import DeepNoiseSuppression
            self._metric_cls = DeepNoiseSuppression
            self._backend = "torchmetrics"
            self._ml_available = True
            logger.info("DNSMOS module initialised (torchmetrics)")
            return
        except (ImportError, Exception):
            pass

        logger.warning("DNSMOS requires torchmetrics[audio]")

    def _extract_audio(self, sample_path: Path) -> Optional[tuple]:
        """Extract audio waveform and sample rate from video/audio file."""
        try:
            import soundfile as sf
            audio, sr = sf.read(str(sample_path))
            return audio, sr
        except Exception:
            pass

        try:
            import librosa
            audio, sr = librosa.load(str(sample_path), sr=16000, mono=True)
            return audio, sr
        except Exception:
            pass

        return None

    def _compute_torchmetrics(self, audio, sr: int) -> Optional[dict]:
        try:
            import torch

            metric = self._metric_cls(fs=sr, personalized=False)
            if not isinstance(audio, torch.Tensor):
                import numpy as np
                if isinstance(audio, np.ndarray):
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)
                    audio = torch.from_numpy(audio).float().unsqueeze(0)
                else:
                    return None

            with torch.no_grad():
                result = metric(audio)

            if isinstance(result, dict):
                return {
                    "sig": float(result.get("SIG", result.get("sig", 0))),
                    "bak": float(result.get("BAK", result.get("bak", 0))),
                    "ovrl": float(result.get("OVRL", result.get("ovrl", 0))),
                }
            return None
        except Exception as e:
            logger.debug(f"DNSMOS torchmetrics failed: {e}")
            return None

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        result = self._extract_audio(sample.path)
        if result is None:
            return sample

        audio, sr = result

        try:
            if self._backend == "torchmetrics":
                scores = self._compute_torchmetrics(audio, sr)
            else:
                scores = None

            if scores is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.dnsmos_overall = scores.get("ovrl")
            sample.quality_metrics.dnsmos_sig = scores.get("sig")
            sample.quality_metrics.dnsmos_bak = scores.get("bak")
            logger.debug(
                f"DNSMOS for {sample.path.name}: "
                f"SIG={scores.get('sig'):.2f} BAK={scores.get('bak'):.2f} OVRL={scores.get('ovrl'):.2f}"
            )
        except Exception as e:
            logger.error(f"DNSMOS failed: {e}")
        return sample
