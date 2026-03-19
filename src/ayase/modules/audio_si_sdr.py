"""SI-SDR (Scale-Invariant Signal-to-Distortion Ratio) module.

Full-reference metric for audio source separation and enhancement.
Measures signal quality independent of the global scale, making it
more robust than classic SDR for comparing audio signals.

Score range: -inf to +inf dB (higher = better).
  >20 dB  excellent
  10-20    good
  0-10     fair
  <0       poor (distortion dominates)

Standard metric in source separation (SiSEC, DNSMOS benchmarks).

References:
    - Le Roux et al. (2019), "SDR — Half-Baked or Well Done?"
      ICASSP 2019
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class AudioSISDRModule(PipelineModule):
    name = "audio_si_sdr"
    description = "Scale-Invariant SDR for audio quality (full-reference)"
    default_config = {
        "target_sr": 16000,
        "warning_threshold": 0.0,  # dB
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.target_sr = self.config.get("target_sr", 16000)
        self.warning_threshold = self.config.get("warning_threshold", 0.0)

    def process(self, sample: Sample) -> Sample:
        reference = getattr(sample, "reference_path", None)
        if reference is None:
            return sample
        reference = Path(reference) if not isinstance(reference, Path) else reference
        if not reference.exists():
            return sample

        try:
            ref_audio = self._load_audio(reference)
            deg_audio = self._load_audio(sample.path)

            if ref_audio is None or deg_audio is None:
                return sample

            min_len = min(len(ref_audio), len(deg_audio))
            if min_len < self.target_sr * 0.1:
                return sample

            ref_audio = ref_audio[:min_len]
            deg_audio = deg_audio[:min_len]

            si_sdr = self._compute_si_sdr(ref_audio, deg_audio)

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.si_sdr_score = round(float(si_sdr), 3)

            if si_sdr < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low SI-SDR: {si_sdr:.2f} dB",
                        details={"si_sdr": float(si_sdr)},
                    )
                )

        except Exception as e:
            logger.warning(f"SI-SDR failed for {sample.path}: {e}")

        return sample

    @staticmethod
    def _compute_si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
        """Compute Scale-Invariant SDR.

        SI-SDR = 10 * log10(||s_target||^2 / ||e_noise||^2)
        where s_target = (<est, ref> / <ref, ref>) * ref
              e_noise  = est - s_target
        """
        ref = reference - np.mean(reference)
        est = estimate - np.mean(estimate)

        dot = np.dot(ref, est)
        s_ref = np.dot(ref, ref)

        if s_ref < 1e-8:
            return 0.0

        s_target = (dot / s_ref) * ref
        e_noise = est - s_target

        si_sdr_val = np.dot(s_target, s_target) / max(np.dot(e_noise, e_noise), 1e-8)
        return 10.0 * np.log10(max(si_sdr_val, 1e-8))

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
