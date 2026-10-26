"""MCD (Mel Cepstral Distortion) module.

Full-reference metric for evaluating TTS and voice conversion quality, computed
following ``pymcd==0.2.1``. Audio is resampled to 22,050 Hz before a WORLD spectral
envelope and 14 SPTK mel-cepstral coefficients (order 13, c0-c13) are extracted
at 5 ms intervals, using FFT size 512 and SPTK alpha 0.65. FastDTW aligns
c1-c13, while the final path distance includes
c0-c13 and uses the Kubichek constant 10·sqrt(2)/ln(10).

Score range: 0.0+ dB (lower = better). Values are meaningful only when the
same feature extraction, alignment, and coefficient convention is used.
The PyMCD 0.2.1 algorithm is implemented directly because its ``pysptk``
dependency has no Windows wheels. PyMCD is MIT licensed; copyright 2021
Chen Qi and contributors.

References:
    - Kubichek (1993), "Mel-Cepstral Distance Measure for Objective
      Speech Quality Assessment"
    - https://github.com/chenqi008/pymcd (implementation used here)
"""

import logging
import math
from pathlib import Path

import numpy as np

from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


def _pymcd_mcep(spectrum: np.ndarray, order: int = 13, alpha: float = 0.65) -> np.ndarray:
    """Reproduce PyMCD's ``pysptk.mcep(..., maxiter=0, itype=3)`` call.

    SPTK squares its amplitude input, applies the configured epsilon, computes
    the real cepstrum, halves the DC and Nyquist terms, and frequency-warps the
    first ``fft_size / 2 + 1`` coefficients. With zero Newton iterations this
    initial value is the final result.
    """

    from ayase.utils.sptk import freqt

    amplitude = np.asarray(spectrum, dtype=np.float64)
    if amplitude.ndim != 2 or amplitude.shape[1] < 2:
        raise ValueError("WORLD spectrum must be a two-dimensional half spectrum")
    fft_size = (amplitude.shape[1] - 1) * 2
    periodogram = amplitude * amplitude + 1.0e-8
    cepstrum = np.fft.irfft(np.log(periodogram), n=fft_size, axis=1)
    cepstrum[:, 0] /= 2.0
    cepstrum[:, fft_size // 2] /= 2.0
    return freqt(cepstrum[:, : fft_size // 2 + 1], order, alpha)


class _PyMCD021:
    """Exact inference subset of MIT-licensed ``pymcd==0.2.1``."""

    sampling_rate = 22050
    frame_period = 5.0
    scale = 10.0 / math.log(10.0) * math.sqrt(2.0)

    def __init__(self, mode: str = "dtw") -> None:
        self.mode = mode

    def _features(self, path: str) -> np.ndarray:
        import librosa
        import pyworld

        waveform, _ = librosa.load(path, sr=self.sampling_rate, mono=True)
        _, spectrum, _ = pyworld.wav2world(
            waveform.astype(np.double),
            fs=self.sampling_rate,
            frame_period=self.frame_period,
            fft_size=512,
        )
        return _pymcd_mcep(spectrum)

    def calculate_mcd(self, reference_audio: str, synthesized_audio: str) -> float:
        from fastdtw import fastdtw
        from scipy.spatial.distance import euclidean

        reference = self._features(reference_audio)
        synthesized = self._features(synthesized_audio)
        if self.mode == "plain":
            count = max(len(reference), len(synthesized))
            reference = np.pad(reference, ((0, count - len(reference)), (0, 0)))
            synthesized = np.pad(synthesized, ((0, count - len(synthesized)), (0, 0)))
            path = [(index, index) for index in range(count)]
            length_factor = 1.0
        else:
            _, path = fastdtw(reference[:, 1:], synthesized[:, 1:], dist=euclidean)
            length_factor = (
                max(len(reference), len(synthesized)) / min(len(reference), len(synthesized))
                if self.mode == "dtw_sl"
                else 1.0
            )
        left = reference[[pair[0] for pair in path]]
        right = synthesized[[pair[1] for pair in path]]
        distance = np.sqrt(((left - right) ** 2).sum(axis=1)).sum()
        return float(length_factor * self.scale * distance / len(path))


class AudioMCDModule(PipelineModule):
    implementation_sources = ("../utils/sptk.py",)
    name = "audio_mcd"
    implementation_status = "implemented"
    implementation_notes = "Executable PyMCD 0.2.1 algorithm port: WORLD spectral envelope, SPTK-compatible mel cepstra, FastDTW path and Kubichek distance. Managed pyworld/fastdtw dependencies exist; a Python port does not itself make the method approximate."
    implementation_review = "Source inspection; confidence high; pretrained inference not validated."
    provenance = "adapted"
    sources = {
        "mcd_score": "MCD-DTW via pymcd 0.2.1 — https://github.com/chenqi008/pymcd",
    }
    deviations = {
        "mcd_score": "pymcd 0.2.1 uses FastDTW on SPTK mel-cepstral coefficients c1-c13 but includes c0-c13 in the final distance; results are not interchangeable with c0-excluding or exact-DTW MCD protocols",
    }
    description = "Mel Cepstral Distortion for TTS/VC quality (full-reference, pymcd)"
    default_config = {
        "warning_threshold": 8.0,  # dB
    }
    models = [
        {
            "id": "pyworld==0.3.5",
            "type": "pip_package",
            "install": "pip install ayase",
            "task": "Binary WORLD analysis for the PyMCD 0.2.1 protocol",
            "notes": "The zero-iteration SPTK transform is implemented in NumPy",
        },
        {
            "id": "fastdtw==0.3.4",
            "type": "pip_package",
            "install": "pip install ayase",
            "task": "Binary FastDTW alignment for the PyMCD 0.2.1 protocol",
        },
    ]
    metric_groups = {
        "mcd_score": "audio",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.warning_threshold = self.config.get("warning_threshold", 8.0)
        self._ml_available = False
        self._backend = "unavailable"

    def setup(self) -> None:
        try:
            import fastdtw  # noqa: F401
            import pyworld  # noqa: F401

            self._ml_available = True
            self._backend = "pymcd_0_2_1_port"
            logger.info("MCD module initialised (PyMCD 0.2.1 protocol)")
        except ImportError:
            logger.warning(
                "MCD requires the managed pyworld and fastdtw backends; score left unset."
            )

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        reference = getattr(sample, "reference_path", None)
        if reference is None:
            return sample
        reference = Path(reference) if not isinstance(reference, Path) else reference
        if not reference.exists():
            return sample

        try:
            mcd_toolbox = _PyMCD021(mode="dtw")
            mcd = mcd_toolbox.calculate_mcd(str(reference), str(sample.path))
            if mcd is None or not np.isfinite(mcd):
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.mcd_score = round(float(mcd), 3)

            if mcd > self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"High Mel Cepstral Distortion: {mcd:.2f} dB",
                        details={"mcd": float(mcd)},
                    )
                )

        except Exception as e:
            logger.warning(f"MCD failed for {sample.path}: {e}")

        return sample
