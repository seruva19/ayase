"""DTW-aligned log-F0 error for paired monophonic speech or singing.

The module compares the fundamental-frequency contours of ``sample.path`` and
``sample.reference_path`` after content-based temporal alignment, following the
ESPnet TTS ``evaluate_f0.py`` protocol.  Both inputs are decoded as mono
16 kHz audio.  F0 is extracted with WORLD Harvest (40-800 Hz, 16 ms frame
period), and the DTW path is computed with fastdtw over mel-cepstra
(cheaptrick + SPTK ``sp2mc``, 24 coefficients, alpha 0.42).  The published
quantity is the RMSE of natural-log F0 over jointly voiced path pairs; it is
reported in cents (``1200/ln2`` unit conversion) as ``audio_log_f0_rmse_cents``.

The metric is emitted only when both inputs contain enough voiced evidence and
the DTW path has jointly-voiced support.  Companion outputs expose DTW-path
voiced/unvoiced mismatch and the joint coverage guard (own additions, not part
of the ESPnet definition).  Silent, fully unvoiced, too-short, over-30-second,
missing, or undecodable pairs are left unset rather than assigned a fabricated
score.

This is a full-reference contour-fidelity metric.  It requires paired audio
with the same linguistic or musical content and a predominantly monophonic
pitched source.  Different utterances, polyphony, overlapping speakers, strong
background music, and pitch-tracker octave errors are outside its reliable
domain.  DTW deliberately relaxes local timing, and the metric does not measure
timbre, loudness, articulation, rhythm, emotion, or overall perceptual quality.

Primary sources:
    - Mauch and Dixon (2014), "pYIN: A Fundamental Frequency Estimator Using
      Probabilistic Threshold Distributions", ICASSP 2014,
      https://doi.org/10.1109/ICASSP.2014.6853678
    - Sakoe and Chiba (1978), "Dynamic Programming Algorithm Optimization for
      Spoken Word Recognition", https://doi.org/10.1109/TASSP.1978.1163055
    - ESPnet TTS evaluation ``evaluate_f0.py`` (WORLD Harvest + mel-cepstrum
      fastdtw + natural-log F0 RMSE),
      https://github.com/espnet/espnet/tree/master/egs2/TEMPLATE/asr1/pyscripts/utils/evaluate_f0.py
    - pyworld WORLD bindings and fastdtw,
      https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ayase.audio import load_audio
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000
# ESPnet evaluate_f0.py defaults at 16 kHz.
_F0_FMIN = 40.0
_F0_FMAX = 800.0
_N_SHIFT = 256           # hop in samples -> 16 ms frame period
_N_FFT = 1024
_MCEP_DIM = 23           # best mcep dim for 16 kHz (sp2mc order, 0-based)
_MCEP_ALPHA = 0.42
_MIN_VOICED_FRAMES = 20
_MIN_JOINT_PAIRS = 20
_MIN_JOINT_COVERAGE = 0.50
_MAX_DURATION_SECONDS = 30.0


@dataclass(frozen=True)
class _PitchFeatures:
    """Time-aligned F0 and mel-cepstrum features for one input."""

    f0: np.ndarray
    voiced: np.ndarray
    mcep: np.ndarray


class AudioLogF0DTWModule(PipelineModule):
    """Measure paired pitch-contour error (ESPnet evaluate_f0 protocol)."""

    name = "audio_log_f0_dtw"
    provenance = {
        "audio_f0_voiced_mismatch": "own",
        "audio_log_f0_rmse_cents": "published",
    }
    sources = {
        "audio_log_f0_rmse_cents": "ESPnet evaluate_f0.py (WORLD Harvest + mel-cepstrum fastdtw + ln-F0 RMSE) — https://github.com/espnet/espnet/tree/master/egs2/TEMPLATE/asr1/pyscripts/utils/evaluate_f0.py",
    }
    deviations = {
        "audio_log_f0_rmse_cents": "units are cents (ESPnet emits ln-RMSE; pure 1200/ln2 conversion); input resampled to 16 kHz; coverage/duration thresholds are own guards",
        "audio_f0_voiced_mismatch": "fraction of the DTW path with mismatched voiced flags — an own metric (not VDE)",
    }
    description = "WORLD/mcep-DTW-aligned log-F0 RMSE in cents (ESPnet evaluate_f0 protocol)"
    default_config = {}
    models = [
        {"id": "pyworld", "type": "pip_package", "install": "pip install pyworld", "task": "WORLD Harvest F0 + cheaptrick"},
        {"id": "pysptk", "type": "pip_package", "install": "pip install pysptk", "task": "SPTK sp2mc mel-cepstrum"},
        {"id": "fastdtw", "type": "pip_package", "install": "pip install fastdtw", "task": "mel-cepstrum DTW"},
    ]
    metric_info = {
        "audio_log_f0_rmse_cents": (
            "Mel-cepstrum-DTW-aligned log-F0 RMSE in cents (0+, lower=better)"
        ),
        "audio_f0_voiced_mismatch": (
            "Share of the constrained DTW path with mismatched voiced/unvoiced flags "
            "(0-1, lower=better)"
        ),
    }
    metric_groups = {
        "audio_log_f0_rmse_cents": "audio",
        "audio_f0_voiced_mismatch": "audio",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._dtw = None
        self._backend = "unavailable"

    def setup(self) -> None:
        try:
            import fastdtw
            import pysptk  # noqa: F401
            import pyworld  # noqa: F401
            from scipy.spatial import distance as _spdist

            self._dtw = lambda x, y: fastdtw.fastdtw(x, y, dist=_spdist.euclidean)
            self._backend = "espnet_f0"
            logger.info("audio_log_f0_dtw initialised (ESPnet evaluate_f0 protocol)")
        except Exception as exc:
            self._dtw = None
            self._backend = "unavailable"
            logger.warning(
                "audio_log_f0_dtw unavailable (%s); requires pyworld, pysptk, fastdtw",
                exc,
            )

    def process(self, sample: Sample) -> Sample:
        if self._backend != "espnet_f0":
            return sample

        reference = sample.reference_path
        if reference is None:
            return sample

        candidate_path = Path(sample.path)
        reference_path = Path(reference)
        if not candidate_path.exists() or not reference_path.exists():
            return sample

        try:
            reference_audio = load_audio(
                reference_path, target_sr=_SAMPLE_RATE, mono=True
            )
            candidate_audio = load_audio(
                candidate_path, target_sr=_SAMPLE_RATE, mono=True
            )
            if not self._valid_duration(reference_audio) or not self._valid_duration(
                candidate_audio
            ):
                return sample

            reference_features = self._extract_features(reference_audio)
            candidate_features = self._extract_features(candidate_audio)
            if reference_features is None or candidate_features is None:
                return sample

            result = self._align_and_score(reference_features, candidate_features)
            if result is None:
                return sample

            rmse_cents, voicing_error, _joint_coverage = result
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.audio_f0_voiced_mismatch = round(voicing_error, 4)
            if rmse_cents is not None:
                sample.quality_metrics.audio_log_f0_rmse_cents = round(
                    rmse_cents, 3
                )
        except Exception as exc:
            logger.warning("audio_log_f0_dtw failed for %s: %s", sample.path, exc)

        return sample

    @staticmethod
    def _valid_duration(audio: Optional[np.ndarray]) -> bool:
        if audio is None:
            return False
        values = np.asarray(audio)
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            return False
        return values.size <= int(_MAX_DURATION_SECONDS * _SAMPLE_RATE)

    def _extract_features(self, audio: np.ndarray) -> Optional[_PitchFeatures]:
        """ESPnet ``world_extract``: Harvest F0 + cheaptrick + sp2mc mel-cepstrum."""
        import pyworld as pw

        x = np.asarray(audio, dtype=np.float64)
        f0, time_axis = pw.harvest(
            x,
            _SAMPLE_RATE,
            f0_floor=_F0_FMIN,
            f0_ceil=_F0_FMAX,
            frame_period=_N_SHIFT / _SAMPLE_RATE * 1000,
        )
        f0 = np.asarray(f0, dtype=np.float64)
        voiced = np.isfinite(f0) & (f0 > 0.0)
        if int(np.count_nonzero(voiced)) < _MIN_VOICED_FRAMES:
            return None

        import pysptk

        sp = pw.cheaptrick(x, f0, time_axis, _SAMPLE_RATE, fft_size=_N_FFT)
        mcep = pysptk.sp2mc(sp, _MCEP_DIM, _MCEP_ALPHA)  # (T, mcep_dim+1)
        mcep = np.asarray(mcep, dtype=np.float64).T  # (dim, T)

        voiced_indices = np.flatnonzero(voiced)
        first = int(voiced_indices[0])
        last = int(voiced_indices[-1]) + 1
        f0 = f0[first:last]
        voiced = voiced[first:last]
        mcep = mcep[:, first:last]
        if not np.isfinite(mcep).all():
            return None

        return _PitchFeatures(f0=f0, voiced=voiced, mcep=mcep)

    def _align_and_score(
        self,
        reference: _PitchFeatures,
        candidate: _PitchFeatures,
    ) -> Optional[Tuple[Optional[float], float, float]]:
        if self._dtw is None:
            return None
        if (
            int(np.count_nonzero(reference.voiced)) < _MIN_VOICED_FRAMES
            or int(np.count_nonzero(candidate.voiced)) < _MIN_VOICED_FRAMES
        ):
            return None

        # ESPnet: fastdtw(gen_mcep, gt_mcep, dist=euclidean) — frames x dims.
        _, path = self._dtw(candidate.mcep.T, reference.mcep.T)
        path = np.asarray(path, dtype=np.int64)
        if path.ndim != 2 or path.shape[0] == 0 or path.shape[1] != 2:
            return None

        cand_indices = path[:, 0]
        ref_indices = path[:, 1]
        ref_voiced = reference.voiced[ref_indices]
        cand_voiced = candidate.voiced[cand_indices]
        joint = ref_voiced & cand_voiced

        voicing_error = float(np.mean(ref_voiced != cand_voiced))
        ref_joint_unique = np.unique(ref_indices[joint]).size
        cand_joint_unique = np.unique(cand_indices[joint]).size
        ref_coverage = ref_joint_unique / int(np.count_nonzero(reference.voiced))
        cand_coverage = cand_joint_unique / int(np.count_nonzero(candidate.voiced))
        joint_coverage = float(min(ref_coverage, cand_coverage))

        rmse_cents: Optional[float] = None
        if (
            int(np.count_nonzero(joint)) >= _MIN_JOINT_PAIRS
            and joint_coverage >= _MIN_JOINT_COVERAGE
        ):
            # ESPnet: RMSE of natural-log F0 over jointly voiced path pairs;
            # reported here in cents (pure unit conversion, 1200/ln2).
            ln_rmse = float(
                np.sqrt(
                    np.mean(
                        np.square(
                            np.log(candidate.f0[cand_indices[joint]])
                            - np.log(reference.f0[ref_indices[joint]])
                        )
                    )
                )
            )
            if np.isfinite(ln_rmse):
                rmse_cents = ln_rmse * 1200.0 / np.log(2.0)

        return rmse_cents, voicing_error, joint_coverage
