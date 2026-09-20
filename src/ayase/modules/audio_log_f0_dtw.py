"""DTW-aligned log-F0 error for paired monophonic speech or singing.

The module compares the fundamental-frequency contours of ``sample.path`` and
``sample.reference_path`` after content-based temporal alignment.  Both inputs
are decoded as mono 16 kHz audio.  F0 and voicing are estimated with pYIN on a
10 ms grid, while the DTW path is computed independently from cepstral-mean-
normalised MFCC coefficients (c1-c12).  The path is endpoint-constrained and
limited by a 10% Sakoe-Chiba band so that pitch itself cannot choose its own
best alignment and large local timing errors are not warped away freely.

The primary output is log-F0 RMSE expressed in cents (0 is identical; lower is
better).  It is emitted only when both inputs contain enough voiced evidence,
at least 20 DTW path pairs are jointly voiced, and at least half of the unique
voiced frames in each contour receive a jointly-voiced match.  Companion
outputs expose DTW-path voiced/unvoiced error and the joint coverage guard.
Silent, fully unvoiced, too-short, over-30-second, missing, or undecodable pairs
are left unset rather than assigned a fabricated score.

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
    - ESPnet TTS evaluation, which applies DTW before log-F0 RMSE,
      https://github.com/espnet/espnet/tree/master/egs2/TEMPLATE/tts1
    - librosa pYIN and exact DTW APIs,
      https://librosa.org/doc/0.10.2/generated/librosa.pyin.html
      https://librosa.org/doc/0.10.2/generated/librosa.sequence.dtw.html
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
_HOP_LENGTH = 160
_F0_FRAME_LENGTH = 2048
_MFCC_N_FFT = 512
_MFCC_WIN_LENGTH = 400
_N_MFCC = 13
_FMIN = 50.0
_FMAX = 800.0
_MIN_VOICED_FRAMES = 20
_MIN_JOINT_PAIRS = 20
_MIN_JOINT_COVERAGE = 0.50
_MAX_DURATION_SECONDS = 30.0
_DTW_BAND_RADIUS = 0.10


@dataclass(frozen=True)
class _PitchFeatures:
    """Time-aligned pitch, voicing, and MFCC features for one input."""

    f0: np.ndarray
    voiced: np.ndarray
    mfcc: np.ndarray


class AudioLogF0DTWModule(PipelineModule):
    """Measure paired pitch-contour error after MFCC-based constrained DTW."""

    name = "audio_log_f0_dtw"
    description = "MFCC-DTW-aligned log-F0 RMSE in cents for paired monophonic audio"
    default_config = {}
    metric_info = {
        "audio_log_f0_rmse_cents": (
            "MFCC-DTW-aligned log-F0 RMSE in cents (0+, lower=better)"
        ),
        "audio_f0_voicing_error": (
            "Share of the constrained DTW path with mismatched voiced/unvoiced flags "
            "(0-1, lower=better)"
        ),
        "audio_f0_joint_coverage": (
            "Lower per-input share of unique voiced frames receiving a jointly-voiced "
            "DTW match (0-1, higher=better)"
        ),
    }
    metric_groups = {
        "audio_log_f0_rmse_cents": "audio",
        "audio_f0_voicing_error": "audio",
        "audio_f0_joint_coverage": "audio",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._librosa = None
        self._backend = "unavailable"

    def setup(self) -> None:
        try:
            import librosa

            self._librosa = librosa
            self._backend = "librosa_pyin"
            logger.info("audio_log_f0_dtw initialised (librosa pYIN + exact DTW)")
        except Exception as exc:
            self._librosa = None
            self._backend = "unavailable"
            logger.warning("audio_log_f0_dtw unavailable: %s", exc)

    def process(self, sample: Sample) -> Sample:
        if self._librosa is None:
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

            rmse_cents, voicing_error, joint_coverage = result
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.audio_f0_voicing_error = round(voicing_error, 4)
            sample.quality_metrics.audio_f0_joint_coverage = round(joint_coverage, 4)
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
        librosa = self._librosa
        if librosa is None:
            return None

        f0, voiced_flag, _ = librosa.pyin(
            np.asarray(audio, dtype=np.float32),
            fmin=_FMIN,
            fmax=_FMAX,
            sr=_SAMPLE_RATE,
            frame_length=_F0_FRAME_LENGTH,
            hop_length=_HOP_LENGTH,
            center=True,
            fill_na=np.nan,
        )
        f0 = np.asarray(f0, dtype=np.float64)
        voiced = (
            np.asarray(voiced_flag, dtype=bool)
            & np.isfinite(f0)
            & (f0 > 0.0)
        )
        if int(np.count_nonzero(voiced)) < _MIN_VOICED_FRAMES:
            return None

        mfcc = librosa.feature.mfcc(
            y=np.asarray(audio, dtype=np.float32),
            sr=_SAMPLE_RATE,
            n_mfcc=_N_MFCC,
            n_fft=_MFCC_N_FFT,
            win_length=_MFCC_WIN_LENGTH,
            hop_length=_HOP_LENGTH,
            center=True,
        )
        mfcc = np.asarray(mfcc[1:_N_MFCC], dtype=np.float64)

        frame_count = min(f0.size, voiced.size, mfcc.shape[1])
        if frame_count == 0:
            return None
        f0 = f0[:frame_count]
        voiced = voiced[:frame_count]
        mfcc = mfcc[:, :frame_count]

        voiced_indices = np.flatnonzero(voiced)
        if voiced_indices.size < _MIN_VOICED_FRAMES:
            return None
        first = int(voiced_indices[0])
        last = int(voiced_indices[-1]) + 1
        f0 = f0[first:last]
        voiced = voiced[first:last]
        mfcc = mfcc[:, first:last]

        # Cepstral mean normalisation reduces stationary channel/speaker bias
        # while retaining the phonetic changes that should determine the path.
        mfcc = mfcc - np.mean(mfcc, axis=1, keepdims=True)
        if not np.isfinite(mfcc).all():
            return None

        return _PitchFeatures(f0=f0, voiced=voiced, mfcc=mfcc)

    def _align_and_score(
        self,
        reference: _PitchFeatures,
        candidate: _PitchFeatures,
    ) -> Optional[Tuple[Optional[float], float, float]]:
        librosa = self._librosa
        if librosa is None:
            return None
        if (
            int(np.count_nonzero(reference.voiced)) < _MIN_VOICED_FRAMES
            or int(np.count_nonzero(candidate.voiced)) < _MIN_VOICED_FRAMES
        ):
            return None

        _, path = librosa.sequence.dtw(
            X=reference.mfcc,
            Y=candidate.mfcc,
            metric="euclidean",
            subseq=False,
            backtrack=True,
            global_constraints=True,
            band_rad=_DTW_BAND_RADIUS,
        )
        path = np.asarray(path, dtype=np.int64)
        if path.ndim != 2 or path.shape[0] == 0 or path.shape[1] != 2:
            return None
        path = path[::-1]

        ref_indices = path[:, 0]
        cand_indices = path[:, 1]
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
            ref_f0 = reference.f0[ref_indices[joint]]
            cand_f0 = candidate.f0[cand_indices[joint]]
            cents = 1200.0 * np.log2(cand_f0 / ref_f0)
            if cents.size and np.isfinite(cents).all():
                rmse_cents = float(np.sqrt(np.mean(np.square(cents))))

        return rmse_cents, voicing_error, joint_coverage
