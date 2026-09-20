"""DTW-aligned relative-energy diagnostics for paired mono speech.

This module compares the energy contours of ``sample.path`` and
``sample.reference_path`` when they contain the same linguistic content.  Both
inputs are decoded as mono 16 kHz audio.  pYIN defines each speech span from
its first voiced frame through its last voiced frame on a 10 ms grid.  An
endpoint-constrained DTW path is then computed independently of energy from
cepstral-mean-normalised MFCC coefficients c1-c12, within a 10% Sakoe-Chiba
band.  The inputs may have different durations, but each must be at most 30 s.

``audio_relative_energy_rmse_db`` is the path RMSE between mean-centred log-RMS
contours (0+ dB, lower is better), using an explicit -80 dBFS energy floor.
``audio_energy_contour_correlation`` is their aligned Pearson correlation
(-1..1, higher is better) and is left unset when either aligned contour has
zero variance.  ``audio_voiced_fraction_difference`` is the absolute pYIN
voiced-fraction difference over the trimmed speech spans (0..1, lower is
better).  ``audio_duration_ratio`` is candidate/reference original duration
(0+, 1 means equal).  ``audio_prosody_warp_ratio`` is the shorter input's frame
count divided by the DTW path length (0..1, higher means less repeated-frame
warping).  This is a path-efficiency diagnostic, not alignment coverage.
Contour outputs require at least 20 path pairs and a warp ratio of at least 0.5.

Mean-centring intentionally removes absolute level offsets: this module does
not assess absolute loudness.  Energy correlation is a transparent contour
diagnostic, not a perceptual prosody score.  The intended domain is paired,
non-overlapping mono speech with identical words and no music.  It does not
measure speaker identity, emotion, pronunciation, MOS, or similarity between
arbitrary references.  Fully unvoiced inputs and spans with fewer than 20
pYIN-voiced frames are rejected.

Primary sources and APIs:
    - Sakoe and Chiba (1978), "Dynamic Programming Algorithm Optimization for
      Spoken Word Recognition", https://doi.org/10.1109/TASSP.1978.1163055
    - librosa pYIN, RMS, MFCC, and DTW APIs:
      https://librosa.org/doc/latest/generated/librosa.pyin.html
      https://librosa.org/doc/latest/generated/librosa.feature.rms.html
      https://librosa.org/doc/latest/generated/librosa.feature.mfcc.html
      https://librosa.org/doc/latest/generated/librosa.sequence.dtw.html
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
_ENERGY_FRAME_LENGTH = 400
_MFCC_N_FFT = 512
_MFCC_WIN_LENGTH = 400
_N_MFCC = 13
_FMIN = 50.0
_FMAX = 800.0
_ENERGY_FLOOR_DB = -80.0
_ENERGY_FLOOR_AMPLITUDE = 10.0 ** (_ENERGY_FLOOR_DB / 20.0)
_MIN_VOICED_FRAMES = 20
_MIN_PATH_PAIRS = 20
_MIN_WARP_RATIO = 0.50
_MAX_DURATION_SECONDS = 30.0
_DTW_BAND_RADIUS = 0.10
_ZERO_VARIANCE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class _ProsodyFeatures:
    """Frame-synchronous features over one first-to-last-voiced speech span."""

    energy_db: np.ndarray
    voiced: np.ndarray
    mfcc: np.ndarray

    @property
    def voiced_fraction(self) -> float:
        return float(np.mean(self.voiced))


class AudioProsodyDTWModule(PipelineModule):
    """Compare paired speech energy contours after independent MFCC-based DTW."""

    name = "audio_prosody_dtw"
    description = (
        "MFCC-DTW-aligned relative-energy and voicing diagnostics for paired speech"
    )
    default_config = {}
    metric_info = {
        "audio_relative_energy_rmse_db": (
            "RMSE between aligned mean-centred log-RMS contours in dB "
            "(0+, lower=better; absolute level removed)"
        ),
        "audio_energy_contour_correlation": (
            "Pearson correlation of aligned mean-centred energy contours "
            "(-1..1, higher=better; not a perceptual prosody score)"
        ),
        "audio_voiced_fraction_difference": (
            "Absolute difference in pYIN voiced fractions over trimmed speech spans "
            "(0-1, lower=better)"
        ),
        "audio_duration_ratio": (
            "Candidate/reference original-duration ratio (0+, 1 means equal)"
        ),
        "audio_prosody_warp_ratio": (
            "Shorter input frame count divided by constrained DTW path length "
            "(0-1, higher=less repeated-frame warping; path-efficiency diagnostic)"
        ),
    }
    metric_groups = {
        "audio_relative_energy_rmse_db": "audio",
        "audio_energy_contour_correlation": "audio",
        "audio_voiced_fraction_difference": "audio",
        "audio_duration_ratio": "audio",
        "audio_prosody_warp_ratio": "audio",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._librosa = None
        self._backend = "unavailable"

    def setup(self) -> None:
        try:
            import librosa

            self._librosa = librosa
            self._backend = "librosa_pyin_dtw"
            logger.info("audio_prosody_dtw initialised (librosa pYIN + exact DTW)")
        except Exception as exc:
            self._librosa = None
            self._backend = "unavailable"
            logger.warning("audio_prosody_dtw unavailable: %s", exc)

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
            if not self._valid_audio(reference_audio) or not self._valid_audio(
                candidate_audio
            ):
                return sample

            reference_values = np.asarray(reference_audio)
            candidate_values = np.asarray(candidate_audio)
            duration_ratio = float(candidate_values.size / reference_values.size)
            self._ensure_metrics(sample)
            qm = sample.quality_metrics
            qm.audio_duration_ratio = round(duration_ratio, 6)

            reference_features = self._extract_features(reference_values)
            candidate_features = self._extract_features(candidate_values)
            if reference_features is None or candidate_features is None:
                return sample

            voiced_fraction_difference = abs(
                candidate_features.voiced_fraction
                - reference_features.voiced_fraction
            )
            qm.audio_voiced_fraction_difference = round(
                voiced_fraction_difference, 6
            )

            result = self._align_and_score(reference_features, candidate_features)
            if result is None:
                return sample

            energy_rmse_db, energy_correlation, warp_ratio = result
            qm.audio_prosody_warp_ratio = round(warp_ratio, 6)
            if energy_rmse_db is not None:
                qm.audio_relative_energy_rmse_db = round(energy_rmse_db, 6)
            if energy_correlation is not None:
                qm.audio_energy_contour_correlation = round(
                    energy_correlation, 6
                )
        except Exception as exc:
            logger.warning("audio_prosody_dtw failed for %s: %s", sample.path, exc)

        return sample

    @staticmethod
    def _ensure_metrics(sample: Sample) -> QualityMetrics:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        return sample.quality_metrics

    @staticmethod
    def _valid_audio(audio: Optional[np.ndarray]) -> bool:
        if audio is None:
            return False
        values = np.asarray(audio)
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            return False
        return values.size <= int(_MAX_DURATION_SECONDS * _SAMPLE_RATE)

    def _extract_features(self, audio: np.ndarray) -> Optional[_ProsodyFeatures]:
        librosa = self._librosa
        if librosa is None:
            return None

        values = np.asarray(audio, dtype=np.float32)
        f0, voiced_flag, _ = librosa.pyin(
            values,
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

        rms = librosa.feature.rms(
            y=values,
            frame_length=_ENERGY_FRAME_LENGTH,
            hop_length=_HOP_LENGTH,
            center=True,
        )
        rms = np.asarray(rms, dtype=np.float64).reshape(-1)
        energy_db = 20.0 * np.log10(np.maximum(rms, _ENERGY_FLOOR_AMPLITUDE))
        energy_db = np.maximum(energy_db, _ENERGY_FLOOR_DB)

        mfcc = librosa.feature.mfcc(
            y=values,
            sr=_SAMPLE_RATE,
            n_mfcc=_N_MFCC,
            n_fft=_MFCC_N_FFT,
            win_length=_MFCC_WIN_LENGTH,
            hop_length=_HOP_LENGTH,
            center=True,
        )
        mfcc = np.asarray(mfcc[1:_N_MFCC], dtype=np.float64)

        frame_count = min(f0.size, voiced.size, energy_db.size, mfcc.shape[1])
        if frame_count == 0:
            return None
        voiced = voiced[:frame_count]
        energy_db = energy_db[:frame_count]
        mfcc = mfcc[:, :frame_count]

        voiced_indices = np.flatnonzero(voiced)
        if voiced_indices.size < _MIN_VOICED_FRAMES:
            return None
        first = int(voiced_indices[0])
        last = int(voiced_indices[-1]) + 1
        voiced = voiced[first:last]
        energy_db = energy_db[first:last]
        mfcc = mfcc[:, first:last]

        # CMN reduces stationary channel/speaker bias while retaining the
        # phonetic changes used for alignment. Energy never selects the path.
        mfcc = mfcc - np.mean(mfcc, axis=1, keepdims=True)
        if not np.isfinite(energy_db).all() or not np.isfinite(mfcc).all():
            return None

        return _ProsodyFeatures(
            energy_db=energy_db,
            voiced=voiced,
            mfcc=mfcc,
        )

    def _align_and_score(
        self,
        reference: _ProsodyFeatures,
        candidate: _ProsodyFeatures,
    ) -> Optional[Tuple[Optional[float], Optional[float], float]]:
        librosa = self._librosa
        if librosa is None:
            return None

        ref_count = reference.energy_db.size
        cand_count = candidate.energy_db.size
        if ref_count == 0 or cand_count == 0:
            return None
        if (
            reference.voiced.size != ref_count
            or candidate.voiced.size != cand_count
            or reference.mfcc.ndim != 2
            or candidate.mfcc.ndim != 2
            or reference.mfcc.shape[1] != ref_count
            or candidate.mfcc.shape[1] != cand_count
        ):
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
        if (
            np.any(ref_indices < 0)
            or np.any(ref_indices >= ref_count)
            or np.any(cand_indices < 0)
            or np.any(cand_indices >= cand_count)
        ):
            return None
        if (
            not np.array_equal(path[0], np.array([0, 0]))
            or not np.array_equal(
                path[-1], np.array([ref_count - 1, cand_count - 1])
            )
        ):
            return None
        steps = np.diff(path, axis=0)
        if np.any(steps < 0) or np.any(steps > 1) or np.any(np.all(steps == 0, axis=1)):
            return None

        warp_ratio = float(min(ref_count, cand_count) / path.shape[0])

        energy_rmse_db: Optional[float] = None
        energy_correlation: Optional[float] = None
        if path.shape[0] >= _MIN_PATH_PAIRS and warp_ratio >= _MIN_WARP_RATIO:
            reference_centered = reference.energy_db - np.mean(reference.energy_db)
            candidate_centered = candidate.energy_db - np.mean(candidate.energy_db)
            ref_aligned = reference_centered[ref_indices]
            cand_aligned = candidate_centered[cand_indices]
            difference = cand_aligned - ref_aligned
            if np.isfinite(difference).all():
                energy_rmse_db = float(np.sqrt(np.mean(np.square(difference))))

            ref_std = float(np.std(ref_aligned))
            cand_std = float(np.std(cand_aligned))
            if (
                ref_std > _ZERO_VARIANCE_TOLERANCE
                and cand_std > _ZERO_VARIANCE_TOLERANCE
            ):
                correlation = float(np.corrcoef(ref_aligned, cand_aligned)[0, 1])
                if np.isfinite(correlation):
                    energy_correlation = float(np.clip(correlation, -1.0, 1.0))

        return energy_rmse_db, energy_correlation, warp_ratio
