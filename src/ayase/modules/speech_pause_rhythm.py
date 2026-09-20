"""Paired-speech pause and activity diagnostics from Silero VAD intervals.

This module is intended for matching-content pairs of predominantly
single-speaker speech.  It decodes each input as mono 16 kHz audio and rejects
inputs longer than 30 seconds.  The official ``silero-vad`` package supplies
half-open speech intervals in sample indices with threshold 0.5, minimum speech
250 ms, minimum silence 200 ms, and zero speech padding.
Ayase's audio loader selects an available decoder (soundfile, librosa.load, or
ffmpeg); decoding does not change the fixed analysis sample rate or channel
layout.

The analyzed span runs from the first speech start through the last speech end,
so leading and trailing silence are excluded independently for each input.
``speech_span_duration_ratio`` is candidate/reference span duration (0+).
``speech_activity_fraction_difference`` is the absolute difference between
speech fractions over those spans (0..1).  ``speech_pause_count_difference`` is
the absolute difference in internal-pause counts (0+).
``speech_pause_duration_wasserstein_ms`` is the empirical 1-Wasserstein
distance between internal-pause durations in milliseconds (0+, lower is more
similar) and is unset if either input has no internal pause.
``speech_activity_pattern_disagreement`` is the exact measure of the symmetric
difference between independently span-normalized speech-interval unions
(0..1); it does not rasterize the intervals.

These are timing diagnostics, not an aggregate quality, identity, speaker,
person, or perceptual-similarity judgment.  They are not meaningful for
different linguistic content, overlapping speakers, music-dominant audio, or
VAD misses, and they do not establish that the same person is speaking.
Noise, reverberation, singing, crosstalk, and domain mismatch can change Silero
VAD segmentation and therefore every reported diagnostic.

Source/API/license: Silero VAD, ``load_silero_vad`` and
``get_speech_timestamps`` from the official ``silero-vad`` package,
https://github.com/snakers4/silero-vad (MIT license).
"""

import logging
import math
from dataclasses import dataclass
from importlib.metadata import version
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from ayase.audio import load_audio
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000
_MAX_DURATION_SECONDS = 30.0
_MAX_SAMPLES = int(_SAMPLE_RATE * _MAX_DURATION_SECONDS)
_VAD_THRESHOLD = 0.5
_MIN_SPEECH_DURATION_MS = 250
_MIN_SILENCE_DURATION_MS = 200
_SPEECH_PAD_MS = 0

_Interval = Tuple[int, int]
_NormalizedInterval = Tuple[float, float]


@dataclass(frozen=True)
class _SpeechActivity:
    """Canonical speech activity within the first-to-last-speech span."""

    span_samples: int
    speech_fraction: float
    pause_durations_samples: Tuple[int, ...]
    normalized_intervals: Tuple[_NormalizedInterval, ...]


@dataclass(frozen=True)
class _RhythmComparison:
    """Paired timing diagnostics prior to assignment to ``QualityMetrics``."""

    span_duration_ratio: float
    activity_fraction_difference: float
    pause_count_difference: float
    pause_duration_wasserstein_ms: Optional[float]
    activity_pattern_disagreement: float


def _canonical_intervals(
    timestamps: Any,
    sample_count: int,
) -> Optional[Tuple[_Interval, ...]]:
    """Validate Silero sample-index timestamps and merge touching intervals.

    Intervals must be ordered, non-overlapping, finite integer sample indices
    satisfying ``0 <= start < end <= sample_count``.  Booleans and floating
    point values are rejected even when they happen to represent integers.
    """
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, Integral)
        or sample_count <= 0
    ):
        return None
    if not isinstance(timestamps, Sequence) or isinstance(
        timestamps, (str, bytes, bytearray)
    ):
        return None

    intervals = []
    previous_end: Optional[int] = None
    for timestamp in timestamps:
        if not isinstance(timestamp, Mapping):
            return None
        start = timestamp.get("start")
        end = timestamp.get("end")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, Integral)
            or not isinstance(end, Integral)
        ):
            return None
        start = int(start)
        end = int(end)
        if start < 0 or start >= end or end > sample_count:
            return None
        if previous_end is not None and start < previous_end:
            return None
        if intervals and start == intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], end)
        else:
            intervals.append((start, end))
        previous_end = end

    return tuple(intervals) if intervals else None


def _speech_activity(
    timestamps: Any,
    sample_count: int,
) -> Optional[_SpeechActivity]:
    """Create exact span-relative activity features from VAD timestamps."""
    intervals = _canonical_intervals(timestamps, sample_count)
    if intervals is None:
        return None

    span_start = intervals[0][0]
    span_end = intervals[-1][1]
    span_samples = span_end - span_start
    if span_samples <= 0:
        return None

    speech_samples = sum(end - start for start, end in intervals)
    pauses = tuple(
        next_start - end
        for (_, end), (next_start, _) in zip(intervals, intervals[1:])
    )
    if any(pause <= 0 for pause in pauses):
        return None

    normalized = tuple(
        (
            (start - span_start) / span_samples,
            (end - span_start) / span_samples,
        )
        for start, end in intervals
    )
    return _SpeechActivity(
        span_samples=span_samples,
        speech_fraction=speech_samples / span_samples,
        pause_durations_samples=pauses,
        normalized_intervals=normalized,
    )


def _empirical_wasserstein(left: Sequence[int], right: Sequence[int]) -> Optional[float]:
    """Return exact one-dimensional empirical Wasserstein distance."""
    if not left or not right:
        return None
    left_values = sorted(float(value) for value in left)
    right_values = sorted(float(value) for value in right)
    if (
        any(not math.isfinite(value) or value < 0.0 for value in left_values)
        or any(not math.isfinite(value) or value < 0.0 for value in right_values)
    ):
        return None

    points = sorted(set(left_values + right_values))
    left_index = 0
    right_index = 0
    distance = 0.0
    for point, next_point in zip(points, points[1:]):
        while left_index < len(left_values) and left_values[left_index] <= point:
            left_index += 1
        while right_index < len(right_values) and right_values[right_index] <= point:
            right_index += 1
        left_cdf = left_index / len(left_values)
        right_cdf = right_index / len(right_values)
        distance += abs(left_cdf - right_cdf) * (next_point - point)
    return distance


def _symmetric_difference_measure(
    left: Sequence[_NormalizedInterval],
    right: Sequence[_NormalizedInterval],
) -> Optional[float]:
    """Measure the symmetric difference of two ordered interval unions."""
    if not left or not right:
        return None

    def valid(intervals: Sequence[_NormalizedInterval]) -> bool:
        previous_end = -math.inf
        for start, end in intervals:
            if (
                not math.isfinite(start)
                or not math.isfinite(end)
                or start < 0.0
                or start >= end
                or end > 1.0
                or start < previous_end
            ):
                return False
            previous_end = end
        return True

    if not valid(left) or not valid(right):
        return None

    left_measure = sum(end - start for start, end in left)
    right_measure = sum(end - start for start, end in right)
    intersection = 0.0
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        left_start, left_end = left[left_index]
        right_start, right_end = right[right_index]
        intersection += max(0.0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1

    disagreement = left_measure + right_measure - 2.0 * intersection
    if not math.isfinite(disagreement):
        return None
    return float(min(1.0, max(0.0, disagreement)))


def _compare_activity(
    reference: _SpeechActivity,
    candidate: _SpeechActivity,
) -> Optional[_RhythmComparison]:
    """Compare two independently trimmed speech-activity interval unions."""
    if reference.span_samples <= 0 or candidate.span_samples <= 0:
        return None
    pattern_disagreement = _symmetric_difference_measure(
        reference.normalized_intervals,
        candidate.normalized_intervals,
    )
    if pattern_disagreement is None:
        return None

    pause_wasserstein_samples = _empirical_wasserstein(
        reference.pause_durations_samples,
        candidate.pause_durations_samples,
    )
    pause_wasserstein_ms = (
        pause_wasserstein_samples * 1000.0 / _SAMPLE_RATE
        if pause_wasserstein_samples is not None
        else None
    )
    return _RhythmComparison(
        span_duration_ratio=candidate.span_samples / reference.span_samples,
        activity_fraction_difference=abs(
            candidate.speech_fraction - reference.speech_fraction
        ),
        pause_count_difference=float(
            abs(
                len(candidate.pause_durations_samples)
                - len(reference.pause_durations_samples)
            )
        ),
        pause_duration_wasserstein_ms=pause_wasserstein_ms,
        activity_pattern_disagreement=pattern_disagreement,
    )


class SpeechPauseRhythmModule(PipelineModule):
    """Compare exact Silero-VAD pause timing in matching-content speech."""

    name = "speech_pause_rhythm"
    description = "Paired-speech pause and activity timing diagnostics from Silero VAD"
    default_config = {}
    models = [
        {
            "id": "silero-vad",
            "type": "pip_package",
            "install": "pip install silero-vad",
            "task": "Speech activity timestamps for paired pause-rhythm diagnostics",
            "url": "https://github.com/snakers4/silero-vad",
            "notes": "Official package; MIT license",
        }
    ]
    metric_info = {
        "speech_span_duration_ratio": (
            "Candidate/reference first-to-last-speech span duration ratio (0+)"
        ),
        "speech_activity_fraction_difference": (
            "Absolute speech-fraction difference over independently trimmed spans (0-1)"
        ),
        "speech_pause_count_difference": (
            "Absolute difference in internal-pause counts (0+)"
        ),
        "speech_pause_duration_wasserstein_ms": (
            "1-Wasserstein distance between internal-pause durations in milliseconds "
            "(0+, lower=more similar; unset if either input has no pause)"
        ),
        "speech_activity_pattern_disagreement": (
            "Exact symmetric-difference measure of independently span-normalized "
            "speech interval unions (0-1, lower=more similar)"
        ),
    }
    metric_groups = {
        "speech_span_duration_ratio": "audio",
        "speech_activity_fraction_difference": "audio",
        "speech_pause_count_difference": "audio",
        "speech_pause_duration_wasserstein_ms": "audio",
        "speech_activity_pattern_disagreement": "audio",
    }
    _metric_fields = tuple(metric_info)

    def __init__(self, config=None):
        super().__init__(config)
        self._torch: Any = None
        self._vad_model: Any = None
        self._get_speech_timestamps: Any = None
        self._backend = "unavailable"

    def setup(self) -> None:
        self._torch = None
        self._vad_model = None
        self._get_speech_timestamps = None
        self._backend = "unavailable"
        try:
            import torch
            from silero_vad import get_speech_timestamps, load_silero_vad

            self._vad_model = load_silero_vad()
            self._get_speech_timestamps = get_speech_timestamps
            self._torch = torch
            try:
                distribution_version = version("silero-vad")
            except Exception:
                distribution_version = "unknown"
            self._backend = f"silero_vad:{distribution_version}"
        except ImportError:
            logger.warning("speech_pause_rhythm requires the official silero-vad package")
        except Exception as exc:
            logger.warning("speech_pause_rhythm setup failed: %s", exc)

    def process(self, sample: Sample) -> Sample:
        if (
            not self._backend.startswith("silero_vad:")
            or self._torch is None
            or self._vad_model is None
            or self._get_speech_timestamps is None
            or sample.reference_path is None
        ):
            return sample

        candidate_path = Path(sample.path)
        reference_path = Path(sample.reference_path)
        if not candidate_path.exists() or not reference_path.exists():
            return sample

        try:
            reference_audio = load_audio(reference_path, target_sr=_SAMPLE_RATE, mono=True)
            candidate_audio = load_audio(candidate_path, target_sr=_SAMPLE_RATE, mono=True)
            if not self._valid_audio(reference_audio) or not self._valid_audio(candidate_audio):
                return sample

            reference_values = np.asarray(reference_audio, dtype=np.float32)
            candidate_values = np.asarray(candidate_audio, dtype=np.float32)
            reference = self._detect_activity(reference_values)
            candidate = self._detect_activity(candidate_values)
            if reference is None or candidate is None:
                return sample
            comparison = _compare_activity(reference, candidate)
            if comparison is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            metrics = sample.quality_metrics
            sample.quality_metrics.speech_span_duration_ratio = comparison.span_duration_ratio
            sample.quality_metrics.speech_activity_fraction_difference = (
                comparison.activity_fraction_difference
            )
            sample.quality_metrics.speech_pause_count_difference = (
                comparison.pause_count_difference
            )
            if comparison.pause_duration_wasserstein_ms is not None:
                sample.quality_metrics.speech_pause_duration_wasserstein_ms = (
                    comparison.pause_duration_wasserstein_ms
                )
            sample.quality_metrics.speech_activity_pattern_disagreement = (
                comparison.activity_pattern_disagreement
            )
            emitted_fields = [
                "speech_span_duration_ratio",
                "speech_activity_fraction_difference",
                "speech_pause_count_difference",
                "speech_activity_pattern_disagreement",
            ]
            if comparison.pause_duration_wasserstein_ms is not None:
                emitted_fields.append("speech_pause_duration_wasserstein_ms")
            for field in emitted_fields:
                metrics.metric_backends[field] = self._backend
        except Exception as exc:
            logger.warning("speech_pause_rhythm failed for %s: %s", sample.path, exc)

        return sample

    @staticmethod
    def _valid_audio(audio: Optional[np.ndarray]) -> bool:
        if audio is None:
            return False
        values = np.asarray(audio)
        return bool(
            values.ndim == 1
            and 0 < values.size <= _MAX_SAMPLES
            and np.isfinite(values).all()
        )

    def _detect_activity(self, audio: np.ndarray) -> Optional[_SpeechActivity]:
        waveform = self._torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        timestamps = self._get_speech_timestamps(
            waveform,
            self._vad_model,
            sampling_rate=_SAMPLE_RATE,
            threshold=_VAD_THRESHOLD,
            min_speech_duration_ms=_MIN_SPEECH_DURATION_MS,
            min_silence_duration_ms=_MIN_SILENCE_DURATION_MS,
            speech_pad_ms=_SPEECH_PAD_MS,
            return_seconds=False,
        )
        return _speech_activity(timestamps, int(audio.size))
