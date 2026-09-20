"""Temporal speaker-verification diagnostics against a reference voice set.

Use this module to inspect whether the speaker embedding of a candidate remains
similar to an enrolled reference speaker over time.  It is intended for speech
recordings with predominantly one audible speaker.  A directory of independent
reference recordings is preferred to a single recording because enrollment then
samples more than one utterance and channel condition.

Audio is decoded as mono 16 kHz.  Candidate windows start every 1.5 seconds and
contain at most 3.0 seconds; a final shorter scheduled window is retained only
when at least 1.0 second remains.  Windows whose RMS is not strictly greater than
1e-4, or for which ECAPA-TDNN cannot produce a finite embedding, are missing
observations.  Reference recordings use their complete decoded waveform, must be
at least 1.0 second long, and apply the same RMS eligibility rule.  Valid unit
reference embeddings are averaged and the mean is normalized to form one
enrollment centroid.  Each valid candidate window is compared with that centroid
by cosine similarity.

Outputs (there is no aggregate score):
    voice_identity_window_coverage
        Valid candidate-window embeddings / scheduled windows, in [0, 1].
        Higher means the temporal diagnostics are more observable, not that the
        identity match is better.
    voice_identity_reference_coverage
        Valid reference embeddings / selected reference files, in [0, 1].
    voice_identity_similarity_p05, voice_identity_similarity_min
        Fifth percentile (NumPy linear method) and minimum candidate-window
        cosine similarity to the enrollment centroid, in [-1, 1]; higher means
        greater speaker-embedding similarity.
    voice_identity_below_threshold_fraction
        Fraction of valid candidate windows below ``similarity_threshold``.
    voice_identity_longest_below_threshold_run_fraction
        Longest consecutive below-threshold run divided by all scheduled
        windows.  An ineligible or failed window breaks a run.
    voice_identity_drift_slope
        Least-squares cosine-similarity slope per full normalized scheduled
        sequence.  Negative values mean similarity declines over time.  It is
        emitted only with at least four valid windows and is not range-clipped.

``similarity_threshold`` defaults to ``None``.  Threshold-dependent fields are
emitted only for a caller-supplied operating point in [-1, 1]; no verification
threshold or identity probability is built into this module.

These are speaker-verification diagnostics, not measurements of prosody, style,
linguistic content, intelligibility, naturalness, or audio quality, and they do
not prove that two recordings contain the same person.  Results inherit the
VoxCeleb training domain and ECAPA-TDNN backend limitations.  Short speech,
silence, noise, reverberation, overlapping speakers, accents or domains unlike
the training data, and microphone/channel mismatch can change coverage and
similarity.  Interpret tail and trend values together with both coverage fields.

Backend and primary sources:
    SpeechBrain ``speechbrain/spkrec-ecapa-voxceleb`` (ECAPA-TDNN), also used by
    :mod:`ayase.modules.voice_identity`.
    Desplanques et al., "ECAPA-TDNN: Emphasized Channel Attention, Propagation
    and Aggregation in TDNN Based Speaker Verification", Interspeech 2020,
    https://arxiv.org/abs/2005.07143
    SpeechBrain reference implementation, https://github.com/speechbrain/speechbrain
"""

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ayase.audio import load_audio
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

MODEL_REPO_ID = "speechbrain/spkrec-ecapa-voxceleb"
TARGET_RATE = 16_000
AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus")
MEDIA_SUFFIXES = (".mp4", ".mkv", ".mov", ".webm", ".avi") + AUDIO_SUFFIXES


class VoiceIdentityDriftModule(PipelineModule):
    """Expose chronological ECAPA-TDNN speaker-identity diagnostics."""

    name = "voice_identity_drift"
    description = "Temporal ECAPA-TDNN speaker-identity tail, coverage, run, and drift diagnostics"
    default_config = {
        "device": "auto",
        "models_dir": "models",
        "window_seconds": 3.0,
        "hop_seconds": 1.5,
        "min_window_seconds": 1.0,
        "silence_rms_threshold": 1e-4,
        "similarity_threshold": None,
        "max_references": 32,
    }
    models = [
        {
            "id": MODEL_REPO_ID,
            "type": "huggingface",
            "task": "ECAPA-TDNN speaker embeddings trained on VoxCeleb",
            "url": f"https://huggingface.co/{MODEL_REPO_ID}",
            "auto_download": True,
            "notes": "Shared with the voice_identity module; weights are downloaded at runtime.",
        }
    ]
    metric_info = {
        "voice_identity_window_coverage": (
            "Valid ECAPA-TDNN candidate windows / scheduled windows (0-1, "
            "higher=more observable)"
        ),
        "voice_identity_reference_coverage": (
            "Valid ECAPA-TDNN reference embeddings / selected references (0-1)"
        ),
        "voice_identity_similarity_p05": (
            "Fifth percentile window cosine similarity to the reference centroid "
            "([-1, 1], higher=more similar)"
        ),
        "voice_identity_similarity_min": (
            "Minimum window cosine similarity to the reference centroid "
            "([-1, 1], higher=more similar)"
        ),
        "voice_identity_below_threshold_fraction": (
            "Share of valid windows below the caller-supplied operating point (0-1)"
        ),
        "voice_identity_longest_below_threshold_run_fraction": (
            "Longest consecutive below-threshold run / all scheduled windows (0-1)"
        ),
        "voice_identity_drift_slope": (
            "Linear speaker-similarity trend per normalized scheduled sequence "
            "(negative=decline)"
        ),
    }
    metric_groups = {field: "audio" for field in metric_info}

    def __init__(self, config=None):
        super().__init__(config)
        self.device = str(self.config.get("device", "auto"))
        self.models_dir = str(self.config.get("models_dir", "models"))
        self.window_seconds = self._positive_config("window_seconds", 3.0)
        self.hop_seconds = self._positive_config("hop_seconds", 1.5)
        self.min_window_seconds = self._positive_config("min_window_seconds", 1.0)
        if self.min_window_seconds > self.window_seconds:
            logger.warning(
                "VoiceIdentityDrift: min_window_seconds exceeds window_seconds; "
                "using window_seconds"
            )
            self.min_window_seconds = self.window_seconds
        self.silence_rms_threshold = max(
            0.0, float(self.config.get("silence_rms_threshold", 1e-4))
        )
        self.max_references = max(1, int(self.config.get("max_references", 32)))
        self.similarity_threshold = self._parse_threshold(
            self.config.get("similarity_threshold")
        )
        self._encoder = None
        self._backend = "unavailable"

    def _positive_config(self, key: str, default: float) -> float:
        try:
            value = float(self.config.get(key, default))
        except (TypeError, ValueError):
            value = default
        if not math.isfinite(value) or value <= 0.0:
            logger.warning("VoiceIdentityDrift: %s must be positive; using %s", key, default)
            return default
        return value

    @staticmethod
    def _parse_threshold(value) -> Optional[float]:
        if value is None:
            return None
        try:
            threshold = float(value)
        except (TypeError, ValueError):
            logger.warning(
                "VoiceIdentityDrift: similarity_threshold must be in [-1, 1]; "
                "threshold diagnostics disabled"
            )
            return None
        if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
            logger.warning(
                "VoiceIdentityDrift: similarity_threshold=%r is outside [-1, 1]; "
                "threshold diagnostics disabled",
                value,
            )
            return None
        return threshold

    def setup(self) -> None:
        if self.test_mode:
            logger.debug("VoiceIdentityDrift: test mode, skipping SpeechBrain setup")
            return
        try:
            import torch
            from speechbrain.inference.speaker import EncoderClassifier

            device = self.device
            if device in {"", "auto"}:
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            elif device.startswith("cuda") and not torch.cuda.is_available():
                logger.warning(
                    "VoiceIdentityDrift: CUDA requested but unavailable; falling back to CPU"
                )
                device = "cpu"

            savedir = Path(self.models_dir) / "voice_identity"
            self._encoder = EncoderClassifier.from_hparams(
                source=MODEL_REPO_ID,
                savedir=str(savedir),
                run_opts={"device": device},
            )
            self._backend = f"speechbrain:{MODEL_REPO_ID}:cosine-centroid"
            logger.info("VoiceIdentityDrift initialised on %s", device)
        except ImportError as exc:
            logger.warning("VoiceIdentityDrift dependencies are unavailable: %s", exc)
        except Exception as exc:
            self._encoder = None
            self._backend = "unavailable"
            logger.warning("VoiceIdentityDrift setup failed: %s", exc)

    def process(self, sample: Sample) -> Sample:
        if self._encoder is None or sample.reference_path is None:
            return sample

        try:
            reference_files = self._reference_files(Path(sample.reference_path))
            if not reference_files:
                return sample
            reference_target, reference_coverage = self._reference_target(reference_files)
            if reference_target is None:
                self._write_metrics(
                    sample,
                    {"voice_identity_reference_coverage": reference_coverage},
                )
                return sample

            candidate = self._load_waveform(Path(sample.path))
            if candidate is None:
                self._write_metrics(
                    sample,
                    {"voice_identity_reference_coverage": reference_coverage},
                )
                return sample

            bounds = self._window_bounds(candidate.size)
            series: List[Tuple[int, float]] = []
            for index, (start, end) in enumerate(bounds):
                window = candidate[start:end]
                if not self._eligible(window, self.min_window_seconds):
                    continue
                try:
                    embedding = self._encode_waveform(window)
                except Exception as exc:
                    logger.debug(
                        "VoiceIdentityDrift candidate window %d failed: %s", index, exc
                    )
                    continue
                if embedding is not None:
                    series.append((index, float(np.dot(embedding, reference_target))))

            values = self._summarize_series(series, len(bounds))
            values["voice_identity_reference_coverage"] = reference_coverage
            self._write_metrics(sample, values)
        except Exception as exc:
            logger.warning("VoiceIdentityDrift failed for %s: %s", sample.path, exc)

        return sample

    def _write_metrics(
        self, sample: Sample, values: Dict[str, Optional[float]]
    ) -> None:
        emitted = {field: value for field, value in values.items() if value is not None}
        if not emitted:
            return
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        for field, value in emitted.items():
            setattr(sample.quality_metrics, field, float(value))
            sample.quality_metrics.metric_backends[field] = self._backend

    def _reference_files(self, reference: Path) -> List[Path]:
        """Return one reference file or sorted supported files from a directory."""
        if reference.is_dir():
            files = sorted(
                path for path in reference.iterdir() if path.suffix.lower() in MEDIA_SUFFIXES
            )
            return files[: self.max_references]
        return [reference] if reference.is_file() else []

    def _reference_target(
        self, references: Sequence[Path]
    ) -> Tuple[Optional[np.ndarray], float]:
        """Return the normalized enrollment centroid and reference coverage."""
        embeddings: List[np.ndarray] = []
        for path in references:
            waveform = self._load_waveform(path)
            if waveform is None or not self._eligible(waveform, self.min_window_seconds):
                continue
            try:
                embedding = self._encode_waveform(waveform)
            except Exception as exc:
                logger.debug("VoiceIdentityDrift reference failed for %s: %s", path, exc)
                continue
            if embedding is not None:
                embeddings.append(embedding)

        coverage = len(embeddings) / float(len(references)) if references else 0.0
        if not embeddings:
            return None, coverage
        centroid = np.mean(np.stack(embeddings, axis=0), axis=0)
        norm = float(np.linalg.norm(centroid))
        if not np.all(np.isfinite(centroid)) or not math.isfinite(norm) or norm <= 0.0:
            return None, coverage
        return centroid / norm, coverage

    def _load_waveform(self, path: Path) -> Optional[np.ndarray]:
        audio = load_audio(path, target_sr=TARGET_RATE, mono=True)
        if audio is None:
            return None
        waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
        if waveform.size == 0 or not np.all(np.isfinite(waveform)):
            return None
        return waveform

    def _eligible(self, waveform: np.ndarray, min_seconds: float) -> bool:
        minimum = int(math.ceil(min_seconds * TARGET_RATE))
        if waveform.size < minimum:
            return False
        rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
        return math.isfinite(rms) and rms > self.silence_rms_threshold

    def _window_bounds(self, sample_count: int) -> List[Tuple[int, int]]:
        """Return deterministic chronological half-open candidate windows."""
        count = max(0, int(sample_count))
        window = max(1, int(round(self.window_seconds * TARGET_RATE)))
        hop = max(1, int(round(self.hop_seconds * TARGET_RATE)))
        minimum = max(1, int(math.ceil(self.min_window_seconds * TARGET_RATE)))
        if count < minimum:
            return []
        return [
            (start, min(count, start + window))
            for start in range(0, count - minimum + 1, hop)
        ]

    def _encode_waveform(self, waveform: np.ndarray) -> Optional[np.ndarray]:
        """Return one finite unit ECAPA-TDNN embedding."""
        import torch

        with torch.inference_mode():
            embedding = (
                self._encoder.encode_batch(torch.from_numpy(waveform).unsqueeze(0))
                .squeeze()
                .detach()
                .cpu()
                .numpy()
            )
        array = np.asarray(embedding, dtype=np.float64).reshape(-1)
        norm = float(np.linalg.norm(array))
        if not np.all(np.isfinite(array)) or not math.isfinite(norm) or norm <= 0.0:
            return None
        return array / norm

    def _summarize_series(
        self,
        series: Sequence[Tuple[int, float]],
        scheduled_window_count: int,
    ) -> Dict[str, Optional[float]]:
        """Summarize similarities while retaining missing-window positions."""
        count = max(0, int(scheduled_window_count))
        result: Dict[str, Optional[float]] = {
            "voice_identity_window_coverage": 0.0 if count else None,
            "voice_identity_similarity_p05": None,
            "voice_identity_similarity_min": None,
            "voice_identity_below_threshold_fraction": None,
            "voice_identity_longest_below_threshold_run_fraction": None,
            "voice_identity_drift_slope": None,
        }
        if count == 0:
            return result

        valid: List[Tuple[int, float]] = []
        seen = set()
        for index, similarity in series:
            index = int(index)
            similarity = float(similarity)
            if index in seen or not 0 <= index < count or not math.isfinite(similarity):
                continue
            seen.add(index)
            valid.append((index, float(np.clip(similarity, -1.0, 1.0))))
        valid.sort(key=lambda item: item[0])

        result["voice_identity_window_coverage"] = len(valid) / float(count)
        if not valid:
            return result

        similarities = np.asarray([value for _, value in valid], dtype=np.float64)
        result["voice_identity_similarity_p05"] = float(
            np.percentile(similarities, 5.0, method="linear")
        )
        result["voice_identity_similarity_min"] = float(np.min(similarities))

        if len(valid) >= 4:
            positions = np.asarray(
                [index / float(max(1, count - 1)) for index, _ in valid],
                dtype=np.float64,
            )
            if float(np.ptp(positions)) > 0.0:
                result["voice_identity_drift_slope"] = float(
                    np.polyfit(positions, similarities, 1)[0]
                )

        threshold = self.similarity_threshold
        if threshold is None:
            return result

        bad_by_index = {index: value < threshold for index, value in valid}
        result["voice_identity_below_threshold_fraction"] = sum(
            bad_by_index.values()
        ) / float(len(valid))

        longest = 0
        current = 0
        for index in range(count):
            if bad_by_index.get(index, False):
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        result["voice_identity_longest_below_threshold_run_fraction"] = (
            longest / float(count)
        )
        return result
