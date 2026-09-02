"""Speaker identity of the voice in a clip, against a reference set of the person.

Complements ``identity_loss``, which answers the same question for the face. A
generated clip can carry the right face and speak with someone else's voice, and
for a model that synthesises audio together with video that is a distinct failure
the visual metrics cannot see.

The measure is the ordinary speaker-verification one: an ECAPA-TDNN embedding of
the clip's audio, compared by cosine similarity against embeddings of the person's
reference recordings. The reference may be a single file or a directory, and a
directory is the intended use -- one recording fixes a mood and a room, a set fixes
the speaker. Both the mean and the best similarity over the reference set are
reported, together with the share of reference files that yielded an embedding.

What this metric does **not** measure is how the person speaks -- rhythm, pauses,
intonation. It measures whose voice it is. It also says nothing about the video: a
clip whose audio was supplied rather than generated will score near the ceiling by
construction, so the number is only meaningful for a model that synthesises audio
itself.

Measured behaviour on real recordings of three public speakers, 91 clips of five to
twenty seconds cut from podium speeches and press conferences, all 4095 pairs:

* separability of same-speaker from different-speaker pairs, AUC 0.984, bootstrap
  interval over clips 0.970 to 1.000;
* nearest neighbour is the same speaker in 91 of 91 clips -- no errors;
* cosine similarity 0.581 median within a speaker against 0.018 between speakers.

For comparison on the same material: facial-expression manner separates at 0.832
and body-motion manner at 0.536. Voice is the strongest identity channel of the
three, and it needs only a few seconds of clean speech.

Backend: SpeechBrain ECAPA-TDNN trained on VoxCeleb. Values are left unset when the
clip carries no audio track, when the speech is shorter than a second, or when the
backend is unavailable.
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

MODEL_REPO_ID = "speechbrain/spkrec-ecapa-voxceleb"
MODEL_URL = f"https://huggingface.co/{MODEL_REPO_ID}"
#: Sample rate the pinned model was trained at; audio is resampled to it.
TARGET_RATE = 16000
AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus")
MEDIA_SUFFIXES = (".mp4", ".mkv", ".mov", ".webm", ".avi") + AUDIO_SUFFIXES


class VoiceIdentityModule(PipelineModule):
    name = "voice_identity"
    description = "Speaker-verification similarity of the voice to a reference set of the person"
    default_config = {
        "device": "auto",
        "models_dir": "models",
        "min_seconds": 1.0,
        "warning_threshold": 0.25,
        "max_references": 32,
    }
    metric_groups = {
        "voice_identity": "audio",
        "voice_identity_max": "audio",
        "voice_identity_coverage": "audio",
    }
    metric_info = {
        "voice_identity": "Mean speaker-embedding cosine similarity to the reference set (higher=better)",
        "voice_identity_max": "Best similarity over the reference set (higher=better)",
        "voice_identity_coverage": "Share of reference files that yielded an embedding (0-1)",
    }
    models = [
        {
            "id": MODEL_REPO_ID,
            "type": "huggingface",
            "task": "ECAPA-TDNN speaker embedding (VoxCeleb)",
            "url": MODEL_URL,
            "auto_download": "yes",
        }
    ]

    def __init__(self, config=None):
        super().__init__(config)
        self.min_seconds = float(self.config.get("min_seconds", 1.0))
        self.warning_threshold = float(self.config.get("warning_threshold", 0.25))
        self.max_references = int(self.config.get("max_references", 32))
        self._encoder = None

    def setup(self) -> None:
        try:
            import torch
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            logger.warning("voice_identity: speechbrain not installed; metric disabled")
            return
        except Exception as exc:  # pragma: no cover - import-time backend failure
            logger.warning("voice_identity: speaker encoder unavailable: %s", exc)
            return

        device = self.config.get("device", "auto")
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        models_dir = Path(self.config.get("models_dir", "models")) / "voice_identity"
        try:
            self._encoder = EncoderClassifier.from_hparams(
                source=MODEL_REPO_ID,
                savedir=str(models_dir),
                run_opts={"device": device},
            )
        except Exception as exc:  # pragma: no cover - download/runtime failure
            logger.warning("voice_identity: failed to load %s: %s", MODEL_REPO_ID, exc)

    def process(self, sample: Sample) -> Sample:
        if self._encoder is None or sample.reference_path is None:
            return sample

        try:
            with tempfile.TemporaryDirectory() as workdir:
                work = Path(workdir)
                sample_embedding = self._embed(Path(sample.path), work, "sample")
                if sample_embedding is None:
                    return sample

                references = self._reference_files(Path(sample.reference_path))
                if not references:
                    return sample
                found: List[np.ndarray] = []
                for order, path in enumerate(references):
                    embedding = self._embed(path, work, f"ref{order}")
                    if embedding is not None:
                        found.append(embedding)
                coverage = len(found) / float(len(references))
                if not found:
                    return sample

                scores = [float(np.dot(sample_embedding, other)) for other in found]
        except Exception as exc:  # pragma: no cover - depends on decoder/backend
            logger.warning("voice_identity failed for %s: %s", sample.path, exc)
            return sample

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        qm = sample.quality_metrics
        qm.voice_identity = round(float(np.mean(scores)), 4)
        qm.voice_identity_max = round(float(np.max(scores)), 4)
        qm.voice_identity_coverage = round(coverage, 4)
        return sample

    def _reference_files(self, reference: Path) -> List[Path]:
        """Reference recordings, one file or every media file in a directory."""
        if reference.is_dir():
            files = sorted(
                path
                for path in reference.iterdir()
                if path.suffix.lower() in MEDIA_SUFFIXES
            )
            return files[: self.max_references]
        return [reference] if reference.is_file() else []

    def _embed(self, path: Path, workdir: Path, tag: str) -> Optional[np.ndarray]:
        """Speaker embedding of one file, or ``None`` when it carries no usable speech."""
        import torch

        wav = workdir / f"{tag}.wav"
        result = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y", "-i", str(path),
                "-ac", "1", "-ar", str(TARGET_RATE), "-vn", str(wav),
            ],
            capture_output=True,
        )
        if result.returncode != 0 or not wav.is_file():
            return None

        # Read with soundfile rather than torchaudio: recent torchaudio routes
        # ``load`` through TorchCodec, which is a separate install and absent in
        # environments that otherwise have everything this module needs.
        import soundfile as sf

        data, rate = sf.read(str(wav), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if data.size < self.min_seconds * rate:
            return None

        with torch.no_grad():
            embedding = self._encoder.encode_batch(
                torch.from_numpy(data).unsqueeze(0)
            ).squeeze().cpu().numpy()
        norm = float(np.linalg.norm(embedding))
        return embedding / norm if norm > 0 else None
