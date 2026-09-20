"""SpeechBERTScore for matching-content reference speech.

The metric compares generated speech with a reference that contains the same
spoken linguistic content; exact time alignment and equal duration are not
required. It is not a no-reference MOS, an arbitrary-reference similarity
measure, or a metric for music and general audio.

This implementation follows the official SpeechBERTScore precision variant:
mono audio is resampled to 16 kHz without amplitude normalization or VAD,
encoded by the pinned WavLM-Large model, and represented by Hugging Face
``hidden_states[14]``. For every generated frame it takes the maximum cosine
similarity over all reference frames, then averages those maxima. The result is
asymmetric, has mathematical range [-1, 1], and is better when higher. Exact
blockwise maxima avoid materializing the full similarity matrix; the encoder is
never windowed, because doing so would change the published metric. Inputs over
30 seconds are skipped by default to bound full-context WavLM memory use.

Method source: Saeki et al., Interspeech 2024, and the MIT-licensed reference
implementation at Takaaki-Saeki/DiscreteSpeechMetrics v1.0.1. Model weights are
downloaded at runtime from ``microsoft/wavlm-large`` at a fixed revision and are
licensed CC BY-SA 3.0; the checkpoint size and SHA-256 are verified before the
pickle checkpoint is loaded.
"""

import hashlib
import logging
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ayase.audio import get_audio_metadata, load_audio
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class SpeechBERTScoreModule(PipelineModule):
    """Compute official WavLM-Large SpeechBERTScore precision."""

    name = "speech_bert_score"
    description = "SpeechBERTScore similarity for matching-content reference speech"
    default_config = {
        "device": "auto",
        "models_dir": "models",
        "min_duration_seconds": 0.1,
        "max_duration_seconds": 30.0,
        "silence_rms_threshold": 1e-5,
        "similarity_block_frames": 1024,
    }

    model_id = "microsoft/wavlm-large"
    model_revision = "c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c"
    checkpoint_filename = "pytorch_model.bin"
    checkpoint_size = 1_261_990_257
    checkpoint_sha256 = "fdee460e529396ddb2f8c8e8ce0ad74cfb747b726bc6f612e666c7c1e1963c9d"
    sample_rate = 16_000
    hidden_state_index = 14

    models = [
        {
            "id": model_id,
            "type": "huggingface",
            "task": "Matching-content reference speech similarity",
            "auto_download": True,
            "size": "1,261,990,257 bytes",
            "url": (
                "https://huggingface.co/microsoft/wavlm-large/blob/"
                f"{model_revision}/pytorch_model.bin"
            ),
            "revision": model_revision,
            "arxiv": "2110.13900",
            "notes": (
                "CC BY-SA 3.0; WavLM-Large at 16 kHz; SHA-256 "
                f"{checkpoint_sha256}; official license: "
                "https://github.com/microsoft/UniSpeech/blob/"
                "8f8cbd22d352fc59dfd5bf19de979b05bb5c7938/LICENSE; "
                "weights are downloaded at runtime and not bundled"
            ),
        }
    ]
    metric_info = {
        "speech_bert_score": (
            "WavLM-Large layer-14 SpeechBERTScore precision for matching-content "
            "speech ([-1, 1], higher=better)"
        )
    }
    metric_groups = {"speech_bert_score": "audio"}

    def __init__(self, config=None):
        super().__init__(config)
        self.device_config = str(self.config.get("device", "auto"))
        self.models_dir = str(self.config.get("models_dir", "models"))
        self.min_duration_seconds = float(
            self.config.get("min_duration_seconds", 0.1)
        )
        self.max_duration_seconds = float(
            self.config.get("max_duration_seconds", 30.0)
        )
        self.silence_rms_threshold = float(
            self.config.get("silence_rms_threshold", 1e-5)
        )
        self.similarity_block_frames = int(
            self.config.get("similarity_block_frames", 1024)
        )
        self._model = None
        self._device = None
        self._backend = "unavailable"

    def setup(self) -> None:
        """Download, verify, and load the fixed official model configuration."""
        try:
            import torch
            from huggingface_hub import snapshot_download
            from transformers import WavLMModel

            if self.similarity_block_frames < 1:
                raise ValueError("similarity_block_frames must be positive")
            if self.min_duration_seconds <= 0:
                raise ValueError("min_duration_seconds must be positive")
            if self.max_duration_seconds <= self.min_duration_seconds:
                raise ValueError(
                    "max_duration_seconds must exceed min_duration_seconds"
                )

            if self.device_config in {"", "auto"}:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                device = torch.device(self.device_config)
                if device.type == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError("CUDA was requested but is not available")

            snapshot_path = Path(
                snapshot_download(
                    repo_id=self.model_id,
                    revision=self.model_revision,
                    cache_dir=self.models_dir,
                    allow_patterns=["config.json", self.checkpoint_filename],
                )
            )
            checkpoint_path = snapshot_path / self.checkpoint_filename
            self._verify_checkpoint(checkpoint_path)

            # Loading only from the verified local snapshot prevents a second,
            # moving network resolution between verification and deserialization.
            model = WavLMModel.from_pretrained(
                str(snapshot_path),
                local_files_only=True,
                use_safetensors=False,
            )
            self._model = model.to(device).eval()
            self._device = device
            self._backend = (
                f"wavlm-large@{self.model_revision}:hidden_states[{self.hidden_state_index}]"
            )
            logger.info("SpeechBERTScore initialised on %s", device)
        except ImportError as exc:
            logger.warning("SpeechBERTScore dependencies are unavailable: %s", exc)
        except Exception as exc:
            self._model = None
            self._device = None
            self._backend = "unavailable"
            logger.warning("SpeechBERTScore setup failed: %s", exc)

    @classmethod
    def _verify_checkpoint(cls, checkpoint_path: Path) -> None:
        """Verify the pinned pickle checkpoint before model deserialization."""
        if not checkpoint_path.is_file():
            raise RuntimeError("SpeechBERTScore checkpoint is missing")
        if checkpoint_path.stat().st_size != cls.checkpoint_size:
            raise RuntimeError("SpeechBERTScore checkpoint size mismatch")

        digest = hashlib.sha256()
        with checkpoint_path.open("rb") as checkpoint_file:
            for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != cls.checkpoint_sha256:
            raise RuntimeError("SpeechBERTScore checkpoint SHA-256 mismatch")

    def process(self, sample: Sample) -> Sample:
        if self._model is None or self._device is None:
            return sample

        reference_path = sample.reference_path
        candidate_path = Path(sample.path)
        if reference_path is None:
            return sample
        reference_path = Path(reference_path)
        if not reference_path.is_file() or not candidate_path.is_file():
            return sample

        try:
            reference_audio = self._prepare_waveform(reference_path)
            candidate_audio = self._prepare_waveform(candidate_path)
            if reference_audio is None or candidate_audio is None:
                return sample

            import torch

            with torch.inference_mode():
                reference_features = self._encode(reference_audio)
                candidate_features = self._encode(candidate_audio)
                precision = self._blockwise_precision(
                    candidate_features,
                    reference_features,
                    self.similarity_block_frames,
                )

            score = float(precision.detach().cpu().item())
            if not math.isfinite(score) or score < -1.0001 or score > 1.0001:
                logger.warning(
                    "SpeechBERTScore returned an invalid score for %s: %r",
                    sample.path,
                    score,
                )
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.speech_bert_score = score
            sample.quality_metrics.metric_backends["speech_bert_score"] = self._backend
        except Exception as exc:
            logger.warning("SpeechBERTScore failed for %s: %s", sample.path, exc)

        return sample

    def _prepare_waveform(self, path: Path):
        """Decode mono audio and reproduce the official 16 kHz resampling step."""
        decoded = self._decode_native(path)
        if decoded is None:
            return None
        audio, source_rate = decoded
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0 or not np.all(np.isfinite(audio)):
            return None

        source_duration = audio.size / float(source_rate)
        if not self.min_duration_seconds <= source_duration <= self.max_duration_seconds:
            logger.debug(
                "Skipping SpeechBERTScore input outside %.3f-%.3f seconds: %s",
                self.min_duration_seconds,
                self.max_duration_seconds,
                path,
            )
            return None
        rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
        if not math.isfinite(rms) or rms <= self.silence_rms_threshold:
            return None

        import torch
        import torchaudio

        waveform = torch.from_numpy(audio).unsqueeze(0).to(self._device).float()
        if source_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=source_rate,
                new_freq=self.sample_rate,
            )
        return waveform

    @staticmethod
    def _decode_native(path: Path) -> Optional[Tuple[np.ndarray, int]]:
        """Decode at native sample rate and average channels without normalization."""
        try:
            import soundfile as sf

            audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
        except Exception:
            metadata = get_audio_metadata(path)
            if metadata is None or metadata.sample_rate <= 0:
                return None
            sample_rate = metadata.sample_rate
            audio = load_audio(path, target_sr=sample_rate, mono=False)
            if audio is None:
                return None

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1, dtype=np.float32)
        elif audio.ndim != 1:
            return None
        return audio, int(sample_rate)

    def _encode(self, waveform):
        outputs = self._model(waveform, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        if hidden_states is None or len(hidden_states) <= self.hidden_state_index:
            raise RuntimeError("WavLM did not return the requested hidden state")
        features = hidden_states[self.hidden_state_index]
        if features.ndim != 3 or features.shape[0] != 1 or features.shape[1] == 0:
            raise RuntimeError("WavLM returned invalid speech features")
        return features.squeeze(0)

    @staticmethod
    def _blockwise_precision(generated, reference, block_frames: int):
        """Compute ``mean_i max_j cosine(generated_i, reference_j)`` exactly."""
        import torch

        if block_frames < 1:
            raise ValueError("block_frames must be positive")
        if generated.ndim != 2 or reference.ndim != 2:
            raise ValueError("SpeechBERTScore features must be two-dimensional")
        if generated.shape[0] == 0 or reference.shape[0] == 0:
            raise ValueError("SpeechBERTScore feature sequences must be non-empty")
        if generated.shape[1] != reference.shape[1]:
            raise ValueError("SpeechBERTScore feature dimensions must match")

        generated_norms = torch.linalg.vector_norm(generated, dim=1)
        reference_norms = torch.linalg.vector_norm(reference, dim=1)
        if torch.any(generated_norms == 0) or torch.any(reference_norms == 0):
            raise ValueError("SpeechBERTScore encountered a zero-norm feature")

        maxima = []
        for generated_start in range(0, generated.shape[0], block_frames):
            generated_end = min(generated_start + block_frames, generated.shape[0])
            generated_block = generated[generated_start:generated_end]
            generated_block_norms = generated_norms[generated_start:generated_end]
            block_maxima = None
            for reference_start in range(0, reference.shape[0], block_frames):
                reference_end = min(reference_start + block_frames, reference.shape[0])
                reference_block = reference[reference_start:reference_end]
                denominator = generated_block_norms[:, None] * reference_norms[
                    reference_start:reference_end
                ][None, :]
                similarities = torch.matmul(generated_block, reference_block.T) / denominator
                current_maxima = torch.max(similarities, dim=1).values
                block_maxima = (
                    current_maxima
                    if block_maxima is None
                    else torch.maximum(block_maxima, current_maxima)
                )
            maxima.append(block_maxima)

        return torch.cat(maxima).mean()
