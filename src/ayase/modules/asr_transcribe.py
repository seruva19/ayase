"""Shared ASR transcription cache for speech metrics.

Provides a common faster-whisper / Whisper transcription step so CER and WER
modules do not run ASR twice for the same file.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

from ayase.models import Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

_TRANSCRIPT_CACHE: Dict[Tuple[str, int, int, str], str] = {}


class ASRTranscribeModule(PipelineModule):
    name = "asr_transcribe"
    description = "Shared Whisper ASR transcription cache"
    default_config = {
        "model_name": "large-v3",
        "device": "auto",
        "language": None,
        "duration": None,
    }
    models = [
        {
            "id": "faster-whisper",
            "type": "pip_package",
            "install": "pip install faster-whisper",
            "task": "Preferred ASR backend",
        },
        {
            "id": "openai-whisper",
            "type": "pip_package",
            "install": "pip install openai-whisper",
            "task": "Fallback ASR backend",
        },
    ]

    def __init__(self, config=None):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "large-v3")
        self.device = self.config.get("device", "auto")
        self.language = self.config.get("language", None)
        self.duration = self.config.get("duration", None)
        self._backend = None
        self._model = None

    def setup(self) -> None:
        self._setup_backend()

    def process(self, sample: Sample) -> Sample:
        try:
            transcribe_sample(sample.path, self.config, model_holder=self)
        except Exception as e:
            logger.debug("ASR transcription skipped for %s: %s", sample.path, e)
        return sample

    def _setup_backend(self) -> None:
        if self._backend is not None:
            return
        try:
            from faster_whisper import WhisperModel

            device = self._resolve_device()
            compute_type = "float16" if device == "cuda" else "int8"
            self._model = WhisperModel(self.model_name, device=device, compute_type=compute_type)
            self._backend = "faster_whisper"
            logger.info("ASR transcription initialised with faster-whisper %s", self.model_name)
            return
        except ImportError:
            pass
        except Exception as e:
            logger.debug("faster-whisper setup failed: %s", e)

        try:
            import whisper

            self._model = whisper.load_model(self.model_name, device=self._resolve_device())
            self._backend = "whisper"
            logger.info("ASR transcription initialised with whisper %s", self.model_name)
        except ImportError:
            logger.warning("ASR transcription requires faster-whisper or openai-whisper")
        except Exception as e:
            logger.warning("ASR transcription setup failed: %s", e)

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


def transcribe_sample(
    path: Path,
    config: Optional[dict] = None,
    model_holder: Optional[ASRTranscribeModule] = None,
) -> Optional[str]:
    """Return cached transcript text for *path* or run an available ASR backend."""
    config = config or {}
    explicit = config.get("transcript")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    sidecar = Path(config.get("transcript_path", "")) if config.get("transcript_path") else None
    if sidecar is None:
        sidecar = Path(path).with_suffix(".asr.txt")
    try:
        if sidecar.exists():
            return sidecar.read_text(encoding="utf-8").strip()
    except Exception:
        logger.debug("Failed to read ASR sidecar %s", sidecar)

    try:
        stat = Path(path).stat()
        key = (str(Path(path).resolve()), int(stat.st_mtime_ns), int(stat.st_size), str(config.get("model_name", "large-v3")))
    except OSError:
        return None
    if key in _TRANSCRIPT_CACHE:
        return _TRANSCRIPT_CACHE[key]

    holder = model_holder or ASRTranscribeModule(config)
    if holder._backend is None:
        holder._setup_backend()
    if holder._backend is None or holder._model is None:
        return None

    transcript = _run_backend(holder, path)
    if transcript:
        _TRANSCRIPT_CACHE[key] = transcript
    return transcript


def _run_backend(holder: ASRTranscribeModule, path: Path) -> Optional[str]:
    try:
        if holder._backend == "faster_whisper":
            segments, _ = holder._model.transcribe(
                str(path),
                language=holder.language,
                vad_filter=True,
            )
            return " ".join(seg.text.strip() for seg in segments if getattr(seg, "text", "").strip()).strip()
        if holder._backend == "whisper":
            result = holder._model.transcribe(str(path), language=holder.language)
            text = result.get("text", "") if isinstance(result, dict) else ""
            return text.strip()
    except Exception as e:
        logger.warning("ASR transcription failed for %s: %s", path, e)
    return None


def normalized_tokens(text: str) -> list[str]:
    import re

    return re.findall(r"[\w']+", text.lower())


def char_error_rate(reference: str, hypothesis: str) -> float:
    ref = "".join(normalized_tokens(reference))
    hyp = "".join(normalized_tokens(hypothesis))
    if not ref:
        return 0.0 if not hyp else 1.0
    return min(_edit_distance(ref, hyp) / len(ref), 1.0)


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = normalized_tokens(reference)
    hyp = normalized_tokens(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return min(_edit_distance(ref, hyp) / len(ref), 1.0)


def _edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            ))
        prev = cur
    return prev[-1]
