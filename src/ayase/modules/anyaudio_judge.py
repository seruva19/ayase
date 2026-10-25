"""AnyAudio-Judge rubric-based audio-instruction alignment.

Runs the pinned author logits scorer. Each rubric item is scored from the maximum
eligible yes/no next-token logits and the reported score is their arithmetic mean.
Caption-only inputs use the author's Qwen decomposition and cleaning prompts.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import math
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

SOURCE_REVISION = "b3a22a78c7407b3f7eb8dd5ae18c0bacde895f52"
SOURCE_ARCHIVE = (
    "https://codeload.github.com/CuCl-2/AnyAudio-Judge/zip/" + SOURCE_REVISION
)
JUDGE_MODELS = {
    "7b": ("cucl2/AnyAudio-Judge-7B", "f939a6512a00cdd112162a045daf6eac697e338f"),
    "30b": ("cucl2/AnyAudio-Judge-30B", "7fd66d33d978dfc48105e16cab3893580b0899a6"),
}
DECOMPOSER_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DECOMPOSER_REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
_REQUIRED_SOURCE = (
    "anyaudio_judge/__init__.py",
    "anyaudio_judge/judge.py",
    "anyaudio_judge/decompose.py",
    "anyaudio_judge/prompts.py",
)
_SUBSET_ALIASES = {
    "speech": "speech", "speech_gen": "speech", "speech-generation": "speech",
    "sound": "sound", "sound_gen": "sound", "sound-generation": "sound",
    "music": "music", "music_gen": "music", "music-generation": "music",
    "mix": "mix", "mixed": "mix", "audio": "mix", "general": "mix",
}


def _verify_source(root: Path, *, managed: bool) -> None:
    if not all((root / name).is_file() for name in _REQUIRED_SOURCE):
        raise ValueError(f"Incomplete AnyAudio-Judge source tree: {root}")
    marker = root / ".ayase-source-revision"
    if managed and not marker.is_file():
        raise ValueError(f"Unverified AnyAudio-Judge source tree: {root}")
    if marker.is_file() and marker.read_text(encoding="ascii").strip() != SOURCE_REVISION:
        raise ValueError(f"AnyAudio-Judge source revision mismatch: {root}")


def _ensure_source(destination: Path) -> Path:
    """Acquire the immutable author source archive without requiring git."""
    from ayase.managed_runtime import _SetupLock

    destination.parent.mkdir(parents=True, exist_ok=True)
    with _SetupLock(destination.with_suffix(".lock")):
        if destination.is_dir():
            _verify_source(destination, managed=True)
            return destination
        with tempfile.TemporaryDirectory(prefix="anyaudio-source-", dir=destination.parent) as tmp:
            work = Path(tmp)
            archive = work / "source.zip"
            urllib.request.urlretrieve(SOURCE_ARCHIVE, archive)
            unpacked = work / "unpacked"
            unpacked.mkdir()
            with zipfile.ZipFile(archive) as bundle:
                members = bundle.infolist()
                roots = {Path(item.filename).parts[0] for item in members if Path(item.filename).parts}
                if len(roots) != 1:
                    raise ValueError("AnyAudio-Judge archive must contain one root")
                base = unpacked.resolve()
                for item in members:
                    target = (unpacked / item.filename).resolve()
                    if target != base and base not in target.parents:
                        raise ValueError(f"Unsafe source archive member: {item.filename}")
                    if item.external_attr >> 16 & 0o170000 == 0o120000:
                        raise ValueError(f"Source archive contains a symlink: {item.filename}")
                bundle.extractall(unpacked)
            extracted = unpacked / roots.pop()
            (extracted / ".ayase-source-revision").write_text(SOURCE_REVISION, encoding="ascii")
            _verify_source(extracted, managed=True)
            extracted.replace(destination)
    return destination


def _normalize_rubric(value: Any) -> Optional[List[Dict[str, str]]]:
    if not isinstance(value, list) or not value:
        return None
    normalized: List[Dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            return None
        question = item.get("question", item.get("题目"))
        if not isinstance(question, str) or not question.strip():
            return None
        dimension = item.get("dimension", item.get("维度", ""))
        basis = item.get("basis", item.get("依据", ""))
        if not isinstance(dimension, str) or not isinstance(basis, str):
            return None
        normalized.append({
            "dimension": dimension, "question": question, "basis": basis
        })
    return normalized


def _rubric_identity(rubric: Sequence[Dict[str, str]]) -> str:
    payload = json.dumps(list(rubric), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _install_generate_adapter(judge: Any) -> None:
    """Apply the Transformers full-Omni first-step compatibility shim."""
    original_generate = judge.model.generate

    def compatible_generate(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("thinker_max_new_tokens", 1)
        kwargs.setdefault("return_audio", False)
        features = kwargs.get("input_features")
        thinker = getattr(judge.model, "thinker", judge.model)
        audio_tower = getattr(thinker, "audio_tower", None)
        if audio_tower is not None and features is not None and features.is_floating_point():
            # Qwen3-Omni does not perform Qwen2.5-Omni's internal input cast.
            kwargs["input_features"] = features.to(dtype=next(audio_tower.parameters()).dtype)
        result = original_generate(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    judge.model.generate = compatible_generate


class AnyAudioJudgeModule(PipelineModule):
    name = "anyaudio_judge"
    description = "AnyAudio-Judge author-logits audio-instruction alignment"
    input_type = "audio"
    provenance = {"anyaudio_judge_score": "published"}
    sources = {
        "anyaudio_judge_score": (
            "AnyAudio-Judge source " + SOURCE_REVISION
            + " — https://github.com/CuCl-2/AnyAudio-Judge/tree/" + SOURCE_REVISION
        )
    }
    implementation_sources = ("../third_party/anyaudio_judge/NOTICE",)
    default_config = {
        "model_size": "7b", "model_path": None, "decomposer_model_path": None,
        "source_root": None, "models_dir": "models", "device_map": "auto",
        "dtype": "bfloat16", "rubric": None, "subset": "mix", "clean_rubric": True,
    }
    models = [
        {"id": model, "type": "huggingface", "revision": revision,
         "auto_download": True, "task": "AnyAudio-Judge audio rubric scorer"}
        for model, revision in JUDGE_MODELS.values()
    ] + [{
        "id": DECOMPOSER_MODEL, "type": "huggingface", "revision": DECOMPOSER_REVISION,
        "auto_download": True, "task": "Caption decomposition and rubric cleaning",
        "notes": "Loaded only for caption inputs without a supplied rubric",
    }]
    metric_info = {
        "anyaudio_judge_score": (
            "Mean author-logits yes probability over valid rubric items (0-1, higher=better)"
        )
    }
    metric_groups = {"anyaudio_judge_score": "audio"}

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._backend: Optional[str] = None
        self._judge = None
        self._source_root: Optional[Path] = None
        self._decomposer_path: Optional[Path] = None
        self._decomposition_cache: Dict[Tuple[str, str, bool], List[Dict[str, str]]] = {}
        self._decomposition_modes: Dict[Tuple[str, str, bool], str] = {}

    def setup(self) -> None:
        if self.test_mode:
            return
        try:
            self._prepare_backend()
        except Exception as exc:
            self._judge = None
            self._backend = "unavailable"
            logger.warning("AnyAudio-Judge setup failed: %s", exc)

    def _prepare_backend(self) -> None:
        from ayase.config import download_hf_snapshot

        size = str(self.config.get("model_size", "7b")).lower()
        if size not in JUDGE_MODELS:
            raise ValueError("model_size must be '7b' or '30b'")
        import torch

        dtype_name = str(self.config.get("dtype", "bfloat16"))
        dtype_by_name = {
            "bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32,
        }
        if dtype_name not in dtype_by_name:
            raise ValueError("dtype must be 'bfloat16', 'float16', or 'float32'")
        dtype = dtype_by_name[dtype_name]
        models_dir = Path(self.config.get("models_dir", "models")).resolve()
        configured_source = self.config.get("source_root")
        self._source_root = (
            Path(configured_source).resolve() if configured_source else
            _ensure_source(models_dir / ".ayase" / "sources" / f"anyaudio-{SOURCE_REVISION}")
        )
        _verify_source(self._source_root, managed=not bool(configured_source))
        if str(self._source_root) not in sys.path:
            sys.path.insert(0, str(self._source_root))
        package = importlib.import_module("anyaudio_judge")
        if self._source_root not in Path(package.__file__).resolve().parents:
            raise ImportError("Imported anyaudio_judge is not the pinned source tree")
        author = package.AnyAudioJudge

        model_id, revision = JUDGE_MODELS[size]
        model_path = self.config.get("model_path")
        if model_path is None:
            model_path = download_hf_snapshot(model_id, str(models_dir), revision=revision)
        self._judge = author.from_pretrained(
            str(model_path), torch_dtype=dtype,
            device_map=self.config.get("device_map", "auto"), trust_remote_code=True,
        )
        _install_generate_adapter(self._judge)
        self._backend = f"anyaudio-author-logits-{size}"

    def _caption(self, sample: Sample) -> Optional[str]:
        if sample.caption is not None and isinstance(sample.caption.text, str):
            text = sample.caption.text.strip()
            if text:
                return text
        sidecar = sample.path.with_suffix(".txt")
        try:
            text = sidecar.read_text(encoding="utf-8").strip()
            return text or None
        except (OSError, UnicodeError):
            return None

    def _subset(self, sample: Sample) -> Optional[str]:
        raw = sample.metadata.get("anyaudio_judge_subset", self.config.get("subset", "mix"))
        return _SUBSET_ALIASES.get(str(raw).strip().lower())

    def _rubric_for(self, sample: Sample) -> Optional[Tuple[List[Dict[str, str]], str, str]]:
        if "anyaudio_judge_rubric" in sample.metadata:
            rubric = _normalize_rubric(sample.metadata["anyaudio_judge_rubric"])
            return (rubric, "sample_metadata", "supplied") if rubric is not None else None
        if self.config.get("rubric") is not None:
            rubric = _normalize_rubric(self.config.get("rubric"))
            return (rubric, "module_config", "supplied") if rubric is not None else None
        caption = self._caption(sample)
        subset = self._subset(sample)
        if caption is None or subset is None:
            return None
        clean = bool(self.config.get("clean_rubric", True))
        key = (caption, subset, clean)
        if key not in self._decomposition_cache:
            rubric, rubric_mode = self._decompose(caption, subset, clean)
            normalized = _normalize_rubric(rubric)
            if normalized is None:
                return None
            self._decomposition_cache[key] = normalized
            self._decomposition_modes[key] = rubric_mode
        rubric = self._decomposition_cache[key]
        return rubric, "caption", self._decomposition_modes[key]

    def _decompose(
        self, caption: str, subset: str, clean: bool
    ) -> Tuple[List[Dict[str, str]], str]:
        from ayase.config import download_hf_snapshot

        if self._source_root is None:
            raise RuntimeError("AnyAudio-Judge source is not loaded")
        if self._decomposer_path is None:
            configured = self.config.get("decomposer_model_path")
            self._decomposer_path = Path(configured).resolve() if configured else download_hf_snapshot(
                DECOMPOSER_MODEL, str(Path(self.config.get("models_dir", "models")).resolve()),
                revision=DECOMPOSER_REVISION,
            )
        source = importlib.import_module("anyaudio_judge.decompose")
        decomposed, final = source.decompose_instruction(
            caption, subset=subset, clean=clean, model_name_or_path=str(self._decomposer_path),
            torch_dtype=str(self.config.get("dtype", "bfloat16")),
            device_map=self.config.get("device_map", "auto"), return_intermediate=True,
        )
        fallback = clean and final != decomposed and final == [{
            "dimension": "整体", "question": caption, "basis": caption,
        }]
        return final, "holistic_fallback" if fallback else "decomposed_cleaned" if clean else "decomposed"

    def _audio_path(self, sample: Sample, work: Path) -> Path:
        if not sample.is_video:
            return sample.path
        from ayase.managed_ffmpeg import resolve_media_tool

        output = work / "audio.wav"
        command = [
            str(resolve_media_tool("ffmpeg", models_dir=self.config.get("models_dir", "models"))),
            "-v", "error", "-nostdin", "-y", "-i", str(sample.path), "-map", "0:a:0",
            "-vn", "-c:a", "pcm_f32le", str(output),
        ]
        subprocess.run(command, check=True, capture_output=True, timeout=3600)
        return output

    @staticmethod
    def _load_source_audio(path: Path) -> Any:
        source = importlib.import_module("anyaudio_judge.judge")
        return source._load_audio(str(path))

    def process(self, sample: Sample) -> Sample:
        if self._backend is None or self._backend == "unavailable" or self._judge is None:
            return sample
        try:
            with tempfile.TemporaryDirectory(prefix="ayase-anyaudio-") as tmp:
                audio = self._load_source_audio(self._audio_path(sample, Path(tmp)))
                import numpy as np
                if not isinstance(audio, np.ndarray) or audio.size == 0 or not np.isfinite(audio).all():
                    logger.warning("AnyAudio-Judge found invalid or empty audio in %s", sample.path)
                    return sample
                resolved = self._rubric_for(sample)
                if resolved is None:
                    logger.warning("AnyAudio-Judge needs a valid rubric or caption for %s", sample.path)
                    return sample
                rubric, rubric_source, rubric_mode = resolved
                # ndarray input bypasses a second author loader pass; this is its exact mono/16 kHz result.
                result = self._judge.judge(audio, rubric, mode="logits")
            score = float(result.score)
            items = [item.to_dict() for item in result.items]
            for item in items:
                probability = item.get("p_yes")
                if isinstance(probability, (int, float)) and not math.isfinite(probability):
                    item["p_yes"] = None
            valid = sum(
                isinstance(item.get("p_yes"), (int, float))
                and math.isfinite(float(item["p_yes"])) for item in items
            )
            if not math.isfinite(score) or not valid:
                return sample
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.anyaudio_judge_score = score
            model_id, model_revision = JUDGE_MODELS[str(self.config.get("model_size", "7b")).lower()]
            sample.metadata["anyaudio_judge"] = {
                "mode": "logits", "backend": self._backend, "items": items,
                "valid_items": valid, "total_items": len(rubric),
                "coverage": valid / len(rubric), "rubric_source": rubric_source,
                "rubric_mode": rubric_mode, "rubric_sha256": _rubric_identity(rubric),
                "source_revision": SOURCE_REVISION, "model_id": model_id,
                "model_revision": model_revision,
                "decomposer_model_id": DECOMPOSER_MODEL if rubric_source == "caption" else None,
                "decomposer_revision": DECOMPOSER_REVISION if rubric_source == "caption" else None,
            }
        except Exception as exc:
            logger.warning("AnyAudio-Judge failed for %s: %s", sample.path, exc)
        return sample

    def on_dispose(self) -> None:
        self._judge = None
        self._backend = None
        self._decomposition_cache.clear()
        self._decomposition_modes.clear()
        source = sys.modules.get("anyaudio_judge.decompose")
        cached = getattr(source, "_cached_decomposer", None) if source is not None else None
        if cached is not None and hasattr(cached, "cache_clear"):
            cached.cache_clear()
        self._decomposer_path = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
