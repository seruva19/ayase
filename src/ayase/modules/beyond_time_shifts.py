"""Score audio-video synchronization with the published Beyond Time Shifts evaluator.

The author protocol keeps the first 60 reader-sampled frames requested at 12
fps and 140 square pixels plus the first 80,000 embedded-audio samples. This is
nominally five seconds, but qwen-omni-utils caps decoding at 768 frames, so the
60-frame window can span longer wall time on long videos. Short inputs are
zero-padded. The final hidden token feeds a trained unbounded linear readout.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import math
import os
import sys
import tempfile
import threading
import urllib.request
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Optional

from ayase.config import download_hf_snapshot, download_model_file
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

SOURCE_REVISION = "f93e5ffaf8af144d37c39784426a8e0029734026"
SOURCE_ARCHIVE = (
    "https://codeload.github.com/chenhaoqcdyq/BeyondTimeShifts/zip/" + SOURCE_REVISION
)
BASE_MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
BASE_MODEL_REVISION = "f75b40e3da2003cdd6e1829b1f420ca70797c34e"
CHECKPOINT_ID = "qianyijie/avsync-evaluator"
CHECKPOINT_REVISION = "5b88707fd6facf12fe627c5ed44196c0e689e467"
CHECKPOINT_FILE = "avsync_eval_weights.pt"
CHECKPOINT_SHA256 = "ba36a405792367a15ee1ff69ac167ec4644bd2f09d87ed3286c1e0aecf10c0e6"
CHECKPOINT_URL = (
    f"https://huggingface.co/{CHECKPOINT_ID}/resolve/{CHECKPOINT_REVISION}/{CHECKPOINT_FILE}"
)
TARGET_AUDIO = 80_000
VIDEO_FPS = 12
VIDEO_SIZE = 140
TARGET_FRAMES = VIDEO_FPS * 5

_REQUIRED_SOURCE = (
    "avsync_eval/models/evaluator.py",
    "avsync_eval/models/hacked_qwen.py",
    "avsync_eval/qwen2_5_omni/modeling_qwen2_5_omni.py",
    "avsync_eval/qwen2_5_omni/processing_qwen2_5_omni.py",
    "avsync_eval/qwen2_vl/video_processing_qwen2_vl.py",
)
_DECODER_ENVIRONMENT_LOCK = threading.RLock()


def _verify_source(root: Path, *, managed: bool) -> None:
    missing = [name for name in _REQUIRED_SOURCE if not (root / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete Beyond Time Shifts source tree: {root}")
    marker = root / ".ayase-source-revision"
    if managed and not marker.is_file():
        raise ValueError(f"Unverified Beyond Time Shifts source tree: {root}")
    if marker.is_file() and marker.read_text(encoding="ascii").strip() != SOURCE_REVISION:
        raise ValueError(f"Beyond Time Shifts source revision mismatch: {root}")


def _ensure_source(destination: Path) -> Path:
    """Acquire and safely extract the immutable author archive without Git."""
    from ayase.managed_runtime import _SetupLock

    destination.parent.mkdir(parents=True, exist_ok=True)
    with _SetupLock(destination.with_suffix(".lock")):
        if destination.is_dir():
            _verify_source(destination, managed=True)
            return destination
        with tempfile.TemporaryDirectory(prefix="bts-source-", dir=destination.parent) as name:
            work = Path(name)
            archive = work / "source.zip"
            urllib.request.urlretrieve(SOURCE_ARCHIVE, archive)
            unpacked = work / "unpacked"
            unpacked.mkdir()
            with zipfile.ZipFile(archive) as bundle:
                members = bundle.infolist()
                roots = {Path(item.filename).parts[0] for item in members if Path(item.filename).parts}
                if len(roots) != 1:
                    raise ValueError("Beyond Time Shifts archive must contain one root")
                base = unpacked.resolve()
                for item in members:
                    target = (unpacked / item.filename).resolve()
                    if target != base and base not in target.parents:
                        raise ValueError(f"Unsafe source archive member: {item.filename}")
                    if item.external_attr >> 16 & 0o170000 == 0o120000:
                        raise ValueError(f"Source archive contains a symlink: {item.filename}")
                bundle.extractall(unpacked)
            extracted = unpacked / roots.pop()
            (extracted / ".ayase-source-revision").write_text(
                SOURCE_REVISION, encoding="ascii"
            )
            _verify_source(extracted, managed=True)
            extracted.replace(destination)
    return destination


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_checkpoint_file(path: Path) -> None:
    actual = _sha256(path)
    if actual != CHECKPOINT_SHA256:
        raise ValueError(
            f"Beyond Time Shifts checkpoint digest mismatch: expected {CHECKPOINT_SHA256}, "
            f"got {actual}"
        )


def _checkpoint_state(checkpoint: Any) -> Mapping[str, Any]:
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, Mapping) else None
    if not isinstance(state, Mapping):
        raise ValueError("Beyond Time Shifts checkpoint contains no state_dict")
    return state


def _validate_checkpoint_state(state: Mapping[str, Any], evaluator: Any) -> None:
    """Reject random-head and materially partial transformer checkpoints."""
    import torch

    expected_head = {
        "regression_head.fc_1.weight": (1, 2048),
        "regression_head.fc_1.bias": (1,),
    }
    for key, shape in expected_head.items():
        tensor = state.get(key)
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise ValueError(f"Missing or invalid trained tensor {key}; expected shape {shape}")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"Checkpoint tensor {key} contains non-finite values")
    supplied_head = {key for key in state if key.startswith("regression_head.")}
    unexpected_head = sorted(supplied_head - set(expected_head))
    if unexpected_head:
        raise ValueError(
            "Checkpoint contains unexpected regression-head tensors: "
            + ", ".join(unexpected_head)
        )

    expected = evaluator.transformer.state_dict()
    supplied = {
        key.removeprefix("transformer."): value
        for key, value in state.items()
        if key.startswith("transformer.")
    }
    if not supplied:
        raise ValueError("Checkpoint contains no transformer tensors")
    unexpected = sorted(set(supplied) - set(expected))
    missing = sorted(set(expected) - set(supplied))
    mismatched = [
        key
        for key in set(supplied) & set(expected)
        if not hasattr(supplied[key], "shape")
        or tuple(supplied[key].shape) != tuple(expected[key].shape)
    ]
    if missing or unexpected or mismatched:
        raise ValueError(
            "Checkpoint transformer layout is incompatible with the pinned author model: "
            f"{len(missing)} missing, {len(unexpected)} unexpected, "
            f"{len(mismatched)} shape mismatches"
        )


@contextmanager
def _decoder_environment(ffmpeg_path: Optional[Path]):
    """Force the tested qwen-utils backend and expose Ayase's private FFmpeg."""
    with _DECODER_ENVIRONMENT_LOCK:
        old_reader = os.environ.get("FORCE_QWENVL_VIDEO_READER")
        old_path = os.environ.get("PATH")
        os.environ["FORCE_QWENVL_VIDEO_READER"] = "torchvision"
        if ffmpeg_path is not None:
            directory = str(ffmpeg_path.parent)
            os.environ["PATH"] = directory + os.pathsep + (old_path or "")
        vision = sys.modules.get("qwen_omni_utils.v2_5.vision_process")
        old_global = (
            getattr(vision, "FORCE_QWENVL_VIDEO_READER", None)
            if vision is not None
            else old_reader
        )
        if vision is not None:
            vision.FORCE_QWENVL_VIDEO_READER = "torchvision"
            cache = getattr(vision, "get_video_reader_backend", None)
            if hasattr(cache, "cache_clear"):
                cache.cache_clear()
        try:
            yield
        finally:
            current_vision = sys.modules.get("qwen_omni_utils.v2_5.vision_process")
            if current_vision is not None:
                current_vision.FORCE_QWENVL_VIDEO_READER = old_global
                cache = getattr(current_vision, "get_video_reader_backend", None)
                if hasattr(cache, "cache_clear"):
                    cache.cache_clear()
            if old_reader is None:
                os.environ.pop("FORCE_QWENVL_VIDEO_READER", None)
            else:
                os.environ["FORCE_QWENVL_VIDEO_READER"] = old_reader
            if old_path is None:
                os.environ.pop("PATH", None)
            else:
                os.environ["PATH"] = old_path


def load_clip(
    video_path: str | Path,
    fps: int = VIDEO_FPS,
    size: int = VIDEO_SIZE,
    ffmpeg_path: Optional[Path] = None,
):
    """Apply the author's paired decode and exact five-second trim/zero padding."""
    import numpy as np
    import torch

    conversation = {
        "role": "user",
        "content": [
            {"type": "text", "text": "This is the video: "},
            {
                "type": "video",
                "video": str(video_path),
                "fps": fps,
                "resized_height": size,
                "resized_width": size,
            },
        ],
    }
    with _decoder_environment(ffmpeg_path):
        from qwen_omni_utils import process_mm_info

        audios, _, videos = process_mm_info([conversation], use_audio_in_video=True)
    if not audios or not videos:
        raise ValueError("video has no paired embedded audio or decoded frames")
    video, audio = videos[0], audios[0]
    if isinstance(video, torch.Tensor):
        video = video.detach().cpu().numpy()
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    video = np.asarray(video)
    audio = np.asarray(audio)
    if video.ndim != 4 or len(video) == 0 or audio.ndim != 1 or len(audio) == 0:
        raise ValueError("video has no paired embedded audio or decoded frames")
    target_frames = fps * 5
    if len(video) < target_frames:
        # Preserve the author demo exactly: np.zeros defaults to float64, so a
        # short non-float64 clip is promoted during concatenation.
        padding = np.concatenate(
            [np.zeros((1, 3, size, size)) for _ in range(target_frames - len(video))],
            axis=0,
        )
        video = np.concatenate((video, padding), axis=0)
    else:
        video = video[:target_frames]
    if len(audio) < TARGET_AUDIO:
        audio = np.pad(audio, (0, TARGET_AUDIO - len(audio)), mode="constant")
    else:
        audio = audio[:TARGET_AUDIO]
    if video.shape != (target_frames, 3, size, size):
        raise ValueError(
            "decoded video does not match the author input shape "
            f"{(target_frames, 3, size, size)}: {video.shape}"
        )
    if audio.shape != (TARGET_AUDIO,):
        raise ValueError(
            f"decoded audio does not match the author input shape {(TARGET_AUDIO,)}: "
            f"{audio.shape}"
        )
    if not bool(np.isfinite(video).all()) or not bool(np.isfinite(audio).all()):
        raise ValueError("assessed audio-video region contains non-finite values")
    return torch.from_numpy(video), torch.from_numpy(audio)


class BeyondTimeShiftsModule(PipelineModule):
    """Published Beyond Time Shifts native audio-video synchronization scorer."""

    name = "beyond_time_shifts"
    description = "Beyond Time Shifts native audio-video synchronization score"
    input_type = "video"
    provenance = "published"
    sources = {
        "beyond_time_shifts_score": (
            "Beyond Time Shifts source " + SOURCE_REVISION
            + " — https://github.com/chenhaoqcdyq/BeyondTimeShifts/tree/" + SOURCE_REVISION
        )
    }
    implementation_sources = (
        "../third_party/beyond_time_shifts/LICENSE",
        "../third_party/beyond_time_shifts/NOTICE",
    )
    models = [
        {
            "id": BASE_MODEL_ID,
            "type": "huggingface",
            "revision": BASE_MODEL_REVISION,
            "task": "Qwen2.5-Omni-3B backbone for the AV-sync evaluator",
            "auto_download": True,
            "notes": "Qwen Research License; non-commercial use only",
        },
        {
            "id": CHECKPOINT_ID,
            "type": "huggingface",
            "revision": CHECKPOINT_REVISION,
            "task": "Full trained Beyond Time Shifts AV-sync evaluator state",
            "size": "9,407,345,470 bytes",
            "vram": (
                "~18.8 GB weights during default SDPA/eager FP32 CUDA transfer; "
                "~9.4 GB BF16 weights afterward, plus activations"
            ),
            "auto_download": True,
            "notes": (
                f"SHA256 {CHECKPOINT_SHA256}; 4,703,464,448 parameters initialized FP32 on CPU; "
                "exact 1,442-tensor transformer state plus "
                "two-tensor trained head; high host/GPU peak memory; base-model terms survive"
            ),
        },
    ]
    metric_info = {
        "beyond_time_shifts_score": (
            "Raw unbounded native AV-sync scalar over the first 60 reader-sampled frames "
            "and 80,000 audio samples (nominally 5s; higher=better)"
        )
    }
    metric_groups = {"beyond_time_shifts_score": "audio"}
    default_config = {
        "model_path": None,
        "checkpoint_path": None,
        "source_root": None,
        "models_dir": "models",
        "auto_download": True,
        "device": "auto",
        "precision_mode": "bf16",
        "attention": "sdpa",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._evaluator = None
        self._backend = None
        self._ml_available = False
        self._source_root: Optional[Path] = None
        self._ffmpeg_path: Optional[Path] = None
        self.runtime_trace = {
            "source_revision": SOURCE_REVISION,
            "base_model": f"{BASE_MODEL_ID}@{BASE_MODEL_REVISION}",
            "checkpoint": f"{CHECKPOINT_ID}@{CHECKPOINT_REVISION}",
            "protocol": (
                "first 60 qwen-utils reader-sampled frames requested at 12fps/140px + "
                "first 80000 audio samples; nominal 5s; 768-frame reader cap applies; zero-pad"
            ),
            "precision": str(self.config.get("precision_mode", "bf16")),
            "attention": str(self.config.get("attention", "sdpa")),
            "parameters": 4_703_464_448,
            "load_flow": "CPU FP32 -> CUDA FP32 -> CUDA BF16 (author SDPA/eager path)",
        }

    def _resolve_assets(self) -> tuple[Path, Path, Path]:
        models_dir = Path(self.config.get("models_dir", "models")).expanduser().resolve()
        source_override = self.config.get("source_root")
        source = (
            Path(source_override).expanduser().resolve()
            if source_override
            else _ensure_source(
                models_dir / ".ayase" / "sources" / f"beyond-time-shifts-{SOURCE_REVISION}"
            )
        )
        _verify_source(source, managed=not bool(source_override))

        model_override = self.config.get("model_path")
        if model_override:
            model = Path(model_override).expanduser().resolve()
        elif self.config.get("auto_download", True):
            model = download_hf_snapshot(
                BASE_MODEL_ID, str(models_dir), revision=BASE_MODEL_REVISION
            )
        else:
            raise FileNotFoundError("Beyond Time Shifts base model is not provisioned")
        if not model.is_dir():
            raise FileNotFoundError(f"Beyond Time Shifts model_path is not a directory: {model}")

        checkpoint_override = self.config.get("checkpoint_path")
        if checkpoint_override:
            checkpoint = Path(checkpoint_override).expanduser().resolve()
        elif self.config.get("auto_download", True):
            checkpoint = download_model_file(
                f"beyond_time_shifts/{CHECKPOINT_FILE}", CHECKPOINT_URL, str(models_dir)
            )
        else:
            raise FileNotFoundError("Beyond Time Shifts checkpoint is not provisioned")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Beyond Time Shifts checkpoint is not a file: {checkpoint}")
        _validate_checkpoint_file(checkpoint)
        return source, model, checkpoint

    def setup(self) -> None:
        if self.test_mode:
            return
        self._evaluator = None
        self._backend = None
        self._ml_available = False
        self._ffmpeg_path = None
        try:
            import torch

            requested = str(self.config.get("device", "auto"))
            if requested not in {"auto", "cuda"} and not requested.startswith("cuda:"):
                raise ValueError("published Beyond Time Shifts inference requires CUDA")
            if not torch.cuda.is_available():
                raise RuntimeError("published Beyond Time Shifts inference requires CUDA")
            device = "cuda" if requested == "auto" else requested
            attention = str(self.config.get("attention", "sdpa"))
            precision = str(self.config.get("precision_mode", "bf16"))
            if attention not in {"sdpa", "eager", "flash_attention_2"}:
                raise ValueError("attention must be sdpa, eager, or flash_attention_2")
            if precision not in {"bf16", "bf16-mixed"}:
                raise ValueError("precision_mode must be bf16 or bf16-mixed")

            from ayase.managed_ffmpeg import resolve_media_tool

            self._ffmpeg_path = resolve_media_tool(
                "ffmpeg", models_dir=self.config.get("models_dir", "models")
            )
            source, model_path, checkpoint_path = self._resolve_assets()
            self._source_root = source
            source_package_root = source / "avsync_eval"
            for path in (source, source_package_root):
                if str(path) not in sys.path:
                    sys.path.insert(0, str(path))
            module = importlib.import_module("avsync_eval.models.evaluator")
            if source not in Path(module.__file__).resolve().parents:
                raise ImportError("Imported avsync_eval is not the pinned source tree")
            for package_name in ("qwen2_5_omni", "qwen2_vl"):
                package = importlib.import_module(package_name)
                package_file = getattr(package, "__file__", None)
                if package_file is None or source_package_root not in Path(package_file).resolve().parents:
                    raise ImportError(
                        f"Imported {package_name} is not the pinned author source tree"
                    )
            evaluator = module.AVSyncEvaluator(
                model_name=str(model_path),
                v_fps=VIDEO_FPS,
                v_size=VIDEO_SIZE,
                device=device,
                dtype=torch.bfloat16,
                precision_mode=precision,
                attn_implementation=attention,
            )
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            state = _checkpoint_state(checkpoint)
            _validate_checkpoint_state(state, evaluator)
            evaluator.load_eval_checkpoint(state)
            evaluator.to_eval_device()
            self._evaluator = evaluator
            self._backend = (
                f"beyond-time-shifts:{SOURCE_REVISION}:qwen2.5-omni-3b:"
                f"{precision}:{attention}"
            )
            self._ml_available = True
        except Exception as exc:
            self._evaluator = None
            self._backend = "unavailable"
            self._ml_available = False
            logger.warning("Beyond Time Shifts unavailable: %s", exc)

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video or not self._ml_available or self._evaluator is None:
            return sample
        try:
            video, audio = load_clip(
                sample.path, VIDEO_FPS, VIDEO_SIZE, ffmpeg_path=self._ffmpeg_path
            )
            scores = self._evaluator.score_batch([video], [audio])
            if len(scores) != 1:
                raise ValueError(f"Beyond Time Shifts returned {len(scores)} scores for one video")
            score = float(scores[0])
            if not math.isfinite(score):
                raise ValueError(f"Beyond Time Shifts returned non-finite score {score!r}")
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.beyond_time_shifts_score = score
            sample.quality_metrics.metric_backends[self.name] = self._backend
            sample.metadata[self.name] = {
                **self.runtime_trace,
                "backend": self._backend,
                "expert_override": bool(
                    self.config.get("source_root")
                    or self.config.get("model_path")
                    or self.config.get("checkpoint_path")
                ),
            }
        except Exception as exc:
            logger.warning("Beyond Time Shifts failed for %s: %s", sample.path, exc)
        return sample

    def on_dispose(self) -> None:
        self._evaluator = None
        self._backend = None
        self._ml_available = False
        self._ffmpeg_path = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        super().on_dispose()


__all__ = ["BeyondTimeShiftsModule"]
