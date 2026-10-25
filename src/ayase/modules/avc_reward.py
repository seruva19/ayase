"""Audio-video consistency scoring with the research-only AVC-Reward model.

The native Transformers adaptation reports three 0--1 judgments and their
arithmetic mean. Higher scores indicate stronger audio-video consistency.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

SOURCE_REVISION = "eb0659862ff6888dcc8e92e3bf69c7590d59d54d"
MODEL_ID = "Tsuki0512/AVC-Reward"
MODEL_REVISION = "84dd3998ecc6070925f08dca37132a424e667a52"
PROTOCOL_URL = (
    "https://raw.githubusercontent.com/Tsuki0512/AVC-Reward/"
    + SOURCE_REVISION
    + "/reward_model/code/rm_reward_client.py"
)
PROTOCOL_SHA256 = "83949a4216e54314ce4db511fd76db47b4fd0066184c24ef566462ffc32b91df"
SCORE_KEYS = (
    "macro_semantic_alignment",
    "temporal_synchronicity",
    "fine_grained_event_correspondence",
)
ALIASES = {
    "macro_semantic_alignment": (
        "macro_semantic_alignment",
        "macro_semantic",
        "semantic_alignment",
        "macro-semantic alignment",
        "macro semantic alignment",
    ),
    "temporal_synchronicity": (
        "temporal_synchronicity",
        "temporal_synchronization",
        "temporal_sync",
        "temporal",
        "temporal synchronicity",
        "temporal synchronization",
    ),
    "fine_grained_event_correspondence": (
        "fine_grained_event_correspondence",
        "fine_grained_correspondence",
        "fine_grained",
        "fine-grained event correspondence",
        "fine grained event correspondence",
    ),
}
_AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wma")


def _creation_flags() -> int:
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0


def _normalized_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def parse_avc_reward_output(text: str) -> Dict[str, float]:
    """Parse the first complete score object, then author-compatible aliases/labels."""
    tail = text.split("</think>", 1)[1] if "</think>" in text else text
    decoder = json.JSONDecoder()
    candidates = []
    for match in re.finditer(r"\{", tail):
        try:
            payload, _ = decoder.raw_decode(tail[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            candidates.append(payload)

    values: Dict[str, Any] = {}
    for payload in candidates:
        normalized = {_normalized_key(key): value for key, value in payload.items()}
        found = {}
        for key, aliases in ALIASES.items():
            for alias in aliases:
                alias_key = _normalized_key(alias)
                if alias_key in normalized:
                    found[key] = normalized[alias_key]
                    break
        if len(found) == len(SCORE_KEYS):
            values = found
            break

    if not values:
        for key, aliases in ALIASES.items():
            for alias in aliases:
                matches = re.findall(
                    rf"[\"']?{re.escape(alias)}[\"']?"
                    rf"\s*[:=]\s*[\"']?([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
                    tail,
                    flags=re.I,
                )
                if matches:
                    values[key] = matches[-1]
                    break

    scores: Dict[str, float] = {}
    for key in SCORE_KEYS:
        if key not in values or isinstance(values[key], bool):
            raise ValueError(f"AVC-Reward output is missing {key}")
        value = float(values[key])
        if not math.isfinite(value):
            raise ValueError(f"AVC-Reward output contains a non-finite {key}")
        scores[key] = max(0.0, min(1.0, value))
    scores["score"] = sum(scores[key] for key in SCORE_KEYS) / len(SCORE_KEYS)
    return scores


def _load_protocol(path: Path) -> Tuple[str, str]:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("AVC-Reward protocol asset checksum mismatch")
    tree = ast.parse(data.decode("utf-8"), filename=str(path))
    constants: Dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value_node = node.value
        for target in targets:
            if isinstance(target, ast.Name) and target.id in {"SYSTEM_PROMPT", "USER_PROMPT"}:
                value = ast.literal_eval(value_node)
                if not isinstance(value, str):
                    raise ValueError(f"AVC-Reward {target.id} is not text")
                constants[target.id] = value
    if set(constants) != {"SYSTEM_PROMPT", "USER_PROMPT"}:
        raise ValueError("AVC-Reward protocol asset is missing prompt constants")
    return constants["SYSTEM_PROMPT"], constants["USER_PROMPT"]


def _ensure_protocol(models_dir: Path, override: Optional[object] = None) -> Path:
    if override is not None:
        path = Path(str(override)).expanduser().resolve()
        _load_protocol(path)
        return path
    destination = (
        models_dir / ".ayase" / "sources" / "avc_reward" / SOURCE_REVISION / "rm_reward_client.py"
    )
    if destination.is_file():
        _load_protocol(destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        urllib.request.urlretrieve(PROTOCOL_URL, temporary)
        _load_protocol(temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _validate_loading_info(loading_info: Dict[str, Any]) -> None:
    """Reject incomplete checkpoints instead of retaining random parameters."""
    failures = {
        key: loading_info.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if loading_info.get(key)
    }
    if failures:
        summary = ", ".join(f"{key}={len(value)}" for key, value in failures.items())
        raise ValueError(f"AVC-Reward checkpoint did not load exactly: {summary}")


class AVCRewardModule(PipelineModule):
    name = "avc_reward"
    description = "AVC-Reward audio-video semantic, temporal, and event consistency"
    input_type = "video"
    provenance = {
        field: "adapted"
        for field in (
            "avc_reward_score",
            "avc_macro_semantic_score",
            "avc_temporal_score",
            "avc_event_correspondence_score",
        )
    }
    sources = {
        field: (
            "Native Transformers adaptation of AVC-Reward protocol "
            + SOURCE_REVISION
            + " — https://github.com/Tsuki0512/AVC-Reward/tree/"
            + SOURCE_REVISION
        )
        for field in provenance
    }
    deviations = {
        field: (
            "Ayase uses pinned Transformers 4.57.6 with SDPA instead of the released vLLM "
            "0.12 runtime. It reproduces vLLM's at-most-32 uniform raw-frame selection and "
            "dropped FPS metadata, but native processing has not been proven numerically identical. "
            "Ayase also rejects "
            "non-finite/bool scores and a missing explicit audio path instead of falling back."
        )
        for field in provenance
    }
    implementation_sources = ("../third_party/avc_reward/NOTICE",)
    default_config = {
        "model_path": None,
        "protocol_path": None,
        "models_dir": "models",
        "device_map": "auto",
        "dtype": "auto",
        "max_new_tokens": 4096,
        "audio_path": None,
    }
    models = [
        {
            "id": MODEL_ID,
            "type": "huggingface",
            "revision": MODEL_REVISION,
            "auto_download": True,
            "size": "63.44 GB",
            "vram": "BF16 weights are 63.44 GB plus activations (CPU float32 is about 126.9 GB)",
            "task": "AVC-Reward audio-video consistency judge",
            "notes": "Research use only; checkpoint license metadata is 'other' and incomplete.",
        }
    ]
    metric_info = {
        "avc_reward_score": "Mean AVC-Reward score (0-1, higher=better)",
        "avc_macro_semantic_score": "Macro-semantic audio-video alignment (0-1)",
        "avc_temporal_score": "Audio-video temporal synchronicity (0-1)",
        "avc_event_correspondence_score": "Fine-grained event correspondence (0-1)",
    }
    metric_groups = {field: "alignment" for field in metric_info}

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._backend: Optional[str] = None
        self._model = None
        self._processor = None
        self._process_mm_info = None
        self._system_prompt: Optional[str] = None
        self._user_prompt: Optional[str] = None
        self._actual_model: Optional[str] = None
        self._model_identity: Dict[str, Any] = {}
        self._last_sampling: Dict[str, Any] = {}

    def setup(self) -> None:
        if self.test_mode:
            return
        try:
            self._prepare_backend()
        except Exception as exc:
            self._backend = "unavailable"
            self._model = self._processor = None
            logger.warning("AVC-Reward setup failed: %s", exc)

    def _prepare_backend(self) -> None:
        import torch
        from qwen_omni_utils import process_mm_info
        from transformers import (
            AutoConfig,
            Qwen3OmniMoeForConditionalGeneration,
            Qwen3OmniMoeProcessor,
        )
        from ayase.config import download_hf_snapshot

        dtype_name = str(self.config.get("dtype", "auto")).lower()
        if dtype_name == "auto":
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        elif dtype_name in {"bfloat16", "float16", "float32"}:
            dtype = getattr(torch, dtype_name)
        else:
            raise ValueError("dtype must be auto, bfloat16, float16, or float32")
        models_dir = Path(self.config.get("models_dir", "models")).expanduser().resolve()
        protocol = _ensure_protocol(models_dir, self.config.get("protocol_path"))
        self._system_prompt, self._user_prompt = _load_protocol(protocol)
        model_path = self.config.get("model_path") or download_hf_snapshot(
            MODEL_ID,
            str(models_dir),
            revision=MODEL_REVISION,
        )
        checkpoint_config = AutoConfig.from_pretrained(str(model_path))
        architectures = tuple(getattr(checkpoint_config, "architectures", ()) or ())
        if (
            getattr(checkpoint_config, "model_type", None) != "qwen3_omni_moe"
            or "Qwen3OmniMoeForConditionalGeneration" not in architectures
        ):
            raise ValueError("AVC-Reward checkpoint is not a full Qwen3-Omni-MoE model")
        configured_model = self.config.get("model_path")
        self._actual_model = str(model_path)
        self._processor = Qwen3OmniMoeProcessor.from_pretrained(str(model_path), use_fast=True)
        model, loading_info = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            device_map=self.config.get("device_map", "auto"),
            attn_implementation="sdpa",
            enable_audio_output=False,
            output_loading_info=True,
        )
        _validate_loading_info(loading_info)
        self._model = model
        self._model.eval()
        self._process_mm_info = process_mm_info
        self._model_identity = {
            "requested": str(configured_model) if configured_model else MODEL_ID,
            "resolved": self._actual_model,
            "revision": None if configured_model else MODEL_REVISION,
            "expert_override": bool(configured_model),
            "dtype": dtype_name,
            "device_map": self.config.get("device_map", "auto"),
            "processor": type(self._processor).__name__,
            "video_processor": type(getattr(self._processor, "video_processor", None)).__name__,
            "image_processor": type(getattr(self._processor, "image_processor", None)).__name__,
            "use_fast": True,
        }
        self._backend = "transformers-4.57.6-native-adaptation"

    def _audio_input(self, sample: Sample, work: Path) -> Tuple[Path, str]:
        explicit = self.config.get("audio_path")
        origin = "config"
        if explicit is None and "avc_audio_path" in sample.metadata:
            explicit = sample.metadata["avc_audio_path"]
            origin = "sample_metadata"
        if explicit is not None:
            path = Path(str(explicit)).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Explicit AVC-Reward audio does not exist: {path}")
            return path, origin
        for extension in _AUDIO_EXTENSIONS:
            sidecar = sample.path.with_suffix(extension)
            if sidecar.is_file():
                return sidecar.resolve(), "sidecar"
        from ayase.managed_ffmpeg import resolve_media_tool

        output = work / "audio.wav"
        command = [
            str(resolve_media_tool("ffmpeg", models_dir=self.config.get("models_dir", "models"))),
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(sample.path),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output),
        ]
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            timeout=3600,
            creationflags=_creation_flags(),
        )
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError("AVC-Reward could not extract a non-empty audio stream")
        return output, "embedded_audio"

    @staticmethod
    def _video_frames(path: Path) -> Tuple[Any, Dict[str, Any]]:
        """Reproduce vLLM 0.12 OpenCVVideoBackend's default frame selection."""
        import cv2
        import numpy as np

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"Cannot open AVC-Reward video: {path}")
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count <= 0:
            capture.release()
            raise ValueError(f"AVC-Reward video reports no frames: {path}")
        selected_count = min(32, count)
        indices = np.linspace(0, count - 1, selected_count, dtype=np.int64)
        selected = {int(index) for index in indices}
        frames = []
        try:
            for frame_index in range(count):
                if not capture.grab():
                    break
                if frame_index in selected:
                    ok, frame = capture.retrieve()
                    if not ok:
                        raise ValueError(f"Cannot decode AVC-Reward frame {frame_index}: {path}")
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
        if len(frames) != selected_count:
            raise ValueError(
                f"AVC-Reward decoded {len(frames)} of {selected_count} selected frames: {path}"
            )
        video = np.stack(frames)
        return video, {
            "decoder": "opencv",
            "total_frames": count,
            "selected_frames": selected_count,
            "indices": indices.tolist(),
            "source_fps": source_fps,
            "processor_fps": 1.0,
            "fps_metadata_forwarded": False,
        }

    def _generate(self, video: Path, audio: Path) -> str:
        if not all(
            (
                self._model,
                self._processor,
                self._process_mm_info,
                self._system_prompt,
                self._user_prompt,
            )
        ):
            raise RuntimeError("AVC-Reward backend is not ready")
        messages = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video.resolve())},
                    {"type": "audio", "audio": str(audio.resolve())},
                    {"type": "text", "text": self._user_prompt},
                ],
            },
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        audio_messages = [{"role": "user", "content": [messages[1]["content"][1]]}]
        audio_media = self._process_mm_info(
            audio_messages, use_audio_in_video=False, return_video_kwargs=True
        )
        audios = audio_media[0]
        import numpy as np

        if not audios:
            raise ValueError(f"AVC-Reward decoded no audio: {audio}")
        unpadded_lengths = [int(len(waveform)) for waveform in audios]
        audios = [np.pad(waveform, (0, (-len(waveform)) % 160)) for waveform in audios]
        padded_lengths = [int(len(waveform)) for waveform in audios]
        video_input, sampling = self._video_frames(video)
        images, videos = None, [video_input]
        self._last_sampling = {
            **sampling,
            "helper": "qwen-omni-utils==0.0.8 (audio only)",
            "use_audio_in_video": False,
            "audio_unpadded_samples": unpadded_lengths,
            "audio_padded_samples": padded_lengths,
        }
        inputs = self._processor(
            text=[text],
            audio=audios,
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
            use_audio_in_video=False,
        )
        parameter = next(self._model.parameters())
        device = parameter.device
        dtype = parameter.dtype
        thinker = getattr(self._model, "thinker", self._model)
        audio_tower = getattr(thinker, "audio_tower", None)
        audio_dtype = next(audio_tower.parameters()).dtype if audio_tower is not None else dtype
        prepared = {}
        for key, value in inputs.items():
            if (
                key in {"input_features", "pixel_values", "pixel_values_videos"}
                and hasattr(value, "is_floating_point")
                and value.is_floating_point()
            ):
                target_dtype = audio_dtype if key == "input_features" else dtype
                prepared[key] = value.to(device=device, dtype=target_dtype)
            else:
                prepared[key] = value.to(device) if hasattr(value, "to") else value
        inputs = prepared
        import torch

        with torch.inference_mode():
            result = self._model.generate(
                **inputs,
                do_sample=False,
                return_audio=False,
                use_audio_in_video=False,
                thinker_return_dict_in_generate=True,
                thinker_max_new_tokens=int(self.config.get("max_new_tokens", 4096)),
            )
        thinker = result[0] if isinstance(result, tuple) else result
        sequences = thinker.sequences
        suffix = sequences[:, inputs["input_ids"].shape[1] :]
        return self._processor.batch_decode(suffix, skip_special_tokens=True)[0]

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video or self._backend in {None, "unavailable"}:
            return sample
        try:
            models_dir = Path(self.config.get("models_dir", "models")).resolve()
            temporary_root = models_dir / ".ayase" / "tmp"
            temporary_root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="avc-reward-", dir=temporary_root) as tmp:
                audio, audio_origin = self._audio_input(sample, Path(tmp))
                response = self._generate(sample.path, audio)
            scores = parse_avc_reward_output(response)
            metrics = sample.quality_metrics or QualityMetrics()
            metrics.avc_reward_score = scores["score"]
            metrics.avc_macro_semantic_score = scores["macro_semantic_alignment"]
            metrics.avc_temporal_score = scores["temporal_synchronicity"]
            metrics.avc_event_correspondence_score = scores["fine_grained_event_correspondence"]
            sample.quality_metrics = metrics
            sample.metadata["avc_reward"] = {
                "response": response,
                "audio_origin": audio_origin,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "source_revision": SOURCE_REVISION,
                "model": self._model_identity,
                "runtime": self._backend,
                "sampling": {
                    **self._last_sampling,
                    "max_new_tokens": int(self.config.get("max_new_tokens", 4096)),
                    "do_sample": False,
                },
            }
        except Exception as exc:
            logger.warning("AVC-Reward failed for %s: %s", sample.path, exc)
        return sample

    def on_dispose(self) -> None:
        super().on_dispose()
        self._model = self._processor = self._process_mm_info = None
        self._backend = None
        self._actual_model = None
        self._model_identity = {}
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
