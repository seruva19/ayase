"""Runtime acceleration helpers for Ayase pipelines.

The helpers in this module are intentionally optional and lightweight. They
avoid importing heavy ML packages until a caller asks for torch-specific
behavior, and they provide graceful fallbacks when a backend is unavailable.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np


_CURRENT_PIPELINE: ContextVar[Optional[Any]] = ContextVar(
    "ayase_current_pipeline",
    default=None,
)
_MISSING = object()


@contextmanager
def pipeline_context(pipeline: Any) -> Iterator[None]:
    """Make *pipeline* discoverable to utility functions for this call stack."""

    token = _CURRENT_PIPELINE.set(pipeline)
    try:
        yield
    finally:
        _CURRENT_PIPELINE.reset(token)


def current_pipeline() -> Optional[Any]:
    """Return the pipeline currently processing a sample, if any."""

    return _CURRENT_PIPELINE.get()


def clone_frames(frames: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Return defensive copies so cached frames cannot be mutated by callers."""

    return [frame.copy() if hasattr(frame, "copy") else frame for frame in frames]


def runtime_module_config(config: Optional[Any]) -> Dict[str, Any]:
    """Return global runtime config values injected into module configs."""

    if config is None or not getattr(config, "general", None):
        return {}

    general = config.general
    return {
        "models_dir": str(general.models_dir),
        "parallel_jobs": general.parallel_jobs,
        "device": general.device,
        "dtype": general.dtype,
        "amp_enabled": general.amp_enabled,
        "attention_backend": general.attention_backend,
        "frame_cache_enabled": general.frame_cache_enabled,
        "timing_enabled": general.timing_enabled,
        "sample_batch_size": general.sample_batch_size,
        "max_clip_images_per_forward": general.max_clip_images_per_forward,
    }


def shared_runtime_resource(owner: Any, key: tuple[Any, ...], factory: Any) -> Any:
    """Return a pipeline-scoped shared resource, or build one without a pipeline."""

    pipeline = getattr(owner, "pipeline", None)
    if pipeline is not None and hasattr(pipeline, "get_runtime_resource"):
        return pipeline.get_runtime_resource(key, factory)
    return factory()


def shared_runtime_value(owner: Any, key: tuple[Any, ...], factory: Any) -> Any:
    """Return a per-sample cached value, or build one without a pipeline."""

    pipeline = getattr(owner, "pipeline", None)
    if pipeline is not None and hasattr(pipeline, "get_runtime_value"):
        return pipeline.get_runtime_value(key, factory)
    return factory()


def cached_runtime_value(owner: Any, key: tuple[Any, ...], default: Any = _MISSING) -> Any:
    """Return a cached runtime value without constructing it."""

    pipeline = getattr(owner, "pipeline", None)
    if pipeline is not None and hasattr(pipeline, "peek_runtime_value"):
        return pipeline.peek_runtime_value(key, default)
    return default


def store_runtime_value(owner: Any, key: tuple[Any, ...], value: Any) -> None:
    """Store a value in the active pipeline runtime cache, when present."""

    pipeline = getattr(owner, "pipeline", None)
    if pipeline is not None and hasattr(pipeline, "set_runtime_value"):
        pipeline.set_runtime_value(key, value)


def _normalize_torch_features(features: Any) -> Any:
    """Extract and L2-normalize a torch feature tensor."""

    from ayase.compat import extract_features

    features = extract_features(features)
    return features / features.norm(p=2, dim=-1, keepdim=True)


def _runtime_config(owner: Any) -> Dict[str, Any]:
    config = getattr(owner, "config", {})
    return config if isinstance(config, dict) else {}


def _max_clip_images_per_forward(owner: Any) -> int:
    config = _runtime_config(owner)
    try:
        return max(1, int(config.get("max_clip_images_per_forward", 64)))
    except (TypeError, ValueError):
        return 64


def _is_cuda_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "out of memory" in message
        and ("cuda" in message or "cublas" in message or "cudnn" in message)
    )


def _empty_cuda_cache(device: Any) -> None:
    if not str(device).startswith("cuda"):
        return
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def cached_clip_text_features(
    owner: Any,
    model: Any,
    processor: Any,
    texts: Sequence[str],
    *,
    model_key: str,
    device: Any,
    cache_key: Optional[tuple[Any, ...]] = None,
    normalize: bool = True,
    truncation: bool = True,
) -> Any:
    """Return cached Hugging Face CLIP text features for a text sequence."""

    text_values = tuple(str(text) for text in texts)
    key = (
        "clip_text_features",
        "transformers",
        model_key,
        str(device),
        bool(normalize),
        bool(truncation),
        cache_key or text_values,
    )

    def compute() -> Any:
        inputs = processor(
            text=list(text_values),
            return_tensors="pt",
            padding=True,
            truncation=truncation,
        ).to(device)
        with torch_inference_context(
            str(device),
            getattr(owner, "config", {}).get("dtype", "auto"),
            bool(getattr(owner, "config", {}).get("amp_enabled", True)),
        ):
            features = model.get_text_features(**inputs)
            if normalize:
                return _normalize_torch_features(features)
            from ayase.compat import extract_features

            return extract_features(features)

    return shared_runtime_value(owner, key, compute)


def clip_image_features_cache_key(
    model_key: str,
    device: Any,
    cache_key: tuple[Any, ...],
    *,
    normalize: bool = True,
) -> tuple[Any, ...]:
    """Return the canonical runtime cache key for CLIP image features."""

    return (
        "clip_image_features",
        "transformers",
        model_key,
        str(device),
        bool(normalize),
        cache_key,
    )


def cached_clip_image_features(
    owner: Any,
    model: Any,
    processor: Any,
    images: Sequence[Any],
    *,
    model_key: str,
    device: Any,
    cache_key: tuple[Any, ...],
    normalize: bool = True,
) -> Any:
    """Return cached Hugging Face CLIP image features for an image sequence."""

    key = clip_image_features_cache_key(
        model_key,
        device,
        cache_key,
        normalize=normalize,
    )

    def compute() -> Any:
        image_values = list(images)
        chunk_size = _max_clip_images_per_forward(owner)
        return _encode_clip_image_chunks(
            owner,
            model,
            processor,
            image_values,
            device=device,
            normalize=normalize,
            chunk_size=chunk_size,
        )

    return shared_runtime_value(owner, key, compute)


def _encode_clip_image_chunks(
    owner: Any,
    model: Any,
    processor: Any,
    images: Sequence[Any],
    *,
    device: Any,
    normalize: bool,
    chunk_size: int,
) -> Any:
    import torch
    from ayase.compat import extract_features

    image_values = list(images)
    if not image_values:
        return torch.empty((0, 0), device=device)

    chunk_size = max(1, int(chunk_size))
    outputs = []
    start = 0
    while start < len(image_values):
        chunk = image_values[start:start + chunk_size]
        try:
            inputs = processor(images=chunk, return_tensors="pt").to(device)
            with torch_inference_context(
                str(device),
                _runtime_config(owner).get("dtype", "auto"),
                bool(_runtime_config(owner).get("amp_enabled", True)),
            ):
                features = model.get_image_features(**inputs)
                if normalize:
                    features = _normalize_torch_features(features)
                else:
                    features = extract_features(features)
                outputs.append(features)
            start += len(chunk)
        except RuntimeError as exc:
            if chunk_size <= 1 or not _is_cuda_oom(exc):
                raise
            _empty_cuda_cache(device)
            chunk_size = max(1, chunk_size // 2)

    if len(outputs) == 1:
        return outputs[0]
    return torch.cat(outputs, dim=0)




def cached_clip_image_feature_groups(
    owner: Any,
    model: Any,
    processor: Any,
    image_groups: Sequence[Sequence[Any]],
    *,
    model_key: str,
    device: Any,
    cache_keys: Sequence[tuple[Any, ...]],
    normalize: bool = True,
) -> List[Any]:
    """Return CLIP image features for multiple samples using one forward pass.

    Cached per-sample groups are reused. Missing groups are flattened, encoded
    together, split back into their original groups, and stored under the same
    per-sample keys used by ``cached_clip_image_features()``.
    """

    if len(image_groups) != len(cache_keys):
        raise ValueError("image_groups and cache_keys must have the same length")

    results: List[Any] = [None] * len(image_groups)
    missing_indices: List[int] = []
    flattened_images: List[Any] = []
    spans: List[tuple[int, int]] = []

    for idx, (images, cache_key) in enumerate(zip(image_groups, cache_keys)):
        key = clip_image_features_cache_key(
            model_key,
            device,
            cache_key,
            normalize=normalize,
        )
        cached = cached_runtime_value(owner, key, _MISSING)
        if cached is not _MISSING:
            results[idx] = cached
            continue
        if not images:
            continue
        start = len(flattened_images)
        flattened_images.extend(images)
        spans.append((start, len(flattened_images)))
        missing_indices.append(idx)

    if flattened_images:
        batch_key = (
            "batch",
            tuple(cache_keys[idx] for idx in missing_indices),
        )
        batch_features = cached_clip_image_features(
            owner,
            model,
            processor,
            flattened_images,
            model_key=model_key,
            device=device,
            cache_key=batch_key,
            normalize=normalize,
        )

        for idx, (start, end) in zip(missing_indices, spans):
            features = batch_features[start:end]
            key = clip_image_features_cache_key(
                model_key,
                device,
                cache_keys[idx],
                normalize=normalize,
            )
            store_runtime_value(owner, key, features)
            results[idx] = features

    return results


def shared_openai_clip_resource(
    owner: Any,
    model_name: str,
    *,
    device: Any,
) -> tuple[Any, Any]:
    """Return a shared OpenAI ``clip`` package model/preprocess pair."""

    def load_clip() -> tuple[Any, Any]:
        import clip

        model, preprocess = clip.load(model_name, device=str(device))
        model.eval()
        return model, preprocess

    return shared_runtime_resource(
        owner,
        ("openai_clip", model_name, str(device)),
        load_clip,
    )


def cached_openai_clip_text_features(
    owner: Any,
    model: Any,
    texts: Sequence[str],
    *,
    model_key: str,
    device: Any,
    cache_key: Optional[tuple[Any, ...]] = None,
    normalize: bool = True,
) -> Any:
    """Return cached OpenAI ``clip`` package text features."""

    text_values = tuple(str(text) for text in texts)
    key = (
        "clip_text_features",
        "openai_clip",
        model_key,
        str(device),
        bool(normalize),
        cache_key or text_values,
    )

    def compute() -> Any:
        import clip
        import torch

        tokens = clip.tokenize(list(text_values)).to(device)
        with torch_inference_context(
            str(device),
            _runtime_config(owner).get("dtype", "auto"),
            bool(_runtime_config(owner).get("amp_enabled", True)),
        ):
            features = model.encode_text(tokens)
            features = features.float()
            if normalize:
                features = features / features.norm(dim=-1, keepdim=True)
            return features

    return shared_runtime_value(owner, key, compute)


def openai_clip_image_features_cache_key(
    model_key: str,
    device: Any,
    cache_key: tuple[Any, ...],
    *,
    normalize: bool = True,
) -> tuple[Any, ...]:
    """Return the canonical runtime cache key for OpenAI CLIP image features."""

    return (
        "clip_image_features",
        "openai_clip",
        model_key,
        str(device),
        bool(normalize),
        cache_key,
    )


def cached_openai_clip_image_features(
    owner: Any,
    model: Any,
    preprocess: Any,
    images: Sequence[Any],
    *,
    model_key: str,
    device: Any,
    cache_key: tuple[Any, ...],
    normalize: bool = True,
) -> Any:
    """Return cached OpenAI ``clip`` package image features."""

    key = openai_clip_image_features_cache_key(
        model_key,
        device,
        cache_key,
        normalize=normalize,
    )

    def compute() -> Any:
        image_values = list(images)
        chunk_size = _max_clip_images_per_forward(owner)
        return _encode_openai_clip_image_chunks(
            owner,
            model,
            preprocess,
            image_values,
            device=device,
            normalize=normalize,
            chunk_size=chunk_size,
        )

    return shared_runtime_value(owner, key, compute)


def _encode_openai_clip_image_chunks(
    owner: Any,
    model: Any,
    preprocess: Any,
    images: Sequence[Any],
    *,
    device: Any,
    normalize: bool,
    chunk_size: int,
) -> Any:
    import torch

    image_values = list(images)
    if not image_values:
        return torch.empty((0, 0), device=device)

    chunk_size = max(1, int(chunk_size))
    outputs = []
    start = 0
    while start < len(image_values):
        chunk = image_values[start:start + chunk_size]
        try:
            inputs = torch.stack([preprocess(image) for image in chunk]).to(device)
            with torch_inference_context(
                str(device),
                _runtime_config(owner).get("dtype", "auto"),
                bool(_runtime_config(owner).get("amp_enabled", True)),
            ):
                features = model.encode_image(inputs).float()
                if normalize:
                    features = features / features.norm(dim=-1, keepdim=True)
                outputs.append(features)
            start += len(chunk)
        except RuntimeError as exc:
            if chunk_size <= 1 or not _is_cuda_oom(exc):
                raise
            _empty_cuda_cache(device)
            chunk_size = max(1, chunk_size // 2)

    if len(outputs) == 1:
        return outputs[0]
    return torch.cat(outputs, dim=0)


def cached_openai_clip_image_feature_groups(
    owner: Any,
    model: Any,
    preprocess: Any,
    image_groups: Sequence[Sequence[Any]],
    *,
    model_key: str,
    device: Any,
    cache_keys: Sequence[tuple[Any, ...]],
    normalize: bool = True,
) -> List[Any]:
    """Return OpenAI CLIP image features for multiple samples in one pass."""

    if len(image_groups) != len(cache_keys):
        raise ValueError("image_groups and cache_keys must have the same length")

    results: List[Any] = [None] * len(image_groups)
    missing_indices: List[int] = []
    flattened_images: List[Any] = []
    spans: List[tuple[int, int]] = []

    for idx, (images, cache_key) in enumerate(zip(image_groups, cache_keys)):
        key = openai_clip_image_features_cache_key(
            model_key,
            device,
            cache_key,
            normalize=normalize,
        )
        cached = cached_runtime_value(owner, key, _MISSING)
        if cached is not _MISSING:
            results[idx] = cached
            continue
        if not images:
            continue
        start = len(flattened_images)
        flattened_images.extend(images)
        spans.append((start, len(flattened_images)))
        missing_indices.append(idx)

    if flattened_images:
        batch_key = (
            "batch",
            tuple(cache_keys[idx] for idx in missing_indices),
        )
        batch_features = cached_openai_clip_image_features(
            owner,
            model,
            preprocess,
            flattened_images,
            model_key=model_key,
            device=device,
            cache_key=batch_key,
            normalize=normalize,
        )

        for idx, (start, end) in zip(missing_indices, spans):
            features = batch_features[start:end]
            key = openai_clip_image_features_cache_key(
                model_key,
                device,
                cache_keys[idx],
                normalize=normalize,
            )
            store_runtime_value(owner, key, features)
            results[idx] = features

    return results


def media_state_key(path: Any) -> tuple[str, Optional[int], Optional[int]]:
    """Return a stable cache key fragment for a media file's current state."""

    from pathlib import Path

    resolved = str(Path(path).resolve())
    try:
        stat = Path(path).stat()
        return resolved, stat.st_size, stat.st_mtime_ns
    except OSError:
        return resolved, None, None


def resolve_torch_device(device_config: str = "auto") -> str:
    """Resolve a torch device string without importing torch at module import time."""

    if device_config and device_config != "auto":
        return str(device_config)

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass

    return "cpu"


def resolve_torch_dtype(device: str, dtype_config: str = "auto") -> Any:
    """Resolve the configured torch dtype.

    Returns ``None`` for fp32/default behavior so callers can omit dtype kwargs.
    """

    try:
        import torch
    except Exception:
        return None

    normalized = str(dtype_config or "auto").lower()
    if normalized in ("fp32", "float32"):
        return torch.float32
    if normalized in ("fp16", "float16", "half"):
        return torch.float16
    if normalized in ("bf16", "bfloat16"):
        return torch.bfloat16
    if normalized in ("none", "default"):
        return None

    if str(device).startswith("cuda"):
        return torch.float16
    return None


@contextmanager
def torch_inference_context(
    device: str,
    dtype_config: str = "auto",
    amp_enabled: bool = True,
) -> Iterator[None]:
    """Return an inference context with autocast when it is safe and useful."""

    try:
        import torch
    except Exception:
        yield
        return

    dtype = resolve_torch_dtype(device, dtype_config)
    with torch.inference_mode():
        if amp_enabled and dtype in (torch.float16, torch.bfloat16) and str(device).startswith("cuda"):
            with torch.autocast("cuda", dtype=dtype):
                yield
        else:
            yield


def attention_kwargs(config: Dict[str, Any], device: str = "cpu") -> Dict[str, Any]:
    """Build safe Hugging Face ``from_pretrained`` attention kwargs."""

    backend = str(config.get("attention_backend", "auto") or "auto").lower()
    if backend in ("auto", "", "default", "none"):
        return {}
    if backend == "flash":
        backend = "flash_attention_2"
    if backend == "flash2":
        backend = "flash_attention_2"

    allowed = {"sdpa", "flash_attention_2", "flash_attention_3", "eager"}
    if backend not in allowed:
        return {}
    if backend.startswith("flash_attention") and not str(device).startswith("cuda"):
        return {}
    return {"attn_implementation": backend}


def from_pretrained_with_attention(
    model_cls: Any,
    model_id: str,
    config: Dict[str, Any],
    device: str = "cpu",
    **kwargs: Any,
) -> Any:
    """Load a Hugging Face model with optional attention backend fallback."""

    extra = attention_kwargs(config, device=device)
    if not extra:
        return model_cls.from_pretrained(model_id, **kwargs)
    try:
        return model_cls.from_pretrained(model_id, **kwargs, **extra)
    except Exception as exc:
        message = str(exc)
        lowered = message.lower()
        if (
            "attn_implementation" not in message
            and "attention" not in lowered
            and "flash" not in lowered
        ):
            raise
        return model_cls.from_pretrained(model_id, **kwargs)


@contextmanager
def maybe_sdpa_kernel(config: Dict[str, Any]) -> Iterator[None]:
    """Optionally force PyTorch SDPA to a selected backend for a small scope."""

    backend = str(config.get("attention_backend", "auto") or "auto").lower()
    if backend not in ("flash", "flash2", "flash_attention_2"):
        yield
        return

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            yield
    except Exception:
        yield
