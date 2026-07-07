"""Image utility functions for Ayase.

Provides small loading and conversion helpers used by image-level metrics and
downstream adapters that need a public image API without constructing a full
pipeline.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}


def is_image_path(path: Path) -> bool:
    """Return True when *path* has a supported still-image extension."""
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def is_video_path(path: Path) -> bool:
    """Return True when *path* has a supported video extension."""
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def load_pil_image(path: Path, mode: str = "RGB") -> Optional[Image.Image]:
    """Load a still image as a PIL image."""
    try:
        with Image.open(path) as img:
            return img.convert(mode)
    except Exception as e:
        logger.debug("Failed to load image %s: %s", path, e)
        return None


def load_image_array(path: Path, color: str = "rgb") -> Optional[np.ndarray]:
    """Load a still image into a numpy array.

    ``color`` can be ``"rgb"``, ``"bgr"``, or ``"gray"``.
    """
    try:
        flag = cv2.IMREAD_GRAYSCALE if color == "gray" else cv2.IMREAD_COLOR
        arr = cv2.imread(str(path), flag)
        if arr is None:
            return None
        if color == "rgb" and arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return arr
    except Exception as e:
        logger.debug("Failed to load image array %s: %s", path, e)
        return None


def image_from_bytes(data: bytes, mode: str = "RGB") -> Optional[Image.Image]:
    """Decode image bytes into a PIL image."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            return img.convert(mode)
    except Exception as e:
        logger.debug("Failed to decode image bytes: %s", e)
        return None


def load_representative_frame(path: Path, color: str = "rgb") -> Optional[np.ndarray]:
    """Load a still image or the middle frame of a video into a numpy array.

    ``color`` is ``"rgb"`` (default), ``"bgr"``, or ``"gray"``. The returned
    array is ALWAYS a READ-ONLY numpy array (``writeable=False``) — both when a
    pipeline is active (a zero-copy view over the shared per-sample frame cache)
    and on the uncached fallback path. Do not mutate it in place; call
    ``frame.copy()`` first if you need a writable array. Returns ``None`` when
    the frame cannot be decoded.
    """
    try:
        from .runtime import current_pipeline

        pipeline = current_pipeline()
        if pipeline is not None and hasattr(pipeline, "load_representative_frame"):
            return pipeline.load_representative_frame(path, color=color)
    except Exception:
        pass
    # Uncached fallback: mark read-only so the contract is identical to the
    # cached path — a module mutating a returned frame fails in tests too,
    # rather than only in production under a live pipeline.
    from .runtime import readonly_view

    frame = _load_representative_frame_uncached(path, color=color)
    return readonly_view(frame) if frame is not None else None


def _load_representative_frame_uncached(path: Path, color: str = "rgb") -> Optional[np.ndarray]:
    """Uncached implementation for loading a representative frame."""
    path = Path(path)
    if is_image_path(path):
        return load_image_array(path, color=color)
    if not is_video_path(path):
        return load_image_array(path, color=color)

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        if color == "gray":
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if color == "rgb":
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    except Exception as e:
        logger.debug("Failed to read representative frame %s: %s", path, e)
        return None
    finally:
        cap.release()


def sample_frames(path: Path, max_frames: int = 8, color: str = "rgb") -> List[np.ndarray]:
    """Load uniformly spaced frames from a video, or one frame for an image.

    ``max_frames`` caps the number of frames; ``color`` is ``"rgb"`` (default),
    ``"bgr"``, or ``"gray"``. The returned arrays are ALWAYS READ-ONLY
    (``writeable=False``) — both when a pipeline is active (zero-copy views over
    the shared per-sample cache, decoded once per file at the highest
    ``max_frames`` requested) and on the uncached fallback path. The returned
    list is fresh each call, but the pixel buffers may be shared. Do NOT mutate
    returned frames in place — copy first (``frame.copy()``) if you need to
    write. Returns an empty list when no frames can be decoded.
    """
    try:
        from .runtime import current_pipeline

        pipeline = current_pipeline()
        if pipeline is not None and hasattr(pipeline, "sample_frames"):
            return pipeline.sample_frames(path, max_frames=max_frames, color=color)
    except Exception:
        pass
    # Uncached fallback: return read-only views so the contract matches the
    # cached path (a violating in-place mutation fails in tests, not only in
    # production under a live pipeline).
    from .runtime import clone_frames

    return clone_frames(_sample_frames_uncached(path, max_frames=max_frames, color=color))


def _convert_frame_color(frame: np.ndarray, color: str) -> np.ndarray:
    """Convert a decoded BGR frame to the requested color space."""
    if color == "gray":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if color == "rgb":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame  # "bgr" (native) or any passthrough


# When CAP_PROP_FRAME_COUNT is unreliable, scan at most this many frames per
# kept frame so a broken container still yields spread-out samples with bounded
# work/memory.
_SEQUENTIAL_FALLBACK_STRIDE = 4


def _sample_frames_sequential(
    cap: "cv2.VideoCapture", max_frames: int, color: str
) -> List[np.ndarray]:
    """Fallback sampler for videos whose frame count is unknown (<=0).

    Some webm / pipe-muxed files report ``CAP_PROP_FRAME_COUNT <= 0`` so
    seek-based sampling returns nothing. Read sequentially instead, keeping
    every ``stride``-th frame up to ``max_frames`` and scanning at most
    ``max_frames * stride`` frames so both memory and time stay bounded.
    """
    frames: List[np.ndarray] = []
    if max_frames <= 0:
        return frames
    stride = _SEQUENTIAL_FALLBACK_STRIDE
    scan_cap = max_frames * stride
    read_idx = 0
    while len(frames) < max_frames and read_idx < scan_cap:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if read_idx % stride == 0:
            frames.append(_convert_frame_color(frame, color))
        read_idx += 1
    return frames


def _sample_frames_uncached_detailed(
    path: Path, max_frames: int = 8, color: str = "rgb"
) -> tuple[List[np.ndarray], Optional[int]]:
    """Uncached frame sampling that also reports the decoder's frame count.

    Returns ``(frames, reliable_total)`` where ``reliable_total`` is the
    decoder-reported total frame count (``CAP_PROP_FRAME_COUNT``) when it is
    trustworthy (``> 0``), or ``None`` when it is unreliable/unknown (frame
    count ``<= 0``, an unopenable file, or a decode error).

    Callers use ``reliable_total`` to tell a source that is *truly exhausted*
    (the decoder says the video has no more frames than were requested) apart
    from a *short read caused by failed seeks* (the video is long, but some
    seek/read attempts were skipped). The number of frames actually returned
    can be smaller than ``min(max_frames, reliable_total)`` when seeks fail, so
    it must NOT be used to infer exhaustion.
    """
    path = Path(path)
    if not is_video_path(path):
        frame = _load_representative_frame_uncached(path, color=color)
        frames = [frame] if frame is not None else []
        # A still image / non-video is a single-frame source: its total is
        # known exactly (the number of frames we could decode).
        return frames, len(frames)

    cap = cv2.VideoCapture(str(path))
    frames: List[np.ndarray] = []
    try:
        if not cap.isOpened():
            return frames, None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            # Frame count unreliable: fall back to a bounded sequential read so
            # frame metrics still get samples instead of silently skipping.
            # Report None so callers never cache an authoritative "exhausted".
            return _sample_frames_sequential(cap, max_frames, color), None
        n = min(max_frames, total)
        for idx in np.linspace(0, total - 1, n, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frames.append(_convert_frame_color(frame, color))
        return frames, total
    except Exception as e:
        logger.debug("Failed to sample frames from %s: %s", path, e)
        return frames, None
    finally:
        cap.release()


def _sample_frames_uncached(path: Path, max_frames: int = 8, color: str = "rgb") -> List[np.ndarray]:
    """Uncached implementation for uniformly spaced frame sampling."""
    return _sample_frames_uncached_detailed(path, max_frames=max_frames, color=color)[0]


def arrays_to_pil(frames: Sequence[np.ndarray]) -> List[Image.Image]:
    """Convert RGB/grayscale arrays to PIL images."""
    images: List[Image.Image] = []
    for frame in frames:
        if frame.ndim == 2:
            images.append(Image.fromarray(frame, mode="L").convert("RGB"))
        else:
            images.append(Image.fromarray(frame.astype(np.uint8)).convert("RGB"))
    return images


def image_stats_features(frame: np.ndarray, bins: int = 16) -> np.ndarray:
    """Return compact deterministic image features for proxy distribution metrics."""
    if frame is None:
        return np.zeros(bins * 3 + 8, dtype=np.float64)
    if frame.ndim == 2:
        rgb = np.stack([frame, frame, frame], axis=-1)
    else:
        rgb = frame[:, :, :3]
    rgb = rgb.astype(np.float32) / 255.0
    h, w = rgb.shape[:2]
    hist_parts = []
    for c in range(3):
        hist, _ = np.histogram(rgb[:, :, c], bins=bins, range=(0.0, 1.0), density=True)
        hist_parts.append(hist.astype(np.float64))
    gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    stats = np.array([
        float(rgb.mean()),
        float(rgb.std()),
        float(gray.mean() / 255.0),
        float(gray.std() / 255.0),
        float(grad.mean() / 255.0),
        float(grad.std() / 255.0),
        float(w / max(h, 1)),
        float(min(h, w) / max(max(h, w), 1)),
    ], dtype=np.float64)
    return np.concatenate(hist_parts + [stats])


def batch_existing(paths: Iterable[Path]) -> List[Path]:
    """Return existing image/video paths from an iterable."""
    return [Path(p) for p in paths if Path(p).exists()]
