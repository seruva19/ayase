"""Inference-time preprocessing for the VideoMAEv2 finetune ViT.

Mirrors the VMBench / VideoMAEv2 test-time transform for a single view:

    Resize(short_side=224, bilinear) -> CenterCrop(224) -> scale to [0, 1]
    -> Normalize(ImageNet mean/std)

Frames are sampled to a fixed clip of ``num_frames`` (16) with a temporal
stride of ``sampling_rate`` (4); when fewer/more frames are supplied they are
sampled uniformly across the available range. The output tensor is laid out as
``[1, C, T, H, W]`` to match ``VisionTransformer.forward``.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch

# VideoMAEv2 finetune datasets normalize with the ImageNet statistics
# (imagenet_default_mean_and_std=True).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

NUM_FRAMES = 16
SAMPLING_RATE = 4
INPUT_SIZE = 224
SHORT_SIDE_SIZE = 224


def _to_rgb_uint8(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def sample_indices(num_available: int,
                   num_frames: int = NUM_FRAMES,
                   sampling_rate: int = SAMPLING_RATE,
                   segment_index: int = 0,
                   num_segments: int = 1) -> List[int]:
    """Pick ``num_frames`` frame indices from ``num_available`` frames.

    Uses a stride of ``sampling_rate``. When ``num_segments`` > 1 the available
    range is split into that many temporal segments and the stride-window is
    centered in segment ``segment_index`` (the VMBench multi-view protocol);
    otherwise the window is centered in the whole clip. Short clips fall back
    to uniform samples across the segment range.
    """
    if num_available <= 0:
        raise ValueError("No frames supplied to VideoMAEv2 preprocessing.")

    num_segments = max(int(num_segments), 1)
    segment_index = min(max(int(segment_index), 0), num_segments - 1)
    lo = int(num_available * segment_index / num_segments)
    hi = int(num_available * (segment_index + 1) / num_segments)
    seg_available = max(hi - lo, 1)

    span = (num_frames - 1) * sampling_rate + 1
    if seg_available >= span:
        start = lo + (seg_available - span) // 2
        return [start + i * sampling_rate for i in range(num_frames)]

    # Short segment: spread indices uniformly across its range.
    idx = np.linspace(lo, hi - 1, num=num_frames)
    return [int(round(x)) for x in idx]


def _resize_short_side(frame: np.ndarray, short_side: int) -> np.ndarray:
    import cv2

    h, w = frame.shape[:2]
    if h <= w:
        new_h = short_side
        new_w = int(round(w * short_side / h))
    else:
        new_w = short_side
        new_h = int(round(h * short_side / w))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def _crop(frame: np.ndarray, size: int, crop_id: int = 0) -> np.ndarray:
    """Three-crop spatial view: 0/1/2 = start/center/end along the long axis."""
    h, w = frame.shape[:2]
    crop_id = int(crop_id) % 3
    if w >= h:
        offsets = [0, max((w - size) // 2, 0), max(w - size, 0)]
        left = min(offsets[crop_id], max(w - size, 0))
        top = max((h - size) // 2, 0)
    else:
        offsets = [0, max((h - size) // 2, 0), max(h - size, 0)]
        top = min(offsets[crop_id], max(h - size, 0))
        left = max((w - size) // 2, 0)
    return frame[top:top + size, left:left + size]


def preprocess_frames(frames_rgb_list: Sequence[np.ndarray],
                      num_frames: int = NUM_FRAMES,
                      sampling_rate: int = SAMPLING_RATE,
                      input_size: int = INPUT_SIZE,
                      short_side_size: int = SHORT_SIDE_SIZE,
                      segment_index: int = 0,
                      num_segments: int = 1,
                      crop_id: int = 0) -> torch.Tensor:
    """Turn a list of RGB frames into a ``[1, C, T, H, W]`` float tensor.

    Arguments:
        frames_rgb_list: Sequence of HxWxC uint8 RGB frames (a decoded clip).
        num_frames: Temporal length of the model input (default 16).
        sampling_rate: Temporal stride used when sampling frames (default 4).
        input_size: Spatial crop size fed to the model (default 224).
        short_side_size: Short-side resize target before cropping (default 224).
        segment_index/num_segments: temporal-view split (VMBench uses 10).
        crop_id: spatial crop view, 0/1/2 = start/center/end (VMBench uses 3).

    Returns:
        A ``torch.FloatTensor`` of shape ``[1, 3, num_frames, input_size,
        input_size]`` normalized with ImageNet statistics.
    """
    frames = [_to_rgb_uint8(f) for f in frames_rgb_list]
    indices = sample_indices(len(frames), num_frames, sampling_rate,
                             segment_index, num_segments)

    processed = []
    for i in indices:
        f = frames[i]
        f = _resize_short_side(f, short_side_size)
        f = _crop(f, input_size, crop_id)
        processed.append(f)

    # [T, H, W, C] uint8 -> float [0, 1]
    clip = np.stack(processed, axis=0).astype(np.float32) / 255.0

    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 1, 3)
    clip = (clip - mean) / std

    # [T, H, W, C] -> [C, T, H, W] -> [1, C, T, H, W]
    tensor = torch.from_numpy(clip).permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    return tensor
