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
                   sampling_rate: int = SAMPLING_RATE) -> List[int]:
    """Pick ``num_frames`` frame indices from ``num_available`` frames.

    Uses a stride of ``sampling_rate`` centered in the clip when the clip is
    long enough; otherwise samples uniformly (with clamping) across whatever is
    available so short clips still yield a full window.
    """
    if num_available <= 0:
        raise ValueError("No frames supplied to VideoMAEv2 preprocessing.")

    span = (num_frames - 1) * sampling_rate + 1
    if num_available >= span:
        start = (num_available - span) // 2
        return [start + i * sampling_rate for i in range(num_frames)]

    # Short clip: spread indices uniformly across the available frames.
    idx = np.linspace(0, num_available - 1, num=num_frames)
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


def _center_crop(frame: np.ndarray, size: int) -> np.ndarray:
    h, w = frame.shape[:2]
    top = max((h - size) // 2, 0)
    left = max((w - size) // 2, 0)
    return frame[top:top + size, left:left + size]


def preprocess_frames(frames_rgb_list: Sequence[np.ndarray],
                      num_frames: int = NUM_FRAMES,
                      sampling_rate: int = SAMPLING_RATE,
                      input_size: int = INPUT_SIZE,
                      short_side_size: int = SHORT_SIDE_SIZE) -> torch.Tensor:
    """Turn a list of RGB frames into a ``[1, C, T, H, W]`` float tensor.

    Arguments:
        frames_rgb_list: Sequence of HxWxC uint8 RGB frames (a decoded clip).
        num_frames: Temporal length of the model input (default 16).
        sampling_rate: Temporal stride used when sampling frames (default 4).
        input_size: Spatial crop size fed to the model (default 224).
        short_side_size: Short-side resize target before cropping (default 224).

    Returns:
        A ``torch.FloatTensor`` of shape ``[1, 3, num_frames, input_size,
        input_size]`` normalized with ImageNet statistics.
    """
    frames = [_to_rgb_uint8(f) for f in frames_rgb_list]
    indices = sample_indices(len(frames), num_frames, sampling_rate)

    processed = []
    for i in indices:
        f = frames[i]
        f = _resize_short_side(f, short_side_size)
        f = _center_crop(f, input_size)
        processed.append(f)

    # [T, H, W, C] uint8 -> float [0, 1]
    clip = np.stack(processed, axis=0).astype(np.float32) / 255.0

    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 1, 3)
    clip = (clip - mean) / std

    # [T, H, W, C] -> [C, T, H, W] -> [1, C, T, H, W]
    tensor = torch.from_numpy(clip).permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    return tensor
