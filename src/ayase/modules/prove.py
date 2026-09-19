"""PROVE removal coherence for masked image and video object removal.

Implements RC-S spatial coherence and RC-T temporal coherence from PROVE
(arXiv:2605.14534) with DINOv2-Giant patch features. RC-S is presented as a
higher-is-better score; RC-T is the raw lower-is-better discrepancy.

This is a behavior-level implementation of the Apache-2.0 reference algorithm
at xiaomi-research/prove commit 7ca299a7a5e12f0fb8285fb58ae744f691607b35.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Sequence

import cv2
import numpy as np

from ayase.image import is_video_path, load_image_array
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

_MODEL_ID = "facebook/dinov2-giant"
_MODEL_REVISION = "611a9d42f2335e0f921f1e313ad3c1b7178d206d"
_PATCH_SIZE = 14
_RGB_PAD_MEAN = (123, 116, 103)


@dataclass(frozen=True)
class _PatchGrid:
    """Patch features plus the geometry used to map a mask onto them."""

    features: np.ndarray
    resized_hw: tuple[int, int]
    padded_hw: tuple[int, int]


class _PatchEncoder(Protocol):
    def encode(self, image: np.ndarray) -> _PatchGrid: ...


class _DinoPatchEncoder:
    """Small adapter around a frozen Hugging Face DINOv2 model."""

    def __init__(self, model, processor, torch_module, device: str, target_size: int):
        self.model = model
        self.processor = processor
        self.torch = torch_module
        self.device = device
        self.target_size = target_size

    def encode(self, image: np.ndarray) -> _PatchGrid:
        padded, resized_hw = _resize_and_pad_rgb(image, self.target_size)
        inputs = self.processor(
            images=padded,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            output = self.model(**inputs)

        tokens = output.last_hidden_state[:, 1:, :]
        padded_hw = padded.shape[:2]
        feature_hw = (padded_hw[0] // _PATCH_SIZE, padded_hw[1] // _PATCH_SIZE)
        expected = feature_hw[0] * feature_hw[1]
        if int(tokens.shape[1]) != expected:
            raise RuntimeError(
                f"DINOv2 patch count mismatch: got {tokens.shape[1]}, expected {expected}"
            )
        features = (
            tokens[0]
            .reshape(feature_hw[0], feature_hw[1], int(tokens.shape[-1]))
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        return _PatchGrid(features, resized_hw, padded_hw)


def _resize_and_pad_rgb(
    image: np.ndarray, target_size: int = 448
) -> tuple[np.ndarray, tuple[int, int]]:
    """Resize the longest side and pad bottom/right to a DINO patch multiple."""

    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("PROVE expects a non-empty RGB image")
    height, width = image.shape[:2]
    scale = float(target_size) / max(height, width)
    resized_hw = (max(1, int(height * scale)), max(1, int(width * scale)))
    resized = cv2.resize(
        image, (resized_hw[1], resized_hw[0]), interpolation=cv2.INTER_LINEAR
    )
    pad_h = (-resized_hw[0]) % _PATCH_SIZE
    pad_w = (-resized_hw[1]) % _PATCH_SIZE
    padded = cv2.copyMakeBorder(
        resized,
        0,
        pad_h,
        0,
        pad_w,
        cv2.BORDER_CONSTANT,
        value=_RGB_PAD_MEAN,
    )
    return padded, resized_hw


def _to_gray_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a mask to one grayscale channel without binarising soft edges."""
    if mask.ndim == 3:
        if mask.shape[2] == 1:
            mask = mask[:, :, 0]
        else:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    if mask.ndim != 2:
        raise ValueError("PROVE masks must be 2-D or RGB arrays")
    return np.asarray(mask)


def _to_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Return the PROVE foreground definition: grayscale values strictly above 127."""

    mask = _to_gray_mask(mask)
    if mask.dtype == np.bool_ or (mask.size and float(np.max(mask)) <= 1.0):
        return np.asarray(mask > 0, dtype=np.uint8)
    return np.asarray(mask > 127, dtype=np.uint8)


def _expanded_square_bbox(binary_mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """Find the centered square mask box and expand it by one third when possible."""

    foreground = np.argwhere(binary_mask > 0)
    if foreground.size == 0:
        return None
    image_h, image_w = binary_mask.shape[:2]
    upper, left = foreground.min(axis=0)[:2]
    lower, right = foreground.max(axis=0)[:2] + 1
    width = int(right - left)
    height = int(lower - upper)
    size = max(width, height)
    center_x = (int(left) + int(right)) / 2.0
    center_y = (int(upper) + int(lower)) / 2.0
    start_x = int(round(center_x - size / 2.0))
    start_y = int(round(center_y - size / 2.0))

    expansion = int(round(size / 3.0))
    available = min(
        start_x,
        start_y,
        image_w - (start_x + size),
        image_h - (start_y + size),
        size // 2,
    )
    expansion = max(0, min(expansion, available))
    x1 = start_x - expansion
    y1 = start_y - expansion
    final_size = size + 2 * expansion

    max_size = min(image_w, image_h)
    if final_size > max_size:
        final_size = max_size
    x1 = max(0, min(x1, image_w - final_size))
    y1 = max(0, min(y1, image_h - final_size))
    return int(x1), int(y1), int(x1 + final_size), int(y1 + final_size)


def _component_boxes(mask: np.ndarray, minimum_area: int = 200) -> list[tuple[int, int, int, int]]:
    """Return PROVE crop boxes for four-connected foreground components."""

    from skimage.measure import label, regionprops

    binary = _to_binary_mask(mask)
    labelled = label(binary, connectivity=1)
    regions = regionprops(labelled)
    if not regions:
        return []
    threshold = min(minimum_area, max(int(region.area) for region in regions))
    boxes = []
    for region in regions:
        if int(region.area) < threshold:
            continue
        component = np.asarray(labelled == region.label, dtype=np.uint8)
        box = _expanded_square_bbox(component)
        if box is not None:
            boxes.append(box)
    return boxes


def _adaptive_max_pool(mask: np.ndarray, output_hw: tuple[int, int]) -> np.ndarray:
    """NumPy equivalent of two-dimensional adaptive max pooling."""

    input_h, input_w = mask.shape
    output_h, output_w = output_hw
    pooled = np.zeros((output_h, output_w), dtype=mask.dtype)
    for out_y in range(output_h):
        y1 = int(math.floor(out_y * input_h / output_h))
        y2 = int(math.ceil((out_y + 1) * input_h / output_h))
        for out_x in range(output_w):
            x1 = int(math.floor(out_x * input_w / output_w))
            x2 = int(math.ceil((out_x + 1) * input_w / output_w))
            pooled[out_y, out_x] = mask[y1:y2, x1:x2].max()
    return pooled


def _mask_to_feature_grid(mask: np.ndarray, encoded: _PatchGrid) -> np.ndarray:
    """Apply the encoder's resize/pad geometry, then max-pool onto its patch grid."""

    aligned_mask = _to_gray_mask(mask)
    resized = cv2.resize(
        aligned_mask,
        (encoded.resized_hw[1], encoded.resized_hw[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    pad_h = encoded.padded_hw[0] - encoded.resized_hw[0]
    pad_w = encoded.padded_hw[1] - encoded.resized_hw[1]
    if pad_h < 0 or pad_w < 0:
        raise ValueError("invalid PROVE encoder geometry")
    padded = np.pad(resized, ((0, pad_h), (0, pad_w)), mode="constant")
    return _adaptive_max_pool(padded, encoded.features.shape[:2]) > 0


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.float32(1e-12))


def _rbf_mmd(x: np.ndarray, y: np.ndarray, sigma: float = 10.0) -> float:
    """Biased RBF MMD over L2-normalized feature rows, scaled by 1000."""

    if len(x) == 0 or len(y) == 0:
        return float("nan")
    x_norm = _normalise_rows(x)
    y_norm = _normalise_rows(y)
    gamma = 1.0 / (2.0 * sigma * sigma)

    def kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        left_sq = np.sum(left * left, axis=1, keepdims=True)
        right_sq = np.sum(right * right, axis=1, keepdims=True).T
        distances = np.maximum(left_sq + right_sq - 2.0 * (left @ right.T), 0.0)
        return np.exp(-gamma * distances)

    value = kernel(x_norm, x_norm).mean()
    value += kernel(y_norm, y_norm).mean()
    value -= 2.0 * kernel(x_norm, y_norm).mean()
    return float(value * 1000.0)


def _scaled_window(feature_hw: tuple[int, int]) -> tuple[int, int]:
    scale = max(feature_hw) / 64.0
    return max(1, int(16 * scale)), max(1, int(8 * scale))


def _window_origins(height: int, width: int, kernel: int, stride: int):
    if kernel > height or kernel > width:
        return
    for top in range(0, height - kernel + 1, stride):
        for left in range(0, width - kernel + 1, stride):
            yield top, left


def _rc_s_from_grid(encoded: _PatchGrid, mask: np.ndarray) -> Optional[float]:
    """Compute one component's RC-S presentation score from encoded patches."""

    features = encoded.features
    mask_grid = _mask_to_feature_grid(mask, encoded)
    height, width, channels = features.shape
    kernel, stride = _scaled_window((height, width))
    global_background = features.reshape(-1, channels)[~mask_grid.reshape(-1)]
    raw_scores: list[float] = []

    for top, left in _window_origins(height, width, kernel, stride):
        window_mask = mask_grid[top : top + kernel, left : left + kernel].reshape(-1)
        foreground_count = int(window_mask.sum())
        background_count = int((~window_mask).sum())
        window_features = features[top : top + kernel, left : left + kernel].reshape(
            -1, channels
        )
        if foreground_count > 5 and background_count > 5:
            raw_scores.append(
                _rbf_mmd(window_features[window_mask], window_features[~window_mask])
            )
        elif foreground_count > 5 and background_count == 0 and len(global_background) > 0:
            raw_scores.append(_rbf_mmd(window_features[window_mask], global_background))

    if not raw_scores:
        return None
    raw = max(0.0, float(np.mean(raw_scores)))
    return math.exp(-raw / 3.0)


def _rc_t_from_grids(
    encoded_t: _PatchGrid,
    encoded_t1: _PatchGrid,
    mask_t: np.ndarray,
    mask_t1: np.ndarray,
) -> Optional[float]:
    """Compute one adjacent-frame RC-T raw discrepancy from encoded patches."""

    if encoded_t.features.shape != encoded_t1.features.shape:
        return None
    features_t = encoded_t.features
    features_t1 = encoded_t1.features
    height, width, channels = features_t.shape
    intersection = _mask_to_feature_grid(mask_t, encoded_t) & _mask_to_feature_grid(
        mask_t1, encoded_t1
    )
    kernel, stride = _scaled_window((height, width))
    raw_scores: list[float] = []

    for top, left in _window_origins(height, width, kernel, stride):
        selected = intersection[top : top + kernel, left : left + kernel].reshape(-1)
        if int(selected.sum()) < 5:
            continue
        patch_t = features_t[top : top + kernel, left : left + kernel].reshape(
            -1, channels
        )
        patch_t1 = features_t1[top : top + kernel, left : left + kernel].reshape(
            -1, channels
        )
        raw_scores.append(_rbf_mmd(patch_t[selected], patch_t1[selected]))

    return float(np.mean(raw_scores)) if raw_scores else None


class PROVEModule(PipelineModule):
    """Compute PROVE RC-S and RC-T from generated media plus a removal mask."""

    name = "prove"
    description = "PROVE masked object-removal spatial and temporal coherence"
    default_config = {
        "model": _MODEL_ID,
        "revision": _MODEL_REVISION,
        "target_size": 448,
        "max_frames": 81,
        "mask_path": None,
        "reference_mask_path": None,
        "device": "auto",
        "models_dir": "models",
    }
    models = [
        {
            "id": _MODEL_ID,
            "type": "huggingface",
            "task": "DINOv2-Giant patch features for PROVE RC-S and RC-T",
            "auto_download": True,
            "revision": _MODEL_REVISION,
            "url": "https://huggingface.co/facebook/dinov2-giant",
            "size": "4.55 GB",
            "vram": "~4.5 GB",
            "notes": f"Apache-2.0; pinned to revision {_MODEL_REVISION}",
        }
    ]
    metric_info = {
        "prove_rc_s_score": (
            "PROVE RC-S masked spatial removal coherence (0-1, higher=better)"
        ),
        "prove_rc_t_score": (
            "PROVE RC-T adjacent-frame masked temporal discrepancy (lower=better)"
        ),
    }
    metric_groups = {
        "prove_rc_s_score": "nr_quality",
        "prove_rc_t_score": "temporal",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.model_id = str(self.config.get("model", _MODEL_ID))
        self.revision = str(self.config.get("revision", _MODEL_REVISION))
        self.target_size = int(self.config.get("target_size", 448))
        self.max_frames = max(1, int(self.config.get("max_frames", 81)))
        self.mask_path = self.config.get("mask_path") or self.config.get(
            "reference_mask_path"
        )
        self.models_dir = str(self.config.get("models_dir", "models"))
        self._device = "cpu"
        self._model = None
        self._processor = None
        self._encoder: Optional[_PatchEncoder] = None
        self._backend = None

    def setup(self) -> None:
        if self.test_mode:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel

            from ayase.config import resolve_model_path
            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            resolved = resolve_model_path(self.model_id, self.models_dir)
            load_kwargs = {
                "revision": self.revision,
                "cache_dir": str(Path(self.models_dir) / "huggingface"),
            }
            self._processor = AutoImageProcessor.from_pretrained(resolved, **load_kwargs)
            self._model = AutoModel.from_pretrained(resolved, **load_kwargs)
            self._model.eval().to(self._device)
            self._encoder = _DinoPatchEncoder(
                self._model,
                self._processor,
                torch,
                self._device,
                self.target_size,
            )
            self._backend = "dinov2_giant"
        except Exception as exc:
            self._model = None
            self._processor = None
            self._encoder = None
            logger.warning("PROVE setup failed: %s", exc)

    def teardown(self) -> None:
        self._encoder = None
        self._processor = None

    def process(self, sample: Sample) -> Sample:
        if self._encoder is None:
            return sample
        mask_path = self._resolve_mask_path(sample)
        if mask_path is None:
            return sample
        try:
            if sample.is_video:
                rc_s, rc_t = self._score_video(Path(sample.path), mask_path)
            else:
                rc_s = self._score_image(Path(sample.path), mask_path)
                rc_t = None

            if rc_s is None and rc_t is None:
                return sample
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            if rc_s is not None and np.isfinite(rc_s):
                sample.quality_metrics.prove_rc_s_score = float(rc_s)
                sample.quality_metrics.metric_backends["prove_rc_s_score"] = self._backend
            if rc_t is not None and np.isfinite(rc_t):
                sample.quality_metrics.prove_rc_t_score = float(rc_t)
                sample.quality_metrics.metric_backends["prove_rc_t_score"] = self._backend
        except Exception as exc:
            logger.warning("PROVE failed for %s: %s", sample.path, exc)
        return sample

    def _score_image(self, image_path: Path, mask_path: Path) -> Optional[float]:
        image = load_image_array(image_path, color="rgb")
        mask = load_image_array(mask_path, color="gray")
        if image is None or mask is None:
            return None
        image = _resize_frame_to_mask(image, mask)
        return self._score_spatial_frame(image, mask)

    def _score_video(
        self, video_path: Path, mask_path: Path
    ) -> tuple[Optional[float], Optional[float]]:
        frames = self._load_video_frames(video_path, color="rgb")
        if not frames:
            return None, None
        if is_video_path(mask_path):
            masks = self._load_video_frames(mask_path, color="gray")
        else:
            static_mask = load_image_array(mask_path, color="gray")
            masks = [static_mask] * len(frames) if static_mask is not None else []
        valid_length = min(len(frames), len(masks))
        if valid_length == 0:
            return None, None

        frames = frames[:valid_length]
        masks = masks[:valid_length]
        aligned_frames = [
            _resize_frame_to_mask(frame, mask) for frame, mask in zip(frames, masks)
        ]
        spatial_scores = [
            score
            for frame, mask in zip(aligned_frames, masks)
            if (score := self._score_spatial_frame(frame, mask)) is not None
        ]
        temporal_scores = [
            score
            for index in range(valid_length - 1)
            if (
                score := self._score_temporal_pair(
                    aligned_frames[index],
                    aligned_frames[index + 1],
                    masks[index],
                    masks[index + 1],
                )
            )
            is not None
        ]
        rc_s = float(np.mean(spatial_scores)) if spatial_scores else None
        rc_t = float(np.mean(temporal_scores)) if temporal_scores else None
        return rc_s, rc_t

    def _score_spatial_frame(self, image: np.ndarray, mask: np.ndarray) -> Optional[float]:
        grayscale = _to_gray_mask(mask)
        binary = _to_binary_mask(mask)
        scores: list[float] = []
        for x1, y1, x2, y2 in _component_boxes(binary):
            cropped_image = image[y1:y2, x1:x2]
            # Upstream uses the binary mask only to discover components, but
            # preserves nonzero antialiased mask pixels for feature-grid pooling.
            cropped_mask = grayscale[y1:y2, x1:x2]
            if cropped_image.size == 0:
                continue
            score = _rc_s_from_grid(self._encoder.encode(cropped_image), cropped_mask)
            if score is not None and np.isfinite(score):
                scores.append(score)
        return float(np.mean(scores)) if scores else None

    def _score_temporal_pair(
        self,
        frame_t: np.ndarray,
        frame_t1: np.ndarray,
        mask_t: np.ndarray,
        mask_t1: np.ndarray,
    ) -> Optional[float]:
        binary_t = _to_binary_mask(mask_t)
        binary_t1 = _to_binary_mask(mask_t1)
        box = _expanded_square_bbox(binary_t | binary_t1)
        if box is None:
            return None
        x1, y1, x2, y2 = box
        crop_t = frame_t[y1:y2, x1:x2]
        crop_t1 = frame_t1[y1:y2, x1:x2]
        crop_mask_t = binary_t[y1:y2, x1:x2]
        crop_mask_t1 = binary_t1[y1:y2, x1:x2]
        if crop_t.size == 0 or crop_t1.size == 0:
            return None
        encoded_t = self._encoder.encode(crop_t)
        encoded_t1 = self._encoder.encode(crop_t1)
        return _rc_t_from_grids(encoded_t, encoded_t1, crop_mask_t, crop_mask_t1)

    def _load_video_frames(self, path: Path, color: str) -> list[np.ndarray]:
        """Read the first frames in order, matching PROVE's default chronology."""

        capture = cv2.VideoCapture(str(path))
        frames: list[np.ndarray] = []
        try:
            if not capture.isOpened():
                return frames
            while len(frames) < self.max_frames:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                if color == "rgb":
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                elif color == "gray":
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(frame)
        finally:
            capture.release()
        return frames

    def _resolve_mask_path(self, sample: Sample) -> Optional[Path]:
        explicit = sample.reference_mask_path or self.mask_path
        if explicit:
            path = Path(explicit)
            if path.is_dir():
                candidates = [
                    path / Path(sample.path).name,
                    path / f"{Path(sample.path).stem}.png",
                    path / f"{Path(sample.path).stem}.mask.png",
                ]
                return next((candidate for candidate in candidates if candidate.is_file()), None)
            return path if path.is_file() else None

        media = Path(sample.path)
        candidates: Sequence[Path] = (
            media.with_name(f"{media.stem}.mask{media.suffix}"),
            media.with_name(f"{media.stem}.mask.png"),
            media.with_name(f"{media.stem}_mask{media.suffix}"),
            media.with_name(f"{media.stem}_mask.png"),
        )
        return next((candidate for candidate in candidates if candidate.is_file()), None)


def _resize_frame_to_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if frame.shape[:2] == mask.shape[:2]:
        return frame
    return cv2.resize(frame, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
