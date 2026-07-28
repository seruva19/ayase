"""RQ-VQA rich quality-aware blind video quality assessment.

Implements the published RQ-VQA ensemble (Sun et al., CVPRW/NTIRE 2024)
using its five real feature streams: trainable Swin-B + BoT spatial features,
SlowFast motion features, Q-Align decoder features, LIQE quality/content
features, and FAST-VQA spatiotemporal features. The released regressors use a
PLCC correlation loss, so their raw output is intentionally unbounded and is
not a calibrated 1-5 MOS. Higher scores indicate better predicted quality.

The upstream single-video test script constructs the two-input base model but
passes all five RQ-VQA features. This module reconstructs the full published
architecture and never substitutes heuristic or proxy scores.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ayase.config import download_model_file
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

_HF_REPO = "AkaneTendo25/ayase-runtime-assets"
_HF_ROOT = "https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main"

# The best published fold comes first, so ensemble_size=1 remains a genuine,
# useful RQ-VQA model. With the default ensemble_size=10, ordering is immaterial.
_CHECKPOINTS = (
    "Swin_b_384_in22k_SlowFast_Fast_LLM_LIQE_FASTVQA_BoT_NTIREVideo_plcc_NR_v2_epoch_21_SRCC_0.924423.pth",
    "Swin_b_384_in22k_SlowFast_Fast_LLM_LIQE_FASTVQA_BoT_NTIREVideo_plcc_NR_v1_epoch_19_SRCC_0.923127.pth",
    "Swin_b_384_in22k_SlowFast_Fast_LLM_LIQE_FASTVQA_BoT_NTIREVideo_plcc_NR_v2_epoch_25_SRCC_0.913571.pth",
    "Swin_b_384_in22k_SlowFast_Fast_LLM_LIQE_FASTVQA_BoT_NTIREVideo_plcc_NR_v4_epoch_6_SRCC_0.905095.pth",
    "Swin_b_384_in22k_SlowFast_Fast_LLM_LIQE_FASTVQA_BoT_NTIREVideo_plcc_NR_v4_epoch_14_SRCC_0.904949.pth",
    "Swin_b_384_in22k_SlowFast_Fast_LLM_LIQE_FASTVQA_BoT_NTIREVideo_plcc_NR_v3_epoch_16_SRCC_0.901800.pth",
    "Swin_b_384_in22k_SlowFast_Fast_LLM_LIQE_FASTVQA_BoT_NTIREVideo_plcc_NR_v3_epoch_8_SRCC_0.896798.pth",
    "Swin_b_384_in22k_SlowFast_Fast_LLM_LIQE_FASTVQA_BoT_NTIREVideo_plcc_NR_v1_epoch_22_SRCC_0.894115.pth",
    "Swin_b_384_in22k_SlowFast_Fast_LLM_LIQE_FASTVQA_BoT_NTIREVideo_plcc_NR_v0_epoch_4_SRCC_0.905999.pth",
    "Swin_b_384_in22k_SlowFast_Fast_LLM_LIQE_FASTVQA_BoT_NTIREVideo_plcc_NR_v0_epoch_9_SRCC_0.885692.pth",
)

_LIQE_FILE = "LIQE.pt"
_SLOWFAST_FILE = "SLOWFAST_8x8_R50.pyth"

_SPATIAL_MEAN = (0.485, 0.456, 0.406)
_SPATIAL_STD = (0.229, 0.224, 0.225)
_MOTION_MEAN = (0.45, 0.45, 0.45)
_MOTION_STD = (0.225, 0.225, 0.225)

_LIQE_DISTORTIONS = (
    "jpeg2000 compression", "jpeg compression", "noise", "blur", "color",
    "contrast", "overexposure", "underexposure", "spatial", "quantization", "other",
)
_LIQE_SCENES = (
    "animal", "cityscape", "human", "indoor", "landscape", "night", "plant",
    "still_life", "others",
)
_LIQE_QUALITIES = ("bad", "poor", "fair", "good", "perfect")


def _remap_swin_state_dict(state_dict: dict) -> dict:
    """Translate the released old-timm Swin layout to current timm."""
    remapped = {}
    for key, value in state_dict.items():
        match = re.match(r"layers\.(\d+)\.downsample\.(.*)", key)
        if match:
            key = f"layers.{int(match.group(1)) + 1}.downsample.{match.group(2)}"
        if key.endswith("relative_position_index") or key.endswith("attn_mask"):
            continue
        remapped[key] = value
    return remapped


def _resolve_dtype(device: str, dtype_name: str):
    import torch

    if not str(device).startswith("cuda"):
        return torch.float32
    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(dtype_name.lower(), torch.float16)


def _build_rqvqa_model(checkpoint_path: Path, device: str, dtype):
    """Build one released RQ-VQA fold with state-compatible clean modules."""
    import timm
    import torch
    import torch.nn as nn

    def rel_to_abs(x):
        batch, heads, length, _ = x.shape
        col_pad = x.new_zeros((batch, heads, length, 1))
        x = torch.cat((x, col_pad), dim=3)
        flat_x = x.reshape(batch, heads, length * 2 * length)
        flat_pad = x.new_zeros((batch, heads, length - 1))
        flat_x = torch.cat((flat_x, flat_pad), dim=2)
        final_x = flat_x.reshape(batch, heads, length + 1, 2 * length - 1)
        return final_x[:, :, :length, length - 1:]

    def relative_logits_1d(q, rel_k):
        batch, heads, height, width, _ = q.shape
        logits = torch.einsum("bnhwd,md->bnhwm", q, rel_k)
        logits = logits.reshape(-1, heads * height, width, 2 * width - 1)
        logits = rel_to_abs(logits)
        logits = logits.reshape(-1, heads, height, width, width)
        logits = logits.unsqueeze(3).expand(-1, -1, -1, height, -1, -1)
        return logits

    class RelativePositionEmbedding(nn.Module):
        def __init__(self, height: int, width: int, dim_head: int) -> None:
            super().__init__()
            scale = dim_head ** -0.5
            self.height = height
            self.width = width
            self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
            self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

        def forward(self, q):
            batch, heads, _, dim = q.shape
            q_w = q.reshape(batch, heads, self.height, self.width, dim)
            logits_w = relative_logits_1d(q_w, self.rel_width)
            logits_w = logits_w.permute(0, 1, 2, 4, 3, 5).reshape(
                batch, heads, self.height * self.width, self.height * self.width
            )

            q_h = q_w.permute(0, 1, 3, 2, 4)
            logits_h = relative_logits_1d(q_h, self.rel_height)
            logits_h = logits_h.permute(0, 1, 4, 2, 5, 3).reshape(
                batch, heads, self.height * self.width, self.height * self.width
            )
            return logits_w + logits_h

    class MultiHeadSpatialAttention(nn.Module):
        def __init__(self, dim: int, fmap_size: Tuple[int, int]) -> None:
            super().__init__()
            self.scale = 128 ** -0.5
            self.heads = 4
            self.to_qk = nn.Conv2d(dim, 4 * 128 * 2, 1, bias=False)
            self.to_v = nn.Conv2d(dim, 4 * 128, 1, bias=False)
            self.softmax = nn.Softmax(dim=-1)
            self.pos_emb = RelativePositionEmbedding(fmap_size[0], fmap_size[1], 128)

        def forward(self, featuremap):
            batch, _, height, width = featuremap.shape
            q, k = self.to_qk(featuremap).chunk(2, dim=1)
            v = self.to_v(featuremap)

            def split_heads(tensor):
                return tensor.reshape(batch, self.heads, -1, height * width).permute(0, 1, 3, 2)

            q, k, v = split_heads(q), split_heads(k), split_heads(v)
            q = q * self.scale
            logits = torch.einsum("bhxd,bhyd->bhxy", q, k)
            logits = logits + self.pos_emb(q)
            weights = self.softmax(logits)
            attended = torch.einsum("bhxy,bhyd->bhxd", weights, v)
            return attended.permute(0, 1, 3, 2).reshape(batch, -1, height, width)

    class BoTBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shortcut = nn.Identity()
            self.net = nn.Sequential(
                nn.Conv2d(1024, 256, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                MultiHeadSpatialAttention(256, (12, 12)),
                nn.Identity(),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 1024, 1, bias=False),
                nn.BatchNorm2d(1024),
            )
            self.activation = nn.ReLU()

        def forward(self, featuremap):
            return self.activation(self.net(featuremap) + self.shortcut(featuremap))

    class BoTStack(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(BoTBlock(), BoTBlock(), BoTBlock())

        def forward(self, x):
            return self.net(x)

    class RQVQAFold(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.feature_extraction = timm.create_model(
                "swin_base_patch4_window12_384.ms_in22k",
                pretrained=False,
                num_classes=0,
                global_pool="",
            )
            self.bot4 = BoTStack()
            self.quality = nn.Sequential(nn.Linear(6639, 128), nn.Linear(128, 1))

        def forward(self, spatial, slowfast, qalign, liqe, fastvqa):
            batch, frames = spatial.shape[:2]
            image_batch = spatial.reshape(-1, *spatial.shape[2:])
            features = self.feature_extraction.forward_features(image_batch)
            if features.ndim == 3:
                features = features.transpose(1, 2).reshape(-1, 1024, 12, 12)
            elif features.ndim == 4 and features.shape[-1] == 1024:
                features = features.permute(0, 3, 1, 2)
            elif features.ndim != 4 or features.shape[1] != 1024:
                raise RuntimeError(f"Unexpected RQ-VQA Swin feature shape: {tuple(features.shape)}")
            spatial_features = self.bot4(features).flatten(2).mean(dim=2)
            combined = torch.cat(
                (
                    spatial_features,
                    slowfast.reshape(-1, 256),
                    qalign.reshape(-1, 4096),
                    liqe.reshape(-1, 495),
                    fastvqa.reshape(-1, 768),
                ),
                dim=1,
            )
            return self.quality(combined).reshape(batch, frames).mean(dim=1)

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    checkpoint = {
        key[len("module."):] if key.startswith("module.") else key: value
        for key, value in checkpoint.items()
    }

    model = RQVQAFold()
    swin_state = {
        key[len("feature_extraction.feature_extraction."):]: value
        for key, value in checkpoint.items()
        if key.startswith("feature_extraction.feature_extraction.")
    }
    bot_state = {
        key[len("bot4."):]: value for key, value in checkpoint.items() if key.startswith("bot4.")
    }
    quality_state = {
        key[len("quality."):]: value for key, value in checkpoint.items() if key.startswith("quality.")
    }
    model.feature_extraction.load_state_dict(_remap_swin_state_dict(swin_state), strict=True)
    model.bot4.load_state_dict(bot_state, strict=True)
    model.quality.load_state_dict(quality_state, strict=True)
    return model.to(device=device, dtype=dtype).eval()


def _build_slowfast(checkpoint_path: Path, device: str, dtype):
    import torch
    import torch.nn as nn
    from pytorchvideo.models.hub import slowfast_r50

    base = slowfast_r50(pretrained=False)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
    base.load_state_dict(state, strict=True)
    stages = nn.Sequential(*list(base.children())[0])

    class SlowFastFeatures(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.feature_extraction = nn.Sequential(*[stages[index] for index in range(5)])
            self.fast_avg_pool = stages[5].pool[1]
            self.output_pool = stages[6].output_pool

        def forward(self, pathways):
            features = self.feature_extraction(pathways)
            return self.output_pool(self.fast_avg_pool(features[1]))

    return SlowFastFeatures().to(device=device, dtype=dtype).eval()


def _build_liqe(checkpoint_path: Path, device: str, dtype):
    from itertools import product

    import open_clip
    import torch

    model = open_clip.create_model("ViT-B-32", pretrained=None)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    checkpoint = {
        key[len("module."):] if key.startswith("module.") else key: value
        for key, value in checkpoint.items()
    }
    model.load_state_dict(checkpoint, strict=True)
    model = model.to(device=device, dtype=dtype).eval()

    prompts = [
        f"a photo of a {scene} with {distortion} artifacts, which is of {quality} quality"
        for quality, scene, distortion in product(
            _LIQE_QUALITIES, _LIQE_SCENES, _LIQE_DISTORTIONS
        )
    ]
    tokens = open_clip.tokenize(prompts).to(device)
    with torch.inference_mode():
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
    return model, text_features


class RQVQAModule(PipelineModule):
    name = "rqvqa"
    description = "RQ-VQA rich quality-aware blind VQA ensemble (raw regression score)"
    default_config = {
        "ensemble_size": 10,
        "device": "auto",
        "dtype": "float16",
        "qalign_dtype": "float16",
        "fastvqa_seed": 42,
        "models_dir": "models",
    }
    metric_groups = {"rqvqa_score": "nr_quality"}
    metric_info = {
        "rqvqa_score": "RQ-VQA raw PLCC-trained regression score (unbounded; higher=better)"
    }
    models = [
        {
            "id": f"rqvqa/{name}",
            "type": "local",
            "url": f"{_HF_ROOT}/rqvqa/{name}",
            "task": "RQ-VQA released ensemble fold",
            "auto_download": "yes",
        }
        for name in _CHECKPOINTS
    ] + [
        {
            "id": f"rqvqa/{_LIQE_FILE}",
            "type": "local",
            "url": f"{_HF_ROOT}/rqvqa/{_LIQE_FILE}",
            "task": "RQ-VQA LIQE feature encoder",
            "auto_download": "yes",
        },
        {
            "id": f"rqvqa/{_SLOWFAST_FILE}",
            "type": "local",
            "url": f"{_HF_ROOT}/rqvqa/{_SLOWFAST_FILE}",
            "task": "RQ-VQA SlowFast-R50 motion encoder",
            "auto_download": "yes",
        },
        {
            "id": "q-future/one-align",
            "type": "huggingface",
            "task": "RQ-VQA Q-Align 4096-D feature encoder",
            "auto_download": "yes",
        },
        {
            "id": "fast_vqa/FAST_VQA_B_1_4.pth",
            "type": "local",
            "url": f"{_HF_ROOT}/fast_vqa/FAST_VQA_B_1_4.pth",
            "task": "RQ-VQA FAST-VQA 768-D feature encoder",
            "auto_download": "yes",
        },
    ]

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self.ensemble_size = max(1, min(int(self.config.get("ensemble_size", 10)), 10))
        self.fastvqa_seed = int(self.config.get("fastvqa_seed", 42))
        self._device = "cpu"
        self._dtype = None
        self._models: List[object] = []
        self._slowfast = None
        self._liqe = None
        self._liqe_text_features = None
        self._qalign_module = None
        self._fastvqa_module = None
        self._backend = "unavailable"
        self._ml_available = False

    def setup(self) -> None:
        if self.test_mode:
            return
        try:
            from ayase.modules.fast_vqa import FastVQAModule
            from ayase.modules.q_align import QAlignModule
            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            self._dtype = _resolve_dtype(self._device, str(self.config.get("dtype", "float16")))
            models_dir = str(self.config.get("models_dir") or "models")

            checkpoint_paths = [
                download_model_file(
                    f"rqvqa/{name}", f"{_HF_ROOT}/rqvqa/{name}", models_dir
                )
                for name in _CHECKPOINTS[:self.ensemble_size]
            ]
            liqe_path = download_model_file(
                f"rqvqa/{_LIQE_FILE}", f"{_HF_ROOT}/rqvqa/{_LIQE_FILE}", models_dir
            )
            slowfast_path = download_model_file(
                f"rqvqa/{_SLOWFAST_FILE}", f"{_HF_ROOT}/rqvqa/{_SLOWFAST_FILE}", models_dir
            )

            self._models = [
                _build_rqvqa_model(path, self._device, self._dtype)
                for path in checkpoint_paths
            ]
            self._slowfast = _build_slowfast(slowfast_path, self._device, self._dtype)
            self._liqe, self._liqe_text_features = _build_liqe(
                liqe_path, self._device, self._dtype
            )

            shared_config = {
                "device": self._device,
                "models_dir": models_dir,
            }
            self._qalign_module = QAlignModule(
                {
                    **shared_config,
                    "dtype": self.config.get("qalign_dtype", "float16"),
                }
            )
            self._qalign_module.setup()
            if not self._qalign_module._ml_available:
                raise RuntimeError("Q-Align feature encoder failed to initialise")

            self._fastvqa_module = FastVQAModule(
                {**shared_config, "model_type": "FAST-VQA"}
            )
            self._fastvqa_module.setup()
            if not self._fastvqa_module._ml_available:
                raise RuntimeError("FAST-VQA feature encoder failed to initialise")

            self._backend = "rqvqa"
            self._ml_available = True
            logger.info(
                "RQ-VQA initialised on %s with %d published ensemble fold(s)",
                self._device,
                len(self._models),
            )
        except Exception as error:
            self._backend = "unavailable"
            self._ml_available = False
            logger.warning(
                "RQ-VQA real backend unavailable (%s: %s); rqvqa_score left unset.",
                type(error).__name__,
                error,
            )

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or self._backend != "rqvqa" or not sample.is_video:
            return sample
        try:
            score = self._score_video(sample)
            if score is None:
                return sample
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            # PLCC training leaves the released regression scale unbounded.
            sample.quality_metrics.rqvqa_score = float(score)
        except Exception as error:
            logger.warning("RQ-VQA failed for %s: %s", sample.path, error)
        return sample

    def _score_video(self, sample: Sample) -> Optional[float]:
        import torch

        key_frames, motion_clips = self._decode_inputs(sample.path)
        if len(key_frames) != 8 or len(motion_clips) != 8:
            return None

        spatial = self._spatial_batch(key_frames)
        slowfast = self._slowfast_features(motion_clips)
        qalign = self._qalign_features(key_frames)
        liqe = self._liqe_features(key_frames)
        fastvqa = self._fastvqa_features(sample.path).unsqueeze(0).repeat(8, 1)

        spatial = spatial.unsqueeze(0).to(device=self._device, dtype=self._dtype)
        slowfast = slowfast.unsqueeze(0).to(device=self._device, dtype=self._dtype)
        qalign = qalign.unsqueeze(0).to(device=self._device, dtype=self._dtype)
        liqe = liqe.unsqueeze(0).to(device=self._device, dtype=self._dtype)
        fastvqa = fastvqa.unsqueeze(0).to(device=self._device, dtype=self._dtype)

        scores = []
        with torch.inference_mode():
            for model in self._models:
                scores.append(
                    model(spatial, slowfast, qalign, liqe, fastvqa).reshape(-1)[0].float()
                )
        if not scores:
            return None
        return float(torch.stack(scores).mean().item())

    @staticmethod
    def _decode_inputs(path: Path) -> Tuple[List[np.ndarray], List[List[np.ndarray]]]:
        """Decode the first eight one-second anchors and their 32-frame clips."""
        import cv2

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return [], []
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = int(round(capture.get(cv2.CAP_PROP_FPS))) or 1
        available_clips = max(1, int(total / fps)) if total > 0 else 8
        starts = [min(index, available_clips - 1) * fps for index in range(8)]
        last_needed = max(starts) + 31

        frames: List[np.ndarray] = []
        while len(frames) <= last_needed:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        capture.release()
        if not frames:
            return [], []
        while len(frames) <= last_needed:
            frames.append(frames[-1])

        key_frames = [frames[start] for start in starts]
        clips = [[frames[start + offset] for offset in range(32)] for start in starts]
        return key_frames, clips

    def _spatial_batch(self, frames: Sequence[np.ndarray]):
        import cv2
        import torch

        tensors = []
        mean = torch.tensor(_SPATIAL_MEAN).view(3, 1, 1)
        std = torch.tensor(_SPATIAL_STD).view(3, 1, 1)
        for frame in frames:
            height, width = frame.shape[:2]
            scale = 384.0 / min(height, width)
            resized = cv2.resize(frame, (max(384, round(width * scale)), max(384, round(height * scale))))
            top = (resized.shape[0] - 384) // 2
            left = (resized.shape[1] - 384) // 2
            crop = resized[top:top + 384, left:left + 384]
            tensor = torch.from_numpy(np.ascontiguousarray(crop, dtype=np.float32)).permute(2, 0, 1) / 255.0
            tensors.append((tensor - mean) / std)
        return torch.stack(tensors)

    def _slowfast_features(self, clips: Sequence[Sequence[np.ndarray]]):
        import cv2
        import torch

        mean = torch.tensor(_MOTION_MEAN).view(3, 1, 1)
        std = torch.tensor(_MOTION_STD).view(3, 1, 1)
        outputs = []
        with torch.inference_mode():
            for clip_frames in clips:
                clip = []
                for frame in clip_frames:
                    image = cv2.resize(frame, (224, 224))
                    tensor = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)).permute(2, 0, 1) / 255.0
                    clip.append((tensor - mean) / std)
                fast = torch.stack(clip).permute(1, 0, 2, 3).unsqueeze(0)
                slow_indices = torch.linspace(0, fast.shape[2] - 1, fast.shape[2] // 4).long()
                slow = torch.index_select(fast, 2, slow_indices)
                feature = self._slowfast(
                    [
                        slow.to(device=self._device, dtype=self._dtype),
                        fast.to(device=self._device, dtype=self._dtype),
                    ]
                )
                outputs.append(feature.reshape(-1).float().cpu())
        return torch.stack(outputs)

    def _qalign_features(self, frames: Sequence[np.ndarray]):
        import torch
        from PIL import Image
        from ayase.vendor.q_align.modeling_mplug_owl2 import (
            IMAGE_TOKEN_INDEX,
            DEFAULT_IMAGE_TOKEN,
            expand2square,
            tokenizer_image_token,
        )

        model = self._qalign_module._model
        processor = model.image_processor
        background = tuple(int(value * 255) for value in processor.image_mean)
        crop_pairs = []
        for frame in frames:
            image = Image.fromarray(frame).convert("RGB")
            width, height = image.size
            crop_pairs.append(
                (
                    expand2square(image.crop((0, 0, width, max(1, height // 2))), background),
                    expand2square(image.crop((0, height // 2, width, height)), background),
                )
            )

        model_device = model.device
        model_dtype = next(model.parameters()).dtype
        prompt = (
            "USER: How would you rate the quality of this image?\n"
            f"{DEFAULT_IMAGE_TOKEN}\nASSISTANT: The quality of the image is"
        )
        input_ids = tokenizer_image_token(
            prompt, model.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(model_device)

        features = []
        with torch.inference_mode():
            for crop_pair in crop_pairs:
                images = processor.preprocess(list(crop_pair), return_tensors="pt")["pixel_values"].to(
                    device=model_device, dtype=model_dtype
                )
                prepared = model.prepare_inputs_labels_for_multimodal(
                    input_ids.repeat(2, 1), None, None, None, images
                )
                _, modality, attention, past, embeddings, _ = prepared
                outputs = model.model(
                    input_ids=None,
                    modality_indicators=modality,
                    attention_mask=attention,
                    past_key_values=past,
                    inputs_embeds=embeddings,
                    use_cache=False,
                    return_dict=True,
                )
                features.append(outputs.last_hidden_state.mean(dim=(0, 1)))
        return torch.stack(features).float().cpu()

    def _liqe_features(self, frames: Sequence[np.ndarray]):
        import cv2
        import torch

        outputs = []
        mean = torch.tensor((0.48145466, 0.4578275, 0.40821073)).view(3, 1, 1)
        std = torch.tensor((0.26862954, 0.26130258, 0.27577711)).view(3, 1, 1)
        with torch.inference_mode():
            for frame in frames:
                height, width = frame.shape[:2]
                if min(height, width) < 224:
                    scale = 224.0 / min(height, width)
                    frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
                tensor = torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32)).permute(2, 0, 1) / 255.0
                tensor = ((tensor - mean) / std).unsqueeze(0)
                patches = tensor.unfold(2, 224, 32).unfold(3, 224, 32)
                patches = patches.permute(2, 3, 0, 1, 4, 5).reshape(-1, 3, 224, 224)
                step = patches.shape[0] // 15
                indices = torch.arange(15, dtype=torch.long) * step
                patches = patches[indices].to(device=self._device, dtype=self._dtype)
                image_features = self._liqe.encode_image(patches)
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                logits = self._liqe.logit_scale.exp() * image_features @ self._liqe_text_features.T
                outputs.append(logits.reshape(1, 15, 495).mean(dim=1).reshape(495).float().cpu())
        return torch.stack(outputs)

    def _fastvqa_features(self, path: Path):
        import torch

        # FAST-VQA's published test fragment sampler uses CPU torch.randint
        # even in evaluation mode. Isolate and seed that RNG so RQ-VQA is
        # reproducible without perturbing the application's global RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.fastvqa_seed)
            prepared = self._fastvqa_module._prepare_input(path)
        fragments = prepared["samples"]["fragments"].to(self._fastvqa_module.device)
        backbone = self._fastvqa_module._model.backbone["fragments"]
        with torch.inference_mode():
            features = backbone(fragments, multi=False, layer=-1)
            if features.ndim != 5 or features.shape[1] != 768:
                raise RuntimeError(f"Unexpected FAST-VQA feature shape: {tuple(features.shape)}")
            return features.mean(dim=(0, 2, 3, 4)).float().cpu()

    def on_dispose(self) -> None:
        self._models = []
        self._slowfast = None
        self._liqe = None
        self._liqe_text_features = None
        self._qalign_module = None
        self._fastvqa_module = None
        self._ml_available = False
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        super().on_dispose()
