"""SimpleVQA — Simple Blind Video Quality Assessment (Sun et al., 2022).

Two-branch no-reference UGC video quality model:

* **Spatial branch** — a Swin-B (patch4, window12, 384, ImageNet-22k) backbone
  extracts a 1024-d feature from each of eight uniformly sampled key frames
  (resize 384, center-crop 384, ImageNet normalisation).
* **Motion branch** — a Kinetics SlowFast-R50 extracts the 256-d *fast-pathway*
  feature from a 32-frame clip anchored at each key frame (224x224, mean 0.45).

Per frame the 1024-d spatial and 256-d motion features are concatenated
(1280-d) and passed through a two-layer MLP quality head (1280->128->1); the
per-frame scores are averaged into ``simplevqa_score`` (higher = better).

The real checkpoint (``Swin_b_384_in22k_SlowFast_Fast_LSVQ.pth``, trained on
LSVQ) is mirrored on the Hugging Face Hub. Its state_dict follows the
``feature_extraction.*`` (Swin) + ``quality.*`` (MLP head) layout of the
upstream ``Swin_b_384_in22k`` module; the Swin sub-tree is re-keyed to the
current ``timm`` Swin layout (the checkpoint predates timm's move of
``PatchMerging`` from the end of a stage to the start of the next one) so it
loads strictly.

Only the real trained model produces ``simplevqa_score``. When the weights or a
required dependency (``timm`` / ``pytorchvideo`` / ``torch``) are missing the
score is left ``None`` — no heuristic substitute.

Upstream: https://github.com/sunwei925/SimpleVQA (model def / preprocessing
mirrored from the ``Swin_b_384_in22k`` base model in
https://github.com/sunwei925/RQ-VQA).

simplevqa_score — higher = better quality
"""

import logging
import re
from typing import List, Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

_HF_REPO = "AkaneTendo25/ayase-models"
_HF_WEIGHTS = "weights/simplevqa/Swin_b_384_in22k_SlowFast_Fast_LSVQ.pth"

_SPATIAL_MEAN = (0.485, 0.456, 0.406)
_SPATIAL_STD = (0.229, 0.224, 0.225)
_MOTION_MEAN = (0.45, 0.45, 0.45)
_MOTION_STD = (0.225, 0.225, 0.225)


def _remap_swin_state_dict(feat_sd: dict) -> dict:
    """Re-key an old-timm Swin state_dict to the current timm Swin layout.

    Two differences between the checkpoint's timm vintage and current timm:

    * ``PatchMerging`` (``downsample``) used to live at the *end* of stages
      0/1/2; current timm attaches it at the *start* of stages 1/2/3 — i.e. the
      stage index is shifted by ``+1``.
    * ``relative_position_index`` / ``attn_mask`` are non-persistent buffers in
      current timm (recomputed per forward), so they are dropped.
    """
    out = {}
    for k, v in feat_sd.items():
        mo = re.match(r"layers\.(\d+)\.downsample\.(.*)", k)
        if mo:
            out[f"layers.{int(mo.group(1)) + 1}.downsample.{mo.group(2)}"] = v
        else:
            out[k] = v
    return {
        k: v
        for k, v in out.items()
        if not (k.endswith("relative_position_index") or k.endswith("attn_mask"))
    }


def _build_model(device: str):
    """Reconstruct the SimpleVQA Swin+SlowFast-fast network and load weights.

    Returns ``(spatial_model, slowfast_model)`` both on ``device`` in eval mode.
    Raises on any missing dependency / weight mismatch so the caller can mark the
    backend unavailable.
    """
    import timm
    import torch
    import torch.nn as nn
    from huggingface_hub import hf_hub_download
    from pytorchvideo.models.hub import slowfast_r50

    class SimpleVQASwin(nn.Module):
        """Spatial branch + quality head (``feature_extraction`` + ``quality``)."""

        def __init__(self) -> None:
            super().__init__()
            # global_pool='avg', num_classes=0 -> timm head is Identity, forward
            # returns the 1024-d pooled feature (matches upstream head=Identity).
            self.feature_extraction = timm.create_model(
                "swin_base_patch4_window12_384.ms_in22k",
                pretrained=False,
                num_classes=0,
                global_pool="avg",
            )
            self.quality = nn.Sequential(
                nn.Linear(1024 + 256, 128),
                nn.Linear(128, 1),
            )

        def forward(self, x, x_3D_features):
            # x: (batch, frames, 3, H, W); x_3D_features: (batch, frames, 256)
            b, f = x.shape[0], x.shape[1]
            x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
            x_3D_features = x_3D_features.view(-1, x_3D_features.shape[2])
            x = self.feature_extraction(x)
            x = torch.cat((x, x_3D_features), dim=1)
            x = self.quality(x)
            x = x.view(b, f)
            return torch.mean(x, dim=1)

    class SlowFastFast(nn.Module):
        """Kinetics SlowFast-R50; exposes the 256-d fast-pathway feature."""

        def __init__(self) -> None:
            super().__init__()
            feats = nn.Sequential(*list(slowfast_r50(pretrained=True).children())[0])
            self.feature_extraction = nn.Sequential()
            self.fast_avg_pool = nn.Sequential()
            self.adp_avg_pool = nn.Sequential()
            for i in range(5):
                self.feature_extraction.add_module(str(i), feats[i])
            self.fast_avg_pool.add_module("fast_avg_pool", feats[5].pool[1])
            self.adp_avg_pool.add_module("adp_avg_pool", feats[6].output_pool)

        def forward(self, x):
            with torch.no_grad():
                x = self.feature_extraction(x)
                fast = self.fast_avg_pool(x[1])
                fast = self.adp_avg_pool(fast)
            return fast

    weights_path = hf_hub_download(repo_id=_HF_REPO, filename=_HF_WEIGHTS)
    sd = torch.load(weights_path, map_location="cpu", weights_only=True)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}

    feat_sd = {
        k[len("feature_extraction."):]: v
        for k, v in sd.items()
        if k.startswith("feature_extraction.")
    }
    qual_sd = {
        k[len("quality."):]: v for k, v in sd.items() if k.startswith("quality.")
    }

    model = SimpleVQASwin()
    # strict=True: the architecture must exactly match the 366-key checkpoint.
    model.feature_extraction.load_state_dict(_remap_swin_state_dict(feat_sd), strict=True)
    model.quality.load_state_dict(qual_sd, strict=True)
    model = model.to(device).eval()

    slowfast = SlowFastFast().to(device).eval()
    return model, slowfast


class SimpleVQAModule(PipelineModule):
    name = "simplevqa"
    description = "SimpleVQA Swin+SlowFast blind VQA (real model only)"
    default_config = {
        "n_frames": 8,        # key frames / motion clips (upstream base model uses 8)
        "clip_len": 32,       # frames per SlowFast fast-pathway clip
        "spatial_size": 384,  # swin_base_patch4_window12_384 native resolution
        "motion_size": 224,   # SlowFast input resolution
        "device": "auto",
    }
    metric_groups = {
        "simplevqa_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.n_frames = self.config.get("n_frames", 8)
        self.clip_len = self.config.get("clip_len", 32)
        self.spatial_size = self.config.get("spatial_size", 384)
        self.motion_size = self.config.get("motion_size", 224)
        self._backend = None
        self._ml_available = False
        self._model = None
        self._slowfast = None
        self._device = "cpu"

    def setup(self) -> None:
        if self.test_mode:
            return
        try:
            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            self._model, self._slowfast = _build_model(self._device)
            self._backend = "real"
            self._ml_available = True
            logger.info("SimpleVQA loaded real model (%s) on %s", _HF_WEIGHTS, self._device)
            return
        except Exception as e:  # missing dep, missing weights, or key mismatch
            logger.warning(
                "SimpleVQA: real model unavailable (%s: %s); simplevqa_score left unset.",
                type(e).__name__,
                e,
            )
        self._backend = "unavailable"
        self._ml_available = False

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or self._backend != "real" or self._model is None:
            # No trained weights available; do not fabricate a score.
            return sample
        if not sample.is_video:
            # SimpleVQA is a video model (motion branch); skip stills.
            return sample

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        try:
            score = self._score_video(sample)
            if score is not None:
                sample.quality_metrics.simplevqa_score = float(score)
        except Exception as e:
            logger.warning("SimpleVQA failed on %s: %s", sample.path, e)
        return sample

    # ------------------------------------------------------------------ #
    # inference
    # ------------------------------------------------------------------ #
    def _score_video(self, sample: Sample) -> Optional[float]:
        import torch

        frames = self._decode_frames(sample)
        if not frames:
            return None
        n = min(self.n_frames, len(frames))
        # Uniformly spaced key-frame anchors across the decoded frames.
        anchors = np.linspace(0, len(frames) - 1, n).round().astype(int).tolist()

        spatial = torch.stack(
            [self._spatial_tensor(frames[a]) for a in anchors]
        ).unsqueeze(0).to(self._device)  # (1, n, 3, S, S)

        motion_feats = []
        for a in anchors:
            clip = self._motion_clip(frames, a)  # (1, 3, clip_len, M, M)
            slow = torch.index_select(
                clip,
                2,
                torch.linspace(0, clip.shape[2] - 1, clip.shape[2] // 4).long(),
            )
            fast = self._slowfast([slow.to(self._device), clip.to(self._device)])
            motion_feats.append(fast.reshape(-1))
        motion = torch.stack(motion_feats).unsqueeze(0).to(self._device)  # (1, n, 256)

        with torch.no_grad():
            out = self._model(spatial, motion)
        return float(out.reshape(-1)[0].item())

    def _decode_frames(self, sample: Sample) -> List[np.ndarray]:
        """Decode a bounded, contiguous run of RGB frames for both branches.

        Motion clips need temporally contiguous frames, so we decode
        sequentially rather than via the seek-based cache. Decoding is bounded
        to ``n_frames`` anchors plus one ``clip_len`` tail.
        """
        import cv2

        cap = cv2.VideoCapture(str(sample.path))
        if not cap.isOpened():
            return []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        cap_frames = max(self.n_frames * self.clip_len, self.clip_len) + self.clip_len
        limit = min(total, cap_frames) if total > 0 else cap_frames
        frames: List[np.ndarray] = []
        idx = 0
        while idx < limit:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            idx += 1
        cap.release()
        return frames

    def _spatial_tensor(self, frame_rgb: np.ndarray):
        """Resize(384) + CenterCrop(384) + ImageNet normalise -> (3, S, S)."""
        import cv2
        import torch

        s = self.spatial_size
        h, w = frame_rgb.shape[:2]
        scale = s / min(h, w)
        img = cv2.resize(frame_rgb, (max(s, round(w * scale)), max(s, round(h * scale))))
        ch, cw = img.shape[:2]
        top = (ch - s) // 2
        left = (cw - s) // 2
        img = img[top:top + s, left:left + s]
        t = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32)).permute(2, 0, 1) / 255.0
        mean = torch.tensor(_SPATIAL_MEAN).view(3, 1, 1)
        std = torch.tensor(_SPATIAL_STD).view(3, 1, 1)
        return (t - mean) / std

    def _motion_clip(self, frames: List[np.ndarray], start: int):
        """Build one SlowFast clip: (1, 3, clip_len, M, M), mean-0.45 normalised."""
        import cv2
        import torch

        m = self.motion_size
        mean = torch.tensor(_MOTION_MEAN).view(3, 1, 1)
        std = torch.tensor(_MOTION_STD).view(3, 1, 1)
        clip = torch.zeros(self.clip_len, 3, m, m)
        for j in range(self.clip_len):
            fi = min(start + j, len(frames) - 1)
            img = cv2.resize(frames[fi], (m, m))
            t = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32)).permute(2, 0, 1) / 255.0
            clip[j] = (t - mean) / std
        # (clip_len, 3, M, M) -> (1, 3, clip_len, M, M)
        return clip.permute(1, 0, 2, 3).unsqueeze(0)
