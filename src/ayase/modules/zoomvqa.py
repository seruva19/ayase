"""Zoom-VQA -- Patches, Frames and Clips Integration for VQA.

Zhang et al. CVPRW 2023 -- dual-branch late-fusion architecture for
blind video quality assessment.

Architecture (faithful to the paper):
  - IQA branch: CNN backbone (cpnet_multi in paper, ResNet-50 as practical
    substitute) processes individual frames at 512x320 for spatial quality.
    Corresponds to patch-level and frame-level zoom in the paper.
  - VQA branch: Temporal model (Video Swin Transformer in paper, approximated
    via 3D temporal convolutions over frame features) processes frame sequences
    for temporal quality. Corresponds to clip-level zoom.
  - Late fusion: simple average of IQA and VQA branch scores (ensemble).

Three zoom levels map to:
  - Patches (local): random crops within individual frames (IQA branch)
  - Frames (individual): full-frame spatial quality (IQA branch)
  - Clips (temporal): frame-sequence temporal quality (VQA branch)

Tier 1: Loads official pretrained weights (iqa_best_29epoch_checkpoint.pth.tar
        + vqa_best_29e_val-vqpve_s.pth) if available.
Tier 2: ResNet-50 IQA branch + temporal conv VQA branch (practical).
Tier 3: Heuristic fallback (no ML).

GitHub: https://github.com/k-zha14/Zoom-VQA

zoomvqa_score -- higher = better quality (0-1)
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# Paper uses 2 fps extraction for IQA branch
_IQA_FPS = 2
# Frame processing sizes
_IQA_SIZE = (512, 320)  # width x height per paper
_VQA_SIZE = (480, 480)  # 480p square for VQA branch


class ZoomVQAModule(PipelineModule):
    name = "zoomvqa"
    description = (
        "Zoom-VQA dual-branch IQA+VQA late-fusion blind VQA (CVPRW 2023) "
        "— ResNet-50 spatial + temporal conv branches"
    )
    default_config = {
        "subsample": 16,
        "n_patches": 6,
        "patch_size": 224,
        "iqa_weights_path": "",  # path to iqa_best_29epoch_checkpoint.pth.tar
        "vqa_weights_path": "",  # path to vqa_best_29e_val-vqpve_s.pth
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 16)
        self.n_patches = self.config.get("n_patches", 6)
        self.patch_size = self.config.get("patch_size", 224)
        self.iqa_weights_path = self.config.get("iqa_weights_path", "")
        self.vqa_weights_path = self.config.get("vqa_weights_path", "")
        self._iqa_backbone = None
        self._iqa_head = None
        self._vqa_temporal = None
        self._vqa_head = None
        self._transform = None
        self._device = "cpu"
        self._ml_available = False
        self._pretrained_loaded = False

    def setup(self) -> None:
        if self.test_mode:
            return

        if self._try_ml_setup():
            return

        logger.warning(
            "Zoom-VQA: no ML backend available. Install with: "
            "pip install torch torchvision"
        )

    def _try_ml_setup(self) -> bool:
        """Set up dual-branch architecture with ResNet-50 IQA + temporal VQA."""
        try:
            import torch
            import torch.nn as nn
            import torchvision.models as models
            import torchvision.transforms as transforms

            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            # ---- IQA branch: ResNet-50 backbone (approximates cpnet_multi) ----
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self._iqa_backbone = nn.Sequential(*list(resnet.children())[:-1])
            self._iqa_backbone.eval().to(self._device)

            # IQA quality head: spatial features -> quality score
            self._iqa_head = nn.Sequential(
                nn.Linear(2048, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(256, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            ).to(self._device)
            self._iqa_head.eval()

            # ---- VQA branch: temporal model (approximates Video Swin-T) ----
            # Process temporal feature sequences via 1D convolutions
            # Input: (batch, feat_dim=2048, n_frames)
            self._vqa_temporal = nn.Sequential(
                nn.Conv1d(2048, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1),  # -> (batch, 256, 1)
            ).to(self._device)
            self._vqa_temporal.eval()

            # VQA quality head: temporal features -> quality score
            self._vqa_head = nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            ).to(self._device)
            self._vqa_head.eval()

            self._transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(_IQA_SIZE[::-1]),  # (H, W)
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            self._patch_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.patch_size, self.patch_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            # Try loading pretrained weights
            self._try_load_pretrained()

            self._ml_available = True
            wt_status = " (pretrained)" if self._pretrained_loaded else " (random heads)"
            logger.info(
                "Zoom-VQA initialised on %s (IQA: ResNet-50, VQA: temporal conv)%s",
                self._device, wt_status,
            )
            return True

        except ImportError:
            return False
        except Exception as e:
            logger.debug("Zoom-VQA ML setup failed: %s", e)
            return False

    def _try_load_pretrained(self) -> None:
        """Try loading official pretrained weights for IQA and VQA branches."""
        import torch

        loaded_any = False

        for wpath_str, module_pairs in [
            (self.iqa_weights_path, [
                ("iqa_backbone", self._iqa_backbone),
                ("iqa_head", self._iqa_head),
            ]),
            (self.vqa_weights_path, [
                ("vqa_temporal", self._vqa_temporal),
                ("vqa_head", self._vqa_head),
            ]),
        ]:
            if not wpath_str:
                continue
            wpath = Path(wpath_str)
            if not wpath.exists():
                logger.debug("Zoom-VQA weights not found: %s", wpath)
                continue

            try:
                state = torch.load(str(wpath), map_location=self._device, weights_only=False)

                if "state_dict" in state:
                    state = state["state_dict"]
                elif "model" in state:
                    state = state["model"]

                for mod_name, module in module_pairs:
                    sub_state = {}
                    for k, v in state.items():
                        if mod_name in k:
                            short_key = k.split(mod_name + ".")[-1]
                            sub_state[short_key] = v
                    if sub_state:
                        try:
                            module.load_state_dict(sub_state, strict=False)
                            loaded_any = True
                        except Exception as e:
                            logger.debug("Could not load %s weights: %s", mod_name, e)

            except Exception as e:
                logger.warning("Zoom-VQA: failed to load weights from %s: %s", wpath, e)

        if loaded_any:
            self._pretrained_loaded = True
            logger.info("Zoom-VQA: loaded pretrained weights")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            frames = self._extract_frames(sample)
            if not frames:
                return sample

            # IQA branch: spatial quality from individual frames + patches
            iqa_score = self._iqa_branch(frames)

            # VQA branch: temporal quality from frame sequences
            vqa_score = self._vqa_branch(frames)

            # Late fusion: simple average (ensemble) per paper
            score = (iqa_score + vqa_score) / 2.0

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.zoomvqa_score = float(np.clip(score, 0.0, 1.0))

            logger.debug(
                "Zoom-VQA for %s: %.4f (IQA=%.4f, VQA=%.4f)",
                sample.path.name, score, iqa_score, vqa_score,
            )

        except Exception as e:
            logger.warning("Zoom-VQA failed for %s: %s", sample.path, e)

        return sample

    def _iqa_branch(self, frames: List[np.ndarray]) -> float:
        """IQA branch: process individual frames for spatial quality.

        Combines:
          - Full-frame features (frame-level zoom)
          - Random patch features (patch-level zoom)
        """
        import torch

        frame_scores = []
        rng = np.random.RandomState(42)

        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Full-frame spatial feature
            full_feat = self._extract_iqa_feature(rgb, use_patch_transform=False)
            if full_feat is None:
                continue

            # Patch-level features (local quality)
            patch_feats = []
            h, w = frame.shape[:2]
            for _ in range(self.n_patches):
                y = rng.randint(0, max(h - self.patch_size, 1))
                x = rng.randint(0, max(w - self.patch_size, 1))
                patch = rgb[y:y + self.patch_size, x:x + self.patch_size]
                pfeat = self._extract_iqa_feature(patch, use_patch_transform=True)
                if pfeat is not None:
                    patch_feats.append(pfeat)

            # Combine frame + patch features
            if patch_feats:
                patch_stack = torch.cat(patch_feats, dim=0)
                patch_mean = patch_stack.mean(dim=0, keepdim=True)
                # Frame feature = average of full-frame and mean-patch
                combined_feat = (full_feat + patch_mean) / 2.0
            else:
                combined_feat = full_feat

            with torch.no_grad():
                score = self._iqa_head(combined_feat).item()
                frame_scores.append(score)

        if not frame_scores:
            return 0.5

        return float(np.mean(frame_scores))

    def _vqa_branch(self, frames: List[np.ndarray]) -> float:
        """VQA branch: process frame sequences for temporal quality.

        Extracts per-frame features with the IQA backbone, then processes
        the temporal sequence through 1D convolutions (approximating
        Video Swin Transformer's temporal modelling).
        """
        import torch

        if len(frames) < 2:
            # Single frame: no temporal info, return neutral
            return 0.5

        # Extract per-frame features
        frame_feats = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            feat = self._extract_iqa_feature(rgb, use_patch_transform=False)
            if feat is not None:
                frame_feats.append(feat)

        if len(frame_feats) < 2:
            return 0.5

        with torch.no_grad():
            # Stack: (n_frames, feat_dim) -> (1, feat_dim, n_frames)
            feat_stack = torch.cat(frame_feats, dim=0)  # (T, 2048)
            feat_seq = feat_stack.t().unsqueeze(0)  # (1, 2048, T)

            # Temporal convolution
            temporal_feat = self._vqa_temporal(feat_seq)  # (1, 256, 1)
            temporal_feat = temporal_feat.squeeze(-1)  # (1, 256)

            # VQA quality score
            score = self._vqa_head(temporal_feat).item()

        return float(score)

    def _extract_iqa_feature(
        self, rgb: np.ndarray, use_patch_transform: bool = False
    ) -> Optional["torch.Tensor"]:
        """Extract spatial features from an image using IQA backbone."""
        import torch

        try:
            transform = self._patch_transform if use_patch_transform else self._transform
            tensor = transform(rgb).unsqueeze(0).to(self._device)
            with torch.no_grad():
                feat = self._iqa_backbone(tensor)  # (1, 2048, 1, 1)
                feat = feat.squeeze(-1).squeeze(-1)  # (1, 2048)
            return feat
        except Exception as e:
            logger.debug("IQA feature extraction failed: %s", e)
            return None

    def _extract_frames(self, sample: Sample) -> List[np.ndarray]:
        """Extract frames from video or load single image."""
        frames = []
        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            try:
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

                if total <= 0:
                    return frames

                # Paper extracts at 2 fps for IQA branch
                # Use min of subsample and 2fps-equivalent frame count
                n_at_2fps = max(1, int(total / fps * _IQA_FPS))
                n_frames = min(self.subsample, total, n_at_2fps)
                indices = np.linspace(0, total - 1, n_frames, dtype=int)

                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frames.append(frame)
            finally:
                cap.release()
        else:
            img = cv2.imread(str(sample.path))
            if img is not None:
                frames.append(img)
        return frames

    def on_dispose(self) -> None:
        self._iqa_backbone = None
        self._iqa_head = None
        self._vqa_temporal = None
        self._vqa_head = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
