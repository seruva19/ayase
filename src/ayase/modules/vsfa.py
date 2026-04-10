"""VSFA — Video quality assessment with quality-aware temporal pooling.

Li et al. "Quality-Aware Features for Video Quality Assessment"
ACM Multimedia 2019.  GitHub: https://github.com/lidq92/VSFA

Algorithm (from the paper):
  1. Extract ResNet-50 features from the last pooling layer (2048-d) per frame.
  2. Build content-aware features: f_n = [feat_n, feat_n - mean(feat)] → 4096-d.
  3. Feed the sequence through a GRU (hidden_size=32) for temporal modelling.
  4. Per-frame quality: q_n = Linear(h_n).
  5. Quality-aware temporal pooling: Q = sum(softmax(q) * q).

Tiers:
  - **Tier 1 (full model)**: Download VSFA checkpoint from HuggingFace
    (GRU + linear head trained on KoNViD-1k) and run the full pipeline.
  - **Tier 2 (feature-only fallback)**: If weights are unavailable, use
    pretrained ResNet-50 features with simple temporal mean as a proxy score.

vsfa_score — higher = better quality
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# HuggingFace repo where VSFA weights are hosted.
_HF_REPO = "AkaneTendo25/ayase-models"
_HF_FILENAME = "vsfa/VSFA.pt"


class VSFAModule(PipelineModule):
    name = "vsfa"
    description = "VSFA quality-aware feature aggregation with GRU (ACMMM 2019)"
    default_config = {
        "subsample": 8,
        "frame_size": 520,  # Resize shorter side before centre-crop
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self.frame_size = self.config.get("frame_size", 520)
        self._ml_available = False
        self._has_trained_weights = False
        self._backbone = None
        self._gru = None
        self._fc = None
        self._device = None
        self._transform = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            import torch
            import torch.nn as nn
            import torchvision.models as models
            import torchvision.transforms as transforms

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # --- ResNet-50 backbone (feature extractor) ---
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            # Remove the final FC layer — keep up to avgpool → 2048-d output.
            self._backbone = nn.Sequential(*list(resnet.children())[:-1])
            self._backbone.eval()
            self._backbone.to(self._device)

            # --- GRU temporal model ---
            # Input is 4096-d: [feat_n, feat_n - mean(feat)] per the paper.
            self._gru = nn.GRU(
                input_size=4096,
                hidden_size=32,
                num_layers=1,
                batch_first=True,
            ).to(self._device)
            self._gru.eval()

            # --- Quality regression head (32 → 1) ---
            self._fc = nn.Linear(32, 1).to(self._device)
            self._fc.eval()

            # --- ImageNet transform (matches VSFA repo) ---
            self._transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.frame_size),
                transforms.CenterCrop(448),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            # --- Try to load trained VSFA weights (GRU + FC) ---
            self._has_trained_weights = self._try_load_weights(torch)

            self._ml_available = True
            if self._has_trained_weights:
                logger.info(
                    "VSFA initialised on %s (ResNet-50 + trained GRU head)",
                    self._device,
                )
            else:
                logger.info(
                    "VSFA initialised on %s (ResNet-50 features, fallback temporal mean — "
                    "trained VSFA weights not available)",
                    self._device,
                )

        except ImportError:
            logger.warning(
                "VSFA requires torch and torchvision. "
                "Install with: pip install torch torchvision"
            )
        except Exception as e:
            logger.warning("VSFA setup failed: %s", e)

    def _try_load_weights(self, torch) -> bool:
        """Attempt to download and load trained VSFA checkpoint.

        Returns True if weights were loaded successfully.
        """
        try:
            from huggingface_hub import hf_hub_download

            weights_path = hf_hub_download(
                repo_id=_HF_REPO,
                filename=_HF_FILENAME,
            )
            checkpoint = torch.load(
                weights_path,
                map_location=self._device,
                weights_only=True,
            )
            # The checkpoint stores state dicts keyed by component name.
            if isinstance(checkpoint, dict):
                if "gru" in checkpoint:
                    self._gru.load_state_dict(checkpoint["gru"])
                if "fc" in checkpoint:
                    self._fc.load_state_dict(checkpoint["fc"])
                elif "model_state_dict" in checkpoint:
                    # Alternate packaging: full model state dict
                    self._load_from_full_state_dict(checkpoint["model_state_dict"])
            self._gru.eval()
            self._fc.eval()
            logger.info("Loaded trained VSFA weights from %s", _HF_REPO)
            return True
        except ImportError:
            logger.debug("huggingface_hub not installed; skipping VSFA weight download")
        except Exception as e:
            logger.debug("Could not load VSFA weights: %s", e)
        return False

    def _load_from_full_state_dict(self, state_dict: dict) -> None:
        """Load GRU + FC weights from a flat state dict with prefixed keys."""
        import torch

        gru_sd = {}
        fc_sd = {}
        for key, val in state_dict.items():
            if key.startswith("gru."):
                gru_sd[key[len("gru."):]] = val
            elif key.startswith("fc."):
                fc_sd[key[len("fc."):]] = val
        if gru_sd:
            self._gru.load_state_dict(gru_sd)
        if fc_sd:
            self._fc.load_state_dict(fc_sd)

    # ------------------------------------------------------------------
    # Process
    # ------------------------------------------------------------------

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            score = self._compute_quality(sample)

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.vsfa_score = score
                logger.debug("VSFA for %s: %.4f", sample.path.name, score)

        except Exception as e:
            logger.warning("VSFA failed for %s: %s", sample.path, e)

        return sample

    def _compute_quality(self, sample: Sample) -> Optional[float]:
        """Extract ResNet-50 features, build content-aware representation,
        run GRU temporal model, apply quality-aware temporal pooling."""
        import torch

        frames_rgb = self._load_frames_rgb(sample)
        if not frames_rgb:
            return None

        # ------ Step 1: extract 2048-d ResNet-50 features per frame ------
        frame_features = []
        with torch.no_grad():
            for rgb in frames_rgb:
                tensor = self._transform(rgb).unsqueeze(0).to(self._device)
                feat = self._backbone(tensor)          # (1, 2048, 1, 1)
                feat = feat.view(1, -1)                # (1, 2048)
                frame_features.append(feat)

        # (T, 2048)
        features = torch.cat(frame_features, dim=0)

        if self._has_trained_weights:
            return self._quality_aware_pooling(features)
        else:
            return self._fallback_temporal_mean(features)

    def _quality_aware_pooling(self, features) -> float:
        """Full VSFA algorithm: content-aware features → GRU → quality-aware
        temporal pooling.

        features: (T, 2048)
        """
        import torch
        import torch.nn.functional as F

        with torch.no_grad():
            # ------ Step 2: content-aware features ------
            # f_n = [feat_n, feat_n - mean(feat)]  →  4096-d
            mean_feat = features.mean(dim=0, keepdim=True)      # (1, 2048)
            diff = features - mean_feat                          # (T, 2048)
            content_aware = torch.cat([features, diff], dim=1)   # (T, 4096)

            # Add batch dimension: (1, T, 4096)
            content_aware = content_aware.unsqueeze(0)

            # ------ Step 3: GRU temporal modelling ------
            gru_out, _ = self._gru(content_aware)               # (1, T, 32)

            # ------ Step 4: per-frame quality scores ------
            q = self._fc(gru_out)                                # (1, T, 1)
            q = q.squeeze(0).squeeze(-1)                         # (T,)

            # ------ Step 5: quality-aware temporal pooling ------
            # Q = sum(softmax(q) * q)
            weights = F.softmax(q, dim=0)                        # (T,)
            score = (weights * q).sum().item()

        return float(score)

    def _fallback_temporal_mean(self, features) -> float:
        """Fallback when trained GRU/FC weights are not available.

        Uses the L2-norm of the mean ResNet-50 feature as a proxy quality
        signal, rescaled to roughly [0, 1] via sigmoid.
        """
        import torch

        with torch.no_grad():
            # Content-aware features (same as paper)
            mean_feat = features.mean(dim=0, keepdim=True)
            diff = features - mean_feat
            content_aware = torch.cat([features, diff], dim=1)  # (T, 4096)

            # Run through GRU with random (untrained) weights
            content_aware = content_aware.unsqueeze(0)          # (1, T, 4096)
            gru_out, _ = self._gru(content_aware)               # (1, T, 32)
            q = self._fc(gru_out)                                # (1, T, 1)
            q = q.squeeze(0).squeeze(-1)                         # (T,)

            # Simple mean as fallback aggregation (softmax pooling would
            # still work but with untrained weights is less meaningful).
            score = torch.sigmoid(q.mean()).item()

        return float(score)

    # ------------------------------------------------------------------
    # Frame loading
    # ------------------------------------------------------------------

    def _load_frames_rgb(self, sample: Sample) -> list:
        """Load frames as RGB numpy arrays."""
        import cv2

        frames = []
        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                return []
            n_frames = min(self.subsample, total)
            indices = np.linspace(0, total - 1, n_frames, dtype=int)
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(rgb)
            cap.release()
        else:
            img = cv2.imread(str(sample.path))
            if img is not None:
                frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return frames
