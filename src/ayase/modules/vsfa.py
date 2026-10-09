"""VSFA — Video quality assessment with quality-aware temporal pooling.

Li et al. "Quality-Aware Features for Video Quality Assessment"
ACM Multimedia 2019.  GitHub: https://github.com/lidq92/VSFA

Faithful port of the official ``VSFA.py`` + ``CNNfeatures.py`` pipeline:
  1. ResNet-50 res5c features per frame: mean + std global spatial pooling
     → 4096-d content-aware feature per frame (all frames, native size,
     ImageNet normalisation only).
  2. ``ANN`` — Linear(4096→128) dimensionality reduction.
  3. ``GRU(128→32)`` temporal modelling → ``Linear(32→1)`` per-frame quality.
  4. ``TP`` subjectively-inspired temporal pooling (tau=12, beta=0.5):
     softmin-weighted windowed mean blended with the running minimum;
     the video score is the mean over pooled positions.

The trained checkpoint is the official ``models/VSFA.pt`` from the VSFA repo
(mirrored on HuggingFace). If the weights cannot be loaded the metric stays
``None`` — an untrained head does not reproduce VSFA.

vsfa_score — higher = better quality
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# Official trained checkpoint (models/VSFA.pt in lidq92/VSFA), mirrored on HF.
_HF_REPO = "AkaneTendo25/ayase-assets"
_HF_FILENAME = "vsfa/VSFA.pt"
_OFFICIAL_URL = "https://github.com/lidq92/VSFA/raw/master/models/VSFA.pt"


def _build_vsfa_head(nn):
    """Official VSFA head: ANN(4096→128) → GRU(128→32) → Linear(32→1)."""
    ann = nn.Linear(4096, 128)  # ANN with n_ANNlayers=1 is fc0 only
    rnn = nn.GRU(128, 32, batch_first=True)
    q = nn.Linear(32, 1)
    for m in (ann, rnn, q):
        m.eval()
    return ann, rnn, q


def _tp_pool(q, torch, F, tau=12, beta=0.5):
    """Verbatim TP() from VSFA.py — subjectively-inspired temporal pooling.

    q: (T,) per-frame quality → (1, 1, T-tau+1) pooled values.
    """
    q = torch.unsqueeze(torch.t(q), 0)
    qm = -float("inf") * torch.ones((1, 1, tau - 1)).to(q.device)
    qp = 10000.0 * torch.ones((1, 1, tau - 1)).to(q.device)
    l = -F.max_pool1d(torch.cat((qm, -q), 2), tau, stride=1)
    m = F.avg_pool1d(torch.cat((q * torch.exp(-q), qp * torch.exp(-qp)), 2), tau, stride=1)
    n = F.avg_pool1d(torch.cat((torch.exp(-q), torch.exp(-qp)), 2), tau, stride=1)
    m = m / n
    return beta * m + (1 - beta) * l


class VSFAModule(PipelineModule):
    name = "vsfa"
    provenance = "published"
    sources = {
        "vsfa_score": "VSFA, Li et al. ACM MM 2019 — https://github.com/lidq92/VSFA",
    }
    deviations = {
        "vsfa_score": "upstream reads all frames via skvideo at native res; no resize/crop, same as the source (frames are normalized by ImageNet statistics without resizing)",
    }
    description = "VSFA quality-aware feature aggregation with GRU (ACMMM 2019)"
    default_config = {
        "frame_batch_size": 64,  # upstream CNNfeatures.py batch
    }
    metric_groups = {
        "vsfa_score": "nr_quality",
    }
    models = [
        {
            "id": "lidq92/VSFA models/VSFA.pt",
            "type": "other",
            "url": _OFFICIAL_URL,
            "task": "Trained VSFA head (ANN+GRU+q) on KoNViD-1k",
        },
    ]

    def __init__(self, config=None):
        super().__init__(config)
        self.frame_batch_size = int(self.config.get("frame_batch_size", 64))
        self._ml_available = False
        self._has_trained_weights = False
        self._backbone = None
        self._ann = None
        self._rnn = None
        self._q = None
        self._device = "cpu"
        self._backend = None

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

            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))

            # --- ResNet-50 backbone up to res5c (children()[:-2]), matching
            # CNNfeatures.py ResNet50 (frozen, returns mean+std pools).
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self._backbone = nn.Sequential(*list(resnet.children())[:-2])
            for p in self._backbone.parameters():
                p.requires_grad = False
            self._backbone.eval()
            self._backbone.to(self._device)

            # --- Official head ---
            self._ann, self._rnn, self._q = _build_vsfa_head(nn)
            for m in (self._ann, self._rnn, self._q):
                m.to(self._device)

            self._has_trained_weights = self._try_load_weights(torch)

            if self._has_trained_weights:
                self._ml_available = True
                self._backend = "vsfa"
                logger.info(
                    "VSFA initialised on %s (res5c mean+std → ANN → GRU → TP)",
                    self._device,
                )
            else:
                self._ml_available = False
                self._backend = "unavailable"
                logger.warning(
                    "VSFA unavailable: trained VSFA.pt weights not available "
                    "(tried %s and %s)", _HF_REPO, _OFFICIAL_URL,
                )

        except ImportError:
            self._backend = "unavailable"
            logger.warning(
                "VSFA requires torch and torchvision. "
                "Install with: pip install torch torchvision"
            )
        except Exception as e:
            self._backend = "unavailable"
            logger.warning("VSFA setup failed: %s", e)

    def _try_load_weights(self, torch) -> bool:
        """Load the official VSFA.pt (ANN + GRU + q head)."""
        try:
            from huggingface_hub import hf_hub_download
            from ayase.config import resolve_assets_repo

            try:
                weights_path = hf_hub_download(
                    repo_id=resolve_assets_repo(self.config), filename=_HF_FILENAME
                )
            except Exception:
                weights_path = self._download_official()
                if weights_path is None:
                    return False
            checkpoint = torch.load(
                weights_path, map_location=self._device, weights_only=True
            )
            if not isinstance(checkpoint, dict):
                return False
            loaded = self._load_official_keys(checkpoint)
            if not loaded:
                logger.warning(
                    "VSFA checkpoint lacks official ann/rnn/q weights; "
                    "refusing to score with randomly initialised layers",
                )
                return False
            for m in (self._ann, self._rnn, self._q):
                m.eval()
            logger.info("Loaded trained VSFA weights")
            return True
        except ImportError:
            logger.debug("huggingface_hub not installed; skipping VSFA weight download")
        except Exception as e:
            logger.debug("Could not load VSFA weights: %s", e)
        return False

    def _download_official(self):
        """Fetch models/VSFA.pt from the official repo into models_dir."""
        try:
            from ayase.config import download_model_file

            models_dir = self.config.get("models_dir", "models")
            return download_model_file("vsfa/VSFA.pt", _OFFICIAL_URL, models_dir)
        except Exception as e:
            logger.debug("Official VSFA.pt download failed: %s", e)
            return None

    def _load_official_keys(self, state_dict: dict) -> bool:
        """Load keys of the official checkpoint form (ann.fc0/rnn/q), plus
        the legacy packaged form (gru/fc) for backward compatibility."""
        import torch  # noqa: F401

        sd = dict(state_dict)
        if "model_state_dict" in sd and isinstance(sd["model_state_dict"], dict):
            sd = sd["model_state_dict"]

        # Official VSFA.pt keys: ann.fc0.*, ann.fc.*, rnn.*, q.*
        if any(k.startswith("rnn.") for k in sd):
            ann_sd = {k[len("ann.fc0."):]: v for k, v in sd.items()
                      if k.startswith("ann.fc0.")}
            rnn_sd = {k[len("rnn."):]: v for k, v in sd.items()
                      if k.startswith("rnn.")}
            q_sd = {k[len("q."):]: v for k, v in sd.items()
                    if k.startswith("q.")}
            if not ann_sd or not rnn_sd or not q_sd:
                return False
            self._ann.load_state_dict(ann_sd)
            self._rnn.load_state_dict(rnn_sd)
            self._q.load_state_dict(q_sd)
            return True

        # Legacy packaged form: {"gru": {...}, "fc": {...}} for the old
        # 4096-in GRU + 32→1 FC — incompatible with the official head shapes
        # (official GRU input is 128); reject rather than mis-score.
        return False

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
        """res5c mean+std features → ANN → GRU → q → TP → mean."""
        import torch
        import torch.nn.functional as F

        frames_rgb = self._load_frames_rgb(sample)
        if not frames_rgb:
            return None

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self._device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self._device)

        feats = []
        with torch.no_grad():
            bs = max(1, self.frame_batch_size)
            for start in range(0, len(frames_rgb), bs):
                batch_np = np.stack(frames_rgb[start:start + bs]).astype(np.float32) / 255.0
                x = torch.from_numpy(batch_np).permute(0, 3, 1, 2).to(self._device)
                x = (x - mean) / std
                feat_map = self._backbone(x)                      # (B,2048,h,w)
                f_mean = F.adaptive_avg_pool2d(feat_map, 1)       # (B,2048,1,1)
                f_std = torch.std(
                    feat_map.view(feat_map.size(0), feat_map.size(1), -1, 1),
                    dim=2, keepdim=True,
                )                                                  # (B,2048,1,1)
                feats.append(torch.cat((f_mean, f_std), dim=1).squeeze(-1).squeeze(-1))

            features = torch.cat(feats, dim=0)                    # (T,4096)
            if features.ndim == 1:
                features = features.unsqueeze(0)

            reduced = self._ann(features.unsqueeze(0))            # (1,T,128)
            rnn_out, _ = self._rnn(reduced)                       # (1,T,32)
            q = self._q(rnn_out).squeeze(0).squeeze(-1)           # (T,)

            if q.numel() < 12:  # TP window tau=12 needs >= tau frames
                # Not enough frames for a single pooling window → None rather
                # than a padded approximation.
                return None
            pooled = _tp_pool(q, torch, F)
            score = torch.mean(pooled).item()

        return float(score) if np.isfinite(score) else None

    # ------------------------------------------------------------------
    # Frame loading
    # ------------------------------------------------------------------

    def _load_frames_rgb(self, sample: Sample) -> list:
        """All frames at native resolution (as upstream skvideo.vread)."""
        import cv2

        frames = []
        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            finally:
                cap.release()
        else:
            img = cv2.imread(str(sample.path))
            if img is not None:
                frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return frames
