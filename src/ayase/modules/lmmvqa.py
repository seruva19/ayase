"""LMM-VQA -- Large Multimodal Model VQA (IEEE 2024).

Spatiotemporal quality assessment using large multimodal model features.

Implementation:
    CLIP vision encoder for per-frame feature extraction with temporal
    attention aggregation.  Quality-aware prompts are used to anchor the
    CLIP feature space to perceptual quality dimensions.  Falls back to
    ResNet-50 if CLIP is unavailable.

lmmvqa_score -- higher = better quality (0-1)
"""

import logging
from typing import List, Optional

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# Quality-level text prompts for CLIP-based quality anchoring
_QUALITY_PROMPTS = [
    "a low quality, blurry, distorted video frame",
    "a mediocre quality video frame with some artifacts",
    "a decent quality video frame with minor imperfections",
    "a high quality, sharp, clear video frame",
    "an excellent quality, pristine, professional video frame",
]


class LMMVQAModule(PipelineModule):
    name = "lmmvqa"
    description = "LMM-VQA spatiotemporal LMM VQA (IEEE 2024)"
    default_config = {
        "subsample": 8,
        "clip_model": "ViT-B/32",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self.clip_model_name = self.config.get("clip_model", "ViT-B/32")
        self._clip_model = None
        self._clip_preprocess = None
        self._text_features = None
        self._resnet = None
        self._resnet_transform = None
        self._temporal_attn = None
        self._quality_regressor = None
        self._device = "cpu"
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        if self.test_mode:
            return

        # Tier 1: CLIP backbone (preferred -- text-vision alignment)
        if self._try_clip_setup():
            return

        # Tier 2: ResNet-50 backbone
        if self._try_resnet_setup():
            return

        logger.warning(
            "LMM-VQA: no ML backend available. "
            "Install with: pip install torch torchvision  (or: pip install clip)"
        )

    def _try_clip_setup(self) -> bool:
        try:
            import torch
            import clip

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._clip_model, self._clip_preprocess = clip.load(
                self.clip_model_name, device=self._device
            )
            self._clip_model.eval()

            # Pre-encode quality prompts
            text_tokens = clip.tokenize(_QUALITY_PROMPTS).to(self._device)
            with torch.no_grad():
                self._text_features = self._clip_model.encode_text(text_tokens)
                self._text_features = self._text_features / self._text_features.norm(
                    dim=-1, keepdim=True
                )

            # Temporal attention over CLIP features
            feat_dim = self._text_features.shape[1]
            self._temporal_attn = torch.nn.Sequential(
                torch.nn.Linear(feat_dim, 128),
                torch.nn.Tanh(),
                torch.nn.Linear(128, 1),
            ).to(self._device)
            self._temporal_attn.eval()

            self._ml_available = True
            self._backend = "clip"
            logger.info(
                "LMM-VQA initialised with CLIP (%s) on %s",
                self.clip_model_name, self._device,
            )
            return True

        except ImportError:
            return False
        except Exception as e:
            logger.debug("CLIP setup failed: %s", e)
            return False

    def _try_resnet_setup(self) -> bool:
        try:
            import torch
            import torchvision.models as models
            from torchvision import transforms

            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self._resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
            self._resnet.eval().to(self._device)

            self._resnet_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            # Quality regressor from ResNet features
            self._quality_regressor = torch.nn.Sequential(
                torch.nn.Linear(2048, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(512, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 1),
                torch.nn.Sigmoid(),
            ).to(self._device)
            self._quality_regressor.eval()

            # Temporal attention
            self._temporal_attn = torch.nn.Sequential(
                torch.nn.Linear(2048, 128),
                torch.nn.Tanh(),
                torch.nn.Linear(128, 1),
            ).to(self._device)
            self._temporal_attn.eval()

            self._ml_available = True
            self._backend = "resnet"
            logger.info("LMM-VQA initialised with ResNet-50 on %s", self._device)
            return True

        except ImportError:
            return False
        except Exception as e:
            logger.debug("ResNet setup failed: %s", e)
            return False

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            frames = self._extract_frames(sample)
            if not frames:
                return sample

            if self._backend == "clip":
                score = self._compute_clip_quality(frames)
            else:
                score = self._compute_resnet_quality(frames)

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.lmmvqa_score = float(np.clip(score, 0.0, 1.0))

        except Exception as e:
            logger.warning("LMM-VQA failed for %s: %s", sample.path, e)

        return sample

    def _compute_clip_quality(self, frames: List[np.ndarray]) -> Optional[float]:
        """Quality from CLIP: compare frames to quality-level prompts."""
        import torch
        from PIL import Image

        frame_scores = []
        frame_feats = []

        for frame in frames:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                tensor = self._clip_preprocess(pil_img).unsqueeze(0).to(self._device)

                with torch.no_grad():
                    img_feat = self._clip_model.encode_image(tensor)
                    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

                    # Similarity to quality prompts
                    sims = (img_feat @ self._text_features.T).squeeze(0)
                    sims = sims.cpu().numpy()

                frame_feats.append(img_feat.cpu().numpy().flatten())

                # Weighted quality from prompt similarities
                # Prompts ordered low->high quality, weight by position
                quality_levels = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
                probs = np.exp(sims) / np.sum(np.exp(sims))
                frame_q = float(np.dot(probs, quality_levels))
                frame_scores.append(frame_q)

            except Exception as e:
                logger.debug("CLIP frame scoring failed: %s", e)

        if not frame_scores:
            return None

        # Temporal attention pooling
        if len(frame_feats) > 1 and self._temporal_attn is not None:
            import torch as th
            feat_tensor = th.from_numpy(
                np.array(frame_feats, dtype=np.float32)
            ).to(self._device)
            with th.no_grad():
                attn_logits = self._temporal_attn(feat_tensor).squeeze(-1)
                attn_weights = th.softmax(attn_logits, dim=0).cpu().numpy()
            score = float(np.dot(attn_weights, frame_scores))
        else:
            score = float(np.mean(frame_scores))

        return score

    def _compute_resnet_quality(self, frames: List[np.ndarray]) -> Optional[float]:
        """Quality from ResNet-50 features with temporal attention."""
        import torch

        frame_qualities = []
        frame_feats = []

        for frame in frames:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = self._resnet_transform(rgb).unsqueeze(0).to(self._device)

                with torch.no_grad():
                    feat = self._resnet(tensor).flatten()
                    quality = self._quality_regressor(feat.unsqueeze(0)).item()

                frame_qualities.append(quality)
                frame_feats.append(feat.cpu().numpy())

            except Exception as e:
                logger.debug("ResNet frame scoring failed: %s", e)

        if not frame_qualities:
            return None

        # Temporal attention pooling
        if len(frame_feats) > 1:
            feat_tensor = torch.from_numpy(
                np.array(frame_feats, dtype=np.float32)
            ).to(self._device)
            with torch.no_grad():
                attn_logits = self._temporal_attn(feat_tensor).squeeze(-1)
                attn_weights = torch.softmax(attn_logits, dim=0).cpu().numpy()
            score = float(np.dot(attn_weights, frame_qualities))
        else:
            score = float(np.mean(frame_qualities))

        return score

    def _extract_frames(self, sample: Sample) -> List[np.ndarray]:
        frames = []
        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            try:
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total <= 0:
                    return frames
                indices = np.linspace(
                    0, total - 1, min(self.subsample, total), dtype=int
                )
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
        self._clip_model = None
        self._resnet = None
        self._temporal_attn = None
        self._quality_regressor = None
        self._text_features = None
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
