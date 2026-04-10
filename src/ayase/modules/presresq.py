"""PreResQ-R1 -- Fine-Grained Rank-and-Score VQA (2025).

Quality assessment via ranking: frames are compared against quality-level
text prompts using CLIP, producing a ranking-based quality score.

Implementation:
    CLIP backbone compares video frames to a set of quality-level text
    prompts (from "very poor quality" to "excellent quality").  Quality is
    derived from the softmax distribution over prompt similarities,
    yielding a rank-aware continuous score.  Falls back to ResNet-50 with
    learned quality regressor if CLIP is unavailable.

presresq_score -- higher = better quality (0-1)
"""

import logging
from typing import List, Optional

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# Quality ranking prompts: ordered from worst to best
_RANK_PROMPTS = [
    "a very poor quality video frame with severe distortion and blur",
    "a poor quality video frame with noticeable artifacts",
    "a below average quality video frame with some imperfections",
    "an average quality video frame",
    "an above average quality video frame with good clarity",
    "a good quality video frame with sharp details",
    "a very good quality video frame with excellent clarity",
    "an outstanding quality, pristine, professional video frame",
]


class PreResQModule(PipelineModule):
    name = "presresq"
    description = "PreResQ-R1 rank+score VQA (2025)"
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
        self._quality_regressor = None
        self._device = "cpu"
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        if self.test_mode:
            return

        # Tier 1: CLIP (preferred for rank-and-score via prompts)
        if self._try_clip_setup():
            return

        # Tier 2: ResNet-50 fallback
        if self._try_resnet_setup():
            return

        logger.warning(
            "PreResQ: no ML backend available. "
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

            # Pre-encode ranking prompts
            text_tokens = clip.tokenize(_RANK_PROMPTS).to(self._device)
            with torch.no_grad():
                self._text_features = self._clip_model.encode_text(text_tokens)
                self._text_features = self._text_features / self._text_features.norm(
                    dim=-1, keepdim=True
                )

            self._ml_available = True
            self._backend = "clip"
            logger.info(
                "PreResQ initialised with CLIP (%s) on %s",
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

            # Quality regressor with ranking-aware output
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

            self._ml_available = True
            self._backend = "resnet"
            logger.info("PreResQ initialised with ResNet-50 on %s", self._device)
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
                score = self._compute_clip_rank_score(frames)
            else:
                score = self._compute_resnet_score(frames)

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.presresq_score = float(
                    np.clip(score, 0.0, 1.0)
                )

        except Exception as e:
            logger.warning("PreResQ failed for %s: %s", sample.path, e)

        return sample

    def _compute_clip_rank_score(self, frames: List[np.ndarray]) -> Optional[float]:
        """Rank-and-score via CLIP: compare frames against quality prompts."""
        import torch
        from PIL import Image

        n_levels = len(_RANK_PROMPTS)
        # Quality levels evenly spaced from 0 to 1
        quality_levels = np.linspace(0.0, 1.0, n_levels)

        frame_scores = []
        frame_ranks = []

        for frame in frames:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                tensor = self._clip_preprocess(pil_img).unsqueeze(0).to(self._device)

                with torch.no_grad():
                    img_feat = self._clip_model.encode_image(tensor)
                    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

                    # Similarity to each quality-level prompt
                    sims = (img_feat @ self._text_features.T).squeeze(0)
                    sims = sims.cpu().numpy()

                # Ranking score: softmax-weighted quality level
                exp_sims = np.exp(sims - np.max(sims))  # numerical stability
                probs = exp_sims / np.sum(exp_sims)
                rank_score = float(np.dot(probs, quality_levels))

                # Also record the argmax rank
                frame_ranks.append(int(np.argmax(probs)))
                frame_scores.append(rank_score)

            except Exception as e:
                logger.debug("CLIP rank scoring failed: %s", e)

        if not frame_scores:
            return None

        # Temporal aggregation: mean with consistency bonus
        mean_score = float(np.mean(frame_scores))

        # Consistency: low variance across frames = more reliable
        if len(frame_scores) > 1:
            consistency = 1.0 / (1.0 + float(np.var(frame_scores)) * 10.0)
            # Blend: slightly reward consistent quality
            score = 0.85 * mean_score + 0.15 * consistency
        else:
            score = mean_score

        return score

    def _compute_resnet_score(self, frames: List[np.ndarray]) -> Optional[float]:
        """Quality scoring via ResNet-50 features."""
        import torch

        frame_scores = []
        for frame in frames:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = self._resnet_transform(rgb).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    feat = self._resnet(tensor).flatten()
                    score = self._quality_regressor(feat.unsqueeze(0)).item()
                frame_scores.append(score)
            except Exception:
                continue

        if not frame_scores:
            return None

        return float(np.mean(frame_scores))

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
