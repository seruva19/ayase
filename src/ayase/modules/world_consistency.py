"""World Consistency Score --- Object Permanence + Causal Compliance (2025).

Paper: https://arxiv.org/abs/2508.00144

Uses DINOv2 (preferred) or CLIP to track visual consistency across video
frames.  Measures embedding drift and object permanence by computing
how stable deep features remain over time.

Sub-scores:
  - Object permanence: mean cosine similarity of consecutive frame embeddings
  - Relation stability: low variance in pairwise embedding similarities
  - Causal compliance: smooth monotonic drift (not abrupt jumps)

world_consistency_score --- higher = better (0-1 range)
"""

import logging
from typing import List, Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class WorldConsistencyModule(PipelineModule):
    name = "world_consistency"
    description = "World Consistency Score: object permanence + causal compliance (2025)"
    default_config = {
        "subsample": 12,
        "permanence_weight": 0.40,
        "stability_weight": 0.30,
        "causal_weight": 0.30,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 12)
        self._w_permanence = self.config.get("permanence_weight", 0.40)
        self._w_stability = self.config.get("stability_weight", 0.30)
        self._w_causal = self.config.get("causal_weight", 0.30)
        self._ml_available = False
        self._backend = None
        self._model = None
        self._processor = None
        self._transform = None
        self._device = None

    def setup(self) -> None:
        if self.test_mode:
            return

        # Try DINOv2 first (better for visual consistency tracking)
        if self._try_dinov2_setup():
            return
        # Fallback to CLIP
        self._try_clip_setup()

    def _try_dinov2_setup(self) -> bool:
        """Try DINOv2 via torch.hub for self-supervised feature tracking."""
        try:
            import torch
            import torchvision.transforms as T

            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self._model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vits14", verbose=False
            )
            self._model.to(self._device).eval()
            self._transform = T.Compose([
                T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self._backend = "dinov2"
            self._ml_available = True
            logger.info("WorldConsistency (DINOv2) initialised on %s", self._device)
            return True
        except (ImportError, Exception) as e:
            logger.debug("DINOv2 not available: %s", e)
            return False

    def _try_clip_setup(self) -> bool:
        """Fallback: use CLIP for frame feature tracking."""
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self._model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self._processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self._model.to(self._device).eval()
            self._backend = "clip"
            self._ml_available = True
            logger.info("WorldConsistency (CLIP) initialised on %s", self._device)
            return True
        except (ImportError, Exception) as e:
            logger.warning("WorldConsistency setup failed: %s", e)
            return False

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.is_video:
            return sample
        try:
            frames = self._extract_frames(sample)
            if len(frames) < 2:
                return sample

            score = self._compute_consistency(frames)
            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.world_consistency_score = score
            logger.debug("WorldConsistency for %s: %.4f", sample.path.name, score)

        except Exception as e:
            logger.warning("WorldConsistency failed for %s: %s", sample.path, e)
        return sample

    def _encode_frames(self, frames: List[np.ndarray]):
        """Encode frames to normalised embeddings using the active backend."""
        import torch
        from PIL import Image

        pil_frames = [
            Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames
        ]

        embeds = []
        with torch.no_grad():
            if self._backend == "dinov2":
                for pil_img in pil_frames:
                    tensor = self._transform(pil_img).unsqueeze(0).to(self._device)
                    emb = self._model(tensor)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                    embeds.append(emb)
            else:  # clip
                for pil_img in pil_frames:
                    inputs = self._processor(
                        images=pil_img, return_tensors="pt"
                    ).to(self._device)
                    emb = self._model.get_image_features(**inputs)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                    embeds.append(emb)

        return embeds

    def _compute_consistency(self, frames: List[np.ndarray]) -> Optional[float]:
        """Compute world consistency from frame embeddings."""
        import torch

        embeds = self._encode_frames(frames)
        if len(embeds) < 2:
            return None

        # --- Object Permanence: consecutive cosine similarity ---
        consec_sims = []
        for i in range(len(embeds) - 1):
            sim = torch.nn.functional.cosine_similarity(
                embeds[i], embeds[i + 1]
            ).item()
            consec_sims.append(sim)
        permanence = float(np.mean(consec_sims))

        # --- Relation Stability: low variance in pairwise similarities ---
        frame_stack = torch.cat(embeds, dim=0)  # (N, D)
        sim_matrix = (frame_stack @ frame_stack.T).cpu().numpy()
        n = sim_matrix.shape[0]
        # Extract upper triangle (excluding diagonal)
        triu_indices = np.triu_indices(n, k=1)
        pairwise_sims = sim_matrix[triu_indices]
        stability_var = float(np.var(pairwise_sims))
        # Low variance = high stability; map to 0-1
        stability = 1.0 / (1.0 + stability_var * 100.0)

        # --- Causal Compliance: smoothness of embedding drift ---
        # Check that consecutive similarities don't have abrupt drops
        if len(consec_sims) > 1:
            diffs = np.diff(consec_sims)
            # Penalise large negative drops (sudden inconsistency)
            drop_penalty = np.sum(np.maximum(-diffs, 0) ** 2)
            causal = float(1.0 / (1.0 + drop_penalty * 50.0))
        else:
            causal = float(max(consec_sims[0], 0.0))

        score = (
            self._w_permanence * permanence
            + self._w_stability * stability
            + self._w_causal * causal
        )
        return float(np.clip(score, 0.0, 1.0))

    def _extract_frames(self, sample: Sample) -> List[np.ndarray]:
        frames = []
        cap = cv2.VideoCapture(str(sample.path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 1:
            cap.release()
            return []
        indices = np.linspace(
            0, total - 1, min(self.subsample, total), dtype=int
        )
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, f = cap.read()
            if ret:
                frames.append(f)
        cap.release()
        return frames
