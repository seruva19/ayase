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

import numpy as np

from ayase.image import arrays_to_pil, sample_frames
from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule
from ayase.runtime import (
    cached_clip_image_feature_groups,
    cached_clip_image_features,
    media_state_key,
)

logger = logging.getLogger(__name__)

# DINOv2 weights come from the ayase-models HF mirror rather than the torch.hub
# entrypoint's fbaipublicfiles original, which is unreliable on some networks and
# bypasses the mirror the rest of the pipeline already relies on. The architecture
# still comes from the (reliable, cacheable) torch.hub repo code. Same file as
# ``spectral`` uses, so the two modules share one download.
_DINOV2_MIRROR_BASE = "https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/"
_DINOV2_MIRROR_RELATIVE = "spectral/dinov2_vits14_pretrain.pth"


class WorldConsistencyModule(PipelineModule):
    name = "world_consistency"
    description = "World Consistency Score: object permanence + causal compliance (2025)"
    default_config = {
        "subsample": 12,
        "permanence_weight": 0.40,
        "stability_weight": 0.30,
        "causal_weight": 0.30,
    }
    metric_groups = {
        "world_consistency_score": "temporal",
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
        self._device = "cpu"

    def setup(self) -> None:
        if self.test_mode:
            return

        # Try DINOv2 first (better for visual consistency tracking)
        if self._try_dinov2_setup():
            return
        # Fallback to CLIP
        self._try_clip_setup()

    def _try_dinov2_setup(self) -> bool:
        """Try DINOv2 for self-supervised feature tracking."""
        try:
            import torch
            import torchvision.transforms as T

            from ayase.config import download_model_file
            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            checkpoint = download_model_file(
                _DINOV2_MIRROR_RELATIVE,
                _DINOV2_MIRROR_BASE + _DINOV2_MIRROR_RELATIVE,
                str(self.config.get("models_dir", "models")),
            )
            self._model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vits14", pretrained=False, verbose=False
            )
            self._model.load_state_dict(torch.load(str(checkpoint), map_location="cpu"))
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
            from transformers import CLIPModel, CLIPProcessor
            from ayase.config import resolve_model_path
            from ayase.runtime import (
                from_pretrained_with_attention,
                resolve_torch_device,
                shared_runtime_resource,
            )

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            model_name = self.config.get("clip_model", "openai/clip-vit-base-patch32")
            models_dir = self.config.get("models_dir", "models")
            resolved = resolve_model_path(model_name, models_dir)

            def load_clip():
                model = from_pretrained_with_attention(
                    CLIPModel,
                    resolved,
                    self.config,
                    device=self._device,
                ).to(self._device).eval()
                processor = CLIPProcessor.from_pretrained(resolved)
                return model, processor

            self._model, self._processor = shared_runtime_resource(
                self,
                (
                    "hf_clip",
                    resolved,
                    self._device,
                    str(self.config.get("attention_backend", "auto")),
                    "default",
                ),
                load_clip,
            )
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

    def process_batch(self, samples: List[Sample]) -> List[Sample]:
        if not self._ml_available:
            return samples
        if self._backend != "clip":
            return super().process_batch(samples)

        try:
            prepared = []
            image_groups = []
            cache_keys = []
            for sample in samples:
                if not sample.is_video:
                    continue
                frames = self._extract_frames(sample)
                if len(frames) < 2:
                    continue
                prepared.append(sample)
                image_groups.append(arrays_to_pil(frames))
                cache_keys.append((self.subsample, media_state_key(sample.path)))

            if not prepared:
                return samples

            model_name = self.config.get("clip_model", "openai/clip-vit-base-patch32")
            feature_groups = cached_clip_image_feature_groups(
                self,
                self._model,
                self._processor,
                image_groups,
                model_key=model_name,
                device=self._device,
                cache_keys=cache_keys,
            )
            for sample, embeddings in zip(prepared, feature_groups):
                score = self._compute_consistency_from_embeddings(embeddings)
                if score is None:
                    continue
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.world_consistency_score = score
        except Exception as e:
            logger.warning("WorldConsistency batch failed: %s", e)

        return samples

    def _encode_frames(self, frames: List[np.ndarray]):
        """Encode frames to normalised embeddings using the active backend."""
        import torch
        pil_frames = arrays_to_pil(frames)

        embeds = []
        with torch.no_grad():
            if self._backend == "dinov2":
                for pil_img in pil_frames:
                    tensor = self._transform(pil_img).unsqueeze(0).to(self._device)
                    emb = self._model(tensor)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                    embeds.append(emb)
            else:  # clip
                model_name = self.config.get("clip_model", "openai/clip-vit-base-patch32")
                return cached_clip_image_features(
                    self,
                    self._model,
                    self._processor,
                    pil_frames,
                    model_key=model_name,
                    device=self._device,
                    cache_key=("direct", self.subsample, len(frames)),
                )

        return embeds

    def _compute_consistency(self, frames: List[np.ndarray]) -> Optional[float]:
        """Compute world consistency from frame embeddings."""
        import torch

        embeds = self._encode_frames(frames)
        if self._backend == "clip":
            return self._compute_consistency_from_embeddings(embeds)
        if len(embeds) < 2:
            return None
        frame_stack = torch.cat(embeds, dim=0)  # (N, D)
        return self._compute_consistency_from_embeddings(frame_stack)

    def _compute_consistency_from_embeddings(self, frame_stack) -> Optional[float]:
        """Compute world consistency from a normalized frame embedding tensor."""
        import torch

        if frame_stack is None or frame_stack.size(0) < 2:
            return None

        # --- Object Permanence: consecutive cosine similarity ---
        consec_sims = []
        for i in range(frame_stack.size(0) - 1):
            sim = torch.nn.functional.cosine_similarity(
                frame_stack[i].unsqueeze(0), frame_stack[i + 1].unsqueeze(0)
            ).item()
            consec_sims.append(sim)
        permanence = float(np.mean(consec_sims))

        # --- Relation Stability: low variance in pairwise similarities ---
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
        try:
            return sample_frames(sample.path, max_frames=self.subsample, color="rgb")
        except Exception as e:
            logger.debug("WorldConsistency frame loading failed for %s: %s", sample.path, e)
            return []
