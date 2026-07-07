"""CLIP Image Similarity — reference-based image-to-image cosine similarity.

Computes cosine similarity between CLIP image embeddings of ``sample.path``
and ``sample.reference_path``. Equivalent to the "CLIP-I" metric used in
image-to-image generation benchmarks (DreamBooth, IP-Adapter, InstantID).

Distinguishes from ``semantic_alignment`` (which is image-text CLIPScore
between the image and ``sample.caption``).

Output:
    clip_image_similarity — cosine similarity 0-1 (higher = closer match).
    Mathematically the cosine is in [-1, 1]; for L2-normalised CLIP image
    embeddings of natural images it is almost always positive.

Requires ``sample.reference_path`` pointing to a reference image. Gracefully
skips (no field written) when no reference is provided, when the file cannot
be loaded, or when the backend cannot be initialised.

Backends:
    1. open_clip (default) — ViT-B/32 with laion2b_s34b_b79k weights, matches
       the legacy ``clip_similarity_i2i`` numerical regime used by downstream
       image-to-image pipelines.
    2. transformers (CLIPModel + CLIPProcessor) — fallback when open_clip is
       unavailable; auto-selected if ``model_name`` does not start with
       ``open_clip:``.

Aliases for backwards compatibility: pipelines that previously consumed
``qm.clip_score`` for image-image similarity can still read the same value
from the dedicated ``qm.clip_image_similarity`` field.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class CLIPImageSimilarityModule(PipelineModule):
    name = "clip_image_similarity"
    description = (
        "CLIP image-to-image cosine similarity vs reference image (CLIP-I)"
    )
    default_config = {
        "backend": "auto",  # auto | open_clip | transformers
        "model_name": "open_clip:ViT-B-32",
        "pretrained": "laion2b_s34b_b79k",
        "device": "auto",
        "warning_threshold": 0.5,
    }
    metric_info = {
        "clip_image_similarity": (
            "Cosine similarity between CLIP image embeddings of sample and "
            "reference image (0-1, higher = closer match)"
        ),
    }
    metric_groups = {
        "clip_image_similarity": "alignment",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self.backend = self.config.get("backend", "auto")
        self.model_name = self.config.get("model_name", "open_clip:ViT-B-32")
        self.pretrained = self.config.get("pretrained", "laion2b_s34b_b79k")
        self.device_config = self.config.get("device", "auto")
        self.warning_threshold = float(self.config.get("warning_threshold", 0.5))

        self._device = "cpu"
        self._ml_available = False
        self._active_backend: Optional[str] = None
        self._backend = "unavailable"
        self._model = None
        self._preprocess = None
        self._processor = None

    def setup(self) -> None:
        if self.backend in ("auto", "open_clip") or (
            self.backend == "auto" and str(self.model_name).startswith("open_clip:")
        ):
            if self._setup_open_clip():
                return
            if self.backend == "open_clip":
                return

        self._setup_transformers()

    def _setup_open_clip(self) -> bool:
        try:
            import torch
            import open_clip
        except ImportError:
            logger.warning(
                "open_clip_torch not installed; CLIP-I open_clip backend disabled"
            )
            return False

        try:
            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.device_config)
            model_name = str(self.model_name)
            if model_name.startswith("open_clip:"):
                model_name = model_name.split(":", 1)[1]
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=self.pretrained,
                device=self._device,
            )
            self._model.eval()
            self._ml_available = True
            self._active_backend = "open_clip"
            self._backend = "open_clip"
            logger.info(
                "CLIPImageSimilarity initialised with OpenCLIP %s/%s on %s",
                model_name,
                self.pretrained,
                self._device,
            )
            return True
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("CLIPImageSimilarity OpenCLIP setup failed: %s", e)
            return False

    def _setup_transformers(self) -> None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            from ayase.runtime import resolve_torch_device
        except ImportError:
            logger.warning(
                "transformers/torch not installed; CLIP-I transformers backend disabled"
            )
            return

        try:
            self._device = resolve_torch_device(self.device_config)
            model_id = str(self.model_name)
            if model_id.startswith("open_clip:"):
                # Reasonable HF fallback when open_clip is not available
                model_id = "openai/clip-vit-base-patch32"
            self._model = CLIPModel.from_pretrained(model_id).to(self._device).eval()
            self._processor = CLIPProcessor.from_pretrained(model_id)
            self._ml_available = True
            self._active_backend = "transformers"
            self._backend = "transformers"
            logger.info(
                "CLIPImageSimilarity initialised with transformers %s on %s",
                model_id,
                self._device,
            )
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("CLIPImageSimilarity transformers setup failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        ref_path = getattr(sample, "reference_path", None)
        if ref_path is None:
            return sample
        ref_path = Path(ref_path)
        if not ref_path.exists():
            logger.debug("CLIP-I reference not found: %s", ref_path)
            return sample

        try:
            sample_img = self._load_image(sample.path)
            if sample_img is None:
                return sample
            ref_img = self._load_image(ref_path)
            if ref_img is None:
                return sample

            score = self._compute_similarity(sample_img, ref_img)
            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.clip_image_similarity = float(score)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("CLIP-I failed for %s: %s", sample.path, e)

        return sample

    def _load_image(self, path: Path) -> Optional[Image.Image]:
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:  # pylint: disable=broad-except
            logger.debug("Failed to load image %s: %s", path, e)
            return None

    def _compute_similarity(
        self, sample_img: Image.Image, ref_img: Image.Image
    ) -> Optional[float]:
        import torch

        if self._active_backend == "open_clip":
            with torch.no_grad():
                tensors = torch.stack(
                    [self._preprocess(sample_img), self._preprocess(ref_img)], dim=0
                ).to(self._device)
                feats = self._model.encode_image(tensors)
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                cos = float((feats[0] @ feats[1]).item())
                return float(np.clip(cos, -1.0, 1.0))

        if self._active_backend == "transformers":
            with torch.no_grad():
                inputs = self._processor(
                    images=[sample_img, ref_img], return_tensors="pt"
                ).to(self._device)
                feats = self._model.get_image_features(**inputs)
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                cos = float((feats[0] @ feats[1]).item())
                return float(np.clip(cos, -1.0, 1.0))

        return None
