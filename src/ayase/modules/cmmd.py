"""CMMD distributional image quality using CLIP-space MMD.

Computes a dataset-level Maximum Mean Discrepancy between generated and
reference image distributions. The preferred backend is CLIP ViT-L/14@336; a
deterministic image-statistics proxy is used when model weights are unavailable.
Lower scores indicate closer distributions.
"""

import logging
from typing import List, Optional

import numpy as np

from ayase.base_modules import BatchMetricModule
from ayase.compat import extract_features
from ayase.image import image_stats_features, load_representative_frame
from ayase.models import Sample

logger = logging.getLogger(__name__)


class CMMDModule(BatchMetricModule):
    name = "cmmd"
    description = "CLIP Maximum Mean Discrepancy for image generation (batch metric)"
    default_config = {
        "model_name": "openai/clip-vit-large-patch14-336",
        "device": "auto",
        "kernel": "rbf",
        "sigma": None,
    }
    models = [
        {
            "id": "openai/clip-vit-large-patch14-336",
            "type": "huggingface",
            "task": "CLIP image encoder for CMMD features",
        },
    ]
    metric_info = {
        "cmmd": "CLIP Maximum Mean Discrepancy between generated and reference sets (lower=better)",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "openai/clip-vit-large-patch14-336")
        self.device_config = self.config.get("device", "auto")
        self.kernel = self.config.get("kernel", "rbf")
        self.sigma = self.config.get("sigma", None)
        self._backend = "image_stats"
        self._model = None
        self._processor = None
        self._device = "cpu"

    def setup(self) -> None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            from ayase.config import resolve_model_path

            if self.device_config == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = self.device_config

            models_dir = self.config.get("models_dir", "models")
            resolved = resolve_model_path(self.model_name, models_dir)
            self._model = CLIPModel.from_pretrained(resolved, use_safetensors=True).to(self._device)
            self._processor = CLIPProcessor.from_pretrained(resolved)
            self._model.eval()
            self._backend = "clip"
            logger.info("CMMD initialised with %s on %s", self.model_name, self._device)
        except ImportError:
            logger.warning("CMMD CLIP backend requires torch and transformers; using image-stat proxy")
        except Exception as e:
            logger.warning("CMMD CLIP setup failed (%s); using image-stat proxy", e)

    def extract_features(self, sample: Sample) -> Optional[np.ndarray]:
        frame = load_representative_frame(sample.path, color="rgb")
        if frame is None:
            return None
        if self._backend == "clip" and self._model is not None:
            return self._extract_clip(frame)
        return image_stats_features(frame)

    def compute_distribution_metric(
        self,
        features: List[np.ndarray],
        reference_features: Optional[List[np.ndarray]] = None,
    ) -> float:
        gen = np.stack(features).astype(np.float64)
        if reference_features:
            ref = np.stack(reference_features).astype(np.float64)
        else:
            mid = len(gen) // 2
            if mid < 1:
                return float("inf")
            ref = gen[:mid]
            gen = gen[mid:]
        if len(gen) < 1 or len(ref) < 1:
            return float("inf")
        return float(max(self._mmd2(gen, ref), 0.0))

    def _extract_clip(self, frame: np.ndarray) -> Optional[np.ndarray]:
        try:
            import torch
            from PIL import Image

            image = Image.fromarray(frame.astype(np.uint8)).convert("RGB")
            inputs = self._processor(images=image, return_tensors="pt").to(self._device)
            with torch.no_grad():
                emb = extract_features(self._model.get_image_features(**inputs))
                emb = emb / emb.norm(dim=-1, keepdim=True)
            return emb.squeeze(0).cpu().numpy().astype(np.float64)
        except Exception as e:
            logger.debug("CMMD CLIP extraction failed: %s", e)
            return None

    def _mmd2(self, x: np.ndarray, y: np.ndarray) -> float:
        kxx = self._kernel(x, x)
        kyy = self._kernel(y, y)
        kxy = self._kernel(x, y)
        return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())

    def _kernel(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.kernel == "linear":
            return x @ y.T
        d2 = _squared_distances(x, y)
        sigma = self.sigma
        if sigma is None:
            positive = d2[d2 > 0]
            sigma = float(np.sqrt(np.median(positive))) if positive.size else 1.0
        gamma = 1.0 / (2.0 * max(float(sigma) ** 2, 1e-12))
        return np.exp(-gamma * d2)


def _squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x2 = np.sum(x * x, axis=1, keepdims=True)
    y2 = np.sum(y * y, axis=1, keepdims=True).T
    return np.maximum(x2 + y2 - 2.0 * (x @ y.T), 0.0)
