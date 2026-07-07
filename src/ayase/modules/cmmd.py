"""CMMD distributional image quality using CLIP-space MMD.

Computes a dataset-level Maximum Mean Discrepancy between generated and
reference image distributions using CLIP ViT-L/14@336 embeddings. When the
CLIP backend cannot be loaded the metric is left unset (no proxy features).
Lower scores indicate closer distributions.
"""

import logging
from typing import List, Optional

import numpy as np

from ayase.base_modules import BatchMetricModule
from ayase.image import arrays_to_pil, load_representative_frame
from ayase.models import Sample
from ayase.runtime import cached_clip_image_features, media_state_key

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
        self._backend = "unavailable"
        self._model = None
        self._processor = None
        self._device = "cpu"

    def setup(self) -> None:
        try:
            from transformers import CLIPModel, CLIPProcessor

            from ayase.config import resolve_model_path
            from ayase.runtime import (
                from_pretrained_with_attention,
                resolve_torch_device,
                shared_runtime_resource,
            )

            self._device = resolve_torch_device(self.device_config)

            models_dir = self.config.get("models_dir", "models")
            resolved = resolve_model_path(self.model_name, models_dir)

            def load_clip():
                model = from_pretrained_with_attention(
                    CLIPModel,
                    resolved,
                    self.config,
                    device=self._device,
                    use_safetensors=True,
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
                    "safetensors",
                ),
                load_clip,
            )
            self._backend = "clip"
            logger.info("CMMD initialised with %s on %s", self.model_name, self._device)
        except ImportError:
            logger.warning("CMMD unavailable: requires torch and transformers (CLIP); metric skipped")
        except Exception as e:
            logger.warning("CMMD CLIP setup failed (%s); metric skipped", e)

    def extract_features(self, sample: Sample) -> Optional[np.ndarray]:
        if self._backend != "clip" or self._model is None:
            return None
        frame = load_representative_frame(sample.path, color="rgb")
        if frame is None:
            return None
        return self._extract_clip(frame, cache_key=("cmmd", media_state_key(sample.path)))

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

    def _extract_clip(self, frame: np.ndarray, cache_key: Optional[tuple] = None) -> Optional[np.ndarray]:
        try:
            emb = cached_clip_image_features(
                self,
                self._model,
                self._processor,
                arrays_to_pil([frame]),
                model_key=self.model_name,
                device=self._device,
                cache_key=cache_key or ("cmmd_frame", frame.shape, frame.tobytes()),
            )
            return emb.squeeze(0).detach().float().cpu().numpy().astype(np.float64)
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
