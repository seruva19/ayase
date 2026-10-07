"""Compare generated and reference image sets with PRDC neighbourhood statistics.

The dataset-level outputs are precision, recall, density, and coverage; larger
values mean greater generated-set inclusion or reference-set coverage under the
corresponding PRDC definition. Each sample contributes one representative image
or video frame. DINOv2 ViT-L/14 features are required — without the encoder or
without a reference set the module emits no metrics: image-statistics features
are not comparable with DINOv2 results, and splitting one input set in half is
self-comparison, not PRDC. Results require enough samples for the configured
k-nearest-neighbour estimate and are distribution diagnostics, not per-sample
quality or identity scores.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

from ayase.base_modules import BatchMetricModule
from ayase.image import load_representative_frame
from ayase.models import Sample

logger = logging.getLogger(__name__)


class PRDCDINOv2Module(BatchMetricModule):
    name = "prdc_dinov2"
    provenance = "published"
    sources = {
        "prdc_coverage": "PRDC (Naeem et al., ICML 2020) on DINOv2 features (Stein et al., NeurIPS 2023) — https://github.com/clovaai/generative-evaluation-prdc",
        "prdc_density": "PRDC density (Naeem et al., ICML 2020) — https://github.com/clovaai/generative-evaluation-prdc",
        "prdc_precision": "PRDC (Naeem et al., ICML 2020) on DINOv2 features (Stein et al., NeurIPS 2023) — https://github.com/clovaai/generative-evaluation-prdc",
        "prdc_recall": "PRDC (Naeem et al., ICML 2020) on DINOv2 features (Stein et al., NeurIPS 2023) — https://github.com/clovaai/generative-evaluation-prdc",
    }
    description = "PRDC precision/recall/density/coverage over DINOv2 image features"
    default_config = {
        "model_name": "facebook/dinov2-large",
        "device": "auto",
        "k": 5,
    }
    models = [
        {
            "id": "facebook/dinov2-large",
            "type": "huggingface",
            "task": "DINOv2 image encoder for PRDC features",
        },
    ]
    metric_info = {
        "prdc_precision": "Fraction of generated samples inside the reference manifold (0-1)",
        "prdc_recall": "Fraction of reference samples covered by generated samples (0-1)",
        "prdc_density": "Average generated-sample density around reference samples",
        "prdc_coverage": "Fraction of reference samples with a generated neighbour in range (0-1)",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "facebook/dinov2-large")
        self.device_config = self.config.get("device", "auto")
        self.k = self.config.get("k", 5)
        self._backend = "unavailable"
        self._model = None
        self._processor = None
        self._device = "cpu"

    def setup(self) -> None:
        try:
            from transformers import AutoImageProcessor, AutoModel

            from ayase.config import resolve_model_path
            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.device_config)

            models_dir = self.config.get("models_dir", "models")
            resolved = resolve_model_path(self.model_name, models_dir)
            self._processor = AutoImageProcessor.from_pretrained(resolved)
            self._model = AutoModel.from_pretrained(resolved).to(self._device).eval()
            self._backend = "dinov2"
            logger.info("PRDC-DINOv2 initialised with %s on %s", self.model_name, self._device)
        except Exception as e:
            logger.warning("PRDC-DINOv2 unavailable — DINOv2 setup failed: %s", e)

    def extract_features(self, sample: Sample) -> Optional[np.ndarray]:
        if self._backend != "dinov2" or self._model is None:
            return None
        frame = load_representative_frame(sample.path, color="rgb")
        if frame is None:
            return None
        return self._extract_dinov2(frame)

    def compute_distribution_metric(
        self,
        features: List[np.ndarray],
        reference_features: Optional[List[np.ndarray]] = None,
    ) -> Optional[float]:
        gen = np.stack(features).astype(np.float64)
        if not reference_features:
            # PRDC needs a real reference set; self-splitting is not PRDC.
            return None
        real = np.stack(reference_features).astype(np.float64)

        scores = self._compute_prdc(real, gen)
        if scores is None:
            return None
        precision, recall, density, coverage = scores
        if hasattr(self, "pipeline") and self.pipeline and hasattr(self.pipeline, "add_dataset_metric"):
            self.pipeline.add_dataset_metric("prdc_precision", precision)
            self.pipeline.add_dataset_metric("prdc_recall", recall)
            self.pipeline.add_dataset_metric("prdc_density", density)
            self.pipeline.add_dataset_metric("prdc_coverage", coverage)
        logger.info(
            "PRDC-DINOv2: precision=%.3f recall=%.3f density=%.3f coverage=%.3f",
            precision,
            recall,
            density,
            coverage,
        )
        return precision

    def on_dispose(self) -> None:
        if len(self._feature_cache) < 2 or not self._reference_cache:
            self._feature_cache = []
            self._reference_cache = []
            return
        try:
            self.compute_distribution_metric(
                self._feature_cache,
                self._reference_cache if self._reference_cache else None,
            )
        except Exception as e:
            logger.error("PRDC-DINOv2 failed: %s", e)
        finally:
            self._feature_cache = []
            self._reference_cache = []

    def _extract_dinov2(self, frame: np.ndarray) -> Optional[np.ndarray]:
        try:
            import torch
            from PIL import Image

            image = Image.fromarray(frame.astype(np.uint8)).convert("RGB")
            inputs = self._processor(images=image, return_tensors="pt").to(self._device)
            with torch.no_grad():
                outputs = self._model(**inputs)
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is None:
                pooled = outputs.last_hidden_state[:, 0]
            pooled = pooled / pooled.norm(dim=-1, keepdim=True)
            return pooled.squeeze(0).cpu().numpy().astype(np.float64)
        except Exception as e:
            logger.debug("PRDC-DINOv2 feature extraction failed: %s", e)
            return None

    def _compute_prdc(
        self, real: np.ndarray, gen: np.ndarray
    ) -> Optional[Tuple[float, float, float, float]]:
        if len(real) < 2 or len(gen) < 2:
            return None

        k_real = min(int(self.k), len(real) - 1)
        k_gen = min(int(self.k), len(gen) - 1)
        real_radii = _kth_radii(real, real, k_real, exclude_self=True)
        gen_radii = _kth_radii(gen, gen, k_gen, exclude_self=True)
        d_rg = _euclidean_distances(real, gen)

        # Official PRDC (Naeem et al., ICML 2020) formulas: strict inequalities,
        # density = per-fake count of containing real balls, divided by k.
        precision = float(np.mean(np.any(d_rg < real_radii[:, None], axis=0)))
        recall = float(np.mean(np.any(d_rg < gen_radii[None, :], axis=1)))
        density = float(np.mean(np.sum(d_rg < real_radii[:, None], axis=0) / max(k_real, 1)))
        coverage = float(np.mean(np.min(d_rg, axis=1) < real_radii))
        return precision, recall, density, coverage


def _kth_radii(x: np.ndarray, y: np.ndarray, k: int, exclude_self: bool = False) -> np.ndarray:
    d = _euclidean_distances(x, y)
    if exclude_self and x.shape == y.shape and np.allclose(x, y):
        np.fill_diagonal(d, np.inf)
    k = max(1, min(k, d.shape[1]))
    return np.partition(d, kth=k - 1, axis=1)[:, k - 1]


def _euclidean_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x2 = np.sum(x * x, axis=1, keepdims=True)
    y2 = np.sum(y * y, axis=1, keepdims=True).T
    return np.sqrt(np.maximum(x2 + y2 - 2.0 * (x @ y.T), 0.0))
