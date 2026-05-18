"""Generative Distribution Metrics module (Precision, Recall, Coverage, Density).

These batch-level metrics evaluate how well a generative model's
output distribution matches a reference distribution:

  Precision — fraction of generated samples that fall within the
              real data manifold (higher = fewer bad samples).
  Recall    — fraction of the real data manifold covered by
              generated samples (higher = fewer missing modes).
  Coverage  — fraction of real samples that have at least one
              generated neighbour nearby (diversity measure).
  Density   — average number of generated samples around each
              real sample, normalised (concentration measure).

Algorithm (k-NN manifold estimation, Kynkäänniemi et al. 2019):
  1. Extract CLIP image embeddings for every sample.
  2. Build k-NN graphs in feature space.
  3. Compute manifold membership via hypersphere radii.

This is a dataset-level (batch) metric, not per-sample.
"""

import logging
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np

from ayase.image import arrays_to_pil, load_representative_frame
from ayase.models import Sample
from ayase.base_modules import BatchMetricModule
from ayase.runtime import cached_clip_image_features, media_state_key

logger = logging.getLogger(__name__)


class GenerativeDistributionModule(BatchMetricModule):
    name = "generative_distribution"
    description = "Precision / Recall / Coverage / Density (batch metric)"
    default_config = {
        "k": 5,  # Neighbours for manifold estimation
        "device": "auto",
        "clip_model": "openai/clip-vit-base-patch32",
    }
    metric_info = {
        "precision": "Generated-sample precision against the real manifold (0-1, higher=better)",
        "recall": "Real-distribution coverage by generated samples (0-1, higher=better)",
        "coverage": "Fraction of real samples covered by generated neighbours (0-1, higher=better)",
        "density": "Average normalized generated-sample density around real samples",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.k = self.config.get("k", 5)
        self.device_config = self.config.get("device", "auto")
        self.clip_model_name = self.config.get("clip_model", "openai/clip-vit-base-patch32")
        self.device = None
        self._ml_available = False
        self._clip_model = None
        self._clip_processor = None

    def setup(self) -> None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            from ayase.config import resolve_model_path
            from ayase.runtime import (
                from_pretrained_with_attention,
                resolve_torch_device,
                shared_runtime_resource,
            )

            device = resolve_torch_device(self.device_config)
            self.device = torch.device(device)
            models_dir = self.config.get("models_dir", "models")
            resolved = resolve_model_path(self.clip_model_name, models_dir)

            def load_clip():
                model = from_pretrained_with_attention(
                    CLIPModel,
                    resolved,
                    self.config,
                    device=device,
                ).to(self.device).eval()
                processor = CLIPProcessor.from_pretrained(resolved)
                return model, processor

            self._clip_model, self._clip_processor = shared_runtime_resource(
                self,
                (
                    "hf_clip",
                    resolved,
                    device,
                    str(self.config.get("attention_backend", "auto")),
                    "default",
                ),
                load_clip,
            )
            self._ml_available = True
            logger.info(f"Generative distribution metrics initialised on {self.device}")

        except ImportError:
            logger.warning("transformers not installed")
        except Exception as e:
            logger.warning(f"Failed to setup generative distribution metrics: {e}")

    def extract_features(self, sample: Sample) -> Optional[np.ndarray]:
        """Extract CLIP image embedding from a sample."""
        if self._clip_model is None:
            return None

        try:
            rgb = load_representative_frame(sample.path, color="rgb")
            if rgb is None:
                return None
            emb = cached_clip_image_features(
                self,
                self._clip_model,
                self._clip_processor,
                arrays_to_pil([rgb]),
                model_key=self.clip_model_name,
                device=self.device,
                cache_key=("generative_distribution", media_state_key(sample.path)),
            )
            return emb.squeeze(0).detach().float().cpu().numpy()

        except Exception as e:
            logger.debug(f"Feature extraction failed for {sample.path}: {e}")
            return None

    def compute_distribution_metric(
        self, features: List[np.ndarray], reference_features: Optional[List[np.ndarray]] = None
    ) -> float:
        """Compute precision, recall, coverage, density."""
        from sklearn.neighbors import NearestNeighbors

        gen = np.stack(features)

        if reference_features and len(reference_features) > 0:
            real = np.stack(reference_features)
        else:
            # Split generated features in half as proxy
            mid = len(gen) // 2
            real = gen[:mid]
            gen = gen[mid:]

        n_real, n_gen = len(real), len(gen)
        k = min(self.k, n_real - 1, n_gen - 1)
        if k < 1:
            return 0.0

        # k-NN for real data
        nn_real = NearestNeighbors(n_neighbors=k, metric="euclidean")
        nn_real.fit(real)
        real_dists, _ = nn_real.kneighbors(real)
        real_radii = real_dists[:, -1]  # k-th neighbour distance

        # k-NN for generated data
        nn_gen = NearestNeighbors(n_neighbors=k, metric="euclidean")
        nn_gen.fit(gen)
        gen_dists, _ = nn_gen.kneighbors(gen)
        gen_radii = gen_dists[:, -1]

        # Precision: fraction of gen that falls within real manifold
        gen_to_real, _ = nn_real.kneighbors(gen)
        gen_to_real_min = gen_to_real[:, 0]
        # gen sample is "in real manifold" if nearest real neighbour
        # is within that real sample's radius
        gen_nn_idx = nn_real.kneighbors(gen, n_neighbors=1, return_distance=False).flatten()
        precision = float(np.mean(gen_to_real_min <= real_radii[gen_nn_idx]))

        # Recall: fraction of real that falls within gen manifold
        real_to_gen, _ = nn_gen.kneighbors(real)
        real_to_gen_min = real_to_gen[:, 0]
        real_nn_idx = nn_gen.kneighbors(real, n_neighbors=1, return_distance=False).flatten()
        recall = float(np.mean(real_to_gen_min <= gen_radii[real_nn_idx]))

        # Coverage: fraction of real samples that have ≥1 gen neighbour
        # within their radius
        coverage = float(np.mean(real_to_gen_min <= real_radii))

        # Density: avg number of gen samples in each real ball / k
        gen_to_real_all, _ = nn_real.kneighbors(gen)
        density_counts = []
        for i in range(n_real):
            inside = np.sum(gen_to_real_all[:, 0] <= real_radii[i])
            density_counts.append(inside)
        density = float(np.mean(density_counts) / k)

        # Store all metrics via pipeline
        if hasattr(self, "pipeline") and self.pipeline:
            if hasattr(self.pipeline, "add_dataset_metric"):
                self.pipeline.add_dataset_metric("precision", precision)
                self.pipeline.add_dataset_metric("recall", recall)
                self.pipeline.add_dataset_metric("coverage", coverage)
                self.pipeline.add_dataset_metric("density", density)

        logger.info(
            f"Generative metrics: P={precision:.3f} R={recall:.3f} "
            f"C={coverage:.3f} D={density:.3f}"
        )

        # Return precision as the "primary" score
        return precision

    def on_dispose(self) -> None:
        if len(self._feature_cache) < 4:
            logger.info(
                f"Generative metrics: not enough samples "
                f"({len(self._feature_cache)})"
            )
            self._feature_cache = []
            self._reference_cache = []
            return

        try:
            self.compute_distribution_metric(
                self._feature_cache,
                self._reference_cache if self._reference_cache else None,
            )
        except Exception as e:
            logger.error(f"Generative distribution metrics failed: {e}")
        finally:
            self._feature_cache = []
            self._reference_cache = []


class GenerativeDistributionCompatModule(GenerativeDistributionModule):
    """Compatibility alias matching filename-based discovery."""

    name = "generative_distribution_metrics"
