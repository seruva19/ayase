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

from ayase.models import Sample
from ayase.base_modules import BatchMetricModule
from ayase.compat import extract_features

logger = logging.getLogger(__name__)


class GenerativeDistributionModule(BatchMetricModule):
    name = "generative_distribution"
    description = "Precision / Recall / Coverage / Density (batch metric)"
    default_config = {
        "k": 5,  # Neighbours for manifold estimation
        "device": "auto",
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
        self.device = None
        self._ml_available = False
        self._clip_model = None
        self._clip_processor = None

    def setup(self) -> None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            if self.device_config == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(self.device_config)

            self._clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            ).to(self.device)
            self._clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self._clip_model.eval()
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
            import torch

            # Load a representative frame
            if sample.is_video:
                cap = cv2.VideoCapture(str(sample.path))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    return None
            else:
                frame = cv2.imread(str(sample.path))
                if frame is None:
                    return None

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            from PIL import Image
            pil_img = Image.fromarray(rgb)

            inputs = self._clip_processor(images=pil_img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                emb = extract_features(self._clip_model.get_image_features(**inputs))
                emb = emb / emb.norm(dim=-1, keepdim=True)

            return emb.squeeze(0).cpu().numpy()

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
