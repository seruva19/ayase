"""Dataset-Level Analytics module.

Computes batch-level diversity, coverage, outlier, and duplicate
statistics across the entire dataset:

  diversity_score     — visual diversity 0-1 (higher=more diverse)
  semantic_coverage   — embedding space coverage 0-1
  outlier_count       — number of statistical outliers
  class_balance_score — category balance 0-1 (higher=more balanced)
  duplicate_pairs     — count of near-duplicate sample pairs

Uses perceptual hashes (dHash) for duplicate detection and CLIP
embeddings for diversity / coverage analysis.  Falls back to
simpler colour histograms if CLIP is unavailable.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ayase.image import arrays_to_pil, load_representative_frame
from ayase.models import Sample
from ayase.base_modules import BatchMetricModule
from ayase.runtime import cached_clip_image_features, media_state_key

logger = logging.getLogger(__name__)


class DatasetAnalyticsModule(BatchMetricModule):
    name = "dataset_analytics"
    description = "Dataset-level diversity, coverage, outliers, duplicates"
    default_config = {
        "duplicate_threshold": 5,  # Hamming distance for dHash match
        "outlier_iqr_factor": 1.5,  # IQR multiplier for outlier detection
        "device": "auto",
    }
    metric_info = {
        "diversity_score": "Dataset visual diversity score (0-1, higher=more diverse)",
        "semantic_coverage": "Embedding-space coverage score (0-1, higher=more coverage)",
        "outlier_count": "Number of statistical outliers detected in the dataset",
        "duplicate_pairs": "Count of near-duplicate sample pairs",
        "class_balance_score": "Class/category balance score (0-1, higher=balanced)",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.duplicate_threshold = self.config.get("duplicate_threshold", 5)
        self.outlier_iqr_factor = self.config.get("outlier_iqr_factor", 1.5)
        self.device_config = self.config.get("device", "auto")

        self._clip_model = None
        self._clip_processor = None
        self._clip_available = False
        self._backend = "histogram"

        # Accumulation buffers (beyond _feature_cache in base class)
        self._hashes: List[Tuple[str, int]] = []  # (sample_name, hash_value)
        self._histograms: List[np.ndarray] = []

    def setup(self) -> None:
        try:
            from transformers import CLIPModel, CLIPProcessor
            from ayase.config import resolve_model_path
            from ayase.runtime import (
                from_pretrained_with_attention,
                resolve_torch_device,
                shared_runtime_resource,
            )

            device = resolve_torch_device(self.device_config)
            models_dir = self.config.get("models_dir", "models")
            model_name = self.config.get("clip_model", "openai/clip-vit-base-patch32")
            resolved = resolve_model_path(model_name, models_dir)

            def load_clip():
                model = from_pretrained_with_attention(
                    CLIPModel,
                    resolved,
                    self.config,
                    device=device,
                ).to(device).eval()
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
            self._clip_device = device
            self._clip_model_name = model_name
            self._clip_available = True
            self._backend = "clip"
            logger.info(f"Dataset analytics: CLIP embeddings on {device}")
        except ImportError:
            logger.info("CLIP unavailable, using histogram-based analytics")
        except Exception as e:
            logger.warning(f"CLIP init failed: {e}")

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _dhash(gray: np.ndarray, hash_size: int = 8) -> int:
        """Compute difference hash (dHash) for a grayscale image."""
        resized = cv2.resize(gray, (hash_size + 1, hash_size))
        diff = resized[:, 1:] > resized[:, :-1]
        bits = diff.flatten()
        h = 0
        for bit in bits:
            h = (h << 1) | int(bit)
        return h

    @staticmethod
    def _colour_histogram(frame_bgr: np.ndarray) -> np.ndarray:
        """Compute normalised HSV colour histogram."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1], None, [16, 16], [0, 180, 0, 256]
        )
        hist = hist.flatten().astype(np.float32)
        total = hist.sum()
        if total > 0:
            hist /= total
        return hist

    def _clip_embedding(
        self,
        frame_bgr: np.ndarray,
        cache_key: Optional[tuple] = None,
    ) -> Optional[np.ndarray]:
        """Extract CLIP image embedding."""
        if not self._clip_available:
            return None
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            emb = cached_clip_image_features(
                self,
                self._clip_model,
                self._clip_processor,
                arrays_to_pil([rgb]),
                model_key=getattr(self, "_clip_model_name", "openai/clip-vit-base-patch32"),
                device=getattr(self, "_clip_device", "cpu"),
                cache_key=cache_key or ("dataset_analytics_frame", rgb.shape, rgb.tobytes()),
            )
            return emb.squeeze(0).detach().float().cpu().numpy()
        except Exception as e:
            logger.debug(f"CLIP embedding failed: {e}")
            return None

    def extract_features(self, sample: Sample) -> Optional[np.ndarray]:
        """Extract features from a sample (overrides BatchMetricModule)."""
        try:
            frame = load_representative_frame(sample.path, color="bgr")
            if frame is None:
                return None

            # dHash for duplicate detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h = self._dhash(gray)
            self._hashes.append((str(sample.path), h))

            # Colour histogram fallback
            self._histograms.append(self._colour_histogram(frame))

            # CLIP embedding (primary features for diversity/coverage)
            emb = self._clip_embedding(
                frame,
                cache_key=("dataset_analytics", media_state_key(sample.path)),
            )
            return emb

        except Exception as e:
            logger.debug(f"Feature extraction failed for {sample.path}: {e}")
            return None

    # ------------------------------------------------------------------
    # Batch computation
    # ------------------------------------------------------------------

    def compute_distribution_metric(
        self, features: List[np.ndarray], reference_features: Optional[List[np.ndarray]] = None
    ) -> float:
        """Compute all dataset-level analytics.  Returns diversity_score."""

        # Use CLIP features if available, else histograms
        if features and len(features) > 0 and features[0] is not None:
            matrix = np.stack(features)
        elif self._histograms:
            matrix = np.stack(self._histograms)
        else:
            return 0.0

        n = len(matrix)
        if n < 3:
            return 0.0

        # 1. Diversity score
        diversity = self._compute_diversity(matrix)

        # 2. Semantic coverage
        coverage = self._compute_coverage(matrix)

        # 3. Outlier detection
        outlier_count = self._detect_outliers(matrix)

        # 4. Near-duplicate detection
        duplicate_pairs = self._detect_duplicates()

        # 5. Class balance (based on k-means clustering)
        balance = self._compute_class_balance(matrix)

        # Store in pipeline stats (skip metrics that could not be computed)
        if hasattr(self, "pipeline") and self.pipeline:
            if hasattr(self.pipeline, "add_dataset_metric"):
                self.pipeline.add_dataset_metric("diversity_score", diversity)
                if coverage is not None:
                    self.pipeline.add_dataset_metric("semantic_coverage", coverage)
                self.pipeline.add_dataset_metric("outlier_count", outlier_count)
                self.pipeline.add_dataset_metric("duplicate_pairs", duplicate_pairs)
                if balance is not None:
                    self.pipeline.add_dataset_metric("class_balance_score", balance)

        coverage_str = f"{coverage:.3f}" if coverage is not None else "n/a"
        balance_str = f"{balance:.3f}" if balance is not None else "n/a"
        logger.info(
            f"Dataset analytics: diversity={diversity:.3f} "
            f"coverage={coverage_str} outliers={outlier_count} "
            f"duplicates={duplicate_pairs} balance={balance_str}"
        )

        return diversity

    @staticmethod
    def _compute_diversity(matrix: np.ndarray) -> float:
        """Compute diversity as average pairwise distance (0-1)."""
        n = len(matrix)
        if n < 2:
            return 0.0

        # Sample pairs for efficiency
        max_pairs = min(n * (n - 1) // 2, 5000)
        distances = []

        if n <= 100:
            # All pairs
            for i in range(n):
                for j in range(i + 1, n):
                    d = float(np.linalg.norm(matrix[i] - matrix[j]))
                    distances.append(d)
        else:
            # Random sampling
            rng = np.random.default_rng(42)
            for _ in range(max_pairs):
                i, j = rng.choice(n, size=2, replace=False)
                d = float(np.linalg.norm(matrix[i] - matrix[j]))
                distances.append(d)

        mean_dist = float(np.mean(distances))
        # Normalise by dimensionality
        dim = matrix.shape[1]
        expected_max = np.sqrt(dim) * 0.5  # rough upper bound
        return float(np.clip(mean_dist / max(expected_max, 1e-6), 0, 1))

    @staticmethod
    def _compute_coverage(matrix: np.ndarray) -> Optional[float]:
        """Estimate embedding space coverage via convex hull volume proxy."""
        n, d = matrix.shape
        if n < d + 1:
            return 0.0

        # PCA to 2D for tractable convex hull estimation
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(2, d))
            reduced = pca.fit_transform(matrix)
            explained = float(sum(pca.explained_variance_ratio_))

            # Convex hull area as coverage proxy
            from scipy.spatial import ConvexHull
            hull = ConvexHull(reduced)
            # Normalise by bounding box area
            ranges = reduced.max(axis=0) - reduced.min(axis=0)
            bbox_area = float(np.prod(ranges + 1e-6))
            hull_ratio = float(hull.volume / max(bbox_area, 1e-6))

            return float(np.clip(hull_ratio * explained, 0, 1))

        except ImportError:
            # Without sklearn/scipy, use std-based proxy
            stds = matrix.std(axis=0)
            coverage = float(np.mean(stds > 0.01))
            return float(np.clip(coverage, 0, 1))
        except Exception:
            return None

    @staticmethod
    def _detect_outliers(matrix: np.ndarray) -> int:
        """Detect outliers using distance from centroid + IQR."""
        centroid = matrix.mean(axis=0)
        distances = np.linalg.norm(matrix - centroid, axis=1)

        q1 = np.percentile(distances, 25)
        q3 = np.percentile(distances, 75)
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr

        return int(np.sum(distances > upper))

    def _detect_duplicates(self) -> int:
        """Count near-duplicate pairs using dHash Hamming distance."""
        n = len(self._hashes)
        if n < 2:
            return 0

        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                h1 = self._hashes[i][1]
                h2 = self._hashes[j][1]
                hamming = bin(h1 ^ h2).count("1")
                if hamming <= self.duplicate_threshold:
                    count += 1

        return count

    @staticmethod
    def _compute_class_balance(matrix: np.ndarray, n_clusters: int = 10) -> Optional[float]:
        """Estimate class balance via k-means cluster sizes."""
        n = len(matrix)
        k = min(n_clusters, n // 2)
        if k < 2:
            return 1.0

        try:
            from sklearn.cluster import MiniBatchKMeans
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3)
            labels = kmeans.fit_predict(matrix)
        except ImportError:
            # Without sklearn, class balance cannot be estimated.
            return None

        # Cluster sizes
        counts = np.bincount(labels, minlength=k).astype(float)
        # Entropy-based balance: max entropy = perfectly balanced
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log(probs)))
        max_entropy = float(np.log(k))

        return float(entropy / max(max_entropy, 1e-6))

    def on_dispose(self) -> None:
        """Override to also clean up extra buffers."""
        # Run parent computation first
        if len(self._feature_cache) < 4 and len(self._histograms) < 4:
            logger.info(
                f"Dataset analytics: not enough samples "
                f"({max(len(self._feature_cache), len(self._histograms))})"
            )
            self._feature_cache = []
            self._reference_cache = []
            self._hashes = []
            self._histograms = []
            return

        try:
            self.compute_distribution_metric(
                self._feature_cache,
                self._reference_cache if self._reference_cache else None,
            )
        except Exception as e:
            logger.error(f"Dataset analytics failed: {e}")
        finally:
            self._feature_cache = []
            self._reference_cache = []
            self._hashes = []
            self._histograms = []
