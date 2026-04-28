"""Vendi Score — Diversity Metric (NeurIPS 2022).

Dataset-level diversity metric based on the matrix entropy of a
pairwise similarity matrix. Higher Vendi Score = more diverse dataset.

pip install vendi_score

vendi_score — higher = more diverse.
"""

import logging
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.base_modules import BatchMetricModule

logger = logging.getLogger(__name__)


class VendiModule(BatchMetricModule):
    name = "vendi"
    description = "Vendi Score dataset diversity (NeurIPS 2022, batch metric)"
    default_config = {
        "feature_dim": 512,
        "max_samples": 1000,
        "resize": 224,
    }
    models = [
        {
            "id": "vendi_score",
            "type": "pip_package",
            "install": "pip install vendi-score",
            "task": "Optional Vendi Score entropy backend",
        },
    ]
    metric_info = {
        "vendi": "Vendi Score dataset diversity from similarity-matrix entropy (higher=better)",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None
        self._ml_available = False
        self.feature_dim = self.config.get("feature_dim", 512)
        self.max_samples = self.config.get("max_samples", 1000)
        self.resize = self.config.get("resize", 224)
        self._processed_count = 0

    def setup(self) -> None:
        # Tier 1: vendi_score package
        try:
            import vendi_score
            self._model = vendi_score
            self._ml_available = True
            logger.info("Vendi Score module initialised (vendi_score package)")
            return
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"vendi_score init failed: {e}")

        # Tier 2: heuristic (eigenvalue entropy of cosine similarity)
        logger.info("Vendi Score module initialised (heuristic fallback)")

    def extract_features(self, sample: Sample) -> Optional[np.ndarray]:
        """Extract a feature vector from a sample (histogram-based)."""
        if self.max_samples and self._processed_count >= self.max_samples:
            return None

        try:
            if sample.is_video:
                feat = self._extract_video_features(sample.path)
            else:
                feat = self._extract_image_features(sample.path)

            if feat is not None:
                self._processed_count += 1
            return feat
        except Exception as e:
            logger.debug(f"Vendi feature extraction failed for {sample.path}: {e}")
            return None

    def _extract_image_features(self, path: Path) -> Optional[np.ndarray]:
        img = cv2.imread(str(path))
        if img is None:
            return None
        img = cv2.resize(img, (self.resize, self.resize))
        return self._compute_feature_vector(img)

    def _extract_video_features(self, path: Path) -> Optional[np.ndarray]:
        cap = cv2.VideoCapture(str(path))
        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                return None
            frame = cv2.resize(frame, (self.resize, self.resize))
            return self._compute_feature_vector(frame)
        finally:
            cap.release()

    def _compute_feature_vector(self, img: np.ndarray) -> np.ndarray:
        """Compute a feature vector from colour and edge histograms."""
        features = []

        # Colour histogram (per channel, 64 bins each = 192)
        for c in range(3):
            hist = cv2.calcHist([img], [c], None, [64], [0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-8)
            features.append(hist)

        # Edge histogram (128 bins)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150).astype(np.float32)
        edge_hist = cv2.calcHist([edges], [0], None, [128], [0, 256])
        edge_hist = edge_hist.flatten() / (edge_hist.sum() + 1e-8)
        features.append(edge_hist)

        feat = np.concatenate(features)
        # Pad or truncate to feature_dim
        if len(feat) < self.feature_dim:
            feat = np.pad(feat, (0, self.feature_dim - len(feat)))
        else:
            feat = feat[:self.feature_dim]
        # L2 normalise
        norm = np.linalg.norm(feat) + 1e-8
        return feat / norm

    def compute_distribution_metric(
        self, features: List[np.ndarray], reference_features: Optional[List[np.ndarray]] = None
    ) -> float:
        """Compute Vendi Score from feature set."""
        try:
            feat_matrix = np.stack(features, axis=0)  # (N, D)

            if self._ml_available and self._model is not None:
                return self._compute_vendi_package(feat_matrix)
            return self._compute_heuristic(feat_matrix)
        except Exception as e:
            logger.error(f"Vendi Score computation failed: {e}")
            return 0.0

    def _compute_vendi_package(self, features: np.ndarray) -> float:
        try:
            from vendi_score import vendi
            # Cosine similarity kernel
            score = vendi.score(features, k=lambda x, y: np.dot(x, y))
            return float(score)
        except Exception as e:
            logger.debug(f"vendi_score package failed: {e}")
            return self._compute_heuristic(features)

    def _compute_heuristic(self, features: np.ndarray) -> float:
        """Eigenvalue entropy of cosine similarity matrix."""
        # Cosine similarity matrix
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        normed = features / norms
        sim_matrix = normed @ normed.T

        # Ensure symmetric positive semi-definite
        sim_matrix = (sim_matrix + sim_matrix.T) / 2.0
        np.fill_diagonal(sim_matrix, 1.0)

        # Eigenvalue decomposition
        eigenvalues = np.linalg.eigvalsh(sim_matrix)
        eigenvalues = np.maximum(eigenvalues, 0.0)

        # Normalise to form a distribution
        total = eigenvalues.sum()
        if total < 1e-8:
            return 1.0
        probs = eigenvalues / total

        # Matrix entropy: exp(Shannon entropy)
        probs = probs[probs > 1e-12]
        entropy = -np.sum(probs * np.log(probs))
        vendi_score = float(np.exp(entropy))
        return vendi_score

    def on_dispose(self) -> None:
        """Compute and store Vendi Score after all samples processed."""
        if len(self._feature_cache) < 2:
            logger.info(f"Vendi: Not enough samples ({len(self._feature_cache)})")
            self._feature_cache = []
            self._reference_cache = []
            return

        try:
            score = self.compute_distribution_metric(self._feature_cache)
            logger.info(f"Vendi Score: {score:.4f} ({len(self._feature_cache)} samples)")

            if hasattr(self, "pipeline") and self.pipeline:
                if hasattr(self.pipeline, "add_dataset_metric"):
                    self.pipeline.add_dataset_metric("vendi", score)
        except Exception as e:
            logger.error(f"Vendi Score failed: {e}")
        finally:
            self._feature_cache = []
            self._reference_cache = []
            self._processed_count = 0
