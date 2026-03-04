"""UMAP Projection module.

Projects dataset samples into a 2-D embedding space for visualisation
and coverage analysis.  Stores per-sample coordinates in
``sample.detections`` and dataset-level ``umap_spread`` /
``umap_coverage`` via ``pipeline.add_dataset_metric()``.

Feature extraction tiers:
  1. CLIP image embeddings (primary)
  2. HSV colour histogram (fallback)

Dimensionality reduction tiers:
  1. ``umap-learn``  (``pip install umap-learn``)
  2. ``sklearn.TSNE``
  3. ``sklearn.PCA``
  4. numpy covariance PCA (no deps)
"""

import logging
from typing import List, Optional

import cv2
import numpy as np

from ayase.base_modules import BatchMetricModule
from ayase.models import Sample

logger = logging.getLogger(__name__)


class UMAPProjectionModule(BatchMetricModule):
    name = "umap_projection"
    description = "UMAP/t-SNE/PCA 2-D projection with spread & coverage"
    default_config = {
        "device": "auto",
        "min_samples": 3,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.device_config = self.config.get("device", "auto")
        self.min_samples = self.config.get("min_samples", 3)

        self._clip_model = None
        self._clip_processor = None
        self._clip_available = False

        self._sample_refs: List[Sample] = []

    def setup(self) -> None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            device = self.device_config
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            self._clip_model = (
                CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                .to(device)
                .eval()
            )
            self._clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self._clip_available = True
            logger.info(f"UMAP projection: CLIP on {device}")
        except ImportError:
            logger.info("CLIP unavailable, using histogram features")
        except Exception as e:
            logger.warning(f"CLIP init failed: {e}")

    # ------------------------------------------------------------------
    # Feature extraction (overrides BatchMetricModule.extract_features)
    # ------------------------------------------------------------------

    def extract_features(self, sample: Sample) -> Optional[np.ndarray]:
        try:
            frame = self._read_frame(sample)
            if frame is None:
                return None

            emb = self._clip_embedding(frame)
            feature = emb if emb is not None else self._colour_histogram(frame)

            # Only track the sample ref when we actually return a feature,
            # so _sample_refs stays in sync with _feature_cache.
            self._sample_refs.append(sample)
            return feature
        except Exception as e:
            logger.debug(f"Feature extraction failed for {sample.path}: {e}")
            return None

    @staticmethod
    def _read_frame(sample: Sample) -> Optional[np.ndarray]:
        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(total // 2, 0))
            ret, frame = cap.read()
            cap.release()
            return frame if ret else None
        frame = cv2.imread(str(sample.path))
        return frame

    def _clip_embedding(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        if not self._clip_available:
            return None
        try:
            import torch
            from PIL import Image

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            inputs = self._clip_processor(images=pil_img, return_tensors="pt")
            device = next(self._clip_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                emb = self._clip_model.get_image_features(**inputs)
                emb = emb / emb.norm(dim=-1, keepdim=True)

            return emb.squeeze(0).cpu().numpy()
        except Exception as e:
            logger.debug(f"CLIP embedding failed: {e}")
            return None

    @staticmethod
    def _colour_histogram(frame_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        hist = hist.flatten().astype(np.float32)
        total = hist.sum()
        if total > 0:
            hist /= total
        return hist

    # ------------------------------------------------------------------
    # Batch computation
    # ------------------------------------------------------------------

    def compute_distribution_metric(
        self,
        features: List[np.ndarray],
        reference_features: Optional[List[np.ndarray]] = None,
    ) -> float:
        if len(features) < self.min_samples:
            return 0.0

        matrix = np.stack(features)
        coords = self._reduce_2d(matrix)

        # Store per-sample coordinates
        for i, s in enumerate(self._sample_refs[: len(coords)]):
            s.detections.append(
                {
                    "type": "umap_coords",
                    "x": float(coords[i, 0]),
                    "y": float(coords[i, 1]),
                }
            )

        spread = float(np.std(coords))
        coverage = self._hull_coverage(coords)

        # Store dataset-level metrics
        if hasattr(self, "pipeline") and self.pipeline:
            if hasattr(self.pipeline, "add_dataset_metric"):
                self.pipeline.add_dataset_metric("umap_spread", spread)
                self.pipeline.add_dataset_metric("umap_coverage", coverage)

        logger.info(f"UMAP projection: spread={spread:.4f} coverage={coverage:.4f}")
        return spread

    @staticmethod
    def _reduce_2d(matrix: np.ndarray) -> np.ndarray:
        n = len(matrix)

        # Tier 1: umap-learn
        try:
            import umap  # noqa: F811

            reducer = umap.UMAP(n_components=2, random_state=42)
            return reducer.fit_transform(matrix)
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"UMAP failed: {e}")

        # Tier 2: sklearn TSNE
        try:
            from sklearn.manifold import TSNE

            perplexity = min(30.0, max(1.0, n / 3.0))
            return TSNE(
                n_components=2, perplexity=perplexity, random_state=42
            ).fit_transform(matrix)
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"TSNE failed: {e}")

        # Tier 3: sklearn PCA
        try:
            from sklearn.decomposition import PCA

            return PCA(n_components=2, random_state=42).fit_transform(matrix)
        except ImportError:
            pass

        # Tier 4: numpy PCA
        matrix_c = matrix - matrix.mean(axis=0)
        cov = np.cov(matrix_c, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Top 2 components (eigenvalues sorted ascending)
        top2 = eigenvectors[:, -2:][:, ::-1]
        return matrix_c @ top2

    @staticmethod
    def _hull_coverage(coords: np.ndarray) -> float:
        if len(coords) < 3:
            return 0.0

        try:
            from scipy.spatial import ConvexHull

            hull = ConvexHull(coords)
            ranges = coords.max(axis=0) - coords.min(axis=0)
            bbox_area = float(np.prod(ranges + 1e-6))
            return float(np.clip(hull.volume / max(bbox_area, 1e-6), 0, 1))
        except ImportError:
            pass
        except Exception:
            pass

        # Fallback: std-based proxy
        stds = np.std(coords, axis=0)
        return float(np.clip(np.mean(stds) / (np.max(stds) + 1e-6), 0, 1))

    def on_dispose(self) -> None:
        if len(self._feature_cache) < self.min_samples:
            logger.info(
                f"UMAP projection: not enough samples ({len(self._feature_cache)})"
            )
            self._feature_cache = []
            self._reference_cache = []
            self._sample_refs = []
            return

        try:
            self.compute_distribution_metric(
                self._feature_cache,
                self._reference_cache if self._reference_cache else None,
            )
        except Exception as e:
            logger.error(f"UMAP projection failed: {e}")
        finally:
            self._feature_cache = []
            self._reference_cache = []
            self._sample_refs = []
