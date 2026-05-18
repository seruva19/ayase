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
from ayase.image import arrays_to_pil, load_representative_frame
from ayase.models import Sample
from ayase.runtime import cached_clip_image_features, media_state_key

logger = logging.getLogger(__name__)


class UMAPProjectionModule(BatchMetricModule):
    name = "umap_projection"
    description = "UMAP/t-SNE/PCA 2-D projection with spread & coverage"
    default_config = {
        "device": "auto",
        "min_samples": 3,
        "clip_model": "openai/clip-vit-base-patch32",
    }
    models = [
        {
            "id": "openai/clip-vit-base-patch32",
            "type": "huggingface",
            "task": "CLIP embedding extractor for projection",
        },
    ]
    metric_info = {
        "umap_spread": "Spread of dataset embeddings in the 2-D projection",
        "umap_coverage": "Coverage of occupied projection space (0-1, higher=better)",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.device_config = self.config.get("device", "auto")
        self.min_samples = self.config.get("min_samples", 3)
        self.clip_model_name = self.config.get("clip_model", "openai/clip-vit-base-patch32")

        self._clip_model = None
        self._clip_processor = None
        self._clip_device = "cpu"
        self._clip_available = False

        self._sample_refs: List[Sample] = []

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
            resolved = resolve_model_path(self.clip_model_name, models_dir)

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

            emb = self._clip_embedding(frame, cache_key=("umap_projection", media_state_key(sample.path)))
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
        return load_representative_frame(sample.path, color="bgr")

    def _clip_embedding(
        self,
        frame_bgr: np.ndarray,
        cache_key: Optional[tuple] = None,
    ) -> Optional[np.ndarray]:
        if not self._clip_available:
            return None
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            emb = cached_clip_image_features(
                self,
                self._clip_model,
                self._clip_processor,
                arrays_to_pil([rgb]),
                model_key=self.clip_model_name,
                device=self._clip_device,
                cache_key=cache_key or ("umap_projection_frame", rgb.shape, rgb.tobytes()),
            )
            return emb.squeeze(0).detach().float().cpu().numpy()
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
