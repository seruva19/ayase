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
    provenance = "published"
    sources = {
        "vendi": "Vendi Score, Friedman & Dieng TMLR 2023 — https://github.com/vertaix/Vendi-Score (Inception pool embeddings)",
    }
    deviations = {
        "vendi": "embeddings are the FID-Inception pool (2048-d) as in published Vendi; for video — mean over up to 8 frames; cosine kernel",
    }
    description = "Vendi Score dataset diversity (NeurIPS 2022, batch metric)"
    default_config = {
        "max_samples": 1000,
        "num_frames": 8,  # frames mean-pooled per video
        "device": "auto",
    }
    models = [
        {
            "id": "mseitzer/pytorch-fid:fid_inception_v3",
            "type": "torch_hub",
            "task": "FID-Inception pool embeddings (shared with FID)",
        },
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
        self._model = None  # vendi_score entropy backend (optional)
        self._embedder = None
        self._transform = None
        self._device = "cpu"
        self._ml_available = False
        self._backend = None
        self.max_samples = self.config.get("max_samples", 1000)
        self.num_frames = self.config.get("num_frames", 8)
        self.device_config = self.config.get("device", "auto")
        self._processed_count = 0

    def setup(self) -> None:
        # Embeddings must come from a published extractor — Inception pool3
        # (the same fid_inception_v3 shared with the FID module). The Vendi
        # formula itself is exact in numpy; the package is an optional speed-up.
        try:
            import torch
            from torchvision import transforms
            from ayase.runtime import resolve_torch_device, shared_runtime_resource

            self._device = resolve_torch_device(self.device_config)

            def load_inception():
                return torch.hub.load(
                    "mseitzer/pytorch-fid", "fid_inception_v3"
                ).to(self._device).eval()

            self._embedder = shared_runtime_resource(
                self,
                ("fid_inception_v3_pt_inception", str(self._device)),
                load_inception,
            )
            self._transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(
                    (299, 299),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
            ])
        except ImportError:
            logger.info("Vendi unavailable: torch/torchvision required for Inception embeddings")
            self._backend = "unavailable"
            return
        except Exception as e:
            logger.warning(f"Vendi Inception setup failed: {e}")
            self._backend = "unavailable"
            return

        try:
            import vendi_score
            self._model = vendi_score
            self._backend = "vendi_score"
            logger.info("Vendi Score module initialised (vendi_score package, Inception embeddings)")
        except ImportError:
            self._backend = "numpy"
            logger.info("Vendi Score module initialised (numpy matrix-entropy, Inception embeddings)")

    def extract_features(self, sample: Sample) -> Optional[np.ndarray]:
        """Extract the Inception pool embedding for a sample."""
        if self._embedder is None:
            return None
        if self.max_samples and self._processed_count >= self.max_samples:
            return None

        try:
            from ayase.image import sample_frames, load_representative_frame

            if sample.is_video and self.num_frames > 1:
                frames = list(
                    sample_frames(sample.path, max_frames=self.num_frames, color="rgb")
                )
                feats = [self._embed_frame(f) for f in frames]
                feats = [f for f in feats if f is not None]
                feat = np.mean(feats, axis=0) if feats else None
            else:
                frame = load_representative_frame(sample.path, color="rgb")
                feat = self._embed_frame(frame) if frame is not None else None

            if feat is not None:
                self._processed_count += 1
                # L2-normalise the published embedding.
                feat = feat / (np.linalg.norm(feat) + 1e-8)
            return feat
        except Exception as e:
            logger.debug(f"Vendi feature extraction failed for {sample.path}: {e}")
            return None

    def _embed_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        try:
            import torch

            tensor = self._transform(
                np.ascontiguousarray(frame, dtype=np.uint8)
            ).unsqueeze(0).to(self._device)
            with torch.no_grad():
                feat = self._embedder(tensor)
            return feat.detach().cpu().numpy().reshape(-1).astype(np.float64)
        except Exception as e:
            logger.debug(f"Vendi Inception embedding failed: {e}")
            return None

    def compute_distribution_metric(
        self, features: List[np.ndarray], reference_features: Optional[List[np.ndarray]] = None
    ) -> Optional[float]:
        """Compute Vendi Score from feature set."""
        try:
            feat_matrix = np.stack(features, axis=0)  # (N, D)

            if self._ml_available and self._model is not None:
                return self._compute_vendi_package(feat_matrix)
            return self._compute_vendi_numpy(feat_matrix)
        except Exception as e:
            logger.error(f"Vendi Score computation failed: {e}")
            return None

    def _compute_vendi_package(self, features: np.ndarray) -> float:
        try:
            from vendi_score import vendi
            # Cosine similarity kernel
            score = vendi.score(features, k=lambda x, y: np.dot(x, y))
            return float(score)
        except Exception as e:
            logger.debug(f"vendi_score package failed: {e}")
            return self._compute_vendi_numpy(features)

    def _compute_vendi_numpy(self, features: np.ndarray) -> float:
        """Exact Vendi Score: exp(Shannon entropy of the eigenvalues of the
        normalised cosine-similarity kernel)."""
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
            if score is None:
                logger.warning("Vendi Score could not be computed")
                return
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
