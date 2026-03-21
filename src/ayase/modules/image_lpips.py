"""Image LPIPS — perceptual distance between image pairs and diversity metric.

Computes the Learned Perceptual Image Patch Similarity (LPIPS) between a sample
image and its reference.  Also computes a dataset-level diversity metric via
``post_process()`` by averaging pairwise LPIPS across a random subset of image
pairs.

Outputs:
    image_lpips      — per-sample LPIPS distance vs reference (0-1, lower=more similar)
    lpips_diversity  — dataset-level average pairwise LPIPS (higher=more diverse)

Requires ``sample.reference_path`` for per-sample LPIPS.

Tiered backends:
    1. **lpips** library (AlexNet, VGG, or SqueezeNet backbone)
    2. **Heuristic** — SSIM-based proxy via OpenCV
"""

import logging
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class ImageLPIPSModule(PipelineModule):
    name = "image_lpips"
    description = "LPIPS perceptual distance between image pairs and diversity metric"
    default_config = {
        "net": "alex",  # "alex", "vgg", "squeeze"
        "resize": 256,  # Resize images before computing LPIPS
        "diversity_max_pairs": 500,  # Max pairs for diversity computation
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._backend = "heuristic"
        self._lpips_model = None
        self._device = None
        # Cache tensors for diversity computation
        self._tensor_cache: List[Tuple[str, object]] = []

    def setup(self) -> None:
        """Load LPIPS model with tiered fallback."""
        # Tier 1: Real LPIPS model
        try:
            import torch
            import lpips

            net = self.config.get("net", "alex")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._lpips_model = lpips.LPIPS(net=net).to(device)
            self._lpips_model.eval()
            self._device = device
            self._backend = "lpips"
            self._ml_available = True
            logger.info("ImageLPIPS: loaded LPIPS-%s on %s", net, device)
            return
        except (ImportError, Exception) as e:
            logger.info("LPIPS library unavailable: %s", e)

        # Tier 2: Heuristic (SSIM-based proxy)
        self._backend = "heuristic"
        self._ml_available = True
        logger.info("ImageLPIPS: using SSIM-based heuristic fallback")

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()

        if not self._ml_available:
            return sample

        try:
            # Load sample image
            sample_img = self._load_image(sample.path)
            if sample_img is None:
                return sample

            # Cache for diversity computation (store path + resized image)
            self._cache_for_diversity(str(sample.path), sample_img)

            # Per-sample FR LPIPS requires reference_path
            ref_path = getattr(sample, "reference_path", None)
            if ref_path is None:
                return sample

            ref_img = self._load_image(ref_path)
            if ref_img is None:
                return sample

            # Compute LPIPS distance
            distance = self._compute_distance(sample_img, ref_img)
            if distance is not None:
                sample.quality_metrics.image_lpips = float(np.clip(distance, 0.0, 1.0))

        except Exception as e:
            logger.warning("ImageLPIPS failed for %s: %s", sample.path, e)

        return sample

    def post_process(self, all_samples: List[Sample]) -> None:
        """Compute dataset-level LPIPS diversity from cached tensors."""
        if len(self._tensor_cache) < 2:
            self._tensor_cache = []
            return

        try:
            max_pairs = self.config.get("diversity_max_pairs", 500)
            n = len(self._tensor_cache)

            # Generate all possible pair indices
            all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

            # Subsample if too many pairs
            if len(all_pairs) > max_pairs:
                pairs = random.sample(all_pairs, max_pairs)
            else:
                pairs = all_pairs

            distances = []
            for i, j in pairs:
                _, img_a = self._tensor_cache[i]
                _, img_b = self._tensor_cache[j]
                dist = self._compute_distance(img_a, img_b)
                if dist is not None:
                    distances.append(dist)

            if distances:
                diversity = float(np.mean(distances))
                # Store in pipeline stats
                if hasattr(self, "pipeline") and self.pipeline:
                    if hasattr(self.pipeline, "add_dataset_metric"):
                        self.pipeline.add_dataset_metric("lpips_diversity", diversity)
                logger.info(
                    "ImageLPIPS diversity: %.4f (from %d pairs)", diversity, len(distances)
                )

        except Exception as e:
            logger.warning("ImageLPIPS diversity computation failed: %s", e)
        finally:
            self._tensor_cache = []

    # -- Internal methods -------------------------------------------------------

    def _load_image(self, path) -> Optional[np.ndarray]:
        """Load and resize an image to the configured size."""
        try:
            img = cv2.imread(str(path))
            if img is None:
                return None
            resize = self.config.get("resize", 256)
            img = cv2.resize(img, (resize, resize))
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return rgb
        except Exception:
            return None

    def _cache_for_diversity(self, path: str, img: np.ndarray) -> None:
        """Cache an image for diversity computation."""
        self._tensor_cache.append((path, img))

    def _compute_distance(self, img_a: np.ndarray, img_b: np.ndarray) -> Optional[float]:
        """Compute perceptual distance between two RGB images using best backend."""
        if self._backend == "lpips":
            return self._compute_lpips(img_a, img_b)
        return self._compute_heuristic(img_a, img_b)

    def _compute_lpips(self, img_a: np.ndarray, img_b: np.ndarray) -> Optional[float]:
        """Compute distance using real LPIPS model."""
        try:
            import torch

            # Convert to tensors: [1, 3, H, W] in [-1, 1]
            t_a = torch.from_numpy(img_a).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
            t_b = torch.from_numpy(img_b).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
            t_a, t_b = t_a.to(self._device), t_b.to(self._device)

            with torch.no_grad():
                dist = self._lpips_model(t_a, t_b).item()
            return float(dist)
        except Exception as e:
            logger.debug("LPIPS computation failed: %s", e)
            return None

    def _compute_heuristic(self, img_a: np.ndarray, img_b: np.ndarray) -> Optional[float]:
        """SSIM-based perceptual distance proxy.

        Converts SSIM (1=identical, -1=different) to a distance in [0, 1].
        """
        try:
            gray_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY).astype(np.float64)
            gray_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY).astype(np.float64)

            # Compute SSIM manually
            c1 = (0.01 * 255) ** 2
            c2 = (0.03 * 255) ** 2

            mu_a = cv2.GaussianBlur(gray_a, (11, 11), 1.5)
            mu_b = cv2.GaussianBlur(gray_b, (11, 11), 1.5)

            mu_a_sq = mu_a ** 2
            mu_b_sq = mu_b ** 2
            mu_ab = mu_a * mu_b

            sigma_a_sq = cv2.GaussianBlur(gray_a ** 2, (11, 11), 1.5) - mu_a_sq
            sigma_b_sq = cv2.GaussianBlur(gray_b ** 2, (11, 11), 1.5) - mu_b_sq
            sigma_ab = cv2.GaussianBlur(gray_a * gray_b, (11, 11), 1.5) - mu_ab

            ssim_map = ((2 * mu_ab + c1) * (2 * sigma_ab + c2)) / (
                (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
            )
            ssim_val = float(np.mean(ssim_map))

            # Convert SSIM to distance: SSIM=1 -> dist=0, SSIM=0 -> dist=0.5
            distance = float(np.clip((1.0 - ssim_val) / 2.0, 0.0, 1.0))
            return distance
        except Exception as e:
            logger.debug("Heuristic SSIM distance failed: %s", e)
            return None
