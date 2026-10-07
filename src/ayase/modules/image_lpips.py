"""Image LPIPS — perceptual distance between image pairs and diversity metric.

Computes the Learned Perceptual Image Patch Similarity (LPIPS) between a sample
image and its reference.  Also computes a dataset-level diversity metric via
``post_process()`` by averaging pairwise LPIPS across a random subset of image
pairs.

Outputs:
    image_lpips      — per-sample LPIPS distance vs reference (0-1, lower=more similar)
    lpips_diversity  — dataset-level average pairwise LPIPS between samples
                       that share the same conditioning input
                       (``reference_path``); higher=more diverse

Requires ``sample.reference_path`` for per-sample LPIPS.

Backend: **lpips** library (AlexNet, VGG, or SqueezeNet backbone).
"""

import logging
import random
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class ImageLPIPSModule(PipelineModule):
    name = "image_lpips"
    provenance = "published"
    sources = {
        "image_lpips": "LPIPS (Zhang et al., CVPR 2018) — https://github.com/richzhang/PerceptualSimilarity",
        "lpips_diversity": "LPIPS-diversity (BicycleGAN/MUNIT) — https://arxiv.org/abs/1711.11586",
    }
    description = "LPIPS perceptual distance between image pairs and diversity metric"
    default_config = {
        "net": "alex",  # "alex", "vgg", "squeeze"
        "resize": 256,  # Resize images before computing LPIPS
        "diversity_max_pairs": 500,  # Max pairs for diversity computation
        "diversity_batch_size": 64,  # Pairs stacked per LPIPS forward pass
        # Seed for reproducible pair subsampling in the diversity metric.
        "seed": 42,
    }
    models = [
        {
            "id": "lpips",
            "type": "pip_package",
            "install": "pip install lpips",
            "task": "LPIPS AlexNet/VGG/SqueezeNet perceptual distance model",
        },
    ]
    metric_info = {
        "image_lpips": "Per-sample LPIPS distance to reference image (lower=better)",
        "lpips_diversity": "Mean pairwise LPIPS between outputs sharing a conditioning input (higher=more diverse)",
    }
    metric_groups = {
        "image_lpips": "fr_quality",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._lpips_model = None
        self._device = "cpu"
        self._backend = None
        # Cache (condition key, resized RGB image) for diversity computation
        self._tensor_cache: List[Tuple[Optional[str], np.ndarray]] = []

    def setup(self) -> None:
        """Load (or reuse a pipeline-shared) LPIPS model."""
        if self.test_mode:
            return
        try:
            import lpips
            from ayase.runtime import resolve_torch_device, shared_runtime_resource

            net = self.config.get("net", "alex")
            self._device = resolve_torch_device(self.config.get("device", "auto"))

            def load_lpips():
                model = lpips.LPIPS(net=net).to(self._device)
                model.eval()
                return model

            # Share the LPIPS backbone across modules/samples in a pipeline.
            self._lpips_model = shared_runtime_resource(
                self,
                ("lpips", net, str(self._device)),
                load_lpips,
            )
            self._ml_available = True
            self._backend = "lpips"
            logger.info("ImageLPIPS: loaded LPIPS-%s on %s", net, self._device)
        except ImportError:
            self._backend = "unavailable"
            logger.warning("ImageLPIPS: lpips library not installed (pip install lpips)")
        except Exception as e:
            self._backend = "unavailable"
            logger.warning("ImageLPIPS: lpips setup failed: %s", e)

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

            # Cache for diversity computation — keyed by the conditioning
            # reference so pairs are only formed within one condition.
            ref_path = getattr(sample, "reference_path", None)
            self._cache_for_diversity(str(ref_path) if ref_path else None, sample_img)

            # Per-sample FR LPIPS requires reference_path
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
        """Compute dataset-level LPIPS diversity from cached images.

        Per the published diversity protocol, pairs are only formed between
        outputs produced for the *same* conditioning input
        (``reference_path``) — not across the whole dataset.
        """
        # Group cached outputs by condition key; conditions with one output
        # contribute nothing.
        groups: dict = {}
        for idx, (key, _img) in enumerate(self._tensor_cache):
            groups.setdefault(key, []).append(idx)

        try:
            max_pairs = self.config.get("diversity_max_pairs", 500)
            seed = int(self.config.get("seed", 42))
            rng = random.Random(seed)

            per_condition: List[float] = []
            for key in sorted(groups):
                indices = groups[key]
                if len(indices) < 2:
                    continue

                all_pairs = [
                    (i, j) for a, i in enumerate(indices) for j in indices[a + 1 :]
                ]
                if len(all_pairs) > max_pairs:
                    all_pairs = rng.sample(all_pairs, max_pairs)

                distances = self._compute_distances_batched(all_pairs)
                if distances:
                    per_condition.append(float(np.mean(distances)))

            if per_condition:
                diversity = float(np.mean(per_condition))
                # Store in pipeline stats
                if hasattr(self, "pipeline") and self.pipeline:
                    if hasattr(self.pipeline, "add_dataset_metric"):
                        self.pipeline.add_dataset_metric("lpips_diversity", diversity)
                logger.info(
                    "ImageLPIPS diversity: %.4f (over %d conditions)",
                    diversity,
                    len(per_condition),
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

    def _cache_for_diversity(self, key: Optional[str], img: np.ndarray) -> None:
        """Cache an image for diversity computation under its condition key."""
        self._tensor_cache.append((key, img))

    def _to_tensor(self, img_rgb: np.ndarray):
        """RGB uint8 HWC -> LPIPS tensor (1,3,H,W) in [-1, 1]."""
        import torch

        arr = np.ascontiguousarray(img_rgb, dtype=np.float32)
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0

    def _compute_distance(self, img_a: np.ndarray, img_b: np.ndarray) -> Optional[float]:
        """Compute perceptual distance between two RGB images using LPIPS."""
        try:
            import torch

            t_a = self._to_tensor(img_a).to(self._device)
            t_b = self._to_tensor(img_b).to(self._device)

            with torch.no_grad():
                dist = self._lpips_model(t_a, t_b).item()
            return float(dist)
        except Exception as e:
            logger.debug("LPIPS computation failed: %s", e)
            return None

    def _compute_distances_batched(self, pairs: List[Tuple[int, int]]) -> List[float]:
        """Compute LPIPS over many pairs, stacking them into batched forwards."""
        try:
            import torch
        except Exception as e:
            logger.debug("LPIPS batched computation missing torch: %s", e)
            return []

        chunk = max(1, int(self.config.get("diversity_batch_size", 64)))
        distances: List[float] = []
        for start in range(0, len(pairs), chunk):
            chunk_pairs = pairs[start:start + chunk]
            a_list = []
            b_list = []
            for i, j in chunk_pairs:
                a_list.append(self._to_tensor(self._tensor_cache[i][1]))
                b_list.append(self._to_tensor(self._tensor_cache[j][1]))
            try:
                t_a = torch.cat(a_list, dim=0).to(self._device)
                t_b = torch.cat(b_list, dim=0).to(self._device)
                with torch.no_grad():
                    out = self._lpips_model(t_a, t_b)
                for val in out.view(-1).cpu().numpy().tolist():
                    distances.append(float(val))
            except Exception as e:
                logger.debug("LPIPS batched forward failed: %s", e)
        return distances
