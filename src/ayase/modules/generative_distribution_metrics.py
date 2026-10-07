"""Generative Distribution Metrics (Precision, Recall, Density, Coverage).

Batch-level metrics comparing generated and reference distributions via the
published ``prdc`` package (Kynkäänniemi et al. NeurIPS 2019 precision/recall;
Naeem et al. ICML 2020 density/coverage). Features are VGG16 fc7 (4096-d)
embeddings of a representative frame, matching the papers' evaluation
protocol. A real reference set is required.

This is a dataset-level (batch) metric, not per-sample.
"""

import logging
from typing import Optional, List

import numpy as np

from ayase.image import arrays_to_pil, load_representative_frame
from ayase.models import Sample
from ayase.base_modules import BatchMetricModule

logger = logging.getLogger(__name__)


class GenerativeDistributionModule(BatchMetricModule):
    name = "generative_distribution"
    provenance = "adapted"
    sources = {
        "coverage": "Density/Coverage (Naeem et al., ICML 2020) via prdc — https://github.com/clovaai/generative-evaluation-prdc",
        "density": "Density/Coverage (Naeem et al., ICML 2020) via prdc — https://github.com/clovaai/generative-evaluation-prdc",
        "precision": "Improved P/R (Kynkäänniemi et al., NeurIPS 2019) via prdc — https://github.com/clovaai/generative-evaluation-prdc",
        "recall": "Improved P/R (Kynkäänniemi et al., NeurIPS 2019) via prdc — https://github.com/clovaai/generative-evaluation-prdc",
    }
    deviations = {
        "precision": "video → 1 representative frame; without a reference the metrics are not emitted",
        "recall": "video → 1 representative frame; without a reference the metrics are not emitted",
        "density": "video → 1 representative frame; without a reference the metrics are not emitted",
        "coverage": "video → 1 representative frame; without a reference the metrics are not emitted",
    }
    description = "Precision / Recall / Density / Coverage (batch metric, prdc)"
    default_config = {
        "k": 5,  # nearest_k for prdc.compute_prdc
        "device": "auto",
    }
    models = [
        {
            "id": "torchvision/vgg16",
            "type": "torchvision",
            "task": "VGG16 fc7 image embeddings (papers' protocol)",
        },
        {
            "id": "prdc",
            "type": "pip_package",
            "install": "pip install prdc",
            "task": "Published precision/recall/density/coverage implementation",
        },
    ]
    metric_info = {
        "precision": "Generated-sample precision against the real manifold (0-1, higher=better)",
        "recall": "Real-distribution coverage by generated samples (0-1, higher=better)",
        "coverage": "Fraction of real samples covered by generated neighbours (0-1, higher=better)",
        "density": "Average normalized generated-sample density around real samples",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.k = int(self.config.get("k", 5))
        self.device_config = self.config.get("device", "auto")
        self.device = None
        self._ml_available = False
        self._feature_extractor = None
        self._preprocess = None
        self._prdc = None

    def setup(self) -> None:
        try:
            import prdc  # noqa: F401
        except ImportError:
            logger.warning(
                "prdc not installed; generative_distribution_metrics unavailable "
                "(pip install prdc)"
            )
            return
        try:
            import torch
            import torchvision
            from ayase.runtime import resolve_torch_device, shared_runtime_resource

            device = resolve_torch_device(self.device_config)
            self.device = torch.device(device)

            def load_vgg():
                weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1
                model = torchvision.models.vgg16(weights=weights).to(self.device).eval()
                # fc7 activations: everything up to the final classifier layer.
                extractor = torch.nn.Sequential(
                    model.features, model.avgpool, torch.nn.Flatten(1),
                    *list(model.classifier.children())[:-1],
                )
                return extractor, weights.transforms()

            self._feature_extractor, self._preprocess = shared_runtime_resource(
                self, ("vgg16_fc7", device), load_vgg
            )
            self._ml_available = True
            logger.info(f"Generative distribution metrics initialised on {self.device}")

        except ImportError:
            logger.warning("torch/torchvision not installed")
        except Exception as e:
            logger.warning(f"Failed to setup generative distribution metrics: {e}")

    def extract_features(self, sample: Sample) -> Optional[np.ndarray]:
        """Extract a VGG16 fc7 embedding from a representative frame."""
        if self._feature_extractor is None:
            return None

        try:
            import torch

            rgb = load_representative_frame(sample.path, color="rgb")
            if rgb is None:
                return None
            img = self._preprocess(arrays_to_pil([rgb])[0]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self._feature_extractor(img)
            return emb.squeeze(0).detach().float().cpu().numpy()

        except Exception as e:
            logger.debug(f"Feature extraction failed for {sample.path}: {e}")
            return None

    def compute_distribution_metric(
        self, features: List[np.ndarray], reference_features: Optional[List[np.ndarray]] = None
    ) -> Optional[float]:
        """Compute precision, recall, density, coverage via ``prdc``."""
        from prdc import compute_prdc

        gen = np.stack(features)

        if reference_features and len(reference_features) > 0:
            real = np.stack(reference_features)
        else:
            logger.info(
                "generative_distribution_metrics: no reference features "
                "provided; metrics are undefined without a reference set"
            )
            return None

        # Published prdc protocol needs > k samples in each set.
        if len(real) <= self.k or len(gen) <= self.k:
            logger.info(
                "generative_distribution_metrics: too few samples "
                f"(real={len(real)}, gen={len(gen)}, k={self.k})"
            )
            return None

        results = compute_prdc(
            real_features=real, fake_features=gen, nearest_k=self.k
        )
        precision = float(results["precision"])
        recall = float(results["recall"])
        density = float(results["density"])
        coverage = float(results["coverage"])

        # Store all metrics via pipeline
        if hasattr(self, "pipeline") and self.pipeline:
            if hasattr(self.pipeline, "add_dataset_metric"):
                self.pipeline.add_dataset_metric("precision", precision)
                self.pipeline.add_dataset_metric("recall", recall)
                self.pipeline.add_dataset_metric("coverage", coverage)
                self.pipeline.add_dataset_metric("density", density)

        logger.info(
            f"Generative metrics: P={precision:.3f} R={recall:.3f} "
            f"D={density:.3f} C={coverage:.3f}"
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
