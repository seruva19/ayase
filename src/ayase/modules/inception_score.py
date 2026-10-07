"""Dataset-level Inception Score (IS).

Published Inception Score (Salimans et al., NeurIPS 2016) is a dataset-level
quantity: ``exp(E[KL(p(y|x) || p(y))])`` over generated images using the
TF-Inception weights, averaged over 10 splits. Per-video values on a handful
of frames are not comparable to published IS numbers, so this module reports
it at dataset level through ``torch-fidelity`` — the canonical PyTorch
backend. When torch-fidelity is unavailable the metric is left unset.

Metric basis: https://arxiv.org/abs/1606.03498
"""

import logging
from typing import Optional, List

from ayase.models import Sample
from ayase.base_modules import BatchMetricModule

logger = logging.getLogger(__name__)


class InceptionScoreModule(BatchMetricModule):
    name = "inception_score"
    provenance = "published"
    sources = {
        "is_score": "Inception Score (Salimans et al., NeurIPS 2016), torch-fidelity backend — https://arxiv.org/abs/1606.03498",
    }
    deviations = {
        "is_score": "dataset-level IS (torch-fidelity, 10 splits, FID-Inception); for video — a representative frame; without the package the metric is not emitted",
    }
    description = "Inception Score (IS), dataset-level via torch-fidelity"
    default_config = {
        "isc_splits": 10,  # Published IS uses 10 splits
    }
    models = [
        {
            "id": "torch-fidelity",
            "type": "pip_package",
            "install": "pip install torch-fidelity",
            "task": "Inception Score backend (TF-Inception weights)",
        },
    ]
    metric_info = {
        "is_score": "Dataset-level Inception Score via torch-fidelity (higher=better)",
    }
    metric_groups = {
        "is_score": "distribution",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.isc_splits = self.config.get("isc_splits", 10)
        self._ml_available = False
        self._backend = None

    def setup(self):
        try:
            import torch_fidelity  # noqa: F401
            self._ml_available = True
            self._backend = "torch_fidelity"
            logger.info("Inception Score module initialised (torch-fidelity backend)")
        except ImportError:
            self._backend = "unavailable"
            logger.info(
                "Inception Score unavailable: requires torch-fidelity "
                "(pip install torch-fidelity)"
            )

    def extract_features(self, sample: Sample) -> Optional[str]:
        """Cache the sample path — torch-fidelity works on directories."""
        if not self._ml_available:
            return None
        return str(sample.path)

    def compute_distribution_metric(
        self, features: List[str], reference_features: Optional[List] = None
    ) -> Optional[float]:
        """Dataset IS via torch-fidelity. ``features`` are image paths."""
        try:
            import shutil
            import tempfile
            from pathlib import Path

            import torch_fidelity

            with tempfile.TemporaryDirectory() as gen_dir:
                # torch-fidelity needs image files; video samples contribute
                # their representative frame.
                import cv2
                from ayase.image import load_representative_frame

                written = 0
                for i, p in enumerate(features):
                    path = Path(p)
                    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
                        shutil.copy(str(path), Path(gen_dir) / f"{i:06d}{path.suffix}")
                        written += 1
                    else:
                        frame = load_representative_frame(path, color="rgb")
                        if frame is None:
                            continue
                        out = Path(gen_dir) / f"{i:06d}.png"
                        cv2.imwrite(str(out), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        written += 1

                if written < self.isc_splits:
                    logger.info(
                        "Inception Score: not enough samples (%d) for %d splits",
                        written, self.isc_splits,
                    )
                    return None

                metrics = torch_fidelity.calculate_metrics(
                    input1=gen_dir,
                    isc=True,
                    isc_splits=self.isc_splits,
                )
                return float(metrics["inception_score_mean"])

        except Exception as e:
            logger.error(f"Failed to compute Inception Score: {e}")
            return None

    def on_dispose(self) -> None:
        if len(self._feature_cache) < 2:
            logger.info(
                "Inception Score: not enough samples (%d)", len(self._feature_cache)
            )
            self._feature_cache = []
            self._reference_cache = []
            return

        try:
            score = self.compute_distribution_metric(self._feature_cache)
            if score is None:
                return
            logger.info(
                "Inception Score: %.4f (%d samples)", score, len(self._feature_cache)
            )
            if hasattr(self, "pipeline") and self.pipeline:
                if hasattr(self.pipeline, "add_dataset_metric"):
                    self.pipeline.add_dataset_metric("is_score", score)
        except Exception as e:
            logger.error(f"Inception Score failed: {e}")
        finally:
            self._feature_cache = []
            self._reference_cache = []
