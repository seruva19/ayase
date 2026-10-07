"""FID image distribution metric.

Computes the Fréchet Inception Distance between generated and reference image
feature distributions using the canonical FID InceptionV3 weights
(``fid_inception_v3`` via ``torch.hub``/pytorch-fid — the TF ``pt_inception``
port used by every published FID number). When torch is unavailable the metric
is left unset (no heuristic stand-in). Lower FID is better.
"""

import logging
from typing import List, Optional

import numpy as np

from ayase.base_modules import BatchMetricModule
from ayase.image import load_representative_frame
from ayase.models import Sample

logger = logging.getLogger(__name__)


class FIDModule(BatchMetricModule):
    name = "fid"
    provenance = "adapted"
    sources = {
        "fid": "FID (Heusel et al., NeurIPS 2017) — fid_inception_v3 via https://github.com/mseitzer/pytorch-fid",
    }
    deviations = {
        "fid": "one representative frame per video; without a reference the metric is not emitted",
    }
    description = "Fréchet Inception Distance for image generation (batch metric)"
    default_config = {
        "device": "auto",
        "resize": 299,
    }
    models = [
        {
            "id": "mseitzer/pytorch-fid:fid_inception_v3",
            "type": "torch_hub",
            "task": "Canonical FID InceptionV3 (pt_inception port)",
        },
    ]
    metric_info = {
        "fid": "Fréchet Inception Distance between generated and reference image sets (lower=better)",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.device_config = self.config.get("device", "auto")
        self.resize = self.config.get("resize", 299)
        self._backend = "unavailable"
        self._model = None
        self._transform = None
        self._device = "cpu"

    def setup(self) -> None:
        try:
            import torch
            from torchvision import transforms
            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.device_config)

            def load_inception():
                # Canonical FID weights: the pytorch-fid port of TF's
                # pt_inception-2015-12-05 checkpoint. Inputs are [0, 1] NCHW.
                return torch.hub.load(
                    "mseitzer/pytorch-fid", "fid_inception_v3"
                ).to(self._device).eval()

            from ayase.runtime import shared_runtime_resource

            self._model = shared_runtime_resource(
                self,
                ("fid_inception_v3_pt_inception", str(self._device)),
                load_inception,
            )
            self._transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(
                    (self.resize, self.resize),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
            ])
            self._backend = "fid_inception_v3"
            logger.info("FID initialised with pytorch-fid fid_inception_v3 on %s", self._device)
        except ImportError:
            logger.warning("FID unavailable: requires torch and torchvision.")
        except Exception as e:
            logger.warning("FID Inception setup failed (%s); metric disabled", e)

    def extract_features(self, sample: Sample) -> Optional[np.ndarray]:
        if self._backend != "fid_inception_v3" or self._model is None:
            return None
        frame = load_representative_frame(sample.path, color="rgb")
        if frame is None:
            return None
        return self._extract_inception(frame)

    def compute_distribution_metric(
        self,
        features: List[np.ndarray],
        reference_features: Optional[List[np.ndarray]] = None,
    ) -> Optional[float]:
        gen = np.stack(features).astype(np.float64)
        if not reference_features:
            logger.info(
                "fid: no reference features provided; "
                "metric is undefined without a reference set"
            )
            return None
        ref = np.stack(reference_features).astype(np.float64)
        return self._frechet_distance(gen, ref)

    def _extract_inception(self, frame: np.ndarray) -> Optional[np.ndarray]:
        try:
            import torch

            tensor = self._transform(frame.astype(np.uint8)).unsqueeze(0).to(self._device)
            with torch.no_grad():
                features = self._model(tensor)
                if hasattr(features, "logits"):
                    features = features.logits
            return features.detach().cpu().numpy().reshape(-1).astype(np.float64)
        except Exception as e:
            logger.debug("FID Inception extraction failed: %s", e)
            return None

    def _frechet_distance(self, x: np.ndarray, y: np.ndarray) -> Optional[float]:
        mu_x = np.mean(x, axis=0)
        mu_y = np.mean(y, axis=0)
        diff = mu_x - mu_y

        if len(x) < 2 or len(y) < 2:
            return float(diff @ diff)

        cov_x = np.atleast_2d(np.cov(x, rowvar=False))
        cov_y = np.atleast_2d(np.cov(y, rowvar=False))
        try:
            from scipy import linalg

            covmean, _ = linalg.sqrtm(cov_x @ cov_y, disp=False)
            if not np.isfinite(covmean).all():
                eps = 1e-6
                offset = np.eye(cov_x.shape[0]) * eps
                covmean, _ = linalg.sqrtm((cov_x + offset) @ (cov_y + offset), disp=False)
            if not np.isfinite(covmean).all():
                logger.warning("FID covariance square root is non-finite; metric left unset")
                return None
            if np.iscomplexobj(covmean):
                max_imag = float(np.max(np.abs(np.diag(covmean).imag)))
                if max_imag > 1e-3:
                    logger.warning(
                        "FID covariance square root has significant imaginary residue "
                        "(%.6g); metric left unset",
                        max_imag,
                    )
                    return None
                covmean = covmean.real
            score = diff @ diff + np.trace(cov_x + cov_y - 2.0 * covmean)
        except Exception as e:
            logger.warning("FID covariance square root failed; metric left unset: %s", e)
            return None
        if not np.isfinite(score):
            logger.warning("FID score is non-finite; metric left unset")
            return None
        return float(max(score, 0.0))
