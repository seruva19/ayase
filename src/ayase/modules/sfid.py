"""sFID — spatial Fréchet Inception Distance.

Dataset-level distribution metric. Unlike FID (which uses the global 2048-d
Inception pool features), sFID uses the *pre-pool spatial* activation of the
canonical FID Inception network — the 8x8x2048 ``Mixed_7c`` map — and treats
each spatial location as a separate 2048-d sample, mirroring the
guided-diffusion evaluator used for every published sFID number. Weights are
``fid_inception_v3`` via ``torch.hub``/pytorch-fid (the TF ``pt_inception``
port). When torch is unavailable the metric is left unset.

sfid_score — LOWER = better (closer distributions)
"""

import logging
from typing import List, Optional

import numpy as np

from ayase.image import load_representative_frame
from ayase.models import Sample
from ayase.base_modules import BatchMetricModule

logger = logging.getLogger(__name__)


class SFIDModule(BatchMetricModule):
    name = "sfid"
    provenance = "published"
    sources = {
        "sfid": "sFID (Nash et al., ICML 2021), guided-diffusion evaluator — FID-Inception via https://github.com/mseitzer/pytorch-fid",
    }
    deviations = {
        "sfid": "1 frame per sample (video → representative frame); without a reference the metric is not emitted",
    }
    description = "sFID spatial Fréchet Inception Distance (InceptionV3 spatial features, lower=better)"
    default_config = {
        "device": "auto",
        "resize": 299,
    }
    models = [
        {
            "id": "mseitzer/pytorch-fid:fid_inception_v3",
            "type": "torch_hub",
            "task": "Canonical FID InceptionV3 — pre-pool spatial features for sFID",
        },
    ]
    metric_info = {
        "sfid": "Spatial Fréchet Inception Distance on FID-Inception pre-pool features (lower=better)",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.device_config = self.config.get("device", "auto")
        self.resize = self.config.get("resize", 299)
        self._model = None
        self._transform = None
        self._device = "cpu"
        self._backend = "unavailable"

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            import torch
            from torchvision import transforms
            from ayase.runtime import resolve_torch_device, shared_runtime_resource

            self._device = resolve_torch_device(self.device_config)

            def load_inception():
                # Canonical FID weights (pt_inception port), retargeted to emit
                # the pre-pool 8x8x2048 spatial map (block 6 = Mixed_7c).
                model = torch.hub.load(
                    "mseitzer/pytorch-fid", "fid_inception_v3"
                ).to(self._device).eval()
                model.output_blocks = [6]
                model.last_included_block = 6
                return model

            self._model = shared_runtime_resource(
                self,
                ("sfid_fid_inception_spatial", str(self._device)),
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
            self._backend = "fid_inception_spatial"
            logger.info("sFID initialised with FID-Inception spatial features on %s", self._device)
        except ImportError:
            logger.warning("sFID unavailable: requires torch and torchvision; sfid left unset.")
        except Exception as e:
            logger.warning("sFID Inception setup failed (%s); metric disabled", e)

    def extract_features(self, sample: Sample) -> Optional[np.ndarray]:
        if self._backend != "fid_inception_spatial" or self._model is None:
            return None
        frame = load_representative_frame(sample.path, color="rgb")
        if frame is None:
            return None
        return self._extract_spatial(frame)

    def _extract_spatial(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Return the per-location spatial vectors of one image.

        The canonical sFID protocol treats every position of the pre-pool
        8x8x2048 map as a separate 2048-d sample, so this returns (64, 2048)
        vectors rather than one flattened descriptor.
        """
        try:
            import torch

            tensor = self._transform(np.ascontiguousarray(frame, dtype=np.uint8))
            tensor = tensor.unsqueeze(0).to(self._device)
            with torch.no_grad():
                spatial = self._model(tensor)  # (1, 2048, 8, 8)
                if isinstance(spatial, (list, tuple)):
                    spatial = spatial[0]
                # (B, C, H, W) -> (B*H*W, C): each spatial location a sample.
                spatial = spatial.permute(0, 2, 3, 1).reshape(-1, spatial.shape[1])
            return spatial.detach().cpu().numpy().astype(np.float64)
        except Exception as e:
            logger.debug("sFID spatial extraction failed: %s", e)
            return None

    def compute_distribution_metric(
        self, features: List, reference_features: Optional[List] = None
    ) -> Optional[float]:
        gen = np.concatenate(features).astype(np.float64)
        if not (reference_features and len(reference_features) >= 2):
            logger.info(
                "sfid: no reference features provided; "
                "metric is undefined without a reference set"
            )
            return None
        ref = np.concatenate(reference_features).astype(np.float64)
        return self._frechet_distance(gen, ref)

    def _frechet_distance(self, x: np.ndarray, y: np.ndarray) -> float:
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
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            score = diff @ diff + np.trace(cov_x + cov_y - 2.0 * covmean)
        except Exception:
            score = diff @ diff + np.trace(cov_x) + np.trace(cov_y)
        return float(max(score, 0.0))
