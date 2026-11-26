"""FID image distribution metric.

Computes the Fréchet Inception Distance between generated and reference image
feature distributions using the canonical pytorch-fid InceptionV3 pool3
features. Video inputs use one representative frame. Lower FID is better.
"""

import logging
from contextlib import nullcontext
from threading import Lock
from typing import List, Optional

import numpy as np

from ayase.base_modules import BatchMetricModule
from ayase.image import load_representative_frame
from ayase.models import Sample

logger = logging.getLogger(__name__)
_CUDNN_PRECISION_LOCK = Lock()


class FIDModule(BatchMetricModule):
    name = "fid"
    implementation_status = "implemented"
    implementation_notes = "Official pytorch-fid pool3 weights and Gaussian Fréchet distance; Ayase applies bicubic resizing and uses one representative frame per video."
    implementation_review = "Official pretrained checkpoint verified. CPU pool3 features match pytorch-fid; CUDA FP32 batch differences stayed below 1e-5 on fixed synthetic inputs."
    provenance = "adapted"
    sources = {
        "fid": "FID (Heusel et al., NeurIPS 2017) — fid_inception_v3 via https://github.com/mseitzer/pytorch-fid",
    }
    deviations = {
        "fid": "Ayase resizes inputs with bicubic interpolation and uses one representative frame per video; no score without references",
    }
    description = "Fréchet Inception Distance for image generation (batch metric)"
    default_config = {
        "device": "auto",
        "resize": 299,
    }
    required_packages = ["pytorch_fid"]
    models = [
        {
            "id": "pt_inception-2015-12-05-6726825d.pth",
            "type": "local",
            "url": "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth",
            "task": "Canonical FID InceptionV3 (pt_inception port)",
            "auto_download": True,
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
            from pytorch_fid.inception import InceptionV3
            from torchvision import transforms
            from ayase.runtime import resolve_torch_device, shared_model_weights

            self._device = resolve_torch_device(self.device_config)

            def load_inception():
                return InceptionV3(
                    output_blocks=(InceptionV3.BLOCK_INDEX_BY_DIM[2048],),
                    resize_input=False,
                ).to(self._device).eval()

            self._model = shared_model_weights(
                self,
                ("pytorch-fid", "0.3.0", 2048, str(self._device)),
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
            logger.warning("FID unavailable: pytorch-fid could not be imported.")
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
            return self._extract_inception_batch([frame])[0]
        except Exception as e:
            logger.debug("FID Inception extraction failed: %s", e)
            return None

    def _extract_inception_batch(self, frames: List[np.ndarray]) -> np.ndarray:
        import torch
        from ayase.runtime import torch_inference_context

        inputs = torch.stack([self._transform(frame) for frame in frames]).to(self._device)
        # FID's covariance estimate is sensitive to reduced precision.
        cudnn = torch.backends.cudnn
        with _CUDNN_PRECISION_LOCK if inputs.is_cuda else nullcontext():
            precision = (
                cudnn.flags(
                    enabled=cudnn.enabled,
                    benchmark=cudnn.benchmark,
                    benchmark_limit=cudnn.benchmark_limit,
                    deterministic=cudnn.deterministic,
                    allow_tf32=False,
                )
                if inputs.is_cuda
                else nullcontext()
            )
            with precision, torch_inference_context(str(self._device), "fp32", False):
                features = self._model(inputs)[0]
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        return features.flatten(1).cpu().numpy().astype(np.float64)

    def process_batch(self, samples: List[Sample]) -> List[Sample]:
        if len(samples) <= 1 or self._backend != "fid_inception_v3" or self._model is None:
            return super().process_batch(samples)

        try:
            generated, references = [], []
            for sample in samples:
                frame = load_representative_frame(sample.path, color="rgb")
                if frame is not None:
                    generated.append(frame)
                reference = sample.reference_path
                if reference is not None and reference.exists():
                    frame = load_representative_frame(reference, color="rgb")
                    if frame is not None:
                        references.append(frame)
            features = self._extract_inception_batch(generated) if generated else []
            reference_features = self._extract_inception_batch(references) if references else []
        except Exception as exc:
            logger.debug("FID batch extraction failed; retrying per sample: %s", exc)
            return super().process_batch(samples)

        self._feature_cache.extend(features)
        self._reference_cache.extend(reference_features)
        return samples

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

            covmean = linalg.sqrtm(cov_x @ cov_y)
            if not np.isfinite(covmean).all():
                eps = 1e-6
                offset = np.eye(cov_x.shape[0]) * eps
                covmean = linalg.sqrtm((cov_x + offset) @ (cov_y + offset))
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
