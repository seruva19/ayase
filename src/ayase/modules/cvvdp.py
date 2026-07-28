"""ColorVideoVDP display-aware full-reference image/video quality.

ColorVideoVDP (SIGGRAPH 2024) jointly models chromatic, achromatic, spatial,
and temporal visual sensitivity under an explicit display model. It reports
quality in Just-Objectionable-Difference (JOD) units: an identical reference
scores 10 and lower values indicate increasingly visible distortion. Scores
may be negative for extremely different content.

The implementation uses the authors' upstream MIT-licensed ``cvvdp`` package.
It supports SDR and HDR display models and processes both image and video file
pairs without substituting a framewise proxy.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)


class ColorVideoVDPModule(ReferenceBasedModule):
    """Run the upstream ColorVideoVDP metric on a test/reference pair."""

    name = "cvvdp"
    description = "ColorVideoVDP display-aware color image/video FR quality"
    metric_field = "cvvdp_score"
    default_config = {
        "display_name": "standard_fhd",
        "device": "auto",
        "gpu_mem_gb": None,
        "max_frames": None,
    }
    models = [
        {
            "id": "cvvdp",
            "type": "pip_package",
            "install": "pip install 'cvvdp>=0.5.6,<0.6'",
            "task": "ColorVideoVDP image/video perceptual metric",
            "auto_download": False,
            "notes": "MIT; calibration and display-model data ship in the package",
        }
    ]
    metric_info = {
        "cvvdp_score": (
            "ColorVideoVDP quality in JOD units (10=reference quality, "
            "lower=worse; can be negative)"
        )
    }
    metric_groups = {"cvvdp_score": "fr_quality"}

    def __init__(self, config=None):
        super().__init__(config)
        self.display_name = str(self.config.get("display_name", "standard_fhd"))
        self.device_config = str(self.config.get("device", "auto"))
        gpu_mem = self.config.get("gpu_mem_gb")
        self.gpu_mem_gb = float(gpu_mem) if gpu_mem is not None else None
        max_frames = self.config.get("max_frames")
        self.max_frames = int(max_frames) if max_frames is not None else None

        self.device = None
        self._metric = None
        self._pycvvdp = None
        self._backend: Optional[str] = None

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            import pycvvdp
            import torch

            if self.device_config == "auto":
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
            elif self.device_config in {"cpu", "cuda"}:
                if self.device_config == "cuda" and not torch.cuda.is_available():
                    logger.warning("ColorVideoVDP requested CUDA but no CUDA GPU is available")
                    return
                self.device = torch.device(self.device_config)
            else:
                logger.warning(
                    "ColorVideoVDP device must be 'auto', 'cpu', or 'cuda', got %r",
                    self.device_config,
                )
                return

            self._metric = pycvvdp.cvvdp(
                display_name=self.display_name,
                device=self.device,
                heatmap=None,
                quiet=True,
                gpu_mem=self.gpu_mem_gb,
            )
            self._pycvvdp = pycvvdp
            self._backend = "cvvdp"
            logger.info(
                "ColorVideoVDP initialised on %s with display model %s",
                self.device,
                self.display_name,
            )
        except ImportError:
            logger.warning(
                "ColorVideoVDP unavailable; install the upstream 'cvvdp' package"
            )
        except Exception as exc:
            logger.warning("ColorVideoVDP setup failed: %s", exc)
            self._backend = None

    def compute_reference_score(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        if (
            self._backend != "cvvdp"
            or self._metric is None
            or self._pycvvdp is None
        ):
            return None

        try:
            test, reference, dim_order, fps = self._load_pair(
                sample_path, reference_path
            )
            score, _stats = self._metric.predict(
                test,
                reference,
                dim_order=dim_order,
                frames_per_second=fps,
            )
            if hasattr(score, "item"):
                score = score.item()
            value = float(score)
            if not np.isfinite(value) or value > 10.0001:
                logger.warning("ColorVideoVDP produced invalid JOD score: %r", value)
                return None
            return value
        except Exception as exc:
            logger.warning(
                "ColorVideoVDP failed for %s vs %s: %s",
                sample_path,
                reference_path,
                exc,
            )
            return None

    def _load_pair(self, sample_path: Path, reference_path: Path):
        """Decode a matched pair for the upstream array-based CVVDP API."""
        image_extensions = {
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff",
            ".exr",
            ".hdr",
        }
        sample_is_image = sample_path.suffix.lower() in image_extensions
        reference_is_image = reference_path.suffix.lower() in image_extensions
        if sample_is_image != reference_is_image:
            raise ValueError("ColorVideoVDP requires two images or two videos")

        if sample_is_image:
            import imageio.v3 as iio

            test = np.asarray(iio.imread(sample_path))
            reference = np.asarray(iio.imread(reference_path))
            if test.shape != reference.shape:
                raise ValueError(
                    "ColorVideoVDP image dimensions must match: "
                    f"{test.shape} != {reference.shape}"
                )
            if test.ndim == 2:
                return test, reference, "HW", 0
            if test.ndim != 3 or test.shape[-1] not in {3, 4}:
                raise ValueError(
                    f"Unsupported ColorVideoVDP image shape: {test.shape}"
                )
            if test.shape[-1] == 4:
                test = test[..., :3]
                reference = reference[..., :3]
            return test, reference, "HWC", 0

        from decord import VideoReader, cpu

        test_reader = VideoReader(str(sample_path), ctx=cpu(0), num_threads=1)
        reference_reader = VideoReader(
            str(reference_path), ctx=cpu(0), num_threads=1
        )
        test_count = len(test_reader)
        reference_count = len(reference_reader)
        if test_count == 0 or reference_count == 0:
            raise ValueError("ColorVideoVDP received an empty video")
        if test_count != reference_count:
            raise ValueError(
                "ColorVideoVDP video frame counts must match: "
                f"{test_count} != {reference_count}"
            )

        test_fps = float(test_reader.get_avg_fps())
        reference_fps = float(reference_reader.get_avg_fps())
        if not np.isclose(test_fps, reference_fps, rtol=0.0, atol=1e-3):
            raise ValueError(
                "ColorVideoVDP video frame rates must match: "
                f"{test_fps:g} != {reference_fps:g}"
            )

        count = (
            min(test_count, self.max_frames)
            if self.max_frames is not None
            else test_count
        )
        indices = list(range(count))
        test = test_reader.get_batch(indices).asnumpy()
        reference = reference_reader.get_batch(indices).asnumpy()
        if test.shape != reference.shape:
            raise ValueError(
                "ColorVideoVDP video dimensions must match: "
                f"{test.shape} != {reference.shape}"
            )
        return test, reference, "FHWC", test_fps

    def on_dispose(self) -> None:
        self._metric = None
        self._pycvvdp = None
        self._backend = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        super().on_dispose()
