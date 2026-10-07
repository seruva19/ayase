"""Full-reference image quality scoring with PyIQA's A-FINE model.

A-FINE (Chen et al., CVPR 2025) is a full-reference metric blending fidelity
and naturalness branches; the published ``afine`` model requires a reference
image (``sample.reference_path``) and emits a higher-is-better quality score.
``afine`` is available from the PyIQA main branch rather than a released
PyIQA package; package versions without it leave the module unavailable.

Model basis: https://github.com/chaofengc/IQA-PyTorch
"""

import logging
from typing import List, Optional

import numpy as np

from ayase.image import sample_frames
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class AFINEModule(PipelineModule):
    name = "afine"
    provenance = "published"
    requires_external_backend = True  # afine ships only in pyiqa main, not released pyiqa
    sources = {
        "afine_score": "A-FINE (Chen et al., CVPR 2025), pyiqa ``afine`` FR model — https://github.com/chaofengc/IQA-PyTorch/blob/main/pyiqa/default_model_configs.py",
    }
    description = "A-FINE adaptive fidelity-naturalness IQA (CVPR 2025)"
    default_config = {"subsample": 4}
    metric_groups = {
        "afine_score": "nr_quality",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._model = None
        self._device = None
        self._backend = "unavailable"

    def setup(self) -> None:
        try:
            import pyiqa
            import torch
            from ayase.runtime import resolve_torch_device

            device = resolve_torch_device(self.config.get("device", "auto"))
            # Published A-FINE is the full-reference blend.
            self._model = pyiqa.create_metric("afine", device=device)
            try:
                self._device = next(self._model.parameters()).device
            except StopIteration:
                self._device = torch.device(device)
            self._ml_available = True
            self._backend = "pyiqa"
            logger.info("A-FINE (NR) model loaded on %s", device)
        except ImportError:
            logger.warning("A-FINE unavailable: pyiqa is not installed (pip install pyiqa)")
        except Exception as e:
            logger.warning("A-FINE unavailable: %s", e)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available:
            return sample

        # Published A-FINE is full-reference — nothing to score without one.
        reference = getattr(sample, "reference_path", None)
        if reference is None:
            return sample

        try:
            import cv2
            import torch

            ref = cv2.imread(str(reference), cv2.IMREAD_COLOR)
            if ref is None:
                return sample
            ref_tensor = (
                torch.from_numpy(np.ascontiguousarray(ref[..., ::-1]))
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                / 255.0
            ).to(self._device)

            frames = self._load_frames(sample)
            if not frames:
                return sample

            scores = []
            for frame in frames:
                tensor = (
                    torch.from_numpy(np.ascontiguousarray(frame))
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                    / 255.0
                )
                tensor = tensor.to(self._device)
                # Match the reference height/width when they differ — pyiqa
                # FR metrics require aligned inputs.
                if tensor.shape[-2:] != ref_tensor.shape[-2:]:
                    tensor = torch.nn.functional.interpolate(
                        tensor, size=ref_tensor.shape[-2:], mode="area"
                    )
                with torch.no_grad():
                    score = self._model(tensor, ref_tensor).item()
                scores.append(score)

            sample.quality_metrics.afine_score = float(np.mean(scores))
        except Exception as e:
            logger.warning("A-FINE processing failed: %s", e)
        return sample

    def _load_frames(self, sample: Sample) -> List[np.ndarray]:
        subsample = self.config.get("subsample", 4)
        try:
            return sample_frames(sample.path, max_frames=subsample, color="rgb")
        except Exception as e:
            logger.debug("A-FINE frame load failed for %s: %s", sample.path, e)
            return []
