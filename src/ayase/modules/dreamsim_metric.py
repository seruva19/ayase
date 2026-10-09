"""Measure perceptual distance with DreamSim on images or sampled video frames.

Requires ``sample.reference_path``: paired frames are averaged. Without a
reference the sample is not scored — mean DreamSim between consecutive frames
of the same video is a different quantity and is not emitted under this field.
Lower is more similar; the distance has no fixed range.
Basis: https://github.com/ssundaram21/dreamsim
"""

import logging
import os
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class DreamSimModule(PipelineModule):
    name = "dreamsim"
    provenance = {"dreamsim": "adapted"}
    sources = {
        "dreamsim": "DreamSim (Fu et al., NeurIPS 2023) — https://github.com/ssundaram21/dreamsim",
    }
    deviations = {
        "dreamsim": "DreamSim is defined for image pairs; video is scored as the mean over position-paired uniformly sampled frames.",
    }
    description = "DreamSim foundation model perceptual similarity (CLIP+DINO ensemble)"
    default_config = {"subsample": 8, "model_type": "ensemble"}
    metric_groups = {
        "dreamsim": "fr_quality",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._model = None
        self._preprocess = None
        self._backend = "unavailable"

    def setup(self) -> None:
        torch = None
        original_hub_dir = None
        try:
            import torch

            original_hub_dir = torch.hub.get_dir()
            self._ensure_dino_cached()
            from dreamsim import dreamsim

            model, preprocess = dreamsim(pretrained=True)
            self._model = model
            self._preprocess = preprocess
            self._ml_available = True
            self._backend = "dreamsim"
            logger.info("DreamSim model loaded")
        except (ImportError, Exception) as e:
            logger.warning("DreamSim unavailable: %s", e)
        finally:
            # DreamSim 0.2.1 calls torch.hub.set_dir("./models") while loading its
            # ensemble. Keep that package-local choice from leaking into later Ayase
            # modules (notably DINOv2 and DOVER) in the same process.
            if torch is not None and original_hub_dir is not None:
                try:
                    torch.hub.set_dir(original_hub_dir)
                except Exception as e:
                    logger.warning("Failed to restore torch.hub directory: %s", e)

    def _prep_batch(self, pil_images):
        """Preprocess a list of PIL images into one ``(N, 3, H, W)`` device tensor.

        DreamSim's ``preprocess`` returns a batched ``(1, 3, H, W)`` tensor per
        image (all resized to the same size), so they concatenate cleanly into a
        single batch that DreamSim scores in one forward pass.
        """
        import torch

        tensors = []
        for pil in pil_images:
            t = self._preprocess(pil)
            if t.dim() == 3:
                t = t.unsqueeze(0)
            tensors.append(t)
        batch = torch.cat(tensors, dim=0)
        return batch.to(next(self._model.parameters()).device)

    def _load_frames(self, path):
        """Uniformly sampled RGB PIL frames (image → one) from the shared cache."""
        from ayase.image import arrays_to_pil, sample_frames

        subsample = self.config.get("subsample", 8)
        arrays = sample_frames(path, max_frames=subsample, color="rgb")
        return arrays_to_pil(arrays)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available:
            return sample

        reference_path = getattr(sample, "reference_path", None)
        if reference_path is None:
            # DreamSim is a full-reference metric; no reference → no score.
            return sample

        try:
            import torch

            ref_frames = self._load_frames(reference_path)
            dist_frames = self._load_frames(sample.path)
            if not ref_frames or not dist_frames:
                return sample

            # Pair frames by position (a single image broadcasts to every video
            # frame), then score every pair in ONE batched forward pass.
            n = max(len(ref_frames), len(dist_frames))
            ref_pil = [ref_frames[i % len(ref_frames)] for i in range(n)]
            dist_pil = [dist_frames[i % len(dist_frames)] for i in range(n)]
            with torch.no_grad():
                distances = self._model(
                    self._prep_batch(ref_pil), self._prep_batch(dist_pil)
                )

            sample.quality_metrics.dreamsim = float(distances.mean().item())
        except Exception as e:
            logger.warning("DreamSim processing failed: %s", e)
        return sample

    # DreamSim's DINO ViT-B/16 backbone is fetched by ``dreamsim(pretrained=True)`` via
    # ``torch.hub`` from dl.fbaipublicfiles.com, which has no timeout and hangs on some
    # networks (and bypasses the mirror the rest of the pipeline relies on). Pre-place it
    # from the ayase-models HF mirror into torch.hub's checkpoints dir so DreamSim loads
    # it offline. Original: https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth
    _DINO_MIRROR_URL = "https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/dreamsim/dino_vitbase16_pretrain.pth"
    _DINO_FILENAME = "dino_vitbase16_pretrain.pth"

    def _ensure_dino_cached(self) -> None:
        """Pre-place DreamSim's base DINO backbone in the torch hub cache if absent."""
        import torch

        hub_dir = torch.hub.get_dir()
        cache_dir = os.path.join(hub_dir, "checkpoints")
        cached = os.path.join(cache_dir, self._DINO_FILENAME)
        if os.path.exists(cached):
            return
        os.makedirs(cache_dir, exist_ok=True)
        logger.info("Pre-caching base DINO backbone for DreamSim from the ayase mirror...")
        from ayase.config import download_model_file

        tmp = download_model_file(
            os.path.join("hub", "checkpoints", self._DINO_FILENAME),
            self._DINO_MIRROR_URL,
            os.path.dirname(hub_dir),  # parent of hub dir
        )
        # Move to torch cache if downloaded elsewhere
        if str(tmp) != cached and os.path.exists(str(tmp)):
            import shutil

            shutil.copy2(str(tmp), cached)


class DreamSimCompatModule(DreamSimModule):
    """Compatibility alias matching filename-based discovery."""

    name = "dreamsim_metric"
