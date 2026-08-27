"""MaSC masked concept-preservation similarity for personalized generation.

Implements the public MaSC masked-maxcos equation: each foreground reference
patch is matched to its best generated-image patch, then those similarities are
averaged. A reference concept mask is required; higher scores are better.
"""

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

_MODEL_ID = "google/siglip2-so400m-patch16-naflex"
_MODEL_REVISION = "cc24074f717b612951c2dead130904ab9b65a81e"


class MaSCModule(PipelineModule):
    name = "masc"
    description = "MaSC masked-maxcos concept preservation similarity"
    default_config = {
        "model": _MODEL_ID,
        "max_num_patches": 1024,
        "foreground_threshold": 0.5,
        "reference_mask_path": None,
        "device": "auto",
        "models_dir": "models",
    }
    models = [
        {
            "id": _MODEL_ID,
            "type": "huggingface",
            "task": "Frozen SigLIP2 NaFlex patch embeddings for MaSC",
            "auto_download": True,
            "notes": f"Apache-2.0; pinned to revision {_MODEL_REVISION}",
        }
    ]
    metric_info = {
        "masc_concept_preservation": (
            "MaSC foreground-reference masked-maxcos score (-1 to 1, higher=better)"
        ),
    }
    metric_groups = {"masc_concept_preservation": "alignment"}

    def __init__(self, config=None):
        super().__init__(config)
        self.model_id = str(self.config.get("model", _MODEL_ID))
        self.max_num_patches = int(self.config.get("max_num_patches", 1024))
        self.foreground_threshold = float(self.config.get("foreground_threshold", 0.5))
        self.reference_mask_path = self.config.get("reference_mask_path")
        self.models_dir = str(self.config.get("models_dir", "models"))
        self._device = "cpu"
        self._model = None
        self._processor = None
        self._torch = None
        self._backend = "unavailable"

    def setup(self) -> None:
        if self.test_mode:
            return
        try:
            import torch
            from transformers import AutoModel, AutoProcessor

            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            common = {
                "revision": _MODEL_REVISION if self.model_id == _MODEL_ID else None,
                "cache_dir": str(Path(self.models_dir) / "huggingface"),
            }
            self._processor = AutoProcessor.from_pretrained(self.model_id, **common)
            self._model = AutoModel.from_pretrained(self.model_id, **common)
            self._model.eval().to(self._device)
            self._torch = torch
            self._backend = f"masc:{self.model_id}:masked-maxcos"
        except Exception as exc:
            self._model = None
            self._processor = None
            logger.warning("MaSC setup failed: %s", exc)

    def process(self, sample: Sample) -> Sample:
        if self._model is None or self._processor is None or self._torch is None:
            return sample
        if sample.reference_path is None:
            return sample
        mask_path = self._resolve_reference_mask(sample)
        if mask_path is None:
            return sample
        try:
            with Image.open(sample.reference_path) as image:
                reference = image.convert("RGB")
            with Image.open(sample.path) as image:
                generated = image.convert("RGB")
            with Image.open(mask_path) as image:
                mask = np.asarray(image.convert("L"), dtype=np.float32)
            score = self._score(reference, generated, mask)
            if not np.isfinite(score):
                return sample
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.masc_concept_preservation = score
        except Exception as exc:
            logger.warning("MaSC failed for %s: %s", sample.path, exc)
        return sample

    def _score(self, reference: Image.Image, generated: Image.Image, mask: np.ndarray) -> float:
        ref_patches, hp, wp = self._encode(reference)
        out_patches, _, _ = self._encode(generated)
        torch = self._torch
        ref_unit = torch.nn.functional.normalize(ref_patches.float(), dim=-1)
        out_unit = torch.nn.functional.normalize(out_patches.float(), dim=-1)
        best = (ref_unit @ out_unit.T).max(dim=-1).values
        mask_tensor = torch.from_numpy(mask / (255.0 if mask.max() > 1.0 else 1.0))
        mask_tensor = mask_tensor[None, None].to(self._device)
        mask_tensor = torch.nn.functional.interpolate(
            mask_tensor, size=(hp, wp), mode="bilinear", align_corners=False
        )[0, 0]
        foreground = mask_tensor.flatten() > self.foreground_threshold
        if not bool(foreground.any()):
            return float("nan")
        return float(best[foreground].mean().detach().cpu())

    def _encode(self, image: Image.Image):
        inputs = self._processor.image_processor(
            images=image,
            max_num_patches=self.max_num_patches,
            return_tensors="pt",
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        vision_kwargs = {
            "pixel_values": inputs["pixel_values"],
            "pixel_attention_mask": inputs["pixel_attention_mask"],
            "spatial_shapes": inputs["spatial_shapes"],
        }
        try:
            output = self._model.vision_model(**vision_kwargs)
        except TypeError as exc:
            # transformers 4.x calls this argument ``attention_mask``;
            # transformers 5.x renamed it to ``pixel_attention_mask``.
            if "pixel_attention_mask" not in str(exc):
                raise
            vision_kwargs["attention_mask"] = vision_kwargs.pop("pixel_attention_mask")
            output = self._model.vision_model(**vision_kwargs)
        valid = inputs["pixel_attention_mask"][0] > 0
        hp, wp = (int(value) for value in inputs["spatial_shapes"][0].tolist())
        patches = output.last_hidden_state[0][valid]
        if patches.shape[0] != hp * wp:
            raise RuntimeError("MaSC patch-grid shape mismatch")
        return patches, hp, wp

    def _resolve_reference_mask(self, sample: Sample):
        explicit = sample.reference_mask_path or self.reference_mask_path
        if explicit:
            path = Path(explicit)
            return path if path.is_file() else None
        reference = Path(sample.reference_path)
        sidecar = reference.with_name(f"{reference.stem}.mask.png")
        return sidecar if sidecar.is_file() else None
