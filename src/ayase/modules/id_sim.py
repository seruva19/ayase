"""Fine-grained visual identity distance using the official ID-Sim metric.

ID-Sim (CVPR 2026) distinguishes intrinsic object identity from context and
viewpoint changes. Lower distance means stronger identity preservation. The
default DINOv2 ViT-B/14 variant uses author-published adapter weights and an
exact backbone checkpoint mirrored on Hugging Face for reproducible loading.
"""

import hashlib
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

_BACKBONE_REVISION = "c21c965fc3f0104bec14efa51504efd12f6e03cf"
_BACKBONE_SHA256 = "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"
_ADAPTER_REVISION = "bcd3f388bec7db42e75b165e235196833111c4ae"


class IDSimModule(PipelineModule):
    name = "id_sim"
    description = "ID-Sim fine-grained visual identity distance (CVPR 2026)"
    default_config = {
        "checkpoint": "dinov2_vitb14_cls_patch",
        "mode": "cls",
        "device": "auto",
        "models_dir": "models",
    }
    models = [
        {
            "id": "chaenayo/id-sim_dinov2_vitb14_cls_patch",
            "type": "huggingface",
            "task": "Official ID-Sim adapter and projection heads",
            "auto_download": True,
            "notes": f"MIT; pinned to revision {_ADAPTER_REVISION}",
        },
        {
            "id": "AkaneTendo25/ayase-runtime-assets",
            "type": "huggingface",
            "task": "Exact DINOv2 ViT-B/14 backbone used by ID-Sim",
            "auto_download": True,
            "notes": (
                f"Upstream Apache-2.0 checkpoint mirrored for HF-only loading; "
                f"pinned to revision {_BACKBONE_REVISION} and SHA-256 {_BACKBONE_SHA256}"
            ),
        },
    ]
    metric_info = {
        "id_sim_distance": "ID-Sim identity distance (lower=more similar)",
    }
    metric_groups = {"id_sim_distance": "face"}

    def __init__(self, config=None):
        super().__init__(config)
        self.checkpoint = str(self.config.get("checkpoint", "dinov2_vitb14_cls_patch"))
        self.mode = str(self.config.get("mode", "cls"))
        self.models_dir = str(self.config.get("models_dir", "models"))
        self._device = "cpu"
        self._model = None
        self._preprocess = None
        self._backend = "unavailable"

    def setup(self) -> None:
        if self.test_mode:
            return
        try:
            from ayase.config import download_model_file
            from ayase.runtime import resolve_torch_device
            from ayase.third_party.id_sim import id_sim

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            cache_dir = Path(self.models_dir) / "id_sim"
            backbone = download_model_file(
                "id_sim/checkpoints/dinov2_vitb14_pretrain.pth",
                f"https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/{_BACKBONE_REVISION}/"
                "id_sim/dinov2_vitb14_pretrain.pth",
                self.models_dir,
            )
            if self._sha256(backbone) != _BACKBONE_SHA256:
                raise RuntimeError("ID-Sim DINOv2 checkpoint SHA-256 mismatch")
            self._model, self._preprocess = id_sim(
                pretrained=True,
                device=self._device,
                cache_dir=str(cache_dir),
                id_sim_type=self.checkpoint,
            )
            self._backend = f"id-sim:{self.checkpoint}:{self.mode}"
        except Exception as exc:
            self._model = None
            self._preprocess = None
            logger.warning("ID-Sim setup failed: %s", exc)

    def process(self, sample: Sample) -> Sample:
        if self._model is None or self._preprocess is None:
            return sample
        reference = getattr(sample, "reference_path", None)
        if reference is None:
            return sample
        try:
            target = self._load_image(Path(sample.path))
            ref = self._load_image(Path(reference))
            if target is None or ref is None:
                return sample
            a = self._preprocess(ref).to(self._device)
            b = self._preprocess(target).to(self._device)
            distance = float(self._model(a, b, mode=self.mode).mean().detach().cpu())
            if not np.isfinite(distance):
                return sample
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.id_sim_distance = max(0.0, distance)
        except Exception as exc:
            logger.warning("ID-Sim failed for %s: %s", sample.path, exc)
        return sample

    @staticmethod
    def _load_image(path: Path):
        if path.is_dir():
            path = next(
                (p for p in sorted(path.iterdir()) if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}),
                None,
            )
            if path is None:
                return None
        with Image.open(path) as image:
            return image.convert("RGB")

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
