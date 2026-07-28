"""Intel CGVQM full-reference metric for rendered-video artifacts.

Uses the upstream CGVQM-2/CGVQM-5 calibration and Kinetics-400 R3D-18
features. It measures spatial and temporal rendering errors such as flicker,
ghosting, aliasing, disocclusion, and reconstruction artifacts. Higher is
better on the paper's perceptual 0-100 scale.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from ayase.base_modules import ReferenceBasedModule
from ayase.config import download_model_file
from ayase.models import QualityMetrics, Sample

logger = logging.getLogger(__name__)

_R3D_18_URL = "https://download.pytorch.org/models/r3d_18-b3b3357e.pth"


class CGVQMModule(ReferenceBasedModule):
    name = "cgvqm"
    description = "Intel CGVQM full-reference rendered-video quality"
    default_config = {
        "variant": "cgvqm-5",
        "patch_pool": "mean",
        "patch_scale": 4,
        "device": "auto",
        "models_dir": "models",
    }
    models = [
        {
            "id": "cgvqm/r3d_18-b3b3357e.pth",
            "type": "local",
            "url": _R3D_18_URL,
            "task": "Kinetics-400 R3D-18 features used by upstream CGVQM",
            "auto_download": True,
        },
        {
            "id": "IntelLabs/cgvqm",
            "type": "other",
            "url": "https://github.com/IntelLabs/cgvqm",
            "task": "Vendored upstream CGVQM-2 and CGVQM-5 calibration weights",
        },
    ]
    metric_info = {
        "cgvqm": (
            "Intel CGVQM perceptual rendered-video quality "
            "(full-reference, nominal 0-100, higher=better)"
        ),
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.variant = str(self.config.get("variant", "cgvqm-5")).lower()
        self.patch_pool = str(self.config.get("patch_pool", "mean")).lower()
        self.patch_scale = int(self.config.get("patch_scale", 4))
        self.device_config = str(self.config.get("device", "auto"))
        self.models_dir = str(self.config.get("models_dir", "models"))
        self._backend = "unavailable"
        self._model = None
        self._torch = None
        self._preprocess = None
        self._device = "cpu"

    def setup(self) -> None:
        if self.variant not in ("cgvqm-2", "cgvqm-5"):
            logger.warning("CGVQM variant must be cgvqm-2 or cgvqm-5")
            return
        if self.patch_pool not in ("mean", "max") or self.patch_scale < 1:
            logger.warning("CGVQM requires patch_pool mean|max and patch_scale >= 1")
            return

        try:
            import torch

            from ayase.third_party.cgvqm.cgvqm import CGVQM
            from ayase.third_party.cgvqm.utils import resnet18
            from ayase.third_party.cgvqm.utils.utils import preprocess

            if self.device_config in ("auto", ""):
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device_config

            backbone_path = download_model_file(
                "cgvqm/r3d_18-b3b3357e.pth",
                _R3D_18_URL,
                self.models_dir,
            )
            backbone = resnet18.r3d_18(weights=None)
            state = torch.load(backbone_path, map_location="cpu", weights_only=True)
            backbone.load_state_dict(state)
            backbone.__class__ = CGVQM

            source_dir = Path(__file__).resolve().parents[1] / "third_party" / "cgvqm"
            calibration = source_dir / "weights" / f"{self.variant}.pickle"
            num_layers = 3 if self.variant == "cgvqm-2" else 6
            backbone.to(device)
            backbone.init_weights(str(calibration), num_layers=num_layers)
            backbone.eval()

            self._torch = torch
            self._model = backbone
            self._preprocess = preprocess
            self._device = device
            self._backend = f"cgvqm_{self.variant}"
            logger.info("CGVQM initialised with %s on %s", self.variant, device)
        except ImportError as exc:
            logger.warning("CGVQM requires torch, torchvision, and av: %s", exc)
        except Exception as exc:
            logger.warning("CGVQM backend initialisation failed: %s", exc)

    def compute_reference_score(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        # Published CGVQM is a spatiotemporal video metric. Reporting it on a
        # single image would be a new, unvalidated metric.
        return None

    @staticmethod
    def _read_video(path: Path) -> Tuple[Optional[np.ndarray], float]:
        capture = cv2.VideoCapture(str(path))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
        if not frames:
            return None, fps
        return np.stack(frames, axis=0), fps

    def _load_aligned_videos(
        self, distorted_path: Path, reference_path: Path
    ) -> Optional[Tuple[object, object, float]]:
        distorted, _ = self._read_video(distorted_path)
        reference, fps = self._read_video(reference_path)
        if distorted is None or reference is None or fps <= 0:
            return None

        target_h, target_w = reference.shape[1:3]
        if distorted.shape[1:3] != (target_h, target_w):
            distorted = np.stack(
                [cv2.resize(frame, (target_w, target_h)) for frame in distorted],
                axis=0,
            )
        if len(distorted) != len(reference):
            indices = np.linspace(0, len(distorted) - 1, len(reference)).round().astype(int)
            distorted = distorted[indices]

        torch = self._torch
        distorted_tensor = torch.from_numpy(np.ascontiguousarray(distorted)).permute(
            0, 3, 1, 2
        )
        reference_tensor = torch.from_numpy(np.ascontiguousarray(reference)).permute(
            0, 3, 1, 2
        )
        return self._preprocess(distorted_tensor), self._preprocess(reference_tensor), fps

    def _score_video(self, path: Path, reference_path: Path) -> Optional[float]:
        if (
            not self._backend.startswith("cgvqm_")
            or self._model is None
            or self._torch is None
            or self._preprocess is None
        ):
            return None
        aligned = self._load_aligned_videos(path, reference_path)
        if aligned is None:
            return None
        distorted, reference, fps = aligned
        torch = self._torch

        distorted = distorted.unsqueeze(0)
        reference = reference.unsqueeze(0)
        clip_size = max(1, int(min(fps, 30)))
        patch_h = max(1, int(distorted.shape[3] / self.patch_scale))
        patch_w = max(1, int(distorted.shape[4] / self.patch_scale))
        pad = (
            0,
            (patch_w - distorted.shape[4] % patch_w) % patch_w,
            0,
            (patch_h - distorted.shape[3] % patch_h) % patch_h,
            0,
            (clip_size - distorted.shape[2] % clip_size) % clip_size,
        )
        distorted = torch.nn.functional.pad(distorted, pad, mode="replicate")
        reference = torch.nn.functional.pad(reference, pad, mode="replicate")

        errors = []
        with torch.inference_mode():
            for start in range(0, distorted.shape[2], clip_size):
                for top in range(0, distorted.shape[3], patch_h):
                    for left in range(0, distorted.shape[4], patch_w):
                        distorted_patch = distorted[
                            :,
                            :,
                            start : start + clip_size,
                            top : top + patch_h,
                            left : left + patch_w,
                        ].to(self._device)
                        reference_patch = reference[
                            :,
                            :,
                            start : start + clip_size,
                            top : top + patch_h,
                            left : left + patch_w,
                        ].to(self._device)
                        error, _ = self._model.feature_diff(
                            distorted_patch, reference_patch
                        )
                        errors.append(error.detach())
        if not errors:
            return None
        if self.patch_pool == "max":
            error = torch.stack(errors).max()
        else:
            error = torch.stack(errors).mean()
        score = float((100.0 - error).cpu().item())
        return score if np.isfinite(score) else None

    def process(self, sample: Sample) -> Sample:
        if not self._backend.startswith("cgvqm_") or not sample.is_video:
            return sample
        reference = getattr(sample, "reference_path", None)
        if reference is None:
            return sample
        reference_path = Path(reference)
        if not reference_path.exists():
            return sample
        try:
            score = self._score_video(sample.path, reference_path)
            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.cgvqm = score
        except Exception as exc:
            logger.warning("CGVQM failed for %s: %s", sample.path, exc)
        return sample

    def on_dispose(self) -> None:
        super().on_dispose()
        self._model = None
        if self._torch is not None and self._device.startswith("cuda"):
            try:
                self._torch.cuda.empty_cache()
            except Exception:
                pass
