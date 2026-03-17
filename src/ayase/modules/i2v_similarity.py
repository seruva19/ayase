"""Image-to-Video Similarity — reference image vs generated video quality.

Measures how closely a generated video matches its source reference image using
three complementary perceptual metrics with a sliding-window approach:

- **CLIP** (Radford et al., 2021): cosine similarity of CLIP embeddings
- **DINOv2** (Oquab et al., 2023): cosine similarity of self-supervised embeddings
- **LPIPS** (Zhang et al., 2018): learned perceptual distance (lower = more similar)

The sliding-window technique (used in VBench, EvalCrafter) splits video frames
into overlapping temporal windows, computes per-window scores against the
reference image, and takes the median — providing a robust aggregate that
handles temporal variation.

Aggregated I2V quality score:
    i2v_quality = clip * 0.4 + (1 - lpips) * 100 * 0.2 + dino * 0.4

Activates only when ``sample.is_video`` and ``sample.reference_path`` points to
an image file.

References:
    - VBench (Huang et al., 2023) — comprehensive video generation benchmark
    - EvalCrafter (Liu et al., 2023) — T2V evaluation framework
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


class I2VSimilarityModule(PipelineModule):
    name = "i2v_similarity"
    description = "Image-to-Video reference similarity using CLIP, DINOv2, and LPIPS (sliding window)"
    default_config = {
        "window_size": 16,
        "stride": 8,
        "max_frames": 256,
        "clip_model": "ViT-B-32",
        "clip_pretrained": "openai",
        "dino_model": "dinov2_vitb14",
        "enable_clip": True,
        "enable_dino": True,
        "enable_lpips": True,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.window_size = self.config.get("window_size", 16)
        self.stride = self.config.get("stride", 8)
        self.max_frames = self.config.get("max_frames", 256)
        self.clip_model_name = self.config.get("clip_model", "ViT-B-32")
        self.clip_pretrained = self.config.get("clip_pretrained", "openai")
        self.dino_model_name = self.config.get("dino_model", "dinov2_vitb14")
        self.enable_clip = self.config.get("enable_clip", True)
        self.enable_dino = self.config.get("enable_dino", True)
        self.enable_lpips = self.config.get("enable_lpips", True)

        self._device = "cpu"
        self._clip_model = None
        self._clip_preprocess = None
        self._dino_model = None
        self._dino_preprocess = None
        self._lpips_model = None
        self._clip_available = False
        self._dino_available = False
        self._lpips_available = False

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    def on_mount(self) -> None:
        super().on_mount()
        try:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            logger.warning("PyTorch not installed. I2V Similarity disabled.")
            return

        if self.enable_clip:
            self._init_clip()
        if self.enable_dino:
            self._init_dino()
        if self.enable_lpips:
            self._init_lpips()

        if not (self._clip_available or self._dino_available or self._lpips_available):
            logger.warning("No I2V sub-metrics available. Module effectively disabled.")

    _CLIP_URLS = {
        "ViT-B-32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    }

    def _init_clip(self) -> None:
        try:
            import open_clip
            from ayase.config import download_model_file

            models_dir = self.config.get("models_dir", "models")
            local_clip = Path(str(models_dir)) / "open_clip" / f"{self.clip_model_name}.pt"

            # Auto-download if missing
            if not local_clip.exists() and self.clip_model_name in self._CLIP_URLS:
                download_model_file(
                    f"open_clip/{self.clip_model_name}.pt",
                    self._CLIP_URLS[self.clip_model_name],
                    models_dir,
                )

            # Patch open_clip to handle TorchScript archives with torch >=2.6
            # (which defaults weights_only=True, incompatible with TorchScript)
            _orig_load_sd = open_clip.factory.load_state_dict
            def _patched_load_sd(checkpoint_path, device='cpu', weights_only=True):
                return _orig_load_sd(checkpoint_path, device=device, weights_only=False)

            logger.info(f"Loading CLIP ({self.clip_model_name}) for I2V on {self._device}...")
            open_clip.factory.load_state_dict = _patched_load_sd
            try:
                if local_clip.exists():
                    logger.info(f"Using local CLIP weights: {local_clip}")
                    model, _, preprocess = open_clip.create_model_and_transforms(
                        self.clip_model_name, pretrained=str(local_clip), device=self._device
                    )
                else:
                    model, _, preprocess = open_clip.create_model_and_transforms(
                        self.clip_model_name, self.clip_pretrained, device=self._device
                    )
            finally:
                open_clip.factory.load_state_dict = _orig_load_sd
            model.eval()
            self._clip_model = model
            self._clip_preprocess = preprocess
            self._clip_available = True
        except ImportError:
            logger.warning("open_clip not installed. I2V CLIP disabled.")
        except Exception as e:
            logger.error(f"Failed to load CLIP for I2V: {e}")

    _DINO_URLS = {
        "dinov2_vitb14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
    }

    def _init_dino(self) -> None:
        try:
            import io
            import os
            import contextlib
            import torch
            from ayase.config import download_model_file

            models_dir = self.config.get("models_dir", "models")
            local_weights = Path(str(models_dir)) / "dinov2" / f"{self.dino_model_name}_pretrain.pth"

            # Auto-download if missing
            if not local_weights.exists() and self.dino_model_name in self._DINO_URLS:
                download_model_file(
                    f"dinov2/{self.dino_model_name}_pretrain.pth",
                    self._DINO_URLS[self.dino_model_name],
                    models_dir,
                )

            # Redirect torch hub cache to models_dir
            if models_dir:
                os.environ["TORCH_HOME"] = str(models_dir)

            logger.info(f"Loading DINOv2 ({self.dino_model_name}) for I2V on {self._device}...")

            # Use local torch hub cache if available to avoid network requests
            hub_cache = Path(os.environ.get("TORCH_HOME", "")) / "hub" / "facebookresearch_dinov2_main"
            hub_source = "local" if hub_cache.is_dir() else "github"

            if local_weights.exists():
                logger.info(f"Using local DINOv2 weights: {local_weights}")
                model = torch.hub.load("facebookresearch/dinov2", self.dino_model_name,
                                       source=hub_source, pretrained=False)
                state_dict = torch.load(str(local_weights), map_location=self._device, weights_only=True)
                model.load_state_dict(state_dict)
            else:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    model = torch.hub.load("facebookresearch/dinov2", self.dino_model_name)
            model.eval().to(self._device)
            self._dino_model = model

            from torchvision import transforms as T

            self._dino_preprocess = T.Compose([
                T.Resize((518, 518), interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(518),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            self._dino_available = True
        except ImportError:
            logger.warning("torch/torchvision not installed properly. I2V DINO disabled.")
        except Exception as e:
            logger.error(f"Failed to load DINOv2 for I2V: {e}")

    _LPIPS_URL = "https://github.com/richzhang/PerceptualSimilarity/raw/master/lpips/weights/v0.1/alex.pth"

    def _init_lpips(self) -> None:
        try:
            import io
            import contextlib
            import warnings
            import lpips as lpips_lib
            from ayase.config import download_model_file

            models_dir = self.config.get("models_dir", "models")
            local_lpips = Path(str(models_dir)) / "lpips" / "alex.pth"

            # Auto-download if missing
            if not local_lpips.exists():
                download_model_file("lpips/alex.pth", self._LPIPS_URL, models_dir)

            logger.info(f"Loading LPIPS (alex) for I2V on {self._device}...")
            warnings.filterwarnings(
                "ignore",
                message="Please be aware that recompiling all source files may be required.*",
            )
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                if local_lpips.exists():
                    logger.info(f"Using local LPIPS weights: {local_lpips}")
                    model = lpips_lib.LPIPS(net="alex", model_path=str(local_lpips)).eval().to(self._device)
                else:
                    model = lpips_lib.LPIPS(net="alex").eval().to(self._device)
            self._lpips_model = model
            self._lpips_available = True
        except ImportError:
            logger.warning("lpips not installed. I2V LPIPS disabled.")
        except Exception as e:
            logger.error(f"Failed to load LPIPS for I2V: {e}")

    # ------------------------------------------------------------------ #
    #  Processing                                                         #
    # ------------------------------------------------------------------ #

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
            return sample
        if not sample.reference_path:
            return sample
        if sample.reference_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            return sample
        if not sample.reference_path.exists():
            logger.debug(f"Reference path does not exist: {sample.reference_path}")
            return sample

        try:
            frames = self._load_video_frames(sample)
            if len(frames) < self.window_size:
                logger.debug(
                    f"Not enough frames ({len(frames)}) for window_size={self.window_size}. Skipping I2V."
                )
                return sample

            ref_path = str(sample.reference_path)

            clip_score = self._compute_clip(ref_path, frames) if self._clip_available else None
            dino_score = self._compute_dino(ref_path, frames) if self._dino_available else None
            lpips_score = self._compute_lpips(ref_path, frames) if self._lpips_available else None

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            if clip_score is not None:
                sample.quality_metrics.i2v_clip = round(clip_score, 4)
            if dino_score is not None:
                sample.quality_metrics.i2v_dino = round(max(0.0, dino_score), 4)
            if lpips_score is not None:
                sample.quality_metrics.i2v_lpips = round(lpips_score, 4)

            # Aggregated quality
            i2v_q = self._aggregate(clip_score, dino_score, lpips_score)
            if i2v_q is not None:
                sample.quality_metrics.i2v_quality = round(i2v_q, 2)

                if i2v_q < 30.0:
                    sample.validation_issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            message=f"Low I2V quality: {i2v_q:.1f}/100",
                            details={
                                "i2v_clip": clip_score,
                                "i2v_dino": dino_score,
                                "i2v_lpips": lpips_score,
                                "i2v_quality": i2v_q,
                            },
                            recommendation="Generated video diverges significantly from the reference image.",
                        )
                    )

        except Exception as e:
            logger.warning(f"I2V Similarity failed for {sample.path}: {e}")

        return sample

    # ------------------------------------------------------------------ #
    #  CLIP sub-metric                                                    #
    # ------------------------------------------------------------------ #

    def _compute_clip(self, image_path: str, frames: List[np.ndarray]) -> float:
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        image_tensor = self._clip_preprocess(image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            image_feat = self._clip_model.encode_image(image_tensor)
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]

        similarities = []
        with torch.no_grad():
            for start in range(0, len(pil_frames) - self.window_size + 1, self.stride):
                window = pil_frames[start : start + self.window_size]
                tensors = torch.stack([self._clip_preprocess(f) for f in window]).to(self._device)
                feats = self._clip_model.encode_image(tensors)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                avg_feat = feats.mean(dim=0, keepdim=True)
                sim = torch.nn.functional.cosine_similarity(image_feat, avg_feat).item()
                similarities.append(sim)

        return float(np.median(similarities)) if similarities else 0.0

    # ------------------------------------------------------------------ #
    #  DINO sub-metric                                                    #
    # ------------------------------------------------------------------ #

    def _compute_dino(self, image_path: str, frames: List[np.ndarray]) -> float:
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        img_tensor = self._dino_preprocess(image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            image_feat = self._dino_model(img_tensor)
            if isinstance(image_feat, dict) and "x" in image_feat:
                image_feat = image_feat["x"]
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]

        similarities = []
        with torch.no_grad():
            for start in range(0, len(pil_frames) - self.window_size + 1, self.stride):
                window = pil_frames[start : start + self.window_size]
                batch = torch.stack([self._dino_preprocess(f) for f in window]).to(self._device)
                feats = self._dino_model(batch)
                if isinstance(feats, dict) and "x" in feats:
                    feats = feats["x"]
                feats = feats / feats.norm(dim=-1, keepdim=True)
                avg_feat = feats.mean(dim=0, keepdim=True)
                sim = torch.nn.functional.cosine_similarity(
                    image_feat, avg_feat
                ).item()
                similarities.append(sim)

        return float(np.median(similarities)) if similarities else 0.0

    # ------------------------------------------------------------------ #
    #  LPIPS sub-metric                                                   #
    # ------------------------------------------------------------------ #

    def _compute_lpips(self, image_path: str, frames: List[np.ndarray]) -> float:
        import torch
        from PIL import Image
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1),  # [0,1] -> [-1,1]
        ])

        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(self._device)

        pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        frame_tensors = [transform(f).unsqueeze(0).to(self._device) for f in pil_frames]

        distances = []
        with torch.no_grad():
            for start in range(0, len(frame_tensors) - self.window_size + 1, self.stride):
                window = frame_tensors[start : start + self.window_size]
                avg_tensor = torch.stack([t.squeeze(0) for t in window]).mean(dim=0, keepdim=True)
                dist = self._lpips_model(image_tensor, avg_tensor)
                distances.append(dist.item())

        return float(np.median(distances)) if distances else 0.0

    # ------------------------------------------------------------------ #
    #  Aggregation                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _aggregate(
        clip_score: Optional[float],
        dino_score: Optional[float],
        lpips_score: Optional[float],
    ) -> Optional[float]:
        """Weighted aggregation into a 0-100 quality score.

        Formula: clip_pct * 0.4 + (1 - lpips) * 100 * 0.2 + dino_pct * 0.4
        Falls back to equal-weight averaging when some sub-metrics are missing.
        """
        parts = []
        weights = []

        if clip_score is not None:
            parts.append(clip_score * 100)
            weights.append(0.4)
        if lpips_score is not None:
            parts.append((1.0 - lpips_score) * 100)
            weights.append(0.2)
        if dino_score is not None:
            parts.append(dino_score * 100)
            weights.append(0.4)

        if not parts:
            return None

        # Renormalize weights to sum to 1
        w_sum = sum(weights)
        return sum(p * w / w_sum for p, w in zip(parts, weights))

    # ------------------------------------------------------------------ #
    #  Frame loading                                                      #
    # ------------------------------------------------------------------ #

    def _load_video_frames(self, sample: Sample) -> List[np.ndarray]:
        frames: List[np.ndarray] = []
        try:
            cap = cv2.VideoCapture(str(sample.path))
            count = 0
            while cap.isOpened() and count < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                count += 1
            cap.release()
        except Exception as e:
            logger.debug(f"Video frame loading failed: {e}")
        return frames
