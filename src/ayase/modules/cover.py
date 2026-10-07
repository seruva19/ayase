"""COVER three-branch video/image quality assessment (He et al., CVPRW 2024).

Runs the official COVER evaluator: a semantic branch (CLIP ViT-L/14 on the
512x512 resized view), a technical branch (Swin-T on 7x7 spatial fragments of
32x32), and an aesthetic branch (ConvNeXt-T on the 224x224 resized view), with
semantic cross-gating of the other two branches. Temporal sampling follows the
published val-ytugc protocol (UnifiedFrameSampler, frame interval 2, one clip
per view) and scores are averaged over clips; ``cover_score`` is the sum of the
three branch scores, as in the official ``fuse_results``. Higher is better.
Scores are produced only by the official model and weights; no proxy is used
when unavailable. Source: https://github.com/taco-group/COVER
"""

import logging
import os
from typing import Optional

from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class COVERModule(PipelineModule):
    name = "cover"
    provenance = "published"
    sources = {
        "cover_aesthetic": "COVER (He et al., CVPRW 2024) — https://github.com/taco-group/COVER",
        "cover_score": "COVER (He et al., CVPRW 2024) — https://github.com/vztu/COVER",
        "cover_semantic": "COVER (He et al., CVPRW 2024) — https://github.com/taco-group/COVER",
        "cover_technical": "COVER (He et al., CVPRW 2024) — https://github.com/taco-group/COVER",
    }
    description = "COVER 3-branch comprehensive video quality (semantic + aesthetic + technical)"
    default_config = {
        "quality_threshold": 30.0,
    }
    metric_groups = {
        "cover_aesthetic": "aesthetic",
        "cover_score": "nr_quality",
        "cover_semantic": "aesthetic",
        "cover_technical": "nr_quality",
    }
    models = [
        {
            "id": "https://github.com/taco-group/COVER/raw/release/Model/COVER.pth",
            "type": "other",
            "task": "COVER evaluator weights",
        },
    ]

    # Official val-ytugc sample_types (semantic first — it gates the others).
    _SAMPLE_TYPES = {
        "semantic": {
            "size_h": 512, "size_w": 512,
            "clip_len": 20, "frame_interval": 2, "t_frag": 20, "num_clips": 1,
        },
        "technical": {
            "fragments_h": 7, "fragments_w": 7,
            "fsize_h": 32, "fsize_w": 32,
            "aligned": 40, "clip_len": 40,
            "t_frag": 20, "frame_interval": 2, "num_clips": 1,
        },
        "aesthetic": {
            "size_h": 224, "size_w": 224,
            "clip_len": 40, "t_frag": 20, "frame_interval": 2, "num_clips": 1,
        },
    }
    _MODEL_ARGS = {
        "backbone": {
            "technical": {"type": "swin_tiny_grpb", "checkpoint": True, "pretrained": None},
            "aesthetic": {"type": "conv_tiny"},
            "semantic": {"type": "clip_iqa+"},
        },
        "backbone_preserve_keys": "technical,aesthetic,semantic",
        "divide_head": True,
        "vqa_head": {"in_channels": 768, "hidden_channels": 64},
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._model = None
        self._device = "cpu"
        self._decompose = None
        self._sampler_cls = None
        self._mean = None
        self._std = None
        self._mean_clip = None
        self._std_clip = None

    def setup(self) -> None:
        try:
            import torch
            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            self._ensure_convnext_cached()

            from ayase.third_party.cover.datasets import (
                UnifiedFrameSampler,
                spatial_temporal_view_decomposition,
            )
            from ayase.third_party.cover.models import COVER

            self._decompose = spatial_temporal_view_decomposition
            self._sampler_cls = UnifiedFrameSampler

            weights = self._resolve_weights()
            if weights is None:
                logger.warning("COVER weights not found; cover_* metrics skipped.")
                return

            model = COVER(**self._MODEL_ARGS)
            try:
                state = torch.load(weights, map_location="cpu", weights_only=True)
            except Exception:
                state = torch.load(weights, map_location="cpu", weights_only=False)
            # strict=False: the prompt_learner ctx lives outside the state dict
            # (CLIP weights are not registered as parameters in upstream).
            model.load_state_dict(
                state.get("state_dict", state) if isinstance(state, dict) else state,
                strict=False,
            )
            self._model = model.to(self._device)
            self._model.eval()

            self._mean = torch.FloatTensor([123.675, 116.28, 103.53])
            self._std = torch.FloatTensor([58.395, 57.12, 57.375])
            self._mean_clip = torch.FloatTensor([122.77, 116.75, 104.09])
            self._std_clip = torch.FloatTensor([68.50, 66.63, 70.32])

            self._ml_available = True
            logger.info("COVER model loaded on %s", self._device)
        except ImportError as e:
            logger.warning("COVER unavailable: %s", e)
        except Exception as e:
            logger.warning("COVER setup failed: %s", e)

    _COVER_WEIGHTS_URL = (
        "https://github.com/taco-group/COVER/raw/release/Model/COVER.pth"
    )

    # Original: https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth
    _CONVNEXT_URL = "https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/dover/convnext_tiny_1k_224_ema.pth"
    _CONVNEXT_FILENAME = "convnext_tiny_1k_224_ema.pth"

    def _ensure_convnext_cached(self) -> None:
        """Download ConvNeXt weights to torch hub cache if not present."""
        import torch

        hub_dir = torch.hub.get_dir()
        cache_dir = os.path.join(hub_dir, "checkpoints")
        cached = os.path.join(cache_dir, self._CONVNEXT_FILENAME)
        if os.path.exists(cached):
            return
        os.makedirs(cache_dir, exist_ok=True)
        logger.info("Downloading ConvNeXt backbone for COVER...")
        from ayase.config import download_model_file

        tmp = download_model_file(
            os.path.join("hub", "checkpoints", self._CONVNEXT_FILENAME),
            self._CONVNEXT_URL,
            os.path.dirname(hub_dir),
        )
        if str(tmp) != cached and os.path.exists(str(tmp)):
            import shutil

            shutil.copy2(str(tmp), cached)

    def _resolve_weights(self) -> Optional[str]:
        """Find COVER.pth weights file, auto-downloading if needed."""
        weights_path = self.config.get("weights_path")
        if weights_path and os.path.exists(weights_path):
            return weights_path

        models_dir = self.config.get("models_dir", "models")
        candidates = [
            os.path.join(models_dir, "cover", "COVER.pth"),
            os.path.join(models_dir, "COVER", "pretrained_weights", "COVER.pth"),
            os.path.join(models_dir, "COVER.pth"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        try:
            from ayase.config import download_model_file

            return str(
                download_model_file(
                    os.path.join("cover", "COVER.pth"),
                    self._COVER_WEIGHTS_URL,
                    models_dir,
                )
            )
        except Exception as e:
            logger.warning("COVER weights download failed: %s", e)
            return None

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available:
            return sample

        try:
            self._process_cover(sample)

            threshold = self.config.get("quality_threshold", 30.0)
            if (
                sample.quality_metrics.cover_score is not None
                and sample.quality_metrics.cover_score < threshold
            ):
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low COVER quality score: {sample.quality_metrics.cover_score:.1f}",
                        recommendation="Review video for quality issues",
                    )
                )
        except Exception as e:
            logger.warning("COVER processing failed: %s", e)
        return sample

    def _process_cover(self, sample: Sample) -> None:
        """Run the official COVER evaluation pipeline on the sample."""
        import torch

        temporal_samplers = {
            stype: self._sampler_cls(
                sopt["clip_len"] // sopt["t_frag"],
                sopt["t_frag"],
                sopt["frame_interval"],
                sopt["num_clips"],
            )
            for stype, sopt in self._SAMPLE_TYPES.items()
        }

        views, _ = self._decompose(
            str(sample.path), self._SAMPLE_TYPES, temporal_samplers
        )

        processed = {}
        for k, v in views.items():
            num_clips = self._SAMPLE_TYPES[k].get("num_clips", 1)
            mean, std = (
                (self._mean_clip, self._std_clip)
                if k == "semantic"
                else (self._mean, self._std)
            )
            processed[k] = (
                ((v.permute(1, 2, 3, 0) - mean) / std)
                .permute(3, 0, 1, 2)
                .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
                .transpose(0, 1)
                .to(self._device)
            )

        with torch.no_grad():
            results = [r.mean().item() for r in self._model(processed)]

        # Official fuse_results: [semantic, technical, aesthetic] view order.
        semantic, technical, aesthetic = results
        sample.quality_metrics.cover_semantic = float(semantic)
        sample.quality_metrics.cover_technical = float(technical)
        sample.quality_metrics.cover_aesthetic = float(aesthetic)
        sample.quality_metrics.cover_score = float(semantic + technical + aesthetic)
