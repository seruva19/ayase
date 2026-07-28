"""Per-sample generated-music quality using the upstream MuQ-Eval A1 metric.

Predicts Musical Impression (MI) on the MusicEval 1-5 expert-MOS scale.
The published frozen-MuQ A1 model reaches utterance-level SRCC 0.838 and
system-level SRCC 0.957. Only MI is exposed: the audio-only checkpoint's
text-alignment head is not a reliable prompt-alignment metric.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ayase.audio import load_audio
from ayase.config import download_hf_snapshot
from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

_CHECKPOINT_REPO = "zhudi2825/MuQ-Eval-A1"
_ENCODER_REPO = "OpenMuQ/MuQ-large-msd-iter"
_IMAGE_SUFFIXES = {
    ".avif",
    ".bmp",
    ".gif",
    ".heic",
    ".heif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


class MuQEvalModule(PipelineModule):
    """frozen-MuQ A1 predictor for generated-music quality."""

    name = "muq_eval"
    description = "MuQ-Eval A1 per-sample generated-music Musical Impression MOS"
    default_config = {
        "sample_rate": 24000,
        "clip_duration": 10.0,
        "warning_threshold": 3.0,
        "device": "auto",
    }
    models = [
        {
            "id": _CHECKPOINT_REPO,
            "type": "huggingface",
            "task": "MuQ-Eval A1 attention-pooling and MOS prediction heads",
            "size": "1.34 GB",
            "auto_download": True,
            "notes": "MIT model repository; upstream recommended A1 checkpoint",
        },
        {
            "id": _ENCODER_REPO,
            "type": "huggingface",
            "task": "MuQ-310M architecture configuration (weights included in A1)",
            "auto_download": True,
            "notes": (
                "Only config.json is downloaded; A1 includes encoder parameters. "
                "Original encoder weights are CC-BY-NC-4.0, so commercial use "
                "still requires separate review"
            ),
        },
        {
            "id": "muq",
            "type": "pip_package",
            "install": "pip install muq",
            "task": "MuQ model implementation",
        },
    ]
    metric_info = {
        "muq_eval_mi_score": (
            "MuQ-Eval A1 Musical Impression prediction on the 1-5 "
            "MusicEval expert-MOS scale (higher=better)"
        ),
    }
    metric_groups = {"muq_eval_mi_score": "audio"}

    def __init__(self, config=None):
        super().__init__(config)
        self.sample_rate = int(self.config.get("sample_rate", 24000))
        self.clip_duration = float(self.config.get("clip_duration", 10.0))
        self.warning_threshold = float(self.config.get("warning_threshold", 3.0))
        self._model = None
        self._device = "cpu"
        self._backend = "unavailable"
        self._ml_available = False

    def setup(self) -> None:
        if self.sample_rate != 24000 or self.clip_duration != 10.0:
            logger.warning(
                "MuQ-Eval disabled: the published A1 metric requires 24 kHz, 10-second inputs"
            )
            return

        try:
            import torch
            import torch.nn as nn
            from muq import MuQ, MuQConfig

            from ayase.runtime import resolve_torch_device

            models_dir = self.config.get("models_dir", "models")
            checkpoint_root = download_hf_snapshot(
                _CHECKPOINT_REPO,
                models_dir,
                allow_patterns=["config.yaml", "model_state_dict.pt"],
            )
            encoder_root = download_hf_snapshot(
                _ENCODER_REPO,
                models_dir,
                allow_patterns=["config.json"],
            )
            state_path = checkpoint_root / "model_state_dict.pt"
            if not state_path.exists():
                raise FileNotFoundError(f"MuQ-Eval state dict not found: {state_path}")

            class _AttentionPooling(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.attention = nn.Sequential(
                        nn.Linear(1024, 128),
                        nn.Tanh(),
                        nn.Linear(128, 1),
                    )

                def forward(self, hidden):
                    weights = torch.softmax(self.attention(hidden).squeeze(-1), dim=-1)
                    return torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)

            class _Encoder(nn.Module):
                def __init__(self, root):
                    super().__init__()
                    with (Path(root) / "config.json").open(encoding="utf-8") as stream:
                        encoder_config = MuQConfig(**json.load(stream))
                    # The upstream A1 state dict already contains every frozen
                    # encoder parameter. Build only the architecture here instead
                    # of downloading a duplicate 1.33 GB base checkpoint.
                    self.encoder = MuQ(encoder_config)
                    self.pooling = _AttentionPooling()

                def forward(self, waveforms):
                    output = self.encoder(waveforms, output_hidden_states=True)
                    return self.pooling(output.last_hidden_state)

            class _PredictionHead(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.mlp = nn.Sequential(
                        nn.Linear(1024, 256),
                        nn.GELU(),
                        nn.Dropout(0.1),
                        nn.Linear(256, 1),
                    )

                def forward(self, features):
                    return self.mlp(features).squeeze(-1)

            class _A1Predictor(nn.Module):
                def __init__(self, root):
                    super().__init__()
                    self.encoder = _Encoder(root)
                    self.heads = nn.ModuleDict(
                        {"MI": _PredictionHead(), "TA": _PredictionHead()}
                    )

                def forward(self, waveforms):
                    features = self.encoder(waveforms)
                    return {name: head(features) for name, head in self.heads.items()}

            model = _A1Predictor(encoder_root)
            try:
                state = torch.load(state_path, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(state_path, map_location="cpu")
            model.load_state_dict(state, strict=True)

            device = resolve_torch_device(self.config.get("device", "auto"))
            model = model.to(device).eval()
            self._model = model
            self._device = device
            self._backend = "a1"
            self._ml_available = True
            logger.info("MuQ-Eval A1 initialised on %s", device)
        except ImportError as exc:
            logger.warning("MuQ-Eval requires torch and the `muq` package: %s", exc)
        except Exception as exc:
            logger.warning("MuQ-Eval setup failed: %s", exc)

    def _prepare_waveform(self, path: Path):
        audio = load_audio(
            path,
            target_sr=self.sample_rate,
            mono=True,
        )
        if audio is None or audio.size == 0:
            return None

        clip_samples = int(self.sample_rate * self.clip_duration)
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size >= clip_samples:
            start = (audio.size - clip_samples) // 2
            audio = audio[start : start + clip_samples]
        else:
            audio = np.pad(audio, (0, clip_samples - audio.size))
        return audio

    def _score(self, waveform) -> Optional[float]:
        try:
            import torch

            tensor = torch.from_numpy(waveform).float().unsqueeze(0).to(self._device)
            with torch.inference_mode():
                output = self._model(tensor)
            score = float(output["MI"].reshape(-1)[0].detach().cpu().item())
            return score if np.isfinite(score) else None
        except Exception as exc:
            logger.debug("MuQ-Eval inference failed: %s", exc)
            return None

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or self._backend != "a1":
            return sample
        if sample.path.suffix.lower() in _IMAGE_SUFFIXES:
            return sample

        try:
            waveform = self._prepare_waveform(sample.path)
            if waveform is None:
                return sample
            score = self._score(waveform)
            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.muq_eval_mi_score = round(score, 4)

            if score < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low MuQ-Eval musical impression: {score:.2f}",
                        details={"muq_eval_mi_score": score},
                    )
                )
        except Exception as exc:
            logger.warning("MuQ-Eval failed for %s: %s", sample.path, exc)
        return sample
