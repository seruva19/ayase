"""Song aesthetic evaluation using SongEval (ASLP-lab, 2025).

Predicts five perceptual aesthetic dimensions for generated songs:
  Coherence    — overall coherence of the song (1-5, higher=better)
  Musicality   — overall musicality (1-5, higher=better)
  Memorability — how memorable the song is (1-5, higher=better)
  Clarity      — clarity of song structure (1-5, higher=better)
  Naturalness  — naturalness of vocal breathing and phrasing (1-5, higher=better)

Based on the SongEval benchmark dataset and MuQ (Music Quality) audio encoder.
Requires ``muq`` pip package and ``torch``.

References:
  - Paper: https://arxiv.org/pdf/2505.10793
  - Code: https://github.com/ASLP-lab/SongEval
  - MuQ: https://huggingface.co/OpenMuQ/MuQ-large-msd-iter
"""

import logging
import os
from pathlib import Path
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class _SongEvalGenerator:
    """SongEval Generator model (from ASLP-lab/SongEval).

    Multi-head attention + FFN architecture that predicts 5 aesthetic
    dimensions from MuQ audio features. Output range: [1, 5].
    """

    def __init__(self, in_features: int = 1024, ffd_hidden_size: int = 4096,
                 num_classes: int = 5, attn_layer_num: int = 4):
        import torch
        import torch.nn as nn

        self.attn = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=in_features,
                num_heads=8,
                dropout=0.2,
                batch_first=True,
            )
            for _ in range(attn_layer_num)
        ])
        self.ffd = nn.Sequential(
            nn.Linear(in_features, ffd_hidden_size),
            nn.ReLU(),
            nn.Linear(ffd_hidden_size, in_features),
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(in_features * 2, num_classes)
        self.proj = nn.Tanh()

    def forward(self, ssl_feature):
        import torch

        B, T, D = ssl_feature.shape
        ssl_feature = self.ffd(ssl_feature)
        tmp_ssl_feature = ssl_feature
        for attn in self.attn:
            tmp_ssl_feature, _ = attn(tmp_ssl_feature, tmp_ssl_feature, tmp_ssl_feature)
        ssl_feature = self.dropout(torch.concat([
            torch.mean(tmp_ssl_feature, dim=1),
            torch.max(ssl_feature, dim=1)[0],
        ], dim=1))
        x = self.fc(ssl_feature)
        x = self.proj(x) * 2.0 + 3
        return x


class SongEvalModule(PipelineModule):
    """Song aesthetic evaluation across 5 perceptual dimensions."""

    name = "song_eval"
    description = "SongEval song aesthetic evaluation — Coherence, Musicality, Memorability, Clarity, Naturalness (1-5)"
    default_config = {
        "sample_rate": 24000,
        "checkpoint_subpath": "song_eval/model.safetensors",
    }

    models = [
        {
            "id": "OpenMuQ/MuQ-large-msd-iter",
            "type": "huggingface",
            "task": "MuQ audio feature encoder for SongEval",
        },
        {
            "id": "song_eval/model.safetensors",
            "type": "local",
            "task": "SongEval Generator aesthetic head weights",
        },
    ]

    metric_info = {
        "song_eval_coherence": "Overall coherence (1-5, higher=better)",
        "song_eval_musicality": "Overall musicality (1-5, higher=better)",
        "song_eval_memorability": "Memorability (1-5, higher=better)",
        "song_eval_clarity": "Clarity of song structure (1-5, higher=better)",
        "song_eval_naturalness": "Naturalness of vocal breathing/phrasing (1-5, higher=better)",
    }

    _CHECKPOINT_URL = "https://github.com/ASLP-lab/SongEval/raw/main/ckpt/model.safetensors"

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None
        self._muq = None
        self._ml_available = False
        self._backend = None
        self.sample_rate = self.config.get("sample_rate", 24000)

    def setup(self) -> None:
        try:
            import torch
            from safetensors.torch import load_file
        except ImportError:
            logger.warning("SongEval requires torch and safetensors. Skipping.")
            return

        try:
            from muq import MuQ
        except ImportError:
            logger.warning("SongEval requires the 'muq' package (pip install muq). Skipping.")
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            checkpoint_path = self._resolve_checkpoint()
            if checkpoint_path is None:
                logger.warning("SongEval checkpoint not found and could not be downloaded.")
                return

            state_dict = load_file(str(checkpoint_path), device="cpu")
            model = _SongEvalGenerator(
                in_features=1024,
                ffd_hidden_size=4096,
                num_classes=5,
                attn_layer_num=4,
            )
            model.load_state_dict(state_dict, strict=False)
            model = model.to(device).eval()

            muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
            muq = muq.to(device).eval()

            self._model = model
            self._muq = muq
            self._device = device
            self._ml_available = True
            self._backend = "songeval"
            logger.info("SongEval initialised (muq + generator)")
        except Exception as e:
            logger.warning(f"SongEval setup failed: {e}")

    def _resolve_checkpoint(self) -> Optional[Path]:
        """Find or download the SongEval generator checkpoint."""
        models_dir = self.config.get("models_dir", "models")
        subpath = self.config.get("checkpoint_subpath", "song_eval/model.safetensors")

        direct = Path(models_dir) / subpath
        if direct.exists():
            return direct

        from ayase.config import resolve_model_path
        resolved = resolve_model_path("song_eval/model.safetensors", models_dir)
        if resolved and Path(resolved).exists():
            return Path(resolved)

        try:
            from ayase.config import download_model_file
            return download_model_file(subpath, self._CHECKPOINT_URL, models_dir)
        except Exception as e:
            logger.debug(f"SongEval checkpoint download failed: {e}")
            return None

    def _extract_audio(self, sample_path: Path) -> Optional[tuple]:
        """Load audio waveform from file using librosa."""
        try:
            import librosa
            import numpy as np
            wav, sr = librosa.load(str(sample_path), sr=self.sample_rate, mono=True)
            return wav.astype(np.float32), sr
        except Exception as e:
            logger.debug(f"Failed to load audio from {sample_path}: {e}")
            return None

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            import torch

            result = self._extract_audio(sample.path)
            if result is None:
                return sample

            wav, _ = result
            audio = torch.tensor(wav).unsqueeze(0).to(self._device)

            with torch.no_grad():
                output = self._muq(audio, output_hidden_states=True)
                features = output["hidden_states"][6]
                scores = self._model(features).squeeze(0)

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.song_eval_coherence = round(scores[0].item(), 4)
            sample.quality_metrics.song_eval_musicality = round(scores[1].item(), 4)
            sample.quality_metrics.song_eval_memorability = round(scores[2].item(), 4)
            sample.quality_metrics.song_eval_clarity = round(scores[3].item(), 4)
            sample.quality_metrics.song_eval_naturalness = round(scores[4].item(), 4)

            logger.debug(
                f"SongEval for {sample.path.name}: "
                f"coh={scores[0]:.2f} mus={scores[1]:.2f} "
                f"mem={scores[2]:.2f} cla={scores[3]:.2f} nat={scores[4]:.2f}"
            )
        except Exception as e:
            logger.warning(f"SongEval failed for {sample.path.name}: {e}")

        return sample
