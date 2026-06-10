"""PAM anti-prompt audio quality score.

Scores no-reference audio quality by contrasting CLAP similarity to positive
quality prompts against anti-prompts such as noisy, distorted, or clipped audio.
Falls back to a signal-quality proxy when CLAP is unavailable.
"""

import logging
from typing import Optional

import numpy as np

from ayase.audio import load_audio, speech_quality_proxy
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class PAMModule(PipelineModule):
    name = "pam"
    description = "PAM anti-prompt no-reference perceptual audio quality"
    default_config = {
        "model_name": "laion/clap-htsat-fused",
        "sample_rate": 48000,
        "device": "auto",
        "positive_prompts": [
            "clear high quality natural audio",
            "clean intelligible speech or music",
        ],
        "negative_prompts": [
            "noisy distorted clipped low quality audio",
            "muffled corrupted unpleasant sound",
        ],
    }
    models = [
        {
            "id": "laion/clap-htsat-fused",
            "type": "huggingface",
            "task": "CLAP encoder for PAM anti-prompt scoring",
        },
    ]
    metric_info = {
        "pam_score": "PAM anti-prompt perceptual audio quality (0-1, higher=better)",
    }
    metric_groups = {
        "pam_score": "audio",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "laion/clap-htsat-fused")
        self.sample_rate = self.config.get("sample_rate", 48000)
        self.device_config = self.config.get("device", "auto")
        self.positive_prompts = self.config.get("positive_prompts", self.default_config["positive_prompts"])
        self.negative_prompts = self.config.get("negative_prompts", self.default_config["negative_prompts"])
        self._backend = "signal_proxy"
        self._model = None
        self._processor = None
        self._device = "cpu"

    def setup(self) -> None:
        try:
            import torch
            from transformers import ClapModel, ClapProcessor

            if self.device_config == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = self.device_config
            models_dir = self.config.get("models_dir", "models")
            self._model = ClapModel.from_pretrained(self.model_name, cache_dir=models_dir).to(self._device)
            self._processor = ClapProcessor.from_pretrained(self.model_name, cache_dir=models_dir)
            self._model.eval()
            self._backend = "clap"
            logger.info("PAM initialised with %s on %s", self.model_name, self._device)
        except ImportError:
            logger.info("PAM CLAP backend requires torch and transformers; using signal proxy")
        except Exception as e:
            logger.warning("PAM setup failed (%s); using signal proxy", e)

    def process(self, sample: Sample) -> Sample:
        try:
            audio = load_audio(sample.path, target_sr=self.sample_rate, duration=10.0)
            if audio is None or len(audio) == 0:
                return sample

            score = self._score_clap(audio) if self._backend == "clap" else None
            if score is None:
                score = speech_quality_proxy(audio, sr=self.sample_rate)

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.pam_score = float(score)
        except Exception as e:
            logger.warning("PAM failed for %s: %s", sample.path, e)
        return sample

    def _score_clap(self, audio) -> Optional[float]:
        try:
            import torch

            prompts = list(self.positive_prompts) + list(self.negative_prompts)
            inputs = self._processor(
                text=prompts,
                audios=[audio],
                return_tensors="pt",
                padding=True,
                sampling_rate=self.sample_rate,
            ).to(self._device)
            with torch.no_grad():
                outputs = self._model(**inputs)
                audio_emb = outputs.audio_embeds
                text_emb = outputs.text_embeds
                audio_emb = audio_emb / audio_emb.norm(dim=-1, keepdim=True)
                text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
                sims = (audio_emb @ text_emb.T).detach().cpu().numpy().reshape(-1)
            n_pos = len(self.positive_prompts)
            pos = float(np.mean(sims[:n_pos]))
            neg = float(np.mean(sims[n_pos:]))
            return float(1.0 / (1.0 + np.exp(-(pos - neg) * 8.0)))
        except Exception as e:
            logger.debug("PAM CLAP scoring failed: %s", e)
            return None
