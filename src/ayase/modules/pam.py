"""PAM — no-reference perceptual audio quality via MS-CLAP anti-prompt softmax.

Implements PAM (Deshmukh et al., Interspeech 2024): the audio is embedded with
Microsoft CLAP (``msclap`` package, ``CLAP(version='2023')``), and
``pam_score`` is the softmax probability of the published positive prompt
"the sound is clear and clean" against the anti-prompt "the sound is noisy
and with artifacts". The whole audio file is used.

MS-CLAP is the paper's backend; without the ``msclap`` package the module
emits no score rather than substituting a different CLAP embedding space
(LAION CLAP similarities are not the published quantity).
"""

import logging
from typing import Optional

import numpy as np

from ayase.audio import load_audio
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

POSITIVE_PROMPT = "the sound is clear and clean"
NEGATIVE_PROMPT = "the sound is noisy and with artifacts"


class PAMModule(PipelineModule):
    name = "pam"
    provenance = "published"
    sources = {
        "pam_score": "PAM (Deshmukh et al., Interspeech 2024) — https://arxiv.org/abs/2402.00282",
    }
    description = "PAM anti-prompt no-reference perceptual audio quality (MS-CLAP)"
    default_config = {
        "device": "auto",
        "msclap_version": "2023",
    }
    models = [
        {
            "id": "msclap",
            "type": "pip_package",
            "install": "pip install msclap",
            "task": "Microsoft CLAP encoder — the PAM backend",
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
        self.device_config = self.config.get("device", "auto")
        self.msclap_version = self.config.get("msclap_version", "2023")
        self._backend = "unavailable"
        self._model = None
        self._device = "cpu"

    def setup(self) -> None:
        try:
            from msclap import CLAP
        except ImportError:
            logger.warning(
                "PAM unavailable: msclap package not installed "
                "(pip install msclap); pam_score left unset."
            )
            return

        try:
            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.device_config)
            self._model = CLAP(
                version=self.msclap_version,
                use_cuda=self._device == "cuda",
            )
            self._backend = "msclap"
            logger.info("PAM initialised with MS-CLAP %s on %s", self.msclap_version, self._device)
        except Exception as e:
            logger.warning("PAM unavailable: setup failed (%s)", e)

    def process(self, sample: Sample) -> Sample:
        if self._backend != "msclap":
            return sample
        try:
            audio = load_audio(sample.path, target_sr=48000, duration=None)
            if audio is None or len(audio) == 0:
                return sample

            score = self._score_msclap(sample.path, audio, sample.is_video)
            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.pam_score = float(score)
        except Exception as e:
            logger.warning("PAM failed for %s: %s", sample.path, e)
        return sample

    def _score_msclap(self, path, audio, is_video: bool) -> Optional[float]:
        try:
            import tempfile
            import torch

            # MS-CLAP's embedding API consumes file paths; video containers get
            # the decoded mono waveform written to a temp WAV first.
            if is_video:
                import os
                import soundfile as sf

                fd, tmp_path = tempfile.mkstemp(suffix=".wav")
                try:
                    os.close(fd)
                    sf.write(tmp_path, np.asarray(audio, dtype=np.float32), 48000)
                    audio_emb = self._model.get_audio_embeddings([tmp_path])
                finally:
                    os.unlink(tmp_path)
            else:
                audio_emb = self._model.get_audio_embeddings([str(path)])
            text_emb = self._model.get_text_embeddings([POSITIVE_PROMPT, NEGATIVE_PROMPT])

            # compute_similarity normalizes and applies MS-CLAP's learned
            # logit scale internally.
            sims = self._model.compute_similarity(audio_emb, text_emb).reshape(-1)
            return float(torch.softmax(sims, dim=0)[0].item())
        except Exception as e:
            logger.debug("PAM MS-CLAP scoring failed: %s", e)
            return None
