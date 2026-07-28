"""MAUVE Audio Divergence (MAD) for human-aligned music evaluation.

Compares generated and reference music distributions using max-pooled
layer-24 MERT-v1-330M embeddings and the upstream ``-log(MAUVE)`` definition.
Lower is better. The metric is dataset-level and requires reference audio.
"""

import logging
from typing import List, Optional

import numpy as np

from ayase.audio import load_audio
from ayase.base_modules import BatchMetricModule
from ayase.config import download_hf_snapshot
from ayase.models import Sample

logger = logging.getLogger(__name__)


class MAUVEAudioDivergenceModule(BatchMetricModule):
    """MAD formulation with an Ayase-managed MERT model cache."""

    name = "mauve_audio_divergence"
    description = "MAUVE Audio Divergence with MERT embeddings (ISMIR 2025)"
    default_config = {
        "model_name": "m-a-p/MERT-v1-330M",
        "sample_rate": 24000,
        "layer": 24,
        "aggregation": "max",
        "device": "auto",
        "models_dir": "models",
    }
    models = [
        {
            "id": "mad_metric",
            "type": "pip_package",
            "install": "pip install mad_metric",
            "task": "MAUVE Audio Divergence implementation",
        },
        {
            "id": "m-a-p/MERT-v1-330M",
            "type": "huggingface",
            "task": "Self-supervised music embeddings used by MAD",
            "auto_download": True,
        },
    ]
    metric_info = {
        "mauve_audio_divergence": (
            "MAD: -log(MAUVE) on max-pooled layer-24 MERT-v1-330M embeddings "
            "(dataset-level, lower=better)"
        ),
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.model_name = str(self.config.get("model_name", "m-a-p/MERT-v1-330M"))
        self.sample_rate = int(self.config.get("sample_rate", 24000))
        self.layer = int(self.config.get("layer", 24))
        self.aggregation = str(self.config.get("aggregation", "max")).lower()
        self.device_config = str(self.config.get("device", "auto"))
        self.models_dir = str(self.config.get("models_dir", "models"))
        self._backend = "unavailable"
        self._model = None
        self._processor = None
        self._torch = None
        self._mauve = None
        self._device = "cpu"

    def setup(self) -> None:
        if self.model_name != "m-a-p/MERT-v1-330M":
            logger.warning(
                "MAD is validated only with m-a-p/MERT-v1-330M; got %r",
                self.model_name,
            )
            return
        if self.sample_rate != 24000 or self.layer != 24 or self.aggregation != "max":
            logger.warning(
                "MAD requires the published setup sample_rate=24000, layer=24, "
                "aggregation=max"
            )
            return

        try:
            import mauve
            import torch
            from transformers import AutoModel, Wav2Vec2FeatureExtractor

            if self.device_config in ("auto", ""):
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device_config

            # The Hub repository also carries a 4 GB fairseq training
            # checkpoint. Transformers inference uses only pytorch_model.bin;
            # avoid downloading the unrelated duplicate.
            model_path = download_hf_snapshot(
                self.model_name,
                self.models_dir,
                ignore_patterns=["MERT-v1-330M_fairseq.pt"],
            )
            processor = Wav2Vec2FeatureExtractor.from_pretrained(
                str(model_path), trust_remote_code=True, local_files_only=True
            )
            model = AutoModel.from_pretrained(
                str(model_path), trust_remote_code=True, local_files_only=True
            )
            model.eval().to(device)

            self._torch = torch
            self._mauve = mauve
            self._processor = processor
            self._model = model
            self._device = device
            self._backend = "mad_mert_mauve"
            logger.info("MAD initialised with upstream MERT/MAUVE setup on %s", device)
        except ImportError as exc:
            logger.warning("MAD requires `pip install mad_metric`: %s", exc)
        except Exception as exc:
            logger.warning("MAD backend initialisation failed: %s", exc)

    def extract_features(self, sample: Sample) -> Optional[np.ndarray]:
        if (
            self._backend != "mad_mert_mauve"
            or self._model is None
            or self._processor is None
            or self._torch is None
        ):
            return None
        try:
            audio = load_audio(sample.path, target_sr=self.sample_rate, mono=True)
            if audio is None or audio.size == 0:
                return None
            inputs = self._processor(
                audio,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True,
            ).to(self._device)
            with self._torch.inference_mode():
                output = self._model(**inputs, output_hidden_states=True)
                hidden_states = output.hidden_states
                if hidden_states is None or self.layer >= len(hidden_states):
                    logger.warning(
                        "MAD MERT output has %d hidden-state layers; requested %d",
                        0 if hidden_states is None else len(hidden_states),
                        self.layer,
                    )
                    return None
                embedding = hidden_states[self.layer].max(dim=1).values
            array = embedding.detach().float().cpu().numpy()
            if array.ndim != 2 or array.shape[0] != 1 or not np.isfinite(array).all():
                logger.warning("MAD produced invalid embedding shape/value %s", array.shape)
                return None
            return array.astype(np.float32, copy=False)
        except Exception as exc:
            logger.warning("MAD feature extraction failed for %s: %s", sample.path, exc)
            return None

    def compute_distribution_metric(
        self,
        features: List[np.ndarray],
        reference_features: Optional[List[np.ndarray]] = None,
    ) -> float:
        if self._mauve is None:
            raise RuntimeError("upstream MAD backend is not initialised")
        if not reference_features:
            raise ValueError("MAD requires an explicit reference music distribution")

        generated = np.concatenate(features, axis=0).astype(np.float32, copy=False)
        reference = np.concatenate(reference_features, axis=0).astype(np.float32, copy=False)
        if len(generated) < 2 or len(reference) < 2:
            raise ValueError("MAD requires at least two samples in each distribution")

        result = self._mauve.compute_mauve(
            p_features=generated,
            q_features=reference,
            verbose=False,
        )
        mauve_score = float(result.mauve)
        if not np.isfinite(mauve_score) or mauve_score <= 0.0 or mauve_score > 1.0:
            raise ValueError(f"MAD received invalid MAUVE score {mauve_score!r}")
        value = float(-np.log(mauve_score))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("MAD returned a non-finite or negative divergence")
        return value

    def on_dispose(self) -> None:
        super().on_dispose()
        self._model = None
        self._processor = None
        if self._torch is not None and self._device.startswith("cuda"):
            try:
                self._torch.cuda.empty_cache()
            except Exception:
                pass
