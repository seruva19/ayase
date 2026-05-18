"""Audio KL divergence — distributional metric over audio-classifier softmax.

Compares the distribution of softmax probabilities produced by a pre-trained
audio classifier on generated audio against the same distribution on reference
audio. The classifier is pretrained on AudioSet (527 sound classes) so the
softmax acts as a perceptual "semantic" descriptor.

This metric is used by audio-generation papers such as MMAudio, FoleyGen and
AudioGen. Two backbones are supported:

* ``panns_cnn14``  — PANNs CNN14 from Kong et al., 2020
  (https://github.com/qiuqiangkong/panns_inference, ``pip install panns_inference``).
* ``passt``        — PaSST from Koutini et al., 2022
  (https://github.com/kkoutini/PaSST, ``pip install hear21passt``).

Formula::

    KL = mean_i( sum_c gen_probs[i, c] * log( gen_probs[i, c] / mean_ref_probs[c] ) )

Lower is better. If neither backend is installed the module logs a warning and
becomes a no-op so the rest of the pipeline keeps running.

The ``audio_kl`` field is not declared on :class:`QualityMetrics`; this is a
batch / dataset-level metric and the final scalar is written through
``pipeline.add_dataset_metric("audio_kl", score)`` in ``on_dispose`` (inherited
from :class:`BatchMetricModule`).
"""

import logging
from typing import List, Optional

import numpy as np

from ayase.base_modules import BatchMetricModule
from ayase.models import Sample

logger = logging.getLogger(__name__)


class AudioKLModule(BatchMetricModule):
    name = "audio_kl"
    description = (
        "KL divergence between audio-classifier softmax distributions "
        "(PANNs/PASST backbone, lower=better)"
    )
    default_config = {
        "backend": "panns_cnn14",
        "sample_rate": 16000,
        "panns_checkpoint_path": None,
        "passt_model_name": "passt_s_kd_p16_128_ap486",
        "device": "auto",
        "duration": 10.0,
    }
    models = [
        {
            "id": "panns_cnn14",
            "type": "pip_package",
            "install": "pip install panns_inference",
            "task": "PANNs CNN14 AudioSet classifier (527 classes) for KL-style scoring",
        },
        {
            "id": "passt_s_kd_p16_128_ap486",
            "type": "pip_package",
            "install": "pip install hear21passt",
            "task": "PaSST AudioSet classifier (527 classes) for KL-style scoring",
        },
    ]
    metric_info = {
        "audio_kl": (
            "KL divergence between audio classifier softmax distributions "
            "(PANNs/PASST backbone, lower=better)"
        ),
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.backend = str(self.config.get("backend", "panns_cnn14")).lower()
        self.sample_rate = int(self.config.get("sample_rate", 16000))
        self.panns_checkpoint_path = self.config.get("panns_checkpoint_path", None)
        self.passt_model_name = self.config.get(
            "passt_model_name", "passt_s_kd_p16_128_ap486"
        )
        self.device_config = self.config.get("device", "auto")
        self.duration = float(self.config.get("duration", 10.0))

        self._model = None
        self._device = "cpu"
        self._ml_available = False
        # Resolved backend (may differ from the config if a fallback kicks in).
        self._active_backend: Optional[str] = None
        # PaSST helper bound at setup time.
        self._passt_get_basic_model = None

    # ------------------------------------------------------------------
    def setup(self) -> None:
        try:
            import torch
        except ImportError:
            logger.warning(
                "audio_kl requires `pip install torch`; module disabled"
            )
            self._ml_available = False
            return

        if self.device_config == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self.device_config

        if self.backend == "panns_cnn14":
            self._setup_panns(torch)
        elif self.backend == "passt":
            self._setup_passt(torch)
        else:
            logger.warning(
                "audio_kl: unknown backend %r (expected 'panns_cnn14' or 'passt')",
                self.backend,
            )
            self._ml_available = False

    def _setup_panns(self, torch) -> None:
        try:
            from panns_inference import AudioTagging
        except ImportError:
            logger.warning(
                "audio_kl backend panns_cnn14 requires `pip install panns_inference`"
            )
            self._ml_available = False
            return

        try:
            kwargs = {"device": self._device}
            if self.panns_checkpoint_path:
                kwargs["checkpoint_path"] = self.panns_checkpoint_path
            model = AudioTagging(**kwargs)
            # ``AudioTagging`` wraps the actual nn.Module on ``.model``.
            inner = getattr(model, "model", None)
            if inner is not None and hasattr(inner, "eval"):
                inner.eval()
            self._model = model
            self._active_backend = "panns_cnn14"
            self._ml_available = True
            logger.info(
                "audio_kl initialised (panns_cnn14) on %s; native sr=32000, "
                "resampling from %d as needed",
                self._device,
                self.sample_rate,
            )
        except Exception as e:
            logger.warning("audio_kl panns_cnn14 setup failed: %s", e)
            self._ml_available = False

    def _setup_passt(self, torch) -> None:
        try:
            from hear21passt.base import get_basic_model
        except ImportError:
            logger.warning(
                "audio_kl backend passt requires `pip install hear21passt`"
            )
            self._ml_available = False
            return

        try:
            model = get_basic_model(mode="logits")
            model.eval()
            model.to(self._device)
            self._model = model
            self._passt_get_basic_model = get_basic_model
            self._active_backend = "passt"
            self._ml_available = True
            logger.info(
                "audio_kl initialised (passt:%s) on %s; native sr=32000, "
                "resampling from %d as needed",
                self.passt_model_name,
                self._device,
                self.sample_rate,
            )
        except Exception as e:
            logger.warning("audio_kl passt setup failed: %s", e)
            self._ml_available = False

    # ------------------------------------------------------------------
    def extract_features(self, sample: Sample) -> Optional[np.ndarray]:
        if not self._ml_available:
            return None

        try:
            from ayase.audio import load_audio

            audio = load_audio(
                sample.path, target_sr=self.sample_rate, duration=self.duration
            )
            if audio is None or len(audio) == 0:
                return None

            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim > 1:
                audio = audio.mean(axis=1).astype(np.float32)

            if self._active_backend == "panns_cnn14":
                return self._probs_panns(audio)
            if self._active_backend == "passt":
                return self._probs_passt(audio)
            return None
        except Exception as e:
            logger.debug("audio_kl feature extraction failed for %s: %s", sample.path, e)
            return None

    def _probs_panns(self, audio: np.ndarray) -> Optional[np.ndarray]:
        # PANNs CNN14 is trained at 32 kHz; resample if loaded at a different rate.
        x = _resample_linear(audio, self.sample_rate, 32000)
        # AudioTagging.inference expects a 2-D batch [B, T].
        batch = x[np.newaxis, :].astype(np.float32)
        try:
            clipwise, _embedding = self._model.inference(batch)
        except Exception as e:
            logger.debug("audio_kl panns inference failed: %s", e)
            return None

        probs = np.asarray(clipwise).reshape(-1)
        # ``clipwise_output`` from panns_inference is sigmoid-multi-label, not a
        # categorical softmax. KL-divergence over distributions requires a
        # probability simplex, so we apply softmax over the logit-space view
        # (log) of the sigmoid probabilities to obtain a valid distribution.
        probs = np.clip(probs, 1e-10, 1.0 - 1e-10)
        logits = np.log(probs) - np.log(1.0 - probs)
        return _softmax(logits)

    def _probs_passt(self, audio: np.ndarray) -> Optional[np.ndarray]:
        try:
            import torch
        except ImportError:
            return None

        x = _resample_linear(audio, self.sample_rate, 32000)
        # PaSST expects ~10 s clips at 32 kHz, shape [B, T].
        tensor = torch.from_numpy(x).float().unsqueeze(0).to(self._device)
        try:
            with torch.no_grad():
                logits = self._model(tensor)
        except Exception as e:
            logger.debug("audio_kl passt inference failed: %s", e)
            return None

        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
        return probs.astype(np.float64)

    # ------------------------------------------------------------------
    def compute_distribution_metric(
        self,
        features: List[np.ndarray],
        reference_features: Optional[List[np.ndarray]] = None,
    ) -> float:
        try:
            gen = np.stack(features, axis=0).astype(np.float64)
        except Exception as e:
            logger.debug("audio_kl: could not stack features: %s", e)
            return float("inf")

        if reference_features:
            try:
                ref = np.stack(reference_features, axis=0).astype(np.float64)
            except Exception as e:
                logger.debug("audio_kl: could not stack reference features: %s", e)
                return float("inf")
        else:
            mid = len(gen) // 2
            if mid < 1:
                return float("inf")
            ref = gen[:mid]
            gen = gen[mid:]

        if len(gen) < 1 or len(ref) < 1:
            return float("inf")

        eps = 1e-10
        # Re-normalise rows in case the backbone produced unnormalised scores.
        gen = gen / np.clip(gen.sum(axis=1, keepdims=True), eps, None)
        ref = ref / np.clip(ref.sum(axis=1, keepdims=True), eps, None)

        mean_ref = ref.mean(axis=0)
        # Per-sample KL( gen_i || mean_ref )
        kl_terms = gen * (np.log(gen + eps) - np.log(mean_ref + eps))
        per_sample = kl_terms.sum(axis=1)
        return float(max(per_sample.mean(), 0.0))


# ---------------------------------------------------------------------------
def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    s = e.sum()
    if s <= 0:
        return np.full_like(e, 1.0 / e.size)
    return e / s


def _resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Cheap linear-interpolation resampler.

    The backbones expect 32 kHz; this keeps the module dependency-free for the
    common case where ``ayase.audio.load_audio`` already returned the requested
    sample rate. For high-fidelity resampling install ``librosa`` and feed the
    module at its native rate via ``sample_rate=32000``.
    """
    if src_sr == dst_sr or len(audio) == 0:
        return audio.astype(np.float32)
    duration = len(audio) / float(src_sr)
    n_target = max(1, int(round(duration * dst_sr)))
    indices = np.linspace(0, len(audio) - 1, n_target)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
