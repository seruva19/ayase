"""Audio KL divergence — paired classifier-logit metric (audioldm_eval protocol).

Follows ``audioldm_eval/metrics/kl.py``: per-sample logits from a pretrained
AudioSet classifier, paired between generated and reference audio **by matching
file basename**, then

::

    audio_kl = mean_i( KL( softmax(ref_logits_i) || softmax(gen_logits_i) ) )

the AudioGen formulation (``kullback_leibler_divergence_softmax``). Lower is
better. Reference logits are collected through ``sample.reference_path`` and
paired with generated samples by identical basename — the same pairing rule
upstream derives from folder structure. If no reference pairs can be formed
the metric is left unset.

Two backbones produce the logits:

* ``panns_cnn14`` — vendored PANNs Cnn14 at 16 kHz with the published
  ``Cnn14_16k_mAP=0.438.pth`` weights (the audioldm_eval backbone);
* ``passt``       — PaSST at 32 kHz (``pip install hear21passt``).

If neither backend is installed the module logs a warning and becomes a no-op.

The ``audio_kl`` field is not declared on :class:`QualityMetrics`; this is a
batch / dataset-level metric and the final scalar is written through
``pipeline.add_dataset_metric("audio_kl", score)``.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from ayase.base_modules import BatchMetricModule
from ayase.models import Sample

logger = logging.getLogger(__name__)

_EPS = 1e-6


class AudioKLModule(BatchMetricModule):
    name = "audio_kl"
    provenance = "adapted"
    sources = {
        "audio_kl": "audioldm_eval paired KL (AudioGen formulation) — https://github.com/haoheliu/audioldm_eval/blob/main/audioldm_eval/metrics/kl.py",
    }
    deviations = {
        "audio_kl": "backend='passt' swaps the classifier for PaSST-32k — a valid KL backbone but not the audioldm_eval one",
    }
    description = (
        "Paired KL divergence between audio-classifier softmax distributions "
        "(audioldm_eval protocol, lower=better)"
    )
    default_config = {
        "backend": "panns_cnn14",
        "panns_checkpoint_path": None,
        "passt_model_name": "passt_s_kd_p16_128_ap486",
        "device": "auto",
        "duration": 10.0,
    }
    models = [
        {
            "id": "Cnn14_16k_mAP=0.438.pth",
            "type": "local",
            "url": "https://zenodo.org/records/3987831/files/Cnn14_16k_mAP%3D0.438.pth",
            "task": "PANNs Cnn14-16k AudioSet classifier — the audioldm_eval KL backbone",
            "auto_download": True,
        },
        {
            "id": "torchlibrosa",
            "type": "pip_package",
            "install": "pip install torchlibrosa",
            "task": "Log-mel front-end for the vendored Cnn14_16k",
        },
        {
            "id": "passt_s_kd_p16_128_ap486",
            "type": "pip_package",
            "install": "pip install hear21passt",
            "task": "Optional PaSST AudioSet classifier backbone (non-default)",
        },
    ]
    metric_info = {
        "audio_kl": (
            "Paired KL divergence between audio classifier softmax distributions "
            "(audioldm_eval protocol, lower=better)"
        ),
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.backend = str(self.config.get("backend", "panns_cnn14")).lower()
        # Each classifier consumes audio at its native rate (Cnn14_16k @ 16 kHz;
        # PaSST is a 32 kHz model).
        self.sample_rate = 32000 if self.backend == "passt" else 16000
        self.panns_checkpoint_path = self.config.get("panns_checkpoint_path", None)
        self.passt_model_name = self.config.get(
            "passt_model_name", "passt_s_kd_p16_128_ap486"
        )
        self.device_config = self.config.get("device", "auto")
        self.duration = float(self.config.get("duration", 10.0))

        self._model = None
        self._device = "cpu"
        self._ml_available = False
        # Sample basename aligned with _feature_cache / _reference_cache.
        self._feature_names: List[str] = []
        self._reference_names: List[str] = []

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
            import torchlibrosa  # noqa: F401
            from ayase.third_party.panns_cnn14 import Cnn14
            from ayase.config import download_model_file
        except ImportError:
            logger.warning(
                "audio_kl backend panns_cnn14 requires torchlibrosa "
                "(pip install torchlibrosa)"
            )
            self._ml_available = False
            return
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("audio_kl: failed to import Cnn14_16k deps: %s", e)
            self._ml_available = False
            return

        try:
            ckpt = self.panns_checkpoint_path
            if not ckpt:
                ckpt = download_model_file(
                    "panns/Cnn14_16k_mAP=0.438.pth",
                    "https://zenodo.org/records/3987831/files/Cnn14_16k_mAP%3D0.438.pth",
                    self.config.get("models_dir", "models"),
                )
            model = Cnn14(
                features_list=["2048", "logits"],
                sample_rate=16000,
                window_size=512,
                hop_size=160,
                mel_bins=64,
                fmin=50,
                fmax=8000,
                classes_num=527,
            )
            model.load_checkpoint(str(ckpt))
            self._model = model.to(self._device).eval()
            self._ml_available = True
            logger.info(
                "audio_kl initialised with PANNs Cnn14_16k on %s", self._device
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
            self._ml_available = True
            logger.info(
                "audio_kl initialised (passt:%s) on %s",
                self.passt_model_name,
                self._device,
            )
        except Exception as e:
            logger.warning("audio_kl passt setup failed: %s", e)
            self._ml_available = False

    # ------------------------------------------------------------------
    def process(self, sample: Sample) -> Sample:
        # Track basenames alongside the base-class caches so pairs can be
        # formed the upstream way (identical file names).
        before_gen = len(self._feature_cache)
        before_ref = len(self._reference_cache)
        sample = super().process(sample)
        if len(self._feature_cache) > before_gen:
            self._feature_names.append(sample.path.name)
        if len(self._reference_cache) > before_ref:
            from pathlib import Path

            self._reference_names.append(Path(sample.reference_path).name)
        return sample

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

            if self.backend == "panns_cnn14":
                return self._logits_panns(audio)
            if self.backend == "passt":
                return self._logits_passt(audio)
            return None
        except Exception as e:
            logger.debug("audio_kl feature extraction failed for %s: %s", sample.path, e)
            return None

    def _logits_panns(self, audio: np.ndarray) -> Optional[np.ndarray]:
        if self._model is None:
            return None
        import torch

        batch = torch.as_tensor(
            np.asarray(audio, dtype=np.float32)[None, :], device=self._device
        )
        try:
            with torch.no_grad():
                out = self._model(batch)
        except Exception as e:
            logger.debug("audio_kl panns inference failed: %s", e)
            return None
        logits = out["logits"] if isinstance(out, dict) else out[0]
        return logits.detach().cpu().numpy().astype(np.float64)[0]

    def _logits_passt(self, audio: np.ndarray) -> Optional[np.ndarray]:
        try:
            import torch
        except ImportError:
            return None

        tensor = torch.as_tensor(audio, dtype=torch.float32, device=self._device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        try:
            with torch.no_grad():
                logits = self._model(tensor)
        except Exception as e:
            logger.debug("audio_kl passt inference failed: %s", e)
            return None
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        return logits.squeeze(0).detach().cpu().numpy().astype(np.float64)

    # ------------------------------------------------------------------
    def compute_distribution_metric(
        self,
        features: List[np.ndarray],
        reference_features: Optional[List[np.ndarray]] = None,
    ) -> Optional[float]:
        """Paired KL(ref_i || gen_i) over samples with identical basenames."""
        if not reference_features:
            logger.info(
                "audio_kl: no reference features provided; "
                "metric is undefined without a reference set"
            )
            return None

        gen_by_name: Dict[str, np.ndarray] = {}
        for name, logits in zip(self._feature_names, features):
            gen_by_name[name] = np.asarray(logits, dtype=np.float64)

        pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        for name, logits in zip(self._reference_names, reference_features):
            gen = gen_by_name.get(name)
            if gen is not None:
                pairs.append((gen, np.asarray(logits, dtype=np.float64)))

        if not pairs:
            logger.info(
                "audio_kl: no generated/reference pairs with matching basenames; "
                "metric is undefined"
            )
            return None

        per_pair = np.empty(len(pairs), dtype=np.float64)
        for i, (gen_logits, ref_logits) in enumerate(pairs):
            # Upstream AudioGen formulation: KL(softmax(ref) || softmax(gen)),
            # summed over classes.
            log_gen = _log_softmax(gen_logits + _EPS)
            ref_probs = _softmax_vec(ref_logits)
            per_pair[i] = float(np.sum(ref_probs * (np.log(ref_probs + _EPS) - log_gen)))
        return float(max(per_pair.mean(), 0.0))

    def on_dispose(self) -> None:
        self._feature_names = []
        self._reference_names = []
        super().on_dispose()


# ---------------------------------------------------------------------------
def _softmax_vec(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    s = e.sum()
    if s <= 0 or not np.isfinite(s):
        return np.full_like(e, 1.0 / e.size)
    return e / s


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    logsum = np.log(np.sum(np.exp(x)))
    return x - logsum
