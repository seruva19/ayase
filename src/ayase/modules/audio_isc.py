"""ISC-Audio Inception Score for Audio.

Intrinsic audio-classifier-based score adapted from Salimans et al. 2016. A
pretrained audio tagger (PANNs CNN14 by default, optionally PaSST) produces a
softmax distribution ``p(y|x)`` per sample; ``IS = exp( E_x[ KL(p(y|x) || p(y)) ] )``
where ``p(y) = mean_x p(y|x)`` is the marginal. Reported as mean and std over
``n_splits`` equal subsets so consumers can plot variance, matching the
convention used in the original IS paper.

Dependency hint::

    pip install panns_inference        # for backend = "panns_cnn14" (default)
    pip install hear21passt            # for backend = "passt"

Both backends are optional; if neither is installed the module logs a warning
and becomes a no-op so the rest of the pipeline keeps running.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

from ayase.audio import load_audio
from ayase.base_modules import BatchMetricModule
from ayase.models import Sample

logger = logging.getLogger(__name__)

_EPS = 1e-16


class AudioISCModule(BatchMetricModule):
    name = "audio_isc"
    description = "Inception Score for Audio (PANNs/PaSST softmax-based intrinsic score, batch metric)"
    default_config = {
        "backend": "panns_cnn14",
        "sample_rate": 16000,
        "panns_checkpoint_path": None,
        "passt_model_name": "passt_s_kd_p16_128_ap486",
        "n_splits": 10,
        "duration": 10.0,
        "device": "auto",
    }
    models = [
        {
            "id": "panns_inference",
            "type": "pip_package",
            "install": "pip install panns_inference",
            "task": "PANNs CNN14 audio tagger (527-class AudioSet softmax) backbone for ISC",
        },
        {
            "id": "Cnn14_mAP=0.431.pth",
            "type": "local",
            "task": "PANNs CNN14 pretrained weights, auto-downloaded by panns_inference on first use",
        },
        {
            "id": "hear21passt",
            "type": "pip_package",
            "install": "pip install hear21passt",
            "task": "Optional PaSST AudioSet classifier backbone for ISC",
        },
    ]
    metric_info = {
        "audio_isc_mean": "Inception Score for Audio, mean over n_splits subsets (PANNs/PASST backbone, higher=better)",
        "audio_isc_std": "Inception Score for Audio, std over n_splits subsets",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.backend = str(self.config.get("backend", "panns_cnn14")).lower()
        self.sample_rate = int(self.config.get("sample_rate", 16000))
        self.panns_checkpoint_path = self.config.get("panns_checkpoint_path", None)
        self.passt_model_name = self.config.get("passt_model_name", "passt_s_kd_p16_128_ap486")
        self.n_splits = max(1, int(self.config.get("n_splits", 10)))
        self.duration = self.config.get("duration", 10.0)
        self._device_cfg = str(self.config.get("device", "auto")).lower()

        self._ml_available = False
        self._tagger = None
        self._passt_model = None
        self._torch = None
        self._device = "cpu"

    # ------------------------------------------------------------------ setup

    def _resolve_device(self) -> str:
        try:
            import torch  # noqa: F401

            self._torch = torch
            if self._device_cfg in ("auto", "", None):
                return "cuda" if torch.cuda.is_available() else "cpu"
            return self._device_cfg
        except Exception:
            return "cpu"

    def setup(self) -> None:
        if self.backend not in ("panns_cnn14", "passt"):
            logger.warning(
                "audio_isc: unknown backend %r, falling back to panns_cnn14", self.backend
            )
            self.backend = "panns_cnn14"

        self._device = self._resolve_device()

        if self.backend == "panns_cnn14":
            self._setup_panns()
        elif self.backend == "passt":
            self._setup_passt()

    def _setup_panns(self) -> None:
        try:
            from panns_inference import AudioTagging
        except ImportError:
            logger.warning(
                "audio_isc backend panns_cnn14 requires `pip install panns_inference`; "
                "module will be a no-op"
            )
            return
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("audio_isc: failed to import panns_inference: %s", e)
            return

        try:
            ckpt = self.panns_checkpoint_path
            self._tagger = AudioTagging(
                checkpoint_path=str(ckpt) if ckpt else None,
                device=self._device,
            )
            self._ml_available = True
            logger.info(
                "audio_isc initialised with PANNs CNN14 on %s (sr=%d)",
                self._device,
                self.sample_rate,
            )
        except Exception as e:
            logger.warning("audio_isc: PANNs CNN14 load failed: %s", e)
            self._tagger = None

    def _setup_passt(self) -> None:
        try:
            from hear21passt.base import get_basic_model
        except ImportError:
            logger.warning(
                "audio_isc backend passt requires `pip install hear21passt`; module will be a no-op"
            )
            return
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("audio_isc: failed to import hear21passt: %s", e)
            return

        try:
            model = get_basic_model(mode="logits")
            if self._torch is not None:
                model = model.to(self._device).eval()
            else:
                model = model.eval()
            self._passt_model = model
            self._ml_available = True
            logger.info(
                "audio_isc initialised with PaSST (%s) on %s",
                self.passt_model_name,
                self._device,
            )
        except Exception as e:
            logger.warning("audio_isc: PaSST load failed: %s", e)
            self._passt_model = None

    # ----------------------------------------------------------- per-sample

    def extract_features(self, sample: Sample) -> Optional[np.ndarray]:
        if not self._ml_available:
            return None

        audio = load_audio(sample.path, target_sr=self.sample_rate, duration=self.duration)
        if audio is None or audio.size == 0:
            return None

        try:
            if self.backend == "panns_cnn14":
                return self._probs_panns(audio)
            if self.backend == "passt":
                return self._probs_passt(audio)
        except Exception as e:
            logger.debug("audio_isc feature extraction failed for %s: %s", sample.path, e)
            return None
        return None

    def _probs_panns(self, audio: np.ndarray) -> Optional[np.ndarray]:
        if self._tagger is None:
            return None
        # panns_inference expects shape [batch, samples], float32 in [-1, 1].
        batch = np.asarray(audio, dtype=np.float32)[None, :]
        clipwise_output, _ = self._tagger.inference(batch)
        # panns_inference already returns sigmoid (multi-label) scores; for IS we
        # need a proper softmax distribution so the KL is well-defined. We re-
        # normalise the per-class scores into a probability simplex with softmax.
        scores = np.asarray(clipwise_output, dtype=np.float64)[0]
        return _softmax(scores)

    def _probs_passt(self, audio: np.ndarray) -> Optional[np.ndarray]:
        if self._passt_model is None or self._torch is None:
            return None
        torch = self._torch
        # PaSST default front-end expects 32 kHz mono float waveform.
        wav = torch.as_tensor(audio, dtype=torch.float32, device=self._device)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        with torch.no_grad():
            logits = self._passt_model(wav)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float64)
        return probs[0]

    # ------------------------------------------------------------ aggregate

    def compute_distribution_metric(
        self,
        features: List[np.ndarray],
        reference_features: Optional[List[np.ndarray]] = None,
    ) -> float:
        """Return the mean IS over ``n_splits`` subsets.

        ISC is an intrinsic metric so ``reference_features`` is ignored. The
        base ``BatchMetricModule`` contract is single-float; we override
        ``on_dispose`` to publish both mean and std.
        """
        mean, _std = self._compute_mean_std(features)
        return mean

    def _compute_mean_std(self, features: List[np.ndarray]) -> Tuple[float, float]:
        if not features:
            return 0.0, 0.0

        probs = np.stack([np.asarray(f, dtype=np.float64) for f in features], axis=0)
        # Guard against negatives / NaN that some classifiers may produce.
        probs = np.clip(probs, 0.0, None)
        row_sum = probs.sum(axis=1, keepdims=True)
        row_sum[row_sum <= 0] = 1.0
        probs = probs / row_sum

        n = probs.shape[0]
        n_splits = self.n_splits
        if n_splits < 1:
            n_splits = 1
        if n < n_splits:
            logger.warning(
                "audio_isc: only %d samples available for n_splits=%d; "
                "computing single-split score (std=0)",
                n,
                n_splits,
            )
            return _inception_score(probs), 0.0

        # Match the canonical IS protocol: equal-size contiguous splits, leftover
        # samples dropped from the tail so every split has the same N.
        split_size = n // n_splits
        scores = np.empty(n_splits, dtype=np.float64)
        for i in range(n_splits):
            start = i * split_size
            end = start + split_size
            scores[i] = _inception_score(probs[start:end])
        return float(np.mean(scores)), float(np.std(scores))

    # -------------------------------------------------------------- dispose

    def on_dispose(self) -> None:
        try:
            if len(self._feature_cache) < 2:
                if self._feature_cache:
                    logger.info(
                        "audio_isc: not enough samples (%d) to compute Inception Score",
                        len(self._feature_cache),
                    )
                return

            try:
                mean, std = self._compute_mean_std(self._feature_cache)
            except Exception as e:
                logger.error("audio_isc computation failed: %s", e)
                return

            logger.info(
                "audio_isc: mean=%.4f std=%.4f (%d samples, backend=%s)",
                mean,
                std,
                len(self._feature_cache),
                self.backend,
            )

            if getattr(self, "pipeline", None) is not None and hasattr(
                self.pipeline, "add_dataset_metric"
            ):
                self.pipeline.add_dataset_metric("audio_isc_mean", float(mean))
                self.pipeline.add_dataset_metric("audio_isc_std", float(std))
        finally:
            self._feature_cache = []
            self._reference_cache = []
            # Let the base/PipelineModule chain release torch resources etc.
            super().on_dispose()


# ----------------------------------------------------------------- helpers


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    ex = np.exp(x)
    s = ex.sum()
    if s <= 0 or not np.isfinite(s):
        return np.full_like(ex, 1.0 / ex.size)
    return ex / s


def _inception_score(p_yx: np.ndarray) -> float:
    """Compute exp(E_x[KL(p(y|x) || p(y))]) for a [N, C] probability matrix."""
    if p_yx.size == 0:
        return 0.0
    p_y = p_yx.mean(axis=0, keepdims=True)
    log_p_yx = np.log(p_yx + _EPS)
    log_p_y = np.log(p_y + _EPS)
    kl = np.sum(p_yx * (log_p_yx - log_p_y), axis=1)
    return float(np.exp(np.mean(kl)))
