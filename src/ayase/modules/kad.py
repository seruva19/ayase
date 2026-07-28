"""Kernel Audio Distance using the upstream KADTK PANNs backend.

KAD is an unbiased, distribution-free MMD estimator over learned audio
embeddings. This implementation uses the paper's recommended PANNs
Wavegram-Logmel encoder and the upstream ``kadtk`` distance function. Lower is
better. Small negative values are valid for the unbiased finite-sample
estimator and are not clipped.
"""

import logging
from typing import List, Optional

import numpy as np

from ayase.audio import load_audio
from ayase.base_modules import BatchMetricModule
from ayase.config import download_model_file
from ayase.models import Sample

logger = logging.getLogger(__name__)

_PANN_WAVEGRAM_URL = (
    "https://zenodo.org/records/3987831/files/"
    "Wavegram_Logmel_Cnn14_mAP%3D0.439.pth"
)


class KADModule(BatchMetricModule):
    name = "kad"
    description = "Kernel Audio Distance with KADTK PANNs embeddings (2025)"
    default_config = {
        "sample_rate": 32000,
        "backbone": "panns-wavegram-logmel",
        "bandwidth": None,
        "kernel": "gaussian",
        "device": "auto",
        "models_dir": "models",
    }
    models = [
        {
            "id": "kadtk",
            "type": "pip_package",
            "install": "pip install kadtk",
            "task": "Kernel Audio Distance implementation and PANNs architecture",
        },
        {
            "id": "kad/Wavegram_Logmel_Cnn14_mAP=0.439.pth",
            "type": "local",
            "url": _PANN_WAVEGRAM_URL,
            "task": "PANNs Wavegram-Logmel audio embedding checkpoint",
            "auto_download": True,
        },
    ]
    metric_info = {
        "kad": (
            "Kernel Audio Distance with PANNs Wavegram-Logmel embeddings "
            "(unbiased finite-sample estimate ×100, lower=better)"
        ),
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.sample_rate = int(self.config.get("sample_rate", 32000))
        self.backbone = str(self.config.get("backbone", "panns-wavegram-logmel"))
        self.bandwidth = self.config.get("bandwidth", None)
        self.kernel = str(self.config.get("kernel", "gaussian"))
        self.device_config = str(self.config.get("device", "auto"))
        self.models_dir = str(self.config.get("models_dir", "models"))
        self._backend = "unavailable"
        self._model = None
        self._torch = None
        self._device = "cpu"
        self._kad_fn = None

    def setup(self) -> None:
        if self.backbone != "panns-wavegram-logmel":
            logger.warning(
                "KAD supports only the validated panns-wavegram-logmel backbone; got %r",
                self.backbone,
            )
            return

        try:
            import torch
            from kadtk.kad import calc_kernel_audio_distance
            from kadtk.model_loader import PANNsModel
            from kadtk.models import panns

            if self.device_config in ("auto", ""):
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device_config

            checkpoint = download_model_file(
                "kad/Wavegram_Logmel_Cnn14_mAP=0.439.pth",
                _PANN_WAVEGRAM_URL,
                self.models_dir,
            )

            # Keep KADTK's exact loader/get_embedding contract, while placing
            # the checkpoint in Ayase's configurable models_dir.
            model = PANNsModel("wavegram-logmel")
            network = panns.Wavegram_Logmel_Cnn14(
                sample_rate=32000,
                window_size=1024,
                hop_size=320,
                mel_bins=64,
                fmin=50,
                fmax=14000,
                classes_num=527,
            )
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            network.load_state_dict(state["model"])
            network.eval().to(device)
            model.model_file = checkpoint
            model.model = network
            model.device = torch.device(device)

            self._torch = torch
            self._device = device
            self._model = model
            self._kad_fn = calc_kernel_audio_distance
            self._backend = "kadtk_panns_wavegram_logmel"
            logger.info("KAD initialised with upstream KADTK PANNs backend on %s", device)
        except ImportError as exc:
            logger.warning("KAD requires the upstream `kadtk` package: %s", exc)
        except Exception as exc:
            logger.warning("KAD backend initialisation failed: %s", exc)

    def extract_features(self, sample: Sample) -> Optional[np.ndarray]:
        if self._backend != "kadtk_panns_wavegram_logmel" or self._model is None:
            return None
        try:
            audio = load_audio(sample.path, target_sr=self.sample_rate, mono=True)
            if audio is None or audio.size == 0:
                return None
            embedding = np.asarray(self._model.get_embedding(audio))
            if embedding.ndim == 1:
                embedding = embedding[None, :]
            if embedding.ndim != 2 or embedding.shape[1] != 2048:
                logger.warning("KAD produced unexpected embedding shape %s", embedding.shape)
                return None
            return embedding.astype(np.float32, copy=False)
        except Exception as exc:
            logger.warning("KAD feature extraction failed for %s: %s", sample.path, exc)
            return None

    def compute_distribution_metric(
        self,
        features: List[np.ndarray],
        reference_features: Optional[List[np.ndarray]] = None,
    ) -> float:
        if self._kad_fn is None or self._torch is None:
            raise RuntimeError("upstream KADTK backend is not initialised")
        if not reference_features:
            raise ValueError("KAD requires an explicit reference audio distribution")

        generated = np.concatenate(features, axis=0).astype(np.float32, copy=False)
        reference = np.concatenate(reference_features, axis=0).astype(np.float32, copy=False)
        if len(generated) < 2 or len(reference) < 2:
            raise ValueError("KAD requires at least two embedding frames in each distribution")

        # KADTK derives the adaptive bandwidth from its second argument. Pass
        # reference second so the score is stable when generated sets change.
        score = self._kad_fn(
            self._torch.from_numpy(generated),
            self._torch.from_numpy(reference),
            (None, None),
            self._device,
            bandwidth=self.bandwidth,
            kernel=self.kernel,
        )
        value = float(score.detach().cpu().item())
        if not np.isfinite(value):
            raise ValueError("KAD returned a non-finite score")
        return value

    def on_dispose(self) -> None:
        super().on_dispose()
        self._model = None
        if self._torch is not None and self._device.startswith("cuda"):
            try:
                self._torch.cuda.empty_cache()
            except Exception:
                pass
