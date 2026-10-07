"""FAD — Frechet Audio Distance (Kilgour et al., Interspeech 2019).

Dataset-level metric measuring the distance between generated and reference
audio distributions, computed over *frame-level* embeddings of the chosen
backbone — the published FAD convention. Three backbones are supported,
routed through the fadtk framework (Gui et al., ICASSP 2024):

* ``vggish`` (default): 128-dim frame embeddings via fadtk's ``VGGishModel``
  (torchvggish, 16 kHz). ``frechet_audio_distance`` is accepted as an
  alternative VGGish package when fadtk is absent.
* ``panns_cnn14``: 2048-dim embeddings per 10 s window via ``panns_inference``,
  wrapped in a fadtk ``ModelLoader``.
* ``passt``: 768-dim embeddings per 10 s window via ``hear21passt``, wrapped
  in a fadtk ``ModelLoader``.

``infinity=True`` computes FAD-inf per the fadtk protocol: FAD measured at
multiple eval-set sizes n (with-replacement frame resampling), then linear
regression of FAD against 1/n — the intercept is FAD-inf.

Dependency hints::

    pip install fadtk                  # VGGish + FAD-inf framework (canonical)
    pip install frechet_audio_distance # alternative VGGish package
    pip install fadtk panns_inference  # for backbone = "panns_cnn14"
    pip install fadtk hear21passt      # for backbone = "passt"

A genuine reference set is required: without ``reference_path`` inputs the
metric is left unset (the sample set is never split as its own reference).

fad_* — lower is better (closer audio distributions).
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List

import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.base_modules import BatchMetricModule

logger = logging.getLogger(__name__)

# fadtk conventions: PANNs and PaSST are AudioSet models trained/evaluated on
# ~10 s clips; embeddings are therefore extracted per non-overlapping window.
_PANNS_WINDOW_SEC = 10.0
_PASST_WINDOW_SEC = 10.0

try:
    from fadtk.model_loader import ModelLoader as _FadtkModelLoader
except ImportError:
    _FadtkModelLoader = None


class _PANNLoader(_FadtkModelLoader or object):
    """fadtk ModelLoader for PANNs CNN14 (panns_inference).

    Emits (n_windows, 2048) frame-level embeddings — one 2048-d penultimate
    vector per non-overlapping ``_PANNS_WINDOW_SEC`` window.
    """

    def __init__(self, checkpoint_path: Optional[str] = None):
        if _FadtkModelLoader is None:
            raise RuntimeError("fadtk is required for the panns_cnn14 backbone")
        super().__init__("panns-cnn14", 2048, 32000)
        self._checkpoint_path = checkpoint_path
        self._tagging = None

    def load_model(self) -> None:
        from panns_inference import AudioTagging

        kwargs = {}
        if self._checkpoint_path:
            kwargs["checkpoint_path"] = str(self._checkpoint_path)
        self._tagging = AudioTagging(**kwargs)

    def _get_embedding(self, audio):
        import torch

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        win = int(round(self.sr * _PANNS_WINDOW_SEC))
        if len(audio) == 0:
            return torch.zeros((0, self.num_features))
        if len(audio) < win:
            audio = np.pad(audio, (0, win - len(audio)))
        n_win = len(audio) // win
        embs = []
        for i in range(n_win):
            clip = audio[i * win:(i + 1) * win][None, :]
            try:
                result = self._tagging.inference(clip, return_embedding=True)
                emb = result[1] if isinstance(result, (tuple, list)) else result
            except TypeError:
                # Older panns_inference without return_embedding — the 2048-d
                # penultimate activation is not exposed; skip the window.
                continue
            emb = np.asarray(emb, dtype=np.float32).reshape(-1)
            if emb.shape[0] == self.num_features:
                embs.append(torch.from_numpy(emb))
        if not embs:
            return torch.zeros((0, self.num_features))
        return torch.stack(embs, dim=0)


class _PaSSTLoader(_FadtkModelLoader or object):
    """fadtk ModelLoader for PaSST (hear21passt).

    Emits (n_windows, 768) frame-level embeddings — one penultimate vector per
    non-overlapping ``_PASST_WINDOW_SEC`` window.
    """

    def __init__(self, model_name: Optional[str] = None):
        if _FadtkModelLoader is None:
            raise RuntimeError("fadtk is required for the passt backbone")
        super().__init__("passt", 768, 32000)
        self._model_name = model_name

    def load_model(self) -> None:
        import torch
        from hear21passt.base import get_basic_model

        kwargs = {"mode": "embed_only"}
        if self._model_name:
            kwargs["model_name"] = self._model_name
        try:
            model = get_basic_model(**kwargs)
        except TypeError:
            model = get_basic_model(mode="embed_only")
        except Exception:
            kwargs["mode"] = "logits"
            try:
                model = get_basic_model(**kwargs)
            except TypeError:
                model = get_basic_model(mode="logits")
        self.model = model.eval().to(self.device)

    def _get_embedding(self, audio):
        import torch

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        win = int(round(self.sr * _PASST_WINDOW_SEC))
        if len(audio) == 0:
            return torch.zeros((0, self.num_features))
        if len(audio) < win:
            audio = np.pad(audio, (0, win - len(audio)))
        n_win = len(audio) // win
        embs = []
        with torch.no_grad():
            for i in range(n_win):
                clip = audio[i * win:(i + 1) * win]
                tensor = torch.from_numpy(clip).unsqueeze(0).to(self.device)
                try:
                    out = self.model(tensor)
                except Exception:
                    continue
                candidate = out[0] if isinstance(out, (tuple, list)) else out
                # 527-dim logits are classifier outputs, not embeddings —
                # route through the base encoder instead of accepting them.
                if candidate.ndim >= 2 and candidate.shape[-1] == 527:
                    net = getattr(self.model, "net", None)
                    if net is not None and hasattr(net, "forward_features"):
                        candidate = net.forward_features(tensor)
                    else:
                        continue
                emb = candidate.detach().reshape(-1)
                if emb.shape[0] == self.num_features:
                    embs.append(emb.float().cpu())
        if not embs:
            return torch.zeros((0, self.num_features))
        return torch.stack(embs, dim=0)


class _FadPackageVGGish:
    """fadtk-style adapter over the frechet_audio_distance package."""

    name = "vggish"
    num_features = 128
    sr = 16000
    min_len = 1

    def __init__(self):
        self._fad = None

    def load_model(self) -> None:
        from frechet_audio_distance import FrechetAudioDistance

        self._fad = FrechetAudioDistance()

    def get_embedding(self, audio: np.ndarray) -> np.ndarray:
        try:
            embd = self._fad.get_embeddings(
                [np.asarray(audio, dtype=np.float32)], sr=self.sr
            )
        except TypeError:
            embd = self._fad.get_embeddings([np.asarray(audio, dtype=np.float32)])
        arr = np.asarray(embd, dtype=np.float32)
        if arr.ndim >= 2:
            arr = arr.reshape(-1, arr.shape[-1])
        return arr


class FADModule(BatchMetricModule):
    name = "fad"
    provenance = "adapted"
    sources = {
        k: "FAD (Kilgour et al., Interspeech 2019); frame embeddings + FAD-inf via fadtk (Gui et al., ICASSP 2024) — https://github.com/microsoft/FADTK"
        for k in (
            "fad", "fad_infinity", "fad_vggish", "fad_vggish_infinity",
            "fad_panns", "fad_panns_infinity", "fad_passt", "fad_passt_infinity",
        )
    }
    deviations = {
        k: "Ayase exposes multiple backbone/runtime variants and optional FAD-inf extrapolation under one module; values are only comparable for the same backend and protocol settings, and a reference set is required"
        for k in (
            "fad", "fad_infinity", "fad_vggish", "fad_vggish_infinity",
            "fad_panns", "fad_panns_infinity", "fad_passt", "fad_passt_infinity",
        )
    }
    description = "Frechet Audio Distance for audio generation (batch metric, 2019)"
    default_config = {
        "subsample_videos": None,
        "infinity": False,
        # FAD-inf protocol parameters (fadtk score_inf conventions).
        "infinity_min_n": 500,
        "infinity_steps": 25,
        "random_seed": 1234,
        # Backbone selection: "vggish" (default), "panns_cnn14", "passt".
        "backbone": "vggish",
        "panns_checkpoint_path": None,
        "passt_model_name": "passt_s_kd_p16_128_ap486",
        "device": "auto",
    }
    models = [
        {
            "id": "fadtk",
            "type": "pip_package",
            "install": "pip install fadtk",
            "task": "Canonical FAD framework (VGGish loader + FAD-inf protocol)",
        },
        {
            "id": "frechet_audio_distance",
            "type": "pip_package",
            "install": "pip install frechet_audio_distance",
            "task": "Alternative VGGish backend when fadtk is absent",
        },
        {
            "id": "panns_inference",
            "type": "pip_package",
            "install": "pip install panns_inference",
            "task": "PANNs CNN14 backbone (2048-dim per-window embeddings) for FAD",
        },
        {
            "id": "hear21passt",
            "type": "pip_package",
            "install": "pip install hear21passt",
            "task": "PaSST backbone (768-dim per-window embeddings) for FAD",
        },
    ]
    metric_info = {
        "fad": "Frechet Audio Distance, VGGish backbone (lower=better)",
        "fad_infinity": "FAD VGGish extrapolated to infinite sample size",
        "fad_vggish": "Frechet Audio Distance, VGGish backbone (lower=better)",
        "fad_vggish_infinity": "FAD VGGish extrapolated to infinite sample size",
        "fad_panns": "Frechet Audio Distance, PANNs Cnn14 backbone",
        "fad_panns_infinity": "FAD PANNs Cnn14 extrapolated to infinite sample size",
        "fad_passt": "Frechet Audio Distance, PASST backbone",
        "fad_passt_infinity": "FAD PASST extrapolated to infinite sample size",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._ml_available = False
        self._backend = "unavailable"
        # Resolved backbone identity: "vggish" | "panns_cnn14" | "passt" | "unavailable".
        self._backend_kind = "unavailable"
        self._loader = None  # fadtk ModelLoader (or compatible adapter)

        self.subsample_videos = self.config.get("subsample_videos", None)
        self.infinity = self.config.get("infinity", False)
        self.infinity_min_n = self.config.get("infinity_min_n", 500)
        self.infinity_steps = self.config.get("infinity_steps", 25)
        self.random_seed = self.config.get("random_seed", 1234)
        self.backbone = str(self.config.get("backbone", "vggish")).lower()
        self.panns_checkpoint_path = self.config.get("panns_checkpoint_path", None)
        self.passt_model_name = self.config.get(
            "passt_model_name", "passt_s_kd_p16_128_ap486"
        )
        self._device_cfg = str(self.config.get("device", "auto")).lower()
        self._processed_count = 0

    # ------------------------------------------------------------------ setup

    def setup(self) -> None:
        if self.backbone not in ("vggish", "panns_cnn14", "passt"):
            logger.warning(
                "FAD: unknown backbone %r (expected 'vggish' | 'panns_cnn14' | 'passt'); "
                "falling back to 'vggish'",
                self.backbone,
            )
            self.backbone = "vggish"

        if self.backbone == "vggish":
            self._setup_vggish()
        elif self.backbone == "panns_cnn14":
            self._setup_panns()
        elif self.backbone == "passt":
            self._setup_passt()

        if self._backend_kind == "unavailable":
            self._backend = "unavailable"
            logger.warning(
                "FAD unavailable: install fadtk plus the backbone package "
                "(panns_inference / hear21passt). fad_* left unset."
            )

    def _bind_loader(self, loader, kind: str, backend: str) -> None:
        try:
            loader.load_model()
        except Exception as e:
            logger.warning("FAD %s loader failed: %s", kind, e)
            return
        self._loader = loader
        self._backend = backend
        self._backend_kind = kind
        self._ml_available = True
        logger.info("FAD module initialised (%s, sr=%d)", loader.name, loader.sr)

    def _setup_vggish(self) -> None:
        # Canonical: fadtk's VGGishModel (torchvggish, frame embeddings).
        try:
            from fadtk.model_loader import VGGishModel

            self._bind_loader(VGGishModel(), "vggish", "fadtk")
            if self._ml_available:
                return
        except ImportError:
            pass

        # Alternative published backend: frechet_audio_distance (also VGGish,
        # frame-level embeddings — we no longer mean-pool them).
        try:
            self._bind_loader(_FadPackageVGGish(), "vggish", "fad_package")
        except Exception:
            pass

    def _resolve_torch_device(self) -> str:
        if self._device_cfg not in ("auto", "", "none"):
            return self._device_cfg
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _setup_panns(self) -> None:
        try:
            import fadtk  # noqa: F401
            import panns_inference  # noqa: F401
        except ImportError as e:
            logger.warning(
                "FAD backbone panns_cnn14 requires fadtk and panns_inference: %s", e
            )
            return
        self._bind_loader(
            _PANNLoader(checkpoint_path=self.panns_checkpoint_path),
            "panns_cnn14",
            "fadtk_panns",
        )

    def _setup_passt(self) -> None:
        try:
            import fadtk  # noqa: F401
            import hear21passt  # noqa: F401
        except ImportError as e:
            logger.warning(
                "FAD backbone passt requires fadtk and hear21passt: %s", e
            )
            return
        self._bind_loader(
            _PaSSTLoader(model_name=self.passt_model_name), "passt", "fadtk_passt"
        )

    # ---------------------------------------------------------- per-sample

    def extract_features(self, sample: Sample) -> Optional[np.ndarray]:
        """Extract *frame-level* embeddings: (n_frames, D), one row per window."""
        if self.subsample_videos is not None and self._processed_count >= self.subsample_videos:
            return None

        if self._backend_kind == "unavailable" or self._loader is None:
            return None

        try:
            audio = self._load_audio(sample.path, self._loader.sr)
            if audio is None:
                return None

            emb = np.asarray(self._loader.get_embedding(audio), dtype=np.float64)
            if emb.ndim == 1:
                emb = emb[None, :]
            if emb.shape[0] == 0:
                return None

            self._processed_count += 1
            return emb
        except Exception as e:
            logger.debug(f"FAD feature extraction failed for {sample.path}: {e}")
            return None

    def _load_audio(self, path: Path, target_sr: int) -> Optional[np.ndarray]:
        """Load audio resampled to the loader's native rate.

        Resampling uses torchaudio's sinc-kaiser resampler (the fadtk
        convention) — never linear interpolation. Video inputs are decoded to
        a temporary WAV via ffmpeg first.
        """
        import soundfile as sf

        audio = None
        sr = None
        try:
            audio, sr = sf.read(str(path))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
        except Exception:
            pass

        if audio is None:
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp = Path(tmpdir) / "audio.wav"
                    cmd = [
                        "ffmpeg", "-y", "-i", str(path),
                        "-vn", "-ac", "1", "-sample_fmt", "s16", str(tmp),
                    ]
                    result = subprocess.run(cmd, capture_output=True, timeout=30)
                    if result.returncode != 0:
                        return None
                    audio, sr = sf.read(tmp, dtype="float32")
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)
            except Exception:
                return None

        audio = np.asarray(audio, dtype=np.float32)
        if sr != target_sr:
            try:
                import torch
                import torchaudio

                x = torch.from_numpy(audio).unsqueeze(0)
                resampler = torchaudio.transforms.Resample(
                    sr, target_sr, resampling_method="sinc_interp_kaiser"
                )
                audio = resampler(x).squeeze(0).numpy()
            except ImportError:
                import librosa

                audio = librosa.resample(
                    audio, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq"
                ).astype(np.float32)
        return audio

    # ---------------------------------------------------------- scoring

    def compute_distribution_metric(
        self, features: List[np.ndarray], reference_features: Optional[List[np.ndarray]] = None
    ) -> Optional[float]:
        """FAD between frame-level embedding distributions."""
        try:
            features_array = _concat_features(features)
            if features_array is None:
                return float("inf")

            if reference_features is not None and len(reference_features) > 0:
                ref_array = _concat_features(reference_features)
                if ref_array is None:
                    return float("inf")
            else:
                logger.info(
                    "FAD: no reference features provided; "
                    "metric is undefined without a reference set"
                )
                return None

            if self.infinity:
                return self._compute_fad_infinity(features_array, ref_array)
            return self._frechet_distance(features_array, ref_array)
        except Exception as e:
            logger.error(f"FAD computation failed: {e}")
            return float("inf")

    def _compute_fad_infinity(self, gen: np.ndarray, ref: np.ndarray) -> float:
        """FAD-inf per fadtk ``score_inf``: FAD at sizes n (with-replacement
        frame resampling) linearly extrapolated on the 1/n axis."""
        mu_ref, cov_ref = _stats(ref)
        if mu_ref is None:
            return float("inf")

        n_max = gen.shape[0]
        min_n = min(int(self.infinity_min_n), n_max)
        ns = sorted({int(n) for n in np.linspace(min_n, n_max, int(self.infinity_steps)) if n >= 2})
        if not ns:
            return self._frechet_distance(gen, ref)

        rng = np.random.default_rng(int(self.random_seed))
        xs, ys = [], []
        for n in ns:
            idx = rng.choice(n_max, size=n, replace=True)
            mu_e, cov_e = _stats(gen[idx])
            ys.append(_frechet(mu_ref, cov_ref, mu_e, cov_e))
            xs.append(1.0 / n)
        slope, intercept = np.polyfit(np.asarray(xs), np.asarray(ys), deg=1)
        return float(max(intercept, 0.0))

    def _frechet_distance(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        mu1, cov1 = _stats(feat1)
        mu2, cov2 = _stats(feat2)
        return _frechet(mu1, cov1, mu2, cov2)

    def on_dispose(self) -> None:
        if len(self._feature_cache) < 1:
            logger.info(f"FAD: Not enough samples ({len(self._feature_cache)})")
            self._feature_cache = []
            self._reference_cache = []
            self._processed_count = 0
            return

        try:
            score = self.compute_distribution_metric(
                self._feature_cache,
                self._reference_cache if self._reference_cache else None,
            )
            if score is None:
                return
            logger.info(
                "FAD: %.4f (%d samples, backbone=%s%s)",
                score,
                len(self._feature_cache),
                self._backend_kind,
                ", infinity" if self.infinity else "",
            )

            if hasattr(self, "pipeline") and self.pipeline:
                if hasattr(self.pipeline, "add_dataset_metric"):
                    backbone_suffix = {
                        "vggish": "vggish",
                        "panns_cnn14": "panns",
                        "passt": "passt",
                    }.get(self._backend_kind, "vggish")
                    inf_suffix = "_infinity" if self.infinity else ""
                    canonical_name = f"fad_{backbone_suffix}{inf_suffix}"
                    self.pipeline.add_dataset_metric(canonical_name, score)

                    # Backward-compat alias: the VGGish backbone keeps emitting
                    # the un-suffixed name so existing consumers see no change.
                    if backbone_suffix == "vggish":
                        alias = "fad_infinity" if self.infinity else "fad"
                        self.pipeline.add_dataset_metric(alias, score)
        except Exception as e:
            logger.error(f"FAD failed: {e}")
        finally:
            self._feature_cache = []
            self._reference_cache = []
            self._processed_count = 0


# ----------------------------------------------------------------- helpers


def _concat_features(features: List[np.ndarray]) -> Optional[np.ndarray]:
    """Concatenate per-sample (n_frames, D) embeddings into one (N, D) array."""
    if not features:
        return None
    try:
        arrs = [np.atleast_2d(np.asarray(f, dtype=np.float64)) for f in features]
    except Exception:
        return None
    if not arrs:
        return None
    dim = arrs[0].shape[1]
    if any(a.shape[1] != dim for a in arrs):
        logger.debug("FAD: inconsistent embedding dimensionality in feature cache")
        return None
    return np.concatenate(arrs, axis=0)


def _stats(feat: np.ndarray):
    """Mean/covariance of an embedding cloud (fadtk calc_embd_statistics)."""
    feat = np.atleast_2d(feat)
    mu = np.mean(feat, axis=0)
    if feat.shape[0] < 2:
        cov = np.zeros((feat.shape[1], feat.shape[1]), dtype=np.float64)
    else:
        cov = np.cov(feat, rowvar=False)
    return mu, np.atleast_2d(cov)


def _frechet(mu1, cov1, mu2, cov2, eps: float = 1e-6) -> float:
    """Frechet distance — the fadtk/pytorch-fid numerically-stable variant."""
    try:
        from scipy import linalg
        from numpy.lib.scimath import sqrt as scisqrt

        diff = mu1 - mu2
        D, V = linalg.eig(cov1.dot(cov2))
        covmean = (V * scisqrt(D)) @ linalg.inv(V)

        if not np.isfinite(covmean).all():
            offset = np.eye(cov1.shape[0]) * eps
            covmean = linalg.sqrtm((cov1 + offset).dot(cov2 + offset))

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        return float(diff.dot(diff) + np.trace(cov1) + np.trace(cov2) - 2 * np.trace(covmean))
    except ImportError:
        diff = mu1 - mu2
        return float(diff @ diff + np.trace(cov1) + np.trace(cov2))
