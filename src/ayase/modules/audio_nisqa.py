"""NISQA non-intrusive speech quality module.

Predicts speech quality on five dimensions (overall MOS, noisiness,
coloration, discontinuity, loudness) without a reference signal. Standard
in modern TTS evaluation pipelines.

Reference: Mittag et al., "NISQA: A Deep CNN-Self-Attention Model for
Multidimensional Speech Quality Prediction with Crowdsourced Datasets"
(Interspeech 2021, arXiv:2104.09494).

The upstream PyPI ``nisqa`` package pins old torch/numpy and cascading-
downgrades half the env on install. To avoid that, the inference code is
vendored at ``ayase/third_party/nisqa/`` (MIT-licensed, source-identical
to https://github.com/gabrielmittag/NISQA) and the ~1 MB checkpoint is
auto-fetched from ``AkaneTendo25/ayase-models``.

Tiered backend:
    1. Vendored NISQA + downloaded ``nisqa.tar`` checkpoint.
    2. UTMOS-derived proxy mapping (torch.hub SpeechMOS).
    3. Spectral-flatness / SNR signal-proxy fallback so the module is always
       runnable in CPU-only test environments.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


_NISQA_WEIGHTS_URL = "https://huggingface.co/AkaneTendo25/ayase-models/resolve/main/nisqa/nisqa.tar"
_NISQA_WEIGHTS_FILENAME = "nisqa/nisqa.tar"


class AudioNISQAModule(PipelineModule):
    name = "audio_nisqa"
    description = "NISQA multidimensional non-intrusive speech quality (MOS, noisiness, coloration, discontinuity, loudness)"
    default_config = {
        "target_sr": 48000,  # NISQA expects 48 kHz internally
        "backend": "auto",  # "auto" | "nisqa" | "utmos_proxy" | "spectral"
        "models_dir": "models",
        "weights_path": None,  # optional override; otherwise auto-download
    }
    models = [
        {
            "id": "nisqa.tar",
            "type": "local",
            "url": _NISQA_WEIGHTS_URL,
            "task": "NISQAv2 multidimensional speech quality (MIT)",
            "notes": "~1 MB; vendored source at ayase/third_party/nisqa/",
        },
    ]
    metric_groups = {
        "nisqa_coloration": "audio",
        "nisqa_discontinuity": "audio",
        "nisqa_loudness": "audio",
        "nisqa_mos": "audio",
        "nisqa_noisiness": "audio",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.target_sr = self.config.get("target_sr", 48000)
        self.backend_pref = self.config.get("backend", "auto")
        self.models_dir = self.config.get("models_dir", "models")
        self.weights_path = self.config.get("weights_path", None)
        self._nisqa_model = None
        self._utmos_model = None
        self._device = "cpu"
        self.active_backend = "spectral"

    def setup(self) -> None:
        if self.backend_pref in ("auto", "nisqa"):
            if self._try_setup_nisqa():
                return
        if self.backend_pref in ("auto", "utmos_proxy"):
            if self._try_setup_utmos():
                return
        logger.info("NISQA module using spectral signal-proxy fallback (no real model loaded)")

    def _try_setup_nisqa(self) -> bool:
        # nisqaModel's __init__ requires ``deg`` to point at a real audio file
        # — so we only resolve weights here and defer model construction to
        # the first process() call (see :meth:`_score_with_nisqa`).
        try:
            from ayase.third_party.nisqa import nisqaModel  # noqa: F401 — import check only
            from ayase.config import download_model_file
            import tempfile
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            weights = self.weights_path
            if not weights:
                weights = str(download_model_file(
                    _NISQA_WEIGHTS_FILENAME, _NISQA_WEIGHTS_URL, self.models_dir,
                ))
            self._nisqa_weights = weights
            self._nisqa_output_dir = tempfile.mkdtemp(prefix="ayase_nisqa_")
            self.active_backend = "nisqa"
            logger.info("NISQA module initialized with vendored real model (weights=%s)", weights)
            return True
        except Exception as e:
            logger.debug(f"NISQA real model unavailable: {e}")
            return False

    def _try_setup_utmos(self) -> bool:
        try:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._utmos_model = torch.hub.load(
                "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
            ).to(self._device)
            self._utmos_model.eval()
            self.active_backend = "utmos_proxy"
            logger.info("NISQA module using UTMOS proxy backend")
            return True
        except Exception as e:
            logger.debug(f"UTMOS proxy unavailable: {e}")
            return False

    def process(self, sample: Sample) -> Sample:
        audio = self._load_or_extract_audio(sample)
        if audio is None or len(audio) < 100:
            return sample

        try:
            if self.active_backend == "nisqa":
                scores = self._score_with_nisqa(audio)
            elif self.active_backend == "utmos_proxy":
                scores = self._score_with_utmos_proxy(audio)
            else:
                scores = self._score_with_spectral(audio)
        except Exception as e:
            logger.debug(f"NISQA scoring failed for {sample.path}: {e}")
            scores = self._score_with_spectral(audio)

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        mos, noi, col, disc, loud = scores
        sample.quality_metrics.nisqa_mos = round(float(mos), 3)
        sample.quality_metrics.nisqa_noisiness = round(float(noi), 3)
        sample.quality_metrics.nisqa_coloration = round(float(col), 3)
        sample.quality_metrics.nisqa_discontinuity = round(float(disc), 3)
        sample.quality_metrics.nisqa_loudness = round(float(loud), 3)
        return sample

    def _score_with_nisqa(self, audio: np.ndarray) -> Tuple[float, float, float, float, float]:
        # NISQA operates on file paths and returns a pandas DataFrame with
        # one row per file. Dump a temp wav, point ``deg`` at it, predict.
        from ayase.third_party.nisqa import nisqaModel
        import tempfile
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, self.target_sr)
            tmp_path = tmp.name
        try:
            if self._nisqa_model is None:
                self._nisqa_model = nisqaModel({
                    "mode": "predict_file",
                    "pretrained_model": self._nisqa_weights,
                    "deg": tmp_path,
                    "output_dir": self._nisqa_output_dir,
                    "tr_bs_val": 1,
                    "tr_num_workers": 0,
                    "ms_channel": 1,
                })
            else:
                self._nisqa_model.args["deg"] = tmp_path
                # Re-load the per-file dataset now that ``deg`` points at the
                # current temp wav (NISQA caches the file list internally).
                self._nisqa_model._loadDatasets()
            df = self._nisqa_model.predict()
            row = df.iloc[0]
            return (
                float(row["mos_pred"]),
                float(row["noi_pred"]),
                float(row["col_pred"]),
                float(row["dis_pred"]),
                float(row["loud_pred"]),
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _score_with_utmos_proxy(self, audio: np.ndarray) -> Tuple[float, float, float, float, float]:
        # UTMOS produces a single MOS; we project the heuristic sub-scores
        # around it so the five output dimensions remain informative.
        import torch

        wav = torch.from_numpy(audio).unsqueeze(0).to(self._device)
        with torch.no_grad():
            mos = float(self._utmos_model(wav, self.target_sr).item())
        _, noi, col, disc, loud = self._score_with_spectral(audio)
        # Anchor sub-scores around the UTMOS overall MOS so they remain
        # consistent rather than drifting from the heuristic.
        shift = mos - 3.0
        return (mos, _clip_mos(noi + shift), _clip_mos(col + shift),
                _clip_mos(disc + shift), _clip_mos(loud + shift))

    def _score_with_spectral(self, audio: np.ndarray) -> Tuple[float, float, float, float, float]:
        # Cheap, deterministic, CPU-only signal-based proxies. Each returns
        # a value on the 1-5 MOS scale.
        audio = audio.astype(np.float32)
        # Overall energy / RMS
        rms = float(np.sqrt(np.mean(audio ** 2) + 1e-12))
        # Spectral flatness — flat spectrum ≈ noisy, peaky ≈ tonal speech
        spectrum = np.abs(np.fft.rfft(audio[: 1 << 14] if len(audio) > 1 << 14 else audio)) + 1e-9
        geo_mean = np.exp(np.mean(np.log(spectrum)))
        arith_mean = np.mean(spectrum)
        flatness = float(geo_mean / arith_mean)
        # Crude SNR estimate from peak vs. low percentile
        p95 = float(np.percentile(np.abs(audio), 95))
        p50 = float(np.percentile(np.abs(audio), 50)) + 1e-9
        snr_like = p95 / p50

        mos = _clip_mos(2.5 + 0.5 * np.log1p(snr_like) - 1.5 * flatness)
        noi = _clip_mos(4.0 - 3.0 * flatness)
        col = _clip_mos(3.0 + 0.5 * (1.0 - flatness))
        disc = _clip_mos(3.5 - 0.5 * abs(np.log10(rms + 1e-3) + 1.5))
        # Loudness preference: target around -20 dBFS RMS
        loud = _clip_mos(4.0 - abs(20.0 * np.log10(rms + 1e-6) + 20.0) / 6.0)
        return (mos, noi, col, disc, loud)

    def _load_or_extract_audio(self, sample: Sample) -> Optional[np.ndarray]:
        audio = self._load_audio(sample.path)
        if audio is None and sample.is_video:
            audio = self._extract_audio_from_video(sample.path)
        return audio

    def _load_audio(self, path: Path) -> Optional[np.ndarray]:
        try:
            import soundfile as sf

            data, sr = sf.read(str(path), dtype="float32")
            if sr != self.target_sr:
                import librosa

                data = librosa.resample(data, orig_sr=sr, target_sr=self.target_sr)
            if data.ndim > 1:
                data = data.mean(axis=1)
            return data.astype(np.float32)
        except ImportError:
            logger.debug("soundfile/librosa not installed; cannot load audio")
            return None
        except Exception as e:
            logger.debug(f"Audio load failed: {e}")
            return None

    def _extract_audio_from_video(self, path: Path) -> Optional[np.ndarray]:
        import subprocess
        import tempfile

        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            cmd = [
                "ffmpeg", "-y", "-i", str(path),
                "-vn", "-ac", "1", "-ar", str(self.target_sr),
                "-sample_fmt", "s16", tmp.name,
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            if result.returncode != 0:
                Path(tmp.name).unlink(missing_ok=True)
                return None
            audio = self._load_audio(Path(tmp.name))
            Path(tmp.name).unlink(missing_ok=True)
            return audio
        except FileNotFoundError:
            return None
        except Exception:
            return None


def _clip_mos(x: float) -> float:
    return float(np.clip(x, 1.0, 5.0))
