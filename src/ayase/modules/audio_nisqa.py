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

Backend: the real vendored NISQA model + checkpoint. When it cannot be loaded
(missing torch, or the checkpoint cannot be downloaded), the NISQA metrics are
left unset — there is no proxy/heuristic stand-in.
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
        self.models_dir = self.config.get("models_dir", "models")
        self.weights_path = self.config.get("weights_path", None)
        self._nisqa_model = None
        self._device = "cpu"
        self.active_backend = "unavailable"
        self._backend = "unavailable"

    def setup(self) -> None:
        if self._try_setup_nisqa():
            return
        logger.warning(
            "NISQA unavailable: the vendored NISQA model/checkpoint could not be "
            "loaded (requires torch and the nisqa.tar checkpoint); NISQA metrics "
            "will be left unset."
        )

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
            self._backend = "vendored:nisqa"
            logger.info("NISQA module initialized with vendored real model (weights=%s)", weights)
            return True
        except Exception as e:
            logger.debug(f"NISQA real model unavailable: {e}")
            return False

    def process(self, sample: Sample) -> Sample:
        if self.active_backend != "nisqa":
            return sample

        audio = self._load_or_extract_audio(sample)
        if audio is None or len(audio) < 100:
            return sample

        try:
            scores = self._score_with_nisqa(audio)
        except Exception as e:
            logger.warning("NISQA scoring failed for %s: %s", sample.path, e)
            return sample

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
