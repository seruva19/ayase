"""ImageBind audio-text alignment score.

Computes cosine similarity between an audio sample and its caption in the
joint ImageBind embedding space (Girdhar et al., CVPR 2023). Useful as a
video-to-audio (v2a) generation evaluation metric, since ImageBind aligns
six modalities (image, text, audio, depth, IMU, thermal) into a single
1024-dim space.

The score ``imagebind_score`` is in [0, 1], higher means the generated audio
better matches the textual prompt.

Backend: ``imagebind_huge`` from the vendored ImageBind source
(https://github.com/facebookresearch/ImageBind).

``QualityMetrics.imagebind_score`` stores the per-sample scalar.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

from ayase.audio import load_audio
from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class ImageBindScoreModule(PipelineModule):
    name = "imagebind_score"
    description = "ImageBind audio-text alignment cosine similarity score"
    default_config = {
        "model_name": "imagebind_huge",
        "sample_rate": 16000,
        "device": "auto",
        "warning_threshold": 0.2,
    }
    models = [
        {
            "id": "imagebind_huge",
            "type": "other",
            "task": "ImageBind joint multimodal embedding for audio-text alignment",
        },
    ]
    metric_info = {
        "imagebind_score": "ImageBind audio-text alignment cosine similarity (0-1, higher=better)",
    }
    metric_groups = {
        "imagebind_score": "audio",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "imagebind_huge")
        self.sample_rate = self.config.get("sample_rate", 16000)
        self.device_config = self.config.get("device", "auto")
        self.warning_threshold = self.config.get("warning_threshold", 0.2)
        self._model = None
        self._device = "cpu"
        self._modality_type = None
        self._data_module = None
        self._ml_available = False

    _WEIGHTS_URLS = {
        "imagebind_huge.pth": (
            "https://huggingface.co/AkaneTendo25/ayase-models/resolve/main/"
            "imagebind/imagebind_huge.pth"
        ),
    }

    def _ensure_weights(self) -> bool:
        """Pre-position ``.checkpoints/imagebind_huge.pth`` from the Ayase HF
        mirror. The vendored ``imagebind_model.imagebind_huge(pretrained=True)``
        otherwise downloads from ``dl.fbaipublicfiles.com`` via
        ``torch.hub.download_url_to_file``, which has no timeout and hangs
        indefinitely on constrained networks. Returns True iff weights are
        present (either pre-existing or freshly downloaded)."""
        import os
        target_dir = ".checkpoints"
        target_path = os.path.join(target_dir, "imagebind_huge.pth")
        if os.path.exists(target_path) and os.path.getsize(target_path) > 1_000_000:
            return True
        url = self._WEIGHTS_URLS.get("imagebind_huge.pth")
        if not url:
            return False
        try:
            from ayase.config import download_model_file
            download_model_file(
                "imagebind_huge.pth",
                url,
                target_dir,
            )
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("ImageBind weights download failed: %s", e)
            return False
        return os.path.exists(target_path) and os.path.getsize(target_path) > 1_000_000

    def setup(self) -> None:
        try:
            import torch
            try:
                from imagebind import data as imagebind_data
                from imagebind.models import imagebind_model
                from imagebind.models.imagebind_model import ModalityType
            except ImportError:
                import sys

                vendor_root = Path(__file__).resolve().parents[1] / "vendor"
                if str(vendor_root) not in sys.path:
                    sys.path.insert(0, str(vendor_root))
                from imagebind import data as imagebind_data
                from imagebind.models import imagebind_model
                from imagebind.models.imagebind_model import ModalityType
        except ImportError:
            logger.warning("ImageBind source unavailable")
            self._ml_available = False
            return

        if not self._ensure_weights():
            logger.warning(
                "ImageBind weights unavailable (cannot download from "
                "AkaneTendo25/ayase-models HF mirror); module disabled"
            )
            self._ml_available = False
            return

        try:
            if self.device_config == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = self.device_config

            model = imagebind_model.imagebind_huge(pretrained=True)
            model.eval()
            model.to(self._device)

            self._model = model
            self._modality_type = ModalityType
            self._data_module = imagebind_data
            self._ml_available = True
            logger.info(
                "ImageBind initialised with %s on %s", self.model_name, self._device
            )
        except Exception as e:
            logger.warning("ImageBind setup failed: %s", e)
            self._ml_available = False

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        caption = _caption_text(sample)
        if not caption:
            return sample

        try:
            audio = load_audio(sample.path, target_sr=self.sample_rate, duration=10.0)
            if audio is None or len(audio) == 0:
                return sample

            score = self._score(audio, caption)
            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.imagebind_score = float(score)

            if score < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low ImageBind audio-text alignment: {score:.3f}",
                        details={
                            "imagebind_score": float(score),
                            "caption": caption[:80],
                            "threshold": float(self.warning_threshold),
                        },
                    )
                )
        except Exception as e:
            logger.warning("ImageBind scoring failed for %s: %s", sample.path, e)
        return sample

    # ------------------------------------------------------------------
    def _score(self, audio, caption: str) -> Optional[float]:
        """Compute audio-text cosine similarity via ImageBind.

        ``imagebind.data.load_and_transform_audio_data`` operates on file
        paths (it internally loads with torchaudio and builds mel-spectrogram
        clips), so we materialise the already-resampled mono PCM array to a
        temporary WAV file. This keeps the API contract identical to the
        upstream ImageBind examples.
        """
        try:
            import numpy as np
            import torch
        except ImportError as e:
            logger.debug("ImageBind scoring missing deps: %s", e)
            return None

        tmp_path: Optional[Path] = None
        try:
            import soundfile as sf

            audio_arr = np.asarray(audio, dtype=np.float32)
            if audio_arr.ndim > 1:
                audio_arr = audio_arr.mean(axis=1).astype(np.float32)

            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            tmp_path = Path(tmp.name)
            sf.write(str(tmp_path), audio_arr, self.sample_rate, subtype="PCM_16")

            ModalityType = self._modality_type
            data = self._data_module

            inputs = {
                ModalityType.TEXT: data.load_and_transform_text(
                    [caption], self._device
                ),
                ModalityType.AUDIO: data.load_and_transform_audio_data(
                    [str(tmp_path)], self._device
                ),
            }

            with torch.no_grad():
                embeddings = self._model(inputs)
                audio_emb = embeddings[ModalityType.AUDIO]
                text_emb = embeddings[ModalityType.TEXT]

                # If audio was chunked into multiple clips, average them.
                if audio_emb.dim() == 3:
                    audio_emb = audio_emb.mean(dim=1)

                audio_emb = audio_emb / audio_emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)

                sim = (audio_emb @ text_emb.T).item()

            return float((sim + 1.0) / 2.0)
        except Exception as e:
            logger.debug("ImageBind scoring failed: %s", e)
            return None
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass


def _caption_text(sample: Sample) -> Optional[str]:
    if sample.caption and sample.caption.text:
        return sample.caption.text
    sidecar = sample.path.with_suffix(".txt")
    try:
        if sidecar.exists():
            return sidecar.read_text(encoding="utf-8").strip()
    except Exception:
        logger.debug("Failed to read caption sidecar %s", sidecar)
    return None
