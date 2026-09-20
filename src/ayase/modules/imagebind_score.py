"""ImageBind audio-text and audio-video semantic correspondence.

``imagebind_av_score`` follows ``calc_imagebind_score(...).sim_av`` from the
official JavisBench implementation in JavisDiT commit ``6821b8d``: it is the
raw cosine similarity between the paired ImageBind video and audio embeddings.
The theoretical cosine range is [-1, 1], and higher means stronger semantic
audio-video correspondence. It does **not** measure synchronization, event
timing, perceptual quality, speaker/subject identity, or causal correctness.

The vendored ImageBind preprocessing samples five 2-second video clips (two
uniformly sampled frames and three spatial crops per clip) and three 2-second
audio clips across the complete files. ImageBind averages those clip embeddings
before the cosine is computed. This is file-level sampling, not an aligned or
sliding audio-video window calculation.

The existing ``imagebind_score`` remains the remapped [0, 1] audio-text cosine.
Both fields use the vendored ``imagebind_huge`` research backend. ImageBind is
licensed CC BY-NC-SA 4.0, so this backend is constrained to non-commercial,
share-alike research use rather than unrestricted commercial use.
"""

import logging
from pathlib import Path
from typing import Optional

from ayase.audio import load_audio
from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class ImageBindScoreModule(PipelineModule):
    name = "imagebind_score"
    description = "ImageBind audio-text and audio-video semantic cosine similarities"
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
            "task": "ImageBind joint multimodal embedding for audio-text and audio-video alignment",
            "notes": (
                "Vendored facebookresearch/ImageBind research backend under CC BY-NC-SA 4.0; "
                "audio-video scoring follows JavisVerse/JavisDiT calc_imagebind_score "
                "sim_av at commit 6821b8d"
            ),
        },
    ]
    metric_info = {
        "imagebind_score": "ImageBind audio-text alignment cosine similarity (0-1, higher=better)",
        "imagebind_av_score": (
            "Raw ImageBind audio-video embedding cosine similarity (-1 to 1 theoretical; "
            "higher=greater semantic correspondence, not synchronization)"
        ),
    }
    metric_groups = {
        "imagebind_score": "audio",
        "imagebind_av_score": "alignment",
    }
    vendor_components = ("imagebind",)

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
        self._backend = None

    _WEIGHTS_URLS = {
        "imagebind_huge.pth": (
            "https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/"
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
        from ayase.licenses import announce

        announce(self.vendor_components)
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
            self._backend = "unavailable"
            return

        if not self._ensure_weights():
            logger.warning(
                "ImageBind weights unavailable (cannot download from "
                "AkaneTendo25/ayase-runtime-assets HF mirror); module disabled"
            )
            self._ml_available = False
            self._backend = "unavailable"
            return

        try:
            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.device_config)

            model = imagebind_model.imagebind_huge(pretrained=True)
            model.eval()
            model.to(self._device)

            self._model = model
            self._modality_type = ModalityType
            self._data_module = imagebind_data
            self._ml_available = True
            self._backend = "imagebind"
            logger.info(
                "ImageBind initialised with %s on %s", self.model_name, self._device
            )
        except Exception as e:
            logger.warning("ImageBind setup failed: %s", e)
            self._ml_available = False
            self._backend = "unavailable"

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        caption = _caption_text(sample)
        video_path = sample.path if sample.is_video else None
        if not caption and video_path is None:
            return sample

        try:
            # Preserve the complete track. The official ImageBind loader then
            # selects its fixed number of clips across the full duration.
            audio = load_audio(sample.path, target_sr=self.sample_rate, duration=None)
            if audio is None or len(audio) == 0:
                return sample

            scores = self._score(audio, caption=caption, video_path=video_path)
            if not scores:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            for field, score in scores.items():
                setattr(sample.quality_metrics, field, float(score))
                sample.quality_metrics.metric_backends[field] = str(self._backend)

            text_score = scores.get("imagebind_score")
            if text_score is not None and text_score < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low ImageBind audio-text alignment: {text_score:.3f}",
                        details={
                            "imagebind_score": float(text_score),
                            "caption": caption[:80] if caption else "",
                            "threshold": float(self.warning_threshold),
                        },
                    )
                )
        except Exception as e:
            logger.warning("ImageBind scoring failed for %s: %s", sample.path, e)
        return sample

    # ------------------------------------------------------------------
    def _score(
        self,
        audio,
        caption: Optional[str] = None,
        video_path: Optional[Path] = None,
    ) -> dict[str, float]:
        """Compute requested ImageBind similarities in one model call.

        Audio uses the same three 2-second clip sampler, mel transform and
        normalization as ImageBind's official file loader, applied directly to
        Ayase's already decoded 16-kHz mono PCM. This avoids Torchaudio's
        version-dependent optional TorchCodec decoder without changing the
        embedding input. Video applies the released ImageBind transforms
        directly, omitting only an obsolete ``EncodedVideo`` keyword rejected
        by current PyTorchVideo, and matches JavisBench preprocessing.
        """
        try:
            import numpy as np
            import torch
        except ImportError as e:
            logger.debug("ImageBind scoring missing deps: %s", e)
            return {}

        try:
            audio_arr = np.asarray(audio, dtype=np.float32)
            if audio_arr.ndim > 1:
                audio_arr = audio_arr.mean(axis=1).astype(np.float32)

            ModalityType = self._modality_type
            data = self._data_module

            inputs = {
                ModalityType.AUDIO: self._transform_audio_array(audio_arr)
            }
            if caption:
                inputs[ModalityType.TEXT] = data.load_and_transform_text(
                    [caption], self._device
                )
            if video_path is not None:
                inputs[ModalityType.VISION] = self._transform_video_path(video_path)

            with torch.no_grad():
                embeddings = self._model(inputs)
                audio_emb = embeddings[ModalityType.AUDIO]
                cosine = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
                scores: dict[str, float] = {}

                if caption:
                    text_emb = embeddings[ModalityType.TEXT]
                    text_cosine = cosine(audio_emb, text_emb).item()
                    # Historical Ayase contract for imagebind_score.
                    scores["imagebind_score"] = float((text_cosine + 1.0) / 2.0)

                if video_path is not None:
                    video_emb = embeddings[ModalityType.VISION]
                    # JavisBench sim_av is the unscaled cosine.
                    scores["imagebind_av_score"] = float(
                        cosine(audio_emb, video_emb).item()
                    )

            return scores
        except Exception as e:
            logger.debug("ImageBind scoring failed: %s", e)
            return {}

    def _transform_audio_array(self, audio) -> object:
        """Apply ImageBind's released audio preprocessing to decoded mono PCM."""
        import torch

        data = self._data_module
        waveform = torch.as_tensor(audio, dtype=torch.float32).reshape(1, -1)
        sampler = data.ConstantClipsPerVideoSampler(
            clip_duration=2, clips_per_video=3
        )
        points = data.get_clip_timepoints(
            sampler, waveform.size(1) / float(self.sample_rate)
        )
        normalize = data.transforms.Normalize(mean=-4.268, std=9.138)
        clips = []
        for start, end in points:
            clip = waveform[
                :, int(start * self.sample_rate) : int(end * self.sample_rate)
            ]
            mel = data.waveform2melspec(
                clip,
                self.sample_rate,
                num_mel_bins=128,
                target_length=204,
            )
            clips.append(normalize(mel).to(self._device))
        return torch.stack([torch.stack(clips, dim=0)], dim=0)

    def _transform_video_path(self, video_path: Path) -> object:
        """Apply ImageBind's released five-clip/three-crop video preprocessing."""
        import torch

        data = self._data_module
        video = data.EncodedVideo.from_path(
            str(video_path), decoder="decord", decode_audio=False
        )
        sampler = data.ConstantClipsPerVideoSampler(
            clip_duration=2, clips_per_video=5
        )
        points = data.get_clip_timepoints(sampler, video.duration)
        frame_sampler = data.pv_transforms.UniformTemporalSubsample(num_samples=2)
        transform = data.transforms.Compose(
            [
                data.pv_transforms.ShortSideScale(224),
                data.NormalizeVideo(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        clips = []
        for start, end in points:
            decoded = video.get_clip(start, end)
            if decoded is None:
                raise ValueError("ImageBind found no decodable video clip")
            clips.append(transform(frame_sampler(decoded["video"]) / 255.0))
        crops = data.SpatialCrop(224, num_crops=3)(clips)
        return torch.stack([torch.stack(crops, dim=0)], dim=0).to(self._device)


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
