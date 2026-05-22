"""CLAP-Score audio-text alignment modules.

This module exposes three thin subclasses of ``HumanCLAPModule`` that share the
same model-agnostic CLAP wrapper but differ only in:

* the registered ``name`` (so each is selectable as a separate pipeline stage),
* the HF checkpoint used as ``default_config["model_name"]``,
* the ``metric_info`` advertised to leaderboards, and
* the ``quality_metrics`` field they write into (``metric_field_name``).

CLAP-Score is the cosine similarity between an audio embedding and a text
embedding in a Contrastive Language-Audio Pretraining space, rescaled to
``[0, 1]``. It is widely used as an audio-text alignment metric for v2a
(video-to-audio) generation.

``QualityMetrics`` declares ``laion_clap_score``, ``ms_clap_score`` and
``clap_score`` as first-class fields. They are written through the inherited
``metric_field_name`` hook so all CLAP variants can share the same processing
implementation.

MS-CLAP backbone note: Microsoft's official ``microsoft/msclap`` HF repo ships
the original ``.pth`` checkpoints, not a ``transformers`` ``ClapModel`` layout.
``MSCLAPScoreModule`` therefore uses the official ``msclap`` Python wrapper,
which downloads ``CLAP_weights_2023.pth`` from the official HF repo.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

from ayase.modules.human_clap import HumanCLAPModule

logger = logging.getLogger(__name__)


class LAIONCLAPScoreModule(HumanCLAPModule):
    """LAION-CLAP audio-text alignment cosine similarity.

    Uses ``laion/clap-htsat-fused``, the most popular open-source CLAP
    checkpoint, as the audio-text encoder.
    """

    name = "laion_clap_score"
    description = "LAION-CLAP audio-text alignment cosine similarity"
    default_config = {
        **HumanCLAPModule.default_config,
        "model_name": "laion/clap-htsat-fused",
    }
    models = [
        {
            "id": "laion/clap-htsat-fused",
            "type": "huggingface",
            "task": "LAION-CLAP audio-text encoder",
        },
    ]
    metric_info = {
        "laion_clap_score": "LAION-CLAP audio-text alignment (0-1, higher=better)",
    }
    metric_field_name = "laion_clap_score"


class MSCLAPScoreModule(HumanCLAPModule):
    """Microsoft CLAP audio-text alignment cosine similarity.

    Uses the official ``msclap`` package and official ``microsoft/msclap``
    HuggingFace weights. The module still writes a 0-1 cosine score to keep the
    metric scale aligned with ``human_clap_score`` and ``laion_clap_score``.
    """

    name = "ms_clap_score"
    description = "Microsoft CLAP audio-text alignment cosine similarity"
    default_config = {
        **HumanCLAPModule.default_config,
        "version": "2023",
    }
    models = [
        {
            "id": "microsoft/msclap",
            "type": "huggingface",
            "task": "Official MS-CLAP audio-text encoder weights",
        },
    ]
    metric_info = {
        "ms_clap_score": "Microsoft CLAP audio-text alignment (0-1, higher=better)",
    }
    metric_field_name = "ms_clap_score"

    def setup(self) -> None:
        try:
            import torch
            from msclap import CLAP

            if self.device_config == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = self.device_config

            version = self.config.get("version", "2023")
            self._model = CLAP(version=version, use_cuda=self._device == "cuda")
            self._processor = None
            self._ml_available = True
            logger.info("MS-CLAP initialised with microsoft/msclap:%s on %s", version, self._device)
        except ImportError:
            logger.warning("MS-CLAP requires the optional `msclap` package")
        except Exception as e:
            logger.warning("MS-CLAP setup failed: %s", e)

    def _score(self, audio, caption: str) -> Optional[float]:
        """Score an in-memory waveform with the official MS-CLAP wrapper."""
        tmp_path: Optional[Path] = None
        try:
            import soundfile as sf
            import torch

            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = Path(tmp.name)
            tmp.close()
            sf.write(str(tmp_path), audio, self.sample_rate)

            with torch.no_grad():
                audio_embeds = self._model.get_audio_embeddings([str(tmp_path)])
                text_embeds = self._model.get_text_embeddings([caption])
                audio_embeds = audio_embeds / audio_embeds.norm(dim=-1, keepdim=True)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                sim = (audio_embeds * text_embeds).sum(dim=-1).item()
            return float(max(0.0, min(1.0, (sim + 1.0) / 2.0)))
        except Exception as e:
            logger.debug("MS-CLAP scoring failed: %s", e)
            return None
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)


class GenericCLAPScoreModule(HumanCLAPModule):
    """Generic CLAP audio-text alignment cosine similarity.

    A backbone-agnostic fallback that writes to the plain ``clap_score``
    column. The default backbone is LAION-CLAP, but users can swap it out
    through ``config["model_name"]`` without changing the leaderboard column.
    """

    name = "clap_score"
    description = "Generic CLAP audio-text alignment cosine similarity (configurable backbone)"
    default_config = {
        **HumanCLAPModule.default_config,
        "model_name": "laion/clap-htsat-fused",
    }
    models = [
        {
            "id": "laion/clap-htsat-fused",
            "type": "huggingface",
            "task": "Generic CLAP audio-text encoder (configurable)",
        },
    ]
    metric_info = {
        "clap_score": "CLAP audio-text alignment (0-1, configurable backbone)",
    }
    metric_field_name = "clap_score"
