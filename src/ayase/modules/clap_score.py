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

TODO (MS-CLAP): Microsoft's official CLAP weights are distributed via the
``msclap`` Python package and the ``microsoft/CLAP`` GitHub release artifacts
rather than as a standard ``transformers``-compatible HF checkpoint. There is
no canonical ``microsoft/msclap`` HF checkpoint loadable with
``ClapModel.from_pretrained`` at the time of writing. We keep
``microsoft/msclap`` as the placeholder identifier per spec; users who want a
working Microsoft-style backbone should either install ``msclap`` and adapt
the loader, or fall back to a community HF mirror (e.g. a
``transformers``-compatible re-upload). Until then, instantiating
``MSCLAPScoreModule`` with the default ``model_name`` will fail at
``setup()`` and the module will gracefully no-op (see ``HumanCLAPModule.setup``).
"""

import logging

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

    TODO: ``microsoft/msclap`` is not a standard ``transformers``-compatible HF
    checkpoint. Microsoft's CLAP is distributed via the ``msclap`` PyPI package
    and ``microsoft/CLAP`` GitHub release artifacts. The identifier here is a
    placeholder; if a canonical HF mirror appears (e.g. a community
    ``transformers``-compatible re-upload) override ``model_name`` via config.
    Until then, ``setup()`` will fail to load and the module will no-op safely.
    """

    name = "ms_clap_score"
    description = "Microsoft CLAP audio-text alignment cosine similarity"
    default_config = {
        **HumanCLAPModule.default_config,
        "model_name": "microsoft/msclap",
    }
    models = [
        {
            "id": "microsoft/msclap",
            "type": "huggingface",
            "task": "Microsoft CLAP audio-text encoder (placeholder HF id)",
        },
    ]
    metric_info = {
        "ms_clap_score": "Microsoft CLAP audio-text alignment (0-1, higher=better)",
    }
    metric_field_name = "ms_clap_score"


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
