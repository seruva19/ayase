"""Tests for the opens2v module (OpenS2V-Eval NexusScore + NaturalScore).

No model downloads: these tests exercise module basics, the strict
graceful-unavailable path, the reference/subject-phrase resolution precedence,
and the NexusScore crop-similarity aggregation math (ported from
OpenS2V-Nexus eval/get_nexusscore.py) with mocked detections/embeddings.
"""

import numpy as np
import pytest

from ..conftest import _test_module_basics
from ayase.models import CaptionMetadata, QualityMetrics, Sample


# --------------------------------------------------------------------------- #
#  Basics / registration                                                      #
# --------------------------------------------------------------------------- #

def test_opens2v_basics():
    from ayase.modules.opens2v import OpenS2VModule

    _test_module_basics(OpenS2VModule, "opens2v")


def test_opens2v_default_config():
    from ayase.modules.opens2v import OpenS2VModule

    cfg = OpenS2VModule.default_config
    # Required config surface per the module contract.
    for key in ("device", "max_frames", "box_threshold", "encoder"):
        assert key in cfg
    assert cfg["encoder"] in ("clip", "dino")
    assert "grounding-dino" in cfg["detector_model"]


def test_opens2v_metric_groups_and_fields():
    from ayase.modules.opens2v import OpenS2VModule

    groups = OpenS2VModule.metric_groups
    assert set(groups.keys()) == {"opens2v_nexus_score", "opens2v_natural_score"}

    qm = QualityMetrics()
    assert hasattr(qm, "opens2v_nexus_score")
    assert hasattr(qm, "opens2v_natural_score")
    assert qm.opens2v_nexus_score is None
    assert qm.opens2v_natural_score is None

    # metric_info/metric_groups keys must be real fields.
    valid = set(QualityMetrics.model_fields.keys())
    assert set(groups.keys()) <= valid
    assert set(OpenS2VModule.metric_info.keys()) <= valid


def test_opens2v_backend_default_unavailable():
    """Before setup, the module advertises no backend."""
    from ayase.modules.opens2v import OpenS2VModule

    m = OpenS2VModule()
    assert m._backend == "unavailable"
    assert m._nexus_available is False
    assert m._natural_available is False


# --------------------------------------------------------------------------- #
#  Graceful-unavailable (strict real-or-none, no deps => both fields None)     #
# --------------------------------------------------------------------------- #

def test_opens2v_graceful_unavailable_image(image_sample):
    from ayase.modules.opens2v import OpenS2VModule

    # test_mode skips setup entirely: no detector, no encoder, no VLM.
    m = OpenS2VModule({"test_mode": True})
    m.on_mount()
    assert m._backend == "unavailable"

    image_sample.quality_metrics = QualityMetrics()
    result = m.process(image_sample)
    assert result is image_sample
    assert result.quality_metrics.opens2v_nexus_score is None
    assert result.quality_metrics.opens2v_natural_score is None


def test_opens2v_graceful_unavailable_video(video_sample):
    from ayase.modules.opens2v import OpenS2VModule

    m = OpenS2VModule({"test_mode": True})
    m.on_mount()
    assert m._backend == "unavailable"

    video_sample.quality_metrics = QualityMetrics()
    result = m.process(video_sample)
    assert result is video_sample
    assert result.quality_metrics.opens2v_nexus_score is None
    assert result.quality_metrics.opens2v_natural_score is None


def test_opens2v_process_no_write_when_unavailable(video_sample):
    """When nothing loaded, process() must not even allocate quality_metrics."""
    from ayase.modules.opens2v import OpenS2VModule

    m = OpenS2VModule({"test_mode": True})
    m.on_mount()
    assert video_sample.quality_metrics is None
    result = m.process(video_sample)
    assert result.quality_metrics is None


# --------------------------------------------------------------------------- #
#  NexusScore aggregation math (filtering + /frame_obj normalization)          #
# --------------------------------------------------------------------------- #

def test_nexus_aggregation_repo_rules():
    """Hand-computable case using the repo's exact gates (bbox>0.6, text>0.30, img!=0).

    detections = (bbox_conf, text_sim, image_sim), frame_obj = 3:
        (0.90, 0.50, 0.80) -> keep 0.80
        (0.40, 0.90, 0.70) -> reject (bbox 0.40 <= 0.60)
        (0.70, 0.20, 0.60) -> reject (text 0.20 <= 0.30)
        (0.95, 0.60, 0.90) -> keep 0.90
        (0.80, 0.40, 0.00) -> reject (image_sim == 0)
    kept = [0.80, 0.90] -> mean 0.85 -> / frame_obj(3) = 0.283333...
    """
    from ayase.modules.opens2v import OpenS2VModule

    detections = [
        (0.90, 0.50, 0.80),
        (0.40, 0.90, 0.70),
        (0.70, 0.20, 0.60),
        (0.95, 0.60, 0.90),
        (0.80, 0.40, 0.00),
    ]
    result = OpenS2VModule._aggregate_nexus(
        detections, frame_obj=3, box_conf_threshold=0.6, text_sim_threshold=0.30
    )
    expected = ((0.80 + 0.90) / 2.0) / 3.0
    assert result == pytest.approx(expected, abs=1e-9)


def test_nexus_aggregation_multisubject_text_gate():
    """Multi-subject robustness: the off-phrase (wrong) subject is filtered by the
    text-relevance gate even though it was detected confidently.

    frame_obj = 1:
        correct subject : (0.90, 0.60, 0.85) -> keep 0.85
        distractor      : (0.90, 0.10, 0.40) -> reject (text 0.10 <= 0.30)
    kept = [0.85] -> mean 0.85 -> / 1 = 0.85 (distractor's 0.40 does not drag it down)
    """
    from ayase.modules.opens2v import OpenS2VModule

    detections = [
        (0.90, 0.60, 0.85),
        (0.90, 0.10, 0.40),
    ]
    result = OpenS2VModule._aggregate_nexus(
        detections, frame_obj=1, box_conf_threshold=0.6, text_sim_threshold=0.30
    )
    assert result == pytest.approx(0.85, abs=1e-9)


def test_nexus_aggregation_frame_normalization():
    """Same kept detections, larger frame_obj -> smaller score (length normalization)."""
    from ayase.modules.opens2v import OpenS2VModule

    dets = [(0.9, 0.9, 0.8), (0.9, 0.9, 0.8)]
    s2 = OpenS2VModule._aggregate_nexus(dets, frame_obj=2, box_conf_threshold=0.6, text_sim_threshold=0.3)
    s4 = OpenS2VModule._aggregate_nexus(dets, frame_obj=4, box_conf_threshold=0.6, text_sim_threshold=0.3)
    assert s2 == pytest.approx(0.8 / 2.0, abs=1e-9)
    assert s4 == pytest.approx(0.8 / 4.0, abs=1e-9)


def test_nexus_aggregation_no_valid_detections():
    """Empty input and all-rejected input both yield 0.0 (repo behavior)."""
    from ayase.modules.opens2v import OpenS2VModule

    assert OpenS2VModule._aggregate_nexus([], frame_obj=0) == 0.0
    # All rejected by the confidence gate.
    rejected = [(0.1, 0.9, 0.9), (0.2, 0.9, 0.9)]
    assert (
        OpenS2VModule._aggregate_nexus(
            rejected, frame_obj=2, box_conf_threshold=0.6, text_sim_threshold=0.3
        )
        == 0.0
    )


def test_nexus_aggregation_default_gates():
    """Default gates (keep_box_conf=0.30, keep_text_sim=0.20) applied when omitted."""
    from ayase.modules.opens2v import OpenS2VModule

    # bbox 0.35 > 0.30 and text 0.25 > 0.20 -> kept; the 0.10/0.05 one rejected.
    dets = [(0.35, 0.25, 0.7), (0.10, 0.9, 0.9)]
    result = OpenS2VModule._aggregate_nexus(dets, frame_obj=1)
    assert result == pytest.approx(0.7, abs=1e-9)


# --------------------------------------------------------------------------- #
#  Reference / subject-phrase resolution precedence                           #
# --------------------------------------------------------------------------- #

def test_resolve_phrase_precedence(image_sample):
    from ayase.modules.opens2v import OpenS2VModule

    # 1. explicit config subject_prompt wins.
    caption_text = "a fluffy cat on a sofa"
    image_sample.caption = CaptionMetadata(text=caption_text, length=len(caption_text))
    m = OpenS2VModule({"subject_prompt": "red sports car"})
    assert m._resolve_phrase(image_sample) == "red sports car"

    # 2. else fall back to caption text.
    m2 = OpenS2VModule()
    assert m2._resolve_phrase(image_sample) == caption_text

    # 3. else None.
    bare = Sample(path=image_sample.path, is_video=False)
    assert m2._resolve_phrase(bare) is None


def test_resolve_reference_precedence(image_sample, tmp_path):
    from ayase.modules.opens2v import OpenS2VModule

    # 1. sample.reference_path (existing) wins.
    image_sample.reference_path = image_sample.path
    m = OpenS2VModule()
    assert m._resolve_reference(image_sample) == image_sample.path

    # 2. else config reference_image (existing).
    bare = Sample(path=image_sample.path, is_video=False)
    m2 = OpenS2VModule({"reference_image": str(image_sample.path)})
    assert m2._resolve_reference(bare) == image_sample.path

    # 3. else None (non-existent path is ignored).
    m3 = OpenS2VModule({"reference_image": str(tmp_path / "missing.png")})
    assert m3._resolve_reference(bare) is None
