"""Tests for tc_bench module."""

import numpy as np

from ..conftest import _test_module_basics
from ayase.models import CaptionMetadata


def test_tc_bench_basics():
    from ayase.modules.tc_bench import TCBenchModule
    _test_module_basics(TCBenchModule, "tc_bench")


def test_tc_bench_regex_decompose():
    from ayase.modules.tc_bench import _regex_decompose
    events = _regex_decompose("First the dog runs, then the cat jumps, finally the bird flies")
    assert len(events) >= 2
    assert "the dog runs" in events[0] or events[0].startswith("the dog")


def test_tc_bench_comma_fallback():
    from ayase.modules.tc_bench import _regex_decompose
    # No connectives → empty result → caller falls back to comma split
    events = _regex_decompose("the cat sits on the mat")
    assert events == []


def test_tc_bench_image_is_noop(image_sample):
    from ayase.modules.tc_bench import TCBenchModule
    m = TCBenchModule()
    image_sample.caption = CaptionMetadata(text="first a, then b", length=15)
    out = m.process(image_sample)
    assert out is image_sample
    if out.quality_metrics is not None:
        assert out.quality_metrics.tcbench_overall is None


def test_tc_bench_single_event_caption_records_neutral(video_sample):
    from ayase.modules.tc_bench import TCBenchModule
    m = TCBenchModule()
    # Single event — temporally trivial → record 1.0 to avoid spurious penalty
    video_sample.caption = CaptionMetadata(text="a green ball bouncing", length=22)
    out = m.process(video_sample)
    assert out is video_sample
    if out.quality_metrics is not None:
        assert out.quality_metrics.tcbench_overall == 1.0


def test_event_fulfillment_counts_grounded_events():
    from ayase.modules.tc_bench import TCBenchModule

    module = TCBenchModule({"event_similarity_threshold": 0.2})
    similarities = np.array(
        [[0.10, 0.25, -0.1], [0.21, 0.15, 0.19], [0.18, 0.30, 0.05]],
        dtype=np.float32,
    )
    assert module._event_fulfillment(similarities) == 2 / 3


def test_custom_clip_does_not_inherit_default_revision():
    from ayase.modules.tc_bench import TCBenchModule

    assert TCBenchModule({"clip_model": "example/custom"}).clip_revision is None
