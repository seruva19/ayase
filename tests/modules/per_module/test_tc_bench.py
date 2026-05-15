"""Tests for tc_bench module."""

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
