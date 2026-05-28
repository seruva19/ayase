"""Tests for qwen_image_bench module."""

from ..conftest import _test_module_basics
from ayase.models import CaptionMetadata, QualityMetrics


def test_qwen_image_bench_basics():
    from ayase.modules.qwen_image_bench import QwenImageBenchModule

    _test_module_basics(QwenImageBenchModule, "qwen_image_bench")


def test_qwen_image_bench_no_backend_is_noop(image_sample):
    from ayase.modules.qwen_image_bench import QwenImageBenchModule

    image_sample.caption = CaptionMetadata(text="a red apple on a table", length=22)
    mod = QwenImageBenchModule()
    out = mod.process(image_sample)

    assert out is image_sample
    if out.quality_metrics is not None:
        assert out.quality_metrics.qwen_image_bench_overall is None


def test_qwen_image_bench_video_is_noop(video_sample):
    from ayase.modules.qwen_image_bench import QwenImageBenchModule

    mod = QwenImageBenchModule()
    out = mod.process(video_sample)

    assert out is video_sample


def test_qwen_image_bench_extracts_json_after_thinking():
    from ayase.modules.qwen_image_bench import _extract_json_from_response

    parsed = _extract_json_from_response(
        '<think>reasoning</think> {"Realism": {"Physical Logic": {"score": 1}}}'
    )

    assert parsed == {"Realism": {"Physical Logic": {"score": 1}}}


def test_qwen_image_bench_score_mapping_and_aggregation():
    from ayase.modules.qwen_image_bench import (
        _compute_dimension_score,
        _fix_score_json,
    )

    fixed = _fix_score_json(
        {
            "Physical Logic": {"score": 1},
            "Material Texture": {"score": 2},
            "Resolution": {"score": "N/A"},
        },
        "Quality",
    )
    result = _compute_dimension_score(fixed)

    assert result["level2_scores"]["Realism"] == 80.0
    assert result["level2_scores"]["Resolution"] is None
    assert result["level1_score"] == 80.0


def test_qwen_image_bench_store_scores(image_sample):
    from ayase.modules.qwen_image_bench import QwenImageBenchModule

    image_sample.quality_metrics = QualityMetrics()
    mod = QwenImageBenchModule()
    mod._store_scores(
        image_sample,
        {
            "Quality": {"level1_score": 60.0},
            "Aesthetics": {"level1_score": 80.0},
            "Alignment": {"level1_score": 100.0},
        },
    )

    metrics = image_sample.quality_metrics
    assert metrics.qwen_image_bench_quality == 60.0
    assert metrics.qwen_image_bench_aesthetics == 80.0
    assert metrics.qwen_image_bench_alignment == 100.0
    assert metrics.qwen_image_bench_overall == 80.0
