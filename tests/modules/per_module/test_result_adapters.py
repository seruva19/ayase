"""Contract tests for upstream 2025-2026 benchmark result adapters."""

import csv
import json

import pytest

from ayase.models import Sample
from tests.modules.conftest import _test_module_basics


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.mark.parametrize(
    ("class_name", "module_name"),
    [
        ("LOVEResultModule", "love_results"),
        ("Ref4DResultModule", "ref4d_results"),
        ("PhyGroundResultModule", "phyground_results"),
    ],
)
def test_reference_result_adapter_basics(class_name, module_name):
    module_path = {
        "LOVEResultModule": "ayase.modules.love",
        "Ref4DResultModule": "ayase.modules.ref4d",
        "PhyGroundResultModule": "ayase.modules.phyground",
    }[class_name]
    module = __import__(module_path, fromlist=[class_name])
    _test_module_basics(getattr(module, class_name), module_name)


def test_love_imports_reference_csv_predictions(tmp_path):
    perception = _write_csv(
        tmp_path / "perception.csv",
        [{"video_name": "clip.mp4", "pred_score": "4.25"}],
    )
    correspondence = _write_csv(
        tmp_path / "correspondence.csv",
        [{"video_name": "clip.mp4", "pred_score": "3.75"}],
    )
    from ayase.modules.love import LOVEResultModule

    sample = Sample(path=tmp_path / "clip.mp4", is_video=True)
    result = LOVEResultModule(
        {
            "perception_results_path": perception,
            "correspondence_results_path": correspondence,
        }
    ).process(sample)

    assert result is sample
    assert result.quality_metrics.love_perception_score == pytest.approx(4.25)
    assert result.quality_metrics.love_correspondence_score == pytest.approx(3.75)


def test_ref4d_imports_four_dimension_summaries(tmp_path):
    paths = {}
    for dimension, score in (
        ("semantic", 80.0),
        ("event", 70.0),
        ("motion", 60.0),
        ("world", 50.0),
    ):
        paths[f"{dimension}_results_path"] = _write_csv(
            tmp_path / f"{dimension}.csv",
            [{"sample_id": "sample-1", f"{dimension}_score_0_100": score}],
        )
    from ayase.modules.ref4d import Ref4DResultModule

    sample = Sample(path=tmp_path / "sample-1.mp4", is_video=True)
    result = Ref4DResultModule(paths).process(sample)
    qm = result.quality_metrics

    assert result is sample
    assert qm.ref4d_semantic_score == pytest.approx(80.0)
    assert qm.ref4d_event_score == pytest.approx(70.0)
    assert qm.ref4d_motion_score == pytest.approx(60.0)
    assert qm.ref4d_world_score == pytest.approx(50.0)
    assert qm.ref4d_overall_score == pytest.approx(65.0)


def test_phyground_imports_structured_scores(tmp_path):
    results = tmp_path / "scores.json"
    results.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "video": "falling_ball",
                        "SA": 4,
                        "PTV": 5,
                        "persistence": 3,
                        "general_avg": 4,
                        "physical": {"avg": 4.5, "coverage": 0.75},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    from ayase.modules.phyground import PhyGroundResultModule

    sample = Sample(path=tmp_path / "falling_ball.mp4", is_video=True)
    result = PhyGroundResultModule({"results_path": results}).process(sample)
    qm = result.quality_metrics

    assert result is sample
    assert qm.phyground_spatial_alignment_score == pytest.approx(4.0)
    assert qm.phyground_prompt_temporal_validity_score == pytest.approx(5.0)
    assert qm.phyground_persistence_score == pytest.approx(3.0)
    assert qm.phyground_general_score == pytest.approx(4.0)
    assert qm.phyground_physical_score == pytest.approx(4.5)
    assert qm.phyground_physical_coverage == pytest.approx(0.75)


def test_unmatched_reference_result_degrades_gracefully(tmp_path):
    results = _write_csv(
        tmp_path / "perception.csv",
        [{"video_name": "another.mp4", "pred_score": "4.0"}],
    )
    from ayase.modules.love import LOVEResultModule

    sample = Sample(path=tmp_path / "clip.mp4", is_video=True)
    result = LOVEResultModule({"perception_results_path": results}).process(sample)

    assert result is sample
    assert result.quality_metrics is None
