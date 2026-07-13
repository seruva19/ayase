"""Contract tests for native WorldModelBench evaluation."""

import json
import sys
from types import SimpleNamespace

import pytest

from ..conftest import _test_module_basics


def test_worldmodelbench_basics():
    from ayase.modules.worldmodelbench import WorldModelBenchModule

    _test_module_basics(WorldModelBenchModule, "worldmodelbench")


def test_worldmodelbench_process_is_noop(video_sample):
    from ayase.modules.worldmodelbench import WorldModelBenchModule

    assert WorldModelBenchModule().process(video_sample) is video_sample


def test_worldmodelbench_native_autodownload_defaults():
    from ayase.modules.worldmodelbench import WorldModelBenchModule

    config = WorldModelBenchModule.default_config
    assert config["model_name"] == "Efficient-Large-Model/vila-ewm-qwen2-1.5b"
    assert "results_path" not in config
    assert "judge_path" not in config
    assert "evaluator_script" not in config
    assert config["benchmark_url"].startswith(
        "https://huggingface.co/AkaneTendo25/ayase-models/resolve/main/"
    )
    assert config["vila_source_url"].startswith(
        "https://huggingface.co/AkaneTendo25/ayase-models/resolve/main/"
    )
    assert config["s2wrapper_source_url"].startswith(
        "https://huggingface.co/AkaneTendo25/ayase-models/resolve/main/"
    )
    assert WorldModelBenchModule.models[0]["auto_download"] is True


def test_worldmodelbench_setup_downloads_and_loads_judge(monkeypatch, tmp_path):
    import ayase.config

    from ayase.modules.worldmodelbench import WorldModelBenchModule

    benchmark = tmp_path / "worldmodelbench.json"
    benchmark.write_text(json.dumps([{"first_frame": "images/a.jpg"}]), encoding="utf-8")
    captured = {}

    def fake_snapshot(repo_id, models_dir, **kwargs):
        captured["repo_id"] = repo_id
        return tmp_path / "judge"

    monkeypatch.setattr(ayase.config, "download_hf_snapshot", fake_snapshot)
    source = tmp_path / "vila"
    source.mkdir()

    def fake_file(relative_path, url, models_dir):
        return benchmark if relative_path.endswith("worldmodelbench.json") else tmp_path / "vila.zip"

    monkeypatch.setattr(ayase.config, "download_model_file", fake_file)
    monkeypatch.setattr(
        WorldModelBenchModule, "_extract_source", staticmethod(lambda *args: source)
    )
    monkeypatch.setattr(
        WorldModelBenchModule,
        "_configure_vila_attention",
        staticmethod(lambda implementation: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "llava",
        SimpleNamespace(
            load=lambda path, **kwargs: SimpleNamespace(generate_content=lambda value: "No")
        ),
    )

    module = WorldModelBenchModule({"models_dir": tmp_path})
    module.setup()

    assert captured["repo_id"] == "Efficient-Large-Model/vila-ewm-qwen2-1.5b"
    assert module._backend == "vila"


def test_worldmodelbench_parses_official_flat_accumulators():
    from ayase.modules.worldmodelbench import WorldModelBenchModule

    metrics = WorldModelBenchModule._parse_metrics(
        {
            "accs": {
                "instruction": [3, 1],
                "physical_laws": [True, True, False, True, False, False, True, True, True, True],
                "common_sense": [True, False, True, True],
            }
        }
    )
    assert metrics["worldmodelbench_instruction_score"] == 2.0
    assert metrics["worldmodelbench_newton_adherence"] == 0.5
    assert metrics["worldmodelbench_physical_score"] == 3.5
    assert metrics["worldmodelbench_total_score"] == 7.0


def test_worldmodelbench_rejects_partial_interleaved_groups():
    from ayase.modules.worldmodelbench import WorldModelBenchModule

    metrics = WorldModelBenchModule._parse_metrics(
        {"accs": {"instruction": [3], "physical_laws": [True] * 4}}
    )
    assert metrics == {"worldmodelbench_instruction_score": 3.0}


def test_worldmodelbench_runs_native_dataset_evaluation(video_sample):
    from ayase.modules.worldmodelbench import WorldModelBenchModule

    recorded = {}

    class PipelineProbe:
        @staticmethod
        def add_dataset_metric(name, value):
            recorded[name] = value

    module = WorldModelBenchModule()
    module._backend = "vila"
    module._benchmark = [
        {"first_frame": f"images/{video_sample.path.stem}.jpg", "text_instruction": "move"}
    ]
    module._evaluate = lambda path, instruction: {
        "instruction": [3.0],
        "physical_laws": [True] * 5,
        "common_sense": [True] * 2,
    }
    module.pipeline = PipelineProbe()
    module.post_process([video_sample])

    assert recorded["worldmodelbench_total_score"] == pytest.approx(10.0)


def test_worldmodelbench_metadata_exposes_dataset_fields():
    from ayase.modules.worldmodelbench import WorldModelBenchModule

    metadata = WorldModelBenchModule.get_metadata()
    assert "worldmodelbench_instruction_score" in metadata["dataset_output_fields"]
    assert "worldmodelbench_total_score" in metadata["dataset_output_fields"]
