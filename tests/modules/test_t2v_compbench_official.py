"""Focused tests for the official T2V-CompBench command adapter."""

import csv
import json
from pathlib import Path


def _metadata(root: Path, category: str):
    folder = root / "meta_data"
    folder.mkdir(exist_ok=True)
    records = [{"prompt": f"prompt {index}"} for index in range(1, 201)]
    (folder / f"{category}.json").write_text(json.dumps(records), encoding="utf-8")
    return records


class _Pipeline:
    def __init__(self):
        self.metrics = {}

    def add_dataset_metric(self, name, value):
        self.metrics[name] = value


def test_category_commands_use_real_scripts_and_no_shell(tmp_path):
    from ayase.modules.t2v_compbench import T2VCompBenchOfficialModule

    module = T2VCompBenchOfficialModule(
        {"upstream_root": str(tmp_path), "category": "motion_binding"}
    )
    commands = module._commands(tmp_path, "motion_binding", tmp_path / "run")
    assert commands[0][0][1].endswith("compbench_motion_binding_seg.py")
    assert commands[1][0][1].endswith("compbench_eval_motion_binding.py")
    assert "--mask_folder" in commands[1][0]
    assert commands[1][1] == tmp_path / "dot"


def test_parse_official_consistent_attribute_normalization(tmp_path):
    from ayase.modules.t2v_compbench import T2VCompBenchOfficialModule

    records = [{"prompt": "one"}, {"prompt": "two"}]
    output = tmp_path / "score.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "prompt", "Score"])
        writer.writerow(["0001_grid.jpg", "one", "15"])
        writer.writerow(["0002_grid.jpg", "two", "1"])
        writer.writerow(["score: ", "0.5"])
    scores, average = T2VCompBenchOfficialModule._parse_scores(
        output, "consistent_attribute_binding", records
    )
    assert scores == {1: 1.0, 2: 0.0}
    assert average == 0.5


def test_parse_rejects_prompt_id_mismatch(tmp_path):
    import pytest
    from ayase.modules.t2v_compbench import T2VCompBenchOfficialModule

    output = tmp_path / "score.csv"
    output.write_text("id,prompt,Score\n0001,wrong,1\nscore: ,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        T2VCompBenchOfficialModule._parse_scores(
            output, "dynamic_attribute_binding", [{"prompt": "right"}]
        )


def test_process_requires_benchmark_id_and_category(tmp_path):
    from ayase.models import Sample
    from ayase.modules.t2v_compbench import T2VCompBenchOfficialModule

    folder = tmp_path / "consistent_attr"
    folder.mkdir()
    valid = folder / "0007.mp4"
    invalid = folder / "clip.mp4"
    valid.touch()
    invalid.touch()
    module = T2VCompBenchOfficialModule()
    module.process(Sample(path=valid, is_video=True))
    module.process(Sample(path=invalid, is_video=True))
    assert list(module._samples["consistent_attribute_binding"]) == [7]


def test_nonprefix_subset_keeps_authoritative_ids_and_prompts(tmp_path):
    from ayase.models import CaptionMetadata, Sample
    from ayase.modules.t2v_compbench import T2VCompBenchOfficialModule

    root = tmp_path / "upstream"
    root.mkdir()
    records = _metadata(root, "generative_numeracy")
    module = T2VCompBenchOfficialModule(
        {
            "upstream_root": str(root),
            "category": "generative_numeracy",
        }
    )
    for identifier in (7, 199):
        path = tmp_path / f"{identifier:04d}.mp4"
        path.touch()
        module.process(
            Sample(
                path=path,
                is_video=True,
                caption=CaptionMetadata(text=records[identifier - 1]["prompt"], length=10),
            )
        )

    accepted = module._validated_samples(root, "generative_numeracy")
    assert [identifier for identifier, _sample in accepted] == [7, 199]
    staged = tmp_path / "stage"
    module._stage(accepted, staged)
    assert sorted(path.name for path in staged.iterdir()) == ["0007.mp4", "0199.mp4"]


def test_fresh_run_stages_only_validated_samples_and_parses_native_csv(tmp_path):
    from ayase.models import CaptionMetadata, Sample
    from ayase.modules.t2v_compbench import T2VCompBenchOfficialModule

    root = tmp_path / "upstream"
    root.mkdir()
    records = _metadata(root, "dynamic_attribute_binding")
    script = root / "LLaVA/llava/eval/compbench_eval_dynamic_attr.py"
    script.parent.mkdir(parents=True)
    script.touch()
    video = tmp_path / "0001.mp4"
    video.touch()
    module = T2VCompBenchOfficialModule(
        {
            "upstream_root": str(root),
            "category": "dynamic_attribute_binding",
            "temp_root": str(tmp_path / "runs"),
        }
    )
    module._available = True
    module._backend = "official_t2v_compbench"
    module.pipeline = _Pipeline()
    sample = Sample(
        path=video,
        is_video=True,
        caption=CaptionMetadata(text=records[0]["prompt"], length=8),
    )
    module.process(sample)

    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        output = Path(command[command.index("--output-path") + 1])
        output.mkdir(parents=True, exist_ok=True)
        with (output / "ayase_dynamic_attr_score.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["name", "prompt", "Score"])
            writer.writerow(["0001_frames.jpg", records[0]["prompt"], "0.75"])
            writer.writerow(["score: ", "0.75"])

    module._runner = runner
    module.post_process([])
    assert len(calls) == 1
    assert calls[0][1]["shell"] is False
    assert calls[0][1]["check"] is True
    assert "capture_output" not in calls[0][1]
    assert calls[0][1]["stderr"] is __import__("subprocess").STDOUT
    assert calls[0][1]["stdout"].name.endswith("evaluator-0.log")
    field = "t2v_compbench_dynamic_attribute_binding"
    assert getattr(sample.quality_metrics, field) == 0.75
    assert sample.quality_metrics.metric_provenance[field] == "adapted"
    assert sample.quality_metrics.metric_backends[module.name] == "official_t2v_compbench"
    assert module.pipeline.metrics[field + "_avg"] == 0.75
    assert not any((tmp_path / "runs").iterdir())


def test_malformed_output_is_rejected(tmp_path):
    import pytest
    from ayase.modules.t2v_compbench import T2VCompBenchOfficialModule

    output = tmp_path / "bad.csv"
    output.write_text("id,Score\n0001,0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed|missing"):
        T2VCompBenchOfficialModule._parse_scores(output, "generative_numeracy", [{"prompt": "x"}])


def test_duplicate_or_inconsistent_native_scores_are_rejected(tmp_path):
    import pytest
    from ayase.modules.t2v_compbench import T2VCompBenchOfficialModule

    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text(
        "id,Score\n0001,0.5\n0001,0.5\nscore: ,0.5\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        T2VCompBenchOfficialModule._parse_scores(
            duplicate, "generative_numeracy", [{"prompt": "x"}]
        )

    inconsistent = tmp_path / "inconsistent.csv"
    inconsistent.write_text(
        "id,Score\n0001,0.5\n0002,0.5\nscore: ,0.8\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match"):
        T2VCompBenchOfficialModule._parse_scores(
            inconsistent,
            "generative_numeracy",
            [{"prompt": "x"}, {"prompt": "y"}],
        )


def test_duplicate_input_id_rejects_category(tmp_path):
    from ayase.models import Sample
    from ayase.modules.t2v_compbench import T2VCompBenchOfficialModule

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = first_dir / "0007.mp4"
    second = second_dir / "0007.mp4"
    first.touch()
    second.touch()
    module = T2VCompBenchOfficialModule({"category": "generative_numeracy"})
    module.process(Sample(path=first, is_video=True))
    module.process(Sample(path=second, is_video=True))
    assert "generative_numeracy" in module._invalid_categories
    assert module._samples["generative_numeracy"] == {}


def test_incomplete_output_does_not_publish_sample_or_average(tmp_path):
    from ayase.models import CaptionMetadata, Sample
    from ayase.modules.t2v_compbench import T2VCompBenchOfficialModule

    root = tmp_path / "upstream"
    root.mkdir()
    records = _metadata(root, "dynamic_attribute_binding")
    video = tmp_path / "0001.mp4"
    video.touch()
    module = T2VCompBenchOfficialModule(
        {
            "upstream_root": str(root),
            "category": "dynamic_attribute_binding",
            "temp_root": str(tmp_path / "runs"),
        }
    )
    module._available = True
    module._backend = "official_t2v_compbench"
    module.pipeline = _Pipeline()
    sample = Sample(
        path=video,
        is_video=True,
        caption=CaptionMetadata(text=records[0]["prompt"], length=8),
    )
    module.process(sample)

    def runner(command, **kwargs):
        del kwargs
        output = Path(command[command.index("--output-path") + 1])
        output.mkdir(parents=True, exist_ok=True)
        with (output / "ayase_dynamic_attr_score.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["name", "prompt", "Score"])
            writer.writerow(["0002_frames.jpg", records[1]["prompt"], "0.75"])
            writer.writerow(["score: ", "0.75"])

    module._runner = runner
    module.post_process([])
    assert sample.quality_metrics is None
    assert module.pipeline.metrics == {}


def test_direct_setup_honors_test_mode_and_reports_missing_assets(tmp_path):
    import pytest
    from ayase.modules.t2v_compbench import T2VCompBenchOfficialModule

    test_module = T2VCompBenchOfficialModule({"test_mode": True})
    test_module.setup()
    assert test_module._available is False
    assert test_module._backend is None

    module = T2VCompBenchOfficialModule({"category": "motion_binding"})
    with pytest.raises(ValueError, match="required evaluator assets are missing"):
        module._check_fixed_assets(tmp_path)


def test_command_failure_reports_only_bounded_log_tail(tmp_path):
    import subprocess
    import pytest
    from ayase.modules.t2v_compbench import T2VCompBenchOfficialModule

    root = tmp_path / "upstream"
    root.mkdir()
    module = T2VCompBenchOfficialModule(
        {
            "upstream_root": str(root),
            "category": "dynamic_attribute_binding",
        }
    )

    def runner(command, **kwargs):
        kwargs["stdout"].write("x" * 3000 + "TAIL_MARKER")
        kwargs["stdout"].flush()
        raise subprocess.TimeoutExpired(command, 1)

    module._runner = runner
    with pytest.raises(RuntimeError) as caught:
        module._run_category(root, "dynamic_attribute_binding", tmp_path / "run")
    message = str(caught.value)
    assert "TAIL_MARKER" in message
    assert message.count("x") <= 2048


def test_setup_auto_provisions_source_assets_and_environments(tmp_path, monkeypatch):
    from ayase.modules import t2v_compbench
    from ayase.modules.t2v_compbench import T2VCompBenchOfficialModule

    monkeypatch.setenv("AYASE_TEST_MODE", "0")
    monkeypatch.setattr(T2VCompBenchOfficialModule, "_global_test_mode", False)

    root = tmp_path / "source"
    root.mkdir()
    records = _metadata(root, "dynamic_attribute_binding")
    script = root / "LLaVA/llava/eval/compbench_eval_dynamic_attr.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import argparse\np=argparse.ArgumentParser()\n"
        + "\n".join(
            f'p.add_argument("{name}")'
            for name in (
                "--video-path",
                "--read-prompt-file",
                "--model-path",
                "--output-path",
                "--t2v-model",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(t2v_compbench.managed, "gpu_is_available", lambda: True)
    monkeypatch.setattr(t2v_compbench.managed, "ensure_source", lambda _models: root)
    assets = []
    monkeypatch.setattr(
        t2v_compbench.managed,
        "ensure_assets",
        lambda source, categories: assets.append((source, categories)),
    )
    monkeypatch.setattr(
        t2v_compbench.managed,
        "ensure_environments",
        lambda *_args, **_kwargs: {"llava": tmp_path / "managed-python"},
    )
    monkeypatch.setattr(T2VCompBenchOfficialModule, "_verify_pinned_sources", lambda *_args: None)
    module = T2VCompBenchOfficialModule(
        {
            "category": "dynamic_attribute_binding",
            "models_dir": str(tmp_path / "models"),
            "check_fixed_assets": False,
        }
    )
    module.setup()
    assert module._available is True
    assert module._python("dynamic_attribute_binding") == str(tmp_path / "managed-python")
    assert assets == [(root, ("dynamic_attribute_binding",))]
    assert len(records) == 200


def test_setup_checks_gpu_before_any_download(tmp_path, monkeypatch):
    from ayase.modules import t2v_compbench
    from ayase.modules.t2v_compbench import T2VCompBenchOfficialModule

    monkeypatch.setattr(t2v_compbench.managed, "gpu_is_available", lambda: False)
    monkeypatch.setattr(
        t2v_compbench.managed,
        "ensure_source",
        lambda _models: (_ for _ in ()).throw(AssertionError("must not download")),
    )
    module = T2VCompBenchOfficialModule({"models_dir": str(tmp_path)})
    module.setup()
    assert module._available is False
