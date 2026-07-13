"""Tests for the official VBench 2.0 dataset-level adapter."""

import json
import sys
from types import ModuleType, SimpleNamespace
from pathlib import Path

import pytest

from ..conftest import _test_module_basics


def test_vbench2_official_basics():
    from ayase.modules.vbench2_official import OfficialVBench2Module

    _test_module_basics(OfficialVBench2Module, "vbench2_official")


def test_vbench2_official_process_is_noop(video_sample):
    from ayase.modules.vbench2_official import OfficialVBench2Module

    assert OfficialVBench2Module().process(video_sample) is video_sample


def test_vbench2_official_uses_autodownload_cache_defaults():
    from ayase.modules.vbench2_official import OfficialVBench2Module

    assert OfficialVBench2Module.default_config["model_repo"] == "Vchitect/VBench-2.0_models"
    assert OfficialVBench2Module.default_config["models_dir"] == "models"
    assert OfficialVBench2Module.default_config["mirror_revision"] == "main"
    assert "load_ckpt_from_local" not in OfficialVBench2Module.default_config
    assert OfficialVBench2Module.models[0]["auto_download"] is True


def test_vbench2_setup_downloads_and_prewarms(monkeypatch, tmp_path):
    import ayase.config

    from ayase.modules.vbench2_official import OfficialVBench2Module

    captured = {}
    fake_utils = SimpleNamespace(
        CACHE_DIR=None,
        init_submodules=lambda dimensions, **kwargs: captured.update(
            dimensions=dimensions, kwargs=kwargs
        ),
    )
    fake_vbench = ModuleType("vbench2")
    fake_vbench.__file__ = str(tmp_path / "vbench2" / "__init__.py")
    fake_vbench.VBench2 = object
    fake_vbench.utils = fake_utils
    monkeypatch.setitem(sys.modules, "vbench2", fake_vbench)
    def fake_snapshot(repo_id, models_dir, **kwargs):
        captured["repo_id"] = repo_id
        captured["snapshot"] = tmp_path
        return tmp_path

    monkeypatch.setattr(ayase.config, "download_hf_snapshot", fake_snapshot)
    source_root = tmp_path / "upstream"
    (source_root / "VBench-2.0").mkdir(parents=True)
    monkeypatch.setattr(
        OfficialVBench2Module,
        "_download_external_artifacts",
        lambda self, checkpoint_root, models_dir: source_root,
    )

    module = OfficialVBench2Module(
        {"models_dir": tmp_path, "dimensions": ["Human_Identity"]}
    )
    module.setup()

    assert captured["snapshot"] == tmp_path
    assert captured["repo_id"] == "Vchitect/VBench-2.0_models"
    assert captured["dimensions"] == ["Human_Identity"]
    assert captured["kwargs"]["local"] is True
    assert module._backend == "official_vbench2"


def test_vbench2_external_artifacts_use_akane_hf_mirror():
    from ayase.modules.vbench2_official import OfficialVBench2Module

    module = OfficialVBench2Module()
    assert module._mirror_url("vbench2/raft/models.zip").startswith(
        "https://huggingface.co/AkaneTendo25/ayase-models/resolve/main/"
    )
    assert "AkaneTendo25/ayase-models" in module._mirror_url(
        "vbench2/torchvision/vgg19-dcbb9e9d.pth"
    )


def test_vbench2_official_extracts_upstream_result_shape():
    from ayase.modules.vbench2_official import OfficialVBench2Module

    assert OfficialVBench2Module._extract_score([0.75, [{"detail": 1}]]) == 0.75
    assert OfficialVBench2Module._extract_score({"score": 0.25}) == 0.25
    assert OfficialVBench2Module._extract_score("invalid") is None


def test_vbench2_official_aggregates_match_upstream_formula():
    from ayase.modules.vbench2_official import DIMENSION_FIELDS, OfficialVBench2Module

    scores = {dimension: 1.0 for dimension in DIMENSION_FIELDS}
    aggregates = OfficialVBench2Module._aggregate_scores(scores)

    assert aggregates["vbench2_creativity_score"] == 1.0
    assert aggregates["vbench2_commonsense_score"] == 1.0
    assert aggregates["vbench2_controllability_score"] == 1.0
    assert aggregates["vbench2_human_fidelity_score"] == 1.0
    assert aggregates["vbench2_physics_score"] == 1.0
    assert aggregates["vbench2_total_score"] == 1.0


def test_vbench2_official_runs_configured_backend(tmp_path: Path):
    from ayase.models import Sample
    from ayase.modules.vbench2_official import DIMENSION_FIELDS, OfficialVBench2Module

    output_dir = tmp_path / "results"
    full_info_path = tmp_path / "VBench2_full_info.json"
    full_info_path.write_text("[]", encoding="utf-8")
    video_path = tmp_path / "example.mp4"
    video_path.write_bytes(b"video")

    class FakeVBench2:
        def __init__(self, device, full_info_path, configured_output_dir):
            self.output_dir = Path(configured_output_dir)
            self.output_dir.mkdir(parents=True)

        def evaluate(self, videos_path, name, dimension_list, **kwargs):
            payload = {dimension: [0.5, []] for dimension in dimension_list}
            (self.output_dir / f"{name}_eval_results.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )

    recorded = {}

    class PipelineProbe:
        @staticmethod
        def add_dataset_metric(name, value):
            recorded[name] = value

    module = OfficialVBench2Module(
        config={
            "full_info_path": full_info_path,
            "output_dir": output_dir,
        }
    )
    module._backend = "official_vbench2"
    module._vbench_cls = FakeVBench2
    module.pipeline = PipelineProbe()

    module.post_process([Sample(path=video_path, is_video=True)])

    assert recorded["vbench2_human_anatomy"] == 0.5
    assert recorded["vbench2_total_score"] == pytest.approx(0.5)


def test_vbench2_official_metadata_exposes_dataset_fields():
    from ayase.modules.vbench2_official import OfficialVBench2Module

    metadata = OfficialVBench2Module.get_metadata()

    assert "vbench2_human_anatomy" in metadata["dataset_output_fields"]
    assert "vbench2_total_score" in metadata["dataset_output_fields"]
