"""Offline regression coverage for source-derived metric and checkpoint metadata."""

import inspect

import pytest

from ayase.models import Sample
from ayase.pipeline import PipelineModule


class _DescribedModule(PipelineModule):
    metric_info = {
        "vmaf": "Module-specific score interpretation (higher=better)",
        "avgen_bench_score": "Coverage-aware dataset aggregate (higher=better)",
    }

    def process(self, sample: Sample) -> Sample:
        sample.quality_metrics.vmaf = 42.0
        return sample

    def post_process(self, all_samples):
        self.pipeline.add_dataset_metric("avgen_bench_score", 42.0)


def test_explicit_metric_descriptions_override_schema_comments():
    metadata = _DescribedModule.get_metadata()
    assert metadata["output_fields"]["vmaf"] == _DescribedModule.metric_info["vmaf"]
    assert metadata["dataset_output_fields"]["avgen_bench_score"] == (
        _DescribedModule.metric_info["avgen_bench_score"]
    )


@pytest.mark.parametrize(
    ("module", "expected"),
    [
        ("brisque_inverted", ["brisque"]),
        ("clip_iqa", ["clipiqa+"]),
        ("face_iqa", ["topiq_nr-face"]),
        ("laion_aesthetic", ["laion_aes"]),
        ("mouth_quality", ["musiq-spaq"]),
        ("topiq", ["topiq_nr"]),
        ("wadiqam", ["wadiqam_nr"]),
    ],
)
def test_pyiqa_uses_backend_metric_instead_of_registry_name(module, expected):
    from ayase.models_doc import _extract_pyiqa_metrics

    imported = __import__(f"ayase.modules.{module}", fromlist=[module])
    assert _extract_pyiqa_metrics(inspect.getsource(imported)) == expected


def test_pyiqa_ignores_prose_and_unrelated_variable_names():
    from ayase.models_doc import _extract_pyiqa_metrics

    source = '''
"""Example: create_metric("not_executed")"""
name = "registry_name"
metric_name = "unused"
# create_metric("comment_only")
backend = pyiqa.create_metric("musiq-spaq")
'''
    assert _extract_pyiqa_metrics(source) == ["musiq-spaq"]


def test_pyiqa_resolves_used_keyword_and_config_defaults():
    from ayase.models_doc import _extract_pyiqa_metrics

    source = """
self.variant = self.config.get("variant", "topiq_nr")
self.backend = pyiqa.create_metric(metric_name=self.variant)
"""
    assert _extract_pyiqa_metrics(source) == ["topiq_nr"]


def test_checkpoint_identity_preserves_files_and_revisions():
    from ayase.models_doc import _checkpoint_key, _hf_checkpoint_reference

    def key(url):
        return _checkpoint_key(_hf_checkpoint_reference(url))

    root = "https://huggingface.co/owner/repo"
    assert key(root + "/resolve/main/one.bin") == key(root + "/blob/main/one.bin")
    assert key(root + "/resolve/main/one.bin") != key(root + "/resolve/main/two.bin")
    assert key(root + "/resolve/main/one.bin") != key(root + "/resolve/pinned/one.bin")
    assert _hf_checkpoint_reference(root) is None
    assert _hf_checkpoint_reference("https://example.com/owner/repo/resolve/main/a.bin") is None


def test_song_eval_checkpoint_is_one_asset_with_declared_purpose():
    from ayase.metric_catalog import build_metric_catalog

    item = build_metric_catalog(["song_eval"]).module("song_eval")
    checkpoints = [model for model in item.models if "model.safetensors" in model.name]
    assert len(checkpoints) == 1
    assert checkpoints[0].task == "SongEval Generator aesthetic head weights"
    assert checkpoints[0].url.endswith("/song_eval/model.safetensors")


def test_generated_checkpoint_cards_keep_real_file_tasks_and_sizes(monkeypatch):
    from ayase import models_doc
    from ayase.modules.csim import CSIMModule
    from ayase.modules.facesim import FaceSimModule

    modules = {"csim": CSIMModule, "facesim": FaceSimModule}
    monkeypatch.setattr(models_doc.ModuleRegistry, "discover_modules", lambda: None)
    monkeypatch.setattr(models_doc.ModuleRegistry, "list_modules", lambda **kw: list(modules))
    monkeypatch.setattr(models_doc.ModuleRegistry, "get_module", modules.get)
    monkeypatch.setattr(
        models_doc, "compute_external_backend_partition", lambda mods: (set(), set())
    )
    monkeypatch.setattr(models_doc, "_generate_charts", lambda *args: {})
    output = models_doc.generate_models_doc(fetch_licenses=False)
    root = "https://huggingface.co/BestWishYsh/OpenS2V-Weight/resolve/main/"
    paths = {
        "face_extractor/models/buffalo_l/det_10g.onnx": "17 MB",
        "face_extractor/models/buffalo_l/w600k_r50.onnx": "174 MB",
        "glint360k_curricular_face_r101_backbone.bin": "261 MB",
    }
    for path, size in paths.items():
        assert output.count(f"]({root}{path})") == 1
        line = next(line for line in output.splitlines() if f"]({root}{path})" in line)
        assert size in line
        assert "face" in line.lower()
    assert "non-commercial research use only" in output
    detector = next(
        line
        for line in output.splitlines()
        if "[`face_extractor/models/buffalo_l/det_10g.onnx`]" in line
    )
    assert "License: non-commercial-research" in detector


def test_checkpoint_only_repos_fetch_license_without_using_repo_parameter_size(monkeypatch):
    from ayase import models_doc

    class CheckpointModule(_DescribedModule):
        name = "catalog_checkpoint_fixture"
        models = [
            {
                "id": "owner/checkpoints",
                "type": "huggingface",
                "url": "https://huggingface.co/owner/checkpoints/resolve/pinned/head.bin",
                "size": "17 MB",
                "vram": "25 MB",
                "arxiv": "2501.12345",
            },
            {
                "id": "owner/checkpoints",
                "type": "huggingface",
                "url": "https://huggingface.co/owner/checkpoints/resolve/pinned/restricted.bin",
                "license": "cc-by-nc-4.0",
            },
        ]

    queries = []

    def metadata(repos):
        queries.extend(repos)
        return {"owner/checkpoints": {"license": "apache-2.0", "parameters": 10**12}}

    monkeypatch.setattr(models_doc.ModuleRegistry, "discover_modules", lambda: None)
    monkeypatch.setattr(models_doc.ModuleRegistry, "list_modules", lambda **kw: ["checkpoint"])
    monkeypatch.setattr(models_doc.ModuleRegistry, "get_module", lambda name: CheckpointModule)
    monkeypatch.setattr(
        models_doc, "compute_external_backend_partition", lambda mods: (set(), set())
    )
    monkeypatch.setattr(models_doc, "_generate_charts", lambda *args: {})
    monkeypatch.setattr(models_doc, "_fetch_hf_info_batch", metadata)
    monkeypatch.setattr(models_doc, "_validate_urls", lambda entries: [])
    output = models_doc.generate_models_doc(fetch_licenses=True)
    assert queries == ["owner/checkpoints"]
    head = next(
        line for line in output.splitlines() if "[head.bin]" in line or "[`head.bin`]" in line
    )
    assert "License: apache-2.0" in head
    assert "17 MB" in head and "VRAM: 25 MB" in head
    assert "[Paper](https://arxiv.org/abs/2501.12345)" in head
    restricted = next(line for line in output.splitlines() if "[`restricted.bin`]" in line)
    assert "License: cc-by-nc-4.0" in restricted
    assert "GB" not in head
