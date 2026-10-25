import json
import sys
import types
import zipfile
from pathlib import Path

import pytest

from tests.modules.conftest import _test_module_basics


def _labels(objects=1):
    return {
        f"object_{index}": {"positive": [[100.0 + index, 200.0]], "negative": []}
        for index in range(1, objects + 1)
    }


def test_morpheus_basics():
    from ayase.modules.morpheus import MorpheusModule

    _test_module_basics(MorpheusModule, "morpheus")


def test_morpheus_test_mode_never_provisions(video_sample, monkeypatch):
    from ayase.modules.morpheus import MorpheusModule

    module = MorpheusModule({"test_mode": True})
    monkeypatch.setattr(module, "_prepare_backend", lambda: (_ for _ in ()).throw(AssertionError()))
    module.on_mount()
    assert module.process(video_sample) is video_sample
    assert video_sample.quality_metrics is None


def test_morpheus_setup_allows_metadata_only_annotations(monkeypatch):
    from ayase.modules.morpheus import MorpheusModule

    monkeypatch.delenv("AYASE_TEST_MODE", raising=False)
    monkeypatch.setattr(MorpheusModule, "_global_test_mode", False)
    module = MorpheusModule()
    called = []
    monkeypatch.setattr(module, "_prepare_backend", lambda: called.append(True))
    module.setup()
    assert called == [True]
    assert module._backend == "morpheus-native"


def test_morpheus_rejects_unknown_scenario_and_missing_labels():
    from ayase.modules.morpheus import MorpheusModule

    unknown = MorpheusModule({"experiment": "fluid", "preprocessing_model": "WAN-2.1", "object_labels": _labels()})
    missing = MorpheusModule({"experiment": "falling_ball", "preprocessing_model": "WAN-2.1"})
    collision = MorpheusModule({"experiment": "collision_equal", "preprocessing_model": "WAN-2.1", "object_labels": _labels()})
    assert not unknown._applicable_config()
    assert not missing._applicable_config()
    assert not collision._applicable_config()

    holonomic = MorpheusModule({"experiment": "holonomic_pendulum", "preprocessing_model": "WAN-2.1", "object_labels": _labels()})
    assert not holonomic._applicable_config()


def test_morpheus_publishes_only_finite_author_outputs(video_sample, monkeypatch):
    from ayase.modules.morpheus import MorpheusModule

    module = MorpheusModule({"experiment": "falling_ball", "preprocessing_model": "WAN-2.1", "object_labels": _labels()})
    module._backend = "morpheus-native"
    module._tracker = object()
    monkeypatch.setattr(module, "_run_source_pipeline", lambda path, annotations: ({"physical_score": 0.75, "statistical_score": float("nan")}, {"components": {"physical": {"energy": 0.75}}}))
    result = module.process(video_sample)
    assert result.quality_metrics.morpheus_physical_score == 0.75
    assert result.quality_metrics.morpheus_dynamical_score is None
    assert result.metadata["morpheus"]["components"]["physical"] == {"energy": 0.75}


def test_morpheus_accepts_per_sample_annotations_without_overwriting_them(video_sample, monkeypatch):
    from ayase.modules.morpheus import MorpheusModule

    annotations = {
        "experiment": "falling_ball",
        "preprocessing_model": "WAN-2.1",
        "object_labels": {"obj_1": {"positive": [[20, 30]], "negative": []}},
    }
    video_sample.metadata["morpheus_annotations"] = annotations
    module = MorpheusModule()
    module._backend = "morpheus-native"
    module._tracker = object()
    observed = {}

    def fake_run(path, resolved):
        observed.update(resolved)
        return {"physical_score": 0.9, "statistical_score": 0.7}, {"experiment": "falling_ball"}

    monkeypatch.setattr(module, "_run_source_pipeline", fake_run)
    module.process(video_sample)
    assert observed["object_labels"] == annotations["object_labels"]
    assert video_sample.metadata["morpheus_annotations"] == annotations
    assert video_sample.metadata["morpheus"] == {"experiment": "falling_ball"}


def test_morpheus_source_archive_materializes_contained_symlink(tmp_path, monkeypatch):
    from ayase.modules.morpheus import _ensure_source

    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("source/pkg/target.yaml", "model: pinned\n")
        link = zipfile.ZipInfo("source/pkg/alias.yaml")
        link.external_attr = (0o120777 << 16)
        bundle.writestr(link, "target.yaml")

    monkeypatch.setattr("urllib.request.urlretrieve", lambda url, target: Path(target).write_bytes(archive.read_bytes()))
    root = _ensure_source("unused", "abc123", tmp_path / "cache" / "source", ("pkg/alias.yaml",))
    assert (root / "pkg/alias.yaml").read_text() == "model: pinned\n"
    assert (root / ".ayase-source-revision").read_text() == "abc123"
    assert _ensure_source("unused", "abc123", root, ("pkg/alias.yaml",)) == root


def test_morpheus_source_archive_rejects_escape(tmp_path, monkeypatch):
    from ayase.modules.morpheus import _ensure_source

    archive = tmp_path / "escape.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("source/../../escape.py", "bad")
    monkeypatch.setattr("urllib.request.urlretrieve", lambda url, target: Path(target).write_bytes(archive.read_bytes()))
    import pytest

    with pytest.raises(ValueError, match="Unsafe"):
        _ensure_source("unused", "abc123", tmp_path / "cache" / "source", ("pkg/file.py",))


def test_morpheus_setup_pins_depth_path_and_restores_source_pipeline(tmp_path, monkeypatch):
    from ayase.modules.morpheus import MorpheusModule
    import ayase.config
    import ayase.runtime

    source = tmp_path / "morpheus-source"
    sam2_source = tmp_path / "sam2-source"
    for path in (
        source / "morpheus/pipeline.py",
        source / "configs/sam2.1_hiera_l.yaml",
        sam2_source / "sam2/__init__.py",
        sam2_source / "sam2/build_sam.py",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n")
    checkpoint = tmp_path / "sam2.pt"
    checkpoint.write_bytes(b"checkpoint")
    depth = tmp_path / "depth"
    depth.mkdir()

    sentinel = object()
    observed = {}
    processing = types.ModuleType("morpheus.tracking.processing")
    processing.pipeline = sentinel
    morpheus_package = types.ModuleType("morpheus")
    morpheus_package.__path__ = [str(source / "morpheus")]
    morpheus_package.__file__ = str(source / "morpheus/__init__.py")
    tracking_package = types.ModuleType("morpheus.tracking")
    tracking_package.__path__ = []
    tracker_module = types.ModuleType("morpheus.tracking.sam2_tracker")

    class FakeTracker:
        def __init__(self, **kwargs):
            observed.update(kwargs)
            observed["depth_result"] = processing.pipeline(model="nielsr/depth-anything-large")

    tracker_module.Sam2Tracker = FakeTracker
    sam2_package = types.ModuleType("sam2")
    sam2_package.__path__ = [str(sam2_source / "sam2")]
    sam2_package.__file__ = str(sam2_source / "sam2/__init__.py")
    hf = types.ModuleType("huggingface_hub")
    hf.snapshot_download = lambda **kwargs: str(depth)
    transformers = types.ModuleType("transformers")
    transformers.pipeline = lambda *args, **kwargs: kwargs["model"]
    for name, module in {
        "morpheus": morpheus_package,
        "morpheus.tracking": tracking_package,
        "morpheus.tracking.processing": processing,
        "morpheus.tracking.sam2_tracker": tracker_module,
        "sam2": sam2_package,
        "huggingface_hub": hf,
        "transformers": transformers,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(ayase.config, "download_model_file", lambda *args: checkpoint)
    monkeypatch.setattr(ayase.runtime, "resolve_torch_device", lambda value: "cpu")

    module = MorpheusModule({
        "source_root": str(source), "sam2_source_root": str(sam2_source),
        "models_dir": str(tmp_path / "models"),
    })
    old_path = list(sys.path)
    try:
        module._prepare_backend()
    finally:
        sys.path[:] = old_path
    assert observed["depth_result"] == str(depth.resolve())
    assert observed["checkpoint"] == str(checkpoint)
    assert processing.pipeline is sentinel


def test_morpheus_native_path_uses_source_preprocess_and_scorer(tmp_path, monkeypatch):
    from ayase.modules.morpheus import MorpheusModule

    observed = {}

    def fake_resize(video, rel_video, output, video_num, use_cache):
        observed["rel_video"] = rel_video
        folder = Path(output) / "single_frame_conditioning" / "WAN-2.1" / "plain" / "falling_ball" / "0"
        folder.mkdir(parents=True)
        (folder / "info.json").write_text(json.dumps({"experiment": "falling_ball", "model": "WAN-2.1", "conditioning": "single_frame_conditioning", "prompt_type": "plain"}))

    def fake_pipeline(folder, labels, tracker, args):
        observed["labels"] = labels
        prepared = Path(folder)
        info = json.loads((prepared / "info.json").read_text())
        observed["scorer_model"] = info["model"]
        destination = Path(args.output_dir) / prepared.relative_to(args.input_dir)
        destination.mkdir(parents=True)
        (destination / "combined_scores.json").write_text(json.dumps({"physical_score": 0.8, "statistical_score": 0.6, "individual_physical_scores": {"energy": 0.8}, "individual_statistical_scores": {"MSE": 0.1, "NMSE": 0.4}}))

    package = types.ModuleType("morpheus")
    package.__path__ = []
    resizing = types.ModuleType("morpheus.resizing")
    resizing.__path__ = []
    transforms = types.ModuleType("morpheus.resizing.transforms")
    transforms.process_mp4_file = fake_resize
    source_pipeline = types.ModuleType("morpheus.pipeline")
    source_pipeline.process_video_folder = fake_pipeline
    monkeypatch.setitem(sys.modules, "morpheus", package)
    monkeypatch.setitem(sys.modules, "morpheus.resizing", resizing)
    monkeypatch.setitem(sys.modules, "morpheus.resizing.transforms", transforms)
    monkeypatch.setitem(sys.modules, "morpheus.pipeline", source_pipeline)
    module = MorpheusModule({"experiment": "falling_ball", "preprocessing_model": "WAN-2.1", "object_labels": _labels(), "source_category": "real-world"})
    module._tracker = object()
    scores, details = module._run_source_pipeline(tmp_path / "input.mp4")
    assert scores["physical_score"] == 0.8
    assert observed["scorer_model"] == "real-world"
    assert observed["labels"]["falling_ball"]["video_0"] == _labels()
    assert details["physical_window_aggregation"] == "author MAX-window recipes"
    assert details["sam2_connected_components"] is False


@pytest.mark.parametrize(
    ("model", "token"),
    [("PyramidalFlow", "pyramid-flow"), ("WAN-2.1", "WAN-2.1"), ("Veo3", "Veo3")],
)
def test_morpheus_preprocessing_path_tokens(model, token):
    from ayase.modules.morpheus import _preprocessing_path_token

    assert _preprocessing_path_token(model) == token
