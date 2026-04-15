import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from ayase.base_modules import BatchMetricModule
from ayase.cli import app
from ayase.config import AyaseConfig
from ayase.models import CaptionMetadata, QualityMetrics, Sample
from ayase.pipeline import AyasePipeline, ModuleRegistry, Pipeline, PipelineModule
from ayase.scanner import sample_from_path, scan_dataset


def _plugin_readiness_label(path: Path) -> str:
    return str(path.resolve())


class _LifecycleModule(PipelineModule):
    name = "lifecycle_test"
    description = "Lifecycle test module"

    def __init__(self, config=None):
        super().__init__(config)
        self.mount_calls = 0
        self.execute_calls = 0

    def on_mount(self) -> None:
        self.mount_calls += 1
        self._mounted = True

    def on_execute(self) -> None:
        self.execute_calls += 1

    def process(self, sample: Sample) -> Sample:
        return sample


def test_pipeline_start_mounts_modules_once():
    module = _LifecycleModule()
    pipeline = Pipeline([module])

    pipeline.start()
    pipeline.start()

    assert module.mount_calls == 1
    assert module.execute_calls == 2


class _MissingPkgModule(PipelineModule):
    """Module whose on_mount() reports missing packages and stays unmounted."""
    name = "missing_pkg_test"
    description = "Simulates missing dependency"
    required_packages = ["nonexistent_pkg_xyz_42"]

    def __init__(self, config=None):
        super().__init__(config)
        self.process_calls = 0

    def process(self, sample: Sample) -> Sample:
        self.process_calls += 1
        return sample


def test_unmounted_module_skipped_during_processing():
    """Modules with missing packages stay unmounted and are skipped in process_sample."""
    # Temporarily disable test_mode so the package check runs normally
    prev = PipelineModule._global_test_mode
    PipelineModule._global_test_mode = False
    try:
        module = _MissingPkgModule()
        pipeline = Pipeline([module])
        pipeline.start()

        # Module should NOT be mounted (missing package)
        assert not module._mounted

        # Processing should skip the unmounted module
        sample = Sample(path=Path("test.mp4"), is_video=True)
        pipeline.process_sample(sample)
        assert module.process_calls == 0
    finally:
        PipelineModule._global_test_mode = prev


def test_scanner_attaches_exact_caption(tmp_path: Path):
    media = tmp_path / "clips" / "sample.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"not-a-real-video")

    caption_file = tmp_path / "clips" / "sample.txt"
    caption_file.write_text("a test caption", encoding="utf-8")

    samples = scan_dataset(tmp_path, include_videos=True, include_images=False)
    assert len(samples) == 1
    assert samples[0].caption is not None
    assert samples[0].caption.text == "a test caption"


def test_scanner_avoids_ambiguous_stem_caption(tmp_path: Path):
    media = tmp_path / "other" / "shared.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"not-a-real-video")

    c1 = tmp_path / "a" / "shared.txt"
    c1.parent.mkdir(parents=True)
    c1.write_text("caption-a", encoding="utf-8")

    c2 = tmp_path / "b" / "shared.txt"
    c2.parent.mkdir(parents=True)
    c2.write_text("caption-b", encoding="utf-8")

    samples = scan_dataset(tmp_path, include_videos=True, include_images=False)
    assert len(samples) == 1
    assert samples[0].caption is None


def test_sample_from_path_attaches_sidecar_caption(tmp_path: Path):
    media = tmp_path / "single.mp4"
    media.write_bytes(b"not-a-real-video")
    caption_file = tmp_path / "single.txt"
    caption_file.write_text("caption text", encoding="utf-8")

    sample = sample_from_path(media)

    assert sample is not None
    assert sample.caption is not None
    assert sample.caption.text == "caption text"


class _SparseMetricModule(PipelineModule):
    name = "sparse_metric_test"
    description = "Sparse metric test module"

    def process(self, sample: Sample) -> Sample:
        if sample.path.name == "second.mp4":
            sample.quality_metrics = QualityMetrics(technical_score=100.0)
        return sample


def test_pipeline_average_ignores_missing_metric_values():
    pipeline = Pipeline([_SparseMetricModule()])
    pipeline.start()
    pipeline.process_sample(Sample(path=Path("first.mp4"), is_video=True))
    pipeline.process_sample(Sample(path=Path("second.mp4"), is_video=True))
    assert pipeline.stats.avg_technical_score == 100.0


def test_video_memorability_is_clamped(tmp_path: Path):
    img = np.full((64, 64, 3), 255, dtype=np.uint8)
    img_path = tmp_path / "bright.png"
    cv2.imwrite(str(img_path), img)

    from ayase.modules.video_memorability import VideoMemorabilityModule

    m = VideoMemorabilityModule()
    sample = Sample(path=img_path, is_video=False)
    result = m.process(sample)

    # Without ML backend, module skips and returns sample unchanged
    if not m._ml_available:
        assert result.quality_metrics is None or result.quality_metrics.video_memorability is None
    else:
        assert result.quality_metrics is not None
        assert result.quality_metrics.video_memorability is not None
        assert 0.0 <= result.quality_metrics.video_memorability <= 1.0


def test_hdr_detector_does_not_flag_bright_uint8_as_hdr():
    from ayase.modules.hdr_sdr_vqa import HDRSDRVQAModule

    frame = np.full((32, 32, 3), 255, dtype=np.uint8)
    assert HDRSDRVQAModule()._detect_hdr(frame) is False


class _BatchProbeModule(BatchMetricModule):
    name = "batch_probe"
    description = "Batch metric cache probe"

    def extract_features(self, sample: Sample):
        return np.array([len(str(sample.path))], dtype=np.float32)

    def compute_distribution_metric(self, features, reference_features=None) -> float:
        return float(len(features) + (len(reference_features) if reference_features else 0))


def test_batch_module_collects_reference_features(tmp_path: Path):
    sample_path = tmp_path / "sample.mp4"
    reference_path = tmp_path / "reference.mp4"
    sample_path.write_bytes(b"a")
    reference_path.write_bytes(b"b")

    module = _BatchProbeModule()
    sample = Sample(path=sample_path, is_video=True, reference_path=reference_path)
    module.process(sample)

    assert len(module._feature_cache) == 1
    assert len(module._reference_cache) == 1


def test_config_env_overrides_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_path = tmp_path / "ayase.toml"
    config_path.write_text("[general]\nparallel_jobs = 16\n", encoding="utf-8")

    monkeypatch.setenv("AYASE_GENERAL__PARALLEL_JOBS", "32")
    cfg = AyaseConfig.load(config_path)

    assert cfg.general.parallel_jobs == 32


def test_config_env_parses_list_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_path = tmp_path / "ayase.toml"
    config_path.write_text('[pipeline]\nmodules = ["metadata"]\n', encoding="utf-8")

    monkeypatch.setenv("AYASE_PIPELINE__MODULES", '["basic_quality", "motion"]')
    monkeypatch.setenv("AYASE_PIPELINE__PLUGIN_FOLDERS", '["plugins", "custom_plugins"]')
    cfg = AyaseConfig.load(config_path)

    assert cfg.pipeline.modules == ["basic_quality", "motion"]
    assert cfg.pipeline.plugin_folders == [Path("plugins"), Path("custom_plugins")]


class _RequiredFileProbeModule(PipelineModule):
    name = "required_file_probe"
    description = "Required file path safety probe"

    def process(self, sample: Sample) -> Sample:
        return sample


def test_required_files_reject_path_escape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls = []

    def _fake_urlopen(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("unsafe path should be rejected before download")

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    module = _RequiredFileProbeModule(
        config={
            "models_dir": str(tmp_path),
            "required_files": {"../escape.bin": "https://example.com/model.bin"},
        }
    )

    module._ensure_required_files()

    assert not calls
    assert not (tmp_path / "escape.bin").exists()


class _CaptionScoreModule(PipelineModule):
    name = "caption_score_probe"
    description = "Probe cache invalidation when sample context changes"

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.technical_score = float(len(sample.caption.text)) if sample.caption else 0.0
        return sample


class _FileSizeScoreModule(PipelineModule):
    name = "file_size_score_probe"
    description = "Probe cache invalidation when media file state changes"

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.technical_score = float(sample.path.stat().st_size)
        return sample


class _ReferenceSizeScoreModule(PipelineModule):
    name = "reference_size_score_probe"
    description = "Probe cache invalidation when reference file state changes"

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        size = sample.reference_path.stat().st_size if sample.reference_path else 0
        sample.quality_metrics.technical_score = float(size)
        return sample


class _TestModeAwareScoreModule(PipelineModule):
    name = "test_mode_score_probe"
    description = "Probe cache/state invalidation across effective test mode"

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.technical_score = 0.0 if self.test_mode else 1.0
        return sample


class _AyasePipelineRunProbeModule(PipelineModule):
    name = "ayase_pipeline_run_probe"
    description = "Probe that AyasePipeline.run() starts from a clean pipeline each time"

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.technical_score = 1.0
        return sample


class _AyasePipelineConfigProbeModule(PipelineModule):
    name = "ayase_pipeline_config_probe"
    description = "Probe that AyasePipeline.run() preserves public module config"

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.technical_score = float(self.config.get("score", 1.0))
        return sample


def test_pipeline_reprocesses_same_path_when_sample_context_changes():
    pipeline = Pipeline([_CaptionScoreModule()])
    pipeline.start()

    first = Sample(
        path=Path("same.mp4"),
        is_video=True,
        caption=CaptionMetadata(text="a", length=1),
    )
    second = Sample(
        path=Path("same.mp4"),
        is_video=True,
        caption=CaptionMetadata(text="abcdef", length=6),
    )

    result_1 = pipeline.process_sample(first)
    result_2 = pipeline.process_sample(second)

    assert result_1 is not result_2
    assert result_1.quality_metrics is not None
    assert result_2.quality_metrics is not None
    assert result_1.quality_metrics.technical_score == 1.0
    assert result_2.quality_metrics.technical_score == 6.0
    assert len(pipeline.results) == 1
    assert pipeline.stats.total_samples == 1
    assert pipeline.stats.avg_technical_score == 6.0


def test_load_state_skips_stale_caption_context_and_rebuilds_stats(tmp_path: Path):
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"video-bytes")
    caption_path = tmp_path / "clip.txt"
    caption_path.write_text("short", encoding="utf-8")

    pipeline = Pipeline([_CaptionScoreModule()])
    pipeline.start()
    pipeline.process_sample(
        Sample(
            path=media_path,
            is_video=True,
            caption=CaptionMetadata(
                text="short",
                length=5,
                source_file=caption_path,
            ),
        )
    )

    state_path = tmp_path / "state" / "pipeline.json"
    pipeline.save_state(state_path)
    assert state_path.exists()

    caption_path.write_text("caption changed materially", encoding="utf-8")

    restored = Pipeline([_CaptionScoreModule()])
    restored.load_state(state_path)

    assert restored.results == {}
    assert restored.stats.total_samples == 0
    assert restored.stats.valid_samples == 0
    assert restored.stats.avg_technical_score is None


def test_load_state_skips_incompatible_pipeline_fingerprint(tmp_path: Path):
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"video")
    state_path = tmp_path / "pipeline_state.json"

    first_pipeline = Pipeline([_FileSizeScoreModule()])
    first_pipeline.start()
    first_pipeline.process_sample(Sample(path=media_path, is_video=True))
    first_pipeline.save_state(state_path)

    second_pipeline = Pipeline([_FileSizeScoreModule(), _CaptionScoreModule()])
    second_pipeline.load_state(state_path)
    second_pipeline.start()

    sample = Sample(
        path=media_path,
        is_video=True,
        caption=CaptionMetadata(text="abc", length=3),
    )
    result = second_pipeline.process_sample(sample)

    assert result.quality_metrics is not None
    assert result.quality_metrics.technical_score == 3.0
    assert second_pipeline.stats.total_samples == 1


def test_load_state_skips_legacy_files_without_pipeline_fingerprint(tmp_path: Path):
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"video")
    state_path = tmp_path / "legacy_state.json"

    pipeline = Pipeline([_FileSizeScoreModule()])
    pipeline.start()
    result = pipeline.process_sample(Sample(path=media_path, is_video=True))
    state_path.write_text(
        json.dumps(
            {
                "results": {str(media_path): result.model_dump(mode="json")},
                "stats": pipeline.stats.model_dump(mode="json"),
                "cache_manifest": {
                    str(media_path): pipeline._sample_state_manifest(result),
                },
            }
        ),
        encoding="utf-8",
    )

    restored = Pipeline([_FileSizeScoreModule()])
    restored.load_state(state_path)

    assert restored.results == {}
    assert restored.stats.total_samples == 0


def test_load_state_replaces_existing_results_when_state_is_incompatible(tmp_path: Path):
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"video")
    state_path = tmp_path / "bad_state.json"

    pipeline = Pipeline([_FileSizeScoreModule()])
    pipeline.start()
    pipeline.process_sample(Sample(path=media_path, is_video=True))
    assert pipeline.stats.total_samples == 1

    state_path.write_text(
        json.dumps(
            {
                "pipeline_fingerprint": {"modules": []},
                "results": {},
                "stats": {
                    "total_samples": 0,
                    "valid_samples": 0,
                    "invalid_samples": 0,
                    "total_size": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    pipeline.load_state(state_path)

    assert pipeline.results == {}
    assert pipeline.stats.total_samples == 0
    assert pipeline.stats.avg_technical_score is None


def test_load_state_replaces_existing_results_when_state_is_legacy(tmp_path: Path):
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"video")
    state_path = tmp_path / "legacy_state.json"

    pipeline = Pipeline([_FileSizeScoreModule()])
    pipeline.start()
    result = pipeline.process_sample(Sample(path=media_path, is_video=True))
    assert pipeline.stats.total_samples == 1

    state_path.write_text(
        json.dumps(
            {
                "results": {str(media_path): result.model_dump(mode="json")},
                "stats": pipeline.stats.model_dump(mode="json"),
            }
        ),
        encoding="utf-8",
    )

    pipeline.load_state(state_path)

    assert pipeline.results == {}
    assert pipeline.stats.total_samples == 0


def test_load_state_preserves_existing_results_when_state_file_is_corrupt(tmp_path: Path):
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"video")
    state_path = tmp_path / "corrupt_state.json"

    pipeline = Pipeline([_FileSizeScoreModule()])
    pipeline.start()
    pipeline.process_sample(Sample(path=media_path, is_video=True))
    assert pipeline.stats.total_samples == 1

    state_path.write_text("{not json", encoding="utf-8")
    pipeline.load_state(state_path)

    assert sorted(pipeline.results) == [str(media_path)]
    assert pipeline.stats.total_samples == 1
    assert pipeline.stats.avg_technical_score == 5.0


def test_load_state_rejects_resume_cache_from_different_test_mode(tmp_path: Path):
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"video")
    state_path = tmp_path / "pipeline_state.json"
    prev = PipelineModule._global_test_mode
    PipelineModule._global_test_mode = False
    try:
        first_pipeline = Pipeline([_TestModeAwareScoreModule(config={"test_mode": True})])
        first_pipeline.start()
        first_pipeline.process_sample(Sample(path=media_path, is_video=True))
        first_pipeline.save_state(state_path)

        second_pipeline = Pipeline([_TestModeAwareScoreModule(config={"test_mode": False})])
        second_pipeline.load_state(state_path)
        second_pipeline.start()

        result = second_pipeline.process_sample(Sample(path=media_path, is_video=True))

        assert result.quality_metrics is not None
        assert result.quality_metrics.technical_score == 1.0
        assert second_pipeline.stats.total_samples == 1
    finally:
        PipelineModule._global_test_mode = prev


def test_pipeline_reprocesses_same_path_when_media_file_changes(tmp_path: Path):
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"1234")

    pipeline = Pipeline([_FileSizeScoreModule()])
    pipeline.start()

    first = pipeline.process_sample(Sample(path=media_path, is_video=True))
    media_path.write_bytes(b"1234567890")
    second = pipeline.process_sample(Sample(path=media_path, is_video=True))

    assert first is not second
    assert first.quality_metrics is not None
    assert second.quality_metrics is not None
    assert first.quality_metrics.technical_score == 4.0
    assert second.quality_metrics.technical_score == 10.0
    assert pipeline.stats.total_samples == 1
    assert pipeline.stats.avg_technical_score == 10.0


def test_pipeline_reprocesses_same_path_when_reference_file_changes(tmp_path: Path):
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"video")
    ref_path = tmp_path / "ref.png"
    ref_path.write_bytes(b"12")

    pipeline = Pipeline([_ReferenceSizeScoreModule()])
    pipeline.start()

    first = pipeline.process_sample(Sample(path=media_path, is_video=True, reference_path=ref_path))
    ref_path.write_bytes(b"123456")
    second = pipeline.process_sample(Sample(path=media_path, is_video=True, reference_path=ref_path))

    assert first is not second
    assert first.quality_metrics is not None
    assert second.quality_metrics is not None
    assert first.quality_metrics.technical_score == 2.0
    assert second.quality_metrics.technical_score == 6.0
    assert pipeline.stats.total_samples == 1
    assert pipeline.stats.avg_technical_score == 6.0


def test_duplicate_module_name_raises():
    class _DuplicateNameA(PipelineModule):
        name = "duplicate_name_probe"
        description = "First duplicate probe"

        def process(self, sample: Sample) -> Sample:
            return sample

    with pytest.raises(ValueError, match="Duplicate module name 'duplicate_name_probe'"):

        class _DuplicateNameB(PipelineModule):
            name = "duplicate_name_probe"
            description = "Second duplicate probe"

            def process(self, sample: Sample) -> Sample:
                return sample


def test_module_registry_readiness_updates():
    original = dict(ModuleRegistry._readiness)
    try:
        ModuleRegistry._readiness.clear()
        ModuleRegistry._record_readiness("demo_probe", False, "first failure")
        ModuleRegistry._record_readiness("demo_probe", True, None)

        assert ModuleRegistry.readiness_report()["demo_probe"] == {
            "status": "ready",
            "error": None,
        }
    finally:
        ModuleRegistry._readiness.clear()
        ModuleRegistry._readiness.update(original)


def test_broken_plugin_does_not_leak_registered_module(tmp_path: Path):
    plugin_path = tmp_path / "broken_plugin.py"
    plugin_path.write_text(
        "from ayase.pipeline import PipelineModule\n"
        "class BrokenPluginModule(PipelineModule):\n"
        "    name = 'broken_plugin_probe'\n"
        "    description = 'broken'\n"
        "    def process(self, sample):\n"
        "        return sample\n"
        "raise RuntimeError('boom')\n",
        encoding="utf-8",
    )

    original_modules = dict(ModuleRegistry._modules)
    original_readiness = dict(ModuleRegistry._readiness)
    original_external_labels = {
        key: set(value) for key, value in ModuleRegistry._external_plugin_labels.items()
    }
    original_external_modules = dict(ModuleRegistry._external_plugin_modules)
    try:
        ModuleRegistry.discover_external_modules([tmp_path])

        assert "broken_plugin_probe" not in ModuleRegistry._modules
        assert (
            ModuleRegistry.readiness_report()[_plugin_readiness_label(plugin_path)]["status"]
            == "missing"
        )
    finally:
        ModuleRegistry._modules.clear()
        ModuleRegistry._modules.update(original_modules)
        ModuleRegistry._readiness.clear()
        ModuleRegistry._readiness.update(original_readiness)
        ModuleRegistry._external_plugin_labels.clear()
        ModuleRegistry._external_plugin_labels.update(original_external_labels)
        ModuleRegistry._external_plugin_modules.clear()
        ModuleRegistry._external_plugin_modules.update(original_external_modules)


def test_removed_plugin_prunes_stale_readiness_entry(tmp_path: Path):
    plugin_path = tmp_path / "broken_plugin.py"
    plugin_path.write_text("raise RuntimeError('boom')\n", encoding="utf-8")

    original_modules = dict(ModuleRegistry._modules)
    original_readiness = dict(ModuleRegistry._readiness)
    original_external_labels = {
        key: set(value) for key, value in ModuleRegistry._external_plugin_labels.items()
    }
    original_external_modules = dict(ModuleRegistry._external_plugin_modules)
    try:
        ModuleRegistry.discover_external_modules([tmp_path])
        assert (
            ModuleRegistry.readiness_report()[_plugin_readiness_label(plugin_path)]["status"]
            == "missing"
        )

        plugin_path.unlink()
        ModuleRegistry.discover_external_modules([tmp_path])

        assert _plugin_readiness_label(plugin_path) not in ModuleRegistry.readiness_report()
    finally:
        ModuleRegistry._modules.clear()
        ModuleRegistry._modules.update(original_modules)
        ModuleRegistry._readiness.clear()
        ModuleRegistry._readiness.update(original_readiness)
        ModuleRegistry._external_plugin_labels.clear()
        ModuleRegistry._external_plugin_labels.update(original_external_labels)
        ModuleRegistry._external_plugin_modules.clear()
        ModuleRegistry._external_plugin_modules.update(original_external_modules)


def test_same_named_plugins_in_different_folders_keep_distinct_readiness_entries(tmp_path: Path):
    first_dir = tmp_path / "first_plugins"
    second_dir = tmp_path / "second_plugins"
    first_dir.mkdir()
    second_dir.mkdir()
    first_plugin = first_dir / "same.py"
    second_plugin = second_dir / "same.py"
    first_plugin.write_text("raise RuntimeError('one')\n", encoding="utf-8")
    second_plugin.write_text("raise RuntimeError('two')\n", encoding="utf-8")

    original_modules = dict(ModuleRegistry._modules)
    original_readiness = dict(ModuleRegistry._readiness)
    original_external_labels = {
        key: set(value) for key, value in ModuleRegistry._external_plugin_labels.items()
    }
    original_external_modules = dict(ModuleRegistry._external_plugin_modules)
    try:
        ModuleRegistry.discover_external_modules([first_dir, second_dir])
        report = ModuleRegistry.readiness_report()

        assert report[_plugin_readiness_label(first_plugin)] == {
            "status": "missing",
            "error": "one",
        }
        assert report[_plugin_readiness_label(second_plugin)] == {
            "status": "missing",
            "error": "two",
        }
    finally:
        ModuleRegistry._modules.clear()
        ModuleRegistry._modules.update(original_modules)
        ModuleRegistry._readiness.clear()
        ModuleRegistry._readiness.update(original_readiness)
        ModuleRegistry._external_plugin_labels.clear()
        ModuleRegistry._external_plugin_labels.update(original_external_labels)
        ModuleRegistry._external_plugin_modules.clear()
        ModuleRegistry._external_plugin_modules.update(original_external_modules)


def test_updated_plugin_is_reloaded_on_rediscovery(tmp_path: Path):
    plugin_path = tmp_path / "hello_plugin.py"
    plugin_path.write_text(
        "from ayase.pipeline import PipelineModule\n"
        "class HelloPluginModule(PipelineModule):\n"
        "    name = 'hello_plugin_probe'\n"
        "    description = 'version one'\n"
        "    def process(self, sample):\n"
        "        return sample\n",
        encoding="utf-8",
    )

    original_modules = dict(ModuleRegistry._modules)
    original_readiness = dict(ModuleRegistry._readiness)
    original_external_labels = {
        key: set(value) for key, value in ModuleRegistry._external_plugin_labels.items()
    }
    original_external_modules = dict(ModuleRegistry._external_plugin_modules)
    try:
        ModuleRegistry.discover_external_modules([tmp_path])
        assert ModuleRegistry.get_module("hello_plugin_probe") is not None
        assert ModuleRegistry.get_module("hello_plugin_probe").description == "version one"

        plugin_path.write_text(
            "from ayase.pipeline import PipelineModule\n"
            "class HelloPluginModule(PipelineModule):\n"
            "    name = 'hello_plugin_probe'\n"
            "    description = 'version two'\n"
            "    def process(self, sample):\n"
            "        return sample\n",
            encoding="utf-8",
        )

        ModuleRegistry.discover_external_modules([tmp_path])

        assert ModuleRegistry.get_module("hello_plugin_probe") is not None
        assert ModuleRegistry.get_module("hello_plugin_probe").description == "version two"
    finally:
        ModuleRegistry._modules.clear()
        ModuleRegistry._modules.update(original_modules)
        ModuleRegistry._readiness.clear()
        ModuleRegistry._readiness.update(original_readiness)
        ModuleRegistry._external_plugin_labels.clear()
        ModuleRegistry._external_plugin_labels.update(original_external_labels)
        ModuleRegistry._external_plugin_modules.clear()
        ModuleRegistry._external_plugin_modules.update(original_external_modules)


def test_removed_successful_plugin_is_unregistered(tmp_path: Path):
    plugin_path = tmp_path / "hello_plugin.py"
    plugin_path.write_text(
        "from ayase.pipeline import PipelineModule\n"
        "class HelloPluginModule(PipelineModule):\n"
        "    name = 'hello_plugin_probe'\n"
        "    description = 'version one'\n"
        "    def process(self, sample):\n"
        "        return sample\n",
        encoding="utf-8",
    )

    original_modules = dict(ModuleRegistry._modules)
    original_readiness = dict(ModuleRegistry._readiness)
    original_external_labels = {
        key: set(value) for key, value in ModuleRegistry._external_plugin_labels.items()
    }
    original_external_modules = dict(ModuleRegistry._external_plugin_modules)
    try:
        ModuleRegistry.discover_external_modules([tmp_path])
        assert ModuleRegistry.get_module("hello_plugin_probe") is not None

        plugin_path.unlink()
        ModuleRegistry.discover_external_modules([tmp_path])

        assert ModuleRegistry.get_module("hello_plugin_probe") is None
    finally:
        ModuleRegistry._modules.clear()
        ModuleRegistry._modules.update(original_modules)
        ModuleRegistry._readiness.clear()
        ModuleRegistry._readiness.update(original_readiness)
        ModuleRegistry._external_plugin_labels.clear()
        ModuleRegistry._external_plugin_labels.update(original_external_labels)
        ModuleRegistry._external_plugin_modules.clear()
        ModuleRegistry._external_plugin_modules.update(original_external_modules)


def test_stats_counts_image_only_dataset():
    runner = CliRunner()
    with runner.isolated_filesystem():
        dataset = Path("dataset")
        dataset.mkdir()
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        cv2.imwrite(str(dataset / "photo.png"), image)

        result = runner.invoke(app, ["stats", str(dataset)])

        assert result.exit_code == 0
        assert "Total samples: 1" in result.output


def test_filter_list_mode_does_not_require_output():
    runner = CliRunner()
    with runner.isolated_filesystem():
        dataset = Path("dataset")
        dataset.mkdir()
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        cv2.imwrite(str(dataset / "photo.png"), image)

        result = runner.invoke(app, ["filter", str(dataset), "--mode", "list"])

        assert result.exit_code == 0
        assert "photo.png" in result.output


def test_scan_with_explicit_output_does_not_create_implicit_artifact_dir():
    runner = CliRunner()
    with runner.isolated_filesystem():
        dataset = Path("dataset")
        dataset.mkdir()
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        cv2.imwrite(str(dataset / "photo.png"), image)

        result = runner.invoke(
            app,
            [
                "scan",
                str(dataset),
                "--modules",
                "metadata",
                "--format",
                "json",
                "--output",
                "report.json",
            ],
        )

        assert result.exit_code == 0
        assert Path("report.json").exists()
        assert not Path("reports").exists()


def test_run_stops_pipeline_when_no_valid_inputs(monkeypatch: pytest.MonkeyPatch):
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("note.txt").write_text("not media", encoding="utf-8")
        stop_calls = []

        monkeypatch.setattr("ayase.cli._parse_pipeline_str", lambda *args, **kwargs: [])

        def _stop(self):
            stop_calls.append(self)

        monkeypatch.setattr("ayase.cli.Pipeline.stop", _stop)

        result = runner.invoke(app, ["run", "note.txt", "--pipeline", "metadata"])

        assert result.exit_code == 0
        assert len(stop_calls) == 1


def test_scan_stops_pipeline_when_scanning_raises(monkeypatch: pytest.MonkeyPatch):
    runner = CliRunner()
    with runner.isolated_filesystem():
        stop_calls = []
        monkeypatch.setattr("ayase.cli._select_modules", lambda *args, **kwargs: [])

        def _stop(self):
            stop_calls.append(self)

        monkeypatch.setattr("ayase.cli.Pipeline.stop", _stop)

        result = runner.invoke(app, ["scan", "missing_dataset"])

        assert result.exit_code != 0
        assert isinstance(result.exception, FileNotFoundError)
        assert len(stop_calls) == 1


class _BrokenReadinessModule(PipelineModule):
    name = "broken_readiness_probe"
    description = "Probe modules check readiness at mount time"
    cleanup_calls = 0

    def on_mount(self) -> None:
        raise RuntimeError("mount failed")

    def on_dispose(self) -> None:
        type(self).cleanup_calls += 1

    def process(self, sample: Sample) -> Sample:
        return sample


def test_modules_check_fails_when_module_mount_fails(monkeypatch: pytest.MonkeyPatch):
    runner = CliRunner()
    original_modules = dict(ModuleRegistry._modules)
    original_readiness = dict(ModuleRegistry._readiness)
    _BrokenReadinessModule.cleanup_calls = 0
    try:
        ModuleRegistry._modules = {_BrokenReadinessModule.name: _BrokenReadinessModule}
        ModuleRegistry._readiness = {_BrokenReadinessModule.name: {"status": "ready", "error": None}}
        monkeypatch.setattr("ayase.cli._discover_all_modules", lambda config: None)
        monkeypatch.setattr("ayase.cli.AyaseConfig.load", classmethod(lambda cls: AyaseConfig()))

        result = runner.invoke(app, ["modules", "check"])

        assert result.exit_code == 1
        assert "mount failed" in result.output
        assert _BrokenReadinessModule.cleanup_calls == 1
    finally:
        ModuleRegistry._modules = original_modules
        ModuleRegistry._readiness = original_readiness


def test_run_rejects_unknown_format(monkeypatch: pytest.MonkeyPatch):
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("clip.jpg").write_bytes(b"image")
        monkeypatch.setattr("ayase.cli._parse_pipeline_str", lambda *args, **kwargs: [])

        result = runner.invoke(app, ["run", "clip.jpg", "--pipeline", "metadata", "--format", "xml"])

        assert result.exit_code == 1
        assert "Unknown format: xml" in result.output


def test_stats_rejects_unknown_format():
    runner = CliRunner()
    with runner.isolated_filesystem():
        dataset = Path("dataset")
        dataset.mkdir()
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        cv2.imwrite(str(dataset / "photo.png"), image)

        result = runner.invoke(app, ["stats", str(dataset), "--format", "xml"])

        assert result.exit_code == 1
        assert "Unknown format: xml" in result.output


class _ConfigAwareReadinessModule(PipelineModule):
    name = "config_aware_readiness_probe"
    description = "Probe that modules check mounts with loaded runtime config"
    cleanup_calls = 0

    def on_mount(self) -> None:
        if self.config.get("models_dir") != str(Path("custom_models")):
            raise RuntimeError(f"wrong models_dir={self.config.get('models_dir')!r}")
        self._mounted = True

    def on_dispose(self) -> None:
        type(self).cleanup_calls += 1

    def process(self, sample: Sample) -> Sample:
        return sample


def test_modules_check_uses_loaded_runtime_config(monkeypatch: pytest.MonkeyPatch):
    runner = CliRunner()
    original_modules = dict(ModuleRegistry._modules)
    original_readiness = dict(ModuleRegistry._readiness)
    _ConfigAwareReadinessModule.cleanup_calls = 0
    try:
        ModuleRegistry._modules = {
            _ConfigAwareReadinessModule.name: _ConfigAwareReadinessModule
        }
        ModuleRegistry._readiness = {
            _ConfigAwareReadinessModule.name: {"status": "ready", "error": None}
        }
        monkeypatch.setattr("ayase.cli._discover_all_modules", lambda config: None)
        monkeypatch.setattr(
            "ayase.cli.AyaseConfig.load",
            classmethod(
                lambda cls: AyaseConfig.model_validate(
                    {"general": {"models_dir": "custom_models"}}
                )
            ),
        )

        result = runner.invoke(app, ["modules", "check"])

        assert result.exit_code == 0
        assert "All 1 module(s) loaded successfully." in result.output
        assert _ConfigAwareReadinessModule.cleanup_calls == 1
    finally:
        ModuleRegistry._modules = original_modules
        ModuleRegistry._readiness = original_readiness


def test_ayase_pipeline_run_replaces_previous_run_state(tmp_path: Path):
    first_dataset = tmp_path / "first"
    second_dataset = tmp_path / "second"
    first_dataset.mkdir()
    second_dataset.mkdir()

    first_image = np.zeros((16, 16, 3), dtype=np.uint8)
    second_image = np.ones((16, 16, 3), dtype=np.uint8) * 255
    cv2.imwrite(str(first_dataset / "first.png"), first_image)
    cv2.imwrite(str(second_dataset / "second.png"), second_image)

    ayase = AyasePipeline(modules=[_AyasePipelineRunProbeModule.name])

    first_results = ayase.run(first_dataset)
    assert sorted(first_results) == [str(first_dataset / "first.png")]
    assert ayase.stats.total_samples == 1

    second_results = ayase.run(second_dataset)

    assert sorted(second_results) == [str(second_dataset / "second.png")]
    assert ayase.results is ayase.pipeline.results
    assert ayase.stats.total_samples == 1


def test_ayase_pipeline_run_preserves_public_pipeline_hooks(tmp_path: Path):
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"video")
    ayase = AyasePipeline(modules=[_CaptionScoreModule.name])
    ayase.pipeline.add_hook(
        _CaptionScoreModule.name,
        before=lambda sample: sample.model_copy(update={"caption": None}),
    )

    results = ayase.run(
        tmp_path,
        samples=[
            Sample(
                path=media_path,
                is_video=True,
                caption=CaptionMetadata(text="abcdef", length=6),
            )
        ],
    )

    result = results[str(media_path)]
    assert result.quality_metrics is not None
    assert result.quality_metrics.technical_score == 0.0


def test_ayase_pipeline_run_preserves_public_pipeline_module_config(tmp_path: Path):
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"video")
    ayase = AyasePipeline(modules=[_AyasePipelineConfigProbeModule.name])
    ayase.pipeline.modules[0].config["score"] = 7.0

    results = ayase.run(
        tmp_path,
        samples=[Sample(path=media_path, is_video=True)],
    )

    result = results[str(media_path)]
    assert result.quality_metrics is not None
    assert result.quality_metrics.technical_score == 7.0


def test_pipeline_repeated_start_resets_previous_run_state(tmp_path: Path):
    first_image = tmp_path / "first.png"
    second_image = tmp_path / "second.png"
    first_image.write_bytes(b"first")
    second_image.write_bytes(b"second")

    pipeline = Pipeline([_AyasePipelineRunProbeModule()])

    pipeline.start()
    first_result = pipeline.process_sample(Sample(path=first_image, is_video=False))
    pipeline.stop()

    assert first_result.quality_metrics is not None
    assert pipeline.stats.total_samples == 1
    assert sorted(pipeline.results) == [str(first_image)]

    pipeline.start()
    second_result = pipeline.process_sample(Sample(path=second_image, is_video=False))
    pipeline.stop()

    assert second_result.quality_metrics is not None
    assert pipeline.stats.total_samples == 1
    assert sorted(pipeline.results) == [str(second_image)]
