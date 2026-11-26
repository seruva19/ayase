"""Contract tests for core pipeline execution, caching, hooks, and statistics.

Covers:
1.  Backend provenance (``QualityMetrics.metric_backends``) persisted
    automatically from ``module._backend`` in both execution paths.
2.  Module exceptions surface as visible issues, mark the sample incomplete,
    and force reprocessing on resume instead of being served from cache.
3.  Before-hook state is always reverted (after-hook in ``finally``) even when
    ``process()`` raises — no caption/state leak to later modules.
4.  Issues appended in ``post_process()`` are reflected in final stats, and a
    resumed run reports the same stats.
5.  ``cache_enabled`` propagation from global config + sane aggregation.
6.  ``default_config`` deep copy + recursive merge (no shared nested dicts).
7.  Stable (process-independent) external plugin module names.
8.  Frame cache: exact sampling grids per file, lazy color
    conversion, read-only zero-copy arrays.
9.  Zero-frame-count videos fall back to bounded sequential reads.
10. ``issues_by_type`` uses structured issue types, not message parsing.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ayase.models import (
    CaptionMetadata,
    QualityMetrics,
    Sample,
    ValidationIssue,
    ValidationSeverity,
)
from ayase.pipeline import ModuleRegistry, Pipeline, PipelineModule


# ---------------------------------------------------------------------------
# Test module helpers (unique "corefix_" names; auto-registered but not
# packaged, so docs-integrity tests ignore them).
# ---------------------------------------------------------------------------


class _BackendProbeModule(PipelineModule):
    name = "corefix_backend_probe"
    description = "Sets a metric and tracks a tiered backend"

    def __init__(self, config=None):
        super().__init__(config)
        self._backend = "heuristic"
        self.calls = 0

    def process(self, sample: Sample) -> Sample:
        self.calls += 1
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.blur_score = 1.0
        return sample


class _NoBackendModule(PipelineModule):
    name = "corefix_no_backend"
    description = "Module without a _backend attribute"

    def process(self, sample: Sample) -> Sample:
        return sample


class _FailingModule(PipelineModule):
    name = "corefix_failing"
    description = "Always raises"

    def __init__(self, config=None):
        super().__init__(config)
        self.calls = 0
        self._backend = "pyiqa"  # must NOT be recorded on failure

    def process(self, sample: Sample) -> Sample:
        self.calls += 1
        raise RuntimeError("boom")


class _CaptionRecorderModule(PipelineModule):
    name = "corefix_caption_recorder"
    description = "Records the caption text it observes"

    def __init__(self, config=None):
        super().__init__(config)
        self.seen = []

    def process(self, sample: Sample) -> Sample:
        self.seen.append(sample.caption.text if sample.caption else None)
        return sample


class _PostIssueModule(PipelineModule):
    name = "corefix_post_issue"
    description = "Appends an ERROR issue during post_process"

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.blur_score = 2.0
        return sample

    def post_process(self, all_samples):
        for s in all_samples:
            s.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    issue_type="post_selection",
                    message="dropped by diversity selection",
                )
            )


class _NestedConfigModule(PipelineModule):
    name = "corefix_nested_config"
    description = "Has nested dicts in default_config"
    default_config = {"weights": {"a": 1.0, "b": 2.0}, "flag": True}

    def process(self, sample: Sample) -> Sample:
        return sample


class _DisposeInvalidatorModule(PipelineModule):
    """Flips processed samples to invalid during ``on_dispose`` (like dedup)."""

    name = "corefix_dispose_invalidator"
    description = "Appends an ERROR issue during on_dispose"

    def __init__(self, config=None):
        super().__init__(config)
        self._targets = []

    def process(self, sample: Sample) -> Sample:
        self._targets.append(sample)
        return sample

    def on_dispose(self) -> None:
        for s in self._targets:
            s.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    issue_type="duplicate",
                    message="near-duplicate removed",
                )
            )
        super().on_dispose()


def _media(tmp_path: Path, name: str = "clip.mp4") -> Path:
    path = tmp_path / name
    path.write_bytes(b"video-bytes")
    return path


# ---------------------------------------------------------------------------
# 1. Backend provenance
# ---------------------------------------------------------------------------


def test_backend_provenance_persisted_single_path(tmp_path: Path):
    media = _media(tmp_path)
    pipeline = Pipeline([_BackendProbeModule(), _NoBackendModule()])
    pipeline.start()

    result = pipeline.process_sample(Sample(path=media, is_video=True))

    assert result.quality_metrics is not None
    assert result.quality_metrics.metric_backends == {
        "corefix_backend_probe": "heuristic"
    }
    # Module without _backend must not appear.
    assert "corefix_no_backend" not in result.quality_metrics.metric_backends


def test_backend_provenance_persisted_batch_path(tmp_path: Path):
    samples = [
        Sample(path=_media(tmp_path, f"{i}.mp4"), is_video=True) for i in range(2)
    ]
    pipeline = Pipeline([_BackendProbeModule()])
    pipeline.start()

    results = pipeline.process_samples(samples, batch_size=2)

    assert len(results) == 2
    for result in results:
        assert result.quality_metrics.metric_backends == {
            "corefix_backend_probe": "heuristic"
        }


def test_backend_provenance_not_a_metric():
    qm = QualityMetrics(metric_backends={"m": "pyiqa"}, blur_score=1.0)
    assert qm.non_null_count() == 1
    assert "metric_backends" not in qm.non_null_metrics()
    grouped = qm.to_grouped_dict()
    for fields in grouped.values():
        assert "metric_backends" not in fields


def test_backend_provenance_state_round_trip(tmp_path: Path):
    media = _media(tmp_path)
    state_path = tmp_path / "state.json"
    pipeline = Pipeline([_BackendProbeModule()])
    pipeline.start()
    pipeline.process_sample(Sample(path=media, is_video=True))
    pipeline.save_state(state_path)

    restored = Pipeline([_BackendProbeModule()])
    restored.load_state(state_path)

    assert len(restored.results) == 1
    sample = next(iter(restored.results.values()))
    assert sample.quality_metrics.metric_backends == {
        "corefix_backend_probe": "heuristic"
    }


def test_load_state_tolerates_legacy_files_without_new_fields(tmp_path: Path):
    """State files written before metric_backends/failed_modules must load."""
    media = _media(tmp_path)
    state_path = tmp_path / "state.json"
    pipeline = Pipeline([_BackendProbeModule()])
    pipeline.start()
    pipeline.process_sample(Sample(path=media, is_video=True))
    pipeline.save_state(state_path)

    raw = json.loads(state_path.read_text(encoding="utf-8"))
    for sample_dict in raw["results"].values():
        sample_dict.pop("failed_modules", None)
        if isinstance(sample_dict.get("quality_metrics"), dict):
            sample_dict["quality_metrics"].pop("metric_backends", None)
    state_path.write_text(json.dumps(raw), encoding="utf-8")

    restored = Pipeline([_BackendProbeModule()])
    restored.load_state(state_path)

    assert len(restored.results) == 1
    sample = next(iter(restored.results.values()))
    assert sample.failed_modules == []
    assert sample.quality_metrics.metric_backends == {}


# ---------------------------------------------------------------------------
# 2. Module exceptions: visible issue + incomplete result + reprocessing
# ---------------------------------------------------------------------------


def test_module_exception_records_issue_and_failed_module(tmp_path: Path):
    media = _media(tmp_path)
    module = _FailingModule()
    pipeline = Pipeline([module])
    pipeline.start()

    result = pipeline.process_sample(Sample(path=media, is_video=True))

    assert result.failed_modules == ["corefix_failing"]
    issues = [i for i in result.validation_issues if i.issue_type == "module_error"]
    assert len(issues) == 1
    issue = issues[0]
    assert issue.severity == ValidationSeverity.WARNING
    assert "corefix_failing" in issue.message
    assert "RuntimeError" in issue.message
    # Tooling failure must not mark the data itself invalid.
    assert result.is_valid
    # A failing module must not advertise a backend.
    assert (
        result.quality_metrics is None
        or "corefix_failing" not in result.quality_metrics.metric_backends
    )


def test_failed_sample_not_served_from_cache_same_run(tmp_path: Path):
    media = _media(tmp_path)
    module = _FailingModule()
    pipeline = Pipeline([module])
    pipeline.start()

    pipeline.process_sample(Sample(path=media, is_video=True))
    pipeline.process_sample(Sample(path=media, is_video=True))

    assert module.calls == 2  # incomplete result is never served from cache
    assert pipeline.stats.total_samples == 1  # result replaced, not duplicated


def test_complete_sample_is_served_from_cache(tmp_path: Path):
    media = _media(tmp_path)
    module = _BackendProbeModule()
    pipeline = Pipeline([module])
    pipeline.start()

    pipeline.process_sample(Sample(path=media, is_video=True))
    pipeline.process_sample(Sample(path=media, is_video=True))

    assert module.calls == 1


def test_failed_sample_reprocessed_after_resume(tmp_path: Path):
    media = _media(tmp_path)
    state_path = tmp_path / "state.json"

    first = Pipeline([_FailingModule()])
    first.start()
    first.process_sample(Sample(path=media, is_video=True))
    first.save_state(state_path)

    module = _FailingModule()
    resumed = Pipeline([module])
    resumed.load_state(state_path)
    resumed.start()
    resumed.process_sample(Sample(path=media, is_video=True))

    assert module.calls == 1  # reprocessed, not served from restored cache


def test_module_exception_batch_path_records_failure(tmp_path: Path):
    samples = [
        Sample(path=_media(tmp_path, f"{i}.mp4"), is_video=True) for i in range(2)
    ]
    module = _FailingModule()
    pipeline = Pipeline([module])
    pipeline.start()

    results = pipeline.process_samples(samples, batch_size=2)

    for result in results:
        assert result.failed_modules == ["corefix_failing"]
        assert any(i.issue_type == "module_error" for i in result.validation_issues)
        assert result.is_valid


# ---------------------------------------------------------------------------
# 3. Hook state reverted when process() raises
# ---------------------------------------------------------------------------


def _add_caption_hooks(pipeline: Pipeline, module_name: str, original: CaptionMetadata):
    pipeline.add_hook(
        module_name,
        before=lambda item: item.model_copy(
            update={"caption": CaptionMetadata(text="condensed", length=9)}
        ),
        after=lambda item: item.model_copy(update={"caption": original}),
    )


def test_hook_reverted_when_module_raises_single_path(tmp_path: Path):
    media = _media(tmp_path)
    original = CaptionMetadata(text="the original caption", length=20)
    recorder = _CaptionRecorderModule()
    pipeline = Pipeline([_FailingModule(), recorder])
    _add_caption_hooks(pipeline, "corefix_failing", original)
    pipeline.start()

    result = pipeline.process_sample(
        Sample(path=media, is_video=True, caption=original)
    )

    # The module after the raising one must see the restored caption...
    assert recorder.seen == ["the original caption"]
    # ...and the stored result must not leak the hook mutation.
    assert result.caption == original
    assert result.failed_modules == ["corefix_failing"]


def test_hook_reverted_when_module_raises_batch_path(tmp_path: Path):
    original = CaptionMetadata(text="the original caption", length=20)
    samples = [
        Sample(
            path=_media(tmp_path, f"{i}.mp4"),
            is_video=True,
            caption=original,
        )
        for i in range(2)
    ]
    recorder = _CaptionRecorderModule()
    pipeline = Pipeline([_FailingModule(), recorder])
    _add_caption_hooks(pipeline, "corefix_failing", original)
    pipeline.start()

    results = pipeline.process_samples(samples, batch_size=2)

    assert recorder.seen == ["the original caption"] * 2
    for result in results:
        assert result.caption == original
        assert result.failed_modules == ["corefix_failing"]


def test_failed_marker_survives_after_hook_returning_copy_single(tmp_path: Path):
    """An after-hook returning a fresh Sample must preserve failure markers.

    The failure marker belongs on the post-hook sample so failed samples remain
    incomplete and eligible for retry.
    """
    media = _media(tmp_path)
    module = _FailingModule()
    pipeline = Pipeline([module])
    # After-hook returns a NEW Sample that carries no failure marker.
    pipeline.add_hook(
        "corefix_failing",
        after=lambda item: item.model_copy(update={"failed_modules": []}),
    )
    pipeline.start()

    result = pipeline.process_sample(Sample(path=media, is_video=True))

    assert result.failed_modules == ["corefix_failing"]
    assert any(i.issue_type == "module_error" for i in result.validation_issues)
    # And the incomplete result must not be served from cache.
    pipeline.process_sample(Sample(path=media, is_video=True))
    assert module.calls == 2


def test_failed_marker_survives_after_hook_returning_copy_batch(tmp_path: Path):
    """The batch after-hook replacement must preserve the failure marker."""
    samples = [
        Sample(path=_media(tmp_path, f"{i}.mp4"), is_video=True) for i in range(2)
    ]
    module = _FailingModule()
    pipeline = Pipeline([module])
    pipeline.add_hook(
        "corefix_failing",
        after=lambda item: item.model_copy(update={"failed_modules": []}),
    )
    pipeline.start()

    results = pipeline.process_samples(samples, batch_size=2)

    for result in results:
        assert result.failed_modules == ["corefix_failing"]
        assert any(i.issue_type == "module_error" for i in result.validation_issues)


def test_raising_hook_does_not_crash_pipeline(tmp_path: Path):
    media = _media(tmp_path)
    module = _BackendProbeModule()
    pipeline = Pipeline([module])

    def bad_hook(sample):
        raise ValueError("hook exploded")

    pipeline.add_hook("corefix_backend_probe", before=bad_hook)
    pipeline.start()

    result = pipeline.process_sample(Sample(path=media, is_video=True))

    assert result is not None
    assert module.calls == 0  # module skipped when its before-hook fails
    assert result.failed_modules == ["corefix_backend_probe"]
    assert any(issue.issue_type == "module_error" for issue in result.validation_issues)


def test_mount_failure_marks_samples_incomplete(tmp_path: Path):
    class MountFailureModule(PipelineModule):
        name = "corefix_mount_failure"
        description = "Test mount failure reporting"

        def on_mount(self) -> None:
            raise RuntimeError("weights unavailable")

        def process(self, sample: Sample) -> Sample:
            raise AssertionError("unmounted module must not execute")

    media = _media(tmp_path)
    pipeline = Pipeline([MountFailureModule()])
    pipeline.start()

    result = pipeline.process_sample(Sample(path=media, is_video=True))

    assert result.failed_modules == ["corefix_mount_failure"]
    assert pipeline.get_run_status()["complete"] is False
    assert "mount failed" in pipeline.get_run_status()["module_failures"][
        "corefix_mount_failure"
    ]


def test_post_process_failure_marks_run_and_samples_incomplete(tmp_path: Path):
    class PostProcessFailureModule(PipelineModule):
        name = "corefix_post_process_failure"
        description = "Test post-process failure reporting"

        def process(self, sample: Sample) -> Sample:
            return sample

        def post_process(self, all_samples):
            raise RuntimeError("aggregate exploded")

    media = _media(tmp_path)
    pipeline = Pipeline([PostProcessFailureModule()])
    pipeline.start()
    result = pipeline.process_sample(Sample(path=media, is_video=True))
    pipeline.stop()

    assert result.failed_modules == ["corefix_post_process_failure"]
    assert pipeline.get_run_status()["complete"] is False
    assert "post_process failed" in pipeline.get_run_status()["module_failures"][
        "corefix_post_process_failure"
    ]


def test_raising_before_hook_marks_every_batched_sample_incomplete(tmp_path: Path):
    samples = [
        Sample(path=_media(tmp_path, f"{index}.mp4"), is_video=True)
        for index in range(2)
    ]
    module = _BackendProbeModule()
    pipeline = Pipeline([module])

    def bad_hook(sample):
        raise ValueError("batch hook exploded")

    pipeline.add_hook("corefix_backend_probe", before=bad_hook)
    pipeline.start()

    results = pipeline.process_samples(samples, batch_size=2)

    assert module.calls == 0
    assert all(result.failed_modules == ["corefix_backend_probe"] for result in results)


# ---------------------------------------------------------------------------
# 4. post_process() issues are reflected in final stats + resume parity
# ---------------------------------------------------------------------------


def test_post_process_issues_enter_final_stats(tmp_path: Path):
    media = _media(tmp_path)
    pipeline = Pipeline([_PostIssueModule()])
    pipeline.start()
    pipeline.process_sample(Sample(path=media, is_video=True))

    # Before stop(): the post_process issue does not exist yet.
    assert pipeline.stats.valid_samples == 1
    assert pipeline.stats.invalid_samples == 0

    pipeline.stop()

    assert pipeline.stats.total_samples == 1
    assert pipeline.stats.valid_samples == 0
    assert pipeline.stats.invalid_samples == 1
    assert pipeline.stats.issues_by_type.get("post_selection") == 1
    assert pipeline.stats.severity_distribution.get("error") == 1


def test_post_process_stats_match_after_resume(tmp_path: Path):
    media = _media(tmp_path)
    state_path = tmp_path / "state.json"
    pipeline = Pipeline([_PostIssueModule()])
    pipeline.start()
    pipeline.process_sample(Sample(path=media, is_video=True))
    pipeline.stop()
    pipeline.save_state(state_path)

    restored = Pipeline([_PostIssueModule()])
    restored.load_state(state_path)

    assert restored.stats.total_samples == pipeline.stats.total_samples
    assert restored.stats.valid_samples == pipeline.stats.valid_samples
    assert restored.stats.invalid_samples == pipeline.stats.invalid_samples
    assert restored.stats.issues_by_type == pipeline.stats.issues_by_type
    assert (
        restored.stats.severity_distribution == pipeline.stats.severity_distribution
    )


def test_on_dispose_validity_flip_reflected_in_final_stats(tmp_path: Path):
    """Validity changes in on_dispose must be folded into exported stats.

    Sample-derived statistics are rebuilt after on_dispose so validity changes
    are reflected in the final aggregate counts.
    """
    media = _media(tmp_path)
    pipeline = Pipeline([_DisposeInvalidatorModule()])
    pipeline.start()
    pipeline.process_sample(Sample(path=media, is_video=True))

    # Before stop(): the on_dispose issue does not exist yet.
    assert pipeline.stats.valid_samples == 1
    assert pipeline.stats.invalid_samples == 0

    pipeline.stop()

    assert pipeline.stats.total_samples == 1
    assert pipeline.stats.valid_samples == 0
    assert pipeline.stats.invalid_samples == 1
    assert pipeline.stats.issues_by_type.get("duplicate") == 1
    assert pipeline.stats.severity_distribution.get("error") == 1


def test_save_state_tolerates_non_json_metadata(tmp_path: Path):
    """A non-JSON value in Sample.metadata must not abort the whole save."""
    media = _media(tmp_path)
    state_path = tmp_path / "state.json"
    pipeline = Pipeline([_BackendProbeModule()])
    pipeline.start()
    result = pipeline.process_sample(Sample(path=media, is_video=True))

    # A module stashed non-JSON-serializable values in metadata.
    result.metadata["arr"] = np.arange(3)
    result.metadata["scalar"] = np.float32(1.5)
    result.metadata["where"] = tmp_path / "sub"
    result.metadata["layout"] = "2x2"  # valid metadata must be preserved

    pipeline.save_state(state_path)

    assert state_path.exists()  # the whole save was NOT aborted
    data = json.loads(state_path.read_text(encoding="utf-8"))
    meta = data["results"][str(media)]["metadata"]
    assert meta["arr"] == [0, 1, 2]
    assert meta["scalar"] == 1.5
    assert meta["layout"] == "2x2"


def test_save_state_round_trips_non_json_metadata(tmp_path: Path):
    """Coerced metadata still loads back into a valid resumable state."""
    media = _media(tmp_path)
    state_path = tmp_path / "state.json"
    pipeline = Pipeline([_BackendProbeModule()])
    pipeline.start()
    result = pipeline.process_sample(Sample(path=media, is_video=True))
    result.metadata["scalar"] = np.int64(7)
    pipeline.save_state(state_path)

    resumed = Pipeline([_BackendProbeModule()])
    resumed.load_state(state_path)
    restored = resumed.results[str(media)]
    assert restored.metadata["scalar"] == 7


# ---------------------------------------------------------------------------
# 5. cache_enabled config wiring + aggregation
# ---------------------------------------------------------------------------


def test_runtime_module_config_propagates_cache_enabled():
    from types import SimpleNamespace

    from ayase.config import GeneralConfig
    from ayase.runtime import runtime_module_config

    disabled = runtime_module_config(
        SimpleNamespace(general=GeneralConfig(cache_enabled=False))
    )
    enabled = runtime_module_config(
        SimpleNamespace(general=GeneralConfig(cache_enabled=True))
    )
    assert disabled["cache_enabled"] is False
    assert enabled["cache_enabled"] is True


def test_global_cache_disable_disables_result_cache(tmp_path: Path):
    media = _media(tmp_path)
    # Global config off => every module receives cache_enabled=False.
    module = _BackendProbeModule({"cache_enabled": False})
    pipeline = Pipeline([module])
    assert pipeline._cache_enabled is False
    pipeline.start()

    pipeline.process_sample(Sample(path=media, is_video=True))
    pipeline.process_sample(Sample(path=media, is_video=True))

    assert module.calls == 2  # nothing served from the result cache


def test_single_module_opt_out_does_not_disable_global_cache(tmp_path: Path):
    media = _media(tmp_path)
    m1 = _BackendProbeModule()
    m2 = _NoBackendModule({"cache_enabled": False})
    pipeline = Pipeline([m1, m2])

    # One module opting out must NOT nuke caching for the whole pipeline.
    assert pipeline._cache_enabled is True
    assert pipeline._frame_cache_enabled is True

    pipeline.start()
    pipeline.process_sample(Sample(path=media, is_video=True))
    pipeline.process_sample(Sample(path=media, is_video=True))
    assert m1.calls == 1


# ---------------------------------------------------------------------------
# 6. default_config deep copy + recursive merge
# ---------------------------------------------------------------------------


def test_default_config_nested_dicts_not_shared():
    m1 = _NestedConfigModule()
    m1.config["weights"]["a"] = 99.0

    # Class attribute untouched.
    assert _NestedConfigModule.default_config["weights"]["a"] == 1.0
    # Future instances unaffected.
    m2 = _NestedConfigModule()
    assert m2.config["weights"]["a"] == 1.0
    assert m2.config["weights"] is not m1.config["weights"]


def test_config_override_merges_nested_dicts_keywise():
    m = _NestedConfigModule(config={"weights": {"b": 5.0}})
    assert m.config["weights"] == {"a": 1.0, "b": 5.0}
    assert m.config["flag"] is True
    # Non-dict override replaces wholesale.
    m2 = _NestedConfigModule(config={"weights": "off"})
    assert m2.config["weights"] == "off"


# ---------------------------------------------------------------------------
# 7. Stable external plugin module names (resume-safe fingerprints)
# ---------------------------------------------------------------------------


def test_external_plugin_module_name_is_stable_digest(tmp_path: Path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    plugin_file = plugin_dir / "corefix_plugin.py"
    plugin_file.write_text(
        "from ayase.pipeline import PipelineModule\n"
        "from ayase.models import Sample\n"
        "\n"
        "class CorefixExtModule(PipelineModule):\n"
        "    name = 'corefix_ext_plugin'\n"
        "    description = 'external plugin probe'\n"
        "\n"
        "    def process(self, sample):\n"
        "        return sample\n",
        encoding="utf-8",
    )
    try:
        ModuleRegistry.discover_external_modules([plugin_dir])
        file_key = str(plugin_file.resolve())
        # The digest is a pure function of the path — identical in any process
        # (unlike builtin hash(), which is per-process randomized).
        expected = hashlib.sha1(file_key.encode("utf-8")).hexdigest()[:10]
        module_name = ModuleRegistry._external_plugin_modules[file_key]
        assert module_name == f"ayase_ext_corefix_plugin_{expected}"
        assert ModuleRegistry.get_module("corefix_ext_plugin") is not None
    finally:
        plugin_file.unlink(missing_ok=True)
        ModuleRegistry.discover_external_modules([plugin_dir])

    # Re-discovery after deletion fully unregisters the plugin.
    assert ModuleRegistry.get_module("corefix_ext_plugin") is None


def test_external_plugin_symlink_cannot_escape_configured_folder(tmp_path: Path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    outside = tmp_path / "outside_plugin.py"
    outside.write_text(
        "from ayase.pipeline import PipelineModule\n"
        "class EscapedPlugin(PipelineModule):\n"
        "    name = 'corefix_escaped_plugin'\n"
        "    def process(self, sample):\n"
        "        return sample\n",
        encoding="utf-8",
    )
    link = plugin_dir / "escaped.py"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("File symlinks are unavailable on this platform")

    try:
        ModuleRegistry.discover_external_modules([plugin_dir])
        assert ModuleRegistry.get_module("corefix_escaped_plugin") is None
        readiness = ModuleRegistry.readiness_report()
        readiness_key = str(link.resolve())
        assert readiness[readiness_key]["status"] == "missing"
        assert "outside configured folder" in readiness[readiness_key]["error"]
    finally:
        link.unlink(missing_ok=True)
        ModuleRegistry.discover_external_modules([plugin_dir])


# ---------------------------------------------------------------------------
# 8. Frame cache: exact sampling grids, lazy color, read-only
# ---------------------------------------------------------------------------


def _install_fake_decoder(monkeypatch, total_frames: int = 8, failed_frames=(), failed_seeks=()):
    import ayase.image as image_utils

    opens = []

    class Capture:
        def __init__(self, path):
            self.position = 0
            self.reads = []
            opens.append(self.reads)

        def isOpened(self):
            return True

        def get(self, prop):
            return total_frames

        def set(self, prop, position):
            if position in failed_seeks:
                return False
            self.position = position
            return True

        def read(self):
            index = self.position
            self.position += 1
            self.reads.append(index)
            if index in failed_frames or (total_frames <= 0 and index >= 8):
                return False, None
            return True, np.full((4, 4, 3), index, dtype=np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(image_utils.cv2, "VideoCapture", Capture)
    return opens


def test_frame_cache_shares_decode_across_colors_for_same_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = _install_fake_decoder(monkeypatch)
    media = _media(tmp_path)
    pipeline = Pipeline([])

    first = pipeline.sample_frames(media, max_frames=8, color="rgb")
    gray = pipeline.sample_frames(media, max_frames=8, color="gray")
    again = pipeline.sample_frames(media, max_frames=8, color="rgb")

    assert len(calls) == 1  # one exact grid serves all color requests
    assert calls[0] == list(range(8))
    assert len(first) == 8
    assert len(gray) == 8
    assert gray[0].ndim == 2  # converted to gray lazily
    assert len(again) == 8
    # Same pixel data, fresh view objects.
    assert again[0] is not first[0]
    assert np.shares_memory(again[0], first[0])


def test_frame_cache_returns_read_only_views(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _install_fake_decoder(monkeypatch)
    media = _media(tmp_path)
    pipeline = Pipeline([])

    frames = pipeline.sample_frames(media, max_frames=4, color="rgb")

    assert frames and frames[0].flags.writeable is False
    with pytest.raises((ValueError, RuntimeError)):
        frames[0][0, 0, 0] = 9


def test_frame_cache_reads_only_new_indices_for_higher_max_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = _install_fake_decoder(monkeypatch)
    media = _media(tmp_path)
    pipeline = Pipeline([])

    low = pipeline.sample_frames(media, max_frames=3, color="rgb")
    high = pipeline.sample_frames(media, max_frames=8, color="rgb")

    assert len(low) == 3
    assert len(high) == 8
    assert calls == [[0, 3, 7], [1, 2, 4, 5, 6]]


def test_frame_cache_reuses_identical_grids_for_short_videos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = _install_fake_decoder(monkeypatch, total_frames=2)
    media = _media(tmp_path)
    pipeline = Pipeline([])

    first = pipeline.sample_frames(media, max_frames=4, color="rgb")
    second = pipeline.sample_frames(media, max_frames=8, color="rgb")

    assert len(first) == 2
    assert len(second) == 2
    assert calls == [[0, 1]]


def test_frame_cache_sampling_is_order_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A smaller grid has the same pixels standalone or after a larger grid."""
    calls = _install_fake_decoder(monkeypatch, total_frames=100)
    media = _media(tmp_path)

    standalone = Pipeline([]).sample_frames(media, max_frames=3, color="bgr")
    ordered_pipeline = Pipeline([])
    ordered_pipeline.sample_frames(media, max_frames=8, color="rgb")
    ordered = ordered_pipeline.sample_frames(media, max_frames=3, color="bgr")

    assert [int(frame[0, 0, 0]) for frame in standalone] == [0, 49, 99]
    assert [int(frame[0, 0, 0]) for frame in ordered] == [0, 49, 99]
    assert calls == [[0, 49, 99], [0, 14, 28, 42, 56, 70, 84, 99], [49]]


@pytest.mark.parametrize("order", [(5, 8, 16, 32), (32, 16, 8, 5)])
def test_exact_frame_grids_decode_each_successful_index_once(tmp_path, monkeypatch, order):
    from ayase.image import _sample_frames_uncached

    calls = _install_fake_decoder(monkeypatch, total_frames=101)
    pipeline = Pipeline([])
    media = _media(tmp_path)
    reference = {
        limit: _sample_frames_uncached(media, max_frames=limit, color="rgb")
        for limit in order
    }
    calls.clear()
    for limit in order:
        frames = pipeline.sample_frames(media, max_frames=limit, color="rgb")
        np.testing.assert_array_equal(frames, reference[limit])
    middle = pipeline.load_representative_frame(media, color="gray")
    assert int(middle[0, 0]) == 50
    decoded = [index for opened in calls for index in opened]
    expected = {50}
    for limit in order:
        expected.update(np.linspace(0, 100, limit, dtype=int))
    assert len(decoded) == len(expected)
    assert set(decoded) == expected


def test_frame_cache_bounds_converted_pixels_and_returns_readonly_views(tmp_path, monkeypatch):
    calls = _install_fake_decoder(monkeypatch, total_frames=8)
    pipeline = Pipeline([])
    # A BGR frame and its RGB conversion each occupy 48 bytes.
    pipeline._frame_cache.max_bytes = 96
    media = _media(tmp_path)
    frames = pipeline.sample_frames(media, max_frames=3, color="rgb")
    assert pipeline._frame_cache.bytes_used <= 96
    assert [int(frame[0, 0, 0]) for frame in frames] == [0, 3, 7]
    assert pipeline.sample_frames(media, max_frames=1, color="gray")[0].shape == (4, 4)
    assert pipeline._frame_cache.bytes_used <= 96
    assert calls == [[0, 3, 7], [0]]
    with pytest.raises(ValueError):
        frames[0].setflags(write=True)


def test_failed_seek_replays_grid_without_mixing_decoder_positions(tmp_path, monkeypatch):
    from ayase.image import _sample_frames_uncached

    _install_fake_decoder(monkeypatch, total_frames=9, failed_seeks={4})
    pipeline = Pipeline([])
    media = _media(tmp_path)
    pipeline.sample_frames(media, max_frames=1, color="rgb")

    frames = pipeline.sample_frames(media, max_frames=3, color="rgb")
    expected = _sample_frames_uncached(media, max_frames=3, color="rgb")
    assert [int(frame[0, 0, 0]) for frame in expected] == [0, 1, 8]
    np.testing.assert_array_equal(frames, expected)


@pytest.mark.parametrize("cache_enabled", [True, False])
def test_decord_batch_respects_grid_and_channel_order(tmp_path, monkeypatch, cache_enabled):
    import sys
    from types import ModuleType

    calls = []

    class Reader:
        def __init__(self, path, *, fault_tol):
            assert 0 < fault_tol < 1 / 9

        def __len__(self):
            return 9

        def get_batch(self, indices):
            calls.append(indices)
            return np.asarray([[[[index, 12, 34]]] for index in indices], dtype=np.uint8)

    decord = ModuleType("decord")
    decord.VideoReader = Reader
    monkeypatch.setitem(sys.modules, "decord", decord)

    class DecoderProbe(PipelineModule):
        def process(self, sample):
            return sample

    pipeline = Pipeline([DecoderProbe({
        "video_decoder": "decord", "frame_cache_enabled": cache_enabled,
    })])
    media = _media(tmp_path)
    assert [frame[0, 0].tolist() for frame in pipeline.sample_frames(media, 3)] == [
        [0, 12, 34], [4, 12, 34], [8, 12, 34],
    ]
    representative = pipeline.load_representative_frame(media, "bgr")
    assert representative[0, 0].tolist() == [34, 12, 4]
    assert not representative.flags.writeable
    assert calls == ([[0, 4, 8]] if cache_enabled else [[0, 4, 8], [4]])


def test_decoder_choice_is_part_of_numeric_state(tmp_path):
    class DecoderProbe(PipelineModule):
        def process(self, sample):
            return sample

    def pipeline(decoder):
        return Pipeline([DecoderProbe({"video_decoder": decoder})])

    opencv = pipeline("opencv")
    auto = pipeline("auto")
    decord = pipeline("decord")
    assert auto._pipeline_fingerprint() == opencv._pipeline_fingerprint()
    assert decord._pipeline_fingerprint()["video_decoder"] == "decord"
    path = _media(tmp_path)
    opencv.process_sample(Sample(path=path, is_video=True))
    checkpoint = tmp_path / "decoder-state.json"
    opencv.save_state(checkpoint)
    auto.load_state(checkpoint)
    decord.load_state(checkpoint)
    assert set(auto.results) == {str(path)}
    assert not decord.results


@pytest.mark.parametrize("unreliable_count", [False, True])
def test_frame_cache_video_sampling_matches_standalone_with_decoder_edge_cases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unreliable_count: bool,
):
    """Exercise the real sampler for failed seeks and unknown frame counts."""
    import ayase.image as image_utils

    class EdgeCaseCapture:
        def __init__(self, path):
            self.position = 0
            self.reads = 0

        def isOpened(self):
            return True

        def get(self, prop):
            return 0.0 if unreliable_count else 100.0

        def set(self, prop, value):
            self.position = int(value)
            return True

        def read(self):
            if unreliable_count:
                if self.reads >= 100:
                    return False, None
                value = self.reads
                self.reads += 1
            else:
                value = self.position
                # Simulate intermittent seek/read failures on the eight-frame
                # grid while leaving the three-frame grid decodable.
                if value in {28, 70}:
                    return False, None
            return True, np.full((2, 2, 3), value, dtype=np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(image_utils.cv2, "VideoCapture", EdgeCaseCapture)
    media = _media(tmp_path)

    standalone = Pipeline([]).sample_frames(media, max_frames=3, color="bgr")
    ordered_pipeline = Pipeline([])
    ordered_pipeline.sample_frames(media, max_frames=8, color="rgb")
    ordered = ordered_pipeline.sample_frames(media, max_frames=3, color="bgr")

    standalone_pixels = [int(frame[0, 0, 0]) for frame in standalone]
    ordered_pixels = [int(frame[0, 0, 0]) for frame in ordered]
    expected = [0, 4, 8] if unreliable_count else [0, 49, 99]
    assert standalone_pixels == expected
    assert ordered_pixels == expected


def test_frame_cache_short_read_not_treated_as_exhausted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A short read from failed seeks on a LONG video must not cache exhausted.

    A cache entry is exhausted only when the decoder confirms the source has no
    additional frames; failed seeks do not establish exhaustion.
    """
    calls = _install_fake_decoder(monkeypatch, total_frames=100, failed_frames={33, 28, 99})
    media = _media(tmp_path)
    pipeline = Pipeline([])

    first = pipeline.sample_frames(media, max_frames=4, color="rgb")
    second = pipeline.sample_frames(media, max_frames=8, color="rgb")

    assert [int(frame[0, 0, 0]) for frame in first] == [0, 66]
    assert [int(frame[0, 0, 0]) for frame in second] == [0, 14, 42, 56, 70, 84]
    assert calls == [[0, 33, 66, 99], [14, 28, 42, 56, 70, 84, 99]]


def test_frame_cache_unreliable_count_allows_redecode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """When the decoder frame count is unreliable, a larger request re-attempts."""
    calls = _install_fake_decoder(monkeypatch, total_frames=0)
    media = _media(tmp_path)
    pipeline = Pipeline([])

    first = pipeline.sample_frames(media, max_frames=4, color="rgb")
    second = pipeline.sample_frames(media, max_frames=8, color="rgb")

    assert calls == [list(range(9)), list(range(9))]
    assert [int(frame[0, 0, 0]) for frame in first] == [0, 4]
    assert [int(frame[0, 0, 0]) for frame in second] == [0, 4]


@pytest.mark.parametrize("options", [{"known_count_only": True}, {"floor_stride": True}])
def test_known_count_only_does_not_read_sequential_fallback(tmp_path, monkeypatch, options):
    from ayase.image import sample_frames

    calls = _install_fake_decoder(monkeypatch, total_frames=0)
    media = _media(tmp_path)
    assert sample_frames(media, 4, **options) == []
    assert Pipeline([]).sample_frames(media, 4, **options) == []
    assert calls == [[], []]


def test_floor_stride_and_uniform_grids_share_only_matching_indices(tmp_path, monkeypatch):
    from ayase.image import sample_frames

    calls = _install_fake_decoder(monkeypatch, total_frames=10)
    media = _media(tmp_path)
    expected = sample_frames(media, 4, color="bgr", floor_stride=True)
    calls.clear()
    pipeline = Pipeline([])
    uniform = pipeline.sample_frames(media, 4, color="bgr")
    stride = pipeline.sample_frames(media, 4, color="bgr", floor_stride=True)
    np.testing.assert_array_equal(stride, expected)
    assert [int(frame[0, 0, 0]) for frame in uniform] == [0, 3, 6, 9]
    assert [int(frame[0, 0, 0]) for frame in stride] == [0, 2, 4, 6]
    assert calls == [[0, 3, 6, 9], [2, 4]]


def test_floor_stride_decodes_explicit_video_with_uncommon_suffix(tmp_path, monkeypatch):
    from ayase.image import sample_frames

    _install_fake_decoder(monkeypatch, total_frames=10)
    media = _media(tmp_path).with_suffix(".mpeg")
    media.write_bytes(b"video")
    for frames in (
        sample_frames(media, 4, color="bgr", floor_stride=True),
        Pipeline([]).sample_frames(media, 4, color="bgr", floor_stride=True),
    ):
        assert [int(frame[0, 0, 0]) for frame in frames] == [0, 2, 4, 6]


@pytest.mark.parametrize("size", [1, 2])
def test_prefetch_overlaps_one_batch_and_keeps_pipeline_on_owner(tmp_path, monkeypatch, size):
    from threading import Event, get_ident

    from ayase import image
    from ayase.cli import _process_samples

    owner = get_ident()
    next_batch_started = Event()
    original = image._prefetch_frames
    opens = _install_fake_decoder(monkeypatch, total_frames=9)
    yielded = []
    hooks = []
    checkpoints = []
    fail_first = False

    def decode(path, *args):
        assert get_ident() != owner
        if fail_first and path.stem == "0":
            raise RuntimeError("simulated prefetch failure")
        if path.stem == str(size):
            next_batch_started.set()
        return original(path, *args)

    monkeypatch.setattr(image, "_prefetch_frames", decode)

    class PrefetchProbe(PipelineModule):
        def frame_prefetch_limits(self):
            return (3, 4, None)

        def process(self, sample):
            assert get_ident() == owner
            if sample.path.stem == "0" and self.config["prefetch_workers"]:
                assert next_batch_started.wait(5)
                assert yielded == list(range(2 * size))
            first = image.sample_frames(sample.path, 3)
            second = image.sample_frames(sample.path, 4)
            middle = image.load_representative_frame(sample.path)
            sample.metadata["decode"] = [
                [int(frame[0, 0, 0]) for frame in first],
                [int(frame[0, 0, 0]) for frame in second],
                int(middle[0, 0, 0]),
            ]
            return sample

    paths = [_media(tmp_path, f"{index}.mp4") for index in range(5)]

    def run(workers):
        module = PrefetchProbe({"prefetch_workers": workers, "sample_batch_size": size})
        module._mounted = True
        pipeline = Pipeline([module])

        def before(sample):
            assert get_ident() == owner
            hooks.append(sample.path)
            return sample

        pipeline.add_hook(module.name, before=before)

        def samples():
            for index, path in enumerate(paths):
                yielded.append(index)
                yield Sample(path=path, is_video=True)

        count = _process_samples(
            pipeline, samples(), checkpoint_every=2,
            checkpoint_fn=lambda p, count: checkpoints.append((count, len(p.results))),
        )
        assert count == len(paths)
        assert hooks == paths
        assert checkpoints == [(2, 2), (4, 4), (5, 5)]
        return [sample.canonical_model_dump(mode="json") for sample in pipeline.results.values()]

    expected = run(0)
    opens.clear()
    yielded.clear()
    hooks.clear()
    checkpoints.clear()
    assert run(2) == expected
    assert opens == [[0, 2, 4, 5, 8]] * len(paths)
    yielded.clear()
    hooks.clear()
    checkpoints.clear()
    next_batch_started.clear()
    fail_first = True
    assert run(2) == expected


def test_sample_frames_detailed_reports_frame_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """``_sample_frames_uncached_detailed`` reports a reliable total, or None."""
    import ayase.image as image_utils

    class _CountCapture(_FakeZeroCountCapture):
        def get(self, prop):
            return 12.0  # reliable CAP_PROP_FRAME_COUNT

    monkeypatch.setattr(image_utils.cv2, "VideoCapture", _CountCapture)
    media = _media(tmp_path, "clip.mp4")
    frames, total = image_utils._sample_frames_uncached_detailed(
        media, max_frames=4, color="rgb"
    )
    assert total == 12  # decoder-reported count, not len(frames)
    assert len(frames) == 4

    # Zero/unreliable count => None (sequential fallback path).
    monkeypatch.setattr(image_utils.cv2, "VideoCapture", _FakeZeroCountCapture)
    media2 = _media(tmp_path, "broken.webm")
    frames2, total2 = image_utils._sample_frames_uncached_detailed(
        media2, max_frames=4, color="rgb"
    )
    assert total2 is None


def test_uncached_sample_frames_are_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The uncached fallback path returns read-only frames.

    The contract must be identical to the pipeline (cached) path so a module
    that mutates a returned frame in place fails in tests, not only in prod.
    """
    import ayase.image as image_utils

    def fake_decode(path, max_frames=8, color="rgb"):
        return [np.full((4, 4, 3), 5, dtype=np.uint8) for _ in range(int(max_frames))]

    monkeypatch.setattr(image_utils, "_sample_frames_uncached", fake_decode)
    monkeypatch.setattr(
        image_utils,
        "_load_representative_frame_uncached",
        lambda path, color="rgb": np.full((4, 4, 3), 5, dtype=np.uint8),
    )

    # No pipeline active => uncached fallback path is taken.
    frames = image_utils.sample_frames(tmp_path / "x.mp4", max_frames=3)
    assert frames and all(f.flags.writeable is False for f in frames)
    with pytest.raises((ValueError, RuntimeError)):
        frames[0][0, 0, 0] = 9

    rep = image_utils.load_representative_frame(tmp_path / "x.mp4")
    assert rep is not None and rep.flags.writeable is False


def test_representative_frame_cached_and_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = _install_fake_decoder(monkeypatch, total_frames=9)
    media = _media(tmp_path)
    pipeline = Pipeline([])

    rgb = pipeline.load_representative_frame(media, color="rgb")
    gray = pipeline.load_representative_frame(media, color="gray")
    rgb2 = pipeline.load_representative_frame(media, color="rgb")
    sampled = pipeline.sample_frames(media, max_frames=3, color="rgb")

    assert calls == [[4], [0, 8]]
    assert rgb is not None and rgb.flags.writeable is False
    assert gray is not None and gray.ndim == 2
    assert rgb2 is not rgb and np.shares_memory(rgb2, rgb)
    assert np.shares_memory(sampled[1], rgb)
    with pytest.raises((ValueError, RuntimeError)):
        rgb[0, 0, 0] = 1


# ---------------------------------------------------------------------------
# 9. Zero-frame-count fallback (webm/pipe-muxed files)
# ---------------------------------------------------------------------------


class _FakeZeroCountCapture:
    """VideoCapture stand-in that reports frame_count=0 but yields frames."""

    frames_available = 100

    def __init__(self, path):
        self.reads = 0

    def isOpened(self):
        return True

    def get(self, prop):
        return 0.0  # CAP_PROP_FRAME_COUNT <= 0

    def set(self, *args):
        return True

    def read(self):
        if self.reads >= self.frames_available:
            return False, None
        self.reads += 1
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def release(self):
        pass


def test_zero_frame_count_video_falls_back_to_sequential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import ayase.image as image_utils

    monkeypatch.setattr(
        image_utils.cv2, "VideoCapture", _FakeZeroCountCapture
    )
    media = _media(tmp_path, "broken.webm")

    frames = image_utils._sample_frames_uncached(media, max_frames=5, color="rgb")

    assert len(frames) == 5
    assert all(f.shape == (4, 4, 3) for f in frames)


def test_zero_frame_count_fallback_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import ayase.image as image_utils

    class _ShortCapture(_FakeZeroCountCapture):
        frames_available = 6

    monkeypatch.setattr(image_utils.cv2, "VideoCapture", _ShortCapture)
    media = _media(tmp_path, "short.webm")

    frames = image_utils._sample_frames_uncached(media, max_frames=5, color="gray")

    # stride=4 over 6 available frames keeps indices 0 and 4.
    assert len(frames) == 2
    assert all(f.ndim == 2 for f in frames)


# ---------------------------------------------------------------------------
# 10. issues_by_type keys
# ---------------------------------------------------------------------------


def _issue(message: str, issue_type=None) -> ValidationIssue:
    return ValidationIssue(
        severity=ValidationSeverity.WARNING,
        message=message,
        issue_type=issue_type,
    )


def test_issue_type_key_prefers_structured_type():
    issue = _issue("C:\\data\\clip.mp4: too dark", issue_type="too_dark")
    assert Pipeline._issue_type_key(issue) == "too_dark"


def test_issue_type_key_does_not_split_windows_paths():
    key = Pipeline._issue_type_key(_issue("C:\\data\\clip.mp4: too dark"))
    assert key != "C"
    key2 = Pipeline._issue_type_key(_issue("/mnt/data/clip.mp4: too dark"))
    assert key2 != "/mnt/data/clip.mp4"[:1]
    assert "/" not in key2 or len(key2) > 1


def test_issue_type_key_uses_clean_message_prefix():
    assert (
        Pipeline._issue_type_key(_issue("too_dark: luminance 12 below 20"))
        == "too_dark"
    )
    assert Pipeline._issue_type_key(_issue("no colon here")) == "no colon here"


def test_module_error_issues_counted_by_structured_type(tmp_path: Path):
    media = _media(tmp_path)
    pipeline = Pipeline([_FailingModule()])
    pipeline.start()
    pipeline.process_sample(Sample(path=media, is_video=True))

    assert pipeline.stats.issues_by_type.get("module_error") == 1
    assert pipeline.stats.severity_distribution.get("warning") == 1


@pytest.mark.parametrize(
    ("module_name", "resource"),
    [
        ("adaface", "_net"),
        ("fad", "_loader"),
        ("prdc_dinov2", "_processor"),
        ("stream_metric", "_stream_instance"),
        ("umap_projection", "_clip_processor"),
    ],
)
def test_custom_dispose_releases_backend_on_empty_run(module_name, resource):
    module_class = ModuleRegistry.get_module(module_name)
    module = module_class()
    module._mounted = True
    setattr(module, resource, object())

    module.on_dispose()

    assert not module._mounted
    assert getattr(module, resource) is None


def test_entity_dispose_releases_group_state_on_error(monkeypatch):
    from ayase.modules.entity_consistency import EntityBenchModule

    module = EntityBenchModule()
    module._mounted = True
    module._face_model = object()
    module._groups["group"] = [object(), object()]

    def fail(_items):
        raise RuntimeError("group failed")

    monkeypatch.setattr(module, "_identity_consistency", fail)
    with pytest.raises(RuntimeError, match="group failed"):
        module.on_dispose()
    assert not module._mounted
    assert not module._groups
    assert module._face_model is None
@pytest.mark.parametrize(("count", "finalized"), [(3, False), (4, True)])
def test_resume_rebuilds_fid_and_fvd_feature_cohort(tmp_path, monkeypatch, count, finalized):
    import numpy as np

    from ayase.modules.fid import FIDModule
    from ayase.modules.fvd import FVDModule

    for i in range(4):
        for part in ("gen", "ref"):
            (tmp_path / f"{part}-{i}.mp4").touch()

    def features(sample):
        part, index = sample.path.stem.split("-")
        i = float(index)
        return np.array([i if part == "ref" else i + i**2, i**2, i**3])

    def samples():
        return [
            Sample(path=tmp_path / f"gen-{i}.mp4", reference_path=tmp_path / f"ref-{i}.mp4",
                   is_video=True)
            for i in range(4)
        ]

    def pipeline():
        fid = FIDModule({"test_mode": True})
        fvd = FVDModule({"test_mode": True})
        for module in (fid, fvd):
            monkeypatch.setattr(module, "extract_features", features)
            monkeypatch.setattr(module, "extract_reference_features", features, raising=False)
        fid._backend = "test_features"
        fvd._ml_available = True
        monkeypatch.setattr(
            fvd, "_load_video_frames",
            lambda path, total: [features(Sample(path=path, is_video=True))] * total,
        )
        monkeypatch.setattr(fvd, "_extract_i3d_batch", lambda clips: [clip[0] for clip in clips])
        return Pipeline([fid, fvd], allow_provenance=["published", "adapted", "own"])

    expected = pipeline()
    expected.start()
    expected.process_samples(samples(), batch_size=2)
    expected.stop()

    interrupted = pipeline()
    interrupted.start()
    interrupted.process_samples(samples()[:count], batch_size=2)
    checkpoint = tmp_path / "state.json"
    if finalized:
        interrupted.stop()
    interrupted.save_state(checkpoint)
    if not finalized:
        interrupted.stop()

    resumed = pipeline()
    resumed.load_state(checkpoint)
    assert len(resumed.results) == count
    resumed.start()
    assert resumed.stats.fid is None and resumed.stats.fvd is None
    resumed.process_samples(samples(), batch_size=2)
    assert [len(module._feature_cache) for module in resumed.modules] == [4, 4]
    resumed.stop()

    assert resumed.stats.fid == expected.stats.fid == pytest.approx(28.5635797968)
    assert resumed.stats.fvd == expected.stats.fvd == pytest.approx(28.5635797968)
