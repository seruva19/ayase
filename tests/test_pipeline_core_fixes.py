"""Focused tests for the core pipeline review fixes.

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
8.  Frame cache: decode-once per file, subsampled views, lazy color
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
        sample.quality_metrics.technical_score = 1.0
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
        sample.quality_metrics.technical_score = 2.0
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


# ---------------------------------------------------------------------------
# 8. Frame cache: decode once, subsampled views, lazy color, read-only
# ---------------------------------------------------------------------------


def _install_fake_decoder(monkeypatch, total_frames: int = 8):
    import ayase.image as image_utils

    calls = []

    def fake_decode(path, max_frames=8, color="rgb"):
        calls.append((int(max_frames), str(color)))
        n = min(int(max_frames), total_frames)
        return [
            np.full((4, 4, 3), fill_value=i, dtype=np.uint8) for i in range(n)
        ]

    monkeypatch.setattr(image_utils, "_sample_frames_uncached", fake_decode)
    return calls


def test_frame_cache_single_decode_across_max_frames_and_colors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = _install_fake_decoder(monkeypatch)
    media = _media(tmp_path)
    pipeline = Pipeline([])

    first = pipeline.sample_frames(media, max_frames=8, color="rgb")
    gray = pipeline.sample_frames(media, max_frames=3, color="gray")
    again = pipeline.sample_frames(media, max_frames=8, color="rgb")

    assert len(calls) == 1  # one decode serves all requests
    assert calls[0] == (8, "bgr")  # decoded once, native BGR
    assert len(first) == 8
    assert len(gray) == 3
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


def test_frame_cache_redecodes_for_higher_max_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = _install_fake_decoder(monkeypatch)
    media = _media(tmp_path)
    pipeline = Pipeline([])

    low = pipeline.sample_frames(media, max_frames=3, color="rgb")
    high = pipeline.sample_frames(media, max_frames=8, color="rgb")

    assert len(low) == 3
    assert len(high) == 8
    assert [c[0] for c in calls] == [3, 8]


def test_frame_cache_does_not_redecode_exhausted_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = _install_fake_decoder(monkeypatch, total_frames=2)
    media = _media(tmp_path)
    pipeline = Pipeline([])

    first = pipeline.sample_frames(media, max_frames=4, color="rgb")
    second = pipeline.sample_frames(media, max_frames=8, color="rgb")

    assert len(first) == 2
    assert len(second) == 2
    assert len(calls) == 1  # source exhausted; no futile re-decode


def test_representative_frame_cached_and_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import ayase.image as image_utils

    calls = {"count": 0}

    def fake_rep(path, color="rgb"):
        calls["count"] += 1
        return np.full((4, 4, 3), fill_value=7, dtype=np.uint8)

    monkeypatch.setattr(
        image_utils, "_load_representative_frame_uncached", fake_rep
    )
    media = _media(tmp_path)
    pipeline = Pipeline([])

    rgb = pipeline.load_representative_frame(media, color="rgb")
    gray = pipeline.load_representative_frame(media, color="gray")
    rgb2 = pipeline.load_representative_frame(media, color="rgb")

    assert calls["count"] == 1  # decoded once, colors converted lazily
    assert rgb is not None and rgb.flags.writeable is False
    assert gray is not None and gray.ndim == 2
    assert rgb2 is not rgb and np.shares_memory(rgb2, rgb)
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
