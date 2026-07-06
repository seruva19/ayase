"""Focused regression tests for the review-fixes branch.

Covers:
1.  AyaseConfig.save() round-trips (Path/None serialization).
2.  Bare unprefixed env vars no longer leak into config.
3.  Explicit nonexistent --config path errors instead of silent fallback.
4.  filter --min-score is float, --max-score exists, missing metric excluded.
5.  filter copy/symlink mirror directory structure; unknown mode rejected.
7.  Checkpoint cadence crosses thresholds with sample_batch_size > 1.
10. Scanner no longer adopts a same-stem caption from another directory.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from ayase.cli import app, _process_samples
from ayase.config import AyaseConfig
from ayase.models import Sample
from ayase.pipeline import Pipeline, PipelineModule
from ayase.scanner import scan_dataset


# ---------------------------------------------------------------------------
# Fix 1 — config.save() serializes Paths / drops None and round-trips.
# ---------------------------------------------------------------------------


def test_config_save_round_trips_default(tmp_path: Path):
    config_path = tmp_path / "ayase.toml"
    AyaseConfig().save(config_path)

    assert config_path.exists()
    loaded = AyaseConfig.load(config_path)
    assert loaded.model_dump() == AyaseConfig().model_dump()


def test_config_save_stringifies_paths(tmp_path: Path):
    config_path = tmp_path / "ayase.toml"
    AyaseConfig().save(config_path)

    text = config_path.read_text(encoding="utf-8")
    # Path fields must be written as strings, not repr'd WindowsPath objects.
    assert "WindowsPath" not in text
    assert "PosixPath" not in text
    assert "models" in text


# ---------------------------------------------------------------------------
# Fix 2 — bare unprefixed env vars do not leak; AYASE_* channel still works.
# ---------------------------------------------------------------------------


def test_config_ignores_bare_unprefixed_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_path = tmp_path / "ayase.toml"
    config_path.write_text("", encoding="utf-8")

    monkeypatch.setenv("DEVICE", "zzz")
    monkeypatch.setenv("PARALLEL_JOBS", "999")
    monkeypatch.setenv("MODELS_DIR", "/bogus")
    monkeypatch.setenv("CACHE_DIR", "/bogus-cache")
    monkeypatch.setenv("MODULES", "nonsense")

    cfg = AyaseConfig.load(config_path)

    assert cfg.general.device == "auto"
    assert cfg.general.parallel_jobs == 8
    assert cfg.general.models_dir == Path("models")
    assert cfg.pipeline.modules == []


def test_config_ayase_prefixed_env_still_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config_path = tmp_path / "ayase.toml"
    config_path.write_text("", encoding="utf-8")

    monkeypatch.setenv("AYASE_GENERAL__DEVICE", "cuda:5")
    cfg = AyaseConfig.load(config_path)

    assert cfg.general.device == "cuda:5"


# ---------------------------------------------------------------------------
# Fix 3 — explicit nonexistent config path errors; implicit default is lenient.
# ---------------------------------------------------------------------------


def test_config_load_explicit_missing_path_raises(tmp_path: Path):
    missing = tmp_path / "nope.toml"
    with pytest.raises(FileNotFoundError):
        AyaseConfig.load(missing)


def test_config_load_implicit_default_is_lenient(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # No ayase.toml present in an empty cwd -> defaults, no error.
    monkeypatch.chdir(tmp_path)
    cfg = AyaseConfig.load()
    assert cfg.general.device == "auto"


# ---------------------------------------------------------------------------
# Fix 4 — float min-score, max-score, missing-metric exclusion + count.
# ---------------------------------------------------------------------------


def _make_image_dataset(root: Path, names) -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image)


def test_filter_min_score_accepts_float_and_excludes_missing_metric():
    runner = CliRunner()
    with runner.isolated_filesystem():
        dataset = Path("dataset")
        dataset.mkdir()
        _make_image_dataset(dataset, ["photo.png"])

        # vmaf is not computed by the balanced default pipeline -> missing.
        # A float threshold (0.5) also confirms --min-score is no longer int-only.
        result = runner.invoke(
            app,
            ["filter", str(dataset), "--metric", "vmaf", "--min-score", "0.5", "--mode", "list"],
        )

        assert result.exit_code == 0, result.output
        assert "1 samples skipped: metric 'vmaf' missing" in result.output
        # Missing metric must NOT silently pass -> the sample is not listed.
        assert "photo.png" not in result.output


def test_filter_warns_min_score_on_lower_is_better_metric():
    runner = CliRunner()
    with runner.isolated_filesystem():
        dataset = Path("dataset")
        dataset.mkdir()
        _make_image_dataset(dataset, ["photo.png"])

        result = runner.invoke(
            app,
            ["filter", str(dataset), "--metric", "niqe", "--min-score", "5", "--mode", "list"],
        )

        assert result.exit_code == 0, result.output
        assert "lower-is-better" in result.output
        assert "--max-score" in result.output


def test_filter_max_score_option_is_accepted():
    runner = CliRunner()
    with runner.isolated_filesystem():
        dataset = Path("dataset")
        dataset.mkdir()
        _make_image_dataset(dataset, ["photo.png"])

        result = runner.invoke(
            app,
            ["filter", str(dataset), "--metric", "niqe", "--max-score", "3.5", "--mode", "list"],
        )

        # niqe is missing -> excluded and counted, no crash, no false-positive.
        assert result.exit_code == 0, result.output
        assert "1 samples skipped: metric 'niqe' missing" in result.output


# ---------------------------------------------------------------------------
# Fix 5 — copy/symlink mirror directory structure; unknown mode rejected.
# ---------------------------------------------------------------------------


def test_filter_copy_mirrors_structure_for_duplicate_basenames():
    runner = CliRunner()
    with runner.isolated_filesystem():
        dataset = Path("dataset")
        _make_image_dataset(dataset, ["a/clip.png", "b/clip.png"])
        out = Path("out")

        result = runner.invoke(
            app,
            ["filter", str(dataset), "--mode", "copy", "--output", str(out)],
        )

        assert result.exit_code == 0, result.output
        # Duplicate basenames in different subdirs must not overwrite each other.
        assert (out / "a" / "clip.png").exists()
        assert (out / "b" / "clip.png").exists()


def test_filter_rejects_unknown_mode_before_running():
    runner = CliRunner()
    with runner.isolated_filesystem():
        dataset = Path("dataset")
        dataset.mkdir()
        _make_image_dataset(dataset, ["photo.png"])
        out = Path("out")

        result = runner.invoke(
            app,
            ["filter", str(dataset), "--mode", "move", "--output", str(out)],
        )

        assert result.exit_code == 1
        assert "Unknown mode: move" in result.output
        # Must fail before creating the output directory.
        assert not out.exists()


# ---------------------------------------------------------------------------
# Fix 7 — checkpoint cadence crosses thresholds under sample_batch_size > 1.
# ---------------------------------------------------------------------------


class _BatchSevenModule(PipelineModule):
    name = "review_batch_seven_probe"
    description = "sample_batch_size=7 checkpoint probe"

    def process(self, sample: Sample) -> Sample:
        return sample

    def process_batch(self, samples):
        return samples


def test_process_samples_checkpoints_cross_threshold_with_batches(tmp_path: Path):
    module = _BatchSevenModule(config={"sample_batch_size": 7})
    pipeline = Pipeline([module])
    pipeline.start()

    samples = []
    for i in range(30):
        media = tmp_path / f"{i}.mp4"
        media.write_bytes(b"v")
        samples.append(Sample(path=media, is_video=True))

    counts = []
    total = _process_samples(
        pipeline,
        samples,
        checkpoint_every=10,
        checkpoint_fn=lambda _p, c: counts.append(c),
    )

    assert total == 30
    assert counts, "expected at least one checkpoint"
    # With batches of 7 and checkpoint_every=10, the first checkpoint must fire
    # as soon as we cross 10 (processed_count == 14) -- NOT at lcm(7, 10) == 70.
    assert counts[0] == 14
    # Final tail is always persisted.
    assert counts[-1] == 30


# ---------------------------------------------------------------------------
# Fix 10 — scanner does not adopt a same-stem caption from another directory.
# ---------------------------------------------------------------------------


def test_scanner_ignores_single_cross_directory_caption(tmp_path: Path):
    media = tmp_path / "videos" / "clip.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"not-a-real-video")

    # Exactly one same-stem caption, but in a DIFFERENT directory. Previously
    # this single candidate was silently adopted; now it must be ignored.
    caption = tmp_path / "captions" / "clip.txt"
    caption.parent.mkdir(parents=True)
    caption.write_text("wrong caption from elsewhere", encoding="utf-8")

    samples = scan_dataset(tmp_path, include_videos=True, include_images=False)

    assert len(samples) == 1
    assert samples[0].caption is None


def test_scanner_still_attaches_same_directory_caption(tmp_path: Path):
    media = tmp_path / "videos" / "clip.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"not-a-real-video")

    caption = tmp_path / "videos" / "clip.txt"
    caption.write_text("correct sidecar", encoding="utf-8")

    samples = scan_dataset(tmp_path, include_videos=True, include_images=False)

    assert len(samples) == 1
    assert samples[0].caption is not None
    assert samples[0].caption.text == "correct sidecar"
