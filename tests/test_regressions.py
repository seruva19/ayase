from pathlib import Path

import cv2
import numpy as np

from ayase.base_modules import BatchMetricModule
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import Pipeline, PipelineModule
from ayase.scanner import scan_dataset


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
    module = _MissingPkgModule()
    pipeline = Pipeline([module])
    pipeline.start()

    # Module should NOT be mounted (missing package)
    assert not module._mounted

    # Processing should skip the unmounted module
    sample = Sample(path=Path("test.mp4"), is_video=True)
    pipeline.process_sample(sample)
    assert module.process_calls == 0


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

    sample = Sample(path=img_path, is_video=False)
    result = VideoMemorabilityModule().process(sample)

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
