"""Tests for verse_bench module."""

from ..conftest import _test_module_basics
from ayase.models import DatasetStats, Sample


def test_verse_bench_basics():
    from ayase.modules.verse_bench import VerseBenchModule

    _test_module_basics(VerseBenchModule, "verse_bench")


def test_verse_bench_process_passthrough(video_sample):
    from ayase.modules.verse_bench import VerseBenchModule

    module = VerseBenchModule()
    result = module.process(video_sample)
    assert result is video_sample


def test_verse_bench_calculate_overall_score():
    from ayase.modules.verse_bench import VerseBenchModule

    module = VerseBenchModule()
    metrics = {
        "AS": 0.9,
        "ID": 0.8,
        "FD": 0.2,
        "KL": 0.1,
        "CS": 0.7,
        "CE": 5.0,
        "CU": 6.0,
        "PC": 2.0,
        "PQ": 7.0,
        "WER": 0.1,
        "LSE-C": 3.0,
        "AV-A": 0.2,
    }

    breakdown = module._calculate_overall_score(metrics)
    assert breakdown["Overall Score"] > 0
    assert "S_joint" in breakdown
    assert "S_video" in breakdown


def test_verse_bench_post_process(monkeypatch, tmp_path):
    from ayase.modules.verse_bench import VerseBenchModule

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    video_path = input_dir / "clip.mp4"
    video_path.write_bytes(b"video")

    dataset_root = tmp_path / "verse_bench"
    (dataset_root / "set1").mkdir(parents=True)
    (dataset_root / "set2").mkdir(parents=True)
    (dataset_root / "set3").mkdir(parents=True)
    (dataset_root / "set1" / "00000.json").write_text("{}", encoding="utf-8")
    (dataset_root / "set1" / "00000.jpg").write_bytes(b"jpg")
    (dataset_root / "set2" / "dummy.wav").write_bytes(b"wav")
    (dataset_root / "set2" / "dummy.jpg").write_bytes(b"jpg")
    (dataset_root / "set3" / "dummy.wav").write_bytes(b"wav")
    (dataset_root / "set3" / "dummy.jpg").write_bytes(b"jpg")

    module = VerseBenchModule(
        {
            "dataset_root": str(dataset_root),
            "input_dir": str(input_dir),
        }
    )
    module.setup()

    class DummyPipeline:
        def __init__(self):
            self.stats = DatasetStats(total_samples=0, valid_samples=0, invalid_samples=0, total_size=0)

        def add_dataset_metric(self, metric_name, value):
            setattr(self.stats, metric_name, value)

    module.pipeline = DummyPipeline()

    class DummyResult:
        metrics = {
            "AS": 0.9,
            "ID": 0.8,
            "FD": 0.2,
            "KL": 0.1,
            "CS": 0.7,
            "CE": 5.0,
            "CU": 6.0,
            "PC": 2.0,
            "PQ": 7.0,
            "WER": 0.1,
            "LSE-C": 3.0,
            "AV-A": 0.2,
        }
        breakdown = {
            "S_joint": 0.6,
            "S_video": 0.7,
            "S_audio": 0.8,
            "S_other": 0.9,
            "Overall Score": 0.72,
        }

    def fake_run_native_benchmark(*args, **kwargs):
        return DummyResult.metrics, DummyResult.breakdown

    monkeypatch.setattr(module, "_run_native_benchmark", fake_run_native_benchmark)

    sample = Sample(path=video_path, is_video=True)
    module.post_process([sample])

    assert module.pipeline.stats.verse_bench_overall == 0.72
    assert module.pipeline.stats.verse_bench_metrics is not None
    assert module.pipeline.stats.verse_bench_breakdown is not None


def test_verse_bench_dataset_stats_fields():
    stats = DatasetStats(total_samples=0, valid_samples=0, invalid_samples=0, total_size=0)
    assert hasattr(stats, "verse_bench_overall")
    assert hasattr(stats, "verse_bench_metrics")
    assert hasattr(stats, "verse_bench_breakdown")


def test_verse_bench_validate_missing_materialized_assets(tmp_path):
    from ayase.modules.verse_bench import VerseBenchModule

    dataset_root = tmp_path / "verse_bench"
    (dataset_root / "set1").mkdir(parents=True)
    (dataset_root / "set2").mkdir(parents=True)
    (dataset_root / "set3").mkdir(parents=True)
    (dataset_root / "set1" / "00000.json").write_text("{}", encoding="utf-8")
    (dataset_root / "set1" / "00000.jpg").write_bytes(b"jpg")

    module = VerseBenchModule()
    assert module._validate_benchmark_assets(dataset_root) is False
