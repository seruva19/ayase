import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


class _Pipeline:
    def __init__(self):
        self.metrics = {}

    def add_dataset_metric(self, name, value):
        self.metrics[name] = value


def _image(path: Path) -> Path:
    Image.new("RGB", (4, 4), (20, 40, 60)).save(path)
    return path


def _dense_mmd(x, y, sigma):
    def kernel(a, b):
        d = ((a[:, None] - b[None]) ** 2).sum(2)
        return np.exp(-d / (2 * sigma**2)).mean()

    return kernel(x, x) + kernel(y, y) - 2 * kernel(x, y)


def test_palate_numerical_oracle_and_block_boundaries(tmp_path):
    from ayase.modules.palate import _compute_scores

    train = np.array([[0.0, 0.0], [0.2, 1.0], [1.0, 0.4]])
    test = np.array([[2.0, 0.0], [2.2, 1.0], [3.0, 0.4]])
    generated = np.array([[0.1, 0.2], [2.1, 0.8], [1.0, 0.5], [2.5, 0.3]])
    sigma = 1.7
    result = _compute_scores(
        train, test, generated, sigma=sigma, threshold=-1e9, block_size=2, temporary_dir=tmp_path
    )
    dt = _dense_mmd(test, generated, sigma / 3)
    dr = _dense_mmd(train, generated, sigma / 3)
    base = _dense_mmd(test, generated, sigma)
    ktt = np.exp(-((test[:, None] - test[None]) ** 2).sum(2) / (2 * sigma**2)).mean()
    kgg = np.exp(-((generated[:, None] - generated[None]) ** 2).sum(2) / (2 * sigma**2)).mean()
    assert result["palate_score"] == pytest.approx(dt / (dt + dr), abs=1e-12)
    assert result["baseline_score"] == pytest.approx(base / (ktt + kgg), abs=1e-12)
    assert result["m_palate_score"] == pytest.approx(
        0.5 * result["baseline_score"] + 0.5 * result["palate_score"]
    )


def test_empty_retained_keeps_only_baseline(tmp_path):
    from ayase.modules.palate import _compute_scores

    train = np.array([[0.0, 0.0], [1.0, 1.0]])
    test = np.array([[2.0, 0.0], [3.0, 1.0]])
    gen = np.array([[10.0, 10.0]])
    result = _compute_scores(train, test, gen, sigma=1.0, threshold=1.0, temporary_dir=tmp_path)
    assert result["baseline_score"] is not None
    assert result["retained_fraction"] == 0
    assert result["palate_score"] is result["m_palate_score"] is None


def test_zero_variance_kde_preserves_baseline_with_reason(tmp_path):
    from ayase.modules.palate import _compute_scores

    train = np.array([[0.0, 1.0], [1.0, 1.0]])
    test = np.array([[2.0, 1.0], [3.0, 1.0]])
    gen = np.array([[1.0, 1.0]])
    result = _compute_scores(train, test, gen, sigma=1.0, temporary_dir=tmp_path)
    assert result["baseline_score"] is not None and "zero-variance" in result["filter_error"]
    assert result["retained_fraction"] is None and result["palate_score"] is None


def test_automatic_sigma_is_exact_distinct_pair_median(tmp_path):
    from ayase.modules.palate import _estimate_sigma

    real = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 4.0]])
    expected = math.sqrt(np.median([4.0, 16.0, 20.0])) / 2
    assert _estimate_sigma(real, 2, tmp_path) == pytest.approx(expected)
    assert not list(tmp_path.iterdir())


def test_even_sigma_median_and_block_invariance(tmp_path):
    from ayase.modules.palate import _compute_scores, _estimate_sigma

    real = np.array([[0.0], [2.0], [5.0], [9.0]])
    distances = sorted((real[i, 0] - real[j, 0]) ** 2 for i in range(4) for j in range(i + 1, 4))
    assert _estimate_sigma(real, 2, tmp_path) == pytest.approx(math.sqrt(np.median(distances)) / 2)
    train = np.array([[0.0, 0.0], [1.0, 2.0]])
    test = np.array([[3.0, 1.0], [4.0, 3.0]])
    generated = np.array([[0.5, 0.5], [3.5, 2.0], [2.0, 1.0]])
    small = _compute_scores(train, test, generated, sigma=2.0, block_size=1)
    large = _compute_scores(train, test, generated, sigma=2.0, block_size=10)
    for key in ("palate_score", "m_palate_score", "baseline_score", "log_density_min"):
        assert small[key] == pytest.approx(large[key], abs=1e-12)


def test_scores_are_translation_and_nondividing_block_invariant():
    from ayase.modules.palate import _compute_scores

    rng = np.random.default_rng(19)
    train, test, generated = (
        rng.normal(size=(5, 7)),
        rng.normal(size=(6, 7)),
        rng.normal(size=(7, 7)),
    )
    expected = _compute_scores(train, test, generated, sigma=2.0, block_size=1024)
    shifted = _compute_scores(train + 1e8, test + 1e8, generated + 1e8, sigma=2.0, block_size=4)
    for key in ("palate_score", "m_palate_score", "baseline_score", "log_density_min"):
        assert shifted[key] == pytest.approx(expected[key], abs=1e-7)


def test_input_validation_precedes_asset_acquisition(tmp_path, monkeypatch):
    import ayase.modules.palate as implementation

    implementation.PALATEModule.set_test_mode(False)
    module = implementation.PALATEModule(
        {"train_paths": [_image(tmp_path / "same.png")], "test_paths": [tmp_path / "same.png"]}
    )
    monkeypatch.setattr(module, "_resolve_assets", lambda: pytest.fail("must not acquire"))
    module.setup()
    assert not module._ml_available


def test_test_mode_never_acquires_assets(monkeypatch):
    from ayase.modules.palate import PALATEModule

    module = PALATEModule({"test_mode": True})
    monkeypatch.setattr(module, "_resolve_assets", lambda: pytest.fail("must not acquire"))
    module.setup()
    assert not module._ml_available


def test_process_rejects_video_pair_duplicate_and_resets(tmp_path):
    from ayase.models import Sample
    from ayase.modules.palate import PALATEModule

    path = _image(tmp_path / "generated.png")
    module = PALATEModule({"test_mode": True})
    assert module.process(Sample(path=path, is_video=False)).path == path
    module.process(Sample(path=path, is_video=False))
    assert module._invalid_generated
    module.on_execute()
    assert module._generated_paths == [] and not module._invalid_generated
    module.process(Sample(path=path, is_video=True))
    assert module._invalid_generated


def test_postprocess_routes_scores_and_empty_retained(tmp_path, monkeypatch):
    from ayase.models import Sample
    from ayase.modules.palate import PALATEModule

    train = [_image(tmp_path / "train.png")]
    test = [_image(tmp_path / "test.png")]
    generated = _image(tmp_path / "gen.png")
    module = PALATEModule(
        {
            "train_paths": train,
            "test_paths": test,
            "sigma": 1.0,
            "kde_threshold": 1.0,
            "test_mode": True,
            "models_dir": str(tmp_path),
        }
    )
    module._ml_available = True
    module.pipeline = _Pipeline()
    module.process(Sample(path=generated, is_video=False))
    monkeypatch.setattr(
        module,
        "_invoke",
        lambda *_: {
            "train": [[0.0, 0.0], [1.0, 1.0]],
            "test": [[2.0, 0.0], [3.0, 1.0]],
            "generated": [[10.0, 10.0]],
            "device": "cpu",
            "source_root": "s",
            "model_path": "m",
        },
    )
    module.post_process([])
    assert "palate_baseline_score" in module.pipeline.metrics
    assert module.pipeline.metrics["palate_retained_fraction"] == 0
    assert "palate_score" not in module.pipeline.metrics


def test_pinned_asset_routing(tmp_path, monkeypatch):
    import ayase.modules.palate as p

    source = tmp_path / "source"
    for relative in p._REQUIRED:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    (source / ".ayase-source-revision").write_text(p.DINOV2_REVISION)
    model = tmp_path / "model.pth"
    model.write_bytes(b"x")
    monkeypatch.setattr(p, "_ensure_source", lambda _: source)
    monkeypatch.setattr(p, "_download_model", lambda _: model)
    module = p.PALATEModule({"models_dir": str(tmp_path)})
    assert module._resolve_assets() == (source, model)
