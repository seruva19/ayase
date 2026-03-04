import numpy as np


def test_umap_projection_basics():
    from ayase.modules.umap_projection import UMAPProjectionModule
    from .conftest import _test_module_basics

    _test_module_basics(UMAPProjectionModule, "umap_projection")


def test_umap_projection_config():
    from ayase.modules.umap_projection import UMAPProjectionModule

    m = UMAPProjectionModule()
    assert "device" in m.default_config
    assert "min_samples" in m.default_config


def test_umap_projection_is_batch_module():
    from ayase.base_modules import BatchMetricModule
    from ayase.modules.umap_projection import UMAPProjectionModule

    m = UMAPProjectionModule()
    assert isinstance(m, BatchMetricModule)
    assert hasattr(m, "_feature_cache")
    assert hasattr(m, "_sample_refs")


def test_umap_projection_numpy_pca_fallback():
    from ayase.modules.umap_projection import UMAPProjectionModule

    rng = np.random.default_rng(42)
    data = rng.standard_normal((10, 8))
    coords = UMAPProjectionModule._reduce_2d(data)
    assert coords.shape == (10, 2)
    assert np.isfinite(coords).all()


def test_umap_projection_hull_coverage_square():
    from ayase.modules.umap_projection import UMAPProjectionModule

    # Square points → high hull/bbox ratio
    coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]], dtype=float)
    cov = UMAPProjectionModule._hull_coverage(coords)
    assert 0.0 <= cov <= 1.0
    assert cov > 0.5


def test_umap_projection_hull_coverage_collinear():
    from ayase.modules.umap_projection import UMAPProjectionModule

    # Collinear points → degenerate hull
    coords = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
    cov = UMAPProjectionModule._hull_coverage(coords)
    assert 0.0 <= cov <= 1.0


def test_umap_projection_too_few_samples():
    from ayase.modules.umap_projection import UMAPProjectionModule

    m = UMAPProjectionModule()
    rng = np.random.default_rng(42)
    features = [rng.standard_normal(8) for _ in range(2)]
    score = m.compute_distribution_metric(features)
    assert score == 0.0


def test_umap_projection_sample_refs_sync(image_sample):
    from ayase.modules.umap_projection import UMAPProjectionModule

    m = UMAPProjectionModule()
    # extract_features returns non-None for a valid image → refs should sync
    feat = m.extract_features(image_sample)
    if feat is not None:
        assert len(m._sample_refs) == 1
    else:
        assert len(m._sample_refs) == 0
