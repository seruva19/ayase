import numpy as np


def test_generative_distribution_basics():
    from ayase.modules.generative_distribution_metrics import (
        GenerativeDistributionModule,
    )
    from .conftest import _test_module_basics

    _test_module_basics(GenerativeDistributionModule, "generative_distribution")


def test_dataset_analytics_basics():
    from ayase.modules.dataset_analytics import DatasetAnalyticsModule
    from .conftest import _test_module_basics

    _test_module_basics(DatasetAnalyticsModule, "dataset_analytics")


def test_fvd_basics():
    from ayase.modules.fvd import FVDModule
    from .conftest import _test_module_basics

    _test_module_basics(FVDModule, "fvd")


def test_fvd_is_batch_metric():
    from ayase.modules.fvd import FVDModule
    from ayase.base_modules import BatchMetricModule

    assert issubclass(FVDModule, BatchMetricModule)


def test_fvd_instantiation():
    from ayase.modules.fvd import FVDModule

    m = FVDModule()
    assert m._feature_cache == []
    assert m._reference_cache == []


def test_fvd_compute_distribution_metric():
    from ayase.modules.fvd import FVDModule

    m = FVDModule()
    np.random.seed(42)
    feats = [np.random.randn(128) for _ in range(10)]
    refs = [np.random.randn(128) + 0.5 for _ in range(10)]
    score = m.compute_distribution_metric(feats, refs)
    assert isinstance(score, float)
    assert score >= 0


def test_kvd_basics():
    from ayase.modules.kvd import KVDModule
    from .conftest import _test_module_basics

    _test_module_basics(KVDModule, "kvd")


def test_kvd_is_batch_metric():
    from ayase.modules.kvd import KVDModule
    from ayase.base_modules import BatchMetricModule

    assert issubclass(KVDModule, BatchMetricModule)


def test_kvd_rbf_kernel():
    from ayase.modules.kvd import KVDModule

    m = KVDModule()
    x = np.random.randn(5, 10)
    y = np.random.randn(3, 10)
    k = m._rbf_kernel(x, y)
    assert k.shape == (5, 3)
    assert np.all(k >= 0) and np.all(k <= 1)


def test_kvd_compute_mmd():
    from ayase.modules.kvd import KVDModule

    m = KVDModule()
    np.random.seed(42)
    x = np.random.randn(20, 10)
    y = np.random.randn(20, 10)
    mmd = m._compute_mmd(x, y)
    assert isinstance(mmd, float)
    assert mmd >= 0


def test_fvmd_basics():
    from ayase.modules.fvmd import FVMDModule
    from .conftest import _test_module_basics

    _test_module_basics(FVMDModule, "fvmd")


def test_fvmd_is_batch_metric():
    from ayase.modules.fvmd import FVMDModule
    from ayase.base_modules import BatchMetricModule

    assert issubclass(FVMDModule, BatchMetricModule)


def test_fvmd_always_available():
    from ayase.modules.fvmd import FVMDModule

    m = FVMDModule()
    assert m._ml_available is True
