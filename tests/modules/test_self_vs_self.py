"""Appendix D: distribution metrics must not split the dataset against itself.

Without a real reference set, ``compute_distribution_metric`` returns None
instead of halving the input and comparing the halves (a number that is not
the published metric).
"""

import numpy as np
import pytest


def _features(n=8, d=16, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(d) for _ in range(n)]


@pytest.mark.parametrize(
    "cls_path",
    [
        ("ayase.modules.audio_kl", "AudioKLModule"),
        ("ayase.modules.cmmd", "CMMDModule"),
        ("ayase.modules.fad", "FADModule"),
        ("ayase.modules.fid", "FIDModule"),
        ("ayase.modules.fvd", "FVDModule"),
        ("ayase.modules.generative_distribution_metrics", "GenerativeDistributionModule"),
        ("ayase.modules.kid", "KIDModule"),
        ("ayase.modules.kvd", "KVDModule"),
        ("ayase.modules.prdc_dinov2", "PRDCDINOv2Module"),
        ("ayase.modules.sfid", "SFIDModule"),
    ],
)
def test_no_reference_returns_none(cls_path):
    import importlib

    mod = importlib.import_module(cls_path[0])
    cls = getattr(mod, cls_path[1])
    module = cls({})
    score = module.compute_distribution_metric(_features())
    assert score is None, (
        f"{cls_path[1]} returned {score!r} without a reference set "
        "(self-split comparison is forbidden)"
    )


def test_prdc_no_reference_returns_none_dict_or_none():
    """PRDC returns a dict of four sub-metrics; all must be absent without ref."""
    from ayase.modules.prdc_dinov2 import PRDCDINOv2Module

    module = PRDCDINOv2Module({})
    result = module.compute_distribution_metric(_features())
    if isinstance(result, dict):
        assert all(v is None for v in result.values())
    else:
        assert result is None
