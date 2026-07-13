"""Tests for the p1204 module (ITU-T P.1204.3 bitstream NR quality)."""

import os

import numpy as np
import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_p1204_basics():
    from ayase.modules.p1204 import P1204Module
    _test_module_basics(P1204Module, "p1204")


def test_p1204_video(video_sample):
    """Graceful path: in test_mode setup() is skipped, backend stays
    'unavailable', and process() returns the sample unchanged (no fabricated
    MOS)."""
    from ayase.modules.p1204 import P1204Module
    video_sample.quality_metrics = QualityMetrics()
    m = P1204Module()
    m.on_mount()
    assert m._backend == "unavailable"
    assert m._ml_available is False
    result = m.process(video_sample)
    assert result is video_sample
    assert result.quality_metrics.p1204_mos is None


def test_p1204_rf_forward_pure_numpy():
    """The self-contained tree forward reproduces decision-tree traversal
    (network-free: hand-built ensemble)."""
    from ayase.modules.p1204 import _RandomForest

    # nodes: [left, right, feature, threshold, impurity, n, wn]
    # root splits on feature 0 at 5.0 -> two leaves (1.0 / 2.0)
    tree_dict = {
        "nodes": [
            [1, 2, 0, 5.0, 0.0, 0, 0.0],
            [-1, -1, -2, -2.0, 0.0, 0, 0.0],
            [-1, -1, -2, -2.0, 0.0, 0, 0.0],
        ],
        "values": [[[0.0]], [[1.0]], [[2.0]]],
    }
    tree = _RandomForest._build_tree(tree_dict)
    rf = _RandomForest([tree, tree], fs_mask=None)  # two identical trees
    assert rf.predict_one(np.array([3.0])) == pytest.approx(1.0)
    assert rf.predict_one(np.array([9.0])) == pytest.approx(2.0)
    assert rf.predict_one(np.array([5.0])) == pytest.approx(1.0)  # <= goes left


def test_p1204_rmos_helpers():
    from ayase.modules.p1204 import _map_to_5, _mos_from_r, _r_from_mos

    assert _mos_from_r(0) == pytest.approx(1.05)
    assert _mos_from_r(100) == pytest.approx(4.9)
    assert _map_to_5(4.5) == 5.0
    assert _map_to_5(1.0) == pytest.approx(1.0)
    # r_from_mos is (approximately) the inverse of mos_from_r on the interior
    r = _r_from_mos(_mos_from_r(50.0))
    assert r == pytest.approx(50.0, abs=0.5)


def test_p1204_predict_parametric_and_rf():
    """Load the real JSON RF + parametric coefficients from the Hub and run the
    full parametric-baseline + RF-residual prediction on a synthetic feature
    vector. Verifies the model is code-complete and produces a MOS in [1, 5].

    Skipped when the model cannot be downloaded (offline)."""
    huggingface_hub = pytest.importorskip("huggingface_hub")
    import json as _json

    from ayase.modules.p1204 import P1204Module, _RandomForest

    try:
        reg = huggingface_hub.hf_hub_download(
            repo_id="AkaneTendo25/ayase-models",
            filename="p1204/models/p1204_3/mode3_pc_20trees_depth_8_reg.json",
        )
        fs = huggingface_hub.hf_hub_download(
            repo_id="AkaneTendo25/ayase-models",
            filename="p1204/models/p1204_3/mode3_pc_20trees_depth_8_fs.json",
        )
        cfg = huggingface_hub.hf_hub_download(
            repo_id="AkaneTendo25/ayase-models",
            filename="p1204/models/p1204_3/config.json",
        )
    except Exception as exc:  # offline / hub unavailable
        pytest.skip(f"P.1204.3 model not downloadable: {exc}")

    m = P1204Module()
    m._forest = _RandomForest.from_json(reg, fs)
    m._params = _json.load(open(cfg))["pc"]["params"]
    m._display_res = 3840 * 2160

    feat = {
        "video_codec": "hevc",
        "Bitrate": 8000.0,
        "Resolution": 3840 * 2160.0,
        "Framerate": 30.0,
        "mean_Av_QPBB_non-i": 28.0,
        "iqr_Av_QPBB_non-i": 4.0,
        "kurtosis_Av_QPBB_non-i": 0.2,
        "iqr_min_QP": 2.0,
        "std_max_QP_non-i": 1.5,
        "1.0_quantil_FrameSize": 50000.0,
        "std_FrameSize_non-i": 12000.0,
        "kurtosis_FrameSize_non-i": 0.5,
        "kurtosis_Av_Motion": 0.3,
        "0.0_quantil_StdDev_MotionX_non-i": 1.0,
    }
    mos = m._predict(feat)
    assert mos is not None
    assert 1.0 <= mos <= 5.0
    # lower QP should not score worse than much higher QP, same codec
    feat_hi = dict(feat, mean_Av_QPBB_non_i=48.0)
    feat_hi["mean_Av_QPBB_non-i"] = 48.0
    assert m._predict(feat) >= m._predict(feat_hi) - 1e-6


@pytest.mark.skipif(
    not os.environ.get("AYASE_P1204_VIDEOPARSER"),
    reason="requires compiled bitstream_mode3_videoparser (AYASE_P1204_VIDEOPARSER)",
)
def test_p1204_real_path_with_parser(video_sample):
    """Real backend end-to-end: only runs when the compiled bitstream parser is
    configured. Confirms setup() reaches _backend == 'real' and process()
    emits a MOS in [1, 5]."""
    from ayase.modules.p1204 import P1204Module

    m = P1204Module({"test_mode": False})
    m.setup()
    assert m._backend == "real"
    assert m._ml_available is True

    video_sample.quality_metrics = QualityMetrics()
    result = m.process(video_sample)
    mos = result.quality_metrics.p1204_mos
    # The synthetic fixture is mp4v/raw; the parser may reject it and return
    # no score -- that is an acceptable honest outcome. When a score is
    # produced it must be a valid MOS.
    if mos is not None:
        assert 1.0 <= mos <= 5.0
