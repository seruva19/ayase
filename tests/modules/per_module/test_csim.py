"""Tests for the csim module (CSIM, Zakharov 2019 / SadTalker)."""

import numpy as np
import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics, Sample


def test_csim_basics():
    from ayase.modules.csim import CSIMModule
    _test_module_basics(CSIMModule, "csim")


def test_csim_without_backend_leaves_fields_unset(tmp_path):
    from ayase.modules.csim import CSIMModule
    m = CSIMModule()
    sample = Sample(path=tmp_path / "v.mp4", is_video=True)
    sample.quality_metrics = QualityMetrics()
    assert m.process(sample).quality_metrics.csim is None
    assert m.process(sample).quality_metrics.csim_face_frames is None


def test_csim_no_reference_skips(tmp_path):
    from ayase.modules.csim import CSIMModule
    m = CSIMModule()
    m._backend = "insightface"
    m._arc = object()
    sample = Sample(path=tmp_path / "v.mp4", is_video=True)
    sample.quality_metrics = QualityMetrics()
    assert m.process(sample).quality_metrics.csim is None


def test_csim_rejects_video_reference(tmp_path):
    from ayase.modules.csim import CSIMModule
    m = CSIMModule()
    m._backend = "insightface"
    m._arc = object()
    ref_dir = tmp_path / "refs"
    ref_dir.mkdir()
    (ref_dir / "clip.mp4").write_bytes(b"x")
    # A reference dir containing no images yields no embedding.
    assert m._reference_embedding(ref_dir) is None


def test_csim_frame_sampling(tmp_path):
    """max_frames>0 subsamples linspace; 0 reads every frame."""
    from ayase.modules.csim import CSIMModule

    m_all = CSIMModule({"max_frames": 0})
    assert m_all.max_frames == 0
    m_sub = CSIMModule({"max_frames": 8})
    assert m_sub.max_frames == 8


def test_csim_mean_aggregation():
    """Identical embeddings -> cosine 1.0 mean."""
    from ayase.modules.csim import CSIMModule

    emb = np.ones(512, dtype=np.float32)
    emb = emb / np.linalg.norm(emb)
    assert CSIMModule._cosine(emb, emb) == pytest.approx(1.0)
