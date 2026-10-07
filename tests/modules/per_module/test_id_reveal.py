"""Tests for the id_reveal module (ID-Reveal, Cozzolino et al. 2021)."""

import os

import numpy as np
import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics, Sample


def test_id_reveal_basics():
    from ayase.modules.id_reveal import IDRevealModule
    _test_module_basics(IDRevealModule, "id_reveal")


def test_id_reveal_without_backend_leaves_fields_unset(tmp_path):
    from ayase.modules.id_reveal import IDRevealModule
    m = IDRevealModule()
    sample = Sample(path=tmp_path / "v.mp4", is_video=True)
    sample.quality_metrics = QualityMetrics()
    assert m.process(sample).quality_metrics.id_reveal_distance is None
    assert m.process(sample).quality_metrics.id_reveal_tracks is None


def test_id_reveal_no_reference_skips(tmp_path):
    from ayase.modules.id_reveal import IDRevealModule
    m = IDRevealModule()
    m._backend = "idreveal"
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    sample = Sample(path=video, is_video=True)
    sample.quality_metrics = QualityMetrics()
    assert m.process(sample).quality_metrics.id_reveal_distance is None


def test_id_reveal_rejects_image_input(tmp_path):
    from ayase.modules.id_reveal import IDRevealModule
    m = IDRevealModule()
    m._backend = "idreveal"
    image = tmp_path / "v.png"
    image.write_bytes(b"x")
    sample = Sample(path=image, is_video=False)
    sample.quality_metrics = QualityMetrics()
    assert m.process(sample).quality_metrics.id_reveal_distance is None


def test_id_reveal_rejects_image_reference(tmp_path):
    from ayase.modules.id_reveal import IDRevealModule
    m = IDRevealModule()
    m._backend = "idreveal"
    m._ref_cache_root = tmp_path / "refcache"
    ref_dir = tmp_path / "refs"
    ref_dir.mkdir()
    (ref_dir / "face.png").write_bytes(b"x")
    assert m._reference_dir(ref_dir) is None


def test_id_reveal_empty_reference_dir(tmp_path):
    from ayase.modules.id_reveal import IDRevealModule
    m = IDRevealModule()
    m._backend = "idreveal"
    m._ref_cache_root = tmp_path / "refcache"
    ref_dir = tmp_path / "empty_refs"
    ref_dir.mkdir()
    assert m._reference_dir(ref_dir) is None


def test_upstream_distance_excludes_reference_by_stem(tmp_path):
    """Document the vendored flat-corpus behavior that Ayase wraps narrowly."""
    from ayase.modules.id_reveal import IDRevealModule

    ref_dir = tmp_path / "refcache" / "k" / "v"
    ref_dir.mkdir(parents=True)
    embs = np.random.default_rng(0).normal(size=(3, 128)).astype(np.float32)
    np.savez(ref_dir / "embs_track0.npz", embs_3dmm=embs, embs_range=np.zeros((3, 2)))

    m = IDRevealModule()
    # ComputeDistance.reset(filename) excludes refs whose stem equals the
    # candidate stem — verify the vendored mechanism loads our cache layout.
    from ayase.third_party.idreveal.grip_unina.util_dist import ComputeDistance

    cd = ComputeDistance("3dmm", str(tmp_path / "refcache" / "k"), normalize=False)
    assert list(cd.list_refs) == ["v"]
    assert cd.list_refs["v"].shape == (3, 128)
    cd.reset(filename="v.mp4")
    assert cd.list_insize == []
    cd.reset(filename="other.mp4")
    assert cd.list_insize == ["v"]


def _cache_ready_module(tmp_path, **config):
    from ayase.modules.id_reveal import IDRevealModule

    module = IDRevealModule(config)
    module._ref_cache_root = tmp_path / "cache"
    module._resources = tmp_path / "weights"
    module._resources.mkdir()
    for name in ("Resnet50_Final.pth", "mb1_120x120.pth", "model_idreveal.th"):
        (module._resources / name).write_bytes(name.encode())
    module._ref_temporal = object()
    module._extract_windows = lambda video, temporal: {
        "embs_track": [0],
        "embs_3dmm": [np.zeros(4, dtype=np.float32)],
        "embs_range": [np.array([0, 100])],
    }
    return module


def test_reference_cache_key_includes_protocol_weights_and_media(tmp_path):
    reference = tmp_path / "reference.mp4"
    reference.write_bytes(b"video-v1")
    module = _cache_ready_module(tmp_path, fps=25)

    first = module._reference_dir(reference)
    assert first is not None

    module.config["fps"] = 30
    protocol_changed = module._reference_dir(reference)
    assert protocol_changed != first

    module.config["fps"] = 25
    (module._resources / "model_idreveal.th").write_bytes(b"different-weight")
    weight_changed = module._reference_dir(reference)
    assert weight_changed not in (first, protocol_changed)

    reference.write_bytes(b"video-v2-with-different-size")
    media_changed = module._reference_dir(reference)
    assert media_changed not in (first, protocol_changed, weight_changed)


def test_distinct_same_stem_reference_is_not_excluded(tmp_path):
    candidate_dir = tmp_path / "candidate"
    reference_dir = tmp_path / "references"
    candidate_dir.mkdir()
    reference_dir.mkdir()
    candidate = candidate_dir / "person.mp4"
    reference = reference_dir / "person.mp4"
    candidate.write_bytes(b"candidate")
    reference.write_bytes(b"reference")

    module = _cache_ready_module(tmp_path)
    prepared = module._reference_dir(reference_dir, exclude_path=candidate)

    assert prepared is not None
    assert list(prepared.glob("person-*/embs_track0.npz"))


def test_reference_set_failure_does_not_publish_complete_and_retries(tmp_path):
    refs = tmp_path / "references"
    refs.mkdir()
    first = refs / "a.mp4"
    second = refs / "b.mp4"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    module = _cache_ready_module(tmp_path)
    calls = []

    def flaky(video, temporal):
        calls.append(video.name)
        if video == second and calls.count("b.mp4") == 1:
            raise RuntimeError("transient")
        return {
            "embs_track": [0],
            "embs_3dmm": [np.zeros(4, dtype=np.float32)],
            "embs_range": [np.array([0, 100])],
        }

    module._extract_windows = flaky
    assert module._reference_dir(refs) is None
    cache_dir = next(p for p in module._ref_cache_root.iterdir() if p.is_dir())
    assert not (cache_dir / ".complete").exists()

    prepared = module._reference_dir(refs)
    assert prepared == cache_dir
    assert (prepared / ".complete").exists()
    assert calls == ["a.mp4", "b.mp4", "b.mp4"]


def test_partial_reference_subdir_is_quarantined_and_rebuilt(tmp_path):
    reference = tmp_path / "reference.mp4"
    reference.write_bytes(b"video")
    module = _cache_ready_module(tmp_path)
    module._extract_windows = lambda video, temporal: (_ for _ in ()).throw(RuntimeError("stop"))
    assert module._reference_dir(reference) is None
    cache_dir = next(p for p in module._ref_cache_root.iterdir() if p.is_dir())
    partial = cache_dir / module._reference_label(reference)
    partial.mkdir()
    np.savez(partial / "embs_track0.npz", embs_3dmm=np.ones((1, 4)))

    rebuilt = []
    module._extract_windows = lambda video, temporal: rebuilt.append(video.name) or {
        "embs_track": [0],
        "embs_3dmm": [np.zeros(4, dtype=np.float32)],
        "embs_range": [np.array([0, 100])],
    }
    prepared = module._reference_dir(reference)

    assert prepared == cache_dir
    assert rebuilt == ["reference.mp4"]
    assert (partial / ".complete").exists()
    assert list((cache_dir / ".staging").glob("quarantine-*"))


def test_empty_reference_completion_is_trusted_without_retry(tmp_path):
    reference = tmp_path / "noface.mp4"
    reference.write_bytes(b"video")
    module = _cache_ready_module(tmp_path)
    calls = []
    module._extract_windows = lambda video, temporal: calls.append(video.name) or None

    assert module._reference_dir(reference) is None
    assert module._reference_dir(reference) is None
    assert calls == ["noface.mp4"]
    cache_dir = next(p for p in module._ref_cache_root.iterdir() if p.is_dir())
    assert (cache_dir / ".complete").exists()
    assert not (cache_dir / ".has_embeddings").exists()


@pytest.mark.skipif(
    os.environ.get("AYASE_IDREVEAL_REAL") != "1",
    reason="real ID-Reveal weights download disabled (set AYASE_IDREVEAL_REAL=1)",
)
def test_id_reveal_real_backend_setup():
    """Full backend: downloads ~150 MB of weights and initialises the nets."""
    from ayase.modules.id_reveal import IDRevealModule

    m = IDRevealModule()
    m.setup()
    assert m._backend == "idreveal"
    assert m._det is not None and m._mm is not None
