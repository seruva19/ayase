"""Tests for face_cross_similarity module."""

import numpy as np
import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics, Sample, DatasetStats


def test_face_cross_similarity_basics():
    from ayase.modules.face_cross_similarity import FaceCrossSimilarityModule

    _test_module_basics(FaceCrossSimilarityModule, "face_cross_similarity")


def test_face_cross_similarity_image(image_sample):
    from ayase.modules.face_cross_similarity import FaceCrossSimilarityModule

    image_sample.quality_metrics = QualityMetrics()
    m = FaceCrossSimilarityModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample


def test_face_cross_similarity_video(video_sample):
    from ayase.modules.face_cross_similarity import FaceCrossSimilarityModule

    video_sample.quality_metrics = QualityMetrics()
    m = FaceCrossSimilarityModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample


def test_face_cross_similarity_empty_cache():
    """post_process with empty cache should not crash."""
    from ayase.modules.face_cross_similarity import FaceCrossSimilarityModule

    m = FaceCrossSimilarityModule()
    # Empty cache, should skip gracefully
    m.post_process([])


def test_face_cross_similarity_cosine_math():
    """Verify cosine similarity computation using synthetic embeddings."""
    from ayase.modules.face_cross_similarity import FaceCrossSimilarityModule

    m = FaceCrossSimilarityModule()

    # Two identical embeddings should have similarity = 1.0
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    m._embeddings_cache["a"] = [emb]
    m._embeddings_cache["b"] = [emb]

    sample_a = Sample(path="a", is_video=False)
    sample_a.quality_metrics = QualityMetrics()
    sample_b = Sample(path="b", is_video=False)
    sample_b.quality_metrics = QualityMetrics()

    m.post_process([sample_a, sample_b])

    assert sample_a.quality_metrics.face_cross_similarity is not None
    assert abs(sample_a.quality_metrics.face_cross_similarity - 1.0) < 1e-5
    assert abs(sample_b.quality_metrics.face_cross_similarity - 1.0) < 1e-5


def test_face_cross_similarity_orthogonal():
    """Orthogonal embeddings should have similarity close to 0."""
    from ayase.modules.face_cross_similarity import FaceCrossSimilarityModule

    m = FaceCrossSimilarityModule()

    emb_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    m._embeddings_cache["a"] = [emb_a]
    m._embeddings_cache["b"] = [emb_b]

    sample_a = Sample(path="a", is_video=False)
    sample_a.quality_metrics = QualityMetrics()
    sample_b = Sample(path="b", is_video=False)
    sample_b.quality_metrics = QualityMetrics()

    m.post_process([sample_a, sample_b])

    assert sample_a.quality_metrics.face_cross_similarity is not None
    assert abs(sample_a.quality_metrics.face_cross_similarity) < 1e-5


def test_face_cross_similarity_clustering():
    """Test identity clustering with synthetic embeddings."""
    from ayase.modules.face_cross_similarity import FaceCrossSimilarityModule

    m = FaceCrossSimilarityModule({"similarity_threshold": 0.9})

    # Two clusters: (a, b) similar, (c) different
    emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb2 = np.array([0.99, 0.1, 0.0], dtype=np.float32)
    emb2 = emb2 / np.linalg.norm(emb2)
    emb3 = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    m._embeddings_cache["a"] = [emb1]
    m._embeddings_cache["b"] = [emb2]
    m._embeddings_cache["c"] = [emb3]

    sample_a = Sample(path="a", is_video=False)
    sample_a.quality_metrics = QualityMetrics()
    sample_b = Sample(path="b", is_video=False)
    sample_b.quality_metrics = QualityMetrics()
    sample_c = Sample(path="c", is_video=False)
    sample_c.quality_metrics = QualityMetrics()

    # Verify clustering logic directly
    # emb1 . emb2 = 0.99 / sqrt(0.99^2 + 0.1^2) ~ 0.995 -> same cluster
    # emb1 . emb3 = 0 -> different cluster
    sim_matrix = np.zeros((3, 3))
    embs = [emb1, emb2, emb3]
    for i in range(3):
        for j in range(3):
            sim_matrix[i, j] = float(np.dot(embs[i], embs[j]))

    clusters = m._cluster_identities(sim_matrix)
    assert clusters == 2, f"Expected 2 clusters, got {clusters}"


def test_face_cross_similarity_fields():
    """Verify fields exist in QualityMetrics and DatasetStats."""
    # QualityMetrics fields
    qm = QualityMetrics()
    assert hasattr(qm, "face_cross_similarity")
    assert hasattr(qm, "face_identity_count")
    assert qm.face_cross_similarity is None
    assert qm.face_identity_count is None

    # DatasetStats fields
    ds = DatasetStats(total_samples=0, valid_samples=0, invalid_samples=0, total_size=0)
    assert hasattr(ds, "face_similarity_matrix")
    assert hasattr(ds, "avg_face_cross_similarity")
    assert hasattr(ds, "identity_cluster_count")


def test_face_cross_similarity_field_groups():
    """Verify field group assignments."""
    qm = QualityMetrics()
    groups = qm._FIELD_GROUPS
    assert groups.get("face_cross_similarity") == "face"
    assert groups.get("face_identity_count") == "face"


def test_face_cross_similarity_max_cache():
    """Module should stop caching when max_cache_size is reached."""
    from ayase.modules.face_cross_similarity import FaceCrossSimilarityModule

    m = FaceCrossSimilarityModule({"max_cache_size": 2})
    m._backend = "insightface"  # pretend we have a backend
    # Fill cache to max
    m._embeddings_cache["a"] = [np.zeros(3)]
    m._embeddings_cache["b"] = [np.zeros(3)]

    # Process should skip because cache is full
    sample = Sample(path="c", is_video=False)
    sample.quality_metrics = QualityMetrics()
    result = m.process(sample)
    assert "c" not in m._embeddings_cache
