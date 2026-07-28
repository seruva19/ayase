"""Tests for deterministic and learned image-to-image metric modules."""

import cv2
import numpy as np
import pytest

from ayase.models import Sample
from tests.modules.conftest import _test_module_basics


def _write_image(path, offset=0):
    yy, xx = np.mgrid[:96, :96]
    image = np.stack(
        (
            (xx * 2 + offset) % 256,
            (yy * 2 + offset) % 256,
            ((xx + yy) * 2 + offset) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    cv2.rectangle(image, (24, 24), (72, 72), (20 + offset, 180, 240), 3)
    assert cv2.imwrite(str(path), image)
    return path


def test_i2i_module_basics():
    from ayase.modules.i2i_fidelity import I2IFidelityModule
    from ayase.modules.i2i_learned import I2ILearnedModule

    _test_module_basics(I2IFidelityModule, "i2i_fidelity")
    _test_module_basics(I2ILearnedModule, "i2i_learned")


def test_i2i_fidelity_identical_pair_populates_compact_fields(tmp_path):
    from ayase.modules.i2i_fidelity import ALL_FIELDS, I2IFidelityModule

    image = _write_image(tmp_path / "image.png")
    sample = Sample(path=image, is_video=False, reference_path=image)
    result = I2IFidelityModule().process(sample)
    metrics = result.quality_metrics

    assert result is sample
    assert metrics is not None
    assert len(ALL_FIELDS) == 19
    values = {field: getattr(metrics, field) for field in ALL_FIELDS}
    assert all(value is not None and np.isfinite(value) for value in values.values())
    assert values["i2i_mse"] == pytest.approx(0.0)
    assert values["i2i_mae"] == pytest.approx(0.0)
    assert values["i2i_exact_match_ratio"] == pytest.approx(1.0)
    assert values["i2i_edge_f1"] == pytest.approx(1.0)
    assert values["i2i_spectral_cosine"] == pytest.approx(1.0)
    assert values["i2i_mutual_information"] > 0


def test_i2i_fidelity_detects_changed_pair(tmp_path):
    from ayase.modules.i2i_fidelity import I2IFidelityModule

    reference = _write_image(tmp_path / "reference.png")
    generated = _write_image(tmp_path / "generated.png", offset=30)
    sample = Sample(path=generated, is_video=False, reference_path=reference)
    metrics = I2IFidelityModule().process(sample).quality_metrics

    assert metrics.i2i_mse > 0
    assert metrics.i2i_mae > 0
    assert metrics.i2i_exact_match_ratio < 1
    assert metrics.i2i_hist_bhattacharyya_red > 0
    assert metrics.i2i_dinov2_cls_similarity is None


def test_i2i_fidelity_without_reference_is_noop(tmp_path):
    from ayase.modules.i2i_fidelity import I2IFidelityModule

    image = _write_image(tmp_path / "image.png")
    sample = Sample(path=image, is_video=False)
    assert I2IFidelityModule().process(sample) is sample
    assert sample.quality_metrics is None


def test_i2i_learned_without_setup_degrades_gracefully(tmp_path):
    from ayase.modules.i2i_learned import I2ILearnedModule

    image = _write_image(tmp_path / "image.png")
    sample = Sample(path=image, is_video=False, reference_path=image)
    assert I2ILearnedModule().process(sample) is sample
    assert sample.quality_metrics is None
