"""Tests for MAUVE Audio Divergence."""

from types import SimpleNamespace

import numpy as np
import pytest

from tests.modules.conftest import _test_module_basics


def test_mauve_audio_divergence_basics():
    from ayase.modules.mauve_audio_divergence import MAUVEAudioDivergenceModule

    _test_module_basics(MAUVEAudioDivergenceModule, "mauve_audio_divergence")


def test_mauve_audio_divergence_graceful_without_setup(tmp_path):
    from ayase.modules.mauve_audio_divergence import MAUVEAudioDivergenceModule
    from ayase.models import Sample

    module = MAUVEAudioDivergenceModule()
    sample = Sample(path=tmp_path / "audio.wav", is_video=False)
    assert module.process(sample) is sample
    assert module._feature_cache == []


def test_mauve_audio_divergence_reference_formula():
    from ayase.modules.mauve_audio_divergence import MAUVEAudioDivergenceModule

    module = MAUVEAudioDivergenceModule()
    module._mauve = SimpleNamespace(
        compute_mauve=lambda **kwargs: SimpleNamespace(mauve=0.25)
    )
    generated = [np.zeros((1, 4), dtype=np.float32), np.ones((1, 4), dtype=np.float32)]
    reference = [
        np.full((1, 4), 2.0, dtype=np.float32),
        np.full((1, 4), 3.0, dtype=np.float32),
    ]

    assert module.compute_distribution_metric(generated, reference) == pytest.approx(
        -np.log(0.25)
    )


def test_mauve_audio_divergence_requires_reference():
    from ayase.modules.mauve_audio_divergence import MAUVEAudioDivergenceModule

    module = MAUVEAudioDivergenceModule()
    module._mauve = SimpleNamespace()
    with pytest.raises(ValueError, match="reference"):
        module.compute_distribution_metric([np.zeros((2, 4), dtype=np.float32)])


@pytest.mark.parametrize("score", [0.0, -0.1, 1.1, float("nan")])
def test_mauve_audio_divergence_rejects_invalid_mauve(score):
    from ayase.modules.mauve_audio_divergence import MAUVEAudioDivergenceModule

    module = MAUVEAudioDivergenceModule()
    module._mauve = SimpleNamespace(
        compute_mauve=lambda **kwargs: SimpleNamespace(mauve=score)
    )
    values = [np.zeros((1, 4), dtype=np.float32), np.ones((1, 4), dtype=np.float32)]
    with pytest.raises(ValueError, match="invalid MAUVE"):
        module.compute_distribution_metric(values, values)


def test_mauve_audio_divergence_rejects_unpublished_configuration():
    from ayase.modules.mauve_audio_divergence import MAUVEAudioDivergenceModule

    module = MAUVEAudioDivergenceModule({"aggregation": "mean"})
    module.setup()
    assert module._backend == "unavailable"
