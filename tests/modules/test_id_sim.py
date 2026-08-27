"""ID-Sim module contract tests without model downloads."""

from pathlib import Path

import pytest
import torch
from PIL import Image

from ayase.models import Sample


def test_id_sim_basics():
    from ayase.modules.id_sim import IDSimModule
    from .conftest import _test_module_basics

    _test_module_basics(IDSimModule, "id_sim")


def test_id_sim_declares_only_huggingface_models():
    from ayase.modules.id_sim import IDSimModule

    assert IDSimModule.models
    assert all(model["type"] == "huggingface" for model in IDSimModule.models)
    assert "id_sim_distance" in IDSimModule.metric_info


def test_id_sim_skips_without_backend(image_sample):
    from ayase.modules.id_sim import IDSimModule

    result = IDSimModule().process(image_sample)
    assert result is image_sample
    assert result.quality_metrics is None


def test_id_sim_skips_without_reference(image_sample):
    from ayase.modules.id_sim import IDSimModule

    module = IDSimModule()
    module._model = object()
    module._preprocess = object()
    assert module.process(image_sample) is image_sample


def test_id_sim_scores_official_distance(tmp_path):
    from ayase.modules.id_sim import IDSimModule

    ref = tmp_path / "reference.png"
    target = tmp_path / "target.png"
    Image.new("RGB", (16, 16), "red").save(ref)
    Image.new("RGB", (16, 16), "blue").save(target)

    class FakeModel:
        def __call__(self, a, b, mode=None):
            assert mode == "cls"
            assert a.shape == b.shape == (1, 3, 2, 2)
            return torch.tensor([0.375])

    module = IDSimModule()
    module._model = FakeModel()
    module._preprocess = lambda image: torch.zeros(1, 3, 2, 2)
    sample = Sample(path=target, is_video=False, reference_path=ref)
    result = module.process(sample)

    assert result is sample
    assert result.quality_metrics is not None
    assert result.quality_metrics.id_sim_distance == pytest.approx(0.375)


def test_id_sim_accepts_reference_directory(tmp_path):
    from ayase.modules.id_sim import IDSimModule

    refs = tmp_path / "refs"
    refs.mkdir()
    Image.new("RGB", (8, 8), "white").save(refs / "a.png")
    assert IDSimModule._load_image(refs).size == (8, 8)


def test_id_sim_empty_reference_directory_returns_none(tmp_path):
    from ayase.modules.id_sim import IDSimModule

    refs = tmp_path / "refs"
    refs.mkdir()
    assert IDSimModule._load_image(refs) is None
