"""NearID module contract tests without model downloads."""

import pytest
import torch
from PIL import Image

from ayase.models import Sample


def test_nearid_basics():
    from ayase.modules.nearid import NearIDModule
    from .conftest import _test_module_basics

    _test_module_basics(NearIDModule, "nearid")


def test_nearid_declares_pinned_huggingface_model():
    from ayase.modules.nearid import NearIDModule, _MODEL_REVISION

    assert NearIDModule.models[0]["type"] == "huggingface"
    assert _MODEL_REVISION in NearIDModule.models[0]["notes"]


def test_nearid_skips_without_backend(image_sample):
    from ayase.modules.nearid import NearIDModule

    result = NearIDModule().process(image_sample)
    assert result is image_sample
    assert result.quality_metrics is None


def test_nearid_scores_official_cosine(tmp_path):
    from ayase.modules.nearid import NearIDModule

    reference = tmp_path / "reference.png"
    target = tmp_path / "target.png"
    Image.new("RGB", (16, 16), "red").save(reference)
    Image.new("RGB", (16, 16), "blue").save(target)

    class FakeProcessor:
        def __call__(self, images, return_tensors=None):
            assert len(images) == 2
            assert return_tensors == "pt"
            return {"pixel_values": torch.zeros(2, 3, 2, 2)}

    class FakeModel:
        def get_image_features(self, **inputs):
            assert inputs["pixel_values"].shape == (2, 3, 2, 2)
            return torch.tensor([[1.0, 0.0], [0.6, 0.8]])

    module = NearIDModule()
    module._processor = FakeProcessor()
    module._model = FakeModel()
    module._torch = torch
    sample = Sample(path=target, is_video=False, reference_path=reference)
    result = module.process(sample)

    assert result is sample
    assert result.quality_metrics.nearid_identity_similarity == pytest.approx(0.6)


def test_nearid_accepts_reference_directory(tmp_path):
    from ayase.modules.nearid import NearIDModule

    refs = tmp_path / "refs"
    refs.mkdir()
    Image.new("RGB", (8, 8), "white").save(refs / "a.png")
    assert NearIDModule._load_image(refs).size == (8, 8)


def test_nearid_empty_reference_directory_returns_none(tmp_path):
    from ayase.modules.nearid import NearIDModule

    refs = tmp_path / "refs"
    refs.mkdir()
    assert NearIDModule._load_image(refs) is None
