"""MaSC module tests without model downloads."""

import numpy as np
import pytest
import torch
from PIL import Image

from ayase.models import Sample


def test_masc_basics():
    from ayase.modules.masc import MaSCModule
    from .conftest import _test_module_basics

    _test_module_basics(MaSCModule, "masc")


def test_masc_requires_reference_and_mask(image_sample):
    from ayase.modules.masc import MaSCModule

    module = MaSCModule()
    assert module.process(image_sample) is image_sample
    assert image_sample.quality_metrics is None


def test_masc_masked_maxcos_equation():
    from ayase.modules.masc import MaSCModule

    module = MaSCModule()
    module._torch = torch
    module._device = "cpu"
    embeddings = iter(
        [
            (torch.tensor([[1.0, 0.0], [0.0, 1.0]]), 1, 2),
            (torch.tensor([[0.6, 0.8], [1.0, 0.0]]), 1, 2),
        ]
    )
    module._encode = lambda image: next(embeddings)
    mask = np.array([[255, 0]], dtype=np.float32)
    score = module._score(Image.new("RGB", (2, 1)), Image.new("RGB", (2, 1)), mask)
    assert score == pytest.approx(1.0)


def test_masc_process_uses_sample_mask_path(tmp_path):
    from ayase.modules.masc import MaSCModule

    reference = tmp_path / "reference.png"
    target = tmp_path / "target.png"
    mask = tmp_path / "mask.png"
    Image.new("RGB", (8, 8), "red").save(reference)
    Image.new("RGB", (8, 8), "blue").save(target)
    Image.new("L", (8, 8), 255).save(mask)
    module = MaSCModule()
    module._model = object()
    module._processor = object()
    module._torch = torch
    module._score = lambda *args: 0.75
    sample = Sample(
        path=target,
        is_video=False,
        reference_path=reference,
        reference_mask_path=mask,
    )
    result = module.process(sample)
    assert result.quality_metrics.masc_concept_preservation == pytest.approx(0.75)


def test_masc_finds_reference_mask_sidecar(tmp_path):
    from ayase.modules.masc import MaSCModule

    reference = tmp_path / "reference.jpg"
    mask = tmp_path / "reference.mask.png"
    reference.touch()
    mask.touch()
    sample = Sample(path=reference, is_video=False, reference_path=reference)
    assert MaSCModule()._resolve_reference_mask(sample) == mask
