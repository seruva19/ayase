"""Unit tests for the dense RAFT flow-field primitive (:mod:`ayase.flow`).

The model itself is monkeypatched, so these run on CPU without the RAFT weights;
they pin the wiring (``/8`` crop, tensor layout, ``(H, W, 2)`` float32 output) that
downstream consumers such as subject-region residual flow depend on.
"""

import numpy as np
import pytest

from ayase.flow import _crop_to_multiple_of_8, raft_flow_field


def test_crop_to_multiple_of_8_floors_dims() -> None:
    """Dims not divisible by 8 are floored to the nearest multiple."""
    cropped = _crop_to_multiple_of_8(np.zeros((30, 45, 3), dtype=np.uint8))
    assert cropped.shape == (24, 40, 3)  # 30 -> 24, 45 -> 40


def test_crop_already_multiple_is_unchanged() -> None:
    """Dims already divisible by 8 pass through untouched."""
    assert _crop_to_multiple_of_8(np.zeros((16, 24, 3), dtype=np.uint8)).shape == (16, 24, 3)


def test_raft_flow_field_shape_and_crop(monkeypatch) -> None:
    """Crops to /8, runs the (fake) model, returns channels-last (H, W, 2) float32."""
    torch = pytest.importorskip("torch")
    captured = {}

    class _FakeModel:
        def __call__(self, img1, img2):
            captured["shape"] = tuple(img1.shape)  # (1, 3, H, W)
            h, w = img1.shape[-2], img1.shape[-1]
            flow = torch.arange(2 * h * w, dtype=torch.float32).reshape(1, 2, h, w)
            return [flow]  # torchvision RAFT returns a list of refinement iterations

    def _fake_loader(device="auto", models_dir="models"):
        return _FakeModel(), (lambda a, b: (a.float(), b.float())), "cpu"

    monkeypatch.setattr("ayase.flow.load_raft_flow_model", _fake_loader)

    field = raft_flow_field(
        np.zeros((30, 45, 3), dtype=np.uint8), np.zeros((30, 45, 3), dtype=np.uint8)
    )
    assert field.shape == (24, 40, 2)  # cropped to /8, channels moved last
    assert field.dtype == np.float32
    assert captured["shape"] == (1, 3, 24, 40)  # model saw the /8-cropped tensor


def test_raft_flow_field_rejects_tiny_frames(monkeypatch) -> None:
    """A frame smaller than 8 px on a side after cropping raises rather than calling RAFT."""
    pytest.importorskip("torch")
    monkeypatch.setattr(
        "ayase.flow.load_raft_flow_model",
        lambda device="auto", models_dir="models": (None, None, "cpu"),
    )
    tiny = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        raft_flow_field(tiny, tiny)
