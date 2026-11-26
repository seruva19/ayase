"""Tests for psnr99 module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics
from ayase.pipeline import Pipeline


def test_psnr99_basics():
    from ayase.modules.psnr99 import PSNR99Module
    _test_module_basics(PSNR99Module, "psnr99")


def test_psnr99_shared_image_video_field_is_adapted_and_opt_in():
    from ayase.modules.psnr99 import PSNR99Module

    assert PSNR99Module.field_provenance()["psnr99"] == "adapted"
    assert "psnr99" in Pipeline([PSNR99Module()])._provenance_excluded
    assert "psnr99" not in Pipeline(
        [PSNR99Module()], allow_provenance=["adapted"]
    )._provenance_excluded

def test_psnr99_no_reference(image_sample):
    from ayase.modules.psnr99 import PSNR99Module
    m = PSNR99Module()
    result = m.process(image_sample)
    assert result is image_sample


def test_psnr99_video_keeps_pair_indices_when_reference_read_fails(monkeypatch):
    import cv2
    import numpy as np
    from ayase.modules.psnr99 import PSNR99Module

    released = []

    class Capture:
        def __init__(self, path):
            self.path = path

        def get(self, prop):
            assert prop == cv2.CAP_PROP_FRAME_COUNT
            return 7 if self.path == "sample" else 5

        def set(self, prop, index):
            assert prop == cv2.CAP_PROP_POS_FRAMES
            self.index = index

        def read(self):
            if self.path == "reference" and self.index == 2:
                return False, None
            return True, np.full((1, 1, 3), self.index, dtype=np.uint8)

        def release(self):
            released.append(self.path)

    pairs = []

    def score(frame, reference):
        pairs.append((int(frame[0, 0, 0]), int(reference[0, 0, 0])))
        return float(frame[0, 0, 0])

    monkeypatch.setattr(cv2, "VideoCapture", Capture)
    module = PSNR99Module({"subsample": 3})
    monkeypatch.setattr(module, "_block_psnr99", score)
    assert module._score_video("sample", "reference") == 2.0
    assert pairs == [(0, 0), (4, 4)]
    assert set(released) == {"sample", "reference"}


def test_psnr99_measures_errors_outside_first_column():
    import cv2
    import numpy as np
    from ayase.modules.psnr99 import PSNR99Module

    reference = np.zeros((20, 20, 3), dtype=np.uint8)
    distorted = reference.copy()
    distorted[5:7, 8:10] = 255
    # Four damaged pixels are exactly the worst one percent. OpenCV's
    # limited-range I420 conversion maps black/white luma to 16/235.
    assert cv2.cvtColor(distorted, cv2.COLOR_BGR2YUV_IYUV)[5, 8] == 235
    expected = 10 * np.log10(255 ** 2 / (235 - 16) ** 2)
    assert np.isclose(PSNR99Module()._block_psnr99(distorted, reference), expected)
