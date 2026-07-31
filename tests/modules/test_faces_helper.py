"""Shared face-detection helper: padded retry for tightly cropped faces."""

import numpy as np

from ayase.faces import detect_largest_face


class _FakeFace:
    def __init__(self, bbox):
        self.bbox = bbox


class _Detector:
    """Detects faces only above a minimum frame size (tight-crop simulation)."""

    def __init__(self, min_size=0, faces=None):
        self.min_size = min_size
        self.faces = faces if faces is not None else [_FakeFace((0, 0, 10, 10))]
        self.calls = []

    def get(self, image):
        self.calls.append(image.shape[:2])
        return list(self.faces) if image.shape[0] > self.min_size else []


def test_detect_returns_largest_face_without_padding():
    small = _FakeFace((0, 0, 10, 10))
    large = _FakeFace((0, 0, 100, 100))
    det = _Detector(faces=[small, large])
    frame = np.zeros((256, 256, 3), dtype=np.uint8)

    face, image = detect_largest_face(det, frame)

    assert face is large
    assert image is frame
    assert len(det.calls) == 1


def test_detect_retries_on_padded_copy():
    det = _Detector(min_size=112)
    frame = np.zeros((112, 112, 3), dtype=np.uint8)

    face, image = detect_largest_face(det, frame, pad_retry=0.25)

    assert face is not None
    assert image.shape[0] == 112 + 2 * 28
    assert det.calls == [(112, 112), (168, 168)]


def test_detect_retry_can_be_disabled():
    det = _Detector(min_size=112)
    frame = np.zeros((112, 112, 3), dtype=np.uint8)

    face, image = detect_largest_face(det, frame, pad_retry=0.0)

    assert face is None
    assert image is frame
    assert det.calls == [(112, 112)]


def test_detect_returns_none_when_padding_does_not_help():
    det = _Detector(min_size=10_000)
    frame = np.zeros((112, 112, 3), dtype=np.uint8)

    face, image = detect_largest_face(det, frame)

    assert face is None
    assert image is frame
    assert len(det.calls) == 2
