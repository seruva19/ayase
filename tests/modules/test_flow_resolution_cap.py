"""Regression tests for the RAFT flow resolution cap.

Full-resolution HD frames make RAFT's correlation volume explode (observed: a
single 1080p pair tried to allocate 62 GiB -> CUDA OOM -> metric silently None).
`_cap_frame_resolution` downscales oversized frames before RAFT while leaving
smaller frames untouched. These tests exercise the helper without GPU/RAFT.
"""

import numpy as np
import pytest

from ayase.modules.advanced_flow import _cap_frame_resolution as cap_flow
from ayase.modules.motion_amplitude import _cap_frame_resolution as cap_motion

CAPS = [cap_flow, cap_motion]


@pytest.mark.parametrize("cap", CAPS)
def test_downscales_hd_frame(cap):
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    out = cap(frame, 512)
    h, w = out.shape[:2]
    assert max(h, w) <= 512
    assert h % 8 == 0 and w % 8 == 0
    # aspect ratio preserved (within /8 rounding)
    assert abs((w / h) - (1920 / 1080)) < 0.05


@pytest.mark.parametrize("cap", CAPS)
def test_small_frame_unchanged(cap):
    frame = np.zeros((256, 300, 3), dtype=np.uint8)
    out = cap(frame, 512)
    assert out.shape == frame.shape


@pytest.mark.parametrize("cap", CAPS)
def test_disabled_when_max_side_zero(cap):
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    out = cap(frame, 0)
    assert out.shape == frame.shape


@pytest.mark.parametrize("cap", CAPS)
def test_square_hd_capped_to_multiple_of_8(cap):
    frame = np.zeros((1080, 1080, 3), dtype=np.uint8)
    out = cap(frame, 512)
    h, w = out.shape[:2]
    assert max(h, w) <= 512
    assert h % 8 == 0 and w % 8 == 0
