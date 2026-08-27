"""Tests for pc_psnr module."""

import sys
from types import SimpleNamespace

import numpy as np

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_pc_psnr_basics():
    from ayase.modules.pc_psnr import PCPSNRModule
    _test_module_basics(PCPSNRModule, "pc_psnr")

def test_pc_psnr_no_reference(image_sample):
    from ayase.modules.pc_psnr import PCPSNRModule
    m = PCPSNRModule()
    result = m.process(image_sample)
    assert result is image_sample


def test_pc_psnr_preserves_d1_when_reference_has_no_normals(monkeypatch):
    from ayase.modules.pc_psnr import PCPSNRModule

    points = [
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]),
    ]

    class FakePointCloud:
        def __init__(self, values):
            self.points = values

        def has_normals(self):
            return False

    fake_open3d = SimpleNamespace(
        io=SimpleNamespace(read_point_cloud=lambda _path: FakePointCloud(points.pop(0)))
    )
    monkeypatch.setitem(sys.modules, "open3d", fake_open3d)

    d1, d2 = PCPSNRModule()._compute("sample.ply", "reference.ply")

    assert np.isfinite(d1)
    assert d2 is None
