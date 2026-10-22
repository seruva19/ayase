"""Tests for privately provisioned canonical metric executables."""

import os
import lzma
import struct
import zlib
from pathlib import Path


def _lzip_member(payload: bytes) -> bytes:
    dictionary_code = 12
    compressor = lzma.LZMACompressor(
        format=lzma.FORMAT_RAW,
        filters=[
            {"id": lzma.FILTER_LZMA1, "dict_size": 1 << dictionary_code, "lc": 3, "lp": 0, "pb": 2}
        ],
    )
    compressed = compressor.compress(payload) + compressor.flush()
    size = 6 + len(compressed) + 20
    return (
        b"LZIP\x01"
        + bytes([dictionary_code])
        + compressed
        + struct.pack("<IQQ", zlib.crc32(payload), len(payload), size)
    )


def test_lzip_decoder_validates_concatenated_members_and_checksum():
    import pytest

    from ayase.managed_native_tools import _decode_lzip

    archive = _lzip_member(b"first") + _lzip_member(b"second")
    assert _decode_lzip(archive) == b"firstsecond"
    corrupted = bytearray(archive)
    corrupted[-20] ^= 1
    with pytest.raises(RuntimeError, match="checksum"):
        _decode_lzip(bytes(corrupted))


def test_resolver_provisions_private_tool_without_path_mutation(tmp_path, monkeypatch):
    from ayase import managed_native_tools

    prefix = tmp_path / "prefix"
    python = prefix / ("python.exe" if os.name == "nt" else "bin/python")
    bin_dir = prefix / "Library/bin" if os.name == "nt" else prefix / "bin"
    bin_dir.mkdir(parents=True)
    tool = bin_dir / ("butteraugli_main.exe" if os.name == "nt" else "butteraugli_main")
    tool.touch()
    before = os.environ.get("PATH")
    monkeypatch.setattr(managed_native_tools.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        "ayase.managed_native.ensure_managed_native_environment",
        lambda *args, **kwargs: python,
    )

    resolved = managed_native_tools.resolve_native_tool(
        "butteraugli_main", ("libjxl-tools=0.12.0",), tmp_path
    )

    assert resolved == tool.resolve()
    assert os.environ.get("PATH") == before


def test_butteraugli_cli_parses_authoritative_score(tmp_path, monkeypatch):
    import subprocess

    import numpy as np

    from ayase.modules.butteraugli import ButteraugliModule

    module = ButteraugliModule({"test_mode": True})
    module._butteraugli_path = tmp_path / "butteraugli_main"
    monkeypatch.setattr(
        "ayase.modules.butteraugli.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, "1.234567\n3-norm: 0.75\n", ""
        ),
    )
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    assert module._compute_cli(image, image) == 1.234567


def test_camera_setup_auto_provisions_colmap_and_glomap(tmp_path, monkeypatch):
    from ayase.modules.camera_trajectory import CameraTrajectoryModule
    from ayase.pipeline import PipelineModule

    monkeypatch.setattr(PipelineModule, "_global_test_mode", False)
    monkeypatch.setattr(CameraTrajectoryModule, "_detect_sfm_binaries", staticmethod(lambda: None))
    monkeypatch.setattr("ayase.runtime.resolve_torch_device", lambda value: "cpu")
    monkeypatch.setattr(
        "ayase.managed_native_tools.resolve_native_tool",
        lambda name, *args, **kwargs: tmp_path / name,
    )
    module = CameraTrajectoryModule({"models_dir": str(tmp_path), "pose_backend": "auto"})

    module.setup()

    assert module._backend == "glomap"
    assert module._sfm_bin == {
        "colmap": str(tmp_path / "colmap"),
        "glomap": str(tmp_path / "glomap"),
    }
