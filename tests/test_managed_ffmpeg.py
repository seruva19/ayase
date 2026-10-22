"""Contracts for the feature-aware FFmpeg resolver."""

import subprocess
import sys
import ast
from pathlib import Path
from types import SimpleNamespace


def test_filter_probe_requires_exact_advertised_filter(monkeypatch):
    from ayase import managed_ffmpeg

    managed_ffmpeg.ffmpeg_has_filter.cache_clear()
    monkeypatch.setattr(
        managed_ffmpeg.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, stdout=" ... libvmaf VV->V Calculate VMAF\n", stderr=""
        ),
    )
    assert managed_ffmpeg.ffmpeg_has_filter("ffmpeg") is True

    managed_ffmpeg.ffmpeg_has_filter.cache_clear()
    monkeypatch.setattr(
        managed_ffmpeg.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, stdout=" ... scale VV->V mentions libvmaf in help\n", stderr=""
        ),
    )
    assert managed_ffmpeg.ffmpeg_has_filter("ffmpeg") is False


def test_probe_failure_is_unavailable_and_cached(monkeypatch):
    from ayase import managed_ffmpeg

    calls = []
    managed_ffmpeg.ffmpeg_has_filter.cache_clear()

    def fail(*_args, **_kwargs):
        calls.append(True)
        raise subprocess.TimeoutExpired("ffmpeg", 15)

    monkeypatch.setattr(managed_ffmpeg.subprocess, "run", fail)
    assert managed_ffmpeg.ffmpeg_has_filter("broken") is False
    assert managed_ffmpeg.ffmpeg_has_filter("broken") is False
    assert len(calls) == 1


def test_resolver_prefers_configured_feature_complete_binary(tmp_path, monkeypatch):
    from ayase import managed_ffmpeg

    configured = tmp_path / "ffmpeg.exe"
    configured.touch()
    monkeypatch.setattr(
        managed_ffmpeg, "ffmpeg_has_filter", lambda path, _filter: path == str(configured.resolve())
    )
    assert managed_ffmpeg.resolve_ffmpeg(configured) == configured.resolve()


def test_resolver_rejects_packaged_binary_without_libvmaf(tmp_path, monkeypatch):
    import pytest
    from ayase import managed_ffmpeg
    from ayase.pipeline import PipelineModule

    packaged = tmp_path / "ffmpeg.exe"
    packaged.touch()
    monkeypatch.setattr(PipelineModule, "_global_test_mode", False)
    monkeypatch.delenv("AYASE_TEST_MODE", raising=False)
    monkeypatch.setattr(managed_ffmpeg.shutil, "which", lambda _name: None)
    monkeypatch.setattr(managed_ffmpeg, "ffmpeg_has_filter", lambda *_args: False)
    monkeypatch.setitem(
        sys.modules,
        "imageio_ffmpeg",
        SimpleNamespace(get_ffmpeg_exe=lambda: str(packaged)),
    )
    monkeypatch.setattr(managed_ffmpeg, "_native_tool", lambda *_args: packaged)
    with pytest.raises(RuntimeError, match="does not provide"):
        managed_ffmpeg.resolve_ffmpeg()


def test_resolver_uses_packaged_binary_without_path_mutation(tmp_path, monkeypatch):
    from ayase import managed_ffmpeg

    packaged = tmp_path / "ffmpeg.exe"
    packaged.touch()
    original_path = __import__("os").environ.get("PATH")
    monkeypatch.setattr(managed_ffmpeg.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        managed_ffmpeg, "ffmpeg_has_filter", lambda path, _filter: path == str(packaged.resolve())
    )
    monkeypatch.setitem(
        sys.modules,
        "imageio_ffmpeg",
        SimpleNamespace(get_ffmpeg_exe=lambda: str(packaged)),
    )
    assert managed_ffmpeg.resolve_ffmpeg() == packaged.resolve()
    assert __import__("os").environ.get("PATH") == original_path


def test_ffprobe_uses_private_native_tool_without_path_mutation(tmp_path, monkeypatch):
    from ayase import managed_ffmpeg
    from ayase.pipeline import PipelineModule

    private = tmp_path / ("ffprobe.exe" if __import__("os").name == "nt" else "ffprobe")
    private.touch()
    monkeypatch.setattr(PipelineModule, "_global_test_mode", False)
    monkeypatch.delenv("AYASE_TEST_MODE", raising=False)
    original_path = __import__("os").environ.get("PATH")
    monkeypatch.setattr(managed_ffmpeg.shutil, "which", lambda _name: None)
    monkeypatch.setattr(managed_ffmpeg, "_native_tool", lambda name, models_dir: private)

    assert managed_ffmpeg.resolve_media_tool("ffprobe") == private.resolve()
    assert __import__("os").environ.get("PATH") == original_path


def test_native_tool_resolves_executable_beside_private_prefix(tmp_path, monkeypatch):
    from ayase import managed_ffmpeg

    prefix = tmp_path / "native"
    interpreter = prefix / ("python.exe" if __import__("os").name == "nt" else "bin/python")
    tool_dir = prefix / "Library/bin" if __import__("os").name == "nt" else prefix / "bin"
    tool_dir.mkdir(parents=True)
    tool = tool_dir / ("ffprobe.exe" if __import__("os").name == "nt" else "ffprobe")
    tool.touch()
    monkeypatch.setattr(
        "ayase.managed_native.ensure_managed_native_environment",
        lambda *args, **kwargs: interpreter,
    )
    assert managed_ffmpeg._native_tool("ffprobe", tmp_path) == tool.resolve()


def test_test_mode_never_provisions_missing_media_tool(monkeypatch):
    import pytest
    from ayase import managed_ffmpeg
    from ayase.pipeline import PipelineModule

    monkeypatch.setattr(managed_ffmpeg.shutil, "which", lambda _name: None)
    monkeypatch.setitem(
        sys.modules,
        "imageio_ffmpeg",
        SimpleNamespace(get_ffmpeg_exe=lambda: (_ for _ in ()).throw(RuntimeError("missing"))),
    )
    monkeypatch.setattr(PipelineModule, "_global_test_mode", True)

    def unexpected_provision(*_args, **_kwargs):
        raise AssertionError("native provisioning must not run in test mode")

    monkeypatch.setattr(managed_ffmpeg, "_native_tool", unexpected_provision)
    with pytest.raises(RuntimeError, match="unavailable in Ayase test mode"):
        managed_ffmpeg.resolve_media_tool("ffprobe")
    with pytest.raises(RuntimeError, match="provisioning is disabled"):
        managed_ffmpeg.resolve_media_tool("ffmpeg")


def test_vendored_t2v_metrics_resolves_ffmpeg_without_importing_vendor_code():
    source = (Path(__file__).parents[1] / "src/ayase/vendor/t2v_metrics/__init__.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)

    imports_resolver = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "ayase.managed_ffmpeg"
        and any(alias.name == "resolve_media_tool" for alias in node.names)
        for node in tree.body
    )
    resolved_ffmpeg_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "resolve_media_tool"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "ffmpeg"
    ]
    assert imports_resolver
    assert resolved_ffmpeg_calls
