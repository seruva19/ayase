"""Pinned package cache and private build contracts without downloads or installs."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def package_runtime():
    from ayase.third_party.avgen_bench import packaging_runtime

    return packaging_runtime


@pytest.fixture
def cached_wheel(tmp_path, monkeypatch, package_runtime):
    from ayase import managed_runtime

    payload = b"publisher wheel fixture"
    artifact = {
        "kind": "bdist_wheel",
        "filename": "example-1.0-py3-none-any.whl",
        "url": "https://publisher.example/example-1.0-py3-none-any.whl",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    downloads = []

    def download(expected, destination):
        downloads.append(expected)
        destination.write_bytes(payload)

    def unexpected(*args, **kwargs):
        pytest.fail("cache tests must not provision an environment or build a wheel")

    monkeypatch.setattr(package_runtime, "_download", download)
    monkeypatch.setattr(managed_runtime, "ensure_managed_environment", unexpected)
    monkeypatch.setattr(package_runtime.subprocess, "run", unexpected)
    models = tmp_path / "models"
    wheel = package_runtime.ensure_package_wheel("example", artifact, models)
    assert downloads == [artifact]

    def no_redownload(*args, **kwargs):
        pytest.fail("a damaged or mismatched cache must not silently become a new download")

    monkeypatch.setattr(package_runtime, "_download", no_redownload)
    return SimpleNamespace(
        wheel=wheel,
        marker=wheel.parent / "manifest.json",
        artifact=artifact,
        payload=payload,
        models=models,
    )


def _change_marker(cached, **changes):
    document = json.loads(cached.marker.read_text(encoding="utf-8"))
    document.update(changes)
    cached.marker.write_text(json.dumps(document), encoding="utf-8")


def test_publisher_wheel_is_reused_without_download_or_private_builder(
    cached_wheel, package_runtime
):
    result = package_runtime.ensure_package_wheel(
        "example", cached_wheel.artifact, cached_wheel.models
    )

    assert result == cached_wheel.wheel
    assert result.read_bytes() == cached_wheel.payload


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("source_sha256", "0" * 64),
        ("source_url", "https://other-publisher.example/unrelated.whl"),
        ("build_tools", ["setuptools==99.0.0", "wheel==0.45.1"]),
    ],
)
def test_cached_wheel_rejects_tampered_source_or_build_identity(
    cached_wheel, package_runtime, field, replacement
):
    _change_marker(cached_wheel, **{field: replacement})

    with pytest.raises(ValueError, match="source identity"):
        package_runtime.ensure_package_wheel("example", cached_wheel.artifact, cached_wheel.models)


@pytest.mark.parametrize("path_kind", ["parent", "backslash_parent", "absolute", "nested"])
def test_cached_wheel_rejects_existing_file_outside_its_safe_basename(
    cached_wheel, package_runtime, path_kind
):
    if path_kind == "nested":
        target = cached_wheel.wheel.parent / "nested" / "other.whl"
        marker_name = "nested/other.whl"
    else:
        target = cached_wheel.wheel.parent.parent / "outside.whl"
        marker_name = {
            "parent": "../outside.whl",
            "backslash_parent": "..\\outside.whl",
            "absolute": str(target.resolve()),
        }[path_kind]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(cached_wheel.payload)
    _change_marker(cached_wheel, filename=marker_name)

    with pytest.raises(ValueError, match="wheel path"):
        package_runtime.ensure_package_wheel("example", cached_wheel.artifact, cached_wheel.models)


@pytest.mark.parametrize("filename", [None, "", "other:stream.whl", "not-a-wheel.tar.gz"])
def test_cached_wheel_rejects_unsafe_or_non_wheel_marker_names(
    cached_wheel, package_runtime, filename
):
    _change_marker(cached_wheel, filename=filename)

    with pytest.raises(ValueError, match="wheel path"):
        package_runtime.ensure_package_wheel("example", cached_wheel.artifact, cached_wheel.models)


def test_cached_wheel_rejects_damaged_payload(cached_wheel, package_runtime):
    cached_wheel.wheel.write_bytes(b"damaged bytes")

    with pytest.raises(ValueError, match="damaged"):
        package_runtime.ensure_package_wheel("example", cached_wheel.artifact, cached_wheel.models)


@pytest.mark.parametrize("alteration", ["sha256", "size", "filename"])
def test_prebuilt_cache_still_matches_publisher_with_self_consistent_local_marker(
    cached_wheel, package_runtime, alteration
):
    if alteration == "filename":
        wheel = cached_wheel.wheel.with_name("unrelated-2.0-py3-none-any.whl")
        cached_wheel.wheel.rename(wheel)
    else:
        wheel = cached_wheel.wheel
        payload = (
            bytes([cached_wheel.payload[0] ^ 1]) + cached_wheel.payload[1:]
            if alteration == "sha256"
            else cached_wheel.payload + b"extra bytes"
        )
        wheel.write_bytes(payload)
    _change_marker(
        cached_wheel,
        filename=wheel.name,
        sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
    )

    with pytest.raises(ValueError, match="publisher artifact"):
        package_runtime.ensure_package_wheel("example", cached_wheel.artifact, cached_wheel.models)


def test_package_test_mode_exits_before_any_path_or_provisioning_io(
    tmp_path, monkeypatch, package_runtime
):
    from ayase import managed_runtime

    def unexpected(*args, **kwargs):
        pytest.fail("test mode must exit before package or filesystem access")

    monkeypatch.setattr(package_runtime, "Path", unexpected)
    monkeypatch.setattr(package_runtime, "_download", unexpected)
    monkeypatch.setattr(package_runtime.tempfile, "TemporaryDirectory", unexpected)
    monkeypatch.setattr(package_runtime.subprocess, "run", unexpected)
    monkeypatch.setattr(managed_runtime, "ensure_managed_environment", unexpected)

    with pytest.raises(RuntimeError, match="test mode"):
        package_runtime.ensure_package_wheel("example", {}, tmp_path / "models", test_mode=True)
    assert not (tmp_path / "models").exists()


@pytest.fixture
def source_build(tmp_path, monkeypatch, package_runtime):
    from ayase import managed_runtime

    archive = b"pinned pure Python source fixture"
    artifact = {
        "kind": "sdist",
        "filename": "legacy-1.0.tar.gz",
        "url": "https://publisher.example/legacy-1.0.tar.gz",
        "bytes": len(archive),
        "sha256": hashlib.sha256(archive).hexdigest(),
    }
    builder = tmp_path / "private build tools" / "python.exe"
    provisions = []

    def provision(*args, **kwargs):
        provisions.append((args, kwargs))
        return builder

    monkeypatch.setattr(
        package_runtime, "_download", lambda expected, path: path.write_bytes(archive)
    )
    monkeypatch.setattr(managed_runtime, "ensure_managed_environment", provision)
    return SimpleNamespace(
        artifact=artifact,
        archive=archive,
        builder=builder,
        provisions=provisions,
        models=tmp_path / "models",
    )


def test_source_build_uses_compatible_private_tools_and_publishes_reusable_wheel(
    source_build, package_runtime, monkeypatch
):
    builds = []

    def build(command, **kwargs):
        builds.append((command, kwargs))
        assert command[:4] == [str(source_build.builder), "-m", "pip", "wheel"]
        assert "--no-build-isolation" in command
        assert "--no-deps" in command
        assert Path(command[-1]).read_bytes() == source_build.archive
        output = Path(command[command.index("--wheel-dir") + 1])
        assert output.is_relative_to(source_build.models)
        assert Path(kwargs["env"]["TEMP"]) == output.parent
        assert Path(kwargs["env"]["TMP"]) == output.parent
        (output / "legacy-1.0-py3-none-any.whl").write_bytes(b"built pure Python wheel")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(package_runtime.subprocess, "run", build)
    wheel = package_runtime.ensure_package_wheel(
        "legacy", source_build.artifact, source_build.models
    )

    assert wheel.read_bytes() == b"built pure Python wheel"
    assert len(source_build.provisions) == 1
    args, kwargs = source_build.provisions[0]
    assert args == (
        "avgen-build-tools",
        ("setuptools==70.3.0", "wheel==0.45.1"),
        source_build.models,
    )
    assert kwargs["python_version"] == "3.10.19"
    assert (
        package_runtime.ensure_package_wheel("legacy", source_build.artifact, source_build.models)
        == wheel
    )
    assert len(builds) == 1


def test_failed_source_build_retains_log_without_publishing_cache(
    source_build, package_runtime, monkeypatch
):
    def fail_build(command, **kwargs):
        kwargs["stdout"].write("deliberate private build failure\n")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(package_runtime.subprocess, "run", fail_build)
    with pytest.raises(RuntimeError, match="wheel build failed; see"):
        package_runtime.ensure_package_wheel("legacy", source_build.artifact, source_build.models)

    logs = list(source_build.models.rglob("*.log"))
    assert len(logs) == 1
    assert logs[0].read_text(encoding="utf-8") == "deliberate private build failure\n"
    assert not list(source_build.models.rglob("manifest.json"))
    assert not list(source_build.models.rglob("*.whl"))


@pytest.mark.parametrize("wheel_count", [0, 2])
def test_successful_build_requires_one_wheel_before_publishing_cache(
    source_build, package_runtime, monkeypatch, wheel_count
):
    def incomplete_build(command, **kwargs):
        output = Path(command[command.index("--wheel-dir") + 1])
        for index in range(wheel_count):
            (output / f"legacy-{index}-py3-none-any.whl").write_bytes(b"ambiguous wheel")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(package_runtime.subprocess, "run", incomplete_build)
    with pytest.raises(ValueError, match="no unique wheel"):
        package_runtime.ensure_package_wheel("legacy", source_build.artifact, source_build.models)
    assert not list(source_build.models.rglob("manifest.json"))
    assert not list(source_build.models.rglob("*.whl"))
