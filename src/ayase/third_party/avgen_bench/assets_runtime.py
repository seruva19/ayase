"""Pinned, automatic pretrained-asset acquisition for AVGen-Bench.

All managed state lives below ``models_dir/.ayase/avgen-bench/assets``.  Assets
are downloaded into a temporary sibling, verified, and published atomically.
Direct publisher URLs without a published cryptographic digest use a recorded
first-download SHA-256; this is intentionally identified as trust-on-first-use
rather than being presented as a publisher checksum.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import urllib.request
import zipfile
import zlib
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping, Optional, Tuple, Union
from urllib.parse import urlsplit

from .protocol import SOURCE_REVISION

Pathish = Union[str, os.PathLike]

_MARKER = ".ayase-asset.json"
_DIMENSION_KEYS = {
    "Vis": ("onealign_model_dir",),
    "Aud": ("audiobox_checkpoint",),
    "AV": ("synchformer_checkpoint", "synchformer_config"),
    "Lip": ("syncnet_checkpoint", "s3fd_checkpoint"),
    "Text": ("ocr_det_dir", "ocr_rec_dir"),
    "Face": ("buffalo_dir",),
    "Music": ("basic_pitch_model",),
    "Speech": ("whisper_model_dir",),
    "Lo-Phy": ("videophy_model_dir",),
    "Hi-Phy": (),
    "Holistic": (),
}


class AssetProvisionError(RuntimeError):
    """An AVGen-Bench asset could not be safely acquired or validated."""


def required_asset_keys(dimension: str) -> Tuple[str, ...]:
    """Return the semantic asset keys consumed by an exact source dimension."""

    try:
        return _DIMENSION_KEYS[dimension]
    except (KeyError, TypeError) as exc:
        raise AssetProvisionError(f"unknown AVGen-Bench source dimension: {dimension!r}") from exc


def _load_manifest() -> Dict[str, Any]:
    path = Path(__file__).with_name("model_manifest.json")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssetProvisionError("cannot read the AVGen-Bench model manifest") from exc
    if (
        manifest.get("schema") != 1
        or manifest.get("source_revision") != SOURCE_REVISION
        or manifest.get("dimensions")
        != {key: list(value) for key, value in _DIMENSION_KEYS.items()}
        or not isinstance(manifest.get("assets"), dict)
    ):
        raise AssetProvisionError("invalid AVGen-Bench model manifest")
    declared = set(manifest["assets"])
    required = {item for values in _DIMENSION_KEYS.values() for item in values}
    if declared != required:
        raise AssetProvisionError("AVGen-Bench model manifest has an incomplete asset key set")
    for key, spec in manifest["assets"].items():
        if not isinstance(spec, dict) or len(_safe_parts(str(spec.get("cache_name", "")))) != 1:
            raise AssetProvisionError(f"invalid cache name in asset manifest: {key}")
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1()
    digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _spec_sha256(spec: Mapping[str, Any]) -> str:
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _safe_parts(name: str) -> Tuple[str, ...]:
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or "\\" in name
        or any(part in {"", ".", ".."} or ":" in part for part in path.parts)
    ):
        raise AssetProvisionError(f"unsafe asset path in manifest or archive: {name!r}")
    return tuple(path.parts)


def _asset_file(root: Path, name: str) -> Path:
    target = root.joinpath(*_safe_parts(name))
    try:
        target.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise AssetProvisionError(f"asset file escapes its cache: {name!r}") from exc
    return target


def _read_marker(root: Path) -> Dict[str, Any]:
    path = root / _MARKER
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssetProvisionError(f"managed asset cache has no valid marker: {root}") from exc
    if not isinstance(data, dict) or data.get("schema") != 1:
        raise AssetProvisionError(f"managed asset cache has an invalid marker: {root}")
    return data


def _write_marker(root: Path, marker: Mapping[str, Any]) -> None:
    (root / _MARKER).write_text(
        json.dumps(dict(marker), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _base_marker(key: str, spec: Mapping[str, Any]) -> Dict[str, Any]:
    marker = {
        "schema": 1,
        "asset_key": key,
        "asset_id": spec["asset_id"],
        "kind": spec["kind"],
        "license": spec["license"],
        "spec_sha256": _spec_sha256(spec),
    }
    if "url" in spec:
        marker["source_url"] = spec["url"]
    if "repo_id" in spec:
        marker.update(repo_id=spec["repo_id"], revision=spec["revision"])
    return marker


def _verify_marker(marker: Mapping[str, Any], key: str, spec: Mapping[str, Any]) -> None:
    expected = _base_marker(key, spec)
    for name, value in expected.items():
        if marker.get(name) != value:
            raise AssetProvisionError(
                f"managed asset cache identity mismatch for {key}: marker field {name!r}"
            )


def _verify_file(path: Path, expected: Mapping[str, Any], label: str) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise AssetProvisionError(f"required asset file is missing or not regular: {label}")
    size = path.stat().st_size
    if size != expected.get("bytes"):
        raise AssetProvisionError(f"asset size mismatch: {label}")
    result: Dict[str, Any] = {"bytes": size}
    if "sha256" in expected:
        actual = _sha256(path)
        if actual != expected["sha256"]:
            raise AssetProvisionError(f"asset SHA-256 mismatch: {label}")
        result["sha256"] = actual
    if "git_blob_sha1" in expected:
        actual = _git_blob_sha1(path)
        if actual != expected["git_blob_sha1"]:
            raise AssetProvisionError(f"asset Git blob mismatch: {label}")
        result["git_blob_sha1"] = actual
    return result


def _verify_managed_file_set(root: Path, expected: set, *, hf_bookkeeping: bool = False) -> None:
    """Reject files that a native loader could prefer over the verified assets."""
    allowed = set(expected) | {_MARKER}
    if hf_bookkeeping:
        allowed.add(".cache/huggingface/.gitignore")
        allowed.update(
            ".cache/huggingface/download/" + name + suffix
            for name in expected
            for suffix in (".metadata", ".lock")
        )
    directories = {
        parent.as_posix()
        for name in allowed
        for parent in PurePosixPath(name).parents
        if parent.as_posix() != "."
    }
    for path in root.rglob("*"):
        name = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise AssetProvisionError(f"managed asset contains a symbolic link: {name}")
        if path.is_dir():
            if name not in directories:
                raise AssetProvisionError(f"managed asset contains an undeclared directory: {name}")
            continue
        if name in allowed:
            continue
        raise AssetProvisionError(f"managed asset contains an undeclared file: {name}")


def _verify_hf_root(root: Path, key: str, spec: Mapping[str, Any]) -> Dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise AssetProvisionError(f"managed Hugging Face asset is not a directory: {root}")
    marker = _read_marker(root)
    _verify_marker(marker, key, spec)
    _verify_managed_file_set(root, set(spec["files"]), hf_bookkeeping=True)
    for name, expected in spec["files"].items():
        _verify_file(_asset_file(root, name), expected, f"{key}/{name}")
    return marker


def _identity(key: str, path: Path, marker: Mapping[str, Any]) -> Dict[str, Any]:
    identity = dict(marker)
    identity.pop("schema", None)
    identity["provenance"] = "managed"
    identity["path"] = str(path.resolve())
    return identity


def _ensure_hf_asset(root: Path, key: str, spec: Mapping[str, Any]) -> Tuple[Path, Dict[str, Any]]:
    final = root / "hf" / spec["cache_name"]
    from ayase.managed_runtime import _SetupLock

    with _SetupLock(root / "locks" / f"{key}.lock"):
        if final.exists():
            marker = _verify_hf_root(final, key, spec)
            return final.resolve(), _identity(key, final, marker)
        final.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(
                prefix=f".{spec['cache_name']}.", dir=final.parent
            ) as tmp:
                work = Path(tmp)
                from ayase.config import download_hf_snapshot

                downloaded = Path(
                    download_hf_snapshot(
                        spec["repo_id"],
                        str(work / "download"),
                        revision=spec["revision"],
                        allow_patterns=list(spec["files"]),
                    )
                ).resolve()
                try:
                    downloaded.relative_to(work.resolve())
                except ValueError as exc:
                    raise AssetProvisionError(
                        "Hugging Face downloader returned an external path"
                    ) from exc
                marker = _base_marker(key, spec)
                _write_marker(downloaded, marker)
                _verify_hf_root(downloaded, key, spec)
                downloaded.replace(final)
        except AssetProvisionError:
            raise
        except Exception as exc:
            raise AssetProvisionError(f"failed to acquire Hugging Face asset {key}") from exc
        return final.resolve(), _identity(key, final, marker)


def _download_url(url: str, destination: Path, expected_bytes: int) -> Dict[str, Optional[str]]:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise AssetProvisionError("AVGen-Bench asset URLs must be credential-free HTTPS URLs")
    request = urllib.request.Request(url, headers={"User-Agent": "Ayase/AVGen-Bench asset manager"})
    size = 0
    with urllib.request.urlopen(request, timeout=300) as response, destination.open("xb") as stream:
        headers = {str(name).lower(): str(value) for name, value in response.headers.items()}
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > expected_bytes:
                raise AssetProvisionError("asset download exceeds its pinned size")
            stream.write(chunk)
    if size != expected_bytes:
        raise AssetProvisionError("asset download has an unexpected size")
    return {"etag": headers.get("etag")}


def _normal_etag(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    if value.startswith("W/"):
        value = value[2:].strip()
    return value.strip('"')


def _verify_direct_root(
    root: Path, key: str, spec: Mapping[str, Any]
) -> Tuple[Path, Dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise AssetProvisionError(f"managed direct asset is not a directory: {root}")
    marker = _read_marker(root)
    _verify_marker(marker, key, spec)
    expected_etag = _normal_etag(spec.get("expected_etag"))
    if expected_etag and _normal_etag(marker.get("observed_etag")) != expected_etag:
        raise AssetProvisionError(f"managed direct asset ETag marker mismatch: {key}")
    path = _asset_file(root, spec["filename"])
    expected = {"bytes": spec["bytes"]}
    if "sha256" in spec:
        expected["sha256"] = spec["sha256"]
    actual = _verify_file(path, expected, key)
    observed = marker.get("observed_sha256")
    verified_sha = actual["sha256"] if "sha256" in actual else _sha256(path)
    if not isinstance(observed, str) or verified_sha != observed:
        raise AssetProvisionError(f"managed direct asset SHA marker mismatch: {key}")
    return path.resolve(), marker


def _ensure_direct_asset(
    root: Path, key: str, spec: Mapping[str, Any]
) -> Tuple[Path, Dict[str, Any]]:
    final = root / "direct" / spec["cache_name"]
    from ayase.managed_runtime import _SetupLock

    with _SetupLock(root / "locks" / f"{key}.lock"):
        if final.exists():
            path, marker = _verify_direct_root(final, key, spec)
            return path, _identity(key, path, marker)
        final.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(
                prefix=f".{spec['cache_name']}.", dir=final.parent
            ) as tmp:
                work = Path(tmp)
                path = _asset_file(work, spec["filename"])
                path.parent.mkdir(parents=True, exist_ok=True)
                response = _download_url(spec["url"], path, int(spec["bytes"]))
                actual_sha = _sha256(path)
                if spec.get("sha256") is not None and actual_sha != spec["sha256"]:
                    raise AssetProvisionError(f"asset SHA-256 mismatch: {key}")
                expected_etag = _normal_etag(spec.get("expected_etag"))
                observed_etag = _normal_etag(response.get("etag"))
                if expected_etag and observed_etag != expected_etag:
                    raise AssetProvisionError(f"asset ETag mismatch: {key}")
                marker = _base_marker(key, spec)
                marker.update(
                    observed_sha256=actual_sha,
                    observed_bytes=path.stat().st_size,
                    observed_etag=observed_etag,
                    digest_authority=("manifest" if spec.get("sha256") else "first_download"),
                )
                _write_marker(work, marker)
                _verify_direct_root(work, key, spec)
                work.replace(final)
        except AssetProvisionError:
            raise
        except Exception as exc:
            raise AssetProvisionError(f"failed to acquire direct asset {key}") from exc
        path = _asset_file(final, spec["filename"])
        return path.resolve(), _identity(key, path, marker)


def _crc32(path: Path) -> str:
    value = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value = zlib.crc32(chunk, value)
    return f"{value & 0xFFFFFFFF:08x}"


def _extract_zip(archive: Path, root: Path, spec: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    expected = spec["members"]
    output = root.joinpath(*_safe_parts(spec["output_subdir"]))
    output.mkdir(parents=True)
    results: Dict[str, Dict[str, Any]] = {}
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        if len({entry.filename for entry in entries}) != len(entries):
            raise AssetProvisionError("archive contains duplicate paths")
        files: Dict[str, zipfile.ZipInfo] = {}
        for entry in entries:
            parts = _safe_parts(entry.filename.rstrip("/"))
            mode = entry.external_attr >> 16
            if entry.is_dir():
                continue
            if stat.S_IFMT(mode) not in {0, stat.S_IFREG} or entry.flag_bits & 1:
                raise AssetProvisionError("archive contains a non-regular or encrypted entry")
            if len(parts) != 1:
                raise AssetProvisionError("buffalo_l archive has an unexpected nested path")
            files[parts[0]] = entry
        if set(files) != set(expected):
            raise AssetProvisionError("buffalo_l archive member set does not match the manifest")
        for name, details in expected.items():
            entry = files[name]
            if entry.file_size != details["bytes"] or f"{entry.CRC:08x}" != details["crc32"]:
                raise AssetProvisionError(f"buffalo_l archive metadata mismatch: {name}")
            target = _asset_file(output, name)
            with bundle.open(entry) as source, target.open("xb") as destination:
                shutil.copyfileobj(source, destination)
            if target.stat().st_size != details["bytes"] or _crc32(target) != details["crc32"]:
                raise AssetProvisionError(f"buffalo_l extracted member mismatch: {name}")
            results[name] = {
                "bytes": target.stat().st_size,
                "crc32": details["crc32"],
                "sha256": _sha256(target),
            }
    return results


def _verify_zip_root(root: Path, key: str, spec: Mapping[str, Any]) -> Dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise AssetProvisionError(f"managed archive asset is not a directory: {root}")
    marker = _read_marker(root)
    _verify_marker(marker, key, spec)
    recorded = marker.get("members")
    if not isinstance(recorded, dict) or set(recorded) != set(spec["members"]):
        raise AssetProvisionError("archive cache marker has an invalid member set")
    pack = root.joinpath(*_safe_parts(spec["output_subdir"]))
    _verify_managed_file_set(
        root,
        {"/".join((*_safe_parts(spec["output_subdir"]), name)) for name in spec["members"]},
    )
    for name, expected in spec["members"].items():
        details = recorded[name]
        if not isinstance(details, dict) or details.get("crc32") != expected["crc32"]:
            raise AssetProvisionError(f"archive cache marker mismatch: {name}")
        path = _asset_file(pack, name)
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != expected["bytes"]
            or _crc32(path) != expected["crc32"]
            or _sha256(path) != details.get("sha256")
        ):
            raise AssetProvisionError(f"archive cache member failed verification: {name}")
    return marker


def _ensure_zip_asset(root: Path, key: str, spec: Mapping[str, Any]) -> Tuple[Path, Dict[str, Any]]:
    final = root / "archives" / spec["cache_name"]
    from ayase.managed_runtime import _SetupLock

    with _SetupLock(root / "locks" / f"{key}.lock"):
        if final.exists():
            marker = _verify_zip_root(final, key, spec)
            return final.resolve(), _identity(key, final, marker)
        final.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(
                prefix=f".{spec['cache_name']}.", dir=final.parent
            ) as tmp:
                work = Path(tmp)
                archive = work / "asset.zip"
                _download_url(spec["url"], archive, int(spec["bytes"]))
                archive_sha = _sha256(archive)
                extracted = work / "extracted"
                extracted.mkdir()
                members = _extract_zip(archive, extracted, spec)
                marker = _base_marker(key, spec)
                marker.update(
                    observed_archive_sha256=archive_sha,
                    observed_archive_bytes=archive.stat().st_size,
                    digest_authority="first_download",
                    members=members,
                )
                _write_marker(extracted, marker)
                _verify_zip_root(extracted, key, spec)
                extracted.replace(final)
        except AssetProvisionError:
            raise
        except Exception as exc:
            raise AssetProvisionError(f"failed to acquire archive asset {key}") from exc
        return final.resolve(), _identity(key, final, marker)


def _directory_override_files(path: Path, spec: Mapping[str, Any]) -> Dict[str, Path]:
    if spec["kind"] == "hf_snapshot":
        return {name: _asset_file(path, name) for name in spec["files"]}
    if spec["kind"] == "zip_directory":
        if (path / "models" / "buffalo_l").is_dir():
            pack = path / "models" / "buffalo_l"
        elif path.name == "buffalo_l" and path.parent.name == "models":
            pack = path
        else:
            raise AssetProvisionError(
                "buffalo_dir override must be an InsightFace root containing models/buffalo_l "
                "or that models/buffalo_l directory"
            )
        return {name: _asset_file(pack, name) for name in spec["members"]}
    raise AssetProvisionError("override asset is not a directory")


def _override_identity(
    key: str, raw: Pathish, spec: Mapping[str, Any]
) -> Tuple[Path, Dict[str, Any]]:
    try:
        path = Path(raw).expanduser().resolve()
    except (TypeError, ValueError, OSError) as exc:
        raise AssetProvisionError(f"expert override for {key} is not a valid path") from exc
    expected_directory = spec["kind"] in {"hf_snapshot", "zip_directory"}
    if expected_directory:
        if not path.is_dir():
            raise AssetProvisionError(f"expert override for {key} is not an existing directory")
        files = _directory_override_files(path, spec)
        evidence: Dict[str, Any] = {}
        for name, target in files.items():
            if not target.is_file():
                raise AssetProvisionError(f"expert override for {key} is missing {name}")
            evidence[name] = {"bytes": target.stat().st_size, "sha256": _sha256(target)}
        tree_payload = json.dumps(evidence, sort_keys=True, separators=(",", ":"))
        actual_sha = hashlib.sha256(tree_payload.encode("utf-8")).hexdigest()
        details: Dict[str, Any] = {"tree_sha256": actual_sha, "files": evidence}
    else:
        if not path.is_file():
            raise AssetProvisionError(f"expert override for {key} is not an existing file")
        details = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    identity = {
        "asset_key": key,
        "asset_id": spec["asset_id"],
        "provenance": "expert_override",
        "path": str(path),
        "license": spec["license"],
        "verification": "existence, runtime layout, and actual SHA-256 only",
        "published_asset_parity": False,
        "actual": details,
    }
    return path, identity


def ensure_assets(
    dimension: str,
    models_dir: Pathish,
    *,
    overrides: Optional[Mapping[str, Pathish]] = None,
    test_mode: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Return verified local paths and provenance for one source dimension.

    ``dimension`` is an exact released source key (for example ``"Vis"`` or
    ``"Lo-Phy"``).  Expert overrides are optional and may cover any subset of
    that dimension's semantic asset keys.  They are hashed and explicitly
    marked as overrides, but are not claimed to match the published bytes.

    Test mode deliberately returns before manifest, path, lock, or network I/O.
    """

    dimension_keys = required_asset_keys(dimension)
    if test_mode:
        return {"paths": {}, "asset_identities": {}}
    if overrides is not None and not isinstance(overrides, Mapping):
        raise AssetProvisionError("asset overrides must be a mapping")
    supplied = dict(overrides or {})
    allowed = set(dimension_keys)
    unknown = set(supplied) - allowed
    if unknown:
        raise AssetProvisionError(
            f"asset overrides do not belong to dimension {dimension}: {sorted(unknown)!r}"
        )

    manifest = _load_manifest()
    cache_root = Path(models_dir).expanduser().resolve() / ".ayase" / "avgen-bench" / "assets"
    paths: Dict[str, Path] = {}
    identities: Dict[str, Dict[str, Any]] = {}
    for key in dimension_keys:
        spec = manifest["assets"][key]
        if key in supplied:
            path, identity = _override_identity(key, supplied[key], spec)
        elif spec["kind"] == "hf_snapshot":
            path, identity = _ensure_hf_asset(cache_root, key, spec)
        elif spec["kind"] == "direct_file":
            path, identity = _ensure_direct_asset(cache_root, key, spec)
        elif spec["kind"] == "zip_directory":
            path, identity = _ensure_zip_asset(cache_root, key, spec)
        else:
            raise AssetProvisionError(f"unsupported asset kind for {key}: {spec.get('kind')!r}")
        paths[key] = path
        identities[key] = identity
    return {"paths": paths, "asset_identities": identities}


__all__ = ["AssetProvisionError", "ensure_assets", "required_asset_keys"]
