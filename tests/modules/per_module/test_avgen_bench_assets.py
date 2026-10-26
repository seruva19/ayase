import hashlib
import json
import shutil
import zipfile
import zlib
from pathlib import Path

import pytest


def _asset_module():
    from ayase.third_party.avgen_bench import assets_runtime

    return assets_runtime


def _blob_sha1(payload: bytes) -> str:
    return hashlib.sha1(f"blob {len(payload)}\0".encode("ascii") + payload).hexdigest()


def _manifest_for(key: str, spec: dict) -> dict:
    return {"assets": {key: spec}}


def _direct_spec(payload: bytes, *, pinned: bool = True) -> dict:
    spec = {
        "kind": "direct_file",
        "cache_name": "test-direct",
        "filename": "checkpoint.bin",
        "url": "https://publisher.example/releases/v1/checkpoint.bin",
        "bytes": len(payload),
        "asset_id": "publisher/checkpoint@v1",
        "license": "test fixture",
    }
    if pinned:
        spec["sha256"] = hashlib.sha256(payload).hexdigest()
    else:
        spec["record_first_download_sha256"] = True
    return spec


def test_manifest_covers_every_released_dimension_and_semantic_asset():
    module = _asset_module()
    manifest = module._load_manifest()

    assert list(manifest["dimensions"]) == [
        "Vis",
        "Aud",
        "AV",
        "Lip",
        "Text",
        "Face",
        "Music",
        "Speech",
        "Lo-Phy",
        "Hi-Phy",
        "Holistic",
    ]
    assert set(manifest["assets"]) == {
        "onealign_model_dir",
        "audiobox_checkpoint",
        "synchformer_checkpoint",
        "synchformer_config",
        "syncnet_checkpoint",
        "s3fd_checkpoint",
        "buffalo_dir",
        "videophy_model_dir",
        "ocr_det_dir",
        "ocr_rec_dir",
        "basic_pitch_model",
        "whisper_model_dir",
    }
    assert manifest["assets"]["onealign_model_dir"]["revision"] == (
        "dcc603b95aa0ebd82afa696d4a1e20d11fc80ddb"
    )
    assert manifest["assets"]["videophy_model_dir"]["revision"] == (
        "ca38370e1dcc79c99c7017dd8755d0a087299afc"
    )
    assert manifest["assets"]["whisper_model_dir"]["revision"] == (
        "edaa852ec7e145841d8ffdb056a99866b5f0a478"
    )
    assert manifest["source_bundled_assets"]["synchformer_s3d"]["sha256"] == (
        "d34e7463855a0541bd2ba6e73c6c62be8ad5a11a8e8c5480b4e36fc875bbe770"
    )

    for spec in manifest["assets"].values():
        if "url" in spec:
            assert spec["url"].startswith("https://")
        if spec["kind"] == "hf_snapshot":
            assert len(spec["revision"]) == 40
            for details in spec["files"].values():
                assert ("sha256" in details) ^ ("git_blob_sha1" in details)
        elif spec["kind"] in {"direct_file", "zip_directory"}:
            assert "sha256" in spec or spec["record_first_download_sha256"] is True


def test_test_mode_is_zero_io_and_returns_empty_maps(monkeypatch, tmp_path):
    module = _asset_module()
    monkeypatch.setattr(
        module, "_load_manifest", lambda: pytest.fail("test mode read the manifest")
    )

    result = module.ensure_assets(
        "Vis",
        tmp_path / "must not exist",
        overrides={"onealign_model_dir": tmp_path / "also missing"},
        test_mode=True,
    )

    assert result == {"paths": {}, "asset_identities": {}}
    assert list(tmp_path.iterdir()) == []


def test_unknown_dimension_is_rejected_without_io(tmp_path):
    module = _asset_module()
    with pytest.raises(module.AssetProvisionError, match="unknown AVGen-Bench"):
        module.ensure_assets("visual", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_required_asset_keys_exposes_exact_per_dimension_contract():
    module = _asset_module()

    assert module.required_asset_keys("AV") == (
        "synchformer_checkpoint",
        "synchformer_config",
    )
    assert module.required_asset_keys("Hi-Phy") == ()
    with pytest.raises(module.AssetProvisionError, match="unknown AVGen-Bench"):
        module.required_asset_keys("av_sync")


def test_no_model_dimensions_need_no_local_assets(tmp_path):
    module = _asset_module()
    assert module.ensure_assets("Hi-Phy", tmp_path) == {
        "paths": {},
        "asset_identities": {},
    }
    assert module.ensure_assets("Holistic", tmp_path) == {
        "paths": {},
        "asset_identities": {},
    }
    assert list(tmp_path.iterdir()) == []


def test_direct_asset_is_atomic_pinned_and_reused(monkeypatch, tmp_path):
    module = _asset_module()
    payload = b"publisher bytes"
    spec = _direct_spec(payload)
    monkeypatch.setattr(
        module, "_load_manifest", lambda: _manifest_for("audiobox_checkpoint", spec)
    )
    calls = []

    def download(url, destination, expected_bytes):
        calls.append((url, expected_bytes))
        destination.write_bytes(payload)
        return {"etag": '"fixture-etag"', "final_url": url}

    monkeypatch.setattr(module, "_download_url", download)
    first = module.ensure_assets("Aud", tmp_path)
    second = module.ensure_assets("Aud", tmp_path)

    path = first["paths"]["audiobox_checkpoint"]
    assert path == second["paths"]["audiobox_checkpoint"]
    assert path.read_bytes() == payload
    assert path.is_relative_to(tmp_path / ".ayase")
    identity = first["asset_identities"]["audiobox_checkpoint"]
    assert identity["provenance"] == "managed"
    assert identity["digest_authority"] == "manifest"
    assert identity["observed_sha256"] == hashlib.sha256(payload).hexdigest()
    assert calls == [(spec["url"], len(payload))]


def test_direct_asset_without_publisher_hash_records_first_download_sha(monkeypatch, tmp_path):
    module = _asset_module()
    payload = b"official versioned bytes without a published sha"
    spec = _direct_spec(payload, pinned=False)
    monkeypatch.setattr(
        module, "_load_manifest", lambda: _manifest_for("audiobox_checkpoint", spec)
    )

    def download(url, destination, expected_bytes):
        destination.write_bytes(payload)
        return {"etag": None, "final_url": url}

    monkeypatch.setattr(module, "_download_url", download)
    result = module.ensure_assets("Aud", tmp_path)
    identity = result["asset_identities"]["audiobox_checkpoint"]

    assert identity["digest_authority"] == "first_download"
    assert identity["observed_sha256"] == hashlib.sha256(payload).hexdigest()


def test_failed_direct_download_publishes_no_partial_cache(monkeypatch, tmp_path):
    module = _asset_module()
    payload = b"expected"
    spec = _direct_spec(payload)
    monkeypatch.setattr(
        module, "_load_manifest", lambda: _manifest_for("audiobox_checkpoint", spec)
    )

    def corrupt_download(url, destination, expected_bytes):
        destination.write_bytes(b"corrupt")
        return {"etag": None, "final_url": url}

    monkeypatch.setattr(module, "_download_url", corrupt_download)
    with pytest.raises(module.AssetProvisionError, match="SHA-256"):
        module.ensure_assets("Aud", tmp_path)

    final = tmp_path / ".ayase" / "avgen-bench" / "assets" / "direct" / "test-direct"
    assert not final.exists()
    assert not list(final.parent.glob(".test-direct.*"))


def test_existing_bad_cache_is_an_error_not_silently_repaired(monkeypatch, tmp_path):
    module = _asset_module()
    payload = b"expected"
    spec = _direct_spec(payload)
    monkeypatch.setattr(
        module, "_load_manifest", lambda: _manifest_for("audiobox_checkpoint", spec)
    )
    final = tmp_path / ".ayase" / "avgen-bench" / "assets" / "direct" / "test-direct"
    final.mkdir(parents=True)
    (final / "checkpoint.bin").write_bytes(b"corrupt")
    monkeypatch.setattr(module, "_download_url", lambda *args: pytest.fail("cache was repaired"))

    with pytest.raises(module.AssetProvisionError, match="marker"):
        module.ensure_assets("Aud", tmp_path)


def test_hf_snapshot_verifies_lfs_and_git_blob_bytes(monkeypatch, tmp_path):
    module = _asset_module()
    config = b'{"model":"fixture"}\n'
    weights = b"small lfs fixture"
    spec = {
        "kind": "hf_snapshot",
        "cache_name": "test-hf",
        "repo_id": "publisher/model",
        "revision": "1" * 40,
        "asset_id": "publisher/model@" + "1" * 40,
        "license": "test fixture",
        "files": {
            "config.json": {"bytes": len(config), "git_blob_sha1": _blob_sha1(config)},
            "model.bin": {"bytes": len(weights), "sha256": hashlib.sha256(weights).hexdigest()},
        },
    }
    monkeypatch.setattr(module, "_load_manifest", lambda: _manifest_for("onealign_model_dir", spec))

    def snapshot(repo_id, models_dir, *, revision, allow_patterns):
        target = Path(models_dir) / repo_id.replace("/", "--")
        target.mkdir(parents=True)
        (target / "config.json").write_bytes(config)
        (target / "model.bin").write_bytes(weights)
        assert revision == spec["revision"]
        assert set(allow_patterns) == set(spec["files"])
        return target

    import ayase.config

    monkeypatch.setattr(ayase.config, "download_hf_snapshot", snapshot)
    result = module.ensure_assets("Vis", tmp_path)

    model_dir = result["paths"]["onealign_model_dir"]
    assert model_dir.is_dir()
    assert (model_dir / "config.json").read_bytes() == config
    assert result["asset_identities"]["onealign_model_dir"]["revision"] == spec["revision"]


def test_zip_traversal_is_rejected_without_publishing(monkeypatch, tmp_path):
    module = _asset_module()
    archive = tmp_path / "malicious.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("model.onnx", b"model")
        bundle.writestr("../escape", b"escape")
    payload = archive.read_bytes()
    spec = {
        "kind": "zip_directory",
        "cache_name": "test-zip",
        "url": "https://publisher.example/releases/v1/model.zip",
        "bytes": len(payload),
        "record_first_download_sha256": True,
        "asset_id": "publisher/archive@v1",
        "license": "test fixture",
        "output_subdir": "models/buffalo_l",
        "members": {
            "model.onnx": {
                "bytes": 5,
                "crc32": f"{zlib.crc32(b'model') & 0xFFFFFFFF:08x}",
            }
        },
    }
    monkeypatch.setattr(module, "_load_manifest", lambda: _manifest_for("buffalo_dir", spec))

    def download(url, destination, expected_bytes):
        shutil.copyfile(archive, destination)
        return {"etag": None, "final_url": url}

    monkeypatch.setattr(module, "_download_url", download)
    with pytest.raises(module.AssetProvisionError, match="unsafe asset path"):
        module.ensure_assets("Face", tmp_path / "models")

    final = tmp_path / "models" / ".ayase" / "avgen-bench" / "assets" / "archives" / "test-zip"
    assert not final.exists()
    assert not (tmp_path / "escape").exists()


def test_valid_zip_is_safely_materialized_as_an_insightface_root(monkeypatch, tmp_path):
    module = _asset_module()
    model = b"onnx fixture"
    archive = tmp_path / "model.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("model.onnx", model)
    payload = archive.read_bytes()
    spec = {
        "kind": "zip_directory",
        "cache_name": "test-zip",
        "url": "https://publisher.example/releases/v1/model.zip",
        "bytes": len(payload),
        "record_first_download_sha256": True,
        "asset_id": "publisher/archive@v1",
        "license": "test fixture",
        "output_subdir": "models/buffalo_l",
        "members": {
            "model.onnx": {
                "bytes": len(model),
                "crc32": f"{zlib.crc32(model) & 0xFFFFFFFF:08x}",
            }
        },
    }
    monkeypatch.setattr(module, "_load_manifest", lambda: _manifest_for("buffalo_dir", spec))
    calls = []

    def download(url, destination, expected_bytes):
        calls.append(url)
        shutil.copyfile(archive, destination)
        return {"etag": None}

    monkeypatch.setattr(module, "_download_url", download)
    first = module.ensure_assets("Face", tmp_path / "models")
    second = module.ensure_assets("Face", tmp_path / "models")

    root = first["paths"]["buffalo_dir"]
    assert root == second["paths"]["buffalo_dir"]
    assert (root / "models" / "buffalo_l" / "model.onnx").read_bytes() == model
    identity = first["asset_identities"]["buffalo_dir"]
    assert identity["digest_authority"] == "first_download"
    assert identity["members"]["model.onnx"]["sha256"] == hashlib.sha256(model).hexdigest()
    assert calls == [spec["url"]]
    (root / "models" / "buffalo_l" / "000-unverified.onnx").write_bytes(b"override")
    with pytest.raises(module.AssetProvisionError, match="undeclared file"):
        module.ensure_assets("Face", tmp_path / "models")


def test_expert_override_is_hashed_and_not_claimed_as_published_parity(monkeypatch, tmp_path):
    module = _asset_module()
    payload = b"expert-selected bytes"
    override = tmp_path / "override.pt"
    override.write_bytes(payload)
    spec = _direct_spec(b"published bytes")
    monkeypatch.setattr(
        module, "_load_manifest", lambda: _manifest_for("audiobox_checkpoint", spec)
    )
    monkeypatch.setattr(module, "_download_url", lambda *args: pytest.fail("override downloaded"))

    result = module.ensure_assets(
        "Aud", tmp_path / "models", overrides={"audiobox_checkpoint": override}
    )
    identity = result["asset_identities"]["audiobox_checkpoint"]

    assert result["paths"]["audiobox_checkpoint"] == override.resolve()
    assert identity["provenance"] == "expert_override"
    assert identity["published_asset_parity"] is False
    assert identity["actual"]["sha256"] == hashlib.sha256(payload).hexdigest()


def test_override_for_another_dimension_is_rejected(tmp_path):
    module = _asset_module()
    with pytest.raises(module.AssetProvisionError, match="do not belong"):
        module.ensure_assets(
            "Vis",
            tmp_path,
            overrides={"whisper_model_dir": tmp_path},
        )


@pytest.mark.parametrize(
    "extra",
    [
        "model.safetensors",
        "model.safetensors.index.json",
        "generation_config.json",
        "tokenizer_config.json",
        "modeling_extra.py",
        "new_module/__init__.py",
    ],
)
def test_managed_hf_cache_rejects_unverified_loader_overrides(tmp_path, extra):
    module = _asset_module()
    root = tmp_path / "model"
    root.mkdir()
    payload = b"verified weights"
    spec = {
        "kind": "hf_snapshot",
        "repo_id": "publisher/model",
        "revision": "1" * 40,
        "asset_id": "fixture",
        "license": "fixture",
        "files": {
            "model.bin": {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        },
    }
    (root / "model.bin").write_bytes(payload)
    module._write_marker(root, module._base_marker("onealign_model_dir", spec))
    module._verify_hf_root(root, "onealign_model_dir", spec)
    additional = root / extra
    additional.parent.mkdir(parents=True, exist_ok=True)
    additional.write_bytes(b"unverified loader input")
    with pytest.raises(module.AssetProvisionError, match="undeclared"):
        module._verify_hf_root(root, "onealign_model_dir", spec)


def test_hf_downloader_bookkeeping_is_allowed_but_not_executable_files(tmp_path):
    module = _asset_module()
    root = tmp_path / "model"
    root.mkdir()
    payload = b"verified config"
    spec = {
        "kind": "hf_snapshot",
        "repo_id": "publisher/model",
        "revision": "1" * 40,
        "asset_id": "fixture",
        "license": "fixture",
        "files": {"config.json": {"bytes": len(payload), "git_blob_sha1": _blob_sha1(payload)}},
    }
    (root / "config.json").write_bytes(payload)
    module._write_marker(root, module._base_marker("onealign_model_dir", spec))
    bookkeeping = root / ".cache/huggingface/download"
    bookkeeping.mkdir(parents=True)
    (bookkeeping / "config.json.metadata").write_text("download metadata")
    (bookkeeping / "config.json.lock").touch()
    (bookkeeping.parent / ".gitignore").write_text("*")
    module._verify_hf_root(root, "onealign_model_dir", spec)
    (bookkeeping / "injected.py").write_text("raise RuntimeError()")
    with pytest.raises(module.AssetProvisionError, match="undeclared file"):
        module._verify_hf_root(root, "onealign_model_dir", spec)


@pytest.mark.parametrize("observed", [None, '"wrong-etag"', 'W/"pinned-etag"'])
def test_declared_etag_is_required_and_verified(monkeypatch, tmp_path, observed):
    module = _asset_module()
    payload = b"versioned checkpoint"
    spec = _direct_spec(payload, pinned=False)
    spec["expected_etag"] = "pinned-etag"
    monkeypatch.setattr(
        module, "_load_manifest", lambda: _manifest_for("audiobox_checkpoint", spec)
    )

    def download(url, destination, expected_bytes):
        destination.write_bytes(payload)
        return {"etag": observed}

    monkeypatch.setattr(module, "_download_url", download)
    if observed != 'W/"pinned-etag"':
        with pytest.raises(module.AssetProvisionError, match="ETag"):
            module.ensure_assets("Aud", tmp_path)
        assert not (tmp_path / ".ayase/avgen-bench/assets/direct/test-direct").exists()
    else:
        result = module.ensure_assets("Aud", tmp_path)
        marker_root = result["paths"]["audiobox_checkpoint"].parent
        marker = module._read_marker(marker_root)
        marker["observed_etag"] = None
        module._write_marker(marker_root, marker)
        with pytest.raises(module.AssetProvisionError, match="ETag marker"):
            module.ensure_assets("Aud", tmp_path)
