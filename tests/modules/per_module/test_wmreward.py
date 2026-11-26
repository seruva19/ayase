def test_wmreward_sample(video_sample, monkeypatch):
    from ayase.modules.wmreward import WMRewardModule

    module = WMRewardModule({"test_mode": True})
    module.on_mount()
    assert module._encoder is None
    assert module.process(video_sample) is video_sample
    assert video_sample.quality_metrics is None

    module._encoder = object()
    monkeypatch.setattr(module, "_score", lambda path: None)
    assert module.process(video_sample).quality_metrics is None
    monkeypatch.setattr(module, "_score", lambda path: 0.42)
    assert module.process(video_sample).quality_metrics.latent_prediction_reward == 0.42
    module.on_dispose()
    assert module._encoder is None


def test_wmreward_checkpoint_verification_cache(tmp_path, monkeypatch):
    import hashlib
    import json

    import ayase.config as config_mod
    import ayase.modules.wmreward as wm_mod

    models_dir = tmp_path / "models"
    ckpt_dir = models_dir / "wmreward"
    ckpt_dir.mkdir(parents=True)
    ckpt_path = ckpt_dir / "vitg.pt"
    ckpt_path.write_bytes(b"valid-weights-payload")
    expected_sha = hashlib.sha256(b"valid-weights-payload").hexdigest()

    monkeypatch.setattr(wm_mod, "WEIGHTS_SHA256", expected_sha)
    monkeypatch.setattr(wm_mod.WMRewardModule, "test_mode", False)
    monkeypatch.setattr(
        config_mod,
        "download_model_file",
        lambda rel, url, mdir: ckpt_path,
    )
    monkeypatch.setattr(wm_mod, "_load_vitg", lambda path, dev: ("enc", "pred"))

    # 1) First setup computes SHA-256 and writes sidecar
    mod1 = wm_mod.WMRewardModule({"models_dir": str(models_dir), "device": "cpu"})
    mod1.setup()
    assert mod1._encoder == "enc"
    sidecar = ckpt_dir / "vitg.pt.verified.json"
    assert sidecar.is_file()
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    assert meta["sha256"] == expected_sha
    assert meta["size"] == ckpt_path.stat().st_size
    assert meta["mtime_ns"] == ckpt_path.stat().st_mtime_ns

    # 2) Second setup with unchanged (size, mtime_ns) skips re-hashing
    real_sha256 = hashlib.sha256

    def fail_sha256():
        raise AssertionError("sha256() should not be called when sidecar matches")

    monkeypatch.setattr(wm_mod.hashlib, "sha256", fail_sha256)
    mod2 = wm_mod.WMRewardModule({"models_dir": str(models_dir), "device": "cpu"})
    mod2.setup()
    assert mod2._encoder == "enc"

    # 3) Modified file triggers re-hash, removes stale sidecar, and rejects corrupt checkpoint
    monkeypatch.setattr(wm_mod.hashlib, "sha256", real_sha256)
    ckpt_path.write_bytes(b"corrupted-payload-changed-size")
    mod3 = wm_mod.WMRewardModule({"models_dir": str(models_dir), "device": "cpu"})
    mod3.setup()
    assert mod3._encoder is None
    assert not sidecar.exists()
