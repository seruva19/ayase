"""Tests for BrightRate module."""

from ayase.models import QualityMetrics

from ..conftest import _test_module_basics


def test_brightrate_basics():
    from ayase.modules.brightrate import BrightRateModule

    _test_module_basics(BrightRateModule, "brightrate")


def test_brightrate_skip_without_backend(video_sample):
    from ayase.modules.brightrate import BrightRateModule

    video_sample.quality_metrics = QualityMetrics()
    result = BrightRateModule().process(video_sample)
    assert result is video_sample
    assert result.quality_metrics.brightrate_score is None


def test_brightrate_field_exists():
    qm = QualityMetrics()
    assert hasattr(qm, "brightrate_score")
    assert qm.brightrate_score is None


def test_brightrate_field_group():
    qm = QualityMetrics()
    assert qm._FIELD_GROUPS.get("brightrate_score") == "hdr"


def test_explicit_asset_path_takes_precedence(tmp_path, monkeypatch):
    from ayase.modules import brightrate

    explicit = tmp_path / "configured.pt"
    explicit.write_bytes(b"test-checkpoint")
    module = brightrate.BrightRateModule({"contrique_path": str(explicit)})

    def unexpected_download(*args, **kwargs):
        raise AssertionError("An explicit asset must not trigger a download")

    monkeypatch.setattr(brightrate, "download_model_file", unexpected_download)
    assert (
        module._resolve_asset_path(
            "contrique", "brightvq/contrique.pt", "https://example.com/model.pt", str(tmp_path)
        )
        == explicit
    )


def test_default_asset_resolution_uses_managed_download(tmp_path, monkeypatch):
    from ayase.modules import brightrate

    resolved = tmp_path / "managed" / "contrique.pt"
    calls = []

    def download(relative_path, url, models_dir):
        calls.append((relative_path, url, models_dir))
        return resolved

    monkeypatch.setattr(brightrate, "download_model_file", download)
    module = brightrate.BrightRateModule()
    assert (
        module._resolve_asset_path(
            "contrique", "brightvq/contrique.pt", "https://example.com/model.pt", str(tmp_path)
        )
        == resolved
    )
    assert calls == [("brightvq/contrique.pt", "https://example.com/model.pt", str(tmp_path))]


def test_missing_explicit_asset_stays_unavailable(tmp_path, monkeypatch):
    from ayase.modules import brightrate

    module = brightrate.BrightRateModule({"contrique_path": str(tmp_path / "missing.pt")})

    def unexpected_download(*args, **kwargs):
        raise AssertionError("An invalid explicit override must not trigger a download")

    monkeypatch.setattr(brightrate, "download_model_file", unexpected_download)
    assert (
        module._resolve_asset_path(
            "contrique", "brightvq/contrique.pt", "https://example.com/model.pt", str(tmp_path)
        )
        is None
    )
