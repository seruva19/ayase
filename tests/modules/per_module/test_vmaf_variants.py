"""VMAF family provenance tests.

- ``vmaf``: the ``vmaf`` pip package exposes no ``compute_vmaf`` — the Python
  fallback branch is dead code and must not pretend to be a backend.
- ``vmaf_neg``: if the ``vmaf_v0.6.1neg`` model run fails there is no
  fallback to the plain model — a normal VMAF score must never be written
  under the NEG field.
- ``vmaf_phone``: ``phone_model`` was removed in libvmaf v2; the equivalent
  is ``enable_transform=true``.
"""

import json
import re
import subprocess
from pathlib import Path

import pytest

from ayase.models import QualityMetrics, Sample
from ayase.modules.vmaf import VMAFModule
from ayase.modules.vmaf_neg import VMAFNEGModule
from ayase.modules.vmaf_phone import VMAFPhoneModule


def _video_sample(tmp_path):
    return Sample(path=tmp_path / "v.mp4", is_video=True)


def _fake_ffmpeg_run(calls, returncode=0, metrics=None, fail_pred=None):
    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if fail_pred is not None and fail_pred(cmd):
            return subprocess.CompletedProcess(cmd, 1, "", "model not found")
        lavfi = next((a for a in cmd if "libvmaf" in a), None)
        if lavfi:
            m = re.search(r"log_path=(.*?)(?::log_fmt=|$)", lavfi)
            if m:
                Path(m.group(1)).write_text(json.dumps(metrics or {}))
        return subprocess.CompletedProcess(cmd, returncode, "", "")
    return fake_run


def test_vmaf_no_dead_python_backend(monkeypatch, tmp_path):
    """No libvmaf → module unavailable; a nonexistent ``vmaf.compute_vmaf``
    python path must not be claimed as a fallback."""
    monkeypatch.setattr(
        "ayase.modules.vmaf.subprocess.run",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()),
    )
    m = VMAFModule()
    m.setup()
    assert m._backend == "unavailable"
    assert not hasattr(m, "_compute_vmaf_python")


def test_vmaf_neg_no_fallback_command(monkeypatch, tmp_path):
    """A failed NEG-model run must not retry with the plain VMAF model."""
    calls = []
    monkeypatch.setattr(
        "ayase.modules.vmaf_neg.subprocess.run",
        _fake_ffmpeg_run(
            calls,
            metrics={"pooled_metrics": {"vmaf": {"mean": 80.0}}},
            fail_pred=lambda cmd: any("vmaf_v0.6.1neg" in a for a in cmd),
        ),
    )
    m = VMAFNEGModule()
    m._ml_available = True
    m._ffmpeg_available = True
    sample = _video_sample(tmp_path)
    sample.reference_path = tmp_path / "ref.mp4"
    sample.reference_path.touch()

    assert m.compute_reference_score(sample.path, sample.reference_path) is None
    assert len(calls) == 1
    assert any("vmaf_v0.6.1neg" in a for a in calls[0])


def test_vmaf_neg_no_arbitrary_metric_key(monkeypatch, tmp_path):
    """If the log has no vmaf score, the module must not grab an unrelated
    pooled metric and label it NEG."""
    monkeypatch.setattr(
        "ayase.modules.vmaf_neg.subprocess.run",
        _fake_ffmpeg_run(
            [], metrics={"pooled_metrics": {"psnr_y": {"mean": 33.0}}}
        ),
    )
    m = VMAFNEGModule()
    m._ml_available = True
    m._ffmpeg_available = True
    assert m.compute_reference_score(
        tmp_path / "v.mp4", tmp_path / "ref.mp4"
    ) is None


def test_vmaf_phone_uses_enable_transform(monkeypatch, tmp_path):
    """libvmaf v2 dropped ``phone_model`` — the phone transform is
    ``enable_transform=true`` on the standard model."""
    calls = []
    monkeypatch.setattr(
        "ayase.modules.vmaf_phone.subprocess.run",
        _fake_ffmpeg_run(
            calls, metrics={"pooled_metrics": {"vmaf": {"mean": 91.0}}}
        ),
    )
    m = VMAFPhoneModule()
    m._ml_available = True
    sample = _video_sample(tmp_path)
    sample.reference_path = tmp_path / "ref.mp4"
    sample.reference_path.touch()

    assert m.compute_reference_score(sample.path, sample.reference_path) == 91.0
    lavfi = next(a for a in calls[0] if "libvmaf" in a)
    assert "enable_transform" in lavfi
    assert "phone_model" not in lavfi
