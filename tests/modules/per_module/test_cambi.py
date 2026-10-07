"""CAMBI module tests.

The FFmpeg ``libvmaf`` filter is a two-input filter (distorted + reference)
even when only no-reference features like CAMBI are requested. CAMBI must
feed the video on both inputs or libvmaf rejects the filtergraph.
"""

import json
import re
import subprocess
from pathlib import Path

from ayase.models import QualityMetrics, Sample
from ayase.modules.cambi import CAMBIModule


def test_cambi_feeds_video_twice(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        lavfi = next(a for a in cmd if "libvmaf" in a)
        log_path = re.search(r"log_path=(.*?)(?::log_fmt=|$)", lavfi).group(1)
        Path(log_path).write_text(
            json.dumps({"pooled_metrics": {"cambi": {"mean": 5.0}}})
        )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("ayase.modules.cambi.subprocess.run", fake_run)
    m = CAMBIModule()
    m._backend = "ffmpeg_libvmaf"
    m._ml_available = True
    sample = Sample(path=tmp_path / "v.mp4", is_video=True)
    sample.quality_metrics = QualityMetrics()

    out = m.process(sample)
    assert out.quality_metrics.cambi == 5.0

    cmd = calls[0]
    assert cmd.count("-i") == 2, "libvmaf needs two inputs (video fed twice)"
    lavfi = next(a for a in cmd if "libvmaf" in a)
    assert "[0:v][1:v]" in lavfi
