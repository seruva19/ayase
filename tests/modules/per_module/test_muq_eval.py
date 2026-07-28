from pathlib import Path

import numpy as np
import soundfile as sf

from ayase.models import Sample
from tests.modules.conftest import _test_module_basics


def _write_wav(path: Path, sr: int = 24000, seconds: float = 0.25) -> None:
    time = np.arange(int(sr * seconds), dtype=np.float32) / sr
    sf.write(path, 0.1 * np.sin(2 * np.pi * 440.0 * time), sr)


def test_muq_eval_basics():
    from ayase.modules.muq_eval import MuQEvalModule

    _test_module_basics(MuQEvalModule, "muq_eval")


def test_muq_eval_without_setup_is_graceful(tmp_path):
    from ayase.modules.muq_eval import MuQEvalModule

    path = tmp_path / "music.wav"
    _write_wav(path)
    sample = Sample(path=path, is_video=False)
    result = MuQEvalModule().process(sample)
    assert result is sample
    assert result.quality_metrics is None


def test_muq_eval_prepares_published_input_shape(tmp_path):
    from ayase.modules.muq_eval import MuQEvalModule

    path = tmp_path / "music.wav"
    _write_wav(path)
    module = MuQEvalModule()
    waveform = module._prepare_waveform(path)
    assert waveform is not None
    assert waveform.dtype == np.float32
    assert waveform.shape == (240000,)


def test_muq_eval_processes_reference_mi_only(tmp_path, monkeypatch):
    from ayase.modules.muq_eval import MuQEvalModule

    path = tmp_path / "music.wav"
    _write_wav(path)
    sample = Sample(path=path, is_video=False)
    module = MuQEvalModule()
    module._ml_available = True
    module._backend = "a1"
    monkeypatch.setattr(module, "_score", lambda waveform: 4.125)

    result = module.process(sample)

    assert result is sample
    assert result.quality_metrics is not None
    assert result.quality_metrics.muq_eval_mi_score == 4.125


def test_muq_eval_rejects_unpublished_input_configuration():
    from ayase.modules.muq_eval import MuQEvalModule

    module = MuQEvalModule({"sample_rate": 16000})
    module.setup()
    assert module._ml_available is False
    assert module._backend == "unavailable"
