"""SIM-o: speaker similarity of a recording to a reference recording (WavLM-TDNN).

Published metric: speaker similarity measured with the WavLM-Large + ECAPA-TDNN speaker-verification
model of UniSpeech, as used by VALL-E (arXiv:2301.02111), Voicebox (arXiv:2306.15687, which names
the variant against the original prompt audio SIM-o), NaturalSpeech 3, Seed-TTS and F5-TTS. This
module ports the official evaluation code of F5-TTS (``src/f5_tts/eval/utils_eval.py:run_sim``),
which is the same computation as seed-tts-eval ``cal_sim.sh`` -> UniSpeech
``verification_pair_list_v2.py``:

* model ``ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large")``, weights
  ``wavlm_large_finetune.pth`` loaded from ``state_dict["model"]``;
* both recordings loaded at their native rate with torchaudio and resampled to 16 kHz with
  ``torchaudio.transforms.Resample`` when the rate differs;
* whole recordings, no trimming or voice-activity selection;
* ``F.cosine_similarity(emb1, emb2)[0]`` - for multi-channel input the first channel, as in the
  official code.

The field is SIM-o only when ``reference_path`` is the original (not codec-resynthesised) reference
audio; SIM-r needs the evaluated model's own codec and is not defined here. For a video input the
audio track is extracted at its native rate before the same steps.

Weights come from Hugging Face and are checked by sha256:

* ``wavlm_large_finetune.pth`` (1 301 926 579 bytes, sha256 51f07e3b...) - the checkpoint of the
  seed-tts-eval / F5-TTS instructions, from ``bezzam/wavlm_large_finetune_seed_tts_eval`` (a verbatim
  copy of the Google Drive file those instructions link; CC BY-SA 3.0, the UniSpeech license);
* ``wavlm_large.pt`` (sha256 6fb4b3c3...) from ``s3prl/converted_ckpts`` - the file s3prl's own
  ``wavlm_large`` entry downloads.

The WavLM upstream is s3prl's own code, vendored in ``ayase.vendor.s3prl_wavlm`` from commit
ec8064b5 (main, 2025-06-13): the official scripts get it through ``torch.hub.load("s3prl/s3prl",
"wavlm_large")``, which downloads the whole s3prl repository at run time and no longer imports with
torchaudio 2.x (``set_audio_backend`` and ``sox_effects`` were removed). The commit named in the
UniSpeech README (7ab62aaf) cannot be used either: its ``wavlm_large`` points to an Azure link that
expired in 2022. Model code: ``ayase.vendor.unispeech_sv`` (CC BY-SA 3.0) and
``ayase.vendor.s3prl_wavlm`` (MIT / Apache-2.0); see NOTICE.md in each.

Validation: F5-TTS reports SIM-o 0.69 for ground truth on LibriSpeech-PC test-clean cross-sentence
(``data/librispeech_pc_test_clean_cross_sentence.lst``, 1127 pairs, 4-10 s; each ground-truth
utterance against its prompt utterance). Reproduced 2026-09-24 on GPU with LibriSpeech test-clean
(Hugging Face ``openslr/librispeech_asr``): 1127 of 1127 pairs, mean 0.6948.
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

_CKPT_REPO = "bezzam/wavlm_large_finetune_seed_tts_eval"
_CKPT_FILE = "wavlm_large_finetune.pth"
_CKPT_SHA256 = "51f07e3b94d9e0262a6a675ef5a087be3dd09e8c62e9d886827f44f82fe7f94b"
_WAVLM_REPO = "s3prl/converted_ckpts"
_WAVLM_FILE = "wavlm_large.pt"
_WAVLM_SHA256 = "6fb4b3c3e6aa567f0a997b30855859cb81528ee8078802af439f7b2da0bf100f"
_AUDIO_SUFFIXES = {".wav", ".flac"}  # read directly; anything else goes through ffmpeg to WAV at its native rate


class SpeakerSimModule(PipelineModule):
    name = "speaker_sim"
    description = "SIM-o speaker similarity to a reference recording (WavLM-TDNN, UniSpeech / F5-TTS eval)"
    default_config = {
        "device": "auto",
        "models_dir": "models",
    }
    models = [
        {"id": "bezzam/wavlm_large_finetune_seed_tts_eval", "type": "huggingface",
         "url": "https://huggingface.co/bezzam/wavlm_large_finetune_seed_tts_eval/resolve/main/wavlm_large_finetune.pth",
         "task": "WavLM-TDNN speaker verification checkpoint (SIM-o)", "size": "1.3 GB", "auto_download": True,
         "notes": "CC BY-SA 3.0 (UniSpeech); copy of the checkpoint linked by seed-tts-eval / F5-TTS"},
        {"id": "s3prl/converted_ckpts", "type": "huggingface",
         "url": "https://huggingface.co/s3prl/converted_ckpts/resolve/main/wavlm_large.pt",
         "task": "WavLM-Large upstream for the WavLM-TDNN model", "size": "1.26 GB", "auto_download": True,
         "notes": "MIT (WavLM, Microsoft); the file s3prl's wavlm_large entry downloads"},
    ]
    metric_info = {
        "sim_o": "SIM-o: WavLM-TDNN cosine between the recording and the original reference recording (-1..1, higher=better)",
    }
    metric_groups = {"sim_o": "audio"}

    def __init__(self, config=None):
        super().__init__(config)
        self._backend = None
        self._model = None
        self._device = None

    def setup(self) -> None:
        if self.test_mode:
            return
        try:
            import hashlib

            import torch
            from huggingface_hub import hf_hub_download

            from ayase.vendor.unispeech_sv.ecapa_tdnn import ECAPA_TDNN_SMALL
        except Exception as e:
            self._backend = "unavailable"
            logger.warning("speaker_sim: dependencies missing (%s); SIM-o left unset", e)
            return
        device = self.config.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        def fetch(repo: str, name: str, digest: str) -> Path:
            path = Path(hf_hub_download(repo_id=repo, filename=name,
                                        cache_dir=str(Path(self.config.get("models_dir", "models")) / "speaker_sim")))
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 24), b""):
                    h.update(chunk)
            if h.hexdigest() != digest:
                raise RuntimeError(f"{name} sha256 {h.hexdigest()} != {digest}")
            return path

        try:
            ckpt = fetch(_CKPT_REPO, _CKPT_FILE, _CKPT_SHA256)
            wavlm = fetch(_WAVLM_REPO, _WAVLM_FILE, _WAVLM_SHA256)
            # ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large") in the official code; the
            # same s3prl WavLM upstream is built from the vendored code and the downloaded file.
            model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None,
                                     hub_repo="ayase-vendored", hub_kwargs={"ckpt": str(wavlm)})
            state_dict = torch.load(str(ckpt), weights_only=True, map_location=lambda storage, loc: storage)
            result = model.load_state_dict(state_dict["model"], strict=False)
            # The official code loads with strict=False; a silent key mismatch would leave the
            # fine-tuned weights unloaded, so any mismatch is an error here - except the training-loss
            # head ("loss_calculator.*"), which the checkpoint carries and inference never uses.
            unexpected = [k for k in result.unexpected_keys if not k.startswith("loss_calculator.")]
            if result.missing_keys or unexpected:
                raise RuntimeError(f"checkpoint keys mismatch: missing {result.missing_keys[:5]}, "
                                   f"unexpected {unexpected[:5]}")
            self._model = model.to(device).eval()
        except Exception as e:
            self._backend = "unavailable"
            logger.warning("speaker_sim: model setup failed (%s); SIM-o left unset", e)
            return
        self._backend = "wavlm_tdnn"

    def _load(self, path: Path):
        """(waveform [channels, samples] at 16 kHz on device) as in the official code, or None."""
        import torchaudio

        src = path
        tmp = None
        if path.suffix.lower() not in _AUDIO_SUFFIXES:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            done = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-vn",
                                   "-acodec", "pcm_s16le", tmp.name], capture_output=True, text=True)
            if done.returncode != 0:
                Path(tmp.name).unlink(missing_ok=True)
                logger.warning("speaker_sim: no audio track in %s", path.name)
                return None
            src = Path(tmp.name)
        try:
            # torchaudio.load in the official code; torchaudio 2.9+ delegates decoding to torchcodec,
            # so the same float samples ([-1, 1], channels first) are read with soundfile here.
            import soundfile as sf
            import torch

            data, sr = sf.read(str(src), dtype="float32", always_2d=True)
            wav = torch.from_numpy(data.T.copy())
        finally:
            if tmp is not None:
                Path(tmp.name).unlink(missing_ok=True)
        wav = wav.to(self._device)
        if sr != 16000:
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000).to(self._device)(wav)
        return wav

    def process(self, sample: Sample) -> Sample:
        if self._backend != "wavlm_tdnn" or self._model is None:
            return sample
        ref = getattr(sample, "reference_path", None)
        if ref is None:
            return sample
        try:
            import torch
            import torch.nn.functional as F

            wav1 = self._load(Path(sample.path))
            wav2 = self._load(Path(ref))
            if wav1 is None or wav2 is None:
                return sample
            with torch.no_grad():
                emb1 = self._model(wav1)
                emb2 = self._model(wav2)
            sim: Optional[float] = float(F.cosine_similarity(emb1, emb2)[0].item())
        except Exception as e:
            logger.warning("speaker_sim failed on %s: %s", Path(sample.path).name, e)
            return sample
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.sim_o = sim
        return sample
