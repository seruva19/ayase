# UniSpeech speaker-verification ECAPA-TDNN (vendored)

Source: microsoft/UniSpeech `downstreams/speaker_verification/models/ecapa_tdnn.py`, via the copy in
SWivid/F5-TTS `src/f5_tts/eval/ecapa_tdnn.py` (branch main, 2026-09-24), the same file that
seed-tts-eval and F5-TTS use to compute SIM-o. License of the UniSpeech original: CC BY-SA 3.0;
this file keeps that license. Parts are borrowed by UniSpeech from lawlict/ECAPA-TDNN.

Local change: the s3prl torch.hub repository is a constructor argument (`hub_repo`) instead of the
hard-coded unpinned `"s3prl/s3prl"`, so Ayase can pin the commit named in the UniSpeech README
(`7ab62aaf2606d83da6c71ee74e7d16e0979edbc3`). The network architecture and weights are unchanged.
