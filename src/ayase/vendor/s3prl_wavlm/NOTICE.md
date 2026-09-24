# s3prl WavLM upstream (vendored)

Copied from s3prl/s3prl at commit ec8064b5889f81ca460fbe2c094ce576a6f120b7 (main, 2025-06-13):
`s3prl/upstream/wavlm/{expert.py, WavLM.py, modules.py}` and `s3prl/upstream/interfaces.py`.
License: s3prl is Apache-2.0 (LICENSE.s3prl); WavLM.py, modules.py and expert.py carry
"Copyright (c) Microsoft Corporation. Licensed under the MIT License."

Local changes, none of which touch the computation:
- `expert.py`: `from ..interfaces` -> `from .interfaces` (flat package);
- `interfaces.py`: `s3prl.utility.helper.show` replaced by a local `show` that prints (the original
  only adds a distributed-leader check).

Why vendored: the official SIM-o scripts load this upstream with `torch.hub.load("s3prl/s3prl",
"wavlm_large")`, which downloads the whole repository at run time and fails to import under
torchaudio 2.x (removed `set_audio_backend`, `sox_effects`). Used by `ayase.modules.speaker_sim`.
