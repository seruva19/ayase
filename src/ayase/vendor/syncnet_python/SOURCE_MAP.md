# Source map

Source baseline: `joonson/syncnet_python` commit
`6efbb1c305c23f47a62b09cf4215a8ac45e97d49` (MIT).

| Local file | MIT upstream source |
| --- | --- |
| `model.py` | `SyncNetModel.py` |
| `scoring.py` | `SyncNetInstance.py` scoring and feature-window operations |
| `face_tracks.py` | face detection, tracking, interpolation, and crop operations from `run_pipeline.py` |
| `s3fd/` | `detectors/s3fd/` |

`ffmpeg.py` and `protocol.py` provide Ayase-owned checked process orchestration and
lifecycle integration. Defensive 224x224 normalization during scoring and the structured
score return are small independent adaptations around the MIT scoring operations.

The non-maximum-suppression function in `s3fd/box_utils.py` retains the notice from
`rbgirshick/py-faster-rcnn` commit
`781a917b378dbfdedb45b6a56189a31982da1b43`; its MIT license is reproduced in
`PY_FASTER_RCNN_LICENSE.md`.

The `wav2lip` string remains accepted solely as the public compatibility token selecting
this SyncNet implementation. No source from the similarly named external repository is
included in this directory.
