# syncnet_python

This package is an adapted port of
[joonson/syncnet_python](https://github.com/joonson/syncnet_python) commit
`6efbb1c305c23f47a62b09cf4215a8ac45e97d49`, licensed under MIT. See
`LICENSE.md`, `NOTICE.md`, and `SOURCE_MAP.md`.

Weights are the original ones: `syncnet_v2.model` and `sfd_face.pth` from
`https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/`.

## Numerics kept from the originals

- ffmpeg conversion to 25 fps (`-qscale:v 2 -async 1 -r 25`), JPEG frames, mono 16 kHz PCM;
- S3FD at `conf_th=0.9` on a `facedet_scale`-downscaled frame, final NMS at IoU 0.1;
- PySceneDetect `ContentDetector` scene split, IoU-based tracking (0.5, `num_failed_det=25`),
  linear interpolation of missed boxes, `min_track` / `min_face_size` filters;
- crop with `crop_scale=0.40`, box smoothing by a 13-tap median filter, padding value 110,
  224x224 XVID crop re-muxed with the matching audio slice;
- scoring on BGR 0-255 frames without normalisation, `python_speech_features.mfcc` defaults,
  5-frame / 20-MFCC-frame windows, `min_length - 5` windows, `vshift=15`, `batch_size=20`;
- LSE-D = minimum over shifts of the mean distance, LSE-C = median minus minimum;
- TF32 disabled and cuDNN deterministic while scoring.

## Runtime behavior

- explicit device selection, `torch.no_grad()`, `torch.load(weights_only=True)`
  and `np.intp` integer indices;
- ffmpeg uses argument lists and checked return codes;
- intermediate files use a per-call temporary directory;
- PySceneDetect uses `open_video`; clips without a detected cut use one scene spanning
  the frames the decoder read;
- prior boxes of S3FD are cached per input size;
- crops are resized to 224x224 and scoring returns structured per-track values.

The model checkpoint files are external data. Their licensing is not established by
the MIT source license and must be assessed separately.
