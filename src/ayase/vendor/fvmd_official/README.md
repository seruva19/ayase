# Vendored official FVMD runtime

This directory contains the minimal inference path needed for the official
Fréchet Video Motion Distance (FVMD): PIPs++ point tracking, velocity and
acceleration fields, dense 1-D motion histograms, and Fréchet distance.

## Provenance

- Upstream: <https://github.com/DSL-Lab/FVMD-frechet-video-motion-distance>
- Snapshot: `875a86a92239e5f5751fda52c30d18a062fcfebc`
- License: Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
- Paper: Liu et al., *Fréchet Video Motion Distance: A Metric for Evaluating
  Motion Consistency in Videos*, arXiv:2407.16124.

## Included source

- `nets/pips2.py`: upstream PIPs++ architecture used by FVMD.
- `utils/`: only helpers reached by PIPs++ inference.
- `tracking.py`: checkpoint loading, point tracking, and motion-field math.
- `features.py`: official dense 1-D velocity/acceleration histogram.
- `frechet.py`: empirical Gaussian statistics and Fréchet calculation.

Dataset loaders, command-line entry points, training/reporting utilities,
TensorBoard, Fire, Loguru, PrettyTable, visualization, and cache writers are
not part of this runtime subset.

## Modifications from the snapshot

Imports were changed from the top-level `fvmd` package to
`ayase.vendor.fvmd_official` package-relative imports. Unused utilities were
reduced to the functions needed for inference, and checkpoint loading uses
PyTorch's library API directly. No CLI or logging side effect is performed.
The release checkpoint contains legacy pickle globals, so unsafe loading is
opt-in via ``trusted_checkpoint=True``; Ayase enables it only after verifying
the downloaded file against its pinned SHA-256.

The released `tracking_fullseq` contains two related acceleration mismatches
with the paper: it passes trajectories to `calc_acceleration`, and that helper
takes `input[:, 2:] - input[:, 1:-1]` while prepending two zero fields. Paper
Eq. (2) instead specifies:

```text
A = concat(0, V[1:F] - V[0:F-1])
```

For numerical compatibility with the official 1.0.0 evaluator, this vendored
runtime deliberately preserves the released code behavior:

```text
A_runtime = concat(0, 0, Y[2:F] - Y[1:F-1])
```

A paper-corrected acceleration feature must use a distinct metric identity;
it is intentionally not substituted into this official FVMD runtime.
