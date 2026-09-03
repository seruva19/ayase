# Vendored third-party code

Ayase runs several research implementations in-process. Their source is kept here
rather than downloaded at run time: a metric must not put unreviewed code from the
network on `sys.path` of an end user's machine, and an installed version has to be
reproducible.

Each tree is a subset of an upstream snapshot at a pinned commit, trimmed to what
inference reaches — datasets, notebooks, demo media, benchmark assets and training
paths are left out. Licence and notice files are kept.

| Tree | Upstream | Pinned commit | Licence |
|---|---|---|---|
| `cotracker` | facebookresearch/co-tracker | inference-only CoTracker2 and CoTracker3-offline | **CC BY-NC 4.0** |
| `imagebind` | facebookresearch/ImageBind | — | **CC BY-NC-SA 4.0** |
| `mj_video` | aiming-lab/MJ-Video | `cc1d2c95` | **no licence file in the snapshot** |
| `q_align` | Q-Future/Q-Align | — | see tree |
| `s2wrapper` | bfshi/scaling_on_scales | `9c008a37` | MIT |
| `sam`, `sam2` | facebookresearch/segment-anything | — | Apache-2.0 |
| `t2v_metrics` | linzhiqiu/t2v_metrics | — | see tree |
| `vbench` | Vchitect/VBench, `VBench-2.0` subtree | `45e79ec1` | Apache-2.0, but see below |
| `verse_bench` | — | — | see tree |
| `videomae` | OpenGVLab/VideoMAEv2 | — | see tree |
| `vila` | NVlabs/VILA, `llava` package | `0f1426e8` | Apache-2.0 |
| `vmbench` | GD-ML/VMBench | — | see tree |
| `vqa2` | Q-Future/Visual-Question-Answering-for-Video-Quality-Assessment | `9087c795` | Apache-2.0 |

## What the licences mean for redistribution

The VBench snapshot carries its own vendored dependencies, and they are not all
Apache-2.0:

| Inside `vbench/vbench2/third_party` | Licence |
|---|---|
| `Instance_detector`, `LLaVA_NeXT` | Apache-2.0 |
| `RAFT` | BSD 3-Clause |
| `YOLO-World/mmyolo`, `ViTDetector/third_party/YOLO-World/mmyolo` | **GPL-3.0** |
| `cotracker` | **CC BY-NC 4.0** |

Three consequences follow, and they are stated here because a package that declares
one licence while shipping another misleads whoever installs it:

- **GPL-3.0** is copyleft: a distributed work containing it has to reach recipients
  under GPL terms. It does not sit quietly inside an MIT distribution.
- **CC BY-NC** permits redistribution for non-commercial purposes only. Offering the
  package under MIT grants recipients a commercial-use right the upstream licence
  does not grant.
- **A snapshot without a licence file** carries no redistribution grant at all;
  default copyright applies.

Two of the four modules affected (`vbench2`, and the CoTracker2 users) can instead
take an operator-provided path to a local checkout, which keeps the published
distribution clean. That is a policy decision, not a technical one.
