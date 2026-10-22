# Official T2V-CompBench backend

`t2v_compbench_official` exposes all seven official T2V-CompBench evaluators
from a normal Ayase installation. No repository checkout, manual package install,
environment variable, or system Python mutation is required.

On first use Ayase checks for a CUDA GPU, then downloads the pinned upstream
snapshot at commit `dd5eff7b93af0550b9efa2bdabbb21b3b017ceda` into
`models/.ayase/sources`. The archive is extracted with path and symlink checks,
and evaluator scripts and metadata are verified against their Git blob IDs.
The upstream project has no blanket root license, so its source is cached at
runtime and is not redistributed in the Ayase wheel.

Ayase creates two private, fingerprinted environments below
`models/.ayase/environments`: one for the four LLaVA evaluators and one for the
GroundingDINO, Segment Anything, Depth Anything, and DOT evaluators. This keeps
their incompatible Torch, Transformers, and timm versions out of Ayase's
process. Both environments resolve their pinned CUDA 12.1 PyTorch wheels from
PyTorch's package index, including on Windows. GroundingDINO normally attempts a local C++/CUDA build. The managed
source selects GroundingDINO's own PyTorch implementation of multi-scale
deformable attention when its optional extension is unavailable, removing the
nvcc and compiler prerequisite while retaining the upstream operator semantics.

Checkpoints are stored under the managed source tree. GroundingDINO and SAM are
acquired for spatial and motion evaluation, while DOT checkpoints are acquired
only for motion binding. LLaVA (`liuhaotian/llava-v1.6-34b` by default) and
Depth Anything use their upstream Hugging Face loaders with `HF_HOME` below
`models`. These downloads are large, so provisioning starts only when the
module is mounted for a real run and a CUDA GPU is visible. `test_mode` performs
no setup.

The categories are `consistent_attribute_binding`, `dynamic_attribute_binding`,
`spatial_relationships`, `motion_binding`, `action_binding`,
`object_interactions`, and `generative_numeracy`. Select one with `category`, or
several with `categories`. With several categories, place a video below a
directory named for its official category or an alias: `consistent_attr`,
`dynamic_attr`, `spatial`, `motion`, `action`, `interaction`, or `numeracy`.

Official benchmark IDs are preserved: filenames begin with `0001` through
`0200`, and optional captions are checked against pinned metadata. Ayase stages
each run in a fresh temporary directory, uses `shell=False`, checks output IDs
and prompts, and publishes no score for incomplete output. Subsets remain
subset scores.

Most users need only this profile:

```json
{
  "name": "compbench",
  "modules": ["t2v_compbench_official"]
}
```

Advanced deployments may set `models_dir`, `temp_root`, `llava_model_id`,
`upstream_root`, or `python_executables`. The last two override the corresponding
managed source or interpreter choice. `manage_runtime`, `manage_assets`, and
`require_gpu` are expert controls and default to enabled.
