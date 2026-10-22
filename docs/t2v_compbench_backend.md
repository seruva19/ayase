# Official T2V-CompBench backend

`t2v_compbench_official` runs the evaluator scripts from a user-prepared
T2V-CompBench checkout. The integration is pinned and tested against commit
`dd5eff7b93af0550b9efa2bdabbb21b3b017ceda`. The repository does not publish a
blanket root license, so Ayase neither vendors nor redistributes its source.

Configure `upstream_root` to the prepared checkout. `metadata_dir` defaults to
its `meta_data` directory. Because upstream uses separate environments, the
`python_executables` mapping accepts a Python executable for each category plus
an optional `default`. The checkout's README remains authoritative for model
weights and environment preparation.

By default Ayase verifies Git blob identities for every evaluator, motion CLI
options module, and metadata JSON used by the adapter against the pinned commit.
It also reads each actual `argparse` declaration with Python's AST and rejects a
command containing an option the pinned parser does not declare.

All seven official categories are supported:

| Ayase category | Official evaluator |
|---|---|
| `consistent_attribute_binding` | LLaVA consistent-attribute script |
| `dynamic_attribute_binding` | LLaVA dynamic-attribute script |
| `action_binding` | LLaVA action-binding script |
| `object_interactions` | LLaVA interaction script |
| `spatial_relationships` | GroundingDINO/SAM/Depth Anything script |
| `generative_numeracy` | GroundingDINO numeracy script |
| `motion_binding` | GroundingSAM segmentation, then DOT tracking |

Use one module instance with `category`, or select several through `categories`.
With multiple categories, a video's ancestor directory must use an official
category name or a documented short name such as `consistent_attr`,
`dynamic_attr`, `spatial`, `motion`, `action`, `interaction`, or `numeracy`.

The official scripts derive the benchmark item from the first four filename
digits and index their JSON metadata by that value. Ayase therefore accepts only
IDs `0001` through `0200`, checks an available sample caption against the
authoritative metadata prompt, and stages canonical filenames into a new
temporary directory for every run. This prevents stale append-mode CSV rows and
prevents directory order from silently pairing a video with the wrong prompt.
`temp_root` is optional and otherwise uses the operating system's temporary
directory; deployments can set it to their managed temporary-artifact root.

Commands are argument arrays executed with `shell=False`, a configurable
timeout, and the upstream working directory. Evaluator output streams to a
temporary `evaluator-N.log`; failures report at most its final 2048 bytes before
the fresh run directory is removed. Native CSV output is parsed by ID
and prompt. Consistent attribute scores use upstream `(raw - 1) / 14`; action
binding and object interaction use `(raw - 1) / 9`; the other four CSVs already
contain their official per-video scale. The trailer score is stored as the
category dataset average. Missing environments, scripts, metadata, trailers, or
valid rows leave that category unavailable rather than producing a proxy score.

The runner creates no model downloads itself. A real evaluation is expensive
and requires the models, checkpoints, CUDA setup, and sometimes distinct Python
environments described by upstream.

Example profile for a prepared numeracy environment:

```json
{
  "name": "compbench_numeracy",
  "modules": ["t2v_compbench_official"],
  "module_config": {
    "t2v_compbench_official": {
      "upstream_root": "/path/to/T2V-CompBench",
      "category": "generative_numeracy",
      "python_executables": {"default": "/path/to/detection-env/bin/python"}
    }
  }
}
```

Load this file with `AyasePipeline(profile="compbench-profile.json")` and run
the directory containing benchmark videos such as `0007.mp4` and `0199.mp4`.
Keep the official IDs; a subset remains a subset score, not the full benchmark.
