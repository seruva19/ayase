# VideoPhy-2 AutoEval backend

Ayase's `videophy2_autoeval` module runs the released VideoPhy-2 AutoEval model. It is
separate from `vlm_phy`, which is an Ayase-defined generic VLM probe and does not reproduce
the published evaluator.

The bundled inference runtime is pinned to `Hritikbansal/videophy` revision
`01d366c979e4a82875852adbf427ccc036e7373a` (MIT repository license; the bundled mPLUG-Owl
modeling files carry Apache-2.0 headers). The independent AVGen-Bench
integration was checked at `microsoft/AVGen-Bench` revision
`1049eabac472d479fe5feeb1ee202961f8e0982a`. Both use the official
`videophysics/videophy_2_auto` mPLUG-Owl-Video checkpoint, the templates in
`VIDEOPHY2/template.py`, 32 uniformly segmented frames loaded by Decord, and deterministic
generation (`do_sample=False`, `top_k=1`, `temperature=0.001`, `max_length=256`).

The backend reports the native model outputs:

- `videophy2_sa_rating`: semantic adherence, integer-valued 1–5;
- `videophy2_pc_rating`: physical commonsense, integer-valued 1–5;
- `videophy2_joint_pass`: 1 exactly when both SA and PC are at least 4, otherwise 0.

The joint field is missing unless both ratings exist. SA is missing when the sample has no
caption. Parse failures and backend failures stay missing; they are not converted to a zero.
Dataset-level benchmark joint performance is the mean of `videophy2_joint_pass` over samples
with both ratings.

By default, Ayase imports the pinned runtime from
`ayase.third_party.videophy2.mplug_owl_video` and downloads the official checkpoint into
`models_dir` through `download_hf_snapshot`. The model snapshot is pinned to Hugging Face
revision `ca38370e1dcc79c99c7017dd8755d0a087299afc`. Set `auto_download=false` to require a
pre-populated local checkpoint and avoid the multi-gigabyte download.

The tokenizer dependency is included in `pip install ayase`. Ayase provisions a
private runtime with the upstream Transformers 4.28.1 stack automatically, so
the host application's Transformers version does not affect this evaluator.
Select the module with `ayase scan ./generated --modules videophy2_autoeval`;
same-directory text sidecars supply the caption for semantic adherence. Python
applications can use `AyasePipeline(modules=["videophy2_autoeval"])`.

Expert override only: for an independently checked-out copy of the upstream runtime,
clone the pinned repository:

```text
git clone https://github.com/Hritikbansal/videophy.git
git -C videophy checkout 01d366c979e4a82875852adbf427ccc036e7373a
```

Set `official_source_dir` to the checkout (or its `VIDEOPHY2` directory) to override the bundled
runtime. Ayase verifies the checkout revision before loading it. Runtime dependencies are
PyTorch, Decord, SentencePiece, Einops, and the packages imported by the official
mPLUG-Owl-Video source. Ayase uses the upstream `transformers==4.28.1` runtime;
full checkpoint inference still requires GPU validation. A
CUDA device with bfloat16 support is required by the released inference path.

This score predicts the released judge's ratings for action-centric generated videos. It is
not a direct measurement or proof that a video obeys physical laws. Benchmark-comparable
claims additionally require the official VideoPhy-2 manifest and dataset aggregation.
