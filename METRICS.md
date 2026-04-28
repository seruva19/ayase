# Ayase Metrics Reference

> **Version 0.1.30** · Generated 2026-04-28 15:05 · **327 modules** · **368 metrics**
>
> `ayase modules docs -o METRICS.md` to regenerate
>
> Tests: **327/327 modules** have static test references · `pytest tests/` (light) · `pytest tests/ --full` (with ML models)

> [!NOTE]
> Static test coverage links are included below. Live pass/fail status was not collected for this regeneration (`--no-tests` was passed). Re-run with `ayase modules docs --run-tests` to add live status.

## Summary

**327** modules · **393** output fields · **368** metrics · **86** tiered · **155** GPU · **21** categories

<table width="100%"><tr>
<td width="50%" valign="top"><h4>Modules by Category</h4><img src="docs/chart_categories.png" width="100%"/></td>
<td width="50%" valign="top"><h4>Input Types</h4><img src="docs/chart_input_types.png" width="100%"/></td>
</tr></table>

<table width="100%"><tr>
<td width="50%" valign="top"><h4>Speed Tiers</h4><img src="docs/chart_speed.png" width="100%"/></td>
<td width="50%" valign="top"><h4>Backend Usage</h4><img src="docs/chart_backends.png" width="100%"/></td>
</tr></table>

<table width="100%"><tr>
<td width="50%" valign="top"><h4>Top Packages</h4><img src="docs/chart_packages.png" width="100%"/></td>
<td width="50%" valign="top"><h4>Metrics per Category</h4><img src="docs/chart_metrics_per_cat.png" width="100%"/></td>
</tr></table>

<a id="categories"></a>

[No-Reference Quality](#no-reference-quality-98-metrics) (98) · [Full-Reference Quality](#full-reference-quality-58-metrics) (58) · [Text-Video Alignment](#text-video-alignment-31-metrics) (31) · [Temporal Consistency](#temporal-consistency-24-metrics) (24) · [Motion & Dynamics](#motion--dynamics-22-metrics) (22) · [Basic Visual Quality](#basic-visual-quality-15-metrics) (15) · [Aesthetics](#aesthetics-9-metrics) (9) · [Audio Quality](#audio-quality-20-metrics) (20) · [Face & Identity](#face--identity-19-metrics) (19) · [Scene & Content](#scene--content-16-metrics) (16) · [Distribution & Generation](#distribution--generation-1-metrics) (1) · [HDR & Color](#hdr--color-13-metrics) (13) · [Codec & Technical](#codec--technical-5-metrics) (5) · [Depth & Spatial](#depth--spatial-5-metrics) (5) · [Production Quality](#production-quality-5-metrics) (5) · [OCR & Text](#ocr--text-7-metrics) (7) · [Safety & Ethics](#safety--ethics-7-metrics) (7) · [Image-to-Video Reference](#image-to-video-reference-4-metrics) (4) · [Meta & Curation](#meta--curation-6-metrics) (6) · [Dataset-Level Metrics](#dataset-level-metrics-28-fields) (28) · [Utility & Validation](#utility--validation-32-modules) (32)

---

## No-Reference Quality (98 metrics)

### `adadqa_score` [↑](#categories)
> Ada-DQA adaptive diverse (higher=better) · ↑ higher=better

**[`adadqa`](src/ayase/modules/adadqa.py)** — Ada-DQA adaptive diverse quality feature VQA (ACM MM 2023)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Backend**: resnet
- **Packages**: gc, torch, torchvision
- **VRAM**: ~200 MB
- **Tests**: covered by [`test_adadqa.py`](tests/modules/per_module/test_adadqa.py)
- **Config**: `subsample=8`, `scales=[1.0, 0.5, 0.25]`

### `afine_score` [↑](#categories)
> A-FINE fidelity-naturalness (CVPR 2025) · ↑ higher=better

**[`afine`](src/ayase/modules/afine.py)** — A-FINE adaptive fidelity-naturalness IQA (CVPR 2025)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: opencv-python, pyiqa, torch
- **Tests**: covered by [`test_afine.py`](tests/modules/per_module/test_afine.py), [`test_image_iqa_metrics.py`](tests/modules/test_image_iqa_metrics.py)
- **Config**: `subsample=4`

### `aigcvqa_aesthetic` [↑](#categories)
> AIGC-VQA aesthetic branch

**[`aigcvqa`](src/ayase/modules/aigcvqa.py)** — AIGC-VQA holistic 3-branch AIGC perception (CVPRW 2024)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: Pillow, torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_aigcvqa.py`](tests/modules/per_module/test_aigcvqa.py)
- **Config**: `subsample=8`, `clip_model=openai/clip-vit-base-patch32`

### `aigcvqa_technical` [↑](#categories)
> AIGC-VQA technical branch

**[`aigcvqa`](src/ayase/modules/aigcvqa.py)** — AIGC-VQA holistic 3-branch AIGC perception (CVPRW 2024)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: Pillow, torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_aigcvqa.py`](tests/modules/per_module/test_aigcvqa.py)
- **Config**: `subsample=8`, `clip_model=openai/clip-vit-base-patch32`

### `aigv_static` [↑](#categories)
> AI video static quality

**[`aigv_assessor`](src/ayase/modules/aigv_assessor.py)** — AI-generated video quality (AIGV-Assessor model or CLIP proxy)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Backend**: aigv_assessor → clip
- **Packages**: Pillow, opencv-python, torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/IntMeGroup/AIGV-Assessor-static_quality" target="_blank">HF</a>
- **Tests**: covered by [`test_aigv_assessor.py`](tests/modules/per_module/test_aigv_assessor.py), [`test_motion_scene_semantic_metrics.py`](tests/modules/test_motion_scene_semantic_metrics.py)
- **Config**: `subsample=8`, `trust_remote_code=True`

### `aigvqa_score` [↑](#categories)
> AIGVQA multi-dimensional (higher=better) · ↑ higher=better

**[`aigvqa`](src/ayase/modules/aigvqa.py)** — AIGVQA multi-dimensional AIGC VQA (ICCVW 2025)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: Pillow, torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_aigvqa.py`](tests/modules/per_module/test_aigvqa.py)
- **Config**: `subsample=8`, `clip_model=openai/clip-vit-base-patch32`, `spatial_weight=0.4`, `temporal_weight=0.3`, `aesthetic_weight=0.3`

### `arniqa_score` [↑](#categories)
> ARNIQA (higher=better) · ↑ higher=better

**[`arniqa`](src/ayase/modules/arniqa.py)** — ARNIQA no-reference image quality assessment

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: opencv-python, pyiqa, torch
- **Tests**: covered by [`test_arniqa.py`](tests/modules/per_module/test_arniqa.py), [`test_iqa_research_metrics.py`](tests/modules/test_iqa_research_metrics.py)
- **Config**: `subsample=8`

### `brisque` [↑](#categories)
> BRISQUE (0-100, lower=better) · ↓ lower=better · 0-100

**[`brisque`](src/ayase/modules/brisque.py)** — BRISQUE no-reference image quality (lower=better)

- **Input**: img/vid · **Speed**: ⏱️ medium
- **Packages**: pyiqa
- **Tests**: covered by [`test_brisque.py`](tests/modules/per_module/test_brisque.py), [`test_fields_general.py`](tests/modules/test_fields_general.py), [`test_ml_basics.py`](tests/modules/test_ml_basics.py)
- **Config**: `subsample=3`, `warning_threshold=50.0`

### `bvqi_score` [↑](#categories)
> BVQI zero-shot blind VQA (higher=better) · ↑ higher=better

**[`bvqi`](src/ayase/modules/bvqi.py)** — BVQI zero-shot blind video quality index (ICME 2023)

- **Input**: img/vid · **Speed**: ⏱️ medium
- **Backend**: native → pyiqa
- **Packages**: bvqi, pyiqa, torch
- **Tests**: covered by [`test_bvqi.py`](tests/modules/per_module/test_bvqi.py)
- **Config**: `subsample=8`

### `chipqa_score` [↑](#categories)
> ChipQA space-time-chip NR-VQA (higher=better) · ↑ higher=better

**[`chipqa`](src/ayase/modules/chipqa.py)** — ChipQA no-reference video quality via official feature extractor and LIVE-Livestream SVR

- **Input**: vid · **Speed**: ⚡ fast
- **Backend**: chipqa
- **Packages**: joblib, matplotlib, numba, opencv-python, scikit-learn, scipy
- **Tests**: covered by [`test_chipqa.py`](tests/modules/per_module/test_chipqa.py)
- **Config**: `timeout_sec=1800`

### `clifvqa_score` [↑](#categories)
> CLiF-VQA human feelings (higher=better) · ↑ higher=better

**[`clifvqa`](src/ayase/modules/clifvqa.py)** — CLiF-VQA human feelings VQA via CLIP (2024)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: Pillow, torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_clifvqa.py`](tests/modules/per_module/test_clifvqa.py)
- **Config**: `subsample=8`, `clip_model=openai/clip-vit-base-patch32`

### `clip_iqa_score` [↑](#categories)
> CLIP-IQA semantic quality (0-1, higher=better) · ↑ higher=better · 0-1

**[`clip_iqa`](src/ayase/modules/clip_iqa.py)** — CLIP-based no-reference image quality assessment

- **Input**: img/vid · **Speed**: ⏱️ medium
- **Packages**: pyiqa
- **Tests**: covered by [`test_clip_iqa.py`](tests/modules/per_module/test_clip_iqa.py), [`test_ml_basics.py`](tests/modules/test_ml_basics.py)
- **Config**: `subsample=5`, `warning_threshold=0.4`

### `clipvqa_score` [↑](#categories)
> CLIPVQA CLIP-based VQA (higher=better) · ↑ higher=better

**[`clipvqa`](src/ayase/modules/clipvqa.py)** — CLIPVQA CLIP-based spatiotemporal VQA (TIP 2024)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Backend**: native → clip
- **Packages**: Pillow, clipvqa, torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_clipvqa.py`](tests/modules/per_module/test_clipvqa.py)
- **Config**: `subsample=8`

### `cnniqa_score` [↑](#categories)
> CNNIQA blind CNN IQA · ↑ higher=better

**[`cnniqa`](src/ayase/modules/cnniqa.py)** — CNNIQA blind CNN-based image quality assessment

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: opencv-python, pyiqa, torch
- **Tests**: covered by [`test_cnniqa.py`](tests/modules/per_module/test_cnniqa.py), [`test_image_iqa_metrics.py`](tests/modules/test_image_iqa_metrics.py)
- **Config**: `subsample=4`

### `compare2score` [↑](#categories)
> Compare2Score comparison-based · ↑ higher=better

**[`compare2score`](src/ayase/modules/compare2score.py)** — Compare2Score comparison-based NR image quality

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: opencv-python, pyiqa, torch
- **Tests**: covered by [`test_compare2score.py`](tests/modules/per_module/test_compare2score.py), [`test_image_iqa_metrics.py`](tests/modules/test_image_iqa_metrics.py)
- **Config**: `subsample=4`

### `contrique_score` [↑](#categories)
> CONTRIQUE contrastive IQA (higher=better) · ↑ higher=better

**[`contrique`](src/ayase/modules/contrique.py)** — Contrastive no-reference IQA

- **Input**: img/vid · **Speed**: ⏱️ medium
- **Packages**: pyiqa
- **Tests**: covered by [`test_contrique.py`](tests/modules/per_module/test_contrique.py), [`test_ml_basics.py`](tests/modules/test_ml_basics.py)
- **Config**: `subsample=5`

### `conviqt_score` [↑](#categories)
> CONVIQT contrastive NR-VQA (higher=better) · ↑ higher=better

**[`conviqt`](src/ayase/modules/conviqt.py)** — CONVIQT contrastive self-supervised NR-VQA (TIP 2023)

- **Input**: img/vid · **Speed**: ⏱️ medium
- **Backend**: native → pyiqa
- **Packages**: conviqt, pyiqa, torch
- **Tests**: covered by [`test_conviqt.py`](tests/modules/per_module/test_conviqt.py)
- **Config**: `subsample=8`

### `cover_score` [↑](#categories)
> COVER overall (higher=better) · ↑ higher=better

**[`cover`](src/ayase/modules/cover.py)** — COVER 3-branch comprehensive video quality (semantic + aesthetic + technical)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Backend**: cover → dover
- **Packages**: cover, opencv-python, pyiqa, torch
- **VRAM**: ~800 MB
- **Tests**: covered by [`test_cover.py`](tests/modules/per_module/test_cover.py), [`test_iqa_research_metrics.py`](tests/modules/test_iqa_research_metrics.py)
- **Config**: `subsample=8`, `quality_threshold=30.0`

### `cover_technical` [↑](#categories)
> COVER technical branch

**[`cover`](src/ayase/modules/cover.py)** — COVER 3-branch comprehensive video quality (semantic + aesthetic + technical)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Backend**: cover → dover
- **Packages**: cover, opencv-python, pyiqa, torch
- **VRAM**: ~800 MB
- **Tests**: covered by [`test_cover.py`](tests/modules/per_module/test_cover.py), [`test_iqa_research_metrics.py`](tests/modules/test_iqa_research_metrics.py)
- **Config**: `subsample=8`, `quality_threshold=30.0`

### `crave_score` [↑](#categories)
> CRAVE next-gen AIGC (higher=better) · ↑ higher=better

**[`crave`](src/ayase/modules/crave.py)** — CRAVE content-rich AIGC video evaluator (2025)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Packages**: Pillow, torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_crave.py`](tests/modules/per_module/test_crave.py)
- **Config**: `subsample=12`, `clip_model=openai/clip-vit-base-patch32`, `quality_weight=0.35`, `richness_weight=0.35`, `coherence_weight=0.3`

### `dbcnn_score` [↑](#categories)
> DBCNN bilinear CNN (higher=better) · ↑ higher=better

**[`dbcnn`](src/ayase/modules/dbcnn.py)** — DBCNN deep bilinear CNN for no-reference IQA

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: opencv-python, pyiqa, torch
- **Tests**: covered by [`test_dbcnn.py`](tests/modules/per_module/test_dbcnn.py), [`test_iqa_research_metrics.py`](tests/modules/test_iqa_research_metrics.py)
- **Config**: `subsample=8`

### `deepdc_score` [↑](#categories)
> DeepDC distribution conformance (lower=better) · ↓ lower=better

**[`deepdc`](src/ayase/modules/deepdc.py)** — DeepDC distribution conformance NR-IQA via pyiqa (2024, lower=better)

- **Input**: img/vid · **Speed**: ⏱️ medium
- **Backend**: pyiqa
- **Packages**: pyiqa, torch
- **Tests**: covered by [`test_deepdc.py`](tests/modules/per_module/test_deepdc.py)
- **Config**: `subsample=8`

### `discovqa_score` [↑](#categories)
> DisCoVQA distortion-content (higher=better) · ↑ higher=better

**[`discovqa`](src/ayase/modules/discovqa.py)** — DisCoVQA temporal distortion-content VQA (2023)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: opencv-python, torch, torchvision
- **VRAM**: ~200 MB
- **Tests**: covered by [`test_discovqa.py`](tests/modules/per_module/test_discovqa.py)
- **Config**: `subsample=8`, `frame_size=224`

### `dover_score` [↑](#categories)
> DOVER overall (higher=better) · ↑ higher=better

**[`dover`](src/ayase/modules/dover.py)** — DOVER disentangled technical + aesthetic VQA (ICCV 2023)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Backend**: heuristic → native → onnx → pyiqa
- **Packages**: onnxruntime, pyiqa, torch
- **VRAM**: ~800 MB
- **Source**: <a href="https://github.com/VQAssessment/DOVER.git" target="_blank">GitHub</a> · <a href="https://huggingface.co/dover/DOVER.pth" target="_blank">HF</a>
- **Tests**: covered by [`test_dover.py`](tests/modules/per_module/test_dover.py), [`test_ml_basics.py`](tests/modules/test_ml_basics.py)
- **Config**: `warning_threshold=0.4`

**[`unified_vqa`](src/ayase/modules/unified_vqa.py)** — Unified-VQA FR+NR multi-task quality assessment (2025)

- **Input**: img/vid +ref · **Speed**: ⏱️ medium · GPU
- **Backend**: clip → resnet
- **Packages**: Pillow, clip (openai), gc, torch, torchvision
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/torchvision/resnet50" target="_blank">HF</a>
- **Tests**: covered by [`test_unified_vqa.py`](tests/modules/per_module/test_unified_vqa.py)
- **Config**: `subsample=8`, `clip_model=ViT-B/32`

### `dover_technical` [↑](#categories)
> DOVER technical quality · 0-1 sigmoid

**[`dover`](src/ayase/modules/dover.py)** — DOVER disentangled technical + aesthetic VQA (ICCV 2023)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Backend**: heuristic → native → onnx → pyiqa
- **Packages**: onnxruntime, pyiqa, torch
- **VRAM**: ~800 MB
- **Source**: <a href="https://github.com/VQAssessment/DOVER.git" target="_blank">GitHub</a> · <a href="https://huggingface.co/dover/DOVER.pth" target="_blank">HF</a>
- **Tests**: covered by [`test_dover.py`](tests/modules/per_module/test_dover.py), [`test_ml_basics.py`](tests/modules/test_ml_basics.py)
- **Config**: `warning_threshold=0.4`

### `fast_vqa_score` [↑](#categories)
> 0-100 · ↑ higher=better

**[`fast_vqa`](src/ayase/modules/fast_vqa.py)** — Deep Learning Video Quality Assessment (FAST-VQA)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Packages**: PyYAML, decord, torch, traceback
- **Tests**: covered by [`test_fast_vqa.py`](tests/modules/per_module/test_fast_vqa.py)
- **Config**: `model_type=FasterVQA`

### `faver_score` [↑](#categories)
> FAVER variable frame rate (higher=better) · ↑ higher=better

**[`faver`](src/ayase/modules/faver.py)** — FAVER blind VQA for variable frame rate videos (2024)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Packages**: gc, torch, torchvision
- **VRAM**: ~200 MB
- **Tests**: covered by [`test_faver.py`](tests/modules/per_module/test_faver.py)
- **Config**: `subsample=16`

### `finevq_score` [↑](#categories)
> FineVQ fine-grained UGC VQA (CVPR 2025) · ↑ higher=better

**[`finevq`](src/ayase/modules/finevq.py)** — Fine-grained video quality (FineVQ model or TOPIQ+handcrafted)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Backend**: finevq → topiq_handcrafted
- **Packages**: Pillow, opencv-python, pyiqa, torch, transformers
- **Source**: <a href="https://huggingface.co/IntMeGroup/FineVQ_score" target="_blank">HF</a>
- **Tests**: covered by [`test_finevq.py`](tests/modules/per_module/test_finevq.py), [`test_video_native_metrics.py`](tests/modules/test_video_native_metrics.py)
- **Config**: `subsample=8`, `trust_remote_code=True`, `weights={'sharpness': 0.2, 'colorfulness': 0.15, 'noise': 0.2, 'temporal_stability': 0.25, 'content_richness': 0.2}`

### `gamival_score` [↑](#categories)
> GAMIVAL cloud gaming NR-VQA (higher=better) · ↑ higher=better

**[`gamival`](src/ayase/modules/gamival.py)** — GAMIVAL cloud gaming NR-VQA: 1156 NSS + 1024 3D-CNN features (2023)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Backend**: nss_only → full
- **Packages**: gc, joblib, torch, torchvision
- **VRAM**: ~200 MB
- **Tests**: covered by [`test_gamival.py`](tests/modules/per_module/test_gamival.py)
- **Config**: `subsample=8`

### `hyperiqa_score` [↑](#categories)
> HyperIQA adaptive NR-IQA · ↑ higher=better

**[`hyperiqa`](src/ayase/modules/hyperiqa.py)** — HyperIQA adaptive hypernetwork NR image quality

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: opencv-python, pyiqa, torch
- **Tests**: covered by [`test_hyperiqa.py`](tests/modules/per_module/test_hyperiqa.py), [`test_image_iqa_metrics.py`](tests/modules/test_image_iqa_metrics.py)
- **Config**: `subsample=4`

### `ilniqe` [↑](#categories)
> IL-NIQE Integrated Local NIQE (lower=better) · ↓ lower=better

**[`ilniqe`](src/ayase/modules/ilniqe.py)** — IL-NIQE integrated local no-reference quality (lower=better)

- **Input**: img/vid · **Speed**: ⏱️ medium
- **Packages**: pyiqa
- **Tests**: covered by [`test_ilniqe.py`](tests/modules/per_module/test_ilniqe.py), [`test_perceptual_metrics.py`](tests/modules/test_perceptual_metrics.py)
- **Config**: `subsample=3`, `warning_threshold=50.0`

### `internvqa_score` [↑](#categories)
> InternVQA video quality (higher=better) · ↑ higher=better

**[`internvqa`](src/ayase/modules/internvqa.py)** — InternVQA lightweight compressed video quality (2025)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Backend**: resnet
- **Packages**: gc, torch, torchvision
- **VRAM**: ~200 MB
- **Tests**: covered by [`test_internvqa.py`](tests/modules/per_module/test_internvqa.py)
- **Config**: `subsample=8`

### `kvq_score` [↑](#categories)
> KVQ saliency-guided VQA (CVPR 2025) · ↑ higher=better

**[`kvq`](src/ayase/modules/kvq.py)** — Saliency-guided video quality (KVQ model or TOPIQ+saliency)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Backend**: kvq → topiq_saliency
- **Packages**: opencv-python, pyiqa, torch, transformers
- **Source**: <a href="https://huggingface.co/lero233/KVQ" target="_blank">HF</a>
- **Tests**: covered by [`test_kvq.py`](tests/modules/per_module/test_kvq.py), [`test_video_native_metrics.py`](tests/modules/test_video_native_metrics.py)
- **Config**: `subsample=8`, `trust_remote_code=True`

### `liqe_score` [↑](#categories)
> LIQE lightweight IQA (higher=better) · ↑ higher=better

**[`liqe`](src/ayase/modules/liqe.py)** — LIQE lightweight no-reference IQA

- **Input**: img/vid · **Speed**: ⏱️ medium
- **Packages**: pyiqa
- **Tests**: covered by [`test_liqe.py`](tests/modules/per_module/test_liqe.py), [`test_ml_basics.py`](tests/modules/test_ml_basics.py)
- **Config**: `subsample=5`, `warning_threshold=2.5`

### `lmmvqa_score` [↑](#categories)
> LMM-VQA spatiotemporal (higher=better) · ↑ higher=better

**[`lmmvqa`](src/ayase/modules/lmmvqa.py)** — LMM-VQA spatiotemporal LMM VQA (IEEE 2024)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Backend**: clip → resnet
- **Packages**: Pillow, clip (openai), gc, torch, torchvision
- **VRAM**: ~600 MB
- **Tests**: covered by [`test_lmmvqa.py`](tests/modules/per_module/test_lmmvqa.py)
- **Config**: `subsample=8`, `clip_model=ViT-B/32`

### `maclip_score` [↑](#categories)
> MACLIP multi-attribute CLIP NR-IQA (higher=better) · ↑ higher=better

**[`maclip`](src/ayase/modules/maclip.py)** — MACLIP multi-attribute CLIP no-reference quality (higher=better)

- **Input**: img/vid · **Speed**: ⏱️ medium
- **Packages**: pyiqa
- **Tests**: covered by [`test_maclip.py`](tests/modules/per_module/test_maclip.py), [`test_perceptual_metrics.py`](tests/modules/test_perceptual_metrics.py)
- **Config**: `subsample=3`

### `maniqa_score` [↑](#categories)
> MANIQA multi-attention (higher=better) · ↑ higher=better

**[`maniqa`](src/ayase/modules/maniqa.py)** — MANIQA multi-dimension attention no-reference IQA

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: opencv-python, pyiqa, torch
- **Tests**: covered by [`test_maniqa.py`](tests/modules/per_module/test_maniqa.py), [`test_iqa_research_metrics.py`](tests/modules/test_iqa_research_metrics.py)
- **Config**: `subsample=8`

### `maxvqa_score` [↑](#categories)
> MaxVQA explainable quality (higher=better) · ↑ higher=better

**[`maxvqa`](src/ayase/modules/maxvqa.py)** — MaxVQA explainable language-prompted VQA (ACM MM 2023)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Backend**: native → clip
- **Packages**: Pillow, maxvqa, torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_maxvqa.py`](tests/modules/per_module/test_maxvqa.py)
- **Config**: `subsample=8`

### `mc360iqa_score` [↑](#categories)
> MC360IQA blind 360 (higher=better) · ↑ higher=better

**[`mc360iqa`](src/ayase/modules/mc360iqa.py)** — MC360IQA blind 360 IQA (2019)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Backend**: resnet
- **Packages**: gc, torch, torchvision
- **VRAM**: ~200 MB
- **Tests**: covered by [`test_mc360iqa.py`](tests/modules/per_module/test_mc360iqa.py)
- **Config**: `subsample=8`, `n_viewports=10`, `viewport_size=224`

### `mdtvsfa_score` [↑](#categories)
> MDTVSFA fragment-based VQA (higher=better) · ↑ higher=better

**[`mdtvsfa`](src/ayase/modules/mdtvsfa.py)** — Multi-Dimensional fragment-based VQA

- **Input**: img/vid · **Speed**: ⏱️ medium
- **Packages**: pyiqa
- **Tests**: covered by [`test_mdtvsfa.py`](tests/modules/per_module/test_mdtvsfa.py), [`test_ml_basics.py`](tests/modules/test_ml_basics.py)
- **Config**: `subsample=5`

### `mdvqa_distortion` [↑](#categories)
> MD-VQA distortion quality (higher=better) · ↑ higher=better

**[`mdvqa`](src/ayase/modules/mdvqa.py)** — MD-VQA multi-dimensional UGC live VQA (CVPR 2023)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: Pillow, clip (openai), opencv-python, torch, torchvision
- **VRAM**: ~600 MB
- **Tests**: covered by [`test_mdvqa.py`](tests/modules/per_module/test_mdvqa.py)
- **Config**: `subsample=8`, `frame_size=224`

### `mdvqa_motion` [↑](#categories)
> MD-VQA motion quality (higher=better) · ↑ higher=better

**[`mdvqa`](src/ayase/modules/mdvqa.py)** — MD-VQA multi-dimensional UGC live VQA (CVPR 2023)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: Pillow, clip (openai), opencv-python, torch, torchvision
- **VRAM**: ~600 MB
- **Tests**: covered by [`test_mdvqa.py`](tests/modules/per_module/test_mdvqa.py)
- **Config**: `subsample=8`, `frame_size=224`

### `mdvqa_semantic` [↑](#categories)
> MD-VQA semantic quality (higher=better) · ↑ higher=better

**[`mdvqa`](src/ayase/modules/mdvqa.py)** — MD-VQA multi-dimensional UGC live VQA (CVPR 2023)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: Pillow, clip (openai), opencv-python, torch, torchvision
- **VRAM**: ~600 MB
- **Tests**: covered by [`test_mdvqa.py`](tests/modules/per_module/test_mdvqa.py)
- **Config**: `subsample=8`, `frame_size=224`

### `memoryvqa_score` [↑](#categories)
> Memory-VQA human memory (higher=better) · ↑ higher=better

**[`memoryvqa`](src/ayase/modules/memoryvqa.py)** — Memory-VQA human memory system VQA (Neurocomputing 2025)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Backend**: resnet
- **Packages**: gc, torch, torchvision
- **VRAM**: ~200 MB
- **Tests**: covered by [`test_memoryvqa.py`](tests/modules/per_module/test_memoryvqa.py)
- **Config**: `subsample=12`, `memory_size=8`

### `mm_pcqa_score` [↑](#categories)
> MM-PCQA multi-modal (higher=better) · ↑ higher=better

**[`mm_pcqa`](src/ayase/modules/mm_pcqa.py)** — MM-PCQA multi-modal point cloud QA (IJCAI 2023)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Backend**: resnet
- **Packages**: gc, open3d, opencv-python, torch, torchvision
- **VRAM**: ~200 MB
- **Tests**: covered by [`test_mm_pcqa.py`](tests/modules/per_module/test_mm_pcqa.py)
- **Config**: `n_views=6`, `render_size=224`

### `modularbvqa_score` [↑](#categories)
> ModularBVQA resolution-aware (higher=better) · ↑ higher=better

**[`modularbvqa`](src/ayase/modules/modularbvqa.py)** — ModularBVQA resolution/framerate-aware blind VQA (CVPR 2024) — CLIP ViT-B backbone + Laplacian spatial + SlowFast temporal rectifiers

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: Pillow, clip (openai), opencv-python, torch, torchvision
- **VRAM**: ~600 MB
- **Tests**: covered by [`test_modularbvqa.py`](tests/modules/per_module/test_modularbvqa.py)
- **Config**: `subsample=8`, `frame_size=224`

### `musiq_score` [↑](#categories)
> MUSIQ multi-scale IQA (higher=better) · ↑ higher=better

**[`musiq`](src/ayase/modules/musiq.py)** — Multi-Scale Image Quality Transformer (no-reference)

- **Input**: img/vid · **Speed**: ⏱️ medium
- **Packages**: pyiqa
- **Tests**: covered by [`test_musiq.py`](tests/modules/per_module/test_musiq.py), [`test_ml_basics.py`](tests/modules/test_ml_basics.py)
- **Config**: `variant=musiq`, `subsample=5`, `warning_threshold=40.0`

### `naturalness_score` [↑](#categories)
> Natural scene statistics · ↑ higher=better

**[`naturalness`](src/ayase/modules/naturalness.py)** — Measures naturalness of content (natural vs synthetic)

- **Input**: img/vid · **Speed**: ⏱️ medium
- **Packages**: Pillow, pyiqa
- **Tests**: covered by [`test_naturalness.py`](tests/modules/per_module/test_naturalness.py), [`test_reference_and_meta_metrics.py`](tests/modules/test_reference_and_meta_metrics.py)
- **Config**: `use_pyiqa=True`, `subsample=2`, `warning_threshold=0.4`

### `niqe` [↑](#categories)
> Natural Image Quality Evaluator (lower=better) · ↓ lower=better

**[`niqe`](src/ayase/modules/niqe.py)** — Natural Image Quality Evaluator (no-reference)

- **Input**: img/vid · **Speed**: ⏱️ medium
- **Packages**: pyiqa
- **Tests**: covered by [`test_niqe.py`](tests/modules/per_module/test_niqe.py), [`test_fields_general.py`](tests/modules/test_fields_general.py), [`test_reference_and_meta_metrics.py`](tests/modules/test_reference_and_meta_metrics.py)
- **Config**: `subsample=2`, `warning_threshold=7.0`

### `nr_gvqm_score` [↑](#categories)
> NR-GVQM cloud gaming VQA (higher=better) · ↑ higher=better

**[`nr_gvqm`](src/ayase/modules/nr_gvqm.py)** — NR-GVQM no-reference gaming video quality (ISM 2018, 9 features)

- **Input**: img/vid · **Speed**: ⚡ fast
- **Backend**: native
- **Tests**: covered by [`test_nr_gvqm.py`](tests/modules/per_module/test_nr_gvqm.py)
- **Config**: `subsample=8`

### `nrqm` [↑](#categories)
> NRQM No-Reference Quality Metric (higher=better) · ↑ higher=better

**[`nrqm`](src/ayase/modules/nrqm.py)** — NRQM no-reference quality metric for super-resolution (higher=better)

- **Input**: img/vid · **Speed**: ⏱️ medium
- **Packages**: pyiqa
- **Tests**: covered by [`test_nrqm.py`](tests/modules/per_module/test_nrqm.py), [`test_perceptual_metrics.py`](tests/modules/test_perceptual_metrics.py)
- **Config**: `subsample=3`

### `paq2piq_score` [↑](#categories)
> PaQ-2-PiQ patch-to-picture (CVPR 2020) · ↑ higher=better

**[`paq2piq`](src/ayase/modules/paq2piq.py)** — PaQ-2-PiQ patch-to-picture NR quality (CVPR 2020)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: opencv-python, pyiqa, torch
- **Tests**: covered by [`test_paq2piq.py`](tests/modules/per_module/test_paq2piq.py), [`test_image_iqa_metrics.py`](tests/modules/test_image_iqa_metrics.py)
- **Config**: `subsample=4`

### `pi_score` [↑](#categories)
> Perceptual Index (PIRM challenge, lower=better) · ↓ lower=better · PIRM challenge

**[`pi`](src/ayase/modules/pi_metric.py)** — Perceptual Index (PIRM challenge metric, lower=better)

- **Input**: img/vid · **Speed**: ⏱️ medium
- **Packages**: pyiqa
- **Tests**: covered by [`test_pi.py`](tests/modules/per_module/test_pi.py), [`test_perceptual_metrics.py`](tests/modules/test_perceptual_metrics.py), [`test_docs_integrity.py`](tests/test_docs_integrity.py)
- **Config**: `subsample=3`

### `piqe` [↑](#categories)
> PIQE perception-based NR-IQA (lower=better) · ↓ lower=better

**[`piqe`](src/ayase/modules/piqe.py)** — PIQE perception-based no-reference quality (lower=better)

- **Input**: img/vid · **Speed**: ⏱️ medium
- **Packages**: pyiqa
- **Tests**: covered by [`test_piqe.py`](tests/modules/per_module/test_piqe.py), [`test_perceptual_metrics.py`](tests/modules/test_perceptual_metrics.py)
- **Config**: `subsample=3`, `warning_threshold=50.0`

### `presresq_score` [↑](#categories)
> PreResQ-R1 rank+score (higher=better) · ↑ higher=better

**[`presresq`](src/ayase/modules/presresq.py)** — PreResQ-R1 rank+score VQA (2025)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Backend**: clip → resnet
- **Packages**: Pillow, clip (openai), gc, torch, torchvision
- **VRAM**: ~600 MB
- **Tests**: covered by [`test_presresq.py`](tests/modules/per_module/test_presresq.py)
- **Config**: `subsample=8`, `clip_model=ViT-B/32`

### `promptiqa_score` [↑](#categories)
> Few-shot NR-IQA score · ↑ higher=better

**[`promptiqa`](src/ayase/modules/promptiqa.py)** — Prompt-guided NR-IQA (PromptIQA via pyiqa, TOPIQ-NR, or CLIP-IQA+ fallback)

- **Input**: img/vid · **Speed**: ⏱️ medium
- **Backend**: none → promptiqa → topiq_nr
- **Packages**: Pillow, opencv-python, pyiqa, torch
- **Tests**: covered by [`test_promptiqa.py`](tests/modules/per_module/test_promptiqa.py), [`test_motion_scene_semantic_metrics.py`](tests/modules/test_motion_scene_semantic_metrics.py)
- **Config**: `subsample=4`

### `provqa_score` [↑](#categories)
> ProVQA progressive 360 (higher=better) · ↑ higher=better

**[`provqa`](src/ayase/modules/provqa.py)** — ProVQA progressive blind 360 VQA (2022)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Backend**: resnet
- **Packages**: gc, torch, torchvision
- **VRAM**: ~200 MB
- **Tests**: covered by [`test_provqa.py`](tests/modules/per_module/test_provqa.py)
- **Config**: `subsample=8`, `n_fine_crops=6`

### `ptmvqa_score` [↑](#categories)
> PTM-VQA multi-PTM fusion (higher=better) · ↑ higher=better

**[`ptmvqa`](src/ayase/modules/ptmvqa.py)** — PTM-VQA multi-PTM fusion VQA (CVPR 2024)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Backend**: multi_ptm
- **Packages**: Pillow, torch, torchvision, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_ptmvqa.py`](tests/modules/per_module/test_ptmvqa.py)
- **Config**: `subsample=8`, `clip_model=openai/clip-vit-base-patch32`

### `qalign_quality` [↑](#categories)
> Q-Align technical quality (1-5, higher=better) · ↑ higher=better · 1-5

**[`q_align`](src/ayase/modules/q_align.py)** — Q-Align unified quality + aesthetic assessment (ICML 2024)

- **Input**: img/vid · **Speed**: 🐌 slow · GPU
- **Packages**: Pillow, torch, transformers
- **VRAM**: ~14 GB
- **Source**: <a href="https://huggingface.co/q-future/one-align" target="_blank">HF</a>
- **Tests**: covered by [`test_q_align.py`](tests/modules/per_module/test_q_align.py), [`test_ml_basics.py`](tests/modules/test_ml_basics.py)
- **Config**: `model_name=q-future/one-align`, `dtype=float16`, `device=auto`, `subsample=8`, `max_frames=16`, `warning_threshold=2.5`, `trust_remote_code=True`

### `qclip_score` [↑](#categories)
> Q-CLIP VLM-based (higher=better) · ↑ higher=better

**[`qclip`](src/ayase/modules/qclip.py)** — Q-CLIP VLM-based VQA (2025)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: Pillow, torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_qclip.py`](tests/modules/per_module/test_qclip.py)
- **Config**: `subsample=8`, `clip_model=openai/clip-vit-base-patch32`

### `qcn_score` [↑](#categories)
> Geometric order blind IQA · ↑ higher=better

**[`qcn`](src/ayase/modules/qcn.py)** — Blind IQA (QCN via pyiqa, or HyperIQA fallback)

- **Input**: img/vid · **Speed**: ⏱️ medium
- **Backend**: none → qcn → hyperiqa
- **Packages**: Pillow, opencv-python, pyiqa, torch
- **Tests**: covered by [`test_qcn.py`](tests/modules/per_module/test_qcn.py), [`test_motion_scene_semantic_metrics.py`](tests/modules/test_motion_scene_semantic_metrics.py)
- **Config**: `subsample=4`

### `qualiclip_score` [↑](#categories)
> QualiCLIP opinion-unaware (higher=better) · ↑ higher=better

**[`qualiclip`](src/ayase/modules/qualiclip.py)** — QualiCLIP opinion-unaware CLIP-based no-reference IQA

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: opencv-python, pyiqa, torch
- **Tests**: covered by [`test_qualiclip.py`](tests/modules/per_module/test_qualiclip.py), [`test_iqa_research_metrics.py`](tests/modules/test_iqa_research_metrics.py)
- **Config**: `subsample=8`

### `rapique_score` [↑](#categories)
> RAPIQUE bandpass+CNN NR-VQA (higher=better) · ↑ higher=better

**[`rapique`](src/ayase/modules/rapique.py)** — RAPIQUE rapid NR-VQA via bandpass NSS + CNN features (IEEE OJSP 2021)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: opencv-python, torch, torchvision
- **VRAM**: ~200 MB
- **Tests**: covered by [`test_rapique.py`](tests/modules/per_module/test_rapique.py)
- **Config**: `subsample=8`, `frame_size=520`

### `rqvqa_score` [↑](#categories)
> RQ-VQA rich quality-aware (CVPR 2024 winner) · ↑ higher=better

**[`rqvqa`](src/ayase/modules/rqvqa.py)** — Multi-attribute video quality (RQ-VQA model or CLIP-IQA+)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Backend**: rqvqa → clipiqa
- **Packages**: opencv-python, pyiqa, torch, transformers
- **Source**: <a href="https://huggingface.co/AkaneTendo25/ayase-models" target="_blank">HF</a>
- **Tests**: covered by [`test_rqvqa.py`](tests/modules/per_module/test_rqvqa.py), [`test_video_native_metrics.py`](tests/modules/test_video_native_metrics.py)
- **Config**: `subsample=8`, `trust_remote_code=True`, `dimensions={'clarity': 0.25, 'aesthetics': 0.2, 'motion_naturalness': 0.25, 'semantic_coherence': 0.15, 'overall_impression': 0.15}`

### `sama_score` [↑](#categories)
> SAMA scaling+masking (higher=better) · ↑ higher=better

**[`sama`](src/ayase/modules/sama.py)** — SAMA scaling+masking VQA (2024)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Backend**: resnet_sama
- **Packages**: Pillow, torch, torchvision
- **VRAM**: ~200 MB
- **Tests**: covered by [`test_sama.py`](tests/modules/per_module/test_sama.py)
- **Config**: `subsample=8`, `mask_ratio=0.5`

### `siamvqa_score` [↑](#categories)
> SiamVQA Siamese high-res (higher=better) · ↑ higher=better

**[`siamvqa`](src/ayase/modules/siamvqa.py)** — SiamVQA Siamese high-resolution VQA (2025)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Backend**: siamese_resnet
- **Packages**: Pillow, torch, torchvision
- **VRAM**: ~200 MB
- **Tests**: covered by [`test_siamvqa.py`](tests/modules/per_module/test_siamvqa.py)
- **Config**: `subsample=8`, `num_crops=5`, `crop_size=224`

### `simplevqa_score` [↑](#categories)
> SimpleVQA Swin+SlowFast (higher=better) · ↑ higher=better

**[`simplevqa`](src/ayase/modules/simplevqa.py)** — SimpleVQA Swin+SlowFast blind VQA (2022)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: opencv-python, torch, torchvision
- **Tests**: covered by [`test_simplevqa.py`](tests/modules/per_module/test_simplevqa.py)
- **Config**: `slow_frames=8`, `fast_frames=32`, `frame_size=224`, `fast_frame_size=112`

### `spectral_entropy` [↑](#categories)
> DINOv2 spectral entropy

**[`spectral_complexity`](src/ayase/modules/spectral.py)** — Analyzes spectral complexity (Effective Rank) of video features (DINOv2)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Packages**: torch, torchvision
- **VRAM**: ~400 MB
- **Source**: <a href="https://huggingface.co/facebookresearch/dinov2" target="_blank">HF</a>
- **Tests**: covered by [`test_spectral_complexity.py`](tests/modules/per_module/test_spectral_complexity.py), [`test_docs_integrity.py`](tests/test_docs_integrity.py)
- **Config**: `model_type=dinov2_vits14`, `sample_rate=8`, `min_rank_ratio=0.05`, `max_entropy_threshold=6.0`

### `spectral_rank` [↑](#categories)
> DINOv2 effective rank ratio

**[`spectral_complexity`](src/ayase/modules/spectral.py)** — Analyzes spectral complexity (Effective Rank) of video features (DINOv2)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Packages**: torch, torchvision
- **VRAM**: ~400 MB
- **Source**: <a href="https://huggingface.co/facebookresearch/dinov2" target="_blank">HF</a>
- **Tests**: covered by [`test_spectral_complexity.py`](tests/modules/per_module/test_spectral_complexity.py), [`test_docs_integrity.py`](tests/test_docs_integrity.py)
- **Config**: `model_type=dinov2_vits14`, `sample_rate=8`, `min_rank_ratio=0.05`, `max_entropy_threshold=6.0`

### `speedqa_score` [↑](#categories)
> SpEED-QA entropic differencing (higher=better) · ↑ higher=better

**[`speedqa`](src/ayase/modules/speedqa.py)** — SpEED-QA spatial efficient entropic differencing NR-VQA (Bampis 2017)

- **Input**: vid · **Speed**: ⚡ fast
- **Backend**: native → speedqa_pkg
- **Packages**: speedqa
- **Tests**: covered by [`test_speedqa.py`](tests/modules/per_module/test_speedqa.py)
- **Config**: `subsample=8`

### `sqi_score` [↑](#categories)
> SQI streaming quality index · ↑ higher=better

**[`sqi`](src/ayase/modules/sqi.py)** — SQI streaming quality index (2016)

- **Input**: vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_sqi.py`](tests/modules/per_module/test_sqi.py)

### `sr4kvqa_score` [↑](#categories)
> SR4KVQA super-resolution 4K (higher=better) · ↑ higher=better

**[`sr4kvqa`](src/ayase/modules/sr4kvqa.py)** — SR4KVQA super-resolution 4K quality (2024)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Backend**: resnet_sr
- **Packages**: Pillow, torch, torchvision
- **VRAM**: ~200 MB
- **Tests**: covered by [`test_sr4kvqa.py`](tests/modules/per_module/test_sr4kvqa.py)
- **Config**: `subsample=8`, `patch_size=224`, `max_patches=9`

### `stablevqa_score` [↑](#categories)
> StableVQA video stability (higher=better) · ↑ higher=better

**[`stablevqa`](src/ayase/modules/stablevqa.py)** — StableVQA video stability quality assessment (ACM MM 2023)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Packages**: opencv-python, torch, torchvision
- **VRAM**: ~200 MB
- **Tests**: covered by [`test_stablevqa.py`](tests/modules/per_module/test_stablevqa.py)
- **Config**: `step=2`, `max_frames=120`, `frame_size=224`

### `t2v_quality` [↑](#categories)
> Video production quality · ↑ higher=better

**[`t2v_score`](src/ayase/modules/t2v_score.py)** — Text-to-Video alignment and quality scoring

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Packages**: torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_t2v_score.py`](tests/modules/per_module/test_t2v_score.py), [`test_fields_general.py`](tests/modules/test_fields_general.py), [`test_reference_and_meta_metrics.py`](tests/modules/test_reference_and_meta_metrics.py)
- **Config**: `model_name=openai/clip-vit-base-patch32`, `use_clip_fallback=True`, `num_frames=8`, `alignment_weight=0.5`, `quality_weight=0.5`, `device=auto`, `warning_threshold=0.6`

### `thqa_score` [↑](#categories)
> THQA talking head quality (higher=better) · ↑ higher=better

**[`thqa`](src/ayase/modules/thqa.py)** — THQA talking head quality assessment (ICIP 2024)

- **Input**: vid · **Speed**: ⚡ fast
- **Backend**: thqa
- **Packages**: thqa
- **Tests**: covered by [`test_thqa.py`](tests/modules/per_module/test_thqa.py)
- **Config**: `subsample=16`

### `tlvqm_score` [↑](#categories)
> TLVQM two-level video quality · ↑ higher=better

**[`tlvqm`](src/ayase/modules/tlvqm.py)** — Two-level video quality model (CNN-TLVQM or handcrafted fallback)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Backend**: handcrafted → cnn → cnn_svr → cnn_pretrained
- **Packages**: joblib, opencv-python, torch, torchvision
- **VRAM**: ~200 MB
- **Source**: <a href="https://github.com/jarikorhonen/cnn-tlvqm" target="_blank">GitHub</a>
- **Tests**: covered by [`test_tlvqm.py`](tests/modules/per_module/test_tlvqm.py), [`test_video_native_metrics.py`](tests/modules/test_video_native_metrics.py)
- **Config**: `subsample=8`

### `topiq_score` [↑](#categories)
> TOPIQ transformer-based IQA (higher=better) · ↑ higher=better

**[`topiq`](src/ayase/modules/topiq.py)** — TOPIQ transformer-based no-reference IQA

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: pyiqa, torch
- **Tests**: covered by [`test_topiq.py`](tests/modules/per_module/test_topiq.py), [`test_ml_basics.py`](tests/modules/test_ml_basics.py)
- **Config**: `variant=topiq_nr`, `subsample=5`, `warning_threshold=0.4`

### `tres_score` [↑](#categories)
> TReS transformer IQA (WACV 2022) · ↑ higher=better

**[`tres`](src/ayase/modules/tres.py)** — TReS transformer-based NR image quality (WACV 2022)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: opencv-python, pyiqa, torch
- **Tests**: covered by [`test_tres.py`](tests/modules/per_module/test_tres.py), [`test_image_iqa_metrics.py`](tests/modules/test_image_iqa_metrics.py)
- **Config**: `subsample=4`

### `uciqe_score` [↑](#categories)
> UCIQE underwater color (higher=better) · ↑ higher=better

**[`uciqe`](src/ayase/modules/uciqe.py)** — UCIQE underwater color image quality evaluation (2015)

- **Input**: img/vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_uciqe.py`](tests/modules/per_module/test_uciqe.py)
- **Config**: `c1=0.468`, `c2=0.2745`, `c3=0.2576`, `subsample=8`

### `ugvq_score` [↑](#categories)
> UGVQ unified generated VQ (higher=better) · ↑ higher=better

**[`ugvq`](src/ayase/modules/ugvq.py)** — UGVQ unified generated video quality (TOMM 2024)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Backend**: clip_ugvq
- **Packages**: Pillow, torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_ugvq.py`](tests/modules/per_module/test_ugvq.py)
- **Config**: `subsample=8`, `clip_model=openai/clip-vit-base-patch32`

### `uiqm_score` [↑](#categories)
> UIQM underwater quality (higher=better) · ↑ higher=better

**[`uiqm`](src/ayase/modules/uiqm.py)** — UIQM underwater image quality measure (Panetta et al. 2016)

- **Input**: img/vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_uiqm.py`](tests/modules/per_module/test_uiqm.py)
- **Config**: `c1=0.0282`, `c2=0.2953`, `c3=3.5753`, `subsample=8`

### `unified_vqa_score` [↑](#categories)
> Unified-VQA FR/NR quality (0-1, higher=better) · ↑ higher=better · 0-1

**[`unified_vqa`](src/ayase/modules/unified_vqa.py)** — Unified-VQA FR+NR multi-task quality assessment (2025)

- **Input**: img/vid +ref · **Speed**: ⏱️ medium · GPU
- **Backend**: clip → resnet
- **Packages**: Pillow, clip (openai), gc, torch, torchvision
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/torchvision/resnet50" target="_blank">HF</a>
- **Tests**: covered by [`test_unified_vqa.py`](tests/modules/per_module/test_unified_vqa.py)
- **Config**: `subsample=8`, `clip_model=ViT-B/32`

### `unique_score` [↑](#categories)
> UNIQUE unified NR-IQA (TIP 2021) · ↑ higher=better

**[`unique`](src/ayase/modules/unique_iqa.py)** — UNIQUE unified NR image quality (TIP 2021)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: opencv-python, pyiqa, torch
- **Tests**: covered by [`test_unique.py`](tests/modules/per_module/test_unique.py), [`test_image_iqa_metrics.py`](tests/modules/test_image_iqa_metrics.py), [`test_docs_integrity.py`](tests/test_docs_integrity.py)
- **Config**: `subsample=4`

### `vader_score` [↑](#categories)
> VADER reward alignment · ↑ higher=better

**[`vader`](src/ayase/modules/vader.py)** — VADER HPS v2 reward signal (ICLR 2025)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Backend**: hpsv2 → clip_aesthetic
- **Packages**: Pillow, hpsv2, torch, transformers
- **VRAM**: ~1.5 GB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-large-patch14" target="_blank">HF</a>
- **Tests**: covered by [`test_vader.py`](tests/modules/per_module/test_vader.py)
- **Config**: `subsample=8`, `clip_model=openai/clip-vit-large-patch14`

### `vbliinds_score` [↑](#categories)
> V-BLIINDS DCT-domain NSS (higher=better) · ↑ higher=better

**[`vbliinds`](src/ayase/modules/vbliinds.py)** — V-BLIINDS blind NR-VQA via DCT-domain GGD + motion coherency (Saad 2014)

- **Input**: img/vid · **Speed**: ⚡ fast
- **Backend**: builtin → skvideo
- **Packages**: scikit-video
- **Tests**: covered by [`test_vbliinds.py`](tests/modules/per_module/test_vbliinds.py)
- **Config**: `subsample=8`

### `video_atlas_score` [↑](#categories)
> Video ATLAS temporal artifacts · ↑ higher=better

**[`video_atlas`](src/ayase/modules/video_atlas.py)** — Video ATLAS temporal artifacts+stalls assessment (2018)

- **Input**: vid · **Speed**: ⚡ fast
- **Backend**: native → video_atlas_pkg
- **Tests**: covered by [`test_video_atlas.py`](tests/modules/per_module/test_video_atlas.py)
- **Config**: `subsample=16`

### `video_memorability` [↑](#categories)
> Memorability prediction

**[`video_memorability`](src/ayase/modules/video_memorability.py)** — Content memorability approximation (CLIP/DINOv2 feature statistics)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Backend**: clip → dinov2
- **Packages**: Pillow, opencv-python, torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_video_memorability.py`](tests/modules/per_module/test_video_memorability.py), [`test_fields_general.py`](tests/modules/test_fields_general.py), [`test_reference_and_meta_metrics.py`](tests/modules/test_reference_and_meta_metrics.py), +1 more
- **Config**: `subsample=5`

### `videoreward_vq` [↑](#categories)
> VideoReward visual quality

**[`videoreward`](src/ayase/modules/videoreward.py)** — VideoReward Kling multi-dim reward model (NeurIPS 2025)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Packages**: Pillow, torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_videoreward.py`](tests/modules/per_module/test_videoreward.py)
- **Config**: `subsample=8`, `clip_model=openai/clip-vit-base-patch32`

### `videoscore2_visual` [↑](#categories)
> VideoScore2 visual quality · ↑ higher=better · 0-10

**[`videoscore2`](src/ayase/modules/videoscore2.py)** — VideoScore2 3-dimensional generative video evaluation

- **Input**: img/vid · **Speed**: 🐌 slow · GPU
- **Backend**: transformers
- **Packages**: qwen-vl-utils, torch, transformers
- **VRAM**: ~16 GB
- **Source**: <a href="https://huggingface.co/TIGER-Lab/VideoScore2" target="_blank">HF</a>
- **Tests**: covered by [`test_videoscore2.py`](tests/modules/per_module/test_videoscore2.py), [`test_iqa_research_metrics.py`](tests/modules/test_iqa_research_metrics.py)
- **Config**: `model_name=TIGER-Lab/VideoScore2`, `infer_fps=2.0`, `max_new_tokens=1024`, `temperature=0.7`, `do_sample=True`, `trust_remote_code=True`

### `videoscore_visual` [↑](#categories)
> VideoScore visual quality · ↑ higher=better

**[`videoscore`](src/ayase/modules/videoscore.py)** — VideoScore 5-dimensional video quality assessment (1-4 scale)

- **Input**: img/vid · **Speed**: 🐌 slow · GPU
- **Packages**: Pillow, opencv-python, torch, transformers
- **Source**: <a href="https://huggingface.co/TIGER-Lab/VideoScore" target="_blank">HF</a>
- **Tests**: covered by [`test_videoscore.py`](tests/modules/per_module/test_videoscore.py), [`test_iqa_research_metrics.py`](tests/modules/test_iqa_research_metrics.py)
- **Config**: `model_name=TIGER-Lab/VideoScore`, `num_frames=8`, `trust_remote_code=True`

### `videval_score` [↑](#categories)
> VIDEVAL 60-feature fusion NR-VQA · ↑ higher=better

**[`videval`](src/ayase/modules/videval.py)** — VIDEVAL 60-feature hand-crafted NR-VQA (Tu et al. 2021)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Backend**: none → svr → linear
- **Packages**: joblib, opencv-python, torch
- **Tests**: covered by [`test_videval.py`](tests/modules/per_module/test_videval.py), [`test_video_native_metrics.py`](tests/modules/test_video_native_metrics.py)
- **Config**: `subsample=8`, `frame_size=520`

### `viideo_score` [↑](#categories)
> VIIDEO blind natural video statistics (lower=better) · ↓ lower=better

**[`viideo`](src/ayase/modules/viideo.py)** — VIIDEO blind NR-VQA via natural video statistics (Mittal 2016, lower=better)

- **Input**: vid · **Speed**: ⚡ fast
- **Backend**: builtin → skvideo
- **Packages**: scikit-video
- **Tests**: covered by [`test_viideo.py`](tests/modules/per_module/test_viideo.py)
- **Config**: `subsample=8`

### `vqa2_score` [↑](#categories)
> VQA² LMM quality (higher=better) · ↑ higher=better

**[`vqa2`](src/ayase/modules/vqa2.py)** — VQA^2 LMM video quality assessment (MM 2025)

- **Input**: img/vid · **Speed**: 🐌 slow · GPU
- **Backend**: qalign → clip
- **Packages**: Pillow, pyiqa, torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_vqa2.py`](tests/modules/per_module/test_vqa2.py)
- **Config**: `subsample=8`, `clip_model=openai/clip-vit-base-patch32`

### `vqathinker_score` [↑](#categories)
> VQAThinker GRPO (higher=better) · ↑ higher=better

**[`vqathinker`](src/ayase/modules/vqathinker.py)** — VQAThinker RL-based explainable VQA (2025)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Backend**: clip_thinker
- **Packages**: Pillow, torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_vqathinker.py`](tests/modules/per_module/test_vqathinker.py)
- **Config**: `subsample=8`, `clip_model=openai/clip-vit-base-patch32`, `temperature=0.07`

### `vqinsight_score` [↑](#categories)
> VQ-Insight ByteDance (higher=better) · ↑ higher=better

**[`vqinsight`](src/ayase/modules/vqinsight.py)** — VQ-Insight ByteDance multi-dim AIGC scoring (AAAI 2026)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Backend**: clip_vqinsight
- **Packages**: Pillow, torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_vqinsight.py`](tests/modules/per_module/test_vqinsight.py)
- **Config**: `subsample=8`, `clip_model=openai/clip-vit-base-patch32`

### `vsfa_score` [↑](#categories)
> VSFA quality-aware feature aggregation (higher=better) · ↑ higher=better

**[`vsfa`](src/ayase/modules/vsfa.py)** — VSFA quality-aware feature aggregation with GRU (ACMMM 2019)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: huggingface_hub, opencv-python, torch, torchvision
- **VRAM**: ~200 MB
- **Tests**: covered by [`test_vsfa.py`](tests/modules/per_module/test_vsfa.py)
- **Config**: `subsample=8`, `frame_size=520`

### `wadiqam_score` [↑](#categories)
> WaDIQaM-NR (higher=better) · ↑ higher=better

**[`wadiqam`](src/ayase/modules/wadiqam.py)** — WaDIQaM-NR weighted averaging deep image quality mapper

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: opencv-python, pyiqa, torch
- **Tests**: covered by [`test_wadiqam.py`](tests/modules/per_module/test_wadiqam.py), [`test_iqa_research_metrics.py`](tests/modules/test_iqa_research_metrics.py)
- **Config**: `subsample=8`

### `zoomvqa_score` [↑](#categories)
> Zoom-VQA multi-level (higher=better) · ↑ higher=better

**[`zoomvqa`](src/ayase/modules/zoomvqa.py)** — Zoom-VQA dual-branch IQA+VQA late-fusion blind VQA (CVPRW 2023) — ResNet-50 spatial + temporal conv branches

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: gc, torch, torchvision
- **VRAM**: ~200 MB
- **Tests**: covered by [`test_zoomvqa.py`](tests/modules/per_module/test_zoomvqa.py)
- **Config**: `subsample=16`, `n_patches=6`, `patch_size=224`, `iqa_weights_path=`, `vqa_weights_path=`


## Full-Reference Quality (58 metrics)

### `ahiq` [↑](#categories)
> Attention Hybrid IQA (higher=better) · ↑ higher=better

**[`ahiq`](src/ayase/modules/ahiq.py)** — Attention-based Hybrid IQA full-reference (higher=better)

- **Input**: img/vid +ref · **Speed**: ⏱️ medium · GPU
- **Packages**: opencv-python, pyiqa, torch
- **Tests**: covered by [`test_ahiq.py`](tests/modules/per_module/test_ahiq.py), [`test_iqa_research_metrics.py`](tests/modules/test_iqa_research_metrics.py)
- **Config**: `subsample=8`

### `artfid_score` [↑](#categories)
> ArtFID style transfer quality (lower=better) · ↓ lower=better

**[`artfid`](src/ayase/modules/artfid.py)** — ArtFID style transfer quality (FR, 2022, lower=better)

- **Input**: img/vid +ref · **Speed**: ⚡ fast
- **Packages**: art_fid
- **Tests**: covered by [`test_artfid.py`](tests/modules/per_module/test_artfid.py)
- **Config**: `subsample=8`

### `avqt_score` [↑](#categories)
> Apple AVQT perceptual (higher=better) · ↑ higher=better

**[`avqt`](src/ayase/modules/avqt.py)** — Apple AVQT perceptual video quality (full-reference)

- **Input**: img/vid +ref · **Speed**: ⏱️ medium · GPU
- **Packages**: gc, torch, torchvision
- **Tests**: covered by [`test_avqt.py`](tests/modules/per_module/test_avqt.py)
- **Config**: `subsample=8`, `hysteresis_weight=0.1`

### `butteraugli` [↑](#categories)
> Butteraugli perceptual distance (lower=better) · ↓ lower=better

**[`butteraugli`](src/ayase/modules/butteraugli.py)** — Butteraugli perceptual distance (Google/JPEG XL, lower=better)

- **Input**: img/vid +ref · **Speed**: ⚡ fast
- **Backend**: jxlpy → butteraugli → approx
- **Packages**: butteraugli, jxlpy
- **Tests**: covered by [`test_butteraugli.py`](tests/modules/per_module/test_butteraugli.py), [`test_perceptual_metrics.py`](tests/modules/test_perceptual_metrics.py)
- **Config**: `subsample=5`, `warning_threshold=2.0`

### `c3dvqa_score` [↑](#categories)
> C3DVQA 3D CNN spatiotemporal FR · ↑ higher=better

**[`c3dvqa`](src/ayase/modules/c3dvqa.py)** — 3D CNN spatiotemporal video quality assessment

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Packages**: opencv-python, torch, torchvision
- **VRAM**: ~200 MB
- **Tests**: covered by [`test_c3dvqa.py`](tests/modules/per_module/test_c3dvqa.py), [`test_video_native_metrics.py`](tests/modules/test_video_native_metrics.py)
- **Config**: `clip_length=16`, `subsample=4`

### `cgvqm` [↑](#categories)
> CGVQM gaming quality (higher=better) · ↑ higher=better

**[`cgvqm`](src/ayase/modules/cgvqm.py)** — CGVQM gaming/rendering quality metric (Intel, higher=better)

- **Input**: img/vid +ref · **Speed**: ⚡ fast
- **Backend**: cgvqm → approx
- **Packages**: cgvqm
- **Tests**: covered by [`test_cgvqm.py`](tests/modules/per_module/test_cgvqm.py), [`test_industry_metrics.py`](tests/modules/test_industry_metrics.py)
- **Config**: `subsample=5`

### `ciede2000` [↑](#categories)
> CIEDE2000 perceptual color difference (lower=better) · ↓ lower=better

**[`ciede2000`](src/ayase/modules/ciede2000.py)** — CIEDE2000 perceptual color difference (lower=better)

- **Input**: img/vid +ref · **Speed**: ⚡ fast
- **Tests**: covered by [`test_ciede2000.py`](tests/modules/per_module/test_ciede2000.py), [`test_industry_metrics.py`](tests/modules/test_industry_metrics.py)
- **Config**: `subsample=5`

### `ckdn_score` [↑](#categories)
> CKDN knowledge distillation FR · ↑ higher=better

**[`ckdn`](src/ayase/modules/ckdn.py)** — CKDN knowledge distillation FR image quality

- **Input**: img/vid +ref · **Speed**: ⏱️ medium · GPU
- **Packages**: opencv-python, pyiqa, torch
- **Tests**: covered by [`test_ckdn.py`](tests/modules/per_module/test_ckdn.py), [`test_image_iqa_metrics.py`](tests/modules/test_image_iqa_metrics.py)
- **Config**: `subsample=4`

### `compressed_vqa_hdr` [↑](#categories)
> CompressedVQA-HDR (higher=better) · ↑ higher=better

**[`compressed_vqa_hdr`](src/ayase/modules/compressed_vqa_hdr.py)** — CompressedVQA-HDR FR quality (ICME 2025)

- **Input**: img/vid +ref · **Speed**: ⚡ fast
- **Tests**: covered by [`test_compressed_vqa_hdr.py`](tests/modules/per_module/test_compressed_vqa_hdr.py)
- **Config**: `subsample=8`

### `cpp_psnr` [↑](#categories)
> Craster Parabolic PSNR (dB, higher=better) · ↑ higher=better · dB

**[`spherical_psnr`](src/ayase/modules/spherical_psnr.py)** — S-PSNR/WS-PSNR/CPP-PSNR spherical PSNR (MPEG/JVET)

- **Input**: img/vid +ref · **Speed**: ⚡ fast
- **Tests**: covered by [`test_spherical_psnr.py`](tests/modules/per_module/test_spherical_psnr.py)
- **Config**: `subsample=8`

### `cw_ssim` [↑](#categories)
> Complex Wavelet SSIM (0-1, higher=better) · ↑ higher=better · 0-1

**[`cw_ssim`](src/ayase/modules/cw_ssim.py)** — Complex Wavelet SSIM full-reference metric (0-1, higher=better)

- **Input**: img/vid +ref · **Speed**: ⏱️ medium · GPU
- **Packages**: opencv-python, pyiqa, torch
- **Tests**: covered by [`test_cw_ssim.py`](tests/modules/per_module/test_cw_ssim.py), [`test_iqa_research_metrics.py`](tests/modules/test_iqa_research_metrics.py)
- **Config**: `subsample=8`

### `deepvqa_score` [↑](#categories)
> DeepVQA spatiotemporal FR (higher=better) · ↑ higher=better

**[`deepvqa`](src/ayase/modules/deepvqa.py)** — DeepVQA spatiotemporal masking FR-VQA (ECCV 2018)

- **Input**: img/vid +ref · **Speed**: ⏱️ medium · GPU
- **Packages**: gc, torch, torchvision
- **Tests**: covered by [`test_deepvqa.py`](tests/modules/per_module/test_deepvqa.py)
- **Config**: `subsample=8`, `minkowski_p=4.0`

### `deepwsd_score` [↑](#categories)
> DeepWSD Wasserstein distance FR · ↓ lower=better

**[`deepwsd`](src/ayase/modules/deepwsd.py)** — DeepWSD Wasserstein distance FR image quality

- **Input**: img/vid +ref · **Speed**: ⏱️ medium · GPU
- **Packages**: opencv-python, pyiqa, torch
- **Tests**: covered by [`test_deepwsd.py`](tests/modules/per_module/test_deepwsd.py), [`test_image_iqa_metrics.py`](tests/modules/test_image_iqa_metrics.py)
- **Config**: `subsample=4`

### `dists` [↑](#categories)
> DISTS (0-1, lower=more similar) · ↓ lower=better · 0-1, lower=more similar

**[`dists`](src/ayase/modules/dists.py)** — Deep Image Structure and Texture Similarity (full-reference)

- **Input**: img/vid +ref · **Speed**: ⏱️ medium · GPU
- **Packages**: piq, torch
- **Tests**: covered by [`test_dists.py`](tests/modules/per_module/test_dists.py), [`test_fields_general.py`](tests/modules/test_fields_general.py), [`test_ml_basics.py`](tests/modules/test_ml_basics.py)
- **Config**: `subsample=5`, `warning_threshold=0.3`, `device=auto`

### `dmm` [↑](#categories)
> DMM Detail Model Metric FR (higher=better) · ↑ higher=better

**[`dmm`](src/ayase/modules/dmm.py)** — DMM detail model metric full-reference (higher=better)

- **Input**: img/vid +ref · **Speed**: ⏱️ medium · GPU
- **Packages**: opencv-python, pyiqa, torch
- **Tests**: covered by [`test_dmm.py`](tests/modules/per_module/test_dmm.py), [`test_perceptual_metrics.py`](tests/modules/test_perceptual_metrics.py)
- **Config**: `subsample=8`

### `dreamsim` [↑](#categories)
> DreamSim CLIP+DINO similarity (lower=more similar) · ↓ lower=better · lower=more similar

**[`dreamsim`](src/ayase/modules/dreamsim_metric.py)** — DreamSim foundation model perceptual similarity (CLIP+DINO ensemble)

- **Input**: img/vid +ref · **Speed**: ⏱️ medium
- **Packages**: Pillow, dreamsim, opencv-python, torch
- **Tests**: covered by [`test_dreamsim.py`](tests/modules/per_module/test_dreamsim.py), [`test_iqa_research_metrics.py`](tests/modules/test_iqa_research_metrics.py), [`test_docs_integrity.py`](tests/test_docs_integrity.py)
- **Config**: `subsample=8`, `model_type=ensemble`

### `erqa_score` [↑](#categories)
> ERQA edge restoration quality (0-1, higher=better) · ↑ higher=better · 0-1

**[`erqa`](src/ayase/modules/erqa.py)** — ERQA edge restoration quality assessment (FR, 2022)

- **Input**: img/vid +ref · **Speed**: ⚡ fast
- **Packages**: erqa
- **Tests**: covered by [`test_erqa.py`](tests/modules/per_module/test_erqa.py)
- **Config**: `subsample=8`

### `flip_score` [↑](#categories)
> NVIDIA FLIP perceptual metric (0-1, lower=better) · ↓ lower=better · 0-1

**[`flip`](src/ayase/modules/flip_metric.py)** — NVIDIA FLIP perceptual difference (0-1, lower=better)

- **Input**: img/vid +ref · **Speed**: ⏱️ medium
- **Backend**: flip_evaluator → flip_torch → approx
- **Packages**: flip-evaluator, flip_torch, torch
- **Tests**: covered by [`test_flip.py`](tests/modules/per_module/test_flip.py), [`test_perceptual_metrics.py`](tests/modules/test_perceptual_metrics.py), [`test_docs_integrity.py`](tests/test_docs_integrity.py)
- **Config**: `subsample=5`, `warning_threshold=0.3`

### `flolpips` [↑](#categories)
> FloLPIPS flow-based perceptual FR

**[`flolpips`](src/ayase/modules/flolpips.py)** — Flow-compensated perceptual distance (RAFT+LPIPS, Farneback+LPIPS, or MSE fallback)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Backend**: farneback_mse → raft_lpips → farneback_lpips
- **Packages**: lpips, opencv-python, torch, torchvision
- **Tests**: covered by [`test_flolpips.py`](tests/modules/per_module/test_flolpips.py), [`test_video_native_fields.py`](tests/modules/test_video_native_fields.py), [`test_video_native_metrics.py`](tests/modules/test_video_native_metrics.py)
- **Config**: `subsample=8`

### `fsim` [↑](#categories)
> Feature Similarity Index (0-1, higher=better) · ↑ higher=better · 0-1

**[`perceptual_fr`](src/ayase/modules/perceptual_fr.py)** — FSIM + GMSD + VSI full-reference perceptual metrics

- **Input**: img/vid +ref · **Speed**: ⏱️ medium · GPU
- **Packages**: piq, torch
- **Tests**: covered by [`test_perceptual_fr.py`](tests/modules/per_module/test_perceptual_fr.py), [`test_ml_basics.py`](tests/modules/test_ml_basics.py)
- **Config**: `subsample=5`, `device=auto`

### `funque_score` [↑](#categories)
> FUNQUE unified quality (beats VMAF) · ↑ higher=better

**[`funque`](src/ayase/modules/funque.py)** — Fused quality evaluator (FUNQUE package, handcrafted FR, or NR fallback)

- **Input**: img/vid +ref · **Speed**: ⚡ fast
- **Backend**: heuristic_nr → funque → heuristic_fr
- **Packages**: funque, opencv-python
- **Tests**: covered by [`test_funque.py`](tests/modules/per_module/test_funque.py), [`test_video_native_metrics.py`](tests/modules/test_video_native_metrics.py)
- **Config**: `subsample=8`

### `gmsd` [↑](#categories)
> Gradient Magnitude Similarity Deviation (lower=better) · ↓ lower=better

**[`perceptual_fr`](src/ayase/modules/perceptual_fr.py)** — FSIM + GMSD + VSI full-reference perceptual metrics

- **Input**: img/vid +ref · **Speed**: ⏱️ medium · GPU
- **Packages**: piq, torch
- **Tests**: covered by [`test_perceptual_fr.py`](tests/modules/per_module/test_perceptual_fr.py), [`test_ml_basics.py`](tests/modules/test_ml_basics.py)
- **Config**: `subsample=5`, `device=auto`

### `graphsim_score` [↑](#categories)
> GraphSIM gradient (higher=better) · ↑ higher=better

**[`graphsim`](src/ayase/modules/graphsim.py)** — GraphSIM graph gradient point cloud quality (2020)

- **Input**: img/vid +ref · **Speed**: ⚡ fast
- **Packages**: open3d, scipy
- **Tests**: covered by [`test_graphsim.py`](tests/modules/per_module/test_graphsim.py)

### `image_lpips` [↑](#categories)
> LPIPS perceptual distance vs reference (0-1, lower=more similar) · ↓ lower=better

**[`image_lpips`](src/ayase/modules/image_lpips.py)** — LPIPS perceptual distance between image pairs and diversity metric

- **Input**: img/vid +ref · **Speed**: ⏱️ medium · GPU
- **Packages**: lpips, torch
- **Tests**: covered by [`test_image_lpips.py`](tests/modules/per_module/test_image_lpips.py)
- **Config**: `net=alex`, `resize=256`, `diversity_max_pairs=500`

### `mad` [↑](#categories)
> Most Apparent Distortion (lower=better) · ↓ lower=better

**[`mad`](src/ayase/modules/mad_metric.py)** — Most Apparent Distortion full-reference metric (lower=better)

- **Input**: img/vid +ref · **Speed**: ⏱️ medium · GPU
- **Packages**: opencv-python, pyiqa, torch
- **Tests**: covered by [`test_mad.py`](tests/modules/per_module/test_mad.py), [`test_iqa_research_metrics.py`](tests/modules/test_iqa_research_metrics.py), [`test_docs_integrity.py`](tests/test_docs_integrity.py)
- **Config**: `subsample=8`

### `movie_score` [↑](#categories)
> MOVIE motion trajectory FR · ↑ higher=better

**[`movie`](src/ayase/modules/movie.py)** — Video quality via spatiotemporal Gabor decomposition (FR or NR fallback)

- **Input**: img/vid +ref · **Speed**: ⚡ fast
- **Packages**: opencv-python
- **Tests**: covered by [`test_movie.py`](tests/modules/per_module/test_movie.py), [`test_video_native_metrics.py`](tests/modules/test_video_native_metrics.py)
- **Config**: `subsample=8`

### `ms_ssim` [↑](#categories)
> Multi-Scale SSIM (0-1) · 0-1

**[`ms_ssim`](src/ayase/modules/ms_ssim.py)** — Multi-Scale SSIM perceptual similarity metric (full-reference)

- **Input**: vid +ref · **Speed**: ⏱️ medium · GPU
- **Packages**: pytorch_msssim, torch
- **Tests**: covered by [`test_ms_ssim.py`](tests/modules/per_module/test_ms_ssim.py), [`test_fields_general.py`](tests/modules/test_fields_general.py), [`test_reference_and_meta_metrics.py`](tests/modules/test_reference_and_meta_metrics.py)
- **Config**: `scales=5`, `weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333]`, `subsample=1`, `warning_threshold=0.85`, `device=auto`

### `nlpd` [↑](#categories)
> Normalized Laplacian Pyramid Distance (lower=better) · ↓ lower=better

**[`nlpd`](src/ayase/modules/nlpd_metric.py)** — Normalized Laplacian Pyramid Distance full-reference (lower=better)

- **Input**: img/vid +ref · **Speed**: ⏱️ medium · GPU
- **Packages**: opencv-python, pyiqa, torch
- **Tests**: covered by [`test_nlpd.py`](tests/modules/per_module/test_nlpd.py), [`test_iqa_research_metrics.py`](tests/modules/test_iqa_research_metrics.py), [`test_docs_integrity.py`](tests/test_docs_integrity.py)
- **Config**: `subsample=8`

### `pc_d1_psnr` [↑](#categories)
> Point-to-point PSNR (dB) · dB

**[`pc_psnr`](src/ayase/modules/pc_psnr.py)** — D1/D2 MPEG point cloud PSNR

- **Input**: img/vid +ref · **Speed**: ⚡ fast
- **Packages**: open3d, scipy
- **Tests**: covered by [`test_pc_psnr.py`](tests/modules/per_module/test_pc_psnr.py)

### `pc_d2_psnr` [↑](#categories)
> Point-to-plane PSNR (dB) · dB

**[`pc_psnr`](src/ayase/modules/pc_psnr.py)** — D1/D2 MPEG point cloud PSNR

- **Input**: img/vid +ref · **Speed**: ⚡ fast
- **Packages**: open3d, scipy
- **Tests**: covered by [`test_pc_psnr.py`](tests/modules/per_module/test_pc_psnr.py)

### `pcqm_score` [↑](#categories)
> PCQM geometry+color (higher=better) · ↑ higher=better

**[`pcqm`](src/ayase/modules/pcqm.py)** — PCQM geometry+color point cloud quality (2020)

- **Input**: img/vid +ref · **Speed**: ⚡ fast
- **Packages**: open3d, scipy
- **Tests**: covered by [`test_pcqm.py`](tests/modules/per_module/test_pcqm.py)

### `pieapp` [↑](#categories)
> PieAPP pairwise preference (lower=better) · ↓ lower=better

**[`pieapp`](src/ayase/modules/pieapp.py)** — PieAPP full-reference perceptual error via pairwise preference (lower=better)

- **Input**: img/vid +ref · **Speed**: ⏱️ medium · GPU
- **Packages**: opencv-python, pyiqa, torch
- **Tests**: covered by [`test_pieapp.py`](tests/modules/per_module/test_pieapp.py), [`test_iqa_research_metrics.py`](tests/modules/test_iqa_research_metrics.py)
- **Config**: `subsample=8`

### `pointssim_score` [↑](#categories)
> PointSSIM structural (higher=better) · ↑ higher=better

**[`pointssim`](src/ayase/modules/pointssim.py)** — PointSSIM structural similarity for point clouds (2020)

- **Input**: img/vid +ref · **Speed**: ⚡ fast
- **Packages**: open3d, scipy
- **Tests**: covered by [`test_pointssim.py`](tests/modules/per_module/test_pointssim.py)

### `psnr99` [↑](#categories)
> PSNR99 worst-case region quality (dB, higher=better) · ↑ higher=better · dB

**[`psnr99`](src/ayase/modules/psnr99.py)** — PSNR99 worst-case region quality for super-resolution (FR, 2025)

- **Input**: img/vid +ref · **Speed**: ⚡ fast
- **Tests**: covered by [`test_psnr99.py`](tests/modules/per_module/test_psnr99.py)
- **Config**: `subsample=8`, `block_size=32`

### `psnr_div` [↑](#categories)
> PSNR_DIV motion-weighted PSNR (dB, higher=better) · ↑ higher=better · dB

**[`psnr_div`](src/ayase/modules/psnr_div.py)** — PSNR_DIV motion-weighted PSNR for frame interpolation (ICIP 2025, FR)

- **Input**: img/vid +ref · **Speed**: ⚡ fast
- **Tests**: covered by [`test_psnr_div.py`](tests/modules/per_module/test_psnr_div.py)
- **Config**: `subsample=8`, `block_size=16`

### `psnr_hvs` [↑](#categories)
> PSNR-HVS perceptually weighted (dB, higher=better) · ↑ higher=better · dB

**[`psnr_hvs`](src/ayase/modules/psnr_hvs.py)** — PSNR-HVS + PSNR-HVS-M perceptually weighted PSNR (dB, higher=better)

- **Input**: img/vid +ref · **Speed**: ⚡ fast
- **Backend**: dct
- **Tests**: covered by [`test_psnr_hvs.py`](tests/modules/per_module/test_psnr_hvs.py), [`test_industry_metrics.py`](tests/modules/test_industry_metrics.py)
- **Config**: `subsample=5`

### `psnr_hvs_m` [↑](#categories)
> PSNR-HVS-M with masking (dB, higher=better) · ↑ higher=better · dB

**[`psnr_hvs`](src/ayase/modules/psnr_hvs.py)** — PSNR-HVS + PSNR-HVS-M perceptually weighted PSNR (dB, higher=better)

- **Input**: img/vid +ref · **Speed**: ⚡ fast
- **Backend**: dct
- **Tests**: covered by [`test_psnr_hvs.py`](tests/modules/per_module/test_psnr_hvs.py), [`test_industry_metrics.py`](tests/modules/test_industry_metrics.py)
- **Config**: `subsample=5`

### `pvmaf_score` [↑](#categories)
> pVMAF predictive VMAF (0-100) · ↑ higher=better · 0-100

**[`pvmaf`](src/ayase/modules/pvmaf.py)** — Predictive VMAF ~35x faster via bitstream+pixel features (2024, 0-100)

- **Input**: img/vid +ref · **Speed**: ⏱️ medium · GPU
- **Backend**: resnet_vmaf
- **Packages**: Pillow, torch, torchvision
- **VRAM**: ~200 MB
- **Tests**: covered by [`test_pvmaf.py`](tests/modules/per_module/test_pvmaf.py)
- **Config**: `subsample=8`

### `rankdvqa_score` [↑](#categories)
> RankDVQA ranking-based FR (higher=better) · ↑ higher=better

**[`rankdvqa`](src/ayase/modules/rankdvqa.py)** — RankDVQA ranking-based FR VQA (WACV 2024)

- **Input**: img/vid +ref · **Speed**: ⚡ fast
- **Tests**: covered by [`test_rankdvqa.py`](tests/modules/per_module/test_rankdvqa.py)
- **Config**: `subsample=8`

### `s_psnr` [↑](#categories)
> Spherical PSNR (dB, higher=better) · ↑ higher=better · dB

**[`spherical_psnr`](src/ayase/modules/spherical_psnr.py)** — S-PSNR/WS-PSNR/CPP-PSNR spherical PSNR (MPEG/JVET)

- **Input**: img/vid +ref · **Speed**: ⚡ fast
- **Tests**: covered by [`test_spherical_psnr.py`](tests/modules/per_module/test_spherical_psnr.py)
- **Config**: `subsample=8`

### `ssimc` [↑](#categories)
> Complex Wavelet SSIM-C FR (higher=better) · ↑ higher=better

**[`ssimc`](src/ayase/modules/ssimc.py)** — SSIM-C complex wavelet structural similarity FR (higher=better)

- **Input**: img/vid +ref · **Speed**: ⏱️ medium · GPU
- **Packages**: opencv-python, pyiqa, torch
- **Tests**: covered by [`test_ssimc.py`](tests/modules/per_module/test_ssimc.py), [`test_perceptual_metrics.py`](tests/modules/test_perceptual_metrics.py)
- **Config**: `subsample=8`

### `ssimulacra2` [↑](#categories)
> SSIMULACRA 2 (0-100, lower=better, JPEG XL standard) · ↓ lower=better · 0-100, JPEG XL standard

**[`ssimulacra2`](src/ayase/modules/ssimulacra2.py)** — SSIMULACRA 2 perceptual distance (JPEG XL standard, lower=better)

- **Input**: img/vid +ref · **Speed**: ⚡ fast
- **Packages**: ssimulacra2
- **Tests**: covered by [`test_ssimulacra2.py`](tests/modules/per_module/test_ssimulacra2.py), [`test_perceptual_metrics.py`](tests/modules/test_perceptual_metrics.py)
- **Config**: `subsample=5`, `warning_threshold=50.0`

### `st_greed_score` [↑](#categories)
> ST-GREED variable frame rate FR · ↑ higher=better

**[`st_greed`](src/ayase/modules/st_greed.py)** — Spatial-temporal entropic quality (FR entropic difference or NR heuristic fallback)

- **Input**: vid +ref · **Speed**: ⚡ fast
- **Packages**: opencv-python
- **Tests**: covered by [`test_st_greed.py`](tests/modules/per_module/test_st_greed.py), [`test_video_native_metrics.py`](tests/modules/test_video_native_metrics.py)
- **Config**: `subsample=16`

### `st_lpips` [↑](#categories)
> ST-LPIPS spatiotemporal perceptual FR

**[`st_lpips`](src/ayase/modules/st_lpips.py)** — Spatiotemporal perceptual video quality (ST-LPIPS model or LPIPS)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Backend**: stlpips → lpips
- **Packages**: lpips, opencv-python, stlpips-pytorch, torch
- **Tests**: covered by [`test_st_lpips.py`](tests/modules/per_module/test_st_lpips.py), [`test_video_native_fields.py`](tests/modules/test_video_native_fields.py), [`test_video_native_metrics.py`](tests/modules/test_video_native_metrics.py)
- **Config**: `subsample=8`

### `st_mad` [↑](#categories)
> ST-MAD spatiotemporal MAD (lower=better) · ↓ lower=better

**[`st_mad`](src/ayase/modules/st_mad.py)** — ST-MAD spatiotemporal MAD (TIP 2012)

- **Input**: img/vid +ref · **Speed**: ⚡ fast
- **Tests**: covered by [`test_st_mad.py`](tests/modules/per_module/test_st_mad.py)
- **Config**: `subsample=8`

### `strred` [↑](#categories)
> STRRED reduced-reference temporal (lower=better) · ↓ lower=better

**[`strred`](src/ayase/modules/strred.py)** — STRRED reduced-reference temporal quality (ITU, lower=better)

- **Input**: img/vid +ref · **Speed**: ⚡ fast
- **Backend**: skvideo → approx
- **Packages**: scikit-video
- **Tests**: covered by [`test_strred.py`](tests/modules/per_module/test_strred.py), [`test_industry_metrics.py`](tests/modules/test_industry_metrics.py)
- **Config**: `subsample=3`

### `topiq_fr` [↑](#categories)
> TOPIQ full-reference (higher=better) · ↑ higher=better

**[`topiq_fr`](src/ayase/modules/topiq_fr.py)** — TOPIQ full-reference top-down semantics-to-distortion IQA (higher=better)

- **Input**: img/vid +ref · **Speed**: ⏱️ medium · GPU
- **Packages**: opencv-python, pyiqa, torch
- **Tests**: covered by [`test_topiq_fr.py`](tests/modules/per_module/test_topiq_fr.py), [`test_iqa_research_metrics.py`](tests/modules/test_iqa_research_metrics.py)
- **Config**: `subsample=8`

### `vfips_score` [↑](#categories)
> VFIPS frame interpolation perceptual (lower=better) · ↓ lower=better

**[`vfips`](src/ayase/modules/vfips.py)** — VFIPS frame interpolation perceptual similarity (ECCV 2022, FR)

- **Input**: img/vid +ref · **Speed**: ⚡ fast
- **Tests**: covered by [`test_vfips.py`](tests/modules/per_module/test_vfips.py)
- **Config**: `subsample=8`

### `vif` [↑](#categories)
> Visual Information Fidelity

**[`vif`](src/ayase/modules/vif.py)** — Visual Information Fidelity metric (full-reference)

- **Input**: img/vid +ref · **Speed**: ⏱️ medium · GPU
- **Packages**: piq, torch
- **Tests**: covered by [`test_vif.py`](tests/modules/per_module/test_vif.py), [`test_fields_general.py`](tests/modules/test_fields_general.py), [`test_reference_and_meta_metrics.py`](tests/modules/test_reference_and_meta_metrics.py)
- **Config**: `subsample=1`, `warning_threshold=0.3`, `device=auto`

### `vmaf` [↑](#categories)
> VMAF (0-100, higher=better) · ↑ higher=better · 0-100

**[`vmaf`](src/ayase/modules/vmaf.py)** — VMAF perceptual video quality metric (full-reference)

- **Input**: vid +ref · **Speed**: ⚡ fast
- **Packages**: vmaf
- **Tests**: covered by [`test_vmaf.py`](tests/modules/per_module/test_vmaf.py), [`test_fields_general.py`](tests/modules/test_fields_general.py), [`test_reference_and_meta_metrics.py`](tests/modules/test_reference_and_meta_metrics.py)
- **Config**: `vmaf_model=vmaf_v0.6.1`, `subsample=1`, `use_ffmpeg=True`, `warning_threshold=70.0`

### `vmaf_4k` [↑](#categories)
> VMAF 4K model (0-100, higher=better) · ↑ higher=better · 0-100

**[`vmaf_4k`](src/ayase/modules/vmaf_4k.py)** — VMAF 4K model for UHD content (0-100, higher=better)

- **Input**: vid +ref · **Speed**: ⚡ fast
- **Tests**: covered by [`test_vmaf_4k.py`](tests/modules/per_module/test_vmaf_4k.py), [`test_industry_metrics.py`](tests/modules/test_industry_metrics.py)

### `vmaf_neg` [↑](#categories)
> VMAF NEG (no enhancement gain, 0-100, higher=better) · ↑ higher=better · no enhancement gain, 0-100

**[`vmaf_neg`](src/ayase/modules/vmaf_neg.py)** — VMAF NEG no-enhancement-gain variant (0-100, higher=better)

- **Input**: vid +ref · **Speed**: ⚡ fast
- **Tests**: covered by [`test_vmaf_neg.py`](tests/modules/per_module/test_vmaf_neg.py), [`test_perceptual_metrics.py`](tests/modules/test_perceptual_metrics.py)
- **Config**: `subsample=1`, `warning_threshold=70.0`

### `vmaf_phone` [↑](#categories)
> VMAF phone model (0-100, higher=better) · ↑ higher=better · 0-100

**[`vmaf_phone`](src/ayase/modules/vmaf_phone.py)** — VMAF phone model for mobile viewing (0-100, higher=better)

- **Input**: vid +ref · **Speed**: ⚡ fast
- **Tests**: covered by [`test_vmaf_phone.py`](tests/modules/per_module/test_vmaf_phone.py), [`test_industry_metrics.py`](tests/modules/test_industry_metrics.py)

### `vsi_score` [↑](#categories)
> Visual Saliency Index (0-1, higher=better) · ↑ higher=better · 0-1

**[`perceptual_fr`](src/ayase/modules/perceptual_fr.py)** — FSIM + GMSD + VSI full-reference perceptual metrics

- **Input**: img/vid +ref · **Speed**: ⏱️ medium · GPU
- **Packages**: piq, torch
- **Tests**: covered by [`test_perceptual_fr.py`](tests/modules/per_module/test_perceptual_fr.py), [`test_ml_basics.py`](tests/modules/test_ml_basics.py)
- **Config**: `subsample=5`, `device=auto`

### `wadiqam_fr` [↑](#categories)
> WaDIQaM full-reference (higher=better) · ↑ higher=better

**[`wadiqam_fr`](src/ayase/modules/wadiqam_fr.py)** — WaDIQaM full-reference deep quality metric (higher=better)

- **Input**: img/vid +ref · **Speed**: ⏱️ medium · GPU
- **Packages**: opencv-python, pyiqa, torch
- **Tests**: covered by [`test_wadiqam_fr.py`](tests/modules/per_module/test_wadiqam_fr.py), [`test_perceptual_metrics.py`](tests/modules/test_perceptual_metrics.py)
- **Config**: `subsample=8`

### `ws_psnr` [↑](#categories)
> Weighted Spherical PSNR (dB, higher=better) · ↑ higher=better · dB

**[`spherical_psnr`](src/ayase/modules/spherical_psnr.py)** — S-PSNR/WS-PSNR/CPP-PSNR spherical PSNR (MPEG/JVET)

- **Input**: img/vid +ref · **Speed**: ⚡ fast
- **Tests**: covered by [`test_spherical_psnr.py`](tests/modules/per_module/test_spherical_psnr.py)
- **Config**: `subsample=8`

### `ws_ssim` [↑](#categories)
> Weighted Spherical SSIM (0-1, higher=better) · ↑ higher=better · 0-1

**[`ws_ssim`](src/ayase/modules/ws_ssim.py)** — WS-SSIM weighted spherical SSIM

- **Input**: img/vid +ref · **Speed**: ⚡ fast
- **Tests**: covered by [`test_ws_ssim.py`](tests/modules/per_module/test_ws_ssim.py)
- **Config**: `subsample=8`

### `xpsnr` [↑](#categories)
> XPSNR perceptual PSNR (dB, higher=better) · ↑ higher=better · dB

**[`xpsnr`](src/ayase/modules/xpsnr.py)** — XPSNR perceptually weighted PSNR (Fraunhofer, dB, higher=better)

- **Input**: img/vid +ref · **Speed**: ⚡ fast
- **Tests**: covered by [`test_xpsnr.py`](tests/modules/per_module/test_xpsnr.py), [`test_industry_metrics.py`](tests/modules/test_industry_metrics.py)


## Text-Video Alignment (31 metrics)

### `aigcvqa_alignment` [↑](#categories)
> AIGC-VQA text-video alignment

**[`aigcvqa`](src/ayase/modules/aigcvqa.py)** — AIGC-VQA holistic 3-branch AIGC perception (CVPRW 2024)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: Pillow, torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_aigcvqa.py`](tests/modules/per_module/test_aigcvqa.py)
- **Config**: `subsample=8`, `clip_model=openai/clip-vit-base-patch32`

### `aigv_alignment` [↑](#categories)
> AI video text-video alignment

**[`aigv_assessor`](src/ayase/modules/aigv_assessor.py)** — AI-generated video quality (AIGV-Assessor model or CLIP proxy)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Backend**: aigv_assessor → clip
- **Packages**: Pillow, opencv-python, torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/IntMeGroup/AIGV-Assessor-static_quality" target="_blank">HF</a>
- **Tests**: covered by [`test_aigv_assessor.py`](tests/modules/per_module/test_aigv_assessor.py), [`test_motion_scene_semantic_metrics.py`](tests/modules/test_motion_scene_semantic_metrics.py)
- **Config**: `subsample=8`, `trust_remote_code=True`

### `blip_bleu` [↑](#categories)

**[`captioning`](src/ayase/modules/captioning.py)** — Generates captions using BLIP + computes BLEU score (EvalCrafter blip_bleu)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: Pillow, opencv-python, torch, transformers
- **Source**: <a href="https://huggingface.co/Salesforce/blip-image-captioning-base" target="_blank">HF</a>
- **Tests**: covered by [`test_captioning.py`](tests/modules/per_module/test_captioning.py)
- **Config**: `model_name=Salesforce/blip-image-captioning-base`, `num_frames=5`

### `clip_score` [↑](#categories)
> Caption-image alignment · ↑ higher=better

**[`semantic_alignment`](src/ayase/modules/semantic_alignment.py)** — Checks alignment between video and caption (CLIP Score)

- **Input**: vid +cap · **Speed**: ⏱️ medium · GPU
- **Packages**: torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_semantic_alignment.py`](tests/modules/per_module/test_semantic_alignment.py)
- **Config**: `model_name=openai/clip-vit-base-patch32`, `max_frames=32`, `warning_threshold=0.2`

### `compbench_action` [↑](#categories)
> Action binding (0-1) · 0-1

**[`t2v_compbench`](src/ayase/modules/t2v_compbench.py)** — T2V-CompBench compositional metrics (YOLO+Depth+CLIP / CLIP)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Backend**: yolo_depth → clip
- **Packages**: Pillow, torch, transformers, ultralytics
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_t2v_compbench.py`](tests/modules/per_module/test_t2v_compbench.py), [`test_vbench2_compbench.py`](tests/modules/test_vbench2_compbench.py)
- **Config**: `subsample=8`, `enable_attribute=True`, `enable_object_rel=True`, `enable_action=True`, `enable_spatial=True`, `enable_numeracy=True`, `enable_scene=True`, `weights=[1, 1, 1, 1, 1, 1]`

### `compbench_attribute` [↑](#categories)
> Attribute binding (0-1) · 0-1

**[`t2v_compbench`](src/ayase/modules/t2v_compbench.py)** — T2V-CompBench compositional metrics (YOLO+Depth+CLIP / CLIP)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Backend**: yolo_depth → clip
- **Packages**: Pillow, torch, transformers, ultralytics
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_t2v_compbench.py`](tests/modules/per_module/test_t2v_compbench.py), [`test_vbench2_compbench.py`](tests/modules/test_vbench2_compbench.py)
- **Config**: `subsample=8`, `enable_attribute=True`, `enable_object_rel=True`, `enable_action=True`, `enable_spatial=True`, `enable_numeracy=True`, `enable_scene=True`, `weights=[1, 1, 1, 1, 1, 1]`

### `compbench_numeracy` [↑](#categories)
> Generative numeracy (0-1) · 0-1

**[`t2v_compbench`](src/ayase/modules/t2v_compbench.py)** — T2V-CompBench compositional metrics (YOLO+Depth+CLIP / CLIP)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Backend**: yolo_depth → clip
- **Packages**: Pillow, torch, transformers, ultralytics
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_t2v_compbench.py`](tests/modules/per_module/test_t2v_compbench.py), [`test_vbench2_compbench.py`](tests/modules/test_vbench2_compbench.py)
- **Config**: `subsample=8`, `enable_attribute=True`, `enable_object_rel=True`, `enable_action=True`, `enable_spatial=True`, `enable_numeracy=True`, `enable_scene=True`, `weights=[1, 1, 1, 1, 1, 1]`

### `compbench_object_rel` [↑](#categories)
> Object relationship (0-1) · 0-1

**[`t2v_compbench`](src/ayase/modules/t2v_compbench.py)** — T2V-CompBench compositional metrics (YOLO+Depth+CLIP / CLIP)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Backend**: yolo_depth → clip
- **Packages**: Pillow, torch, transformers, ultralytics
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_t2v_compbench.py`](tests/modules/per_module/test_t2v_compbench.py), [`test_vbench2_compbench.py`](tests/modules/test_vbench2_compbench.py)
- **Config**: `subsample=8`, `enable_attribute=True`, `enable_object_rel=True`, `enable_action=True`, `enable_spatial=True`, `enable_numeracy=True`, `enable_scene=True`, `weights=[1, 1, 1, 1, 1, 1]`

### `compbench_overall` [↑](#categories)
> Overall composition (0-1) · 0-1

**[`t2v_compbench`](src/ayase/modules/t2v_compbench.py)** — T2V-CompBench compositional metrics (YOLO+Depth+CLIP / CLIP)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Backend**: yolo_depth → clip
- **Packages**: Pillow, torch, transformers, ultralytics
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_t2v_compbench.py`](tests/modules/per_module/test_t2v_compbench.py), [`test_vbench2_compbench.py`](tests/modules/test_vbench2_compbench.py)
- **Config**: `subsample=8`, `enable_attribute=True`, `enable_object_rel=True`, `enable_action=True`, `enable_spatial=True`, `enable_numeracy=True`, `enable_scene=True`, `weights=[1, 1, 1, 1, 1, 1]`

### `compbench_scene` [↑](#categories)
> Scene composition (0-1) · 0-1

**[`t2v_compbench`](src/ayase/modules/t2v_compbench.py)** — T2V-CompBench compositional metrics (YOLO+Depth+CLIP / CLIP)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Backend**: yolo_depth → clip
- **Packages**: Pillow, torch, transformers, ultralytics
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_t2v_compbench.py`](tests/modules/per_module/test_t2v_compbench.py), [`test_vbench2_compbench.py`](tests/modules/test_vbench2_compbench.py)
- **Config**: `subsample=8`, `enable_attribute=True`, `enable_object_rel=True`, `enable_action=True`, `enable_spatial=True`, `enable_numeracy=True`, `enable_scene=True`, `weights=[1, 1, 1, 1, 1, 1]`

### `compbench_spatial` [↑](#categories)
> Spatial relationship (0-1) · 0-1

**[`t2v_compbench`](src/ayase/modules/t2v_compbench.py)** — T2V-CompBench compositional metrics (YOLO+Depth+CLIP / CLIP)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Backend**: yolo_depth → clip
- **Packages**: Pillow, torch, transformers, ultralytics
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_t2v_compbench.py`](tests/modules/per_module/test_t2v_compbench.py), [`test_vbench2_compbench.py`](tests/modules/test_vbench2_compbench.py)
- **Config**: `subsample=8`, `enable_attribute=True`, `enable_object_rel=True`, `enable_action=True`, `enable_spatial=True`, `enable_numeracy=True`, `enable_scene=True`, `weights=[1, 1, 1, 1, 1, 1]`

### `dsg_score` [↑](#categories)
> DSG Davidsonian Scene Graph (higher=better) · ↑ higher=better

**[`dsg`](src/ayase/modules/dsg.py)** — DSG Davidsonian Scene Graph faithfulness (ICLR 2024, Google)

- **Input**: img/vid +cap · **Speed**: ⚡ fast
- **Packages**: dsg
- **Tests**: covered by [`test_dsg.py`](tests/modules/per_module/test_dsg.py)
- **Config**: `subsample=4`

### `hpsv3_score` [↑](#categories)
> HPSv3 human preference reward mu (higher=better) · ↑ higher=better

**[`hpsv3`](src/ayase/modules/hpsv3.py)** — HPSv3 wide-spectrum human preference scoring (frame-averaged on video)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Backend**: hpsv3
- **Packages**: huggingface_hub, safetensors, torch, transformers
- **VRAM**: ~16 GB
- **Source**: <a href="https://huggingface.co/MizzenAI/HPSv3" target="_blank">HF</a>
- **Tests**: covered by [`test_hpsv3.py`](tests/modules/per_module/test_hpsv3.py)
- **Config**: `num_frames=5`, `device=auto`

### `image_reward_score` [↑](#categories)
> Human preference reward (-2..+2, higher=better) · ↑ higher=better · -2..+2

**[`image_reward`](src/ayase/modules/image_reward.py)** — Human preference prediction for text-to-image quality (ImageReward)

- **Input**: vid +cap · **Speed**: ⏱️ medium
- **Packages**: ImageReward, transformers
- **Tests**: covered by [`test_image_reward.py`](tests/modules/per_module/test_image_reward.py)
- **Config**: `model_name=ImageReward-v1.0`, `num_frames=5`, `warning_threshold=0.0`

### `pickscore_score` [↑](#categories)
> PickScore prompt-image preference score (higher=better) · ↑ higher=better

**[`pickscore`](src/ayase/modules/pickscore.py)** — PickScore prompt-conditioned human preference scoring (frame-averaged on video)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Backend**: pickscore
- **Packages**: torch, transformers
- **VRAM**: ~2.5 GB
- **Source**: <a href="https://huggingface.co/yuvalkirstain/PickScore_v1" target="_blank">HF</a>
- **Tests**: covered by [`test_pickscore.py`](tests/modules/per_module/test_pickscore.py)
- **Config**: `model_name=yuvalkirstain/PickScore_v1`, `processor_name=laion/CLIP-ViT-H-14-laion2B-s32B-b79K`, `num_frames=5`, `device=auto`

### `sd_score` [↑](#categories)
> SD-reference similarity (0-1) · ↑ higher=better · 0-1

**[`sd_reference`](src/ayase/modules/sd_reference.py)** — SD Score — CLIP similarity between video frames and SDXL-generated reference images

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: Pillow, diffusers, torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_sd_reference.py`](tests/modules/per_module/test_sd_reference.py)
- **Config**: `clip_model=openai/clip-vit-base-patch32`, `sdxl_model=stabilityai/stable-diffusion-xl-base-1.0`, `num_sd_images=5`, `num_video_frames=8`, `sd_steps=20`, `cache_dir=.ayase_sd_cache`

### `t2v_alignment` [↑](#categories)
> Text-video semantic alignment

**[`t2v_score`](src/ayase/modules/t2v_score.py)** — Text-to-Video alignment and quality scoring

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Packages**: torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_t2v_score.py`](tests/modules/per_module/test_t2v_score.py), [`test_fields_general.py`](tests/modules/test_fields_general.py), [`test_reference_and_meta_metrics.py`](tests/modules/test_reference_and_meta_metrics.py)
- **Config**: `model_name=openai/clip-vit-base-patch32`, `use_clip_fallback=True`, `num_frames=8`, `alignment_weight=0.5`, `quality_weight=0.5`, `device=auto`, `warning_threshold=0.6`

### `t2v_score` [↑](#categories)
> T2VScore alignment + quality · ↑ higher=better

**[`t2v_score`](src/ayase/modules/t2v_score.py)** — Text-to-Video alignment and quality scoring

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Packages**: torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_t2v_score.py`](tests/modules/per_module/test_t2v_score.py), [`test_fields_general.py`](tests/modules/test_fields_general.py), [`test_reference_and_meta_metrics.py`](tests/modules/test_reference_and_meta_metrics.py)
- **Config**: `model_name=openai/clip-vit-base-patch32`, `use_clip_fallback=True`, `num_frames=8`, `alignment_weight=0.5`, `quality_weight=0.5`, `device=auto`, `warning_threshold=0.6`

### `t2veval_score` [↑](#categories)
> T2VEval consistency+realness (higher=better) · ↑ higher=better

**[`t2veval`](src/ayase/modules/t2veval.py)** — T2VEval text-video consistency+realness (2025)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: Pillow, torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_t2veval.py`](tests/modules/per_module/test_t2veval.py)
- **Config**: `subsample=8`, `clip_model=openai/clip-vit-base-patch32`, `alignment_weight=0.35`, `realness_weight=0.35`, `quality_weight=0.3`

### `tifa_score` [↑](#categories)
> VQA faithfulness (0-1, higher=better) · ↑ higher=better · 0-1

**[`tifa`](src/ayase/modules/tifa.py)** — TIFA text-to-image faithfulness via VQA question answering (ICCV 2023)

- **Input**: img/vid +cap · **Speed**: ⏱️ medium · GPU
- **Backend**: vilt → clip
- **Packages**: Pillow, torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/dandelin/vilt-b32-finetuned-vqa" target="_blank">HF</a>
- **Tests**: covered by [`test_tifa.py`](tests/modules/per_module/test_tifa.py), [`test_tifa.py`](tests/modules/test_tifa.py)
- **Config**: `vqa_model=dandelin/vilt-b32-finetuned-vqa`, `num_questions=8`, `subsample=4`

### `umtscore` [↑](#categories)
> UMTScore video-text alignment · ↑ higher=better

**[`umtscore`](src/ayase/modules/umtscore.py)** — UMTScore video-text alignment via UMT features

- **Input**: img/vid +cap · **Speed**: ⏱️ medium
- **Backend**: native → clip
- **Packages**: Pillow, torch, transformers, umt
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_umtscore.py`](tests/modules/per_module/test_umtscore.py)
- **Config**: `subsample=8`

### `video_reward_score` [↑](#categories)
> Human preference reward · ↑ higher=better

**[`video_reward`](src/ayase/modules/video_reward.py)** — VideoAlign human preference reward model (NeurIPS 2025)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: Pillow, opencv-python, torch, transformers
- **Source**: <a href="https://huggingface.co/KlingTeam/VideoReward" target="_blank">HF</a>
- **Tests**: covered by [`test_video_reward.py`](tests/modules/per_module/test_video_reward.py), [`test_motion_scene_semantic_metrics.py`](tests/modules/test_motion_scene_semantic_metrics.py)
- **Config**: `model_name=KlingTeam/VideoReward`, `subsample=8`, `trust_remote_code=True`

### `video_text_score` [↑](#categories)
> Video-text alignment via X-CLIP/CLIP (0-1) · ↑ higher=better · 0-1

**[`video_text_matching`](src/ayase/modules/video_text_matching.py)** — ViCLIP / X-CLIP (Temporal alignment) or Frame-averaged CLIP

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: Pillow, torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_video_text_matching.py`](tests/modules/per_module/test_video_text_matching.py)
- **Config**: `use_xclip=False`, `model_name=openai/clip-vit-base-patch32`, `xclip_model_name=microsoft/xclip-base-patch32`, `min_score_threshold=0.2`, `consistency_std_threshold=0.1`

### `videoreward_ta` [↑](#categories)
> VideoReward text alignment

**[`videoreward`](src/ayase/modules/videoreward.py)** — VideoReward Kling multi-dim reward model (NeurIPS 2025)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Packages**: Pillow, torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_videoreward.py`](tests/modules/per_module/test_videoreward.py)
- **Config**: `subsample=8`, `clip_model=openai/clip-vit-base-patch32`

### `videoscore2_alignment` [↑](#categories)
> VideoScore2 text-video alignment · ↑ higher=better · 0-10

**[`videoscore2`](src/ayase/modules/videoscore2.py)** — VideoScore2 3-dimensional generative video evaluation

- **Input**: img/vid · **Speed**: 🐌 slow · GPU
- **Backend**: transformers
- **Packages**: qwen-vl-utils, torch, transformers
- **VRAM**: ~16 GB
- **Source**: <a href="https://huggingface.co/TIGER-Lab/VideoScore2" target="_blank">HF</a>
- **Tests**: covered by [`test_videoscore2.py`](tests/modules/per_module/test_videoscore2.py), [`test_iqa_research_metrics.py`](tests/modules/test_iqa_research_metrics.py)
- **Config**: `model_name=TIGER-Lab/VideoScore2`, `infer_fps=2.0`, `max_new_tokens=1024`, `temperature=0.7`, `do_sample=True`, `trust_remote_code=True`

### `videoscore2_physical` [↑](#categories)
> VideoScore2 physical/common-sense consistency · ↑ higher=better · 0-10

**[`videoscore2`](src/ayase/modules/videoscore2.py)** — VideoScore2 3-dimensional generative video evaluation

- **Input**: img/vid · **Speed**: 🐌 slow · GPU
- **Backend**: transformers
- **Packages**: qwen-vl-utils, torch, transformers
- **VRAM**: ~16 GB
- **Source**: <a href="https://huggingface.co/TIGER-Lab/VideoScore2" target="_blank">HF</a>
- **Tests**: covered by [`test_videoscore2.py`](tests/modules/per_module/test_videoscore2.py), [`test_iqa_research_metrics.py`](tests/modules/test_iqa_research_metrics.py)
- **Config**: `model_name=TIGER-Lab/VideoScore2`, `infer_fps=2.0`, `max_new_tokens=1024`, `temperature=0.7`, `do_sample=True`, `trust_remote_code=True`

### `videoscore_alignment` [↑](#categories)
> VideoScore text-video alignment · ↑ higher=better

**[`videoscore`](src/ayase/modules/videoscore.py)** — VideoScore 5-dimensional video quality assessment (1-4 scale)

- **Input**: img/vid · **Speed**: 🐌 slow · GPU
- **Packages**: Pillow, opencv-python, torch, transformers
- **Source**: <a href="https://huggingface.co/TIGER-Lab/VideoScore" target="_blank">HF</a>
- **Tests**: covered by [`test_videoscore.py`](tests/modules/per_module/test_videoscore.py), [`test_iqa_research_metrics.py`](tests/modules/test_iqa_research_metrics.py)
- **Config**: `model_name=TIGER-Lab/VideoScore`, `num_frames=8`, `trust_remote_code=True`

### `videoscore_factual` [↑](#categories)
> VideoScore factual consistency · ↑ higher=better

**[`videoscore`](src/ayase/modules/videoscore.py)** — VideoScore 5-dimensional video quality assessment (1-4 scale)

- **Input**: img/vid · **Speed**: 🐌 slow · GPU
- **Packages**: Pillow, opencv-python, torch, transformers
- **Source**: <a href="https://huggingface.co/TIGER-Lab/VideoScore" target="_blank">HF</a>
- **Tests**: covered by [`test_videoscore.py`](tests/modules/per_module/test_videoscore.py), [`test_iqa_research_metrics.py`](tests/modules/test_iqa_research_metrics.py)
- **Config**: `model_name=TIGER-Lab/VideoScore`, `num_frames=8`, `trust_remote_code=True`

### `vqa_a_score` [↑](#categories)
> ↑ higher=better

**[`aesthetic`](src/ayase/modules/aesthetic.py)** — Estimates aesthetic quality using Aesthetic Predictor V2.5

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: aesthetic_predictor_v2_5, torch
- **Tests**: covered by [`test_aesthetic.py`](tests/modules/per_module/test_aesthetic.py), [`test_vbench2_compbench.py`](tests/modules/test_vbench2_compbench.py)
- **Config**: `num_frames=5`, `trust_remote_code=True`

### `vqa_score_alignment` [↑](#categories)
> ↑ higher=better · 0-1

**[`vqa_score`](src/ayase/modules/vqa_score.py)** — VQAScore text-visual alignment via VQA probability (0-1, higher=better)

- **Input**: img/vid +cap · **Speed**: ⏱️ medium · GPU
- **Packages**: Pillow, clip (openai), opencv-python, torch
- **VRAM**: ~600 MB
- **Tests**: covered by [`test_vqa_score.py`](tests/modules/per_module/test_vqa_score.py), [`test_iqa_research_metrics.py`](tests/modules/test_iqa_research_metrics.py)
- **Config**: `model=clip-flant5-xxl`, `subsample=4`

### `vqa_t_score` [↑](#categories)
> ↑ higher=better

**[`basic_quality`](src/ayase/modules/basic.py)** — Comprehensive technical quality assessment (blur, noise, artifacts, contrast)

- **Input**: img/vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_basic_quality.py`](tests/modules/per_module/test_basic_quality.py), [`test_docs_integrity.py`](tests/test_docs_integrity.py), [`test_profiles.py`](tests/test_profiles.py), +3 more
- **Config**: `threshold=40.0`, `blur_threshold=100.0`, `noise_threshold=50.0`


## Temporal Consistency (24 metrics)

### `aigv_temporal` [↑](#categories)
> AI video temporal smoothness

**[`aigv_assessor`](src/ayase/modules/aigv_assessor.py)** — AI-generated video quality (AIGV-Assessor model or CLIP proxy)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Backend**: aigv_assessor → clip
- **Packages**: Pillow, opencv-python, torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/IntMeGroup/AIGV-Assessor-static_quality" target="_blank">HF</a>
- **Tests**: covered by [`test_aigv_assessor.py`](tests/modules/per_module/test_aigv_assessor.py), [`test_motion_scene_semantic_metrics.py`](tests/modules/test_motion_scene_semantic_metrics.py)
- **Config**: `subsample=8`, `trust_remote_code=True`

### `background_consistency` [↑](#categories)
> ↑ higher=better

**[`background_consistency`](src/ayase/modules/background_consistency.py)** — Background consistency using CLIP (all pairwise frame similarity)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Packages**: torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_background_consistency.py`](tests/modules/per_module/test_background_consistency.py), [`test_integration_synthetic.py`](tests/test_integration_synthetic.py)
- **Config**: `model_name=openai/clip-vit-base-patch32`, `max_frames=16`, `warning_threshold=0.5`

### `cdc_score` [↑](#categories)
> CDC color distribution consistency (lower=better) · ↓ lower=better

**[`cdc`](src/ayase/modules/cdc.py)** — CDC color distribution consistency for video colorization (2024)

- **Input**: vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_cdc.py`](tests/modules/per_module/test_cdc.py)
- **Config**: `subsample=16`, `hist_bins=32`

### `chronomagic_ch_score` [↑](#categories)
> Chrono-hallucination (0-1, lower=fewer) · ↓ lower=better · 0-1, lower=fewer

**[`chronomagic`](src/ayase/modules/chronomagic.py)** — ChronoMagic-Bench MTScore + CHScore (CLIP)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Packages**: Pillow, torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_chronomagic.py`](tests/modules/per_module/test_chronomagic.py), [`test_vbench2_compbench.py`](tests/modules/test_vbench2_compbench.py)
- **Config**: `subsample=16`, `hallucination_threshold=2.0`

### `chronomagic_mt_score` [↑](#categories)
> Metamorphic temporal (0-1, higher=better) · ↑ higher=better · 0-1

**[`chronomagic`](src/ayase/modules/chronomagic.py)** — ChronoMagic-Bench MTScore + CHScore (CLIP)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Packages**: Pillow, torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_chronomagic.py`](tests/modules/per_module/test_chronomagic.py), [`test_vbench2_compbench.py`](tests/modules/test_vbench2_compbench.py)
- **Config**: `subsample=16`, `hallucination_threshold=2.0`

### `clip_temp` [↑](#categories)

**[`clip_temporal`](src/ayase/modules/clip_temporal.py)** — CLIP temporal consistency + face/identity consistency (EvalCrafter clip_temp & face_consistency)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Packages**: Pillow, torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_clip_temporal.py`](tests/modules/per_module/test_clip_temporal.py)
- **Config**: `model_name=openai/clip-vit-base-patch32`, `max_frames=32`, `temp_threshold=0.9`, `face_threshold=0.85`

### `davis_f` [↑](#categories)
> DAVIS F boundary accuracy (higher=better) · ↑ higher=better

**[`davis_jf`](src/ayase/modules/davis_jf.py)** — DAVIS J&F video segmentation quality (FR, 2016)

- **Input**: img/vid +ref · **Speed**: ⚡ fast
- **Packages**: opencv-python
- **Tests**: covered by [`test_davis_jf.py`](tests/modules/per_module/test_davis_jf.py)
- **Config**: `subsample=8`, `boundary_threshold=2`

### `davis_j` [↑](#categories)
> DAVIS J region similarity IoU (higher=better) · ↑ higher=better

**[`davis_jf`](src/ayase/modules/davis_jf.py)** — DAVIS J&F video segmentation quality (FR, 2016)

- **Input**: img/vid +ref · **Speed**: ⚡ fast
- **Packages**: opencv-python
- **Tests**: covered by [`test_davis_jf.py`](tests/modules/per_module/test_davis_jf.py)
- **Config**: `subsample=8`, `boundary_threshold=2`

### `depth_temporal_consistency` [↑](#categories)
> Depth map correlation 0-1 (higher=better) · ↑ higher=better

**[`depth_consistency`](src/ayase/modules/depth_consistency.py)** — Monocular depth temporal consistency

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Packages**: torch
- **Source**: <a href="https://huggingface.co/intel-isl/MiDaS" target="_blank">HF</a>
- **Tests**: covered by [`test_depth_consistency.py`](tests/modules/per_module/test_depth_consistency.py), [`test_depth_and_multiview.py`](tests/modules/test_depth_and_multiview.py)
- **Config**: `model_type=MiDaS_small`, `device=auto`, `subsample=3`, `max_frames=200`, `warning_threshold=0.7`

### `flicker_score` [↑](#categories)
> Flicker severity 0-100 (lower=better) · ↓ lower=better

**[`flicker_detection`](src/ayase/modules/flicker_detection.py)** — Detects temporal luminance flicker

- **Input**: vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_flicker_detection.py`](tests/modules/per_module/test_flicker_detection.py), [`test_opencv_modules.py`](tests/modules/test_opencv_modules.py)
- **Config**: `max_frames=600`, `warning_threshold=30.0`

### `flow_coherence` [↑](#categories)
> Bidirectional optical flow consistency (0-1) · 0-1

**[`flow_coherence`](src/ayase/modules/flow_coherence.py)** — Bidirectional optical flow consistency (0-1, higher=coherent)

- **Input**: vid · **Speed**: ⚡ fast
- **Packages**: opencv-python
- **Tests**: covered by [`test_flow_coherence.py`](tests/modules/per_module/test_flow_coherence.py), [`test_curation_metrics.py`](tests/modules/test_curation_metrics.py), [`test_video_native_fields.py`](tests/modules/test_video_native_fields.py)
- **Config**: `subsample=8`

### `judder_score` [↑](#categories)
> Judder severity 0-100 (lower=better) · ↓ lower=better

**[`judder_stutter`](src/ayase/modules/judder_stutter.py)** — Detects judder (uneven cadence) and stutter (duplicate frames)

- **Input**: vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_judder_stutter.py`](tests/modules/per_module/test_judder_stutter.py), [`test_opencv_modules.py`](tests/modules/test_opencv_modules.py)
- **Config**: `max_frames=600`, `duplicate_threshold=1.0`, `warning_threshold=20.0`

### `jump_cut_score` [↑](#categories)
> Jump cut absence (0-1, 1=no cuts) · ↑ higher=better · 0-1, 1=no cuts

**[`jump_cut`](src/ayase/modules/jump_cut.py)** — Jump cut / abrupt transition detection (0-1, 1=no cuts)

- **Input**: vid · **Speed**: ⚡ fast
- **Packages**: opencv-python
- **Tests**: covered by [`test_jump_cut.py`](tests/modules/per_module/test_jump_cut.py), [`test_curation_metrics.py`](tests/modules/test_curation_metrics.py)
- **Config**: `threshold=40.0`

### `lse_c` [↑](#categories)
> LSE-C lip sync error confidence (higher=better) · ↑ higher=better

**[`lip_sync`](src/ayase/modules/lip_sync.py)** — LSE-D/LSE-C lip sync error (SyncNet/Wav2Lip, 2020)

- **Input**: audio · **Speed**: ⚡ fast
- **Backend**: syncnet
- **Packages**: soundfile, syncnet
- **Tests**: covered by [`test_lip_sync.py`](tests/modules/per_module/test_lip_sync.py)
- **Config**: `subsample=16`, `sample_rate=16000`

### `lse_d` [↑](#categories)
> LSE-D lip sync error distance (lower=better) · ↓ lower=better

**[`lip_sync`](src/ayase/modules/lip_sync.py)** — LSE-D/LSE-C lip sync error (SyncNet/Wav2Lip, 2020)

- **Input**: audio · **Speed**: ⚡ fast
- **Backend**: syncnet
- **Packages**: soundfile, syncnet
- **Tests**: covered by [`test_lip_sync.py`](tests/modules/per_module/test_lip_sync.py)
- **Config**: `subsample=16`, `sample_rate=16000`

### `object_permanence_score` [↑](#categories)
> ↑ higher=better

**[`object_permanence`](src/ayase/modules/object_permanence.py)** — Object tracking consistency (ID switches, disappearances)

- **Input**: vid · **Speed**: ⚡ fast
- **Packages**: ultralytics
- **Tests**: covered by [`test_object_permanence.py`](tests/modules/per_module/test_object_permanence.py), [`test_opencv_modules.py`](tests/modules/test_opencv_modules.py)
- **Config**: `backend=auto`, `subsample=2`, `max_frames=300`, `match_distance=80.0`, `warning_threshold=50.0`

### `scene_stability` [↑](#categories)

**[`scene_detection`](src/ayase/modules/scene_detection.py)** — Scene stability metric — penalises rapid cuts (0-1, higher=more stable)

- **Input**: vid · **Speed**: ⚡ fast
- **Packages**: opencv-python, transnetv2
- **Tests**: covered by [`test_scene_detection.py`](tests/modules/per_module/test_scene_detection.py), [`test_motion_scene_semantic_metrics.py`](tests/modules/test_motion_scene_semantic_metrics.py)
- **Config**: `threshold=0.5`

### `semantic_consistency` [↑](#categories)
> Segmentation temporal IoU 0-1 (higher=better) · ↑ higher=better

**[`semantic_segmentation_consistency`](src/ayase/modules/semantic_segmentation_consistency.py)** — Temporal stability of semantic segmentation

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Packages**: Pillow, torch, transformers
- **Source**: <a href="https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512" target="_blank">HF</a>
- **Tests**: covered by [`test_semantic_segmentation_consistency.py`](tests/modules/per_module/test_semantic_segmentation_consistency.py), [`test_depth_and_multiview.py`](tests/modules/test_depth_and_multiview.py)
- **Config**: `backend=auto`, `device=auto`, `subsample=3`, `max_frames=150`, `num_clusters=8`, `warning_threshold=0.6`

### `stutter_score` [↑](#categories)
> Duplicate/dropped frames 0-100 (lower=better) · ↓ lower=better

**[`judder_stutter`](src/ayase/modules/judder_stutter.py)** — Detects judder (uneven cadence) and stutter (duplicate frames)

- **Input**: vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_judder_stutter.py`](tests/modules/per_module/test_judder_stutter.py), [`test_opencv_modules.py`](tests/modules/test_opencv_modules.py)
- **Config**: `max_frames=600`, `duplicate_threshold=1.0`, `warning_threshold=20.0`

### `subject_consistency` [↑](#categories)
> Subject identity consistency (0-1, higher=better) · ↑ higher=better · 0-1

**[`subject_consistency`](src/ayase/modules/subject_consistency.py)** — Subject consistency using DINOv2-base (all pairwise frame similarity)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Packages**: torch, transformers
- **VRAM**: ~400 MB
- **Source**: <a href="https://huggingface.co/facebook/dinov2-base" target="_blank">HF</a>
- **Tests**: covered by [`test_subject_consistency.py`](tests/modules/per_module/test_subject_consistency.py), [`test_integration_synthetic.py`](tests/test_integration_synthetic.py)
- **Config**: `model_name=facebook/dinov2-base`, `max_frames=16`, `warning_threshold=0.6`

### `video_text_temporal` [↑](#categories)
> Video-text temporal consistency (0-1) · 0-1

**[`video_text_matching`](src/ayase/modules/video_text_matching.py)** — ViCLIP / X-CLIP (Temporal alignment) or Frame-averaged CLIP

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: Pillow, torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_video_text_matching.py`](tests/modules/per_module/test_video_text_matching.py)
- **Config**: `use_xclip=False`, `model_name=openai/clip-vit-base-patch32`, `xclip_model_name=microsoft/xclip-base-patch32`, `min_score_threshold=0.2`, `consistency_std_threshold=0.1`

### `videoscore_temporal` [↑](#categories)
> VideoScore temporal consistency · ↑ higher=better

**[`videoscore`](src/ayase/modules/videoscore.py)** — VideoScore 5-dimensional video quality assessment (1-4 scale)

- **Input**: img/vid · **Speed**: 🐌 slow · GPU
- **Packages**: Pillow, opencv-python, torch, transformers
- **Source**: <a href="https://huggingface.co/TIGER-Lab/VideoScore" target="_blank">HF</a>
- **Tests**: covered by [`test_videoscore.py`](tests/modules/per_module/test_videoscore.py), [`test_iqa_research_metrics.py`](tests/modules/test_iqa_research_metrics.py)
- **Config**: `model_name=TIGER-Lab/VideoScore`, `num_frames=8`, `trust_remote_code=True`

### `warping_error` [↑](#categories)
> ↓ lower=better

**[`temporal_flickering`](src/ayase/modules/temporal_flickering.py)** — Warping Error using RAFT optical flow with occlusion masking

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Packages**: torch, torchvision
- **Tests**: covered by [`test_temporal_flickering.py`](tests/modules/per_module/test_temporal_flickering.py), [`test_motion_scene_semantic_metrics.py`](tests/modules/test_motion_scene_semantic_metrics.py), [`test_integration_synthetic.py`](tests/test_integration_synthetic.py)
- **Config**: `warning_threshold=0.02`, `max_frames=300`

### `world_consistency_score` [↑](#categories)
> WCS object permanence (higher=better) · ↑ higher=better

**[`world_consistency`](src/ayase/modules/world_consistency.py)** — World Consistency Score: object permanence + causal compliance (2025)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Backend**: dinov2 → clip
- **Packages**: Pillow, torch, torchvision, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/facebookresearch/dinov2" target="_blank">HF</a>
- **Tests**: covered by [`test_world_consistency.py`](tests/modules/per_module/test_world_consistency.py)
- **Config**: `subsample=12`, `permanence_weight=0.4`, `stability_weight=0.3`, `causal_weight=0.3`


## Motion & Dynamics (22 metrics)

### `aigv_dynamic` [↑](#categories)
> AI video dynamic degree

**[`aigv_assessor`](src/ayase/modules/aigv_assessor.py)** — AI-generated video quality (AIGV-Assessor model or CLIP proxy)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Backend**: aigv_assessor → clip
- **Packages**: Pillow, opencv-python, torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/IntMeGroup/AIGV-Assessor-static_quality" target="_blank">HF</a>
- **Tests**: covered by [`test_aigv_assessor.py`](tests/modules/per_module/test_aigv_assessor.py), [`test_motion_scene_semantic_metrics.py`](tests/modules/test_motion_scene_semantic_metrics.py)
- **Config**: `subsample=8`, `trust_remote_code=True`

### `bas_score` [↑](#categories)
> BAS beat alignment score (higher=better) · ↑ higher=better

**[`beat_alignment`](src/ayase/modules/beat_alignment.py)** — BAS beat alignment score — audio-motion sync (EDGE/CVPR 2023)

- **Input**: audio · **Speed**: ⚡ fast
- **Backend**: native → librosa
- **Packages**: librosa
- **Tests**: covered by [`test_beat_alignment.py`](tests/modules/per_module/test_beat_alignment.py)
- **Config**: `tolerance=0.1`, `subsample=2`

### `camera_jitter_score` [↑](#categories)
> Camera stability (0-1, 1=stable) · ↓ lower=better · 0-1, 1=stable

**[`camera_jitter`](src/ayase/modules/camera_jitter.py)** — Camera jitter/shake detection (0-1, 1=stable)

- **Input**: vid · **Speed**: ⚡ fast
- **Packages**: opencv-python
- **Tests**: covered by [`test_camera_jitter.py`](tests/modules/per_module/test_camera_jitter.py), [`test_curation_metrics.py`](tests/modules/test_curation_metrics.py)
- **Config**: `subsample=16`

### `camera_motion_score` [↑](#categories)
> Camera motion intensity · ↑ higher=better

**[`camera_motion`](src/ayase/modules/camera_motion.py)** — Analyzes camera motion stability (VMBench) using Homography

- **Input**: vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_camera_motion.py`](tests/modules/per_module/test_camera_motion.py)

### `dynamics_controllability` [↑](#categories)
> Motion control fidelity

**[`dynamics_controllability`](src/ayase/modules/dynamics_controllability.py)** — Assesses motion controllability based on text-motion alignment

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Backend**: farneback → cotracker
- **Packages**: torch
- **Source**: <a href="https://huggingface.co/facebookresearch/co-tracker" target="_blank">HF</a>
- **Tests**: covered by [`test_dynamics_controllability.py`](tests/modules/per_module/test_dynamics_controllability.py), [`test_fields_general.py`](tests/modules/test_fields_general.py), [`test_opencv_modules.py`](tests/modules/test_opencv_modules.py), +2 more
- **Config**: `subsample=16`

### `dynamics_range` [↑](#categories)
> Extent of content variation

**[`dynamics_range`](src/ayase/modules/dynamics_range.py)** — Measures extent of motion and content variation (DEVIL protocol)

- **Input**: vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_dynamics_range.py`](tests/modules/per_module/test_dynamics_range.py), [`test_fields_general.py`](tests/modules/test_fields_general.py), [`test_opencv_modules.py`](tests/modules/test_opencv_modules.py), +1 more
- **Config**: `scene_change_threshold=30.0`

### `flow_score` [↑](#categories)
> ↑ higher=better

**[`advanced_flow`](src/ayase/modules/advanced_flow.py)** — RAFT optical flow: flow_score (all consecutive pairs)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Packages**: torch, torchvision
- **Tests**: covered by [`test_advanced_flow.py`](tests/modules/per_module/test_advanced_flow.py), [`test_integration_synthetic.py`](tests/test_integration_synthetic.py)
- **Config**: `use_large_model=True`, `max_frames=150`

### `kandinsky_camera_motion_score` [↑](#categories)
> Kandinsky camera motion prediction · ↑ higher=better · higher=more camera motion

**[`kandinsky_motion`](src/ayase/modules/kandinsky_motion.py)** — Video/Camera Motion Analysis using Kandinsky Video Tools (VideoMAE-V2)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Packages**: torch
- **Source**: <a href="https://huggingface.co/ai-forever/kandinsky-video-motion-predictor" target="_blank">HF</a>
- **Tests**: covered by [`test_kandinsky_motion.py`](tests/modules/per_module/test_kandinsky_motion.py)

### `kandinsky_dynamics_score` [↑](#categories)
> Kandinsky dynamics prediction · ↑ higher=better · higher=more dynamic

**[`kandinsky_motion`](src/ayase/modules/kandinsky_motion.py)** — Video/Camera Motion Analysis using Kandinsky Video Tools (VideoMAE-V2)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Packages**: torch
- **Source**: <a href="https://huggingface.co/ai-forever/kandinsky-video-motion-predictor" target="_blank">HF</a>
- **Tests**: covered by [`test_kandinsky_motion.py`](tests/modules/per_module/test_kandinsky_motion.py)

### `kandinsky_object_motion_score` [↑](#categories)
> Kandinsky object motion prediction · ↑ higher=better · higher=more object motion

**[`kandinsky_motion`](src/ayase/modules/kandinsky_motion.py)** — Video/Camera Motion Analysis using Kandinsky Video Tools (VideoMAE-V2)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Packages**: torch
- **Source**: <a href="https://huggingface.co/ai-forever/kandinsky-video-motion-predictor" target="_blank">HF</a>
- **Tests**: covered by [`test_kandinsky_motion.py`](tests/modules/per_module/test_kandinsky_motion.py)

### `motion_ac_score` [↑](#categories)
> ↑ higher=better

**[`motion_amplitude`](src/ayase/modules/motion_amplitude.py)** — Motion amplitude classification vs caption (motion_ac_score via RAFT)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Packages**: torch, torchvision
- **Tests**: covered by [`test_motion_amplitude.py`](tests/modules/per_module/test_motion_amplitude.py), [`test_integration_synthetic.py`](tests/test_integration_synthetic.py)
- **Config**: `amplitude_threshold=5.0`, `max_frames=150`, `scoring_mode=binary`

### `motion_score` [↑](#categories)
> Scene motion intensity · ↑ higher=better

**[`motion`](src/ayase/modules/motion.py)** — Analyzes motion dynamics (optical flow, flickering)

- **Input**: vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_motion.py`](tests/modules/per_module/test_motion.py), [`test_vbench2_compbench.py`](tests/modules/test_vbench2_compbench.py), [`test_integration_synthetic.py`](tests/test_integration_synthetic.py), +3 more
- **Config**: `sample_rate=5`, `low_motion_threshold=0.5`, `high_motion_threshold=20.0`

### `motion_smoothness` [↑](#categories)
> Motion smoothness (0-1, higher=better) · ↑ higher=better · 0-1

**[`motion_smoothness`](src/ayase/modules/motion_smoothness.py)** — Motion smoothness via RIFE VFI reconstruction error (VBench)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Packages**: rife_model, torch
- **Source**: <a href="https://huggingface.co/rife/flownet.pkl" target="_blank">HF</a>
- **Tests**: covered by [`test_motion_smoothness.py`](tests/modules/per_module/test_motion_smoothness.py), [`test_integration_synthetic.py`](tests/test_integration_synthetic.py)
- **Config**: `vfi_error_threshold=0.08`, `max_frames=64`

### `physics_score` [↑](#categories)
> Physics plausibility (0-1, higher=better) · ↑ higher=better · 0-1

**[`physics`](src/ayase/modules/physics.py)** — Physics plausibility via trajectory analysis (CoTracker / Lucas-Kanade)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Backend**: cotracker → lk
- **Packages**: torch
- **Source**: <a href="https://huggingface.co/facebookresearch/co-tracker" target="_blank">HF</a>
- **Tests**: covered by [`test_physics.py`](tests/modules/per_module/test_physics.py), [`test_vbench2_compbench.py`](tests/modules/test_vbench2_compbench.py)
- **Config**: `subsample=16`, `accel_threshold=50.0`

### `playback_speed_score` [↑](#categories)
> Normal speed (1.0=normal) · ↑ higher=better

**[`playback_speed`](src/ayase/modules/playback_speed.py)** — Playback speed normality detection (1.0=normal)

- **Input**: vid · **Speed**: ⚡ fast
- **Packages**: opencv-python
- **Tests**: covered by [`test_playback_speed.py`](tests/modules/per_module/test_playback_speed.py), [`test_curation_metrics.py`](tests/modules/test_curation_metrics.py)
- **Config**: `subsample=16`

### `ptlflow_motion_score` [↑](#categories)
> ptlflow optical flow magnitude · ↑ higher=better

**[`ptlflow_motion`](src/ayase/modules/ptlflow_motion.py)** — ptlflow optical flow motion scoring (dpflow model)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Packages**: opencv-python, ptlflow, torch
- **Tests**: covered by [`test_ptlflow_motion.py`](tests/modules/per_module/test_ptlflow_motion.py), [`test_motion_scene_semantic_metrics.py`](tests/modules/test_motion_scene_semantic_metrics.py)
- **Config**: `model_name=dpflow`, `ckpt_path=things`, `subsample=8`

### `raft_motion_score` [↑](#categories)
> RAFT optical flow magnitude · ↑ higher=better

**[`raft_motion`](src/ayase/modules/raft_motion.py)** — RAFT optical flow motion scoring (torchvision)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Packages**: opencv-python, torch, torchvision
- **Tests**: covered by [`test_raft_motion.py`](tests/modules/per_module/test_raft_motion.py), [`test_motion_scene_semantic_metrics.py`](tests/modules/test_motion_scene_semantic_metrics.py)
- **Config**: `subsample=8`

### `stabilized_camera_score` [↑](#categories)
> Stabilized camera motion estimate · ↑ higher=better

**[`stabilized_motion`](src/ayase/modules/stabilized_motion.py)** — Calculates motion scores with camera stabilization (ORB+Homography)

- **Input**: vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_stabilized_motion.py`](tests/modules/per_module/test_stabilized_motion.py)
- **Config**: `step=2`, `threshold_px=0.5`, `stabilize=True`, `high_camera_motion_threshold=5.0`, `static_threshold=0.1`

### `stabilized_motion_score` [↑](#categories)
> Stabilized scene motion (camera-invariant) · ↑ higher=better

**[`stabilized_motion`](src/ayase/modules/stabilized_motion.py)** — Calculates motion scores with camera stabilization (ORB+Homography)

- **Input**: vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_stabilized_motion.py`](tests/modules/per_module/test_stabilized_motion.py)
- **Config**: `step=2`, `threshold_px=0.5`, `stabilize=True`, `high_camera_motion_threshold=5.0`, `static_threshold=0.1`

### `trajan_score` [↑](#categories)
> Point track motion consistency · ↑ higher=better

**[`trajan`](src/ayase/modules/trajan.py)** — Motion consistency via point tracking (CoTracker or Lucas-Kanade fallback)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Backend**: lk → cotracker
- **Packages**: cotracker, opencv-python, torch
- **Source**: <a href="https://huggingface.co/facebookresearch/co-tracker" target="_blank">HF</a>
- **Tests**: covered by [`test_trajan.py`](tests/modules/per_module/test_trajan.py), [`test_motion_scene_semantic_metrics.py`](tests/modules/test_motion_scene_semantic_metrics.py)
- **Config**: `num_frames=16`, `num_points=256`

### `videoreward_mq` [↑](#categories)
> VideoReward motion quality

**[`videoreward`](src/ayase/modules/videoreward.py)** — VideoReward Kling multi-dim reward model (NeurIPS 2025)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Packages**: Pillow, torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_videoreward.py`](tests/modules/per_module/test_videoreward.py)
- **Config**: `subsample=8`, `clip_model=openai/clip-vit-base-patch32`

### `videoscore_dynamic` [↑](#categories)
> VideoScore dynamic degree · ↑ higher=better

**[`videoscore`](src/ayase/modules/videoscore.py)** — VideoScore 5-dimensional video quality assessment (1-4 scale)

- **Input**: img/vid · **Speed**: 🐌 slow · GPU
- **Packages**: Pillow, opencv-python, torch, transformers
- **Source**: <a href="https://huggingface.co/TIGER-Lab/VideoScore" target="_blank">HF</a>
- **Tests**: covered by [`test_videoscore.py`](tests/modules/per_module/test_videoscore.py), [`test_iqa_research_metrics.py`](tests/modules/test_iqa_research_metrics.py)
- **Config**: `model_name=TIGER-Lab/VideoScore`, `num_frames=8`, `trust_remote_code=True`


## Basic Visual Quality (15 metrics)

### `artifacts_score` [↑](#categories)
> ↑ higher=better

**[`basic_quality`](src/ayase/modules/basic.py)** — Comprehensive technical quality assessment (blur, noise, artifacts, contrast)

- **Input**: img/vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_basic_quality.py`](tests/modules/per_module/test_basic_quality.py), [`test_docs_integrity.py`](tests/test_docs_integrity.py), [`test_profiles.py`](tests/test_profiles.py), +3 more
- **Config**: `threshold=40.0`, `blur_threshold=100.0`, `noise_threshold=50.0`

### `blur_score` [↑](#categories)
> Laplacian variance · ↑ higher=better

**[`basic_quality`](src/ayase/modules/basic.py)** — Comprehensive technical quality assessment (blur, noise, artifacts, contrast)

- **Input**: img/vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_basic_quality.py`](tests/modules/per_module/test_basic_quality.py), [`test_docs_integrity.py`](tests/test_docs_integrity.py), [`test_profiles.py`](tests/test_profiles.py), +3 more
- **Config**: `threshold=40.0`, `blur_threshold=100.0`, `noise_threshold=50.0`

### `brightness` [↑](#categories)

**[`basic_quality`](src/ayase/modules/basic.py)** — Comprehensive technical quality assessment (blur, noise, artifacts, contrast)

- **Input**: img/vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_basic_quality.py`](tests/modules/per_module/test_basic_quality.py), [`test_docs_integrity.py`](tests/test_docs_integrity.py), [`test_profiles.py`](tests/test_profiles.py), +3 more
- **Config**: `threshold=40.0`, `blur_threshold=100.0`, `noise_threshold=50.0`

### `compression_artifacts` [↑](#categories)
> Artifact severity (0-100) · 0-100

**[`compression_artifacts`](src/ayase/modules/compression_artifacts.py)** — Detects compression artifacts (blocking, ringing, mosquito noise)

- **Input**: vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_compression_artifacts.py`](tests/modules/per_module/test_compression_artifacts.py), [`test_fields_general.py`](tests/modules/test_fields_general.py), [`test_opencv_modules.py`](tests/modules/test_opencv_modules.py), +1 more
- **Config**: `subsample=3`, `warning_threshold=40.0`

### `contrast` [↑](#categories)

**[`basic_quality`](src/ayase/modules/basic.py)** — Comprehensive technical quality assessment (blur, noise, artifacts, contrast)

- **Input**: img/vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_basic_quality.py`](tests/modules/per_module/test_basic_quality.py), [`test_docs_integrity.py`](tests/test_docs_integrity.py), [`test_profiles.py`](tests/test_profiles.py), +3 more
- **Config**: `threshold=40.0`, `blur_threshold=100.0`, `noise_threshold=50.0`

### `cpbd_score` [↑](#categories)
> CPBD perceptual blur detection (0-1, higher=sharper) · ↑ higher=better · 0-1, higher=sharper

**[`cpbd`](src/ayase/modules/cpbd.py)** — Cumulative Probability of Blur Detection (Perceptual Blur)

- **Input**: img/vid · **Speed**: ⚡ fast
- **Packages**: cpbd
- **Tests**: covered by [`test_cpbd.py`](tests/modules/per_module/test_cpbd.py)
- **Config**: `threshold_cpbd=0.65`, `threshold_heuristic=0.3`

### `imaging_artifacts_score` [↑](#categories)
> Imaging edge-density artifacts (0-1, higher=cleaner) · ↑ higher=better · 0-1, higher=cleaner

**[`imaging_quality`](src/ayase/modules/imaging_quality.py)** — Assesses technical quality (Noise, Blockiness) - Proxy for MUSIQ/DOVER

- **Input**: img/vid · **Speed**: ⚡ fast
- **Packages**: Pillow, brisque, imquality
- **VRAM**: ~800 MB
- **Tests**: covered by [`test_imaging_quality.py`](tests/modules/per_module/test_imaging_quality.py)
- **Config**: `noise_threshold=20.0`

### `imaging_noise_score` [↑](#categories)
> Imaging noise level (0-1, higher=cleaner) · ↑ higher=better · 0-1, higher=cleaner

**[`imaging_quality`](src/ayase/modules/imaging_quality.py)** — Assesses technical quality (Noise, Blockiness) - Proxy for MUSIQ/DOVER

- **Input**: img/vid · **Speed**: ⚡ fast
- **Packages**: Pillow, brisque, imquality
- **VRAM**: ~800 MB
- **Tests**: covered by [`test_imaging_quality.py`](tests/modules/per_module/test_imaging_quality.py)
- **Config**: `noise_threshold=20.0`

### `letterbox_ratio` [↑](#categories)
> Border/letterbox fraction (0-1, 0=no borders) · 0-1, 0=no borders

**[`letterbox`](src/ayase/modules/letterbox.py)** — Border/letterbox detection (0-1, 0=no borders)

- **Input**: img/vid · **Speed**: ⚡ fast
- **Packages**: opencv-python
- **Tests**: covered by [`test_letterbox.py`](tests/modules/per_module/test_letterbox.py), [`test_curation_metrics.py`](tests/modules/test_curation_metrics.py)
- **Config**: `threshold=16`, `subsample=4`

### `noise_score` [↑](#categories)
> ↑ higher=better

**[`basic_quality`](src/ayase/modules/basic.py)** — Comprehensive technical quality assessment (blur, noise, artifacts, contrast)

- **Input**: img/vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_basic_quality.py`](tests/modules/per_module/test_basic_quality.py), [`test_docs_integrity.py`](tests/test_docs_integrity.py), [`test_profiles.py`](tests/test_profiles.py), +3 more
- **Config**: `threshold=40.0`, `blur_threshold=100.0`, `noise_threshold=50.0`

### `saturation` [↑](#categories)
> Advanced metrics

**[`basic_quality`](src/ayase/modules/basic.py)** — Comprehensive technical quality assessment (blur, noise, artifacts, contrast)

- **Input**: img/vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_basic_quality.py`](tests/modules/per_module/test_basic_quality.py), [`test_docs_integrity.py`](tests/test_docs_integrity.py), [`test_profiles.py`](tests/test_profiles.py), +3 more
- **Config**: `threshold=40.0`, `blur_threshold=100.0`, `noise_threshold=50.0`

### `spatial_information` [↑](#categories)
> ITU-T P.910 SI (higher=more detail) · higher=more detail

**[`ti_si`](src/ayase/modules/ti_si.py)** — ITU-T P.910 Temporal & Spatial Information

- **Input**: vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_ti_si.py`](tests/modules/per_module/test_ti_si.py), [`test_opencv_modules.py`](tests/modules/test_opencv_modules.py)
- **Config**: `max_frames=300`

### `technical_score` [↑](#categories)
> Composite technical score · ↑ higher=better

Used by: [`usability_rate`](src/ayase/modules/usability_rate.py)

**[`basic_quality`](src/ayase/modules/basic.py)** — Comprehensive technical quality assessment (blur, noise, artifacts, contrast)

- **Input**: img/vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_basic_quality.py`](tests/modules/per_module/test_basic_quality.py), [`test_docs_integrity.py`](tests/test_docs_integrity.py), [`test_profiles.py`](tests/test_profiles.py), +3 more
- **Config**: `threshold=40.0`, `blur_threshold=100.0`, `noise_threshold=50.0`

### `temporal_information` [↑](#categories)
> ITU-T P.910 TI (higher=more motion) · higher=more motion

**[`ti_si`](src/ayase/modules/ti_si.py)** — ITU-T P.910 Temporal & Spatial Information

- **Input**: vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_ti_si.py`](tests/modules/per_module/test_ti_si.py), [`test_opencv_modules.py`](tests/modules/test_opencv_modules.py)
- **Config**: `max_frames=300`

### `tonal_dynamic_range` [↑](#categories)
> Luminance histogram span (0-100) · 0-100

**[`tonal_dynamic_range`](src/ayase/modules/tonal_dynamic_range.py)** — Luminance histogram tonal range (0-100)

- **Input**: img/vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_tonal_dynamic_range.py`](tests/modules/per_module/test_tonal_dynamic_range.py), [`test_tonal_dynamic_range.py`](tests/modules/test_tonal_dynamic_range.py)
- **Config**: `low_percentile=1`, `high_percentile=99`, `subsample=8`


## Aesthetics (9 metrics)

### `aesthetic_mlp_score` [↑](#categories)
> LAION Aesthetics MLP (1-10) · ↑ higher=better · 1-10

**[`aesthetic_scoring`](src/ayase/modules/aesthetic_scoring.py)** — Calculates aesthetic score (1-10) using LAION-Aesthetics MLP

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: Pillow, torch, transformers
- **VRAM**: ~1.5 GB
- **Source**: <a href="https://github.com/christophschuhmann/improved-aesthetic-predictor" target="_blank">GitHub</a> · <a href="https://huggingface.co/openai/clip-vit-large-patch14" target="_blank">HF</a>
- **Tests**: covered by [`test_aesthetic_scoring.py`](tests/modules/per_module/test_aesthetic_scoring.py)

### `aesthetic_score` [↑](#categories)
> 0-100, normalized from aesthetic predictor · ↑ higher=better · 0-100

Used by: [`knowledge_graph`](src/ayase/modules/knowledge_graph.py), [`usability_rate`](src/ayase/modules/usability_rate.py)

**[`aesthetic`](src/ayase/modules/aesthetic.py)** — Estimates aesthetic quality using Aesthetic Predictor V2.5

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: aesthetic_predictor_v2_5, torch
- **Tests**: covered by [`test_aesthetic.py`](tests/modules/per_module/test_aesthetic.py), [`test_vbench2_compbench.py`](tests/modules/test_vbench2_compbench.py)
- **Config**: `num_frames=5`, `trust_remote_code=True`

### `cover_aesthetic` [↑](#categories)
> COVER aesthetic branch

**[`cover`](src/ayase/modules/cover.py)** — COVER 3-branch comprehensive video quality (semantic + aesthetic + technical)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Backend**: cover → dover
- **Packages**: cover, opencv-python, pyiqa, torch
- **VRAM**: ~800 MB
- **Tests**: covered by [`test_cover.py`](tests/modules/per_module/test_cover.py), [`test_iqa_research_metrics.py`](tests/modules/test_iqa_research_metrics.py)
- **Config**: `subsample=8`, `quality_threshold=30.0`

### `cover_semantic` [↑](#categories)
> COVER semantic branch

**[`cover`](src/ayase/modules/cover.py)** — COVER 3-branch comprehensive video quality (semantic + aesthetic + technical)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Backend**: cover → dover
- **Packages**: cover, opencv-python, pyiqa, torch
- **VRAM**: ~800 MB
- **Tests**: covered by [`test_cover.py`](tests/modules/per_module/test_cover.py), [`test_iqa_research_metrics.py`](tests/modules/test_iqa_research_metrics.py)
- **Config**: `subsample=8`, `quality_threshold=30.0`

### `creativity_score` [↑](#categories)
> Artistic novelty (0-1, higher=better) · ↑ higher=better · 0-1

**[`creativity`](src/ayase/modules/creativity.py)** — Artistic novelty assessment (VLM / CLIP)

- **Input**: img/vid · **Speed**: 🐌 slow · GPU
- **Backend**: vlm → clip
- **Packages**: Pillow, pyiqa, torch, torchvision, transformers
- **VRAM**: ~14 GB
- **Source**: <a href="https://huggingface.co/llava-hf/llava-1.5-7b-hf" target="_blank">HF</a>
- **Tests**: covered by [`test_creativity.py`](tests/modules/per_module/test_creativity.py), [`test_vbench2_compbench.py`](tests/modules/test_vbench2_compbench.py)
- **Config**: `vlm_model=llava-hf/llava-1.5-7b-hf`

### `dover_aesthetic` [↑](#categories)
> DOVER aesthetic quality · 0-1 sigmoid

**[`dover`](src/ayase/modules/dover.py)** — DOVER disentangled technical + aesthetic VQA (ICCV 2023)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Backend**: heuristic → native → onnx → pyiqa
- **Packages**: onnxruntime, pyiqa, torch
- **VRAM**: ~800 MB
- **Source**: <a href="https://github.com/VQAssessment/DOVER.git" target="_blank">GitHub</a> · <a href="https://huggingface.co/dover/DOVER.pth" target="_blank">HF</a>
- **Tests**: covered by [`test_dover.py`](tests/modules/per_module/test_dover.py), [`test_ml_basics.py`](tests/modules/test_ml_basics.py)
- **Config**: `warning_threshold=0.4`

### `laion_aesthetic` [↑](#categories)
> LAION Aesthetics V2 (0-10) · 0-10

**[`laion_aesthetic`](src/ayase/modules/laion_aesthetic.py)** — LAION Aesthetics V2 predictor (0-10, industry standard)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: opencv-python, pyiqa, torch
- **Tests**: covered by [`test_laion_aesthetic.py`](tests/modules/per_module/test_laion_aesthetic.py), [`test_image_iqa_metrics.py`](tests/modules/test_image_iqa_metrics.py)
- **Config**: `subsample=4`

### `nima_score` [↑](#categories)
> NIMA aesthetic+technical (1-10, higher=better) · ↑ higher=better · 1-10

**[`nima`](src/ayase/modules/nima.py)** — NIMA aesthetic and technical image quality (1-10 scale)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: opencv-python, pyiqa, torch
- **Tests**: covered by [`test_nima.py`](tests/modules/per_module/test_nima.py), [`test_iqa_research_metrics.py`](tests/modules/test_iqa_research_metrics.py)
- **Config**: `subsample=8`

### `qalign_aesthetic` [↑](#categories)
> Q-Align aesthetic quality (1-5, higher=better) · ↑ higher=better · 1-5

**[`q_align`](src/ayase/modules/q_align.py)** — Q-Align unified quality + aesthetic assessment (ICML 2024)

- **Input**: img/vid · **Speed**: 🐌 slow · GPU
- **Packages**: Pillow, torch, transformers
- **VRAM**: ~14 GB
- **Source**: <a href="https://huggingface.co/q-future/one-align" target="_blank">HF</a>
- **Tests**: covered by [`test_q_align.py`](tests/modules/per_module/test_q_align.py), [`test_ml_basics.py`](tests/modules/test_ml_basics.py)
- **Config**: `model_name=q-future/one-align`, `dtype=float16`, `device=auto`, `subsample=8`, `max_frames=16`, `warning_threshold=2.5`, `trust_remote_code=True`


## Audio Quality (20 metrics)

### `audiobox_enjoyment` [↑](#categories)
> Audiobox content enjoyment

**[`audiobox_aesthetics`](src/ayase/modules/audiobox_aesthetics.py)** — Meta Audiobox Aesthetics audio quality (2025)

- **Input**: audio · **Speed**: ⚡ fast
- **Backend**: audiobox
- **Packages**: audiobox_aesthetics, soundfile
- **Tests**: covered by [`test_audiobox_aesthetics.py`](tests/modules/per_module/test_audiobox_aesthetics.py)
- **Config**: `sample_rate=16000`

### `audiobox_production` [↑](#categories)
> Audiobox production quality

**[`audiobox_aesthetics`](src/ayase/modules/audiobox_aesthetics.py)** — Meta Audiobox Aesthetics audio quality (2025)

- **Input**: audio · **Speed**: ⚡ fast
- **Backend**: audiobox
- **Packages**: audiobox_aesthetics, soundfile
- **Tests**: covered by [`test_audiobox_aesthetics.py`](tests/modules/per_module/test_audiobox_aesthetics.py)
- **Config**: `sample_rate=16000`

### `av_sync_offset` [↑](#categories)
> Audio-video sync offset in ms

**[`av_sync`](src/ayase/modules/audio_visual_sync.py)** — Audio-video synchronisation offset detection

- **Input**: audio · **Speed**: ⚡ fast
- **Packages**: soundfile
- **Tests**: covered by [`test_av_sync.py`](tests/modules/per_module/test_av_sync.py), [`test_ml_basics.py`](tests/modules/test_ml_basics.py), [`test_docs_integrity.py`](tests/test_docs_integrity.py)
- **Config**: `max_frames=600`, `warning_threshold_ms=80.0`

### `dnsmos_bak` [↑](#categories)
> DNSMOS background quality (1-5, higher=better) · ↑ higher=better · 1-5

**[`dnsmos`](src/ayase/modules/dnsmos.py)** — DNSMOS non-intrusive audio quality (Microsoft, 1-5 MOS)

- **Input**: audio · **Speed**: ⏱️ medium
- **Backend**: torchmetrics
- **Packages**: librosa, soundfile, torch, torchmetrics
- **Tests**: covered by [`test_dnsmos.py`](tests/modules/per_module/test_dnsmos.py), [`test_industry_metrics.py`](tests/modules/test_industry_metrics.py)

### `dnsmos_overall` [↑](#categories)
> DNSMOS overall MOS (1-5, higher=better) · ↑ higher=better · 1-5

**[`dnsmos`](src/ayase/modules/dnsmos.py)** — DNSMOS non-intrusive audio quality (Microsoft, 1-5 MOS)

- **Input**: audio · **Speed**: ⏱️ medium
- **Backend**: torchmetrics
- **Packages**: librosa, soundfile, torch, torchmetrics
- **Tests**: covered by [`test_dnsmos.py`](tests/modules/per_module/test_dnsmos.py), [`test_industry_metrics.py`](tests/modules/test_industry_metrics.py)

### `dnsmos_sig` [↑](#categories)
> DNSMOS signal quality (1-5, higher=better) · ↑ higher=better · 1-5

**[`dnsmos`](src/ayase/modules/dnsmos.py)** — DNSMOS non-intrusive audio quality (Microsoft, 1-5 MOS)

- **Input**: audio · **Speed**: ⏱️ medium
- **Backend**: torchmetrics
- **Packages**: librosa, soundfile, torch, torchmetrics
- **Tests**: covered by [`test_dnsmos.py`](tests/modules/per_module/test_dnsmos.py), [`test_industry_metrics.py`](tests/modules/test_industry_metrics.py)

### `estoi_score` [↑](#categories)
> ESTOI intelligibility (0-1, higher=better) · ↑ higher=better · 0-1

**[`audio_estoi`](src/ayase/modules/audio_estoi.py)** — ESTOI speech intelligibility (full-reference)

- **Input**: audio +ref · **Speed**: ⚡ fast
- **Packages**: librosa, pystoi, soundfile
- **Tests**: covered by [`test_audio_estoi.py`](tests/modules/per_module/test_audio_estoi.py), [`test_audio_metrics.py`](tests/test_audio_metrics.py)
- **Config**: `target_sr=10000`, `warning_threshold=0.5`

### `lpdist_score` [↑](#categories)
> Log-Power Spectral Distance (lower=better) · ↓ lower=better

**[`audio_lpdist`](src/ayase/modules/audio_lpdist.py)** — Log-Power Spectral Distance (full-reference audio)

- **Input**: audio +ref · **Speed**: ⚡ fast
- **Packages**: librosa
- **Tests**: covered by [`test_audio_lpdist.py`](tests/modules/per_module/test_audio_lpdist.py), [`test_audio_metrics.py`](tests/test_audio_metrics.py)
- **Config**: `target_sr=16000`, `n_mels=80`, `warning_threshold=4.0`

### `mcd_score` [↑](#categories)
> Mel Cepstral Distortion (dB, lower=better) · ↓ lower=better · dB

**[`audio_mcd`](src/ayase/modules/audio_mcd.py)** — Mel Cepstral Distortion for TTS/VC quality (full-reference)

- **Input**: audio +ref · **Speed**: ⚡ fast
- **Packages**: librosa
- **Tests**: covered by [`test_audio_mcd.py`](tests/modules/per_module/test_audio_mcd.py), [`test_audio_metrics.py`](tests/test_audio_metrics.py)
- **Config**: `target_sr=16000`, `n_mfcc=13`, `warning_threshold=8.0`

### `oavqa_score` [↑](#categories)
> OAVQA omnidirectional AV (higher=better) · ↑ higher=better

**[`oavqa`](src/ayase/modules/oavqa.py)** — OAVQA omnidirectional audio-visual QA (2024)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Backend**: resnet
- **Packages**: gc, torch, torchaudio, torchvision
- **VRAM**: ~200 MB
- **Tests**: covered by [`test_oavqa.py`](tests/modules/per_module/test_oavqa.py)
- **Config**: `subsample=8`, `n_mels=64`, `audio_sr=16000`

### `p1203_mos` [↑](#categories)
> ITU-T P.1203 streaming QoE MOS (1-5) · 1-5

**[`p1203`](src/ayase/modules/p1203.py)** — ITU-T P.1203 streaming QoE estimation (1-5 MOS)

- **Input**: vid · **Speed**: ⚡ fast
- **Backend**: official → parametric
- **Packages**: itu_p1203
- **Tests**: covered by [`test_p1203.py`](tests/modules/per_module/test_p1203.py), [`test_industry_metrics.py`](tests/modules/test_industry_metrics.py)
- **Config**: `display_size=phone`

### `pesq_score` [↑](#categories)
> PESQ (-0.5 to 4.5, higher=better) · ↑ higher=better · -0.5 to 4.5

**[`audio_pesq`](src/ayase/modules/audio_pesq.py)** — PESQ speech quality (full-reference, ITU-T P.862)

- **Input**: audio +ref · **Speed**: ⚡ fast
- **Packages**: librosa, pesq, soundfile
- **Tests**: covered by [`test_audio_pesq.py`](tests/modules/per_module/test_audio_pesq.py), [`test_ml_basics.py`](tests/modules/test_ml_basics.py)
- **Config**: `target_sr=16000`, `warning_threshold=3.0`

### `si_sdr_score` [↑](#categories)
> Scale-Invariant SDR (dB, higher=better) · ↑ higher=better · dB

**[`audio_si_sdr`](src/ayase/modules/audio_si_sdr.py)** — Scale-Invariant SDR for audio quality (full-reference)

- **Input**: audio +ref · **Speed**: ⚡ fast
- **Packages**: librosa, soundfile
- **Tests**: covered by [`test_audio_si_sdr.py`](tests/modules/per_module/test_audio_si_sdr.py), [`test_audio_metrics.py`](tests/test_audio_metrics.py)
- **Config**: `target_sr=16000`, `warning_threshold=0.0`

### `song_eval_clarity` [↑](#categories)
> SongEval clarity of song structure (1-5, higher=better) · ↑ higher=better · 1-5

**[`song_eval`](src/ayase/modules/song_eval.py)** — SongEval song aesthetic evaluation — Coherence, Musicality, Memorability, Clarity, Naturalness (1-5)

- **Input**: audio · **Speed**: ⏱️ medium · GPU
- **Backend**: songeval
- **Packages**: librosa, muq, safetensors, torch
- **Source**: <a href="https://github.com/ASLP-lab/SongEval" target="_blank">GitHub</a> · <a href="https://huggingface.co/song_eval/model.safetensors" target="_blank">HF</a>
- **Tests**: covered by [`test_song_eval.py`](tests/modules/per_module/test_song_eval.py)
- **Config**: `sample_rate=24000`, `checkpoint_subpath=song_eval/model.safetensors`

### `song_eval_coherence` [↑](#categories)
> SongEval overall coherence (1-5, higher=better) · ↑ higher=better · 1-5

**[`song_eval`](src/ayase/modules/song_eval.py)** — SongEval song aesthetic evaluation — Coherence, Musicality, Memorability, Clarity, Naturalness (1-5)

- **Input**: audio · **Speed**: ⏱️ medium · GPU
- **Backend**: songeval
- **Packages**: librosa, muq, safetensors, torch
- **Source**: <a href="https://github.com/ASLP-lab/SongEval" target="_blank">GitHub</a> · <a href="https://huggingface.co/song_eval/model.safetensors" target="_blank">HF</a>
- **Tests**: covered by [`test_song_eval.py`](tests/modules/per_module/test_song_eval.py)
- **Config**: `sample_rate=24000`, `checkpoint_subpath=song_eval/model.safetensors`

### `song_eval_memorability` [↑](#categories)
> SongEval memorability (1-5, higher=better) · ↑ higher=better · 1-5

**[`song_eval`](src/ayase/modules/song_eval.py)** — SongEval song aesthetic evaluation — Coherence, Musicality, Memorability, Clarity, Naturalness (1-5)

- **Input**: audio · **Speed**: ⏱️ medium · GPU
- **Backend**: songeval
- **Packages**: librosa, muq, safetensors, torch
- **Source**: <a href="https://github.com/ASLP-lab/SongEval" target="_blank">GitHub</a> · <a href="https://huggingface.co/song_eval/model.safetensors" target="_blank">HF</a>
- **Tests**: covered by [`test_song_eval.py`](tests/modules/per_module/test_song_eval.py)
- **Config**: `sample_rate=24000`, `checkpoint_subpath=song_eval/model.safetensors`

### `song_eval_musicality` [↑](#categories)
> SongEval overall musicality (1-5, higher=better) · ↑ higher=better · 1-5

**[`song_eval`](src/ayase/modules/song_eval.py)** — SongEval song aesthetic evaluation — Coherence, Musicality, Memorability, Clarity, Naturalness (1-5)

- **Input**: audio · **Speed**: ⏱️ medium · GPU
- **Backend**: songeval
- **Packages**: librosa, muq, safetensors, torch
- **Source**: <a href="https://github.com/ASLP-lab/SongEval" target="_blank">GitHub</a> · <a href="https://huggingface.co/song_eval/model.safetensors" target="_blank">HF</a>
- **Tests**: covered by [`test_song_eval.py`](tests/modules/per_module/test_song_eval.py)
- **Config**: `sample_rate=24000`, `checkpoint_subpath=song_eval/model.safetensors`

### `song_eval_naturalness` [↑](#categories)
> SongEval vocal breathing/phrasing naturalness (1-5, higher=better) · ↑ higher=better · 1-5

**[`song_eval`](src/ayase/modules/song_eval.py)** — SongEval song aesthetic evaluation — Coherence, Musicality, Memorability, Clarity, Naturalness (1-5)

- **Input**: audio · **Speed**: ⏱️ medium · GPU
- **Backend**: songeval
- **Packages**: librosa, muq, safetensors, torch
- **Source**: <a href="https://github.com/ASLP-lab/SongEval" target="_blank">GitHub</a> · <a href="https://huggingface.co/song_eval/model.safetensors" target="_blank">HF</a>
- **Tests**: covered by [`test_song_eval.py`](tests/modules/per_module/test_song_eval.py)
- **Config**: `sample_rate=24000`, `checkpoint_subpath=song_eval/model.safetensors`

### `utmos_score` [↑](#categories)
> UTMOS predicted MOS (1-5, higher=better) · ↑ higher=better · 1-5

**[`audio_utmos`](src/ayase/modules/audio_utmos.py)** — UTMOS no-reference MOS prediction for speech quality

- **Input**: audio · **Speed**: ⏱️ medium · GPU
- **Packages**: librosa, soundfile, torch
- **Tests**: covered by [`test_audio_utmos.py`](tests/modules/per_module/test_audio_utmos.py), [`test_audio_metrics.py`](tests/test_audio_metrics.py)
- **Config**: `target_sr=16000`, `warning_threshold=3.0`

### `visqol` [↑](#categories)
> ViSQOL audio quality MOS (1-5, higher=better) · ↑ higher=better · 1-5

**[`visqol`](src/ayase/modules/visqol.py)** — ViSQOL audio quality MOS (Google, 1-5, higher=better)

- **Input**: img/vid +ref · **Speed**: ⚡ fast
- **Backend**: python → cli
- **Packages**: visqol
- **Source**: <a href="https://github.com/google/visqol" target="_blank">GitHub</a>
- **Tests**: covered by [`test_visqol.py`](tests/modules/per_module/test_visqol.py), [`test_industry_metrics.py`](tests/modules/test_industry_metrics.py)
- **Config**: `mode=audio`


## Face & Identity (19 metrics)

### `celebrity_id_score` [↑](#categories)
> ↑ higher=better

**[`celebrity_id`](src/ayase/modules/celebrity_id.py)** — Face identity verification using DeepFace (EvalCrafter celebrity_id_score)

- **Input**: img/vid · **Speed**: ⚡ fast
- **Packages**: Pillow, deepface, glob
- **Tests**: covered by [`test_celebrity_id.py`](tests/modules/per_module/test_celebrity_id.py)
- **Config**: `reference_dir=`, `num_frames=8`, `consistency_threshold=0.4`, `model_name=VGG-Face`

### `concept_face_count` [↑](#categories)
> Number of faces detected · type: int

**[`concept_presence`](src/ayase/modules/concept_presence.py)** — Detect concept presence via face detection, CLIP-based object/style detection

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Packages**: insightface, mediapipe, torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_concept_presence.py`](tests/modules/per_module/test_concept_presence.py)
- **Config**: `detection_mode=auto`, `clip_model=openai/clip-vit-base-patch32`, `clip_threshold=0.25`, `face_detection_confidence=0.5`, `concepts=[]`, `num_frames=5`

### `crfiqa_score` [↑](#categories)
> CR-FIQA classifiability (higher=better) · ↑ higher=better

**[`crfiqa`](src/ayase/modules/crfiqa.py)** — CR-FIQA face quality via classifiability (CVPR 2023)

- **Input**: img/vid · **Speed**: ⚡ fast
- **Packages**: gc, insightface
- **Tests**: covered by [`test_crfiqa.py`](tests/modules/per_module/test_crfiqa.py)
- **Config**: `subsample=4`, `face_model=buffalo_l`, `det_size=640`, `norm_min=10.0`, `norm_max=30.0`

### `dino_face_identity` [↑](#categories)
> DINOv2 face identity cosine similarity (0-1, higher=better) · ↑ higher=better · 0-1

**[`dino_face_identity`](src/ayase/modules/dino_face_identity.py)** — Face identity similarity using DINOv2 on face crops (better than ArcFace for AI-generated)

- **Input**: img/vid +ref · **Speed**: ⏱️ medium · GPU
- **Packages**: gc, insightface, torch, torchvision
- **VRAM**: ~400 MB
- **Source**: <a href="https://huggingface.co/facebookresearch/dinov2" target="_blank">HF</a>
- **Tests**: covered by [`test_dino_face_identity.py`](tests/modules/per_module/test_dino_face_identity.py)
- **Config**: `model_name=dinov2_vitb14`, `face_model=buffalo_l`, `subsample=8`, `face_margin=0.3`, `warning_threshold=0.3`

### `dino_face_identity_max` [↑](#categories)
> Max DINOv2 face identity across frames (0-1, higher=better) · ↑ higher=better · 0-1

**[`dino_face_identity`](src/ayase/modules/dino_face_identity.py)** — Face identity similarity using DINOv2 on face crops (better than ArcFace for AI-generated)

- **Input**: img/vid +ref · **Speed**: ⏱️ medium · GPU
- **Packages**: gc, insightface, torch, torchvision
- **VRAM**: ~400 MB
- **Source**: <a href="https://huggingface.co/facebookresearch/dinov2" target="_blank">HF</a>
- **Tests**: covered by [`test_dino_face_identity.py`](tests/modules/per_module/test_dino_face_identity.py)
- **Config**: `model_name=dinov2_vitb14`, `face_model=buffalo_l`, `subsample=8`, `face_margin=0.3`, `warning_threshold=0.3`

### `face_consistency` [↑](#categories)
> ↑ higher=better

**[`clip_temporal`](src/ayase/modules/clip_temporal.py)** — CLIP temporal consistency + face/identity consistency (EvalCrafter clip_temp & face_consistency)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Packages**: Pillow, torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_clip_temporal.py`](tests/modules/per_module/test_clip_temporal.py)
- **Config**: `model_name=openai/clip-vit-base-patch32`, `max_frames=32`, `temp_threshold=0.9`, `face_threshold=0.85`

### `face_count` [↑](#categories)
> type: int

**[`face_fidelity`](src/ayase/modules/face_fidelity.py)** — Face detection and per-face quality assessment

- **Input**: img/vid · **Speed**: ⚡ fast
- **Packages**: mediapipe
- **Tests**: covered by [`test_face_fidelity.py`](tests/modules/per_module/test_face_fidelity.py), [`test_face_modules.py`](tests/modules/test_face_modules.py)
- **Config**: `backend=haar`, `subsample=5`, `max_frames=60`, `min_face_size=64`, `blur_threshold=50.0`, `warning_threshold=40.0`

### `face_cross_similarity` [↑](#categories)
> Avg pairwise face similarity (0-1, higher=more consistent) · ↑ higher=better

**[`face_cross_similarity`](src/ayase/modules/face_cross_similarity.py)** — Pairwise ArcFace cosine similarity matrix across dataset faces

- **Input**: img/vid · **Speed**: ⚡ fast
- **Backend**: insightface → deepface → mediapipe
- **Packages**: Pillow, deepface, insightface, mediapipe
- **Tests**: covered by [`test_face_cross_similarity.py`](tests/modules/per_module/test_face_cross_similarity.py)
- **Config**: `model_name=buffalo_l`, `max_faces_per_image=5`, `similarity_threshold=0.3`, `subsample=8`, `max_cache_size=10000`, `device=auto`

### `face_expression_smoothness` [↑](#categories)

**[`face_landmark_quality`](src/ayase/modules/face_landmark_quality.py)** — Facial landmark jitter, expression smoothness, identity consistency

- **Input**: vid · **Speed**: ⚡ fast
- **Packages**: mediapipe
- **Tests**: covered by [`test_face_landmark_quality.py`](tests/modules/per_module/test_face_landmark_quality.py), [`test_face_modules.py`](tests/modules/test_face_modules.py)
- **Config**: `subsample=2`, `max_frames=300`, `jitter_warning=30.0`

### `face_identity_consistency` [↑](#categories)
> Temporal face identity stability (0-1) · ↑ higher=better · 0-1

**[`face_landmark_quality`](src/ayase/modules/face_landmark_quality.py)** — Facial landmark jitter, expression smoothness, identity consistency

- **Input**: vid · **Speed**: ⚡ fast
- **Packages**: mediapipe
- **Tests**: covered by [`test_face_landmark_quality.py`](tests/modules/per_module/test_face_landmark_quality.py), [`test_face_modules.py`](tests/modules/test_face_modules.py)
- **Config**: `subsample=2`, `max_frames=300`, `jitter_warning=30.0`

### `face_identity_count` [↑](#categories)
> Number of unique identities detected · type: int

**[`face_cross_similarity`](src/ayase/modules/face_cross_similarity.py)** — Pairwise ArcFace cosine similarity matrix across dataset faces

- **Input**: img/vid · **Speed**: ⚡ fast
- **Backend**: insightface → deepface → mediapipe
- **Packages**: Pillow, deepface, insightface, mediapipe
- **Tests**: covered by [`test_face_cross_similarity.py`](tests/modules/per_module/test_face_cross_similarity.py)
- **Config**: `model_name=buffalo_l`, `max_faces_per_image=5`, `similarity_threshold=0.3`, `subsample=8`, `max_cache_size=10000`, `device=auto`

### `face_iqa_score` [↑](#categories)
> TOPIQ-face face quality (higher=better) · ↑ higher=better

**[`face_iqa`](src/ayase/modules/face_iqa.py)** — Face-specific IQA via TOPIQ-face (GFIQA-trained, higher=better)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: opencv-python, pyiqa, torch
- **Tests**: covered by [`test_face_iqa.py`](tests/modules/per_module/test_face_iqa.py), [`test_iqa_research_metrics.py`](tests/modules/test_iqa_research_metrics.py)
- **Config**: `subsample=8`

### `face_landmark_jitter` [↑](#categories)
> Landmark jitter 0-100 (lower=better) · ↓ lower=better

**[`face_landmark_quality`](src/ayase/modules/face_landmark_quality.py)** — Facial landmark jitter, expression smoothness, identity consistency

- **Input**: vid · **Speed**: ⚡ fast
- **Packages**: mediapipe
- **Tests**: covered by [`test_face_landmark_quality.py`](tests/modules/per_module/test_face_landmark_quality.py), [`test_face_modules.py`](tests/modules/test_face_modules.py)
- **Config**: `subsample=2`, `max_frames=300`, `jitter_warning=30.0`

### `face_quality_score` [↑](#categories)
> Composite face quality 0-100 (higher=better) · ↑ higher=better

**[`face_fidelity`](src/ayase/modules/face_fidelity.py)** — Face detection and per-face quality assessment

- **Input**: img/vid · **Speed**: ⚡ fast
- **Packages**: mediapipe
- **Tests**: covered by [`test_face_fidelity.py`](tests/modules/per_module/test_face_fidelity.py), [`test_face_modules.py`](tests/modules/test_face_modules.py)
- **Config**: `backend=haar`, `subsample=5`, `max_frames=60`, `min_face_size=64`, `blur_threshold=50.0`, `warning_threshold=40.0`

### `face_recognition_score` [↑](#categories)
> Face identity cosine similarity (0-1, higher=better) · ↑ higher=better · 0-1

**[`identity_loss`](src/ayase/modules/identity_loss.py)** — Face identity preservation metric (cosine distance/similarity vs reference)

- **Input**: img/vid +ref · **Speed**: ⚡ fast
- **Backend**: insightface → deepface → mediapipe
- **Packages**: Pillow, deepface, insightface, mediapipe
- **Tests**: covered by [`test_identity_loss.py`](tests/modules/per_module/test_identity_loss.py), [`test_identity_loss.py`](tests/modules/test_identity_loss.py)
- **Config**: `model_name=buffalo_l`, `subsample=8`, `warning_threshold=0.5`

### `grafiqs_score` [↑](#categories)
> GraFIQs gradient-based (higher=better) · ↑ higher=better

**[`grafiqs`](src/ayase/modules/grafiqs.py)** — GraFIQs gradient face quality (CVPRW 2024)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: gc, insightface, the, torch, torchvision
- **VRAM**: ~200 MB
- **Tests**: covered by [`test_grafiqs.py`](tests/modules/per_module/test_grafiqs.py)
- **Config**: `subsample=4`, `face_model=buffalo_l`, `det_size=640`, `gradient_scale=10000.0`

### `identity_loss` [↑](#categories)
> Face identity cosine distance (0-1, lower=better) · ↓ lower=better · 0-1

**[`identity_loss`](src/ayase/modules/identity_loss.py)** — Face identity preservation metric (cosine distance/similarity vs reference)

- **Input**: img/vid +ref · **Speed**: ⚡ fast
- **Backend**: insightface → deepface → mediapipe
- **Packages**: Pillow, deepface, insightface, mediapipe
- **Tests**: covered by [`test_identity_loss.py`](tests/modules/per_module/test_identity_loss.py), [`test_identity_loss.py`](tests/modules/test_identity_loss.py)
- **Config**: `model_name=buffalo_l`, `subsample=8`, `warning_threshold=0.5`

### `magface_score` [↑](#categories)
> MagFace magnitude quality (higher=better) · ↑ higher=better

**[`magface`](src/ayase/modules/magface.py)** — MagFace face magnitude quality (CVPR 2021)

- **Input**: img/vid · **Speed**: ⚡ fast
- **Packages**: gc, insightface
- **Tests**: covered by [`test_magface.py`](tests/modules/per_module/test_magface.py)
- **Config**: `subsample=4`, `face_model=buffalo_l`, `det_size=640`, `norm_min=10.0`, `norm_max=30.0`

### `serfiq_score` [↑](#categories)
> SER-FIQ embedding robustness (higher=better) · ↑ higher=better

**[`serfiq`](src/ayase/modules/serfiq.py)** — SER-FIQ face quality via embedding robustness (2020)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: gc, insightface, torch, torchvision
- **VRAM**: ~200 MB
- **Tests**: covered by [`test_serfiq.py`](tests/modules/per_module/test_serfiq.py)
- **Config**: `subsample=4`, `face_model=buffalo_l`, `n_forward_passes=10`, `noise_std=5.0`, `det_size=640`, `dropout_rate=0.1`


## Scene & Content (16 metrics)

### `action_confidence` [↑](#categories)
> Top-1 action confidence (0-100) · 0-100

**[`action_recognition`](src/ayase/modules/action_recognition.py)** — Recognizes human actions (VideoMAE / UMT) - Supports Heavy Models

- **Input**: vid +cap · **Speed**: ⏱️ medium · GPU
- **Packages**: open-clip-torch, torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/MCG-NJU/videomae-large-finetuned-kinetics" target="_blank">HF</a>
- **Tests**: covered by [`test_action_recognition.py`](tests/modules/per_module/test_action_recognition.py), [`test_integration_synthetic.py`](tests/test_integration_synthetic.py)
- **Config**: `model_name=MCG-NJU/videomae-large-finetuned-kinetics`, `caption_matching=False`, `matching_mode=weighted`, `clip_model=openai/clip-vit-base-patch32`, `top_k=5`

### `action_score` [↑](#categories)
> Caption-action fidelity (0-100) · ↑ higher=better · 0-100

**[`action_recognition`](src/ayase/modules/action_recognition.py)** — Recognizes human actions (VideoMAE / UMT) - Supports Heavy Models

- **Input**: vid +cap · **Speed**: ⏱️ medium · GPU
- **Packages**: open-clip-torch, torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/MCG-NJU/videomae-large-finetuned-kinetics" target="_blank">HF</a>
- **Tests**: covered by [`test_action_recognition.py`](tests/modules/per_module/test_action_recognition.py), [`test_integration_synthetic.py`](tests/test_integration_synthetic.py)
- **Config**: `model_name=MCG-NJU/videomae-large-finetuned-kinetics`, `caption_matching=False`, `matching_mode=weighted`, `clip_model=openai/clip-vit-base-patch32`, `top_k=5`

### `avg_scene_duration` [↑](#categories)
> Average scene duration in seconds

**[`scene_detection`](src/ayase/modules/scene_detection.py)** — Scene stability metric — penalises rapid cuts (0-1, higher=more stable)

- **Input**: vid · **Speed**: ⚡ fast
- **Packages**: opencv-python, transnetv2
- **Tests**: covered by [`test_scene_detection.py`](tests/modules/per_module/test_scene_detection.py), [`test_motion_scene_semantic_metrics.py`](tests/modules/test_motion_scene_semantic_metrics.py)
- **Config**: `threshold=0.5`

### `color_score` [↑](#categories)
> ↑ higher=better

**[`color_consistency`](src/ayase/modules/color_consistency.py)** — Verifies color attributes in prompt vs video content

- **Input**: img/vid +cap · **Speed**: ⚡ fast
- **Tests**: covered by [`test_color_consistency.py`](tests/modules/per_module/test_color_consistency.py)

### `commonsense_score` [↑](#categories)
> Common sense adherence (0-1, higher=better) · ↑ higher=better · 0-1

**[`commonsense`](src/ayase/modules/commonsense.py)** — Common sense adherence (VLM / ViLT VQA)

- **Input**: img/vid · **Speed**: 🐌 slow · GPU
- **Backend**: vlm → vilt
- **Packages**: Pillow, torch, transformers
- **VRAM**: ~14 GB
- **Source**: <a href="https://huggingface.co/dandelin/vilt-b32-finetuned-vqa" target="_blank">HF</a>
- **Tests**: covered by [`test_commonsense.py`](tests/modules/per_module/test_commonsense.py), [`test_vbench2_compbench.py`](tests/modules/test_vbench2_compbench.py)
- **Config**: `model_name=dandelin/vilt-b32-finetuned-vqa`, `vlm_model=llava-hf/llava-1.5-7b-hf`

### `concept_count` [↑](#categories)
> Number of detected instances of target concept · type: int

**[`concept_presence`](src/ayase/modules/concept_presence.py)** — Detect concept presence via face detection, CLIP-based object/style detection

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Packages**: insightface, mediapipe, torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_concept_presence.py`](tests/modules/per_module/test_concept_presence.py)
- **Config**: `detection_mode=auto`, `clip_model=openai/clip-vit-base-patch32`, `clip_threshold=0.25`, `face_detection_confidence=0.5`, `concepts=[]`, `num_frames=5`

### `concept_presence` [↑](#categories)
> Concept presence confidence (0-1, higher=more confident) · 0-1, higher=more confident

**[`concept_presence`](src/ayase/modules/concept_presence.py)** — Detect concept presence via face detection, CLIP-based object/style detection

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Packages**: insightface, mediapipe, torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_concept_presence.py`](tests/modules/per_module/test_concept_presence.py)
- **Config**: `detection_mode=auto`, `clip_model=openai/clip-vit-base-patch32`, `clip_threshold=0.25`, `face_detection_confidence=0.5`, `concepts=[]`, `num_frames=5`

### `count_score` [↑](#categories)
> ↑ higher=better

**[`object_detection`](src/ayase/modules/object_detection.py)** — Detects objects (GRiT / YOLOv8) - Supports Heavy Models

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: grit, torch, ultralytics
- **Tests**: covered by [`test_object_detection.py`](tests/modules/per_module/test_object_detection.py)
- **Config**: `model_name=yolov8n.pt`, `use_yolo_world=False`, `use_grit=False`

### `detection_diversity` [↑](#categories)
> Object detection category entropy

**[`object_detection`](src/ayase/modules/object_detection.py)** — Detects objects (GRiT / YOLOv8) - Supports Heavy Models

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: grit, torch, ultralytics
- **Tests**: covered by [`test_object_detection.py`](tests/modules/per_module/test_object_detection.py)
- **Config**: `model_name=yolov8n.pt`, `use_yolo_world=False`, `use_grit=False`

### `detection_score` [↑](#categories)
> ↑ higher=better

**[`object_detection`](src/ayase/modules/object_detection.py)** — Detects objects (GRiT / YOLOv8) - Supports Heavy Models

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: grit, torch, ultralytics
- **Tests**: covered by [`test_object_detection.py`](tests/modules/per_module/test_object_detection.py)
- **Config**: `model_name=yolov8n.pt`, `use_yolo_world=False`, `use_grit=False`

### `gradient_detail` [↑](#categories)
> Sobel gradient detail (0-100) · 0-100

**[`basic_quality`](src/ayase/modules/basic.py)** — Comprehensive technical quality assessment (blur, noise, artifacts, contrast)

- **Input**: img/vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_basic_quality.py`](tests/modules/per_module/test_basic_quality.py), [`test_docs_integrity.py`](tests/test_docs_integrity.py), [`test_profiles.py`](tests/test_profiles.py), +3 more
- **Config**: `threshold=40.0`, `blur_threshold=100.0`, `noise_threshold=50.0`

### `human_fidelity_score` [↑](#categories)
> Body/hand/face quality (0-1, higher=better) · ↑ higher=better · 0-1

**[`human_fidelity`](src/ayase/modules/human_fidelity.py)** — Human body/hand/face fidelity (DWPose / MediaPipe)

- **Input**: img/vid · **Speed**: ⚡ fast
- **Backend**: dwpose → mediapipe
- **Packages**: dwpose, mediapipe
- **Tests**: covered by [`test_human_fidelity.py`](tests/modules/per_module/test_human_fidelity.py), [`test_vbench2_compbench.py`](tests/modules/test_vbench2_compbench.py)

### `ram_tags` [↑](#categories)
> Comma-separated RAM auto-tags · type: str

**[`ram_tagging`](src/ayase/modules/ram_tagging.py)** — RAM (Recognize Anything Model) auto-tagging for video frames

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: Pillow, opencv-python, torch, transformers
- **Source**: <a href="https://huggingface.co/xinyu1205/recognize-anything-plus-model" target="_blank">HF</a>
- **Tests**: covered by [`test_ram_tagging.py`](tests/modules/per_module/test_ram_tagging.py), [`test_motion_scene_semantic_metrics.py`](tests/modules/test_motion_scene_semantic_metrics.py)
- **Config**: `model_name=xinyu1205/recognize-anything-plus-model`, `subsample=4`, `trust_remote_code=False`

### `scene_complexity` [↑](#categories)
> Visual complexity score

**[`scene_complexity`](src/ayase/modules/scene_complexity.py)** — Spatial and temporal scene complexity analysis

- **Input**: vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_scene_complexity.py`](tests/modules/per_module/test_scene_complexity.py), [`test_fields_general.py`](tests/modules/test_fields_general.py), [`test_opencv_modules.py`](tests/modules/test_opencv_modules.py), +2 more
- **Config**: `subsample=2`, `spatial_weight=0.5`, `temporal_weight=0.5`

### `video_type` [↑](#categories)
> Content type (real, animated, game, etc.) · type: str

**[`video_type_classifier`](src/ayase/modules/video_type_classifier.py)** — CLIP zero-shot video content type classification

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: Pillow, opencv-python, torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_video_type_classifier.py`](tests/modules/per_module/test_video_type_classifier.py), [`test_motion_scene_semantic_metrics.py`](tests/modules/test_motion_scene_semantic_metrics.py)
- **Config**: `subsample=4`

### `video_type_confidence` [↑](#categories)
> Classification confidence

**[`video_type_classifier`](src/ayase/modules/video_type_classifier.py)** — CLIP zero-shot video content type classification

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: Pillow, opencv-python, torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_video_type_classifier.py`](tests/modules/per_module/test_video_type_classifier.py), [`test_motion_scene_semantic_metrics.py`](tests/modules/test_motion_scene_semantic_metrics.py)
- **Config**: `subsample=4`


## Distribution & Generation (1 metrics)

### `is_score` [↑](#categories)
> ↑ higher=better

**[`inception_score`](src/ayase/modules/inception_score.py)** — Inception Score (IS) using InceptionV3 — EvalCrafter quality metric

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: torch, torchvision
- **VRAM**: ~200 MB
- **Tests**: covered by [`test_inception_score.py`](tests/modules/per_module/test_inception_score.py)
- **Config**: `num_frames=16`, `splits=1`


## HDR & Color (13 metrics)

### `brightrate_score` [↑](#categories)
> BrightRate HDR UGC NR-VQA (higher=better) · ↑ higher=better

**[`brightrate`](src/ayase/modules/brightrate.py)** — BrightRate HDR no-reference video quality via official BrightVQ inference script

- **Input**: vid · **Speed**: ⏱️ medium
- **Backend**: brightrate
- **Packages**: imageio_ffmpeg, joblib, numba, pandas, pyiqa, scikit-learn, scipy, torch, torchvision
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/CLIP/clip_feats.py" target="_blank">HF</a>
- **Tests**: covered by [`test_brightrate.py`](tests/modules/per_module/test_brightrate.py)
- **Config**: `timeout_sec=3600`, `num_frames=30`, `num_workers=1`, `parallel_level=video`, `ffmpeg_path=`, `read_yuv=False`

### `delta_ictcp` [↑](#categories)
> Delta ICtCp HDR color difference (lower=better) · ↓ lower=better

**[`delta_ictcp`](src/ayase/modules/delta_ictcp.py)** — Delta ICtCp HDR perceptual color difference (lower=better)

- **Input**: img/vid +ref · **Speed**: ⚡ fast
- **Tests**: covered by [`test_delta_ictcp.py`](tests/modules/per_module/test_delta_ictcp.py), [`test_industry_metrics.py`](tests/modules/test_industry_metrics.py)
- **Config**: `subsample=5`

### `hdr_chipqa_score` [↑](#categories)
> HDR-ChipQA HDR NR-VQA (higher=better) · ↑ higher=better

**[`hdr_chipqa`](src/ayase/modules/hdr_chipqa.py)** — HDR-ChipQA no-reference HDR video quality via official feature extractor and LIVE-HDR SVR

- **Input**: vid · **Speed**: ⚡ fast
- **Backend**: hdr_chipqa
- **Packages**: joblib, matplotlib, numba, opencv-python, scikit-learn, scipy
- **Source**: <a href="https://huggingface.co/utils/colour_utils.py" target="_blank">HF</a>
- **Tests**: covered by [`test_hdr_chipqa.py`](tests/modules/per_module/test_hdr_chipqa.py)
- **Config**: `timeout_sec=1800`, `width=3840`, `height=2160`, `bit_depth=10`, `color_space=BT2020`

### `hdr_quality` [↑](#categories)
> HDR-specific quality · ↑ higher=better

**[`hdr_sdr_vqa`](src/ayase/modules/hdr_sdr_vqa.py)** — HDR/SDR-aware video quality assessment

- **Input**: vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_4k_vqa.py`](tests/modules/per_module/test_4k_vqa.py), [`test_hdr_sdr_vqa.py`](tests/modules/per_module/test_hdr_sdr_vqa.py), [`test_reference_and_meta_metrics.py`](tests/modules/test_reference_and_meta_metrics.py), +3 more
- **Config**: `subsample=5`

### `hdr_technical_score` [↑](#categories)
> HDR/SDR-aware technical quality (0-1) · ↑ higher=better · 0-1

**[`4k_vqa`](src/ayase/modules/hdr_sdr_vqa.py)** — Memory-efficient quality assessment for 4K+ videos

- **Input**: vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_4k_vqa.py`](tests/modules/per_module/test_4k_vqa.py), [`test_docs_integrity.py`](tests/test_docs_integrity.py)
- **Config**: `tile_size=512`, `subsample=10`

### `hdr_vdp` [↑](#categories)
> HDR-VDP visual difference predictor (higher=better) · ↑ higher=better

**[`hdr_vdp`](src/ayase/modules/hdr_vdp.py)** — HDR-VDP visual difference predictor (higher=better)

- **Input**: img/vid +ref · **Speed**: ⚡ fast
- **Backend**: python → approx
- **Packages**: hdrvdp
- **Tests**: covered by [`test_hdr_vdp.py`](tests/modules/per_module/test_hdr_vdp.py), [`test_industry_metrics.py`](tests/modules/test_industry_metrics.py)
- **Config**: `subsample=5`

### `hdr_vqm` [↑](#categories)
> HDR-VQM HDR video quality FR

**[`hdr_vqm`](src/ayase/modules/hdr_vqm.py)** — HDR-aware video quality (PU21+wavelet FR or gamma heuristic fallback)

- **Input**: img/vid +ref · **Speed**: ⚡ fast
- **Backend**: gamma_heuristic → pu21_wavelet
- **Packages**: PyWavelets, opencv-python
- **Tests**: covered by [`test_hdr_vqm.py`](tests/modules/per_module/test_hdr_vqm.py), [`test_video_native_fields.py`](tests/modules/test_video_native_fields.py), [`test_video_native_metrics.py`](tests/modules/test_video_native_metrics.py)
- **Config**: `subsample=8`

### `hdrmax_score` [↑](#categories)
> HDRMAX / HDR-VMAF family score (higher=better) · ↑ higher=better

**[`hdrmax`](src/ayase/modules/hdrmax.py)** — HDRMAX full-reference HDR video quality via official HDRMAX feature and prediction scripts

- **Input**: vid +ref · **Speed**: ⚡ fast
- **Backend**: hdrmax
- **Packages**: PyWavelets, colour-science, joblib, matplotlib, pandas, pyrtools, scikit-image, scipy
- **Tests**: covered by [`test_hdrmax.py`](tests/modules/per_module/test_hdrmax.py)
- **Config**: `mode=hdrvmaf`, `timeout_sec=3600`, `ffmpeg_bin=ffmpeg`, `njobs=1`

### `max_cll` [↑](#categories)
> MaxCLL content light level (nits)

**[`hdr_metadata`](src/ayase/modules/hdr_metadata.py)** — MaxFALL + MaxCLL HDR static metadata analysis

- **Input**: vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_hdr_metadata.py`](tests/modules/per_module/test_hdr_metadata.py), [`test_industry_metrics.py`](tests/modules/test_industry_metrics.py)
- **Config**: `subsample=3`, `peak_nits=10000.0`

### `max_fall` [↑](#categories)
> MaxFALL frame average light level (nits)

**[`hdr_metadata`](src/ayase/modules/hdr_metadata.py)** — MaxFALL + MaxCLL HDR static metadata analysis

- **Input**: vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_hdr_metadata.py`](tests/modules/per_module/test_hdr_metadata.py), [`test_industry_metrics.py`](tests/modules/test_industry_metrics.py)
- **Config**: `subsample=3`, `peak_nits=10000.0`

### `pu_psnr` [↑](#categories)
> PU-PSNR perceptually uniform HDR (dB, higher=better) · ↑ higher=better · dB

**[`pu_metrics`](src/ayase/modules/pu_metrics.py)** — PU-PSNR + PU-SSIM for HDR content (perceptually uniform)

- **Input**: img/vid +ref · **Speed**: ⚡ fast
- **Tests**: covered by [`test_pu_metrics.py`](tests/modules/per_module/test_pu_metrics.py), [`test_industry_metrics.py`](tests/modules/test_industry_metrics.py)
- **Config**: `subsample=5`, `assume_nits_range=10000.0`

### `pu_ssim` [↑](#categories)
> PU-SSIM perceptually uniform HDR (0-1, higher=better) · ↑ higher=better · 0-1

**[`pu_metrics`](src/ayase/modules/pu_metrics.py)** — PU-PSNR + PU-SSIM for HDR content (perceptually uniform)

- **Input**: img/vid +ref · **Speed**: ⚡ fast
- **Tests**: covered by [`test_pu_metrics.py`](tests/modules/per_module/test_pu_metrics.py), [`test_industry_metrics.py`](tests/modules/test_industry_metrics.py)
- **Config**: `subsample=5`, `assume_nits_range=10000.0`

### `sdr_quality` [↑](#categories)
> SDR-specific quality · ↑ higher=better

**[`hdr_sdr_vqa`](src/ayase/modules/hdr_sdr_vqa.py)** — HDR/SDR-aware video quality assessment

- **Input**: vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_4k_vqa.py`](tests/modules/per_module/test_4k_vqa.py), [`test_hdr_sdr_vqa.py`](tests/modules/per_module/test_hdr_sdr_vqa.py), [`test_reference_and_meta_metrics.py`](tests/modules/test_reference_and_meta_metrics.py), +3 more
- **Config**: `subsample=5`


## Codec & Technical (5 metrics)

### `cambi` [↑](#categories)
> CAMBI banding index (0-24, lower=better) · ↓ lower=better · 0-24

**[`cambi`](src/ayase/modules/cambi.py)** — CAMBI banding/contouring detector (Netflix, 0-24, lower=better)

- **Input**: vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_cambi.py`](tests/modules/per_module/test_cambi.py), [`test_industry_metrics.py`](tests/modules/test_industry_metrics.py)
- **Config**: `warning_threshold=5.0`

### `codec_artifacts` [↑](#categories)
> Block artifact severity 0-100 (lower=better) · ↓ lower=better

**[`codec_specific_quality`](src/ayase/modules/codec_specific_quality.py)** — Codec-level efficiency, GOP quality, and artifact detection

- **Input**: vid · **Speed**: ⚡ fast
- **Source**: <a href="https://huggingface.co/30/1" target="_blank">HF</a>
- **Tests**: covered by [`test_codec_specific_quality.py`](tests/modules/per_module/test_codec_specific_quality.py), [`test_opencv_modules.py`](tests/modules/test_opencv_modules.py)
- **Config**: `max_frames=100`, `subsample=10`, `warning_efficiency=30.0`, `warning_artifacts=40.0`

### `codec_efficiency` [↑](#categories)
> Quality-per-bit efficiency 0-100 (higher=better) · ↑ higher=better

**[`codec_specific_quality`](src/ayase/modules/codec_specific_quality.py)** — Codec-level efficiency, GOP quality, and artifact detection

- **Input**: vid · **Speed**: ⚡ fast
- **Source**: <a href="https://huggingface.co/30/1" target="_blank">HF</a>
- **Tests**: covered by [`test_codec_specific_quality.py`](tests/modules/per_module/test_codec_specific_quality.py), [`test_opencv_modules.py`](tests/modules/test_opencv_modules.py)
- **Config**: `max_frames=100`, `subsample=10`, `warning_efficiency=30.0`, `warning_artifacts=40.0`

### `gop_quality` [↑](#categories)
> GOP structure appropriateness 0-100 (higher=better) · ↑ higher=better

**[`codec_specific_quality`](src/ayase/modules/codec_specific_quality.py)** — Codec-level efficiency, GOP quality, and artifact detection

- **Input**: vid · **Speed**: ⚡ fast
- **Source**: <a href="https://huggingface.co/30/1" target="_blank">HF</a>
- **Tests**: covered by [`test_codec_specific_quality.py`](tests/modules/per_module/test_codec_specific_quality.py), [`test_opencv_modules.py`](tests/modules/test_opencv_modules.py)
- **Config**: `max_frames=100`, `subsample=10`, `warning_efficiency=30.0`, `warning_artifacts=40.0`

### `p1204_mos` [↑](#categories)
> ITU-T P.1204.3 bitstream MOS (1-5) · 1-5

**[`p1204`](src/ayase/modules/p1204.py)** — ITU-T P.1204.3 bitstream NR quality (2020)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Backend**: resnet
- **Packages**: gc, torch, torchvision
- **VRAM**: ~200 MB
- **Tests**: covered by [`test_p1204.py`](tests/modules/per_module/test_p1204.py)
- **Config**: `subsample=4`


## Depth & Spatial (5 metrics)

### `depth_anything_consistency` [↑](#categories)
> Temporal depth consistency · ↑ higher=better

**[`depth_anything`](src/ayase/modules/depth_anything.py)** — Depth Anything V2 monocular depth estimation and consistency

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: Pillow, opencv-python, torch, transformers
- **Source**: <a href="https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf" target="_blank">HF</a>
- **Tests**: covered by [`test_depth_anything.py`](tests/modules/per_module/test_depth_anything.py), [`test_motion_scene_semantic_metrics.py`](tests/modules/test_motion_scene_semantic_metrics.py)
- **Config**: `model_name=depth-anything/Depth-Anything-V2-Small-hf`, `subsample=8`

### `depth_anything_score` [↑](#categories)
> Monocular depth quality · ↑ higher=better

**[`depth_anything`](src/ayase/modules/depth_anything.py)** — Depth Anything V2 monocular depth estimation and consistency

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: Pillow, opencv-python, torch, transformers
- **Source**: <a href="https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf" target="_blank">HF</a>
- **Tests**: covered by [`test_depth_anything.py`](tests/modules/per_module/test_depth_anything.py), [`test_motion_scene_semantic_metrics.py`](tests/modules/test_motion_scene_semantic_metrics.py)
- **Config**: `model_name=depth-anything/Depth-Anything-V2-Small-hf`, `subsample=8`

### `depth_quality` [↑](#categories)
> Depth map quality 0-100 (higher=better) · ↑ higher=better

**[`depth_map_quality`](src/ayase/modules/depth_map_quality.py)** — Monocular depth map quality (sharpness, completeness, edge alignment)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: torch
- **Source**: <a href="https://huggingface.co/intel-isl/MiDaS" target="_blank">HF</a>
- **Tests**: covered by [`test_depth_map_quality.py`](tests/modules/per_module/test_depth_map_quality.py), [`test_depth_and_multiview.py`](tests/modules/test_depth_and_multiview.py)
- **Config**: `model_type=MiDaS_small`, `device=auto`, `subsample=10`, `max_frames=30`

### `multiview_consistency` [↑](#categories)
> Geometric consistency 0-1 (higher=better) · ↑ higher=better

**[`multi_view_consistency`](src/ayase/modules/multi_view_consistency.py)** — Geometric multi-view consistency via epipolar analysis

- **Input**: vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_multi_view_consistency.py`](tests/modules/per_module/test_multi_view_consistency.py), [`test_depth_and_multiview.py`](tests/modules/test_depth_and_multiview.py)
- **Config**: `subsample=5`, `max_pairs=30`, `min_matches=20`

### `stereo_comfort_score` [↑](#categories)
> Stereo viewing comfort 0-100 (higher=better) · ↑ higher=better

**[`stereoscopic_quality`](src/ayase/modules/stereoscopic_quality.py)** — Stereo 3D comfort and quality assessment

- **Input**: vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_stereoscopic_quality.py`](tests/modules/per_module/test_stereoscopic_quality.py), [`test_depth_and_multiview.py`](tests/modules/test_depth_and_multiview.py)
- **Config**: `stereo_format=auto`, `subsample=10`, `max_frames=30`, `max_disparity_percent=3.0`, `warning_threshold=50.0`


## Production Quality (5 metrics)

### `banding_severity` [↑](#categories)
> Colour banding 0-100 (lower=better) · ↓ lower=better

**[`production_quality`](src/ayase/modules/production_quality.py)** — Professional production quality (colour, exposure, focus, banding)

- **Input**: img/vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_production_quality.py`](tests/modules/per_module/test_production_quality.py), [`test_opencv_modules.py`](tests/modules/test_opencv_modules.py)
- **Config**: `max_frames=150`

### `color_grading_score` [↑](#categories)
> Colour consistency 0-100 · ↑ higher=better · 0-100

**[`production_quality`](src/ayase/modules/production_quality.py)** — Professional production quality (colour, exposure, focus, banding)

- **Input**: img/vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_production_quality.py`](tests/modules/per_module/test_production_quality.py), [`test_opencv_modules.py`](tests/modules/test_opencv_modules.py)
- **Config**: `max_frames=150`

### `exposure_consistency` [↑](#categories)
> Exposure stability 0-100 · ↑ higher=better · 0-100

**[`production_quality`](src/ayase/modules/production_quality.py)** — Professional production quality (colour, exposure, focus, banding)

- **Input**: img/vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_production_quality.py`](tests/modules/per_module/test_production_quality.py), [`test_opencv_modules.py`](tests/modules/test_opencv_modules.py)
- **Config**: `max_frames=150`

### `focus_quality` [↑](#categories)
> Sharpness/focus quality 0-100 · ↑ higher=better · 0-100

**[`production_quality`](src/ayase/modules/production_quality.py)** — Professional production quality (colour, exposure, focus, banding)

- **Input**: img/vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_production_quality.py`](tests/modules/per_module/test_production_quality.py), [`test_opencv_modules.py`](tests/modules/test_opencv_modules.py)
- **Config**: `max_frames=150`

### `white_balance_score` [↑](#categories)
> White balance accuracy 0-100 · ↑ higher=better · 0-100

**[`production_quality`](src/ayase/modules/production_quality.py)** — Professional production quality (colour, exposure, focus, banding)

- **Input**: img/vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_production_quality.py`](tests/modules/per_module/test_production_quality.py), [`test_opencv_modules.py`](tests/modules/test_opencv_modules.py)
- **Config**: `max_frames=150`


## OCR & Text (7 metrics)

### `auto_caption` [↑](#categories)
> Generated caption · type: str

**[`captioning`](src/ayase/modules/captioning.py)** — Generates captions using BLIP + computes BLEU score (EvalCrafter blip_bleu)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: Pillow, opencv-python, torch, transformers
- **Source**: <a href="https://huggingface.co/Salesforce/blip-image-captioning-base" target="_blank">HF</a>
- **Tests**: covered by [`test_captioning.py`](tests/modules/per_module/test_captioning.py)
- **Config**: `model_name=Salesforce/blip-image-captioning-base`, `num_frames=5`

### `ocr_area_ratio` [↑](#categories)
> 0-1 · 0-1

**[`text_detection`](src/ayase/modules/text.py)** — Detects text/watermarks using OCR (PaddleOCR / Tesseract)

- **Input**: img/vid · **Speed**: ⚡ fast
- **Packages**: paddleocr, pytesseract
- **Tests**: covered by [`test_text_detection.py`](tests/modules/per_module/test_text_detection.py), [`test_docs_integrity.py`](tests/test_docs_integrity.py)
- **Config**: `use_paddle=True`, `max_text_area=0.05`

### `ocr_cer` [↑](#categories)
> Character Error Rate (0-1, lower=better) · ↓ lower=better · 0-1

**[`ocr_fidelity`](src/ayase/modules/ocr_fidelity.py)** — Checks whether text requested in the caption actually appears in video frames (EvalCrafter OCR)

- **Input**: img/vid · **Speed**: ⚡ fast
- **Packages**: paddleocr
- **Tests**: covered by [`test_ocr_fidelity.py`](tests/modules/per_module/test_ocr_fidelity.py), [`test_integration_synthetic.py`](tests/test_integration_synthetic.py)
- **Config**: `num_frames=8`, `lang=en`

### `ocr_fidelity` [↑](#categories)
> OCR text accuracy vs caption (0-100, higher=better) · ↑ higher=better · 0-100

**[`ocr_fidelity`](src/ayase/modules/ocr_fidelity.py)** — Checks whether text requested in the caption actually appears in video frames (EvalCrafter OCR)

- **Input**: img/vid · **Speed**: ⚡ fast
- **Packages**: paddleocr
- **Tests**: covered by [`test_ocr_fidelity.py`](tests/modules/per_module/test_ocr_fidelity.py), [`test_integration_synthetic.py`](tests/test_integration_synthetic.py)
- **Config**: `num_frames=8`, `lang=en`

### `ocr_score` [↑](#categories)
> ↑ higher=better

**[`ocr_fidelity`](src/ayase/modules/ocr_fidelity.py)** — Checks whether text requested in the caption actually appears in video frames (EvalCrafter OCR)

- **Input**: img/vid · **Speed**: ⚡ fast
- **Packages**: paddleocr
- **Tests**: covered by [`test_ocr_fidelity.py`](tests/modules/per_module/test_ocr_fidelity.py), [`test_integration_synthetic.py`](tests/test_integration_synthetic.py)
- **Config**: `num_frames=8`, `lang=en`

### `ocr_wer` [↑](#categories)
> Word Error Rate (0-1, lower=better) · ↓ lower=better · 0-1

**[`ocr_fidelity`](src/ayase/modules/ocr_fidelity.py)** — Checks whether text requested in the caption actually appears in video frames (EvalCrafter OCR)

- **Input**: img/vid · **Speed**: ⚡ fast
- **Packages**: paddleocr
- **Tests**: covered by [`test_ocr_fidelity.py`](tests/modules/per_module/test_ocr_fidelity.py), [`test_integration_synthetic.py`](tests/test_integration_synthetic.py)
- **Config**: `num_frames=8`, `lang=en`

### `text_overlay_score` [↑](#categories)
> Text overlay severity (0-1) · ↑ higher=better · 0-1

**[`text_overlay`](src/ayase/modules/text_overlay.py)** — Text overlay / subtitle detection in video frames

- **Input**: img/vid · **Speed**: ⚡ fast
- **Packages**: opencv-python
- **Tests**: covered by [`test_text_overlay.py`](tests/modules/per_module/test_text_overlay.py), [`test_motion_scene_semantic_metrics.py`](tests/modules/test_motion_scene_semantic_metrics.py)
- **Config**: `subsample=4`, `edge_threshold=0.15`


## Safety & Ethics (7 metrics)

### `ai_generated_probability` [↑](#categories)
> AI-generated content likelihood 0-1 · 0-1

**[`watermark_classifier`](src/ayase/modules/watermark_classifier.py)** — Classifies video for watermarks using a pretrained model or custom ResNet-50 weights

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: Pillow, torch, torchvision, transformers
- **VRAM**: ~200 MB
- **Source**: <a href="https://huggingface.co/umm-maybe/AI-image-detector" target="_blank">HF</a>
- **Tests**: covered by [`test_watermark_classifier.py`](tests/modules/per_module/test_watermark_classifier.py)
- **Config**: `model_weights_path=`, `hf_model=umm-maybe/AI-image-detector`, `threshold=0.5`

### `bias_score` [↑](#categories)
> Representation imbalance indicator 0-1 · ↑ higher=better · 0-1

**[`bias_detection`](src/ayase/modules/bias_detection.py)** — Demographic representation analysis (face count, age distribution)

- **Input**: img/vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_bias_detection.py`](tests/modules/per_module/test_bias_detection.py), [`test_opencv_modules.py`](tests/modules/test_opencv_modules.py)
- **Config**: `subsample=10`, `max_frames=30`, `warning_threshold=0.7`

### `deepfake_probability` [↑](#categories)
> Synthetic/deepfake likelihood 0-1 · 0-1

**[`deepfake_detection`](src/ayase/modules/deepfake_detection.py)** — Synthetic media / deepfake likelihood estimation

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: Pillow, scipy, torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_deepfake_detection.py`](tests/modules/per_module/test_deepfake_detection.py), [`test_safety_modules.py`](tests/modules/test_safety_modules.py)
- **Config**: `subsample=10`, `max_frames=60`, `warning_threshold=0.6`

### `harmful_content_score` [↑](#categories)
> Violence/gore severity 0-1 · ↑ higher=better · 0-1

**[`harmful_content`](src/ayase/modules/harmful_content.py)** — Violence, gore, and disturbing content detection

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: Pillow, torch, transformers
- **VRAM**: ~600 MB
- **Source**: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">HF</a>
- **Tests**: covered by [`test_harmful_content.py`](tests/modules/per_module/test_harmful_content.py), [`test_safety_modules.py`](tests/modules/test_safety_modules.py)
- **Config**: `subsample=10`, `max_frames=60`, `warning_threshold=0.4`

### `nsfw_score` [↑](#categories)
> 0-1, likelihood of being NSFW · ↑ higher=better · 0-1

**[`nsfw`](src/ayase/modules/nsfw.py)** — Detects NSFW (adult/violent) content using ViT

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: opencv-python, torch, transformers
- **Source**: <a href="https://huggingface.co/Falconsai/nsfw_image_detection" target="_blank">HF</a>
- **Tests**: covered by [`test_nsfw.py`](tests/modules/per_module/test_nsfw.py)
- **Config**: `model_name=Falconsai/nsfw_image_detection`, `threshold=0.5`, `num_frames=8`

### `watermark_probability` [↑](#categories)
> 0-1 · 0-1

**[`watermark_classifier`](src/ayase/modules/watermark_classifier.py)** — Classifies video for watermarks using a pretrained model or custom ResNet-50 weights

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Packages**: Pillow, torch, torchvision, transformers
- **VRAM**: ~200 MB
- **Source**: <a href="https://huggingface.co/umm-maybe/AI-image-detector" target="_blank">HF</a>
- **Tests**: covered by [`test_watermark_classifier.py`](tests/modules/per_module/test_watermark_classifier.py)
- **Config**: `model_weights_path=`, `hf_model=umm-maybe/AI-image-detector`, `threshold=0.5`

### `watermark_strength` [↑](#categories)
> Invisible watermark strength 0-1 · 0-1

**[`watermark_robustness`](src/ayase/modules/watermark_robustness.py)** — Invisible watermark detection and strength estimation

- **Input**: img/vid · **Speed**: ⚡ fast
- **Packages**: imwatermark
- **Tests**: covered by [`test_watermark_robustness.py`](tests/modules/per_module/test_watermark_robustness.py), [`test_safety_modules.py`](tests/modules/test_safety_modules.py)
- **Config**: `subsample=15`, `max_frames=30`


## Image-to-Video Reference (4 metrics)

### `i2v_clip` [↑](#categories)
> CLIP image-video similarity (0-1) · 0-1

**[`i2v_similarity`](src/ayase/modules/i2v_similarity.py)** — Image-to-Video reference similarity using CLIP, DINOv2, and LPIPS (sliding window)

- **Input**: vid +ref · **Speed**: ⏱️ medium · GPU
- **Packages**: Pillow, lpips, open-clip-torch, timm, torch, torchvision
- **VRAM**: ~600 MB
- **Source**: <a href="https://github.com/richzhang/PerceptualSimilarity" target="_blank">GitHub</a> · <a href="https://huggingface.co/lpips/alex.pth" target="_blank">HF</a>
- **Tests**: covered by [`test_i2v_similarity.py`](tests/modules/per_module/test_i2v_similarity.py), [`test_integration_synthetic.py`](tests/test_integration_synthetic.py)
- **Config**: `window_size=16`, `stride=8`, `max_frames=256`, `clip_model=ViT-B-32`, `clip_pretrained=openai`, `dino_model=dinov2_vitb14`, `enable_clip=True`, `enable_dino=True`, `enable_lpips=True`

### `i2v_dino` [↑](#categories)
> DINOv2 image-video similarity (0-1) · 0-1

**[`i2v_similarity`](src/ayase/modules/i2v_similarity.py)** — Image-to-Video reference similarity using CLIP, DINOv2, and LPIPS (sliding window)

- **Input**: vid +ref · **Speed**: ⏱️ medium · GPU
- **Packages**: Pillow, lpips, open-clip-torch, timm, torch, torchvision
- **VRAM**: ~600 MB
- **Source**: <a href="https://github.com/richzhang/PerceptualSimilarity" target="_blank">GitHub</a> · <a href="https://huggingface.co/lpips/alex.pth" target="_blank">HF</a>
- **Tests**: covered by [`test_i2v_similarity.py`](tests/modules/per_module/test_i2v_similarity.py), [`test_integration_synthetic.py`](tests/test_integration_synthetic.py)
- **Config**: `window_size=16`, `stride=8`, `max_frames=256`, `clip_model=ViT-B-32`, `clip_pretrained=openai`, `dino_model=dinov2_vitb14`, `enable_clip=True`, `enable_dino=True`, `enable_lpips=True`

### `i2v_lpips` [↑](#categories)
> LPIPS image-video distance (0-1, lower=better) · ↓ lower=better · 0-1

**[`i2v_similarity`](src/ayase/modules/i2v_similarity.py)** — Image-to-Video reference similarity using CLIP, DINOv2, and LPIPS (sliding window)

- **Input**: vid +ref · **Speed**: ⏱️ medium · GPU
- **Packages**: Pillow, lpips, open-clip-torch, timm, torch, torchvision
- **VRAM**: ~600 MB
- **Source**: <a href="https://github.com/richzhang/PerceptualSimilarity" target="_blank">GitHub</a> · <a href="https://huggingface.co/lpips/alex.pth" target="_blank">HF</a>
- **Tests**: covered by [`test_i2v_similarity.py`](tests/modules/per_module/test_i2v_similarity.py), [`test_integration_synthetic.py`](tests/test_integration_synthetic.py)
- **Config**: `window_size=16`, `stride=8`, `max_frames=256`, `clip_model=ViT-B-32`, `clip_pretrained=openai`, `dino_model=dinov2_vitb14`, `enable_clip=True`, `enable_dino=True`, `enable_lpips=True`

### `i2v_quality` [↑](#categories)
> Aggregated I2V quality (0-100) · ↑ higher=better · 0-100

**[`i2v_similarity`](src/ayase/modules/i2v_similarity.py)** — Image-to-Video reference similarity using CLIP, DINOv2, and LPIPS (sliding window)

- **Input**: vid +ref · **Speed**: ⏱️ medium · GPU
- **Packages**: Pillow, lpips, open-clip-torch, timm, torch, torchvision
- **VRAM**: ~600 MB
- **Source**: <a href="https://github.com/richzhang/PerceptualSimilarity" target="_blank">GitHub</a> · <a href="https://huggingface.co/lpips/alex.pth" target="_blank">HF</a>
- **Tests**: covered by [`test_i2v_similarity.py`](tests/modules/per_module/test_i2v_similarity.py), [`test_integration_synthetic.py`](tests/test_integration_synthetic.py)
- **Config**: `window_size=16`, `stride=8`, `max_frames=256`, `clip_model=ViT-B-32`, `clip_pretrained=openai`, `dino_model=dinov2_vitb14`, `enable_clip=True`, `enable_dino=True`, `enable_lpips=True`


## Meta & Curation (6 metrics)

### `confidence_score` [↑](#categories)
> Prediction confidence · ↑ higher=better

**[`unqa`](src/ayase/modules/unqa.py)** — UNQA unified no-reference quality for audio/image/video (2024)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Backend**: resnet_unified
- **Packages**: Pillow, torch, torchaudio, torchvision
- **VRAM**: ~200 MB
- **Tests**: covered by [`test_unqa.py`](tests/modules/per_module/test_unqa.py)
- **Config**: `subsample=8`

### `llm_qa_score` [↑](#categories)
> LMM descriptive quality rating (0-1) · ↑ higher=better · 0-1

**[`llm_descriptive_qa`](src/ayase/modules/llm_descriptive_qa.py)** — LMM-based interpretable quality assessment with explanations

- **Input**: img/vid · **Speed**: 🐌 slow · GPU
- **Packages**: Pillow, openai, torch, transformers
- **VRAM**: ~14 GB
- **Source**: <a href="https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf" target="_blank">HF</a>
- **Tests**: covered by [`test_llm_descriptive_qa.py`](tests/modules/per_module/test_llm_descriptive_qa.py), [`test_reference_and_meta_metrics.py`](tests/modules/test_reference_and_meta_metrics.py)
- **Config**: `model_name=llava-hf/llava-v1.6-mistral-7b-hf`, `use_openai=False`, `num_frames=4`, `device=auto`

### `nemo_quality_label` [↑](#categories)
> Quality label (Low/Medium/High) · ↑ higher=better · type: str

**[`nemo_curator`](src/ayase/modules/nemo_curator.py)** — Caption text quality scoring (DeBERTa/FastText)

- **Input**: img/vid +cap · **Speed**: ⏱️ medium · GPU
- **Backend**: deberta → fasttext
- **Packages**: fasttext, torch, transformers
- **Tests**: covered by [`test_nemo_curator.py`](tests/modules/per_module/test_nemo_curator.py), [`test_nemo_curator.py`](tests/modules/test_nemo_curator.py)
- **Config**: `backend=auto`, `model_name=nvidia/quality-classifier-deberta`, `min_length=10`, `max_length=2000`

### `nemo_quality_score` [↑](#categories)
> Caption text quality (0-1) · ↑ higher=better · 0-1

**[`nemo_curator`](src/ayase/modules/nemo_curator.py)** — Caption text quality scoring (DeBERTa/FastText)

- **Input**: img/vid +cap · **Speed**: ⏱️ medium · GPU
- **Backend**: deberta → fasttext
- **Packages**: fasttext, torch, transformers
- **Tests**: covered by [`test_nemo_curator.py`](tests/modules/per_module/test_nemo_curator.py), [`test_nemo_curator.py`](tests/modules/test_nemo_curator.py)
- **Config**: `backend=auto`, `model_name=nvidia/quality-classifier-deberta`, `min_length=10`, `max_length=2000`

### `usability_rate` [↑](#categories)
> Percentage of usable frames

**[`usability_rate`](src/ayase/modules/usability_rate.py)** — Computes percentage of usable frames based on quality thresholds

- **Input**: img/vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_usability_rate.py`](tests/modules/per_module/test_usability_rate.py), [`test_fields_general.py`](tests/modules/test_fields_general.py), [`test_reference_and_meta_metrics.py`](tests/modules/test_reference_and_meta_metrics.py)
- **Config**: `quality_threshold=50.0`

### `vtss` [↑](#categories)
> Video Training Suitability Score (0-1) · 0-1

**[`vtss`](src/ayase/modules/vtss.py)** — Video Training Suitability Score (0-1, meta-metric)

- **Input**: img/vid · **Speed**: ⚡ fast
- **VRAM**: ~800 MB
- **Tests**: covered by [`test_vtss.py`](tests/modules/per_module/test_vtss.py), [`test_curation_metrics.py`](tests/modules/test_curation_metrics.py)
- **Config**: `weights={'aesthetic': 0.15, 'technical': 0.15, 'motion': 0.1, 'clip_temp': 0.15, 'blur': 0.1, 'noise': 0.1, 'scene_stability': 0.1, 'resolution': 0.15}`


## Dataset-Level Metrics (28 fields)

Fields stored on `DatasetStats` via `pipeline.add_dataset_metric()` after batch/post-processing.

### `avg_face_cross_similarity` [↑](#categories)
> Dataset-level average · ↑ higher=better · type: float

**[`face_cross_similarity`](src/ayase/modules/face_cross_similarity.py)** — Dataset-wide average pairwise face similarity

- **Input**: img/vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_face_cross_similarity.py`](tests/modules/per_module/test_face_cross_similarity.py)

### `class_balance_score` [↑](#categories)
> Category balance 0-1 (higher=balanced) · ↑ higher=better · type: float

**[`dataset_analytics`](src/ayase/modules/dataset_analytics.py)** — Class/category balance score (0-1, higher=balanced)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Tests**: covered by [`test_dataset_analytics.py`](tests/modules/per_module/test_dataset_analytics.py), [`test_dataset_modules.py`](tests/modules/test_dataset_modules.py)

### `coverage` [↑](#categories)
> Diversity of generated samples (0-1) · type: float

**[`generative_distribution`](src/ayase/modules/generative_distribution_metrics.py)** — Fraction of real samples covered by generated neighbours (0-1, higher=better)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Tests**: covered by [`test_generative_distribution.py`](tests/modules/per_module/test_generative_distribution.py), [`test_dataset_modules.py`](tests/modules/test_dataset_modules.py)

**[`generative_distribution_metrics`](src/ayase/modules/generative_distribution_metrics.py)** — Fraction of real samples covered by generated neighbours (0-1, higher=better)

- **Input**: img/vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_generative_distribution.py`](tests/modules/per_module/test_generative_distribution.py), [`test_generative_distribution_metrics.py`](tests/modules/per_module/test_generative_distribution_metrics.py), [`test_dataset_modules.py`](tests/modules/test_dataset_modules.py)

### `density` [↑](#categories)
> Concentration around real samples · type: float

**[`generative_distribution`](src/ayase/modules/generative_distribution_metrics.py)** — Average normalized generated-sample density around real samples

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Tests**: covered by [`test_generative_distribution.py`](tests/modules/per_module/test_generative_distribution.py), [`test_dataset_modules.py`](tests/modules/test_dataset_modules.py)

**[`generative_distribution_metrics`](src/ayase/modules/generative_distribution_metrics.py)** — Average normalized generated-sample density around real samples

- **Input**: img/vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_generative_distribution.py`](tests/modules/per_module/test_generative_distribution.py), [`test_generative_distribution_metrics.py`](tests/modules/per_module/test_generative_distribution_metrics.py), [`test_dataset_modules.py`](tests/modules/test_dataset_modules.py)

### `diversity_score` [↑](#categories)
> Visual diversity 0-1 (higher=more diverse) · ↑ higher=better · type: float

**[`dataset_analytics`](src/ayase/modules/dataset_analytics.py)** — Dataset visual diversity score (0-1, higher=more diverse)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Tests**: covered by [`test_dataset_analytics.py`](tests/modules/per_module/test_dataset_analytics.py), [`test_dataset_modules.py`](tests/modules/test_dataset_modules.py)

### `duplicate_pairs` [↑](#categories)
> Count of near-duplicate pairs · type: int

**[`dataset_analytics`](src/ayase/modules/dataset_analytics.py)** — Count of near-duplicate sample pairs

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Tests**: covered by [`test_dataset_analytics.py`](tests/modules/per_module/test_dataset_analytics.py), [`test_dataset_modules.py`](tests/modules/test_dataset_modules.py)

### `face_similarity_matrix` [↑](#categories)
> NxN pairwise similarity · ↑ higher=better · type: float

**[`face_cross_similarity`](src/ayase/modules/face_cross_similarity.py)** — Dataset NxN pairwise face similarity matrix

- **Input**: img/vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_face_cross_similarity.py`](tests/modules/per_module/test_face_cross_similarity.py)

### `fad` [↑](#categories)
> Frechet Audio Distance (lower=better) · ↓ lower=better · type: float

**[`fad`](src/ayase/modules/fad.py)** — Frechet Audio Distance between generated and reference audio distributions (lower=better)

- **Input**: audio · **Speed**: ⚡ fast
- **Tests**: covered by [`test_fad.py`](tests/modules/per_module/test_fad.py)

### `fgd` [↑](#categories)
> Frechet Gesture Distance (lower=better) · ↓ lower=better · type: float

**[`fgd`](src/ayase/modules/fgd.py)** — Frechet Gesture Distance between generated and reference motion distributions (lower=better)

- **Input**: vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_fgd.py`](tests/modules/per_module/test_fgd.py)

### `fmd` [↑](#categories)
> Frechet Motion Distance (lower=better) · ↓ lower=better · type: float

**[`fmd`](src/ayase/modules/fmd.py)** — Frechet Motion Distance between generated and reference motion distributions (lower=better)

- **Input**: vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_fmd.py`](tests/modules/per_module/test_fmd.py)

### `fvd` [↑](#categories)
> Fréchet Video Distance · ↓ lower=better · type: float

**[`fvd`](src/ayase/modules/fvd.py)** — Frechet Video Distance between generated and reference video distributions (lower=better)

- **Input**: vid +ref · **Speed**: ⏱️ medium · GPU
- **Tests**: covered by [`test_fvd.py`](tests/modules/per_module/test_fvd.py), [`test_dataset_modules.py`](tests/modules/test_dataset_modules.py), [`test_fields_general.py`](tests/modules/test_fields_general.py), +1 more

### `fvmd` [↑](#categories)
> Fréchet Video Motion Distance · ↓ lower=better · type: float

**[`fvmd`](src/ayase/modules/fvmd.py)** — Frechet Video Motion Distance from optical-flow features (lower=better)

- **Input**: vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_fvmd.py`](tests/modules/per_module/test_fvmd.py), [`test_dataset_modules.py`](tests/modules/test_dataset_modules.py), [`test_fields_general.py`](tests/modules/test_fields_general.py), +1 more

### `identity_cluster_count` [↑](#categories)
> Number of identity clusters · type: int

**[`face_cross_similarity`](src/ayase/modules/face_cross_similarity.py)** — Estimated number of identity clusters in the dataset

- **Input**: img/vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_face_cross_similarity.py`](tests/modules/per_module/test_face_cross_similarity.py)

### `jedi` [↑](#categories)
> JEDi (V-JEPA + MMD, ICLR 2025) · type: float

**[`jedi`](src/ayase/modules/jedi_metric.py)** — JEDi V-JEPA embedding distance via MMD (lower=better)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Tests**: covered by [`test_jedi.py`](tests/modules/per_module/test_jedi.py), [`test_motion_scene_semantic_metrics.py`](tests/modules/test_motion_scene_semantic_metrics.py)

**[`jedi_metric`](src/ayase/modules/jedi_metric.py)** — JEDi V-JEPA embedding distance via MMD (lower=better)

- **Input**: vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_jedi.py`](tests/modules/per_module/test_jedi.py), [`test_jedi_metric.py`](tests/modules/per_module/test_jedi_metric.py), [`test_motion_scene_semantic_metrics.py`](tests/modules/test_motion_scene_semantic_metrics.py)

### `kid` [↑](#categories)
> Kernel Inception Distance (lower=better) · ↓ lower=better · type: float

**[`kid`](src/ayase/modules/kid.py)** — Kernel Inception Distance estimate (lower=better)

- **Input**: img/vid +ref · **Speed**: ⏱️ medium · GPU
- **Tests**: covered by [`test_kid.py`](tests/modules/per_module/test_kid.py)

### `kid_std` [↑](#categories)
> KID standard deviation · type: float

**[`kid`](src/ayase/modules/kid.py)** — Standard deviation over KID subsets

- **Input**: img/vid +ref · **Speed**: ⏱️ medium · GPU
- **Tests**: covered by [`test_kid.py`](tests/modules/per_module/test_kid.py)

### `kvd` [↑](#categories)
> Kernel Video Distance · ↓ lower=better · type: float

**[`kvd`](src/ayase/modules/kvd.py)** — Kernel Video Distance via MMD over video features (lower=better)

- **Input**: vid · **Speed**: ⏱️ medium · GPU
- **Tests**: covered by [`test_kvd.py`](tests/modules/per_module/test_kvd.py), [`test_dataset_modules.py`](tests/modules/test_dataset_modules.py), [`test_fields_general.py`](tests/modules/test_fields_general.py), +1 more

### `lpips_diversity` [↑](#categories)
> Average pairwise LPIPS across dataset (higher=more diverse) · type: float

**[`image_lpips`](src/ayase/modules/image_lpips.py)** — Dataset average pairwise LPIPS distance (higher=more diverse)

- **Input**: img/vid +ref · **Speed**: ⏱️ medium · GPU
- **Tests**: covered by [`test_image_lpips.py`](tests/modules/per_module/test_image_lpips.py)

### `outlier_count` [↑](#categories)
> Number of statistical outliers · type: int

**[`dataset_analytics`](src/ayase/modules/dataset_analytics.py)** — Number of statistical outliers detected in the dataset

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Tests**: covered by [`test_dataset_analytics.py`](tests/modules/per_module/test_dataset_analytics.py), [`test_dataset_modules.py`](tests/modules/test_dataset_modules.py)

### `precision` [↑](#categories)
> Quality of generated samples (0-1) · type: float

**[`generative_distribution`](src/ayase/modules/generative_distribution_metrics.py)** — Generated-sample precision against the real manifold (0-1, higher=better)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Tests**: covered by [`test_generative_distribution.py`](tests/modules/per_module/test_generative_distribution.py), [`test_dataset_modules.py`](tests/modules/test_dataset_modules.py)

**[`generative_distribution_metrics`](src/ayase/modules/generative_distribution_metrics.py)** — Generated-sample precision against the real manifold (0-1, higher=better)

- **Input**: img/vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_generative_distribution.py`](tests/modules/per_module/test_generative_distribution.py), [`test_generative_distribution_metrics.py`](tests/modules/per_module/test_generative_distribution_metrics.py), [`test_dataset_modules.py`](tests/modules/test_dataset_modules.py)

### `recall` [↑](#categories)
> Coverage of real distribution (0-1) · type: float

**[`generative_distribution`](src/ayase/modules/generative_distribution_metrics.py)** — Real-distribution coverage by generated samples (0-1, higher=better)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Tests**: covered by [`test_generative_distribution.py`](tests/modules/per_module/test_generative_distribution.py), [`test_dataset_modules.py`](tests/modules/test_dataset_modules.py)

**[`generative_distribution_metrics`](src/ayase/modules/generative_distribution_metrics.py)** — Real-distribution coverage by generated samples (0-1, higher=better)

- **Input**: img/vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_generative_distribution.py`](tests/modules/per_module/test_generative_distribution.py), [`test_generative_distribution_metrics.py`](tests/modules/per_module/test_generative_distribution_metrics.py), [`test_dataset_modules.py`](tests/modules/test_dataset_modules.py)

### `semantic_coverage` [↑](#categories)
> Embedding space coverage 0-1 · type: float

**[`dataset_analytics`](src/ayase/modules/dataset_analytics.py)** — Embedding-space coverage score (0-1, higher=more coverage)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Tests**: covered by [`test_dataset_analytics.py`](tests/modules/per_module/test_dataset_analytics.py), [`test_dataset_modules.py`](tests/modules/test_dataset_modules.py)

### `umap_coverage` [↑](#categories)
> UMAP projection coverage (0-1) · type: float

**[`umap_projection`](src/ayase/modules/umap_projection.py)** — Coverage of occupied projection space (0-1, higher=better)

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Tests**: covered by [`test_umap_projection.py`](tests/modules/per_module/test_umap_projection.py), [`test_umap_projection.py`](tests/modules/test_umap_projection.py)

### `umap_spread` [↑](#categories)
> UMAP projection spread · type: float

**[`umap_projection`](src/ayase/modules/umap_projection.py)** — Spread of dataset embeddings in the 2-D projection

- **Input**: img/vid · **Speed**: ⏱️ medium · GPU
- **Tests**: covered by [`test_umap_projection.py`](tests/modules/per_module/test_umap_projection.py), [`test_umap_projection.py`](tests/modules/test_umap_projection.py)

### `vendi` [↑](#categories)
> Vendi Score diversity (higher=better) · ↑ higher=better · type: float

**[`vendi`](src/ayase/modules/vendi.py)** — Vendi Score dataset diversity from similarity-matrix entropy (higher=better)

- **Input**: img/vid · **Speed**: ⚡ fast
- **Tests**: covered by [`test_vendi.py`](tests/modules/per_module/test_vendi.py)

### `verse_bench_breakdown` [↑](#categories)
> Verse-Bench subscores and overall · type: float

**[`verse_bench`](src/ayase/modules/verse_bench.py)** — Subscore dict: S_joint, S_video, S_audio, S_other, Overall Score

- **Input**: img/vid · **Speed**: 🐌 slow
- **Tests**: covered by [`test_verse_bench.py`](tests/modules/per_module/test_verse_bench.py)

### `verse_bench_metrics` [↑](#categories)
> Raw Verse-Bench component metrics · type: float

**[`verse_bench`](src/ayase/modules/verse_bench.py)** — Raw 12-component metric dict: AS, ID, FD, KL, CS, CE, CU, PC, PQ, WER, LSE-C, AV-A

- **Input**: img/vid · **Speed**: 🐌 slow
- **Tests**: covered by [`test_verse_bench.py`](tests/modules/per_module/test_verse_bench.py)

### `verse_bench_overall` [↑](#categories)
> Official Verse-Bench final score · type: float

**[`verse_bench`](src/ayase/modules/verse_bench.py)** — Weighted aggregate score (0-1, higher=better) from S_joint(50%), S_video(20%), S_audio(20%), S_other(10%)

- **Input**: img/vid · **Speed**: 🐌 slow
- **Tests**: covered by [`test_verse_bench.py`](tests/modules/per_module/test_verse_bench.py)

## Utility & Validation (32 modules)

Modules that perform validation, embedding, deduplication, or dataset-level analysis without writing individual QualityMetrics fields.

- **[`audio`](src/ayase/modules/audio.py)** — Validates audio stream quality and presence · Input: vid · Speed: ⚡ fast · Tests: covered by [`test_audio.py`](tests/modules/per_module/test_audio.py), [`test_docs_integrity.py`](tests/test_docs_integrity.py), [`test_integration_synthetic.py`](tests/test_integration_synthetic.py)
- **[`audio_text_alignment`](src/ayase/modules/audio_text_alignment.py)** — Multimodal alignment check (Audio-Text) using CLAP · Input: audio +cap · Speed: ⏱️ medium · GPU · Tests: covered by [`test_audio_text_alignment.py`](tests/modules/per_module/test_audio_text_alignment.py)
- **[`background_diversity`](src/ayase/modules/background_diversity.py)** — Checks background complexity (entropy) to detect concept bleeding · Input: img/vid · Speed: ⚡ fast · Tests: covered by [`test_background_diversity.py`](tests/modules/per_module/test_background_diversity.py)
- **[`bd_rate`](src/ayase/modules/bd_rate.py)** — BD-Rate codec comparison (dataset-level, negative%=better) · Input: img/vid · Speed: ⚡ fast · Tests: covered by [`test_bd_rate.py`](tests/modules/per_module/test_bd_rate.py), [`test_industry_metrics.py`](tests/modules/test_industry_metrics.py)
- **[`codec_compatibility`](src/ayase/modules/codec_compatibility.py)** — Validates codec, pixel format, and container for ML dataloader compatibility · Input: vid · Speed: ⚡ fast · Tests: covered by [`test_codec_compatibility.py`](tests/modules/per_module/test_codec_compatibility.py)
- **[`decoder_stress`](src/ayase/modules/decoder_stress.py)** — Random access decoder stress test · Input: vid · Speed: ⚡ fast · Tests: covered by [`test_decoder_stress.py`](tests/modules/per_module/test_decoder_stress.py)
- **[`dedup`](src/ayase/modules/dedup.py)** — Detects duplicates using Perceptual Hashing (pHash) · Input: img/vid · Speed: ⚡ fast · Tests: covered by [`test_dedup.py`](tests/modules/per_module/test_dedup.py), [`test_deduplication.py`](tests/modules/per_module/test_deduplication.py), [`test_docs_integrity.py`](tests/test_docs_integrity.py)
- **[`deduplication`](src/ayase/modules/dedup.py)** — Detects duplicates using Perceptual Hashing (pHash) · Input: img/vid · Speed: ⚡ fast · Tests: covered by [`test_deduplication.py`](tests/modules/per_module/test_deduplication.py), [`test_docs_integrity.py`](tests/test_docs_integrity.py)
- **[`diversity`](src/ayase/modules/diversity_selection.py)** — Flags redundant samples using embedding similarity (Deduplication) · Input: img/vid · Speed: ⚡ fast · Tests: covered by [`test_diversity.py`](tests/modules/per_module/test_diversity.py)
- **[`diversity_selection`](src/ayase/modules/diversity_selection.py)** — Flags redundant samples using embedding similarity (Deduplication) · Input: img/vid · Speed: ⚡ fast · Tests: covered by [`test_diversity.py`](tests/modules/per_module/test_diversity.py), [`test_diversity_selection.py`](tests/modules/per_module/test_diversity_selection.py)
- **[`embedding`](src/ayase/modules/embedding.py)** — Calculates X-CLIP embeddings for similarity search · Input: img/vid · Speed: ⏱️ medium · GPU · Tests: covered by [`test_embedding.py`](tests/modules/per_module/test_embedding.py)
- **[`exposure`](src/ayase/modules/exposure.py)** — Checks for overexposure, underexposure, and low contrast using histograms · Input: img/vid · Speed: ⚡ fast · Tests: covered by [`test_exposure.py`](tests/modules/per_module/test_exposure.py)
- **[`knowledge_graph`](src/ayase/modules/knowledge_graph.py)** — Generates a conceptual knowledge graph of the video dataset · Input: img/vid · Speed: ⚡ fast · Tests: covered by [`test_knowledge_graph.py`](tests/modules/per_module/test_knowledge_graph.py)
- **[`llm_advisor`](src/ayase/modules/llm_advisor.py)** — Rule-based improvement recommendations derived from quality metrics (no LLM used) · Input: img/vid · Speed: 🐌 slow · Tests: covered by [`test_llm_advisor.py`](tests/modules/per_module/test_llm_advisor.py)
- **[`metadata`](src/ayase/modules/metadata.py)** — Checks video/image metadata (resolution, FPS, duration, integrity) · Input: img/vid · Speed: ⚡ fast · Tests: covered by [`test_metadata.py`](tests/modules/per_module/test_metadata.py), [`test_integration_synthetic.py`](tests/test_integration_synthetic.py), [`test_profiles.py`](tests/test_profiles.py), +3 more
- **[`msswd`](src/ayase/modules/msswd.py)** — MSSWD multi-scale sliced Wasserstein distance via pyiqa (batch, lower=better) · Input: img/vid · Speed: ⏱️ medium · Tests: covered by [`test_msswd.py`](tests/modules/per_module/test_msswd.py)
- **[`multiple_objects`](src/ayase/modules/multiple_objects.py)** — Verifies object count matches caption (VBench multiple_objects dimension) · Input: img/vid +cap · Speed: ⚡ fast · Tests: covered by [`test_multiple_objects.py`](tests/modules/per_module/test_multiple_objects.py)
- **[`paranoid_decoder`](src/ayase/modules/paranoid_decoder.py)** — Deep bitstream validation using FFmpeg (Paranoid Mode) · Input: vid · Speed: ⚡ fast · Tests: covered by [`test_paranoid_decoder.py`](tests/modules/per_module/test_paranoid_decoder.py)
- **[`resolution_bucketing`](src/ayase/modules/resolution_bucketing.py)** — Validates resolution/aspect-ratio fit for training buckets · Input: img/vid · Speed: ⚡ fast · Tests: covered by [`test_resolution_bucketing.py`](tests/modules/per_module/test_resolution_bucketing.py)
- **[`scene`](src/ayase/modules/scene.py)** — Detects scene cuts and shots using PySceneDetect · Input: vid · Speed: ⚡ fast · Tests: covered by [`test_concept_presence.py`](tests/modules/per_module/test_concept_presence.py), [`test_scene.py`](tests/modules/per_module/test_scene.py), [`test_vbench2_compbench.py`](tests/modules/test_vbench2_compbench.py), +1 more
- **[`scene_tagging`](src/ayase/modules/scene_tagging.py)** — Tags scene context (Proxy for Tag2Text/RAM using CLIP) · Input: img/vid · Speed: ⏱️ medium · GPU · Tests: covered by [`test_scene_tagging.py`](tests/modules/per_module/test_scene_tagging.py)
- **[`semantic_selection`](src/ayase/modules/semantic_selection.py)** — Selects diverse samples based on VLM-extracted semantic traits · Input: img/vid · Speed: ⚡ fast · Tests: covered by [`test_semantic_selection.py`](tests/modules/per_module/test_semantic_selection.py)
- **[`sfid`](src/ayase/modules/sfid.py)** — SFID spatial Fréchet Inception Distance via pyiqa (batch, lower=better) · Input: img/vid · Speed: ⏱️ medium · Tests: covered by [`test_sfid.py`](tests/modules/per_module/test_sfid.py)
- **[`spatial_relationship`](src/ayase/modules/spatial_relationship.py)** — Verifies spatial relations (left/right/top/bottom) in prompt vs detections · Input: img/vid +cap · Speed: ⚡ fast · Tests: covered by [`test_spatial_relationship.py`](tests/modules/per_module/test_spatial_relationship.py)
- **[`spectral_upscaling`](src/ayase/modules/spectral_upscaling.py)** — Detection of upscaled/fake high-resolution content · Input: img/vid · Speed: ⚡ fast · Tests: covered by [`test_spectral_upscaling.py`](tests/modules/per_module/test_spectral_upscaling.py)
- **[`stream_metric`](src/ayase/modules/stream_metric.py)** — STREAM spatial/temporal generation eval (ICLR 2024) · Input: img/vid · Speed: ⚡ fast · Tests: covered by [`test_stream_metric.py`](tests/modules/per_module/test_stream_metric.py)
- **[`structural`](src/ayase/modules/structural.py)** — Checks structural integrity (scene cuts, black bars) · Input: vid · Speed: ⚡ fast · Tests: covered by [`test_structural.py`](tests/modules/per_module/test_structural.py)
- **[`style_consistency`](src/ayase/modules/style_consistency.py)** — Appearance Style verification (Gram Matrix Consistency) · Input: vid · Speed: ⚡ fast · Tests: covered by [`test_style_consistency.py`](tests/modules/per_module/test_style_consistency.py)
- **[`temporal_style`](src/ayase/modules/temporal_style.py)** — Analyzes temporal style (Slow Motion, Timelapse, Speed) · Input: vid · Speed: ⚡ fast · Tests: covered by [`test_temporal_style.py`](tests/modules/per_module/test_temporal_style.py)
- **[`vfr_detection`](src/ayase/modules/vfr_detection.py)** — Variable Frame Rate (VFR) and jitter detection · Input: vid · Speed: ⚡ fast · Tests: covered by [`test_vfr_detection.py`](tests/modules/per_module/test_vfr_detection.py)
- **[`vlm_judge`](src/ayase/modules/vlm_judge.py)** — Advanced semantic verification using VLM (e.g. LLaVA) · Input: img/vid · Speed: 🐌 slow · GPU · Tests: covered by [`test_vlm_judge.py`](tests/modules/per_module/test_vlm_judge.py), [`test_vlm_presets.py`](tests/modules/test_vlm_presets.py)
- **[`worldscore`](src/ayase/modules/worldscore.py)** — WorldScore world generation evaluation (ICCV 2025) · Input: vid · Speed: ⚡ fast · Tests: covered by [`test_worldscore.py`](tests/modules/per_module/test_worldscore.py)
