# Ayase Model Weights Reference

Complete inventory of all pretrained model weights used by Ayase modules.
Models are downloaded and cached automatically on first use. Use this reference for manual management or auditing.

---

## 1. HuggingFace Transformers Models

Downloaded via `transformers.*.from_pretrained()` or `huggingface_hub.snapshot_download()`.
All support `cache_dir=models_dir` for local storage.

### Core (used by many modules)

| Model ID | Used By | License | Size |
|----------|---------|---------|------|
| `openai/clip-vit-base-patch32` | semantic_alignment, clip_temporal, sd_reference, action_recognition, background_consistency, deepfake_detection, generative_distribution_metrics, harmful_content, scene_tagging, t2v_score, video_memorability, video_text_matching, video_type_classifier, aigv_assessor, dataset_analytics, tifa, umap_projection | MIT | ~600 MB |
| `openai/clip-vit-large-patch14` | aesthetic_scoring | MIT | ~1.7 GB |

### Vision-Language

| Model ID | Used By | License | Size |
|----------|---------|---------|------|
| `Salesforce/blip-image-captioning-base` | captioning (default) | BSD-3-Clause | ~1 GB |
| `Salesforce/blip2-opt-2.7b` | captioning (configurable BLIP-2 mode) | MIT | ~15 GB |
| `dandelin/vilt-b32-finetuned-vqa` | commonsense, tifa | Apache 2.0 | ~500 MB |
| `llava-hf/llava-v1.6-mistral-7b-hf` | llm_descriptive_qa | Apache 2.0 | ~15 GB |
| `llava-hf/llava-1.5-7b-hf` | vlm_judge | LLaMA 2 license | ~14 GB |

### Video Understanding

| Model ID | Used By | License | Size |
|----------|---------|---------|------|
| `MCG-NJU/videomae-large-finetuned-kinetics` | action_recognition | CC-BY-NC 4.0 | ~1.2 GB |
| `microsoft/xclip-base-patch32` | embedding, video_text_matching | MIT | ~600 MB |

### Video Quality Assessment

| Model ID | Used By | License | Size |
|----------|---------|---------|------|
| `wangjiarui153/AIGV-Assessor` | aigv_assessor | Research | ~2 GB |
| `IntMeGroup/FineVQ_score` | finevq | Apache 2.0 | ~2 GB |
| `qyp2000/KVQ` | kvq | Research | ~2 GB |
| `sunwei925/RQ-VQA` | rqvqa | Research | ~2 GB |
| `q-future/one-align` | q_align | MIT | ~7 GB |
| `TIGER-Lab/VideoScore` | videoscore | Apache 2.0 | ~7 GB |
| `KlingTeam/VideoAlign-Reward` | video_reward | Research (gated) | ~2 GB |
| `TIGER-Lab/T2VScore` | t2v_score | Research | ~2 GB |

### Image Generation / Diffusion

| Model ID | Used By | License | Size |
|----------|---------|---------|------|
| `stabilityai/stable-diffusion-xl-base-1.0` | sd_reference | CreativeML Open RAIL++-M | ~7 GB |

### Segmentation & Detection

| Model ID | Used By | License | Size |
|----------|---------|---------|------|
| `nvidia/segformer-b0-finetuned-ade-512-512` | semantic_segmentation_consistency | Other (Nvidia) | ~15 MB |
| `Falconsai/nsfw_image_detection` | nsfw | Apache 2.0 | ~350 MB |
| `facebook/dinov2-base` | subject_consistency | Apache 2.0 | ~350 MB |

### Audio

| Model ID | Used By | License | Size |
|----------|---------|---------|------|
| `laion/clap-htsat-fused` | audio_text_alignment | Apache 2.0 | ~600 MB |

### Text Quality

| Model ID | Used By | License | Size |
|----------|---------|---------|------|
| `nvidia/quality-classifier-deberta` | nemo_curator | Apache 2.0 | ~1.5 GB |

---

## 2. Direct Download Weights

Downloaded from specific URLs. Stored in `models/` subdirectories.

### DOVER (Video Quality Assessment)

| File | URL | License | Size |
|------|-----|---------|------|
| `DOVER.pth` | https://github.com/VQAssessment/DOVER/releases/download/v0.1.0/DOVER.pth | MIT | ~240 MB |

**Used by:** dover module (native backend)
**Local path:** `models/dover/DOVER.pth`

### RAFT Optical Flow (TorchVision)

| File | URL | License | Size |
|------|-----|---------|------|
| `raft_small_C_T_V2-01064c6d.pth` | https://download.pytorch.org/models/raft_small_C_T_V2-01064c6d.pth | BSD-3-Clause | 4 MB |
| `raft_large_C_T_SKHT_V2-ff5fadd5.pth` | https://download.pytorch.org/models/raft_large_C_T_SKHT_V2-ff5fadd5.pth | BSD-3-Clause | 21 MB |

**Used by:** motion_amplitude (small), advanced_flow (large/small), temporal_flickering (small), flolpips (small), raft_motion (large)
**Local path:** `models/raft/`

### FAST-VQA / FasterVQA

| File | URL | License | Size |
|------|-----|---------|------|
| `FAST_VQA_3D_1_1.pth` | https://github.com/VQAssessment/FAST-VQA-and-FasterVQA/releases/download/v2.0.0/FAST_VQA_3D_1_1.pth | MIT | 127 MB |
| `FAST_VQA_B_1_4.pth` | https://github.com/VQAssessment/FAST-VQA-and-FasterVQA/releases/download/v2.0.0/FAST_VQA_B_1_4.pth | MIT | 127 MB |
| `FAST_VQA_M_1_4.pth` | https://github.com/VQAssessment/FAST-VQA-and-FasterVQA/releases/download/v2.0.0/FAST_VQA_M_1_4.pth | MIT | 111 MB |

**Used by:** fast_vqa module (FasterVQA, FAST-VQA, FAST-VQA-M variants)
**Local path:** `models/fast_vqa/`

### Aesthetic Scoring MLP Head

| File | URL | License | Size |
|------|-----|---------|------|
| `sac+logos+ava1-l14-linearMSE.pth` | https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth | MIT | ~3 KB |

**Used by:** aesthetic_scoring (on top of CLIP ViT-L/14)
**Local path:** `models/aesthetic/sac+logos+ava1-l14-linearMSE.pth`

### open_clip ViT-B-32 (OpenAI weights)

| File | URL | License | Size |
|------|-----|---------|------|
| `ViT-B-32.pt` | https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt | MIT | 354 MB |

**Used by:** i2v_similarity (CLIP sub-metric, via open_clip library)
**Local path:** `models/open_clip/ViT-B-32.pt`

### LPIPS AlexNet

| File | URL | License | Size |
|------|-----|---------|------|
| `alex.pth` | https://github.com/richzhang/PerceptualSimilarity/raw/master/lpips/weights/v0.1/alex.pth | BSD-2-Clause | ~6 KB |

**Used by:** i2v_similarity (LPIPS sub-metric)
**Local path:** `models/lpips/alex.pth`

---

## 3. Torch Hub Models

Downloaded via `torch.hub.load()`. Cached in `$TORCH_HOME/hub/`.

| Repository | Model Name | Used By | License | Size |
|-----------|-----------|---------|---------|------|
| `facebookresearch/dinov2` | `dinov2_vitb14` | i2v_similarity | Apache 2.0 | ~346 MB |
| `facebookresearch/dinov2` | `dinov2_vits14` | spectral (default), video_memorability | Apache 2.0 | ~86 MB |
| `intel-isl/MiDaS` | `MiDaS_small` | depth_consistency, depth_map_quality | MIT | ~25 MB |
| `intel-isl/MiDaS` | `DPT_Hybrid` | depth_consistency (large mode) | MIT | ~470 MB |
| `intel-isl/MiDaS` | `DPT_Large` | depth_consistency (large mode) | MIT | ~1.3 GB |
| `facebookresearch/co-tracker` | `cotracker2` | trajan | CC-BY-NC 4.0 | ~40 MB |

---

## 4. TorchVision Pretrained Weights

Downloaded automatically by TorchVision on first use. Cached in `$TORCH_HOME/hub/checkpoints/`.

| Weights Enum | Used By | License | Size |
|-------------|---------|---------|------|
| `Raft_Small_Weights.DEFAULT` | motion_amplitude, temporal_flickering, advanced_flow (fallback), flolpips | BSD-3-Clause | 4 MB |
| `Raft_Large_Weights.DEFAULT` | advanced_flow, raft_motion | BSD-3-Clause | 21 MB |
| `R3D_18_Weights.DEFAULT` / `KINETICS400_V1` | c3dvqa, fvd | BSD-3-Clause | ~125 MB |
| `ResNet18_Weights.DEFAULT` | tlvqm | BSD-3-Clause | ~45 MB |

---

## 5. PyIQA Metrics

Downloaded automatically by `pyiqa.create_metric()`. Weights managed by PyIQA internally.

| Metric Name | Used By | Category |
|------------|---------|----------|
| `dover` | dover (fallback), cover | Video Quality |
| `clipiqa+` | clip_iqa, promptiqa, rqvqa | Image Quality |
| `topiq_nr` | topiq, finevq, kvq, promptiqa | Image Quality |
| `topiq_fr` | topiq_fr | Image Quality (FR) |
| `topiq_nr-face` | face_iqa | Face Quality |
| `brisque` | brisque, naturalness | Image Quality |
| `niqe` | niqe | Image Quality |
| `musiq` | musiq (default variant) | Image Quality |
| `musiq-koniq` | musiq (configurable) | Image Quality |
| `musiq-spaq` | musiq (configurable) | Image Quality |
| `hyperiqa` | hyperiqa, qcn | Image Quality |
| `maniqa` | maniqa | Image Quality |
| `cnniqa` | cnniqa | Image Quality |
| `dbcnn` | dbcnn | Image Quality |
| `nima` | nima | Image Quality |
| `paq2piq` | paq2piq | Image Quality |
| `tres` | tres | Image Quality |
| `unique` | unique_iqa | Image Quality |
| `contrique` | contrique | Image Quality |
| `liqe` | liqe | Image Quality |
| `ilniqe` | ilniqe | Image Quality |
| `ahiq` | ahiq | Image Quality |
| `arniqa` | arniqa | Image Quality |
| `qualiclip` | qualiclip | Image Quality |
| `laion_aes` | laion_aesthetic | Aesthetic |
| `compare2score` | compare2score | Image Quality |
| `wadiqam_nr` | wadiqam | Image Quality |
| `wadiqam_fr` | wadiqam_fr | Image Quality (FR) |
| `deepwsd` | deepwsd | Image Quality |
| `dmm` | dmm | Image Quality |
| `ssimc` | ssimc | Image Quality |
| `ckdn` | ckdn | Image Quality |
| `cw_ssim` | cw_ssim | Image Quality (FR) |
| `mad` | mad_metric | Image Quality (FR) |
| `afine_nr` | afine | Image Quality |
| `nrqm` | nrqm | Image Quality |
| `pi` | pi_metric | Image Quality |
| `piqe` | piqe | Image Quality |
| `pieapp` | pieapp | Image Quality (FR) |
| `nlpd` | nlpd_metric | Image Quality (FR) |
| `qcn` | qcn | Image Quality |
| `maclip` | maclip | Image Quality |
| `promptiqa` | promptiqa | Image Quality |
| `mdtvsfa` | mdtvsfa | Video Quality |

---

## 6. Other Libraries

### PaddleOCR

| Component | Auto-downloaded | Used By | License |
|-----------|----------------|---------|---------|
| Detection model (`en_PP-OCRv4_det`) | Yes | ocr_fidelity, text | Apache 2.0 |
| Recognition model (`en_PP-OCRv4_rec`) | Yes | ocr_fidelity, text | Apache 2.0 |
| Angle classifier (`ch_ppocr_mobile_v2.0_cls`) | Yes | ocr_fidelity, text | Apache 2.0 |

**Note:** PaddleOCR downloads models to `~/.paddleocr/` on first use. Language configurable (default: `en`).

### open_clip

| Model | Pretrained | Used By | License |
|-------|-----------|---------|---------|
| `ViT-B-32` | `openai` | i2v_similarity | MIT |

**Note:** open_clip can also use the direct download URL listed in section 2.

### LPIPS

| Network | Used By | License |
|---------|---------|---------|
| `alex` (AlexNet) | i2v_similarity | BSD-2-Clause |

### InsightFace

| Model | Used By | License |
|-------|---------|---------|
| `buffalo_l` (ArcFace) | identity_loss | MIT |

**Note:** Auto-downloaded by InsightFace on first use to `~/.insightface/models/buffalo_l/`. Requires `insightface>=0.7.0` and `onnxruntime>=1.14.0` (optional dependency group `v-identity`).

### DeepFace

| Model | Used By | License |
|-------|---------|---------|
| `ArcFace` | identity_loss (fallback) | MIT |

**Note:** Auto-downloaded by DeepFace on first use to `~/.deepface/weights/`. Already available if `deepface` is installed (used by `celebrity_id` module).

### MediaPipe

| Model | Used By | License |
|-------|---------|---------|
| FaceMesh (468 landmarks) | identity_loss (geometric fallback), face_landmark_quality | Apache 2.0 |

**Note:** Bundled with `mediapipe` package, no separate download needed.

### Ultralytics YOLO

| Model | Used By | License |
|-------|---------|---------|
| `yolov8n.pt` | object_detection, object_permanence | AGPL-3.0 |
| `yolov8s-world.pt` | object_detection (open vocabulary) | AGPL-3.0 |

**Note:** Auto-downloaded by ultralytics on first use.

### rembg (Background Removal)

| Model | Used By | License |
|-------|---------|---------|
| `u2net` | background_diversity | Apache 2.0 |

**Note:** Auto-downloaded by rembg on first use.

---

## 7. Modules with Tiered Fallback Loading

These modules try multiple models in order, falling back to lighter alternatives:

| Module | Tier 1 (preferred) | Tier 2 (fallback) | Tier 3 (minimal) |
|--------|--------------------|--------------------|-------------------|
| dover | Native DOVER.pth | pyiqa "dover" | - |
| sd_reference | SDXL (GPU only) | CLIP text-image proxy | - |
| captioning | BLIP-2 | BLIP base | - |
| finevq | IntMeGroup/FineVQ_score | pyiqa "topiq_nr" | Heuristics |
| kvq | qyp2000/KVQ | pyiqa "topiq_nr" | Heuristics |
| rqvqa | sunwei925/RQ-VQA | pyiqa "clipiqa+" | Heuristics |
| promptiqa | pyiqa "promptiqa" | pyiqa "topiq_nr" | pyiqa "clipiqa+" |
| qcn | pyiqa "qcn" | pyiqa "hyperiqa" | - |
| t2v_score | TIGER-Lab/T2VScore | CLIP (ViT-B/32) | - |
| aigv_assessor | AIGV-Assessor | CLIP (ViT-B/32) | Heuristics |
| object_detection | GRiT (optional) | YOLO v8 | - |
| llm_descriptive_qa | LLaVA 1.6 (local) | OpenAI GPT-4V (API) | - |
| depth_consistency | MiDaS (various sizes) | OpenCV disparity | - |
| motion_amplitude | RAFT Small | OpenCV Farneback | - |
| temporal_flickering | RAFT Small | OpenCV Farneback | - |
| identity_loss | InsightFace ArcFace | DeepFace ArcFace | MediaPipe FaceMesh |
| tifa | ViLT VQA + rule-based QG | CLIP similarity proxy | Heuristic |
| nemo_curator | nvidia/quality-classifier-deberta | FastText | Heuristic |
| umap_projection | UMAP | t-SNE | sklearn PCA / numpy PCA |

---

## License Summary

| License | Models |
|---------|--------|
| **MIT** | CLIP (OpenAI), DOVER, FAST-VQA, open_clip, aesthetic scoring, BLIP-2, X-CLIP, Q-Align, InsightFace, DeepFace |
| **Apache 2.0** | DINOv2, PaddleOCR, NSFW, CLAP (LAION), rembg, FineVQ, VideoScore, ViLT, nvidia/quality-classifier-deberta, MediaPipe |
| **BSD-3-Clause** | RAFT, R3D-18, BLIP base |
| **BSD-2-Clause** | LPIPS |
| **CC-BY-NC 4.0** | VideoMAE, Co-Tracker |
| **CreativeML Open RAIL++-M** | Stable Diffusion XL |
| **AGPL-3.0** | YOLOv8 |
| **LLaMA 2 License** | LLaVA-1.5-7B |
| **Other (Nvidia)** | SegFormer |
| **Research / Gated** | AIGV-Assessor, KVQ, RQ-VQA, VideoAlign-Reward, T2VScore |

---

## Storage Estimates

| Category | Approximate Size |
|----------|-----------------|
| Core models (CLIP B/32, DOVER, RAFT, BLIP base) | ~2.5 GB |
| Action recognition (VideoMAE) | ~1.2 GB |
| DINOv2 + open_clip + LPIPS | ~0.8 GB |
| FAST-VQA (all 3 variants) | ~0.4 GB |
| SDXL (sd_reference) | ~7 GB |
| BLIP-2 (captioning upgrade) | ~15 GB |
| CLIP ViT-L/14 (aesthetic) | ~1.7 GB |
| MiDaS (depth) | ~0.5 GB |
| PyIQA metrics (all) | ~2-5 GB |
| LLaVA 7B (llm_descriptive_qa) | ~15 GB |
| InsightFace buffalo_l (identity_loss) | ~0.3 GB |
| nvidia/quality-classifier-deberta (nemo_curator) | ~1.5 GB |
| Everything else | ~5-10 GB |
| **Total (all modules)** | **~52-62 GB** |
| **Typical usage (10-15 modules)** | **~5-15 GB** |
