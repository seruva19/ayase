# Ayase Models Reference

> **Version 0.1.30** · Generated 2026-04-28 14:36 · **179 models** across **10 sources**
>
> `ayase modules models -o MODELS.md` to regenerate

## Summary

**179** models · **54** HuggingFace · **61** pyiqa · **10** sources

*License labels in this document cover model weights and runtime assets referenced by Ayase modules.*
*They do not describe the license of Ayase source code or vendored third-party source trees.*
*Resolution order: hardcoded source mappings, HuggingFace metadata when available, then parent-repo inheritance for weight files.*

<table width="100%"><tr>
<td width="50%" valign="top"><h4>Models by Source</h4><img src="docs/models_sources.png" width="100%"/></td>
<td width="50%" valign="top"><h4>License Distribution</h4><img src="docs/models_licenses.png" width="100%"/></td>
</tr></table>

<table width="100%"><tr>
<td width="50%" valign="top"><h4>VRAM Tiers</h4><img src="docs/models_vram.png" width="100%"/></td>
<td width="50%" valign="top"><h4>Top Used Models</h4><img src="docs/models_top_used.png" width="100%"/></td>
</tr></table>

**Estimated total download size (all models):** ~81 GB

*Note: Most modules auto-download only the models they need on first use. You rarely need all models at once.*

> [!WARNING]
> **Commercial use:** Stick to modules whose models are marked "Commercial OK" above. Most pyiqa metrics marked "research" are re-implementations under pyiqa's MIT license, but the original training data or architecture may carry restrictions — verify before commercial deployment.

<a id="categories"></a>

[HuggingFace (37)](#huggingface-models) · [Weight Files (17)](#weight-file-repos) · [pyiqa (61)](#pyiqa-metrics) · [torchvision (12)](#torchvision-models) · [CLIP / OpenCLIP (2)](#clip--openclip) · [torch.hub (4)](#torchhub) · [FFmpeg (7)](#ffmpeg) · [pip Packages (15)](#pip-packages) · [Quick Install Guide](#quick-install-guide)

---

## HuggingFace Models

### <a href="https://huggingface.co/AkaneTendo25/ayase-models" target="_blank">`AkaneTendo25/ayase-models`</a> [↑](#categories)

- **Used by**: `rqvqa`

### <a href="https://huggingface.co/Falconsai/nsfw_image_detection" target="_blank">`Falconsai/nsfw_image_detection`</a> [↑](#categories)

- **Used by**: `nsfw`

### <a href="https://huggingface.co/FunAudioLLM/SenseVoiceSmall" target="_blank">`FunAudioLLM/SenseVoiceSmall`</a> [↑](#categories)

- **Used by**: `verse_bench`

### <a href="https://huggingface.co/IntMeGroup/AIGV-Assessor-static_quality" target="_blank">`IntMeGroup/AIGV-Assessor-static_quality`</a> [↑](#categories)

- **Used by**: `aigv_assessor`

### <a href="https://huggingface.co/IntMeGroup/FineVQ_score" target="_blank">`IntMeGroup/FineVQ_score`</a> [↑](#categories)

- **Used by**: `finevq`

### <a href="https://huggingface.co/KlingTeam/VideoReward" target="_blank">`KlingTeam/VideoReward`</a> [↑](#categories)

- **Used by**: `video_reward`

### <a href="https://huggingface.co/MCG-NJU/videomae-large-finetuned-kinetics" target="_blank">`MCG-NJU/videomae-large-finetuned-kinetics`</a> [↑](#categories)

- **Used by**: `action_recognition`
- **VRAM**: ~1.5 GB · **Disk**: ~1.3 GB

### <a href="https://huggingface.co/MizzenAI/HPSv3" target="_blank">`MizzenAI/HPSv3`</a> [↑](#categories)

- **Used by**: `hpsv3`

### <a href="https://huggingface.co/OpenMuQ/MuQ-large-msd-iter" target="_blank">`OpenMuQ/MuQ-large-msd-iter`</a> [↑](#categories)

- **Used by**: `song_eval`

### <a href="https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct" target="_blank">`Qwen/Qwen2-VL-7B-Instruct`</a> [↑](#categories)

- **Used by**: `hpsv3`
- **VRAM**: ~16 GB · **Disk**: ~15 GB

### <a href="https://huggingface.co/Salesforce/blip-image-captioning-base" target="_blank">`Salesforce/blip-image-captioning-base`</a> [↑](#categories)

- **Used by**: `captioning`
- **VRAM**: ~1 GB · **Disk**: ~990 MB

### <a href="https://huggingface.co/TIGER-Lab/VideoScore" target="_blank">`TIGER-Lab/VideoScore`</a> [↑](#categories)

- **Used by**: `videoscore`
- **VRAM**: ~14 GB · **Disk**: ~14 GB

### <a href="https://huggingface.co/TIGER-Lab/VideoScore2" target="_blank">`TIGER-Lab/VideoScore2`</a> [↑](#categories)

- **Used by**: `videoscore2`
- **VRAM**: ~16 GB · **Disk**: ~15 GB

### <a href="https://huggingface.co/ai-forever/kandinsky-video-motion-predictor" target="_blank">`ai-forever/kandinsky-video-motion-predictor`</a> [↑](#categories)

- **Used by**: `kandinsky_motion`

### <a href="https://huggingface.co/dandelin/vilt-b32-finetuned-vqa" target="_blank">`dandelin/vilt-b32-finetuned-vqa`</a> [↑](#categories)

- **Used by**: `commonsense`, `tifa`
- **VRAM**: ~500 MB · **Disk**: ~450 MB

### <a href="https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf" target="_blank">`depth-anything/Depth-Anything-V2-Small-hf`</a> [↑](#categories)

- **Used by**: `depth_anything`, `t2v_compbench`
- **VRAM**: ~200 MB · **Disk**: ~100 MB

### <a href="https://huggingface.co/facebook/dinov2-base" target="_blank">`facebook/dinov2-base`</a> [↑](#categories)

- **Used by**: `subject_consistency`

### <a href="https://huggingface.co/facebook/dinov2-large" target="_blank">`facebook/dinov2-large`</a> [↑](#categories)

- **Used by**: `verse_bench`

### <a href="https://huggingface.co/facebook/vjepa2-vitg-fpc64-256" target="_blank">`facebook/vjepa2-vitg-fpc64-256`</a> [↑](#categories)

- **Used by**: `jedi`, `jedi_metric`

### `fsmn-vad` [↑](#categories)

- **Used by**: `verse_bench`

### <a href="https://huggingface.co/google/siglip-so400m-patch14-384" target="_blank">`google/siglip-so400m-patch14-384`</a> [↑](#categories)

- **Used by**: `verse_bench`

### <a href="https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K" target="_blank">`laion/CLIP-ViT-H-14-laion2B-s32B-b79K`</a> [↑](#categories)

- **Used by**: `pickscore`

### <a href="https://huggingface.co/laion/clap-htsat-fused" target="_blank">`laion/clap-htsat-fused`</a> [↑](#categories)

- **Used by**: `audio_text_alignment`
- **VRAM**: ~600 MB · **Disk**: ~600 MB

### <a href="https://huggingface.co/lero233/KVQ" target="_blank">`lero233/KVQ`</a> [↑](#categories)

- **Used by**: `kvq`

### <a href="https://huggingface.co/llava-hf/llava-1.5-7b-hf" target="_blank">`llava-hf/llava-1.5-7b-hf`</a> [↑](#categories)

- **Used by**: `commonsense`, `creativity`, `vlm_judge`
- **VRAM**: ~14 GB · **Disk**: ~14 GB

### <a href="https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf" target="_blank">`llava-hf/llava-v1.6-mistral-7b-hf`</a> [↑](#categories)

- **Used by**: `llm_descriptive_qa`
- **VRAM**: ~14 GB · **Disk**: ~14 GB

### <a href="https://huggingface.co/microsoft/xclip-base-patch32" target="_blank">`microsoft/xclip-base-patch32`</a> [↑](#categories)

- **Used by**: `embedding`, `video_text_matching`
- **VRAM**: ~600 MB · **Disk**: ~600 MB

### <a href="https://huggingface.co/nvidia/quality-classifier-deberta" target="_blank">`nvidia/quality-classifier-deberta`</a> [↑](#categories)

- **Used by**: `nemo_curator`

### <a href="https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512" target="_blank">`nvidia/segformer-b0-finetuned-ade-512-512`</a> [↑](#categories)

- **Used by**: `semantic_segmentation_consistency`

### <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">`openai/clip-vit-base-patch32`</a> [↑](#categories)

- **Used by**: `action_recognition`, `aigcvqa`, `aigv_assessor`, `aigvqa`, `background_consistency`, `chronomagic`, `clifvqa`, `clip_temporal`, `clipvqa`, `concept_presence`, `crave`, `creativity`, `dataset_analytics`, `deepfake_detection`, `generative_distribution`, `harmful_content`, `maxvqa`, `ptmvqa`, `qclip`, `scene_tagging`, `sd_reference`, `semantic_alignment`, `t2v_compbench`, `t2v_score`, `t2veval`, `tifa`, `ugvq`, `umap_projection`, `umtscore`, `video_memorability`, `video_text_matching`, `video_type_classifier`, `videoreward`, `vqa2`, `vqathinker`, `vqinsight`, `world_consistency`
- **VRAM**: ~600 MB · **Disk**: ~600 MB

### <a href="https://huggingface.co/openai/clip-vit-large-patch14" target="_blank">`openai/clip-vit-large-patch14`</a> [↑](#categories)

- **Used by**: `aesthetic_scoring`, `vader`
- **VRAM**: ~1.5 GB · **Disk**: ~1.7 GB

### <a href="https://huggingface.co/q-future/one-align" target="_blank">`q-future/one-align`</a> [↑](#categories)

- **Used by**: `q_align`

### `roberta-base` [↑](#categories)

- **Used by**: `verse_bench`

### <a href="https://huggingface.co/song_eval/model.safetensors" target="_blank">`song_eval/model.safetensors`</a> [↑](#categories)

- **Used by**: `song_eval`

### <a href="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0" target="_blank">`stabilityai/stable-diffusion-xl-base-1.0`</a> [↑](#categories)

- **Used by**: `sd_reference`

### <a href="https://huggingface.co/xinyu1205/recognize-anything-plus-model" target="_blank">`xinyu1205/recognize-anything-plus-model`</a> [↑](#categories)

- **Used by**: `ram_tagging`

### <a href="https://huggingface.co/yuvalkirstain/PickScore_v1" target="_blank">`yuvalkirstain/PickScore_v1`</a> [↑](#categories)

- **Used by**: `pickscore`

## Weight File Repos

### <a href="https://huggingface.co/AkaneTendo25/ayase-models" target="_blank">`AkaneTendo25/ayase-models`</a> [↑](#categories)
> Pre-trained weight files for ayase modules

- `CLIPIQA+_ViTL14_512-e66488f2.pth` — used by `brightrate`
- `CONTRIQUE_checkpoint25.tar` — used by `brightrate`
- `DOVER.pth` — used by `dover`
- `FAST_VQA_3D_1_1.pth` — used by `fast_vqa`
- `FAST_VQA_B_1_4.pth` — used by `fast_vqa`
- `FAST_VQA_M_1_4.pth` — used by `fast_vqa`
- `ViT-B-32.safetensors` — used by `brightrate`
- `ViT-B-32.safetensors` — used by `i2v_similarity`
- `ViT-L-14.safetensors` — used by `brightrate`
- `alex.pth` — used by `i2v_similarity`
- `brightrate_brightvq.pt` — used by `brightrate`
- `convnext_tiny_1k_224_ema.pth` — used by `dover`
- `dinov2_vitb14_pretrain.pth` — used by `i2v_similarity`
- `flownet.pkl` — used by `motion_smoothness`
- `frames_modelparameters.mat` — used by `brightrate`
- `onnx_dover.onnx` — used by `dover`
- `sac+logos+ava1-l14-linearMSE.pth` — used by `aesthetic_scoring`

## pyiqa Metrics (61)

<a href="https://github.com/chaofengc/IQA-PyTorch" target="_blank">pyiqa</a> is an MIT-licensed collection of image/video quality metrics. Weights auto-download on first `pyiqa.create_metric()` call. `pip install pyiqa`

| Metric | Task | License | Commercial | Used By |
|--------|------|---------|------------|---------|
| `afine` | Adaptive fidelity-naturalness IQA | research | — | `afine` |
| `afine_nr` | A-FINE NR fidelity-naturalness | research | — | `afine` |
| `ahiq` | Attention-based hybrid FR-IQA | research | — | `ahiq` |
| `arniqa` | Artifact-aware NR-IQA | research | — | `arniqa` |
| `brisque` | Blind naturalness statistics NR-IQA | BSD-2-Clause (OpenCV) | Yes | `brisque`, `naturalness` |
| `bvqi` | Zero-shot blind VQA | research | — | `bvqi` |
| `ckdn` | Conditional knowledge distillation FR-IQA | research | — | `ckdn` |
| `clip_iqa` | CLIP image quality assessment | research | — | `clip_iqa` |
| `clipiqa+` | CLIP-based image quality assessment | MIT (pyiqa) | Yes | `clip_iqa`, `promptiqa`, `rqvqa` |
| `cnniqa` | CNN-based blind image quality | research | — | `cnniqa` |
| `compare2score` | Comparative-to-absolute quality scoring | research | — | `compare2score` |
| `contrique` | Contrastive image quality representation | research | — | `contrique` |
| `conviqt` | Contrastive NR-VQA | research | — | `conviqt` |
| `cover` | Comprehensive video evaluation and rating | research | — | `cover` |
| `creativity` | Creative quality assessment | research | — | `creativity` |
| `cw_ssim` | Complex wavelet SSIM | MIT (pyiqa) | Yes | `cw_ssim` |
| `dbcnn` | Deep bilinear CNN for blind IQA | research | — | `dbcnn` |
| `deepdc` | Deep distribution conformance | research | — | `deepdc` |
| `deepwsd` | Deep Wasserstein distance IQA | research | — | `deepwsd` |
| `dmm` | Detail model metric FR-IQA | research | — | `dmm` |
| `dover` | Disentangled objective video evaluation | MIT (pyiqa) | Yes | `cover`, `dover` |
| `face_iqa` | TOPIQ face-specific quality | research | — | `face_iqa` |
| `finevq` | Fine-grained UGC video quality | research | — | `finevq` |
| `hyperiqa` | Adaptive hypernetwork NR-IQA | research | — | `hyperiqa`, `qcn` |
| `ilniqe` | Integrated local NIQE | BSD-2-Clause | Yes | `ilniqe` |
| `kvq` | Key-frame saliency-guided VQA | research | — | `kvq` |
| `laion_aes` | LAION aesthetic scoring (CLIP-based) | MIT | Yes | `creativity`, `laion_aesthetic` |
| `laion_aesthetic` | LAION Aesthetics V2 predictor | research | — | `laion_aesthetic` |
| `liqe` | Learned image quality evaluator (multi-task) | research | — | `liqe` |
| `maclip` | Multi-attribute CLIP quality scoring | research | — | `maclip` |
| `mad` | Most apparent distortion FR-IQA | research | — | `mad` |
| `maniqa` | Multi-dimension attention NR-IQA | Apache-2.0 | Yes | `maniqa` |
| `mdtvsfa` | Multi-dimensional temporal-spatial VQA | research | — | `mdtvsfa` |
| `msswd` | Multi-scale sliced Wasserstein distance | research | — | `msswd` |
| `musiq` | Multi-scale image quality transformer | Apache-2.0 (Google) | Yes | `musiq` |
| `naturalness` | Natural scene statistics | research | — | `naturalness` |
| `nima` | Neural image assessment (aesthetic + technical) | Apache-2.0 (Google) | Yes | `nima` |
| `niqe` | Natural image quality evaluator (statistics-based) | BSD-2-Clause (OpenCV) | Yes | `niqe` |
| `nlpd` | Normalized Laplacian pyramid distance | research | — | `nlpd` |
| `nrqm` | No-reference quality metric | research | — | `nrqm` |
| `paq2piq` | Patches-as-questions for image quality | research | — | `paq2piq` |
| `pi` | Perceptual index (PIRM challenge) | research | — | `pi` |
| `pieapp` | Pairwise learned perceptual distance | research | — | `pieapp` |
| `piqe` | Perception-based blind NR-IQA | BSD-2-Clause | Yes | `piqe` |
| `promptiqa` | Few-shot prompt-based NR-IQA | research | — | `promptiqa` |
| `qalign` | IQA | research | — | `vqa2` |
| `qcn` | Geometric order blind IQA | research | — | `qcn` |
| `qualiclip` | Quality-aware CLIP embeddings | research | — | `qualiclip` |
| `rqvqa` | Rich quality-aware VQA | research | — | `rqvqa` |
| `sfid` | Spatial FID distribution metric | research | — | `sfid` |
| `ssimc` | Complex wavelet SSIM-C FR | MIT (pyiqa) | Yes | `ssimc` |
| `topiq` | TOPIQ transformer quality | research | — | `topiq` |
| `topiq_fr` | Transformer-based FR image quality | MIT (pyiqa) | Yes | `topiq_fr` |
| `topiq_nr` | Transformer-based NR image quality | MIT (pyiqa) | Yes | `finevq`, `kvq`, `promptiqa`, `topiq` |
| `topiq_nr-face` | TOPIQ face-specific quality | MIT (pyiqa) | Yes | `face_iqa` |
| `tres` | Transformer relative quality estimation | research | — | `tres` |
| `unique` | Unified NR-IQA with contrastive learning | research | — | `unique` |
| `vqa2` | IQA | research | — | `vqa2` |
| `wadiqam` | Weighted average deep IQA | research | — | `wadiqam` |
| `wadiqam_fr` | Weighted average deep FR-IQA | research | — | `wadiqam_fr` |
| `wadiqam_nr` | Weighted average deep NR-IQA | research | — | `wadiqam` |

## torchvision Models

Bundled with `pip install torchvision`. Weights download on first use.

### `torchvision/inception_v3` [↑](#categories)
> torchvision · BSD-3-Clause

- **Used by**: `inception_score`, `kid`
- **VRAM**: ~200 MB · **Disk**: ~100 MB

### `torchvision/r3d_18` [↑](#categories)
> torchvision · BSD-3-Clause

- **Used by**: `c3dvqa`, `fvd`, `gamival`
- **VRAM**: ~200 MB · **Disk**: ~130 MB

### `torchvision/raft_large` [↑](#categories)
> torchvision · BSD-3-Clause

- **Used by**: `advanced_flow`, `raft_motion`
- **VRAM**: ~200 MB · **Disk**: ~20 MB

### `torchvision/raft_small` [↑](#categories)
> torchvision · BSD-3-Clause

- **Used by**: `advanced_flow`, `flolpips`, `motion_amplitude`, `temporal_flickering`
- **VRAM**: ~100 MB · **Disk**: ~20 MB

### `torchvision/resnet18` [↑](#categories)
> torchvision · BSD-3-Clause

- **Used by**: `tlvqm`
- **VRAM**: ~100 MB · **Disk**: ~45 MB

### `torchvision/resnet50` [↑](#categories)
> torchvision

- **Used by**: `adadqa`, `discovqa`, `faver`, `grafiqs`, `internvqa`, `lmmvqa`, `mc360iqa`, `mdvqa`, `memoryvqa`, `mm_pcqa`, `modularbvqa`, `oavqa`, `p1204`, `presresq`, `provqa`, `ptmvqa`, `pvmaf`, `rapique`, `sama`, `serfiq`, `siamvqa`, `sr4kvqa`, `stablevqa`, `unified_vqa`, `unqa`, `vsfa`, `watermark_classifier`, `zoomvqa`

### `torchvision/swin_b` [↑](#categories)
> torchvision

- **Used by**: `simplevqa`

### `torchvision/inception_v3` [↑](#categories)
> torchvision · BSD-3-Clause

- **Used by**: `kid`

### `torchvision/r3d_18` [↑](#categories)
> torchvision · BSD-3-Clause

- **Used by**: `fvd`, `kvd`

### `torchvision/resnet50` [↑](#categories)
> torchvision

- **Used by**: `unified_vqa`

### `torchvision/vgg16` [↑](#categories)
> torchvision

- **Used by**: `avqt`, `deepvqa`

### `torchvision/video` [↑](#categories)
> torchvision · BSD-3-Clause

- **Used by**: `c3dvqa`, `gamival`

## CLIP / OpenCLIP

### `CLIP ViT-B-32` [↑](#categories)

- **Used by**: `i2v_similarity`

### `CLIP ViT-B/32` [↑](#categories)
> MIT (OpenAI)

- **Used by**: `lmmvqa`, `mdvqa`, `modularbvqa`, `presresq`, `unified_vqa`, `vqa_score`
- **VRAM**: ~600 MB · **Disk**: ~340 MB

## torch.hub

### `facebookresearch/co-tracker` [↑](#categories)
> torch.hub · Apache-2.0

- **Used by**: `dynamics_controllability`, `physics`, `trajan`

### `facebookresearch/dinov2` [↑](#categories)
> torch.hub · Apache-2.0

- **Used by**: `dino_face_identity`, `ptmvqa`, `spectral_complexity`, `video_memorability`, `world_consistency`

### `intel-isl/MiDaS` [↑](#categories)
> torch.hub · MIT

- **Used by**: `depth_consistency`, `depth_map_quality`
- **VRAM**: ~400 MB · **Disk**: ~400 MB

### `tarepan/SpeechMOS:v1.2.0` [↑](#categories)
> torch.hub · MIT

- **Used by**: `audio_utmos`
- **VRAM**: ~200 MB · **Disk**: ~100 MB

## FFmpeg

Require FFmpeg compiled with libvmaf. No separate download needed.

### `ffmpeg/cambi` [↑](#categories)
> built-in · BSD-2-Clause (Netflix)

- **Used by**: `cambi`

### `ffmpeg/libvmaf` [↑](#categories)
> built-in · BSD-2-Clause (Netflix)

- **Used by**: `cambi`, `vmaf`, `vmaf_4k`, `vmaf_neg`, `vmaf_phone`

### `ffmpeg/vmaf_4k_v0.6.1` [↑](#categories)
> built-in · BSD-2-Clause (Netflix)

- **Used by**: `vmaf_4k`

### `ffmpeg/vmaf_phone_model` [↑](#categories)
> built-in · BSD-2-Clause (Netflix)

- **Used by**: `vmaf_phone`

### `ffmpeg/vmaf_v0.6.1` [↑](#categories)
> built-in · BSD-2-Clause (Netflix)

- **Used by**: `vmaf_neg`, `vmaf_phone`

### `ffmpeg/vmaf_v0.6.1neg` [↑](#categories)
> built-in · BSD-2-Clause (Netflix)

- **Used by**: `vmaf_neg`

### `ffmpeg/xpsnr` [↑](#categories)
> built-in · BSD (FFmpeg)

- **Used by**: `xpsnr`

## pip Packages

### `aesthetic-predictor-v2-5` [↑](#categories)
> Aesthetic Predictor V2.5 (SigLIP)

- **Used by**: `aesthetic`
- **Install**: `pip install aesthetic-predictor-v2-5`

### `deepface` [↑](#categories)
> DeepFace (face recognition/verification)

- **Used by**: `celebrity_id`, `face_cross_similarity`, `identity_loss`
- **Install**: `pip install deepface`

### `dreamsim` [↑](#categories)
> DreamSim CLIP+DINO similarity

- **Used by**: `dreamsim`
- **Install**: `pip install dreamsim`

### `erqa` [↑](#categories)
> ERQA edge restoration quality

- **Used by**: `erqa`
- **Install**: `pip install erqa`

### `fasttext` [↑](#categories)
> FastText (text classification)

- **Used by**: `nemo_curator`
- **Install**: `pip install fasttext`

### `insightface` [↑](#categories)
> InsightFace (face recognition)

- **Used by**: `concept_presence`, `crfiqa`, `dino_face_identity`, `face_cross_similarity`, `grafiqs`, `identity_loss`, `magface`, `serfiq`
- **Install**: `pip install insightface`

### `joblib` [↑](#categories)
> Joblib (serialized model storage)

- **Used by**: `brightrate`, `chipqa`, `gamival`, `hdr_chipqa`, `hdrmax`, `tlvqm`, `videval`
- **Install**: `pip install joblib`

### `jxlpy` [↑](#categories)
> JPEG XL codec library

- **Used by**: `butteraugli`
- **Install**: `pip install jxlpy`

### `mediapipe` [↑](#categories)
> MediaPipe (face/pose/hand detection)

- **Used by**: `concept_presence`, `face_cross_similarity`, `face_fidelity`, `face_landmark_quality`, `human_fidelity`, `identity_loss`
- **Install**: `pip install mediapipe`

### `onnxruntime` [↑](#categories)
> ONNX Runtime (model inference)

- **Used by**: `dover`
- **Install**: `pip install onnxruntime`

### `piq` [↑](#categories)
> piq (PyTorch Image Quality)

- **Used by**: `dists`, `perceptual_fr`, `vif`
- **Install**: `pip install piq`

### `ptlflow` [↑](#categories)
> ptlflow optical flow models

- **Used by**: `ptlflow_motion`
- **Install**: `pip install ptlflow`

### `stlpips-pytorch` [↑](#categories)
> ST-LPIPS spatiotemporal perceptual

- **Used by**: `st_lpips`
- **Install**: `pip install stlpips-pytorch`

### `torchmetrics[audio]` [↑](#categories)
> TorchMetrics (DNSMOS, etc.)

- **Used by**: `dnsmos`
- **Install**: `pip install torchmetrics[audio]`

### `ultralytics` [↑](#categories)
> YOLOv8 object detection

- **Used by**: `object_detection`, `object_permanence`, `t2v_compbench`
- **Install**: `pip install ultralytics`

## Quick Install Guide

Install Ayase with the bundled runtime dependencies:

```bash
pip install ayase
```
