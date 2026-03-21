# Ayase Models Reference

> **Version 0.1.17** · Generated 2026-03-21 18:45 · **130 models** across **7 sources**
>
> `ayase modules models -o MODELS.md` to regenerate

## Summary

**130** models · **37** HuggingFace · **59** pyiqa · **7** sources

<table width="100%"><tr>
<td width="50%" valign="top"><h4>Models by Source</h4><img src="docs/models_sources.png" width="100%"/></td>
<td width="50%" valign="top"><h4>License Distribution</h4><img src="docs/models_licenses.png" width="100%"/></td>
</tr></table>

<table width="100%"><tr>
<td width="50%" valign="top"><h4>VRAM Tiers</h4><img src="docs/models_vram.png" width="100%"/></td>
<td width="50%" valign="top"><h4>Top Used Models</h4><img src="docs/models_top_used.png" width="100%"/></td>
</tr></table>

**Estimated total download size (all models):** ~82 GB

*Note: Most modules auto-download only the models they need on first use. You rarely need all models at once.*

> [!WARNING]
> **Commercial use:** Stick to modules whose models are marked "Commercial OK" above. Most pyiqa metrics marked "research" are re-implementations under pyiqa's MIT license, but the original training data or architecture may carry restrictions — verify before commercial deployment.

<a id="categories"></a>

[HuggingFace (26)](#huggingface-models) · [Weight Files (11)](#weight-file-repos) · [pyiqa (59)](#pyiqa-metrics) · [torchvision (7)](#torchvision-models) · [CLIP / OpenCLIP (1)](#clip--openclip) · [torch.hub (4)](#torchhub) · [FFmpeg (7)](#ffmpeg) · [pip Packages (15)](#pip-packages) · [Quick Install Guide](#quick-install-guide)

---

## HuggingFace Models

### [`Falconsai/nsfw_image_detection`](https://huggingface.co/Falconsai/nsfw_image_detection) [↑](#categories)
> image-classification · apache-2.0

- **Used by**: `nsfw`
- **Downloads**: 41.7M
- **Source**: [arXiv](https://arxiv.org/abs/2010.11929)

### [`IntMeGroup/FineVQ_score`](https://huggingface.co/IntMeGroup/FineVQ_score) [↑](#categories)
> apache-2.0

- **Used by**: `finevq`
- **Parameters**: 8.2B · **Downloads**: 671
- **Disk**: ~30.5 GB

### [`KlingTeam/VideoAlign-Reward`](https://huggingface.co/KlingTeam/VideoAlign-Reward) [↑](#categories)

- **Used by**: `video_reward`

### [`MCG-NJU/videomae-large-finetuned-kinetics`](https://huggingface.co/MCG-NJU/videomae-large-finetuned-kinetics) [↑](#categories)
> video-classification · cc-by-nc-4.0

- **Used by**: `action_recognition`
- **Parameters**: 304M · **Downloads**: 2K
- **VRAM**: ~1.5 GB · **Disk**: ~1.3 GB
- **Source**: [arXiv](https://arxiv.org/abs/2203.12602)

### [`Salesforce/blip-image-captioning-base`](https://huggingface.co/Salesforce/blip-image-captioning-base) [↑](#categories)
> image-to-text · bsd-3-clause

- **Used by**: `captioning`
- **Downloads**: 3.5M
- **VRAM**: ~1 GB · **Disk**: ~990 MB
- **Source**: [arXiv](https://arxiv.org/abs/2201.12086)

### [`TIGER-Lab/T2VScore`](https://huggingface.co/TIGER-Lab/T2VScore) [↑](#categories)

- **Used by**: `t2v_score`

### [`TIGER-Lab/VideoScore`](https://huggingface.co/TIGER-Lab/VideoScore) [↑](#categories)
> visual-question-answering · apache-2.0

- **Used by**: `videoscore`
- **Parameters**: 8.3B · **Downloads**: 2K
- **VRAM**: ~14 GB · **Disk**: ~14 GB
- **Source**: [arXiv](https://arxiv.org/abs/2406.15252)

### [`ai-forever/kandinsky-video-motion-predictor`](https://huggingface.co/ai-forever/kandinsky-video-motion-predictor) [↑](#categories)

- **Used by**: `kandinsky_motion`
- **Downloads**: 86K

### [`dandelin/vilt-b32-finetuned-vqa`](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa) [↑](#categories)
> visual-question-answering · apache-2.0

- **Used by**: `commonsense`, `tifa`
- **Downloads**: 73K
- **VRAM**: ~500 MB · **Disk**: ~450 MB
- **Source**: [arXiv](https://arxiv.org/abs/2102.03334)

### [`depth-anything/Depth-Anything-V2-Small-hf`](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) [↑](#categories)
> depth-estimation · apache-2.0

- **Used by**: `depth_anything`, `t2v_compbench`
- **Parameters**: 25M · **Downloads**: 1.4M
- **VRAM**: ~200 MB · **Disk**: ~100 MB
- **Source**: [arXiv](https://arxiv.org/abs/2406.09414)

### [`facebook/dinov2-base`](https://huggingface.co/facebook/dinov2-base) [↑](#categories)
> image-feature-extraction · apache-2.0

- **Used by**: `subject_consistency`
- **Parameters**: 87M · **Downloads**: 1.2M
- **Disk**: ~330 MB
- **Source**: [arXiv](https://arxiv.org/abs/2304.07193)

### [`facebook/vjepa-giant`](https://huggingface.co/facebook/vjepa-giant) [↑](#categories)

- **Used by**: `jedi`

### [`laion/clap-htsat-fused`](https://huggingface.co/laion/clap-htsat-fused) [↑](#categories)
> audio-classification · apache-2.0

- **Used by**: `audio_text_alignment`
- **Parameters**: 154M · **Downloads**: 25.8M
- **VRAM**: ~600 MB · **Disk**: ~600 MB
- **Source**: [arXiv](https://arxiv.org/abs/2211.06687)

### [`llava-hf/llava-1.5-7b-hf`](https://huggingface.co/llava-hf/llava-1.5-7b-hf) [↑](#categories)
> image-text-to-text · llama2

- **Used by**: `commonsense`, `creativity`, `vlm_judge`
- **Parameters**: 7.1B · **Downloads**: 4.7M
- **VRAM**: ~14 GB · **Disk**: ~14 GB

### [`llava-hf/llava-v1.6-mistral-7b-hf`](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf) [↑](#categories)
> image-text-to-text · apache-2.0

- **Used by**: `llm_descriptive_qa`
- **Parameters**: 7.6B · **Downloads**: 683K
- **VRAM**: ~14 GB · **Disk**: ~14 GB
- **Source**: [arXiv](https://arxiv.org/abs/2310.03744)

### [`microsoft/xclip-base-patch32`](https://huggingface.co/microsoft/xclip-base-patch32) [↑](#categories)
> video-classification · mit

- **Used by**: `embedding`, `video_text_matching`
- **Parameters**: 197M · **Downloads**: 140K
- **VRAM**: ~600 MB · **Disk**: ~600 MB
- **Source**: [arXiv](https://arxiv.org/abs/2208.02816)

### [`nvidia/quality-classifier-deberta`](https://huggingface.co/nvidia/quality-classifier-deberta) [↑](#categories)
> apache-2.0

- **Used by**: `nemo_curator`
- **Downloads**: 3K
- **Source**: [arXiv](https://arxiv.org/abs/2111.09543)

### [`nvidia/segformer-b0-finetuned-ade-512-512`](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512) [↑](#categories)
> image-segmentation · other

- **Used by**: `semantic_segmentation_consistency`
- **Parameters**: 4M · **Downloads**: 503K
- **Disk**: ~14 MB
- **Source**: [arXiv](https://arxiv.org/abs/2105.15203)

### [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32) [↑](#categories)
> zero-shot-image-classification

- **Used by**: `action_recognition`, `aigv_assessor`, `background_consistency`, `chronomagic`, `clip_temporal`, `clipvqa`, `creativity`, `dataset_analytics`, `deepfake_detection`, `generative_distribution`, `harmful_content`, `maxvqa`, `scene_tagging`, `sd_reference`, `semantic_alignment`, `t2v_compbench`, `t2v_score`, `tifa`, `umap_projection`, `umtscore`, `video_memorability`, `video_text_matching`, `video_type_classifier`
- **Downloads**: 20.0M
- **VRAM**: ~600 MB · **Disk**: ~600 MB
- **Source**: [arXiv](https://arxiv.org/abs/2103.00020)

### [`openai/clip-vit-large-patch14`](https://huggingface.co/openai/clip-vit-large-patch14) [↑](#categories)
> zero-shot-image-classification

- **Used by**: `aesthetic_scoring`
- **Parameters**: 428M · **Downloads**: 20.2M
- **VRAM**: ~1.5 GB · **Disk**: ~1.7 GB
- **Source**: [arXiv](https://arxiv.org/abs/2103.00020)

### [`q-future/one-align`](https://huggingface.co/q-future/one-align) [↑](#categories)
> zero-shot-image-classification · mit

- **Used by**: `q_align`
- **Downloads**: 310K
- **Source**: [arXiv](https://arxiv.org/abs/2312.17090)

### [`qyp2000/KVQ`](https://huggingface.co/qyp2000/KVQ) [↑](#categories)

- **Used by**: `kvq`

### [`stabilityai/stable-diffusion-xl-base-1.0`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) [↑](#categories)
> text-to-image · openrail++

- **Used by**: `sd_reference`
- **Downloads**: 2.2M
- **Source**: [arXiv](https://arxiv.org/abs/2307.01952)

### [`sunwei925/RQ-VQA`](https://huggingface.co/sunwei925/RQ-VQA) [↑](#categories)

- **Used by**: `rqvqa`

### [`wangjiarui153/AIGV-Assessor`](https://huggingface.co/wangjiarui153/AIGV-Assessor) [↑](#categories)

- **Used by**: `aigv_assessor`

### [`xinyu1205/recognize-anything-plus-model`](https://huggingface.co/xinyu1205/recognize-anything-plus-model) [↑](#categories)
> zero-shot-image-classification · apache-2.0

- **Used by**: `ram_tagging`
- **Source**: [arXiv](https://arxiv.org/abs/2306.03514)

## Weight File Repos

### [`AkaneTendo25/ayase-models`](https://huggingface.co/AkaneTendo25/ayase-models) [↑](#categories)
> Pre-trained weight files for ayase modules

- `DOVER.pth` — used by `dover`
- `FAST_VQA_3D_1_1.pth` — used by `fast_vqa`
- `FAST_VQA_B_1_4.pth` — used by `fast_vqa`
- `FAST_VQA_M_1_4.pth` — used by `fast_vqa`
- `ViT-B-32.pt` — used by `i2v_similarity`
- `alex.pth` — used by `i2v_similarity`
- `convnext_tiny_1k_224_ema.pth` — used by `dover`
- `dinov2_vitb14_pretrain.pth` — used by `i2v_similarity`
- `flownet.pkl` — used by `motion_smoothness`
- `onnx_dover.onnx` — used by `dover`
- `sac+logos+ava1-l14-linearMSE.pth` — used by `aesthetic_scoring`

## pyiqa Metrics (59)

[pyiqa](https://github.com/chaofengc/IQA-PyTorch) is an MIT-licensed collection of image/video quality metrics. Weights auto-download on first `pyiqa.create_metric()` call. `pip install pyiqa`

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
| `deepwsd` | Deep Wasserstein distance FR-IQA | research | — | `deepwsd` |
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
| `wadiqam` | Weighted average deep IQA | research | — | `wadiqam` |
| `wadiqam_fr` | Weighted average deep FR-IQA | research | — | `wadiqam_fr` |
| `wadiqam_nr` | Weighted average deep NR-IQA | research | — | `wadiqam` |

## torchvision Models

Bundled with `pip install torchvision`. Weights download on first use.

### `torchvision/inception_v3` [↑](#categories)
> torchvision · BSD-3-Clause

- **Used by**: `inception_score`
- **VRAM**: ~200 MB · **Disk**: ~100 MB

### `torchvision/r3d_18` [↑](#categories)
> torchvision · BSD-3-Clause

- **Used by**: `c3dvqa`, `fvd`
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

- **Used by**: `watermark_classifier`

### `torchvision/video` [↑](#categories)
> torchvision · BSD-3-Clause

- **Used by**: `c3dvqa`

## CLIP / OpenCLIP

### `CLIP ViT-B/32` [↑](#categories)
> MIT (OpenAI)

- **Used by**: `vqa_score`
- **VRAM**: ~600 MB · **Disk**: ~340 MB

## torch.hub

### `facebookresearch/co-tracker` [↑](#categories)
> torch.hub · Apache-2.0

- **Used by**: `dynamics_controllability`, `physics`, `trajan`

### `facebookresearch/dinov2` [↑](#categories)
> torch.hub · Apache-2.0

- **Used by**: `spectral_complexity`, `video_memorability`

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

- **Used by**: `celebrity_id`, `identity_loss`
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

- **Used by**: `identity_loss`
- **Install**: `pip install insightface`

### `joblib` [↑](#categories)
> Joblib (serialized model storage)

- **Used by**: `tlvqm`, `videval`
- **Install**: `pip install joblib`

### `jxlpy` [↑](#categories)
> JPEG XL codec library

- **Used by**: `butteraugli`
- **Install**: `pip install jxlpy`

### `mediapipe` [↑](#categories)
> MediaPipe (face/pose/hand detection)

- **Used by**: `face_fidelity`, `face_landmark_quality`, `human_fidelity`, `identity_loss`
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

Install all model dependencies at once:

```bash
# Core (covers ~80% of modules)
pip install torch torchvision pyiqa piq opencv-python Pillow transformers

# Audio metrics
pip install librosa soundfile pesq pystoi

# Additional NR/FR metrics
pip install lpips dreamsim ssimulacra2 stlpips-pytorch

# Video-specific
pip install decord scenedetect

# Optional heavy models (LLaVA, Q-Align)
pip install accelerate bitsandbytes  # for efficient LLM loading
```
