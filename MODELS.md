# Ayase Models Reference

> **Version 0.1.17** · Generated 2026-03-21 17:47 · **124 models** across **7 sources**
>
> `ayase modules models -o MODELS.md` to regenerate

## Summary

**124** models · **38** HuggingFace · **59** pyiqa · **7** sources

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

[HuggingFace (28)](#huggingface-models) · [Weight Files (10)](#weight-file-repos) · [pyiqa (59)](#pyiqa-metrics) · [torchvision (7)](#torchvision-models) · [CLIP / OpenCLIP (1)](#clip--openclip) · [torch.hub (4)](#torchhub) · [FFmpeg (7)](#ffmpeg) · [pip Packages (8)](#pip-packages) · [Quick Install Guide](#quick-install-guide)

---

## HuggingFace Models

### [`Falconsai/nsfw_image_detection`](https://huggingface.co/Falconsai/nsfw_image_detection)
> image-classification · apache-2.0

- **Used by**: `nsfw`
- **Downloads**: 41.7M
- **Source**: [arXiv](https://arxiv.org/abs/2010.11929)

### [`IntMeGroup/FineVQ_score`](https://huggingface.co/IntMeGroup/FineVQ_score)
> apache-2.0

- **Used by**: `finevq`
- **Parameters**: 8.2B · **Downloads**: 671
- **Disk**: ~30.5 GB

### [`KlingTeam/VideoAlign-Reward`](https://huggingface.co/KlingTeam/VideoAlign-Reward)

- **Used by**: `video_reward`

### [`MCG-NJU/videomae-large-finetuned-kinetics`](https://huggingface.co/MCG-NJU/videomae-large-finetuned-kinetics)
> video-classification · cc-by-nc-4.0

- **Used by**: `action_recognition`
- **Parameters**: 304M · **Downloads**: 2K
- **VRAM**: ~1.5 GB · **Disk**: ~1.3 GB
- **Source**: [arXiv](https://arxiv.org/abs/2203.12602)

### [`Salesforce/blip-image-captioning-base`](https://huggingface.co/Salesforce/blip-image-captioning-base)
> image-to-text · bsd-3-clause

- **Used by**: `captioning`
- **Downloads**: 3.5M
- **VRAM**: ~1 GB · **Disk**: ~990 MB
- **Source**: [arXiv](https://arxiv.org/abs/2201.12086)

### [`TIGER-Lab/T2VScore`](https://huggingface.co/TIGER-Lab/T2VScore)

- **Used by**: `t2v_score`

### [`TIGER-Lab/VideoScore`](https://huggingface.co/TIGER-Lab/VideoScore)
> visual-question-answering · apache-2.0

- **Used by**: `videoscore`
- **Parameters**: 8.3B · **Downloads**: 2K
- **VRAM**: ~14 GB · **Disk**: ~14 GB
- **Source**: [arXiv](https://arxiv.org/abs/2406.15252)

### [`ai-forever/kandinsky-video-tools`](https://huggingface.co/ai-forever/kandinsky-video-tools)

- **Used by**: `kandinsky_motion`

### [`dandelin/vilt-b32-finetuned-vqa`](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa)
> visual-question-answering · apache-2.0

- **Used by**: `commonsense`, `tifa`
- **Downloads**: 73K
- **VRAM**: ~500 MB · **Disk**: ~450 MB
- **Source**: [arXiv](https://arxiv.org/abs/2102.03334)

### [`depth-anything/Depth-Anything-V2-Small-hf`](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)
> depth-estimation · apache-2.0

- **Used by**: `depth_anything`, `t2v_compbench`
- **Parameters**: 25M · **Downloads**: 1.4M
- **VRAM**: ~200 MB · **Disk**: ~100 MB
- **Source**: [arXiv](https://arxiv.org/abs/2406.09414)

### [`facebook/dinov2-base`](https://huggingface.co/facebook/dinov2-base)
> image-feature-extraction · apache-2.0

- **Used by**: `subject_consistency`
- **Parameters**: 87M · **Downloads**: 1.2M
- **Disk**: ~330 MB
- **Source**: [arXiv](https://arxiv.org/abs/2304.07193)

### [`facebook/vjepa-giant`](https://huggingface.co/facebook/vjepa-giant)

- **Used by**: `jedi`

### [`facebookresearch/dinov2`](https://huggingface.co/facebookresearch/dinov2)

- **Used by**: `video_memorability`

### [`laion/clap-htsat-fused`](https://huggingface.co/laion/clap-htsat-fused)
> audio-classification · apache-2.0

- **Used by**: `audio_text_alignment`
- **Parameters**: 154M · **Downloads**: 25.8M
- **VRAM**: ~600 MB · **Disk**: ~600 MB
- **Source**: [arXiv](https://arxiv.org/abs/2211.06687)

### [`llava-hf/llava-1.5-7b-hf`](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
> image-text-to-text · llama2

- **Used by**: `commonsense`, `creativity`, `vlm_judge`
- **Parameters**: 7.1B · **Downloads**: 4.7M
- **VRAM**: ~14 GB · **Disk**: ~14 GB

### [`llava-hf/llava-v1.6-mistral-7b-hf`](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf)
> image-text-to-text · apache-2.0

- **Used by**: `llm_descriptive_qa`
- **Parameters**: 7.6B · **Downloads**: 683K
- **VRAM**: ~14 GB · **Disk**: ~14 GB
- **Source**: [arXiv](https://arxiv.org/abs/2310.03744)

### [`microsoft/xclip-base-patch32`](https://huggingface.co/microsoft/xclip-base-patch32)
> video-classification · mit

- **Used by**: `embedding`, `video_text_matching`
- **Parameters**: 197M · **Downloads**: 140K
- **VRAM**: ~600 MB · **Disk**: ~600 MB
- **Source**: [arXiv](https://arxiv.org/abs/2208.02816)

### [`models/video_motion_predictor`](https://huggingface.co/models/video_motion_predictor)

- **Used by**: `kandinsky_motion`

### [`nvidia/quality-classifier-deberta`](https://huggingface.co/nvidia/quality-classifier-deberta)
> apache-2.0

- **Used by**: `nemo_curator`
- **Downloads**: 3K
- **Source**: [arXiv](https://arxiv.org/abs/2111.09543)

### [`nvidia/segformer-b0-finetuned-ade-512-512`](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)
> image-segmentation · other

- **Used by**: `semantic_segmentation_consistency`
- **Parameters**: 4M · **Downloads**: 503K
- **Disk**: ~14 MB
- **Source**: [arXiv](https://arxiv.org/abs/2105.15203)

### [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32)
> zero-shot-image-classification

- **Used by**: `action_recognition`, `aigv_assessor`, `background_consistency`, `chronomagic`, `clip_temporal`, `clipvqa`, `creativity`, `dataset_analytics`, `deepfake_detection`, `generative_distribution`, `harmful_content`, `maxvqa`, `scene_tagging`, `sd_reference`, `semantic_alignment`, `t2v_compbench`, `t2v_score`, `tifa`, `umap_projection`, `umtscore`, `video_memorability`, `video_text_matching`, `video_type_classifier`
- **Downloads**: 20.0M
- **VRAM**: ~600 MB · **Disk**: ~600 MB
- **Source**: [arXiv](https://arxiv.org/abs/2103.00020)

### [`openai/clip-vit-large-patch14`](https://huggingface.co/openai/clip-vit-large-patch14)
> zero-shot-image-classification

- **Used by**: `aesthetic_scoring`
- **Parameters**: 428M · **Downloads**: 20.2M
- **VRAM**: ~1.5 GB · **Disk**: ~1.7 GB
- **Source**: [arXiv](https://arxiv.org/abs/2103.00020)

### [`q-future/one-align`](https://huggingface.co/q-future/one-align)
> zero-shot-image-classification · mit

- **Used by**: `q_align`
- **Downloads**: 310K
- **Source**: [arXiv](https://arxiv.org/abs/2312.17090)

### [`qyp2000/KVQ`](https://huggingface.co/qyp2000/KVQ)

- **Used by**: `kvq`

### [`stabilityai/stable-diffusion-xl-base-1.0`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
> text-to-image · openrail++

- **Used by**: `sd_reference`
- **Downloads**: 2.2M
- **Source**: [arXiv](https://arxiv.org/abs/2307.01952)

### [`sunwei925/RQ-VQA`](https://huggingface.co/sunwei925/RQ-VQA)

- **Used by**: `rqvqa`

### [`wangjiarui153/AIGV-Assessor`](https://huggingface.co/wangjiarui153/AIGV-Assessor)

- **Used by**: `aigv_assessor`

### [`xinyu1205/recognize-anything-plus-model`](https://huggingface.co/xinyu1205/recognize-anything-plus-model)
> zero-shot-image-classification · apache-2.0

- **Used by**: `ram_tagging`
- **Source**: [arXiv](https://arxiv.org/abs/2306.03514)

## Weight File Repos

### [`AkaneTendo25/ayase-models`](https://huggingface.co/AkaneTendo25/ayase-models)
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

## pyiqa Metrics

All auto-download weights on first `pyiqa.create_metric()` call. pyiqa itself is MIT-licensed; underlying model licenses vary.

### `pyiqa/afine`

- **Used by**: `afine`

### `pyiqa/afine_nr`
> research

- **Used by**: `afine`

### `pyiqa/ahiq`
> Attention-based hybrid FR-IQA · research

- **Used by**: `ahiq`
- **VRAM**: ~300 MB · **Disk**: ~150 MB

### `pyiqa/arniqa`
> research

- **Used by**: `arniqa`

### `pyiqa/brisque`
> No-reference image quality (naturalness) · BSD-2-Clause (OpenCV)

- **Used by**: `brisque`, `naturalness`
- **VRAM**: ~50 MB · **Disk**: ~1 MB
- **Commercial**: Yes

### `pyiqa/bvqi`

- **Used by**: `bvqi`

### `pyiqa/ckdn`
> research

- **Used by**: `ckdn`

### `pyiqa/clip_iqa`

- **Used by**: `clip_iqa`

### `pyiqa/clipiqa+`
> CLIP-based image quality assessment · MIT (pyiqa)

- **Used by**: `clip_iqa`, `promptiqa`, `rqvqa`
- **VRAM**: ~600 MB · **Disk**: ~600 MB
- **Commercial**: Yes

### `pyiqa/cnniqa`
> research

- **Used by**: `cnniqa`

### `pyiqa/compare2score`
> research

- **Used by**: `compare2score`

### `pyiqa/contrique`
> research

- **Used by**: `contrique`

### `pyiqa/conviqt`

- **Used by**: `conviqt`

### `pyiqa/cover`

- **Used by**: `cover`

### `pyiqa/creativity`

- **Used by**: `creativity`

### `pyiqa/cw_ssim`
> MIT (pyiqa)

- **Used by**: `cw_ssim`
- **Commercial**: Yes

### `pyiqa/dbcnn`
> Deep bilinear CNN for blind IQA · research

- **Used by**: `dbcnn`
- **VRAM**: ~200 MB · **Disk**: ~100 MB

### `pyiqa/deepdc`

- **Used by**: `deepdc`

### `pyiqa/deepwsd`
> research

- **Used by**: `deepwsd`

### `pyiqa/dmm`
> research

- **Used by**: `dmm`

### `pyiqa/dover`
> MIT (pyiqa)

- **Used by**: `cover`, `dover`
- **Commercial**: Yes

### `pyiqa/face_iqa`

- **Used by**: `face_iqa`

### `pyiqa/finevq`

- **Used by**: `finevq`

### `pyiqa/hyperiqa`
> Adaptive hypernetwork NR image quality · research

- **Used by**: `hyperiqa`, `qcn`
- **VRAM**: ~200 MB · **Disk**: ~100 MB

### `pyiqa/ilniqe`
> BSD-2-Clause

- **Used by**: `ilniqe`
- **Commercial**: Yes

### `pyiqa/kvq`

- **Used by**: `kvq`

### `pyiqa/laion_aes`
> MIT

- **Used by**: `creativity`, `laion_aesthetic`
- **Commercial**: Yes

### `pyiqa/laion_aesthetic`

- **Used by**: `laion_aesthetic`

### `pyiqa/liqe`

- **Used by**: `liqe`

### `pyiqa/maclip`
> research

- **Used by**: `maclip`

### `pyiqa/mad`
> research

- **Used by**: `mad`

### `pyiqa/maniqa`
> Multi-dimension attention NR-IQA · Apache-2.0

- **Used by**: `maniqa`
- **VRAM**: ~300 MB · **Disk**: ~150 MB
- **Commercial**: Yes

### `pyiqa/mdtvsfa`

- **Used by**: `mdtvsfa`

### `pyiqa/msswd`

- **Used by**: `msswd`

### `pyiqa/musiq`
> Multi-scale image quality transformer · Apache-2.0 (Google)

- **Used by**: `musiq`
- **VRAM**: ~300 MB · **Disk**: ~150 MB
- **Commercial**: Yes

### `pyiqa/naturalness`

- **Used by**: `naturalness`

### `pyiqa/nima`
> Neural image assessment (aesthetic + technical) · Apache-2.0 (Google)

- **Used by**: `nima`
- **VRAM**: ~200 MB · **Disk**: ~100 MB
- **Commercial**: Yes

### `pyiqa/niqe`
> No-reference image quality (naturalness statistics) · BSD-2-Clause (OpenCV)

- **Used by**: `niqe`
- **VRAM**: ~50 MB · **Disk**: ~1 MB
- **Commercial**: Yes

### `pyiqa/nlpd`
> research

- **Used by**: `nlpd`

### `pyiqa/nrqm`
> research

- **Used by**: `nrqm`

### `pyiqa/paq2piq`
> research

- **Used by**: `paq2piq`

### `pyiqa/pi`
> research

- **Used by**: `pi`

### `pyiqa/pieapp`
> research

- **Used by**: `pieapp`

### `pyiqa/piqe`
> BSD-2-Clause

- **Used by**: `piqe`
- **Commercial**: Yes

### `pyiqa/promptiqa`
> research

- **Used by**: `promptiqa`

### `pyiqa/qcn`
> research

- **Used by**: `qcn`

### `pyiqa/qualiclip`
> research

- **Used by**: `qualiclip`

### `pyiqa/rqvqa`

- **Used by**: `rqvqa`

### `pyiqa/sfid`

- **Used by**: `sfid`

### `pyiqa/ssimc`
> MIT (pyiqa)

- **Used by**: `ssimc`
- **Commercial**: Yes

### `pyiqa/topiq`

- **Used by**: `topiq`

### `pyiqa/topiq_fr`
> Transformer-based FR image quality · MIT (pyiqa)

- **Used by**: `topiq_fr`
- **VRAM**: ~300 MB · **Disk**: ~150 MB
- **Commercial**: Yes

### `pyiqa/topiq_nr`
> Transformer-based NR image quality · MIT (pyiqa)

- **Used by**: `finevq`, `kvq`, `promptiqa`, `topiq`
- **VRAM**: ~300 MB · **Disk**: ~150 MB
- **Commercial**: Yes

### `pyiqa/topiq_nr-face`
> MIT (pyiqa)

- **Used by**: `face_iqa`
- **Commercial**: Yes

### `pyiqa/tres`
> research

- **Used by**: `tres`

### `pyiqa/unique`
> research

- **Used by**: `unique`

### `pyiqa/wadiqam`

- **Used by**: `wadiqam`

### `pyiqa/wadiqam_fr`
> research

- **Used by**: `wadiqam_fr`

### `pyiqa/wadiqam_nr`
> research

- **Used by**: `wadiqam`

## torchvision Models

Bundled with `pip install torchvision`. Weights download on first use.

### `torchvision/inception_v3`
> torchvision · BSD-3-Clause

- **Used by**: `inception_score`
- **VRAM**: ~200 MB · **Disk**: ~100 MB

### `torchvision/r3d_18`
> torchvision · BSD-3-Clause

- **Used by**: `c3dvqa`, `fvd`
- **VRAM**: ~200 MB · **Disk**: ~130 MB

### `torchvision/raft_large`
> torchvision · BSD-3-Clause

- **Used by**: `advanced_flow`, `raft_motion`
- **VRAM**: ~200 MB · **Disk**: ~20 MB

### `torchvision/raft_small`
> torchvision · BSD-3-Clause

- **Used by**: `advanced_flow`, `flolpips`, `motion_amplitude`, `temporal_flickering`
- **VRAM**: ~100 MB · **Disk**: ~20 MB

### `torchvision/resnet18`
> torchvision · BSD-3-Clause

- **Used by**: `tlvqm`
- **VRAM**: ~100 MB · **Disk**: ~45 MB

### `torchvision/resnet50`
> torchvision

- **Used by**: `watermark_classifier`

### `torchvision/video`
> torchvision · BSD-3-Clause

- **Used by**: `c3dvqa`

## CLIP / OpenCLIP

### `CLIP ViT-B/32`
> MIT (OpenAI)

- **Used by**: `vqa_score`
- **VRAM**: ~600 MB · **Disk**: ~340 MB

## torch.hub

### `facebookresearch/co-tracker`
> torch.hub · Apache-2.0

- **Used by**: `dynamics_controllability`, `physics`, `trajan`

### `facebookresearch/dinov2`
> torch.hub · Apache-2.0

- **Used by**: `spectral_complexity`, `video_memorability`

### `intel-isl/MiDaS`
> torch.hub · MIT

- **Used by**: `depth_consistency`, `depth_map_quality`
- **VRAM**: ~400 MB · **Disk**: ~400 MB

### `tarepan/SpeechMOS:v1.2.0`
> torch.hub · MIT

- **Used by**: `audio_utmos`
- **VRAM**: ~200 MB · **Disk**: ~100 MB

## FFmpeg

Require FFmpeg compiled with libvmaf. No separate download needed.

### `ffmpeg/cambi`
> built-in · BSD-2-Clause (Netflix)

- **Used by**: `cambi`

### `ffmpeg/libvmaf`
> built-in · BSD-2-Clause (Netflix)

- **Used by**: `cambi`, `vmaf`, `vmaf_4k`, `vmaf_neg`, `vmaf_phone`

### `ffmpeg/vmaf_4k_v0.6.1`
> built-in · BSD-2-Clause (Netflix)

- **Used by**: `vmaf_4k`

### `ffmpeg/vmaf_phone_model`
> built-in · BSD-2-Clause (Netflix)

- **Used by**: `vmaf_phone`

### `ffmpeg/vmaf_v0.6.1`
> built-in · BSD-2-Clause (Netflix)

- **Used by**: `vmaf_neg`, `vmaf_phone`

### `ffmpeg/vmaf_v0.6.1neg`
> built-in · BSD-2-Clause (Netflix)

- **Used by**: `vmaf_neg`

### `ffmpeg/xpsnr`
> built-in · BSD (FFmpeg)

- **Used by**: `xpsnr`

## pip Packages

### `aesthetic-predictor-v2-5`
> Aesthetic Predictor V2.5 (SigLIP)

- **Used by**: `aesthetic`
- **Install**: `pip install aesthetic-predictor-v2-5`

### `dreamsim`
> DreamSim CLIP+DINO similarity

- **Used by**: `dreamsim`
- **Install**: `pip install dreamsim`

### `erqa`
> ERQA edge restoration quality

- **Used by**: `erqa`
- **Install**: `pip install erqa`

### `piq`
> piq (PyTorch Image Quality)

- **Used by**: `dists`, `perceptual_fr`, `vif`
- **Install**: `pip install piq`

### `ptlflow`
> ptlflow optical flow models

- **Used by**: `ptlflow_motion`
- **Install**: `pip install ptlflow`

### `stlpips-pytorch`
> ST-LPIPS spatiotemporal perceptual

- **Used by**: `st_lpips`
- **Install**: `pip install stlpips-pytorch`

### `torchmetrics[audio]`
> TorchMetrics (DNSMOS, etc.)

- **Used by**: `dnsmos`
- **Install**: `pip install torchmetrics[audio]`

### `ultralytics`
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
