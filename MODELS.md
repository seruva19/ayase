# Ayase Models Reference

> **Version 0.1.17** · Generated 2026-03-21 16:52 · **124 models** across **7 sources**
>
> `ayase modules models -o MODELS.md` to regenerate

## Summary

![Summary](docs/models_summary.png)

### Models by Source

![Models by Source](docs/models_sources.png)

**Estimated total download size (all models):** ~82 GB

*Note: Most modules auto-download only the models they need on first use. You rarely need all models at once.*

### License Overview

![License Distribution](docs/models_licenses.png)

> [!WARNING]
> **Commercial use:** Stick to modules whose models are marked "Commercial OK" above. Most pyiqa metrics marked "research" are re-implementations under pyiqa's MIT license, but the original training data or architecture may carry restrictions — verify before commercial deployment.


## HuggingFace Models

| Model | License | Params | Downloads | Task | Used By |
|-------|---------|--------|-----------|------|---------|
| [Falconsai/nsfw_image_detection](https://huggingface.co/Falconsai/nsfw_image_detection) [[paper]](https://arxiv.org/abs/2010.11929) | apache-2.0 | ? | 41.7M | image-classification | `nsfw` |
| [IntMeGroup/FineVQ_score](https://huggingface.co/IntMeGroup/FineVQ_score) | apache-2.0 | 8.2B | 671 | ? | `finevq` |
| [KlingTeam/VideoAlign-Reward](https://huggingface.co/KlingTeam/VideoAlign-Reward) | ? | ? | ? | ? | `video_reward` |
| [MCG-NJU/videomae-large-finetuned-kinetics](https://huggingface.co/MCG-NJU/videomae-large-finetuned-kinetics) [[paper]](https://arxiv.org/abs/2203.12602) | cc-by-nc-4.0 | 304M | 2K | video-classification | `action_recognition` |
| [Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base) [[paper]](https://arxiv.org/abs/2201.12086) | bsd-3-clause | ? | 3.5M | image-to-text | `captioning` |
| [TIGER-Lab/T2VScore](https://huggingface.co/TIGER-Lab/T2VScore) | ? | ? | ? | ? | `t2v_score` |
| [TIGER-Lab/VideoScore](https://huggingface.co/TIGER-Lab/VideoScore) [[paper]](https://arxiv.org/abs/2406.15252) | apache-2.0 | 8.3B | 2K | visual-question-answering | `videoscore` |
| [ai-forever/kandinsky-video-tools](https://huggingface.co/ai-forever/kandinsky-video-tools) | ? | ? | ? | ? | `kandinsky_motion` |
| [dandelin/vilt-b32-finetuned-vqa](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa) [[paper]](https://arxiv.org/abs/2102.03334) | apache-2.0 | ? | 73K | visual-question-answering | `commonsense`, `tifa` |
| [depth-anything/Depth-Anything-V2-Small-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) [[paper]](https://arxiv.org/abs/2406.09414) | apache-2.0 | 25M | 1.4M | depth-estimation | `depth_anything`, `t2v_compbench` |
| [facebook/dinov2-base](https://huggingface.co/facebook/dinov2-base) [[paper]](https://arxiv.org/abs/2304.07193) | apache-2.0 | 87M | 1.2M | image-feature-extraction | `subject_consistency` |
| [facebook/vjepa-giant](https://huggingface.co/facebook/vjepa-giant) | ? | ? | ? | ? | `jedi` |
| [facebookresearch/dinov2](https://huggingface.co/facebookresearch/dinov2) | ? | ? | ? | ? | `video_memorability` |
| [laion/clap-htsat-fused](https://huggingface.co/laion/clap-htsat-fused) [[paper]](https://arxiv.org/abs/2211.06687) | apache-2.0 | 154M | 25.8M | audio-classification | `audio_text_alignment` |
| [llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf) | llama2 | 7.1B | 4.7M | image-text-to-text | `commonsense`, `creativity`, `vlm_judge` |
| [llava-hf/llava-v1.6-mistral-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf) [[paper]](https://arxiv.org/abs/2310.03744) | apache-2.0 | 7.6B | 683K | image-text-to-text | `llm_descriptive_qa` |
| [microsoft/xclip-base-patch32](https://huggingface.co/microsoft/xclip-base-patch32) [[paper]](https://arxiv.org/abs/2208.02816) | mit | 197M | 140K | video-classification | `embedding`, `video_text_matching` |
| [models/video_motion_predictor](https://huggingface.co/models/video_motion_predictor) | ? | ? | ? | ? | `kandinsky_motion` |
| [nvidia/quality-classifier-deberta](https://huggingface.co/nvidia/quality-classifier-deberta) [[paper]](https://arxiv.org/abs/2111.09543) | apache-2.0 | ? | 3K | ? | `nemo_curator` |
| [nvidia/segformer-b0-finetuned-ade-512-512](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512) [[paper]](https://arxiv.org/abs/2105.15203) | other | 4M | 503K | image-segmentation | `semantic_segmentation_consistency` |
| [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) [[paper]](https://arxiv.org/abs/2103.00020) | ? | ? | 20.0M | zero-shot-image-classification | `action_recognition`, `aigv_assessor`, `background_consistency`, `chronomagic`, `clip_temporal` +18 |
| [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) [[paper]](https://arxiv.org/abs/2103.00020) | ? | 428M | 20.2M | zero-shot-image-classification | `aesthetic_scoring` |
| [q-future/one-align](https://huggingface.co/q-future/one-align) [[paper]](https://arxiv.org/abs/2312.17090) | mit | ? | 310K | zero-shot-image-classification | `q_align` |
| [qyp2000/KVQ](https://huggingface.co/qyp2000/KVQ) | ? | ? | ? | ? | `kvq` |
| [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) [[paper]](https://arxiv.org/abs/2307.01952) | openrail++ | ? | 2.2M | text-to-image | `sd_reference` |
| [sunwei925/RQ-VQA](https://huggingface.co/sunwei925/RQ-VQA) | ? | ? | ? | ? | `rqvqa` |
| [wangjiarui153/AIGV-Assessor](https://huggingface.co/wangjiarui153/AIGV-Assessor) | ? | ? | ? | ? | `aigv_assessor` |
| [xinyu1205/recognize-anything-plus-model](https://huggingface.co/xinyu1205/recognize-anything-plus-model) [[paper]](https://arxiv.org/abs/2306.03514) | apache-2.0 | ? | ? | zero-shot-image-classification | `ram_tagging` |
| [DOVER.pth](https://huggingface.co/AkaneTendo25/ayase-models/resolve/main/dover/DOVER.pth) | ? | ? | ? | ? | `dover` |
| [convnext_tiny_1k_224_ema.pth](https://huggingface.co/AkaneTendo25/ayase-models/resolve/main/dover/convnext_tiny_1k_224_ema.pth) | ? | ? | ? | ? | `dover` |
| [onnx_dover.onnx](https://huggingface.co/AkaneTendo25/ayase-models/resolve/main/dover/onnx_dover.onnx) | ? | ? | ? | ? | `dover` |
| [FAST_VQA_3D_1_1.pth](https://huggingface.co/AkaneTendo25/ayase-models/resolve/main/fast_vqa/FAST_VQA_3D_1_1.pth) | ? | ? | ? | ? | `fast_vqa` |
| [FAST_VQA_B_1_4.pth](https://huggingface.co/AkaneTendo25/ayase-models/resolve/main/fast_vqa/FAST_VQA_B_1_4.pth) | ? | ? | ? | ? | `fast_vqa` |
| [FAST_VQA_M_1_4.pth](https://huggingface.co/AkaneTendo25/ayase-models/resolve/main/fast_vqa/FAST_VQA_M_1_4.pth) | ? | ? | ? | ? | `fast_vqa` |
| [ViT-B-32.pt](https://huggingface.co/AkaneTendo25/ayase-models/resolve/main/i2v_similarity/ViT-B-32.pt) | ? | ? | ? | ? | `i2v_similarity` |
| [alex.pth](https://huggingface.co/AkaneTendo25/ayase-models/resolve/main/i2v_similarity/alex.pth) | ? | ? | ? | ? | `i2v_similarity` |
| [dinov2_vitb14_pretrain.pth](https://huggingface.co/AkaneTendo25/ayase-models/resolve/main/i2v_similarity/dinov2_vitb14_pretrain.pth) | ? | ? | ? | ? | `i2v_similarity` |
| [flownet.pkl](https://huggingface.co/AkaneTendo25/ayase-models/resolve/main/motion_smoothness/flownet.pkl) | ? | ? | ? | ? | `motion_smoothness` |

## pyiqa Metrics

All auto-download weights on first `pyiqa.create_metric()` call. pyiqa itself is MIT-licensed; underlying model licenses vary.

| Metric | License | Commercial | Task | VRAM | Used By |
|--------|---------|------------|------|------|---------|
| `pyiqa/afine` | research | ? | IQA | ~200 MB | `afine` |
| `pyiqa/afine_nr` | research | ? | IQA | ~200 MB | `afine` |
| `pyiqa/ahiq` | research | ? | Attention-based hybrid FR-IQA | ~300 MB | `ahiq` |
| `pyiqa/arniqa` | research | ? | IQA | ~200 MB | `arniqa` |
| `pyiqa/brisque` | BSD-2-Clause (OpenCV) | Yes | No-reference image quality (naturalness) | ~50 MB | `brisque`, `naturalness` |
| `pyiqa/bvqi` | research | ? | IQA | ~200 MB | `bvqi` |
| `pyiqa/ckdn` | research | ? | IQA | ~200 MB | `ckdn` |
| `pyiqa/clip_iqa` | research | ? | IQA | ~200 MB | `clip_iqa` |
| `pyiqa/clipiqa+` | MIT (pyiqa) | Yes | CLIP-based image quality assessment | ~600 MB | `clip_iqa`, `promptiqa`, `rqvqa` |
| `pyiqa/cnniqa` | research | ? | IQA | ~200 MB | `cnniqa` |
| `pyiqa/compare2score` | research | ? | IQA | ~200 MB | `compare2score` |
| `pyiqa/contrique` | research | ? | IQA | ~200 MB | `contrique` |
| `pyiqa/conviqt` | research | ? | IQA | ~200 MB | `conviqt` |
| `pyiqa/cover` | research | ? | IQA | ~200 MB | `cover` |
| `pyiqa/creativity` | research | ? | IQA | ~200 MB | `creativity` |
| `pyiqa/cw_ssim` | MIT (pyiqa) | Yes | IQA | ~200 MB | `cw_ssim` |
| `pyiqa/dbcnn` | research | ? | Deep bilinear CNN for blind IQA | ~200 MB | `dbcnn` |
| `pyiqa/deepdc` | research | ? | IQA | ~200 MB | `deepdc` |
| `pyiqa/deepwsd` | research | ? | IQA | ~200 MB | `deepwsd` |
| `pyiqa/dmm` | research | ? | IQA | ~200 MB | `dmm` |
| `pyiqa/dover` | MIT (pyiqa) | Yes | IQA | ~200 MB | `cover`, `dover` |
| `pyiqa/face_iqa` | research | ? | IQA | ~200 MB | `face_iqa` |
| `pyiqa/finevq` | research | ? | IQA | ~200 MB | `finevq` |
| `pyiqa/hyperiqa` | research | ? | Adaptive hypernetwork NR image quality | ~200 MB | `hyperiqa`, `qcn` |
| `pyiqa/ilniqe` | BSD-2-Clause | Yes | IQA | ~200 MB | `ilniqe` |
| `pyiqa/kvq` | research | ? | IQA | ~200 MB | `kvq` |
| `pyiqa/laion_aes` | MIT | Yes | IQA | ~200 MB | `creativity`, `laion_aesthetic` |
| `pyiqa/laion_aesthetic` | research | ? | IQA | ~200 MB | `laion_aesthetic` |
| `pyiqa/liqe` | research | ? | IQA | ~200 MB | `liqe` |
| `pyiqa/maclip` | research | ? | IQA | ~200 MB | `maclip` |
| `pyiqa/mad` | research | ? | IQA | ~200 MB | `mad` |
| `pyiqa/maniqa` | Apache-2.0 | Yes | Multi-dimension attention NR-IQA | ~300 MB | `maniqa` |
| `pyiqa/mdtvsfa` | research | ? | IQA | ~200 MB | `mdtvsfa` |
| `pyiqa/msswd` | research | ? | IQA | ~200 MB | `msswd` |
| `pyiqa/musiq` | Apache-2.0 (Google) | Yes | Multi-scale image quality transformer | ~300 MB | `musiq` |
| `pyiqa/naturalness` | research | ? | IQA | ~200 MB | `naturalness` |
| `pyiqa/nima` | Apache-2.0 (Google) | Yes | Neural image assessment (aesthetic + technical) | ~200 MB | `nima` |
| `pyiqa/niqe` | BSD-2-Clause (OpenCV) | Yes | No-reference image quality (naturalness statistics) | ~50 MB | `niqe` |
| `pyiqa/nlpd` | research | ? | IQA | ~200 MB | `nlpd` |
| `pyiqa/nrqm` | research | ? | IQA | ~200 MB | `nrqm` |
| `pyiqa/paq2piq` | research | ? | IQA | ~200 MB | `paq2piq` |
| `pyiqa/pi` | research | ? | IQA | ~200 MB | `pi` |
| `pyiqa/pieapp` | research | ? | IQA | ~200 MB | `pieapp` |
| `pyiqa/piqe` | BSD-2-Clause | Yes | IQA | ~200 MB | `piqe` |
| `pyiqa/promptiqa` | research | ? | IQA | ~200 MB | `promptiqa` |
| `pyiqa/qcn` | research | ? | IQA | ~200 MB | `qcn` |
| `pyiqa/qualiclip` | research | ? | IQA | ~200 MB | `qualiclip` |
| `pyiqa/rqvqa` | research | ? | IQA | ~200 MB | `rqvqa` |
| `pyiqa/sfid` | research | ? | IQA | ~200 MB | `sfid` |
| `pyiqa/ssimc` | MIT (pyiqa) | Yes | IQA | ~200 MB | `ssimc` |
| `pyiqa/topiq` | research | ? | IQA | ~200 MB | `topiq` |
| `pyiqa/topiq_fr` | MIT (pyiqa) | Yes | Transformer-based FR image quality | ~300 MB | `topiq_fr` |
| `pyiqa/topiq_nr` | MIT (pyiqa) | Yes | Transformer-based NR image quality | ~300 MB | `finevq`, `kvq`, `promptiqa`, `topiq` |
| `pyiqa/topiq_nr-face` | MIT (pyiqa) | Yes | IQA | ~200 MB | `face_iqa` |
| `pyiqa/tres` | research | ? | IQA | ~200 MB | `tres` |
| `pyiqa/unique` | research | ? | IQA | ~200 MB | `unique` |
| `pyiqa/wadiqam` | research | ? | IQA | ~200 MB | `wadiqam` |
| `pyiqa/wadiqam_fr` | research | ? | IQA | ~200 MB | `wadiqam_fr` |
| `pyiqa/wadiqam_nr` | research | ? | IQA | ~200 MB | `wadiqam` |

## torchvision Models

Bundled with `pip install torchvision`. Weights download on first use.

| Model | Disk | VRAM | Used By |
|-------|------|------|---------|
| `torchvision/inception_v3` | ~100 MB | ~200 MB | `inception_score` |
| `torchvision/r3d_18` | ~130 MB | ~200 MB | `c3dvqa`, `fvd` |
| `torchvision/raft_large` | ~20 MB | ~200 MB | `advanced_flow`, `raft_motion` |
| `torchvision/raft_small` | ~20 MB | ~100 MB | `advanced_flow`, `flolpips`, `motion_amplitude`, `temporal_flickering` |
| `torchvision/resnet18` | ~45 MB | ~100 MB | `tlvqm` |
| `torchvision/resnet50` | ? | ? | `watermark_classifier` |
| `torchvision/video` | ? | ? | `c3dvqa` |

## CLIP Models

| Model | Disk | VRAM | Used By |
|-------|------|------|---------|
| `CLIP ViT-B/32` | ~340 MB | ~600 MB | `vqa_score` |

## torch.hub Models

| Repo | Disk | VRAM | Used By |
|------|------|------|---------|
| `facebookresearch/co-tracker` | ? | ? | `dynamics_controllability`, `physics`, `trajan` |
| `facebookresearch/dinov2` | ? | ? | `spectral_complexity`, `video_memorability` |
| `intel-isl/MiDaS` | ~400 MB | ~400 MB | `depth_consistency`, `depth_map_quality` |
| `tarepan/SpeechMOS:v1.2.0` | ~100 MB | ~200 MB | `audio_utmos` |

## FFmpeg Models

Require FFmpeg compiled with libvmaf. No separate download needed.

| Model | Used By |
|-------|---------|
| `ffmpeg/cambi` | `cambi` |
| `ffmpeg/libvmaf` | `cambi`, `vmaf`, `vmaf_4k`, `vmaf_neg`, `vmaf_phone` |
| `ffmpeg/vmaf_4k_v0.6.1` | `vmaf_4k` |
| `ffmpeg/vmaf_phone_model` | `vmaf_phone` |
| `ffmpeg/vmaf_v0.6.1` | `vmaf_neg`, `vmaf_phone` |
| `ffmpeg/vmaf_v0.6.1neg` | `vmaf_neg` |
| `ffmpeg/xpsnr` | `xpsnr` |

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
