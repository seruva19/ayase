# Ayase Models Reference

Auto-generated catalog of all ML models, weights, and external assets used by Ayase pipeline modules. Run `ayase modules models` to regenerate.

## Summary

| Stat | Value |
|------|-------|
| Total model references | **88** |
| HuggingFace models | 31 |
| pyiqa metrics | 40 |
| torchvision models | 5 |
| CLIP variants | 1 |
| torch.hub repos | 4 |
| FFmpeg models | 7 |
| Local weight files | 0 |

### By Source

```
  pyiqa        ██████████████████████████████ 40
  huggingface  ███████████████████████ 31
  ffmpeg       █████ 7
  torchvision  ███ 5
  torch_hub    ███ 4
  clip          1
```

**Estimated total download size (all models):** ~49 GB

*Note: Most modules auto-download only the models they need on first use. You rarely need all models at once.*

### License Overview

```
  Commercial OK           █████████████████████████████ 43
  Non-commercial           1
  Research / unspecified  ██████████████████████████████ 44
```

> **For commercial use:** Stick to modules whose models are marked "Commercial: Yes" below. Most pyiqa metrics marked "research" are re-implementations under pyiqa's MIT license, but the original training data or architecture may carry restrictions — verify before commercial deployment.


## HuggingFace Models

| Model | License | Commercial | Disk | VRAM | Used By |
|-------|---------|------------|------|------|---------|
| [Falconsai/nsfw_image_detection](https://huggingface.co/Falconsai/nsfw_image_detection) | apache-2.0 | Yes | ? | ? | `nsfw` |
| [KlingTeam/VideoAlign-Reward](https://huggingface.co/KlingTeam/VideoAlign-Reward) | ? | ? | ? | ? | `video_reward` |
| [MCG-NJU/videomae-large-finetuned-kinetics](https://huggingface.co/MCG-NJU/videomae-large-finetuned-kinetics) | cc-by-nc-4.0 | No | ~1.3 GB | ~1.5 GB | `action_recognition` |
| [Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base) | bsd-3-clause | Yes | ~990 MB | ~1 GB | `captioning` |
| [TIGER-Lab/T2VScore](https://huggingface.co/TIGER-Lab/T2VScore) | ? | ? | ? | ? | `t2v_score` |
| [TIGER-Lab/VideoScore](https://huggingface.co/TIGER-Lab/VideoScore) | apache-2.0 | Yes | ~14 GB | ~14 GB | `videoscore` |
| [ai-forever/kandinsky-video-tools](https://huggingface.co/ai-forever/kandinsky-video-tools) | ? | ? | ? | ? | `kandinsky_motion` |
| [dandelin/vilt-b32-finetuned-vqa](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa) | apache-2.0 | Yes | ~450 MB | ~500 MB | `commonsense` |
| [depth-anything/Depth-Anything-V2-Small-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) | apache-2.0 | Yes | ~100 MB | ~200 MB | `depth_anything` |
| [facebook/dinov2-base](https://huggingface.co/facebook/dinov2-base) | apache-2.0 | Yes | ? | ? | `subject_consistency` |
| [laion/clap-htsat-fused](https://huggingface.co/laion/clap-htsat-fused) | apache-2.0 | Yes | ~600 MB | ~600 MB | `audio_text_alignment` |
| [llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf) | llama2 | ? | ~14 GB | ~14 GB | `commonsense`, `creativity`, `vlm_judge` |
| [llava-hf/llava-v1.6-mistral-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf) | apache-2.0 | Yes | ~14 GB | ~14 GB | `llm_descriptive_qa` |
| [microsoft/xclip-base-patch32](https://huggingface.co/microsoft/xclip-base-patch32) | mit | Yes | ~600 MB | ~600 MB | `embedding` |
| [nvidia/quality-classifier-deberta](https://huggingface.co/nvidia/quality-classifier-deberta) | apache-2.0 | Yes | ? | ? | `nemo_curator` |
| [nvidia/segformer-b0-finetuned-ade-512-512](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512) | other | ? | ? | ? | `semantic_segmentation_consistency` |
| [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) | ? | ? | ~600 MB | ~600 MB | `aigv_assessor`, `background_consistency`, `chronomagic`, `clip_temporal`, `creativity` +9 |
| [q-future/one-align](https://huggingface.co/q-future/one-align) | mit | Yes | ? | ? | `q_align` |
| [qyp2000/KVQ](https://huggingface.co/qyp2000/KVQ) | ? | ? | ? | ? | `kvq` |
| [sunwei925/RQ-VQA](https://huggingface.co/sunwei925/RQ-VQA) | ? | ? | ? | ? | `rqvqa` |
| [xinyu1205/recognize-anything-plus-model](https://huggingface.co/xinyu1205/recognize-anything-plus-model) | apache-2.0 | Yes | ? | ? | `ram_tagging` |
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
| `pyiqa/afine_nr` | research | ? | IQA | ~200 MB | `afine` |
| `pyiqa/ahiq` | research | ? | Attention-based hybrid FR-IQA | ~300 MB | `ahiq` |
| `pyiqa/arniqa` | research | ? | IQA | ~200 MB | `arniqa` |
| `pyiqa/brisque` | BSD-2-Clause (OpenCV) | Yes | No-reference image quality (naturalness) | ~50 MB | `brisque`, `naturalness` |
| `pyiqa/ckdn` | research | ? | IQA | ~200 MB | `ckdn` |
| `pyiqa/clipiqa+` | MIT (pyiqa) | Yes | CLIP-based image quality assessment | ~600 MB | `clip_iqa`, `promptiqa`, `rqvqa` |
| `pyiqa/cnniqa` | research | ? | IQA | ~200 MB | `cnniqa` |
| `pyiqa/compare2score` | research | ? | IQA | ~200 MB | `compare2score` |
| `pyiqa/contrique` | research | ? | IQA | ~200 MB | `contrique` |
| `pyiqa/cw_ssim` | MIT (pyiqa) | Yes | IQA | ~200 MB | `cw_ssim` |
| `pyiqa/dbcnn` | research | ? | Deep bilinear CNN for blind IQA | ~200 MB | `dbcnn` |
| `pyiqa/deepwsd` | research | ? | IQA | ~200 MB | `deepwsd` |
| `pyiqa/dmm` | research | ? | IQA | ~200 MB | `dmm` |
| `pyiqa/dover` | MIT (pyiqa) | Yes | IQA | ~200 MB | `cover`, `dover` |
| `pyiqa/hyperiqa` | research | ? | Adaptive hypernetwork NR image quality | ~200 MB | `hyperiqa`, `qcn` |
| `pyiqa/ilniqe` | BSD-2-Clause | Yes | IQA | ~200 MB | `ilniqe` |
| `pyiqa/laion_aes` | MIT | Yes | IQA | ~200 MB | `creativity`, `laion_aesthetic` |
| `pyiqa/liqe` | research | ? | IQA | ~200 MB | `liqe` |
| `pyiqa/maclip` | research | ? | IQA | ~200 MB | `maclip` |
| `pyiqa/mad` | research | ? | IQA | ~200 MB | `mad` |
| `pyiqa/maniqa` | Apache-2.0 | Yes | Multi-dimension attention NR-IQA | ~300 MB | `maniqa` |
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
| `pyiqa/ssimc` | MIT (pyiqa) | Yes | IQA | ~200 MB | `ssimc` |
| `pyiqa/topiq_fr` | MIT (pyiqa) | Yes | Transformer-based FR image quality | ~300 MB | `topiq_fr` |
| `pyiqa/topiq_nr` | MIT (pyiqa) | Yes | Transformer-based NR image quality | ~300 MB | `finevq`, `kvq`, `promptiqa` |
| `pyiqa/topiq_nr-face` | MIT (pyiqa) | Yes | IQA | ~200 MB | `face_iqa` |
| `pyiqa/tres` | research | ? | IQA | ~200 MB | `tres` |
| `pyiqa/unique` | research | ? | IQA | ~200 MB | `unique` |
| `pyiqa/wadiqam_fr` | research | ? | IQA | ~200 MB | `wadiqam_fr` |
| `pyiqa/wadiqam_nr` | research | ? | IQA | ~200 MB | `wadiqam` |

## torchvision Models

Bundled with `pip install torchvision`. Weights download on first use.

| Model | Disk | VRAM | Used By |
|-------|------|------|---------|
| `torchvision/r3d_18` | ~130 MB | ~200 MB | `c3dvqa`, `fvd` |
| `torchvision/raft_large` | ~20 MB | ~200 MB | `advanced_flow`, `raft_motion` |
| `torchvision/raft_small` | ~20 MB | ~100 MB | `advanced_flow`, `flolpips`, `motion_amplitude`, `temporal_flickering` |
| `torchvision/resnet18` | ~45 MB | ~100 MB | `tlvqm` |
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
