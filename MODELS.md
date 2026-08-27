# Ayase Models Reference

> **Version 0.1.72** · Generated 2026-08-27 22:39 · **272 models** across **9 sources**
>
> `ayase modules models -o MODELS.md` to regenerate

## Summary

**272** models · **102** HuggingFace · **55** pyiqa · **9** sources

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

**Estimated total download size (all models):** ~628 GB

*Note: Most modules auto-download only the models they need on first use. You rarely need all models at once.*

> [!WARNING]
> **Commercial use:** Stick to modules whose models are marked "Commercial OK" above. Most pyiqa metrics marked "research" are re-implementations under pyiqa's MIT license, but the original training data or architecture may carry restrictions — verify before commercial deployment.

<a id="categories"></a>

[HuggingFace (73)](#huggingface-models) · [Weight Files (29)](#weight-file-repos) · [pyiqa (55)](#pyiqa-metrics) · [torchvision (14)](#torchvision-models) · [CLIP / OpenCLIP (2)](#clip--openclip) · [torch.hub (5)](#torchhub) · [FFmpeg (7)](#ffmpeg) · [pip Packages (37)](#pip-packages) · [Local Weights (39)](#local-weight-files) · [Quick Install Guide](#quick-install-guide)

---

## HuggingFace Models

### <a href="https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/409c832ac7a30524a48ab642455bf963c2a95d1f/expression_following/face_landmarker.task" target="_blank">`AkaneTendo25/ayase-runtime-assets`</a> [↑](#categories)

- **Used by**: `expression_following`, `expression_similarity`, `eyebrow_dynamics`, `head_motion_dynamics`, `lip_dynamics`, `mj_video`, `mouth_quality`, `silent_lip_stability`, `vbench2`, `vebench`, `worldmodelbench`

### <a href="https://huggingface.co/ByteDance/EvoQuality" target="_blank">`ByteDance/EvoQuality`</a> [↑](#categories)
> image-text-to-text · apache-2.0

- **Used by**: `evoquality`
- **Parameters**: 8.3B · **Downloads**: 111
- **Disk**: ~30.9 GB
- **Source**: <a href="https://arxiv.org/abs/2509.25787" target="_blank">arXiv</a>

### <a href="https://huggingface.co/ByteDance/Q-Insight" target="_blank">`ByteDance/Q-Insight`</a> [↑](#categories)
> apache-2.0

- **Used by**: `vqinsight`
- **Source**: <a href="https://arxiv.org/abs/2503.22679" target="_blank">arXiv</a>

### <a href="https://huggingface.co/Efficient-Large-Model/vila-ewm-qwen2-1.5b" target="_blank">`Efficient-Large-Model/vila-ewm-qwen2-1.5b`</a> [↑](#categories)

- **Used by**: `worldmodelbench`
- **Downloads**: 56

### <a href="https://huggingface.co/Falconsai/nsfw_image_detection" target="_blank">`Falconsai/nsfw_image_detection`</a> [↑](#categories)
> image-classification · apache-2.0

- **Used by**: `nsfw`
- **Parameters**: 86M · **Downloads**: 4.1M
- **Disk**: ~327 MB
- **Source**: <a href="https://arxiv.org/abs/2010.11929" target="_blank">arXiv</a>

### <a href="https://huggingface.co/FunAudioLLM/SenseVoiceSmall" target="_blank">`FunAudioLLM/SenseVoiceSmall`</a> [↑](#categories)
> automatic-speech-recognition · other

- **Used by**: `verse_bench`
- **Downloads**: 31K

### <a href="https://huggingface.co/GD-ML/VMBench" target="_blank">`GD-ML/VMBench`</a> [↑](#categories)
> apache-2.0

- **Used by**: `video_edit_motion_fidelity`, `vmbench_cas`, `vmbench_pas`, `vmbench_tcs`
- **Source**: <a href="https://arxiv.org/abs/2503.10076" target="_blank">arXiv</a>

### <a href="https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3" target="_blank">`HuggingFaceM4/Idefics3-8B-Llama3`</a> [↑](#categories)
> image-text-to-text · apache-2.0

- **Used by**: `dice_edit`
- **Parameters**: 8.5B · **Downloads**: 137K
- **VRAM**: ~20 GB in bfloat16 · **Disk**: ~17 GB
- **Source**: <a href="https://arxiv.org/abs/2306.16527" target="_blank">arXiv</a>

### <a href="https://huggingface.co/IDEA-Research/grounding-dino-tiny" target="_blank">`IDEA-Research/grounding-dino-tiny`</a> [↑](#categories)
> zero-shot-object-detection · apache-2.0

- **Used by**: `opens2v`
- **Parameters**: 172M · **Downloads**: 992K
- **Disk**: ~657 MB
- **Source**: <a href="https://arxiv.org/abs/2303.05499" target="_blank">arXiv</a>

### <a href="https://huggingface.co/IntMeGroup/AIGV-Assessor-static_quality" target="_blank">`IntMeGroup/AIGV-Assessor-static_quality`</a> [↑](#categories)
> apache-2.0

- **Used by**: `aigv_assessor`
- **Parameters**: 8.2B · **Downloads**: 14
- **Disk**: ~30.4 GB

### <a href="https://huggingface.co/IntMeGroup/FineVQ_score" target="_blank">`IntMeGroup/FineVQ_score`</a> [↑](#categories)
> apache-2.0

- **Used by**: `finevq`
- **Parameters**: 8.2B · **Downloads**: 698
- **Disk**: ~30.5 GB

### <a href="https://huggingface.co/JZHWS/slowfast" target="_blank">`JZHWS/slowfast`</a> [↑](#categories)
> apache-2.0

- **Used by**: `vqa2`
- **Disk**: 139 MB

### <a href="https://huggingface.co/KlingTeam/VideoReward" target="_blank">`KlingTeam/VideoReward`</a> [↑](#categories)
> apache-2.0

- **Used by**: `video_reward`
- **Source**: <a href="https://arxiv.org/abs/2501.13918" target="_blank">arXiv</a>

### <a href="https://huggingface.co/KwaiVGI/VideoReward" target="_blank">`KwaiVGI/VideoReward`</a> [↑](#categories)
> apache-2.0

- **Used by**: `video_reward`
- **Source**: <a href="https://arxiv.org/abs/2501.13918" target="_blank">arXiv</a>

### <a href="https://huggingface.co/MCG-NJU/videomae-large-finetuned-kinetics" target="_blank">`MCG-NJU/videomae-large-finetuned-kinetics`</a> [↑](#categories)
> video-classification · cc-by-nc-4.0

- **Used by**: `action_recognition`
- **Parameters**: 304M · **Downloads**: 3K
- **VRAM**: ~1.5 GB · **Disk**: ~1.3 GB
- **Source**: <a href="https://arxiv.org/abs/2203.12602" target="_blank">arXiv</a>

### <a href="https://huggingface.co/MJ-Bench/MJ-VIDEO-2B" target="_blank">`MJ-Bench/MJ-VIDEO-2B`</a> [↑](#categories)

- **Used by**: `mj_video`
- **Parameters**: 2.2B · **Downloads**: 13
- **Disk**: ~8.2 GB

### <a href="https://huggingface.co/MizzenAI/HPSv3" target="_blank">`MizzenAI/HPSv3`</a> [↑](#categories)
> image-text-to-text · apache-2.0

- **Used by**: `hpsv3`
- **Downloads**: 209
- **Source**: <a href="https://arxiv.org/abs/2508.03789" target="_blank">arXiv</a>

### <a href="https://huggingface.co/NU-World-Model-Embodied-AI/phyjudge-9B" target="_blank">`NU-World-Model-Embodied-AI/phyjudge-9B`</a> [↑](#categories)
> video-text-to-text · other

- **Used by**: `phyground_results`
- **Downloads**: 20
- **Source**: <a href="https://arxiv.org/abs/2605.10806" target="_blank">arXiv</a>

### <a href="https://huggingface.co/OpenMuQ/MuQ-large-msd-iter" target="_blank">`OpenMuQ/MuQ-large-msd-iter`</a> [↑](#categories)
> audio-classification · cc-by-nc-4.0

- **Used by**: `muq_eval`, `song_eval`
- **Parameters**: 333M · **Downloads**: 423K
- **Disk**: ~1.2 GB
- **Source**: <a href="https://arxiv.org/abs/2501.01108" target="_blank">arXiv</a>

### <a href="https://huggingface.co/Qwen/Qwen-Image-Bench" target="_blank">`Qwen/Qwen-Image-Bench`</a> [↑](#categories)
> image-text-to-text · apache-2.0

- **Used by**: `qwen_image_bench`
- **Parameters**: 27.4B · **Downloads**: 41K
- **Disk**: ~101.9 GB
- **Source**: <a href="https://arxiv.org/abs/2605.28091" target="_blank">arXiv</a>

### <a href="https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct" target="_blank">`Qwen/Qwen2-VL-7B-Instruct`</a> [↑](#categories)
> image-text-to-text · apache-2.0

- **Used by**: `hpsv3`
- **Parameters**: 8.3B · **Downloads**: 1.3M
- **VRAM**: ~16 GB · **Disk**: ~15 GB
- **Source**: <a href="https://arxiv.org/abs/2409.12191" target="_blank">arXiv</a>

### <a href="https://huggingface.co/Qwen/Qwen2.5-Omni-7B" target="_blank">`Qwen/Qwen2.5-Omni-7B`</a> [↑](#categories)
> any-to-any · other

- **Used by**: `aqascore`
- **Parameters**: 10.7B · **Downloads**: 314K
- **Disk**: ~40.0 GB
- **Source**: <a href="https://arxiv.org/abs/2503.20215" target="_blank">arXiv</a>

### <a href="https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct" target="_blank">`Qwen/Qwen2.5-VL-7B-Instruct`</a> [↑](#categories)
> image-text-to-text · apache-2.0

- **Used by**: `camerabench`
- **Parameters**: 8.3B · **Downloads**: 8.4M
- **Disk**: ~30.9 GB
- **Source**: <a href="https://arxiv.org/abs/2309.00071" target="_blank">arXiv</a>

### <a href="https://huggingface.co/Salesforce/blip-image-captioning-base" target="_blank">`Salesforce/blip-image-captioning-base`</a> [↑](#categories)
> image-to-text · bsd-3-clause

- **Used by**: `captioning`
- **Downloads**: 2.0M
- **VRAM**: ~1 GB · **Disk**: ~990 MB
- **Source**: <a href="https://arxiv.org/abs/2201.12086" target="_blank">arXiv</a>

### <a href="https://huggingface.co/Salesforce/blip-itm-large-coco" target="_blank">`Salesforce/blip-itm-large-coco`</a> [↑](#categories)
> bsd-3-clause

- **Used by**: `blip_score`
- **Downloads**: 4K
- **Source**: <a href="https://arxiv.org/abs/2201.12086" target="_blank">arXiv</a>

### <a href="https://huggingface.co/THUDM/VisionReward-Image" target="_blank">`THUDM/VisionReward-Image`</a> [↑](#categories)
> text-generation · other

- **Used by**: `vision_reward`

### <a href="https://huggingface.co/THUDM/VisionReward-Video" target="_blank">`THUDM/VisionReward-Video`</a> [↑](#categories)
> text-generation · other

- **Used by**: `vision_reward`
- **Parameters**: 12.5B · **Downloads**: 2K
- **Disk**: ~46.6 GB

### <a href="https://huggingface.co/TIGER-Lab/VideoScore" target="_blank">`TIGER-Lab/VideoScore`</a> [↑](#categories)
> visual-question-answering · apache-2.0

- **Used by**: `videoscore`
- **Parameters**: 8.3B · **Downloads**: 47
- **VRAM**: ~14 GB · **Disk**: ~14 GB
- **Source**: <a href="https://arxiv.org/abs/2406.15252" target="_blank">arXiv</a>

### <a href="https://huggingface.co/TIGER-Lab/VideoScore2" target="_blank">`TIGER-Lab/VideoScore2`</a> [↑](#categories)
> visual-question-answering · apache-2.0

- **Used by**: `videoscore2`
- **Parameters**: 8.3B · **Downloads**: 3K
- **VRAM**: ~16 GB · **Disk**: ~15 GB
- **Source**: <a href="https://arxiv.org/abs/2509.22799" target="_blank">arXiv</a>

### <a href="https://huggingface.co/Vchitect/VBench-2.0_models" target="_blank">`Vchitect/VBench-2.0_models`</a> [↑](#categories)

- **Used by**: `vbench2`
- **Disk**: 2.26 GB

### <a href="https://huggingface.co/ai-forever/kandinsky-video-motion-predictor" target="_blank">`ai-forever/kandinsky-video-motion-predictor`</a> [↑](#categories)

- **Used by**: `kandinsky_motion`
- **Parameters**: 115M · **Downloads**: 69
- **Disk**: ~440 MB

### <a href="https://huggingface.co/aimagelab/DICE_coherence_Idefics" target="_blank">`aimagelab/DICE_coherence_Idefics`</a> [↑](#categories)

- **Used by**: `dice_edit`
- **Disk**: ~2.8 GB

### <a href="https://huggingface.co/aimagelab/DICE_differencedet_Idefics" target="_blank">`aimagelab/DICE_differencedet_Idefics`</a> [↑](#categories)

- **Used by**: `dice_edit`
- **VRAM**: ~20 GB in bfloat16 · **Disk**: ~20 GB

### <a href="https://huggingface.co/anonymousdb/LOVE-Correspondence" target="_blank">`anonymousdb/LOVE-Correspondence`</a> [↑](#categories)
> apache-2.0

- **Used by**: `love_results`
- **Parameters**: 9.2B · **Downloads**: 12
- **Disk**: ~34.4 GB

### <a href="https://huggingface.co/anonymousdb/LOVE-Perception" target="_blank">`anonymousdb/LOVE-Perception`</a> [↑](#categories)
> apache-2.0

- **Used by**: `love_results`
- **Parameters**: 9.2B · **Downloads**: 11
- **Disk**: ~34.4 GB

### <a href="https://huggingface.co/chancharikm/qwen2.5-vl-7b-cam-motion" target="_blank">`chancharikm/qwen2.5-vl-7b-cam-motion`</a> [↑](#categories)
> video-text-to-text · other

- **Used by**: `camerabench`
- **Parameters**: 8.3B · **Downloads**: 869
- **Disk**: ~30.9 GB
- **Source**: <a href="https://arxiv.org/abs/2404.01291" target="_blank">arXiv</a>

### <a href="https://huggingface.co/cromsc/nima-mobilenet-aesthetic" target="_blank">`cromsc/nima-mobilenet-aesthetic`</a> [↑](#categories)

- **Used by**: `nima_onnx`

### <a href="https://huggingface.co/dandelin/vilt-b32-finetuned-vqa" target="_blank">`dandelin/vilt-b32-finetuned-vqa`</a> [↑](#categories)
> visual-question-answering · apache-2.0

- **Used by**: `commonsense`, `tifa`
- **Downloads**: 56K
- **VRAM**: ~500 MB · **Disk**: ~450 MB
- **Source**: <a href="https://arxiv.org/abs/2102.03334" target="_blank">arXiv</a>

### <a href="https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf" target="_blank">`depth-anything/Depth-Anything-V2-Small-hf`</a> [↑](#categories)
> depth-estimation · apache-2.0

- **Used by**: `depth_anything`
- **Parameters**: 25M · **Downloads**: 1.6M
- **VRAM**: ~200 MB · **Disk**: ~100 MB
- **Source**: <a href="https://arxiv.org/abs/2406.09414" target="_blank">arXiv</a>

### <a href="https://huggingface.co/facebook/VGGT-1B" target="_blank">`facebook/VGGT-1B`</a> [↑](#categories)
> image-to-3d · cc-by-nc-4.0

- **Used by**: `camera_trajectory`
- **Parameters**: 1.3B · **Downloads**: 420K
- **Disk**: ~4.7 GB
- **Source**: <a href="https://arxiv.org/abs/2503.11651" target="_blank">arXiv</a>

### <a href="https://huggingface.co/facebook/dinov2-base" target="_blank">`facebook/dinov2-base`</a> [↑](#categories)
> image-feature-extraction · apache-2.0

- **Used by**: `entitybench`, `fvd`, `subject_consistency`
- **Parameters**: 87M · **Downloads**: 2.6M
- **Disk**: ~330 MB
- **Source**: <a href="https://arxiv.org/abs/2304.07193" target="_blank">arXiv</a>

### <a href="https://huggingface.co/facebook/dinov2-large" target="_blank">`facebook/dinov2-large`</a> [↑](#categories)
> image-feature-extraction · apache-2.0

- **Used by**: `prdc_dinov2`, `verse_bench`
- **Parameters**: 304M · **Downloads**: 799K
- **Disk**: ~1.1 GB
- **Source**: <a href="https://arxiv.org/abs/2304.07193" target="_blank">arXiv</a>

### <a href="https://huggingface.co/facebook/dinov2-small" target="_blank">`facebook/dinov2-small`</a> [↑](#categories)
> image-feature-extraction · apache-2.0

- **Used by**: `i2i_learned`
- **Parameters**: 22M · **Downloads**: 4.7M
- **Disk**: ~84 MB
- **Source**: <a href="https://arxiv.org/abs/2304.07193" target="_blank">arXiv</a>

### <a href="https://huggingface.co/facebook/vjepa2-vitg-fpc64-256" target="_blank">`facebook/vjepa2-vitg-fpc64-256`</a> [↑](#categories)
> video-classification · apache-2.0

- **Used by**: `jedi`, `jedi_metric`
- **Parameters**: 1.0B · **Downloads**: 140K
- **Disk**: ~3.9 GB

### `fsmn-vad` [↑](#categories)

- **Used by**: `verse_bench`

### <a href="https://huggingface.co/google/siglip-base-patch16-224" target="_blank">`google/siglip-base-patch16-224`</a> [↑](#categories)
> zero-shot-image-classification · apache-2.0

- **Used by**: `i2i_learned`
- **Parameters**: 203M · **Downloads**: 1.4M
- **Disk**: ~775 MB
- **Source**: <a href="https://arxiv.org/abs/2303.15343" target="_blank">arXiv</a>

### <a href="https://huggingface.co/google/siglip-so400m-patch14-384" target="_blank">`google/siglip-so400m-patch14-384`</a> [↑](#categories)
> zero-shot-image-classification · apache-2.0

- **Used by**: `verse_bench`
- **Parameters**: 878M · **Downloads**: 1.5M
- **Disk**: ~3.3 GB
- **Source**: <a href="https://arxiv.org/abs/2303.15343" target="_blank">arXiv</a>

### <a href="https://huggingface.co/internlm/internlm2-chat-1_8b" target="_blank">`internlm/internlm2-chat-1_8b`</a> [↑](#categories)
> text-generation · other

- **Used by**: `mj_video`
- **Parameters**: 1.9B · **Downloads**: 5K
- **Disk**: ~7.0 GB
- **Source**: <a href="https://arxiv.org/abs/2403.17297" target="_blank">arXiv</a>

### <a href="https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K" target="_blank">`laion/CLIP-ViT-H-14-laion2B-s32B-b79K`</a> [↑](#categories)
> zero-shot-image-classification · mit

- **Used by**: `pickscore`
- **Parameters**: 986M · **Downloads**: 503K
- **Disk**: ~3.7 GB
- **Source**: <a href="https://arxiv.org/abs/1910.04867" target="_blank">arXiv</a>

### <a href="https://huggingface.co/laion/clap-htsat-fused" target="_blank">`laion/clap-htsat-fused`</a> [↑](#categories)
> audio-classification · apache-2.0

- **Used by**: `audio_text_alignment`, `clap_score`, `human_clap`, `laion_clap_score`, `ms_clap_score`, `pam`
- **Parameters**: 154M · **Downloads**: 8.3M
- **VRAM**: ~600 MB · **Disk**: ~600 MB
- **Source**: <a href="https://arxiv.org/abs/2211.06687" target="_blank">arXiv</a>

### <a href="https://huggingface.co/lero233/KVQ" target="_blank">`lero233/KVQ`</a> [↑](#categories)

- **Used by**: `kvq`

### <a href="https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf" target="_blank">`llava-hf/LLaVA-NeXT-Video-7B-hf`</a> [↑](#categories)
> video-text-to-text · llama2

- **Used by**: `videophy`
- **Parameters**: 7.1B · **Downloads**: 104K
- **Disk**: ~26.3 GB
- **Source**: <a href="https://arxiv.org/abs/2405.21075" target="_blank">arXiv</a>

### <a href="https://huggingface.co/llava-hf/llava-1.5-7b-hf" target="_blank">`llava-hf/llava-1.5-7b-hf`</a> [↑](#categories)
> image-text-to-text · llama2

- **Used by**: `commonsense`, `creativity`, `opens2v`, `vlm_judge`
- **Parameters**: 7.1B · **Downloads**: 2.3M
- **VRAM**: ~14 GB · **Disk**: ~14 GB

### <a href="https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf" target="_blank">`llava-hf/llava-v1.6-mistral-7b-hf`</a> [↑](#categories)
> image-text-to-text · apache-2.0

- **Used by**: `llm_descriptive_qa`
- **Parameters**: 7.6B · **Downloads**: 496K
- **VRAM**: ~14 GB · **Disk**: ~14 GB
- **Source**: <a href="https://arxiv.org/abs/2310.03744" target="_blank">arXiv</a>

### <a href="https://huggingface.co/m-a-p/MERT-v1-330M" target="_blank">`m-a-p/MERT-v1-330M`</a> [↑](#categories)
> audio-classification · cc-by-nc-4.0

- **Used by**: `mauve_audio_divergence`
- **Downloads**: 180K
- **Source**: <a href="https://arxiv.org/abs/2306.00107" target="_blank">arXiv</a>

### <a href="https://huggingface.co/microsoft/msclap" target="_blank">`microsoft/msclap`</a> [↑](#categories)
> ms-pl

- **Used by**: `ms_clap_score`
- **Source**: <a href="https://arxiv.org/abs/2309.05767" target="_blank">arXiv</a>

### <a href="https://huggingface.co/microsoft/xclip-base-patch32" target="_blank">`microsoft/xclip-base-patch32`</a> [↑](#categories)
> video-classification · mit

- **Used by**: `embedding`, `video_text_matching`
- **Parameters**: 197M · **Downloads**: 100K
- **VRAM**: ~600 MB · **Disk**: ~600 MB
- **Source**: <a href="https://arxiv.org/abs/2208.02816" target="_blank">arXiv</a>

### <a href="https://huggingface.co/minchul/cvlface_adaface_ir101_ms1mv2/resolve/afdb94f8190f4cd8ea1467258ce65f1d76033b63/model.safetensors" target="_blank">`minchul/cvlface_adaface_ir101_ms1mv2`</a> [↑](#categories)
> feature-extraction

- **Used by**: `adaface`
- **Parameters**: 65M · **Downloads**: 46
- **Disk**: 261.0 MB
- **Source**: <a href="https://arxiv.org/abs/2204.00964" target="_blank">arXiv</a>

### <a href="https://huggingface.co/minchul/cvlface_adaface_ir101_webface12m/resolve/54f602a0737bd1ee4a4e7e9fd089a485f397fefd/model.safetensors" target="_blank">`minchul/cvlface_adaface_ir101_webface12m`</a> [↑](#categories)
> feature-extraction

- **Used by**: `adaface`
- **Parameters**: 65M · **Downloads**: 648
- **Disk**: 261.0 MB
- **Source**: <a href="https://arxiv.org/abs/2204.00964" target="_blank">arXiv</a>

### <a href="https://huggingface.co/minchul/cvlface_adaface_ir101_webface4m/resolve/f2b38d9e24bfe301490d8dd081d8924b102333dd/model.safetensors" target="_blank">`minchul/cvlface_adaface_ir101_webface4m`</a> [↑](#categories)
> feature-extraction

- **Used by**: `adaface`
- **Parameters**: 65M · **Downloads**: 303
- **Disk**: 261.0 MB
- **Source**: <a href="https://arxiv.org/abs/2204.00964" target="_blank">arXiv</a>

### <a href="https://huggingface.co/minchul/cvlface_adaface_ir18_webface4m/resolve/0dd53f188fa27968b0a1326970ebf4aeb37ce2ca/model.safetensors" target="_blank">`minchul/cvlface_adaface_ir18_webface4m`</a> [↑](#categories)
> feature-extraction

- **Used by**: `adaface`
- **Parameters**: 24M · **Downloads**: 106
- **Disk**: 97.1 MB
- **Source**: <a href="https://arxiv.org/abs/2204.00964" target="_blank">arXiv</a>

### <a href="https://huggingface.co/minchul/cvlface_adaface_ir50_webface4m/resolve/60a65befbcf7e19284c4f3ac730f56867ed29594/model.safetensors" target="_blank">`minchul/cvlface_adaface_ir50_webface4m`</a> [↑](#categories)
> feature-extraction

- **Used by**: `adaface`
- **Parameters**: 44M · **Downloads**: 151
- **Disk**: 175.4 MB
- **Source**: <a href="https://arxiv.org/abs/2204.00964" target="_blank">arXiv</a>

### <a href="https://huggingface.co/nvidia/quality-classifier-deberta" target="_blank">`nvidia/quality-classifier-deberta`</a> [↑](#categories)
> apache-2.0

- **Used by**: `nemo_curator`
- **Parameters**: 184M · **Downloads**: 5K
- **Disk**: ~701 MB
- **Source**: <a href="https://arxiv.org/abs/2111.09543" target="_blank">arXiv</a>

### <a href="https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512" target="_blank">`nvidia/segformer-b0-finetuned-ade-512-512`</a> [↑](#categories)
> image-segmentation · other

- **Used by**: `semantic_segmentation_consistency`
- **Parameters**: 4M · **Downloads**: 316K
- **Disk**: ~14 MB
- **Source**: <a href="https://arxiv.org/abs/2105.15203" target="_blank">arXiv</a>

### <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">`openai/clip-vit-base-patch32`</a> [↑](#categories)
> zero-shot-image-classification

- **Used by**: `action_recognition`, `background_consistency`, `clifvqa`, `clip_image_similarity`, `clip_temporal`, `concept_presence`, `creativity`, `dataset_analytics`, `deepfake_detection`, `entitybench`, `generative_distribution`, `geneval`, `harmful_content`, `i2i_learned`, `opens2v`, `scene_tagging`, `sd_reference`, `semantic_alignment`, `tc_bench`, `umap_projection`, `video_text_matching`, `video_type_classifier`, `world_consistency`
- **Downloads**: 20.0M
- **VRAM**: ~600 MB · **Disk**: ~600 MB
- **Source**: <a href="https://arxiv.org/abs/2103.00020" target="_blank">arXiv</a>

### <a href="https://huggingface.co/openai/clip-vit-large-patch14" target="_blank">`openai/clip-vit-large-patch14`</a> [↑](#categories)
> zero-shot-image-classification

- **Used by**: `aesthetic_scoring`
- **Parameters**: 428M · **Downloads**: 6.3M
- **VRAM**: ~1.5 GB · **Disk**: ~1.7 GB
- **Source**: <a href="https://arxiv.org/abs/2103.00020" target="_blank">arXiv</a>

### <a href="https://huggingface.co/openai/clip-vit-large-patch14-336" target="_blank">`openai/clip-vit-large-patch14-336`</a> [↑](#categories)
> zero-shot-image-classification

- **Used by**: `cmmd`
- **Downloads**: 2.7M

### <a href="https://huggingface.co/q-future/VQA-UGC-Scorer-llava_qwen" target="_blank">`q-future/VQA-UGC-Scorer-llava_qwen`</a> [↑](#categories)
> apache-2.0

- **Used by**: `vqa2`
- **Parameters**: 8.1B · **Downloads**: 19
- **VRAM**: ~18 GB · **Disk**: 16.2 GB

### <a href="https://huggingface.co/q-future/one-align" target="_blank">`q-future/one-align`</a> [↑](#categories)
> zero-shot-image-classification · mit

- **Used by**: `q_align`, `rqvqa`, `vmbench_mss`
- **Downloads**: 150K
- **Source**: <a href="https://arxiv.org/abs/2312.17090" target="_blank">arXiv</a>

### `roberta-base` [↑](#categories)

- **Used by**: `verse_bench`

### <a href="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0" target="_blank">`stabilityai/stable-diffusion-xl-base-1.0`</a> [↑](#categories)
> text-to-image · openrail++

- **Used by**: `sd_reference`
- **Parameters**: 2.6B · **Downloads**: 1.6M
- **Disk**: ~9.6 GB
- **Source**: <a href="https://arxiv.org/abs/2307.01952" target="_blank">arXiv</a>

### <a href="https://huggingface.co/yuvalkirstain/PickScore_v1" target="_blank">`yuvalkirstain/PickScore_v1`</a> [↑](#categories)
> zero-shot-image-classification

- **Used by**: `pickscore`
- **Parameters**: 986M · **Downloads**: 248K
- **Disk**: ~3.7 GB
- **Source**: <a href="https://arxiv.org/abs/2305.01569" target="_blank">arXiv</a>

### <a href="https://huggingface.co/zhudi2825/MuQ-Eval-A1" target="_blank">`zhudi2825/MuQ-Eval-A1`</a> [↑](#categories)
> audio-classification · mit

- **Used by**: `muq_eval`
- **Downloads**: 2K
- **Disk**: 1.34 GB
- **Source**: <a href="https://arxiv.org/abs/2603.22677" target="_blank">arXiv</a>

## Weight File Repos

### <a href="https://huggingface.co/AkaneTendo25/ayase-runtime-assets" target="_blank">`AkaneTendo25/ayase-runtime-assets`</a> [↑](#categories)
> Pre-trained weight files for ayase modules

- `24-01-04T16-39-21.pt` — used by `audio_visual_sync`, `av_desync`, `av_sync`
- `CLIPIQA+_ViTL14_512-e66488f2.pth` — used by `brightrate`
- `CONTRIQUE_checkpoint25.tar` — used by `brightrate`
- `DOVER.pth` — used by `dover`
- `FAST_VQA_3D_1_1.pth` — used by `fast_vqa`
- `FAST_VQA_B_1_4.pth` — used by `fast_vqa`
- `FAST_VQA_M_1_4.pth` — used by `fast_vqa`
- `SAMA-baseline_val-ltest_s_dev_v0.0.pth` — used by `sama`
- `ViT-B-32.safetensors` — used by `brightrate`
- `ViT-B-32.safetensors` — used by `i2v_similarity`
- `ViT-L-14.safetensors` — used by `brightrate`
- `alex.pth` — used by `i2v_similarity`
- `brightrate_brightvq.pt` — used by `brightrate`
- `convnext_tiny_1k_224_ema.pth` — used by `dover`
- `dino_vitbase16_pretrain.pth` — used by `dreamsim`, `dreamsim_metric`
- `dinov2_vitb14_pretrain.pth` — used by `i2v_similarity`
- `flownet.pkl` — used by `motion_smoothness`
- `frames_modelparameters.mat` — used by `brightrate`
- `imagebind_huge.pth` — used by `imagebind_score`
- `model.safetensors` — used by `song_eval`
- `nisqa.tar` — used by `audio_nisqa`
- `onnx_dover.onnx` — used by `dover`
- `raft_large_C_T_SKHT_V2-ff5fadd5.pth` — used by `advanced_flow`
- `raft_small_C_T_V2-01064c6d.pth` — used by `advanced_flow`
- `rtmpose_m.onnx` — used by `object_integrity`
- `sac+logos+ava1-l14-linearMSE.pth` — used by `aesthetic_scoring`
- `yolox_m.onnx` — used by `object_integrity`

### <a href="https://huggingface.co/cromsc/nima-mobilenet-aesthetic" target="_blank">`cromsc/nima-mobilenet-aesthetic`</a> [↑](#categories)
> Pre-trained weight files for ayase modules

- `nima_mobilenet_aesthetic.onnx` — used by `nima_onnx`

### <a href="https://huggingface.co/facebook/cotracker" target="_blank">`facebook/cotracker`</a> [↑](#categories)
> Pre-trained weight files for ayase modules

- `cotracker2.pth` — used by `vbench2`

## Local Weight Files

Checkpoint files downloaded directly by Ayase modules or supplied through a local model path.

### `24-01-04T16-39-21.pt` [↑](#categories)

- **Used by**: `verse_bench`
- **Task**: Syncformer AV sync model (AST + MotionFormer)

### `630k-audioset-fusion-best.pt` [↑](#categories)

- **Used by**: `verse_bench`
- **Task**: LAION CLAP fusion model for audio-text similarity and FAD

### `Cnn14_mAP=0.431.pth` [↑](#categories)

- **Used by**: `audio_isc`
- **Task**: PANNs CNN14 pretrained weights, auto-downloaded by panns_inference on first use

### <a href="https://github.com/hjbahng/cyclereward/releases/download/v1.0.0/CycleReward-Combo.pth" target="_blank">`CycleReward-Combo.pth`</a> [↑](#categories)

- **Used by**: `cycle_reward`
- **Task**: CycleReward combined I2T/T2I preference checkpoint

### <a href="https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/dover/DOVER.pth" target="_blank">`DOVER.pth`</a> [↑](#categories)

- **Used by**: `dover`
- **Task**: Native DOVER video quality weights
- **Notes**: Resolved from weights_path or models_dir

### <a href="https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/fast_vqa/FAST_VQA_3D_1_1.pth" target="_blank">`FAST_VQA_3D_1_1.pth`</a> [↑](#categories)

- **Used by**: `fast_vqa`
- **Task**: FAST-VQA / FasterVQA video quality checkpoint

### <a href="https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/fast_vqa/FAST_VQA_B_1_4.pth" target="_blank">`FAST_VQA_B_1_4.pth`</a> [↑](#categories)

- **Used by**: `fast_vqa`
- **Task**: FAST-VQA base video quality checkpoint

### <a href="https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/fast_vqa/FAST_VQA_M_1_4.pth" target="_blank">`FAST_VQA_M_1_4.pth`</a> [↑](#categories)

- **Used by**: `fast_vqa`
- **Task**: FAST-VQA motion-aware video quality checkpoint

### `Synchformer` [↑](#categories)

- **Used by**: `audio_visual_sync`, `av_desync`, `av_sync`
- **Task**: Optional learned A/V offset backend when local weights are configured

### <a href="https://github.com/Q-Future/Visual-Question-Answering-for-Video-Quality-Assessment/archive/9087c7952052088a6eb01bac4408bff903ab9e41.zip" target="_blank">`VQA2-source-9087c7952052088a6eb01bac4408bff903ab9e41.zip`</a> [↑](#categories)

- **Used by**: `vqa2`
- **Task**: Pinned upstream VQA² LLaVA runtime source
- **Notes**: Apache-2.0

### `aesthetic_predictor_v2_5.pth` [↑](#categories)

- **Used by**: `verse_bench`
- **Task**: Aesthetic Predictor V2.5 head weights

### <a href="https://download.pytorch.org/models/r3d_18-b3b3357e.pth" target="_blank">`cgvqm/r3d_18-b3b3357e.pth`</a> [↑](#categories)

- **Used by**: `cgvqm`
- **Task**: Kinetics-400 R3D-18 features used by upstream CGVQM

### `ckpt_koniq10k.pt` [↑](#categories)

- **Used by**: `verse_bench`
- **Task**: MANIQA Swin-T quality assessment (KonIQ-10k)

### <a href="https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/dover/convnext_tiny_1k_224_ema.pth" target="_blank">`convnext_tiny_1k_224_ema.pth`</a> [↑](#categories)

- **Used by**: `dover`
- **Task**: ConvNeXt-Tiny aesthetic backbone

### <a href="https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/fast_vqa/FAST_VQA_B_1_4.pth" target="_blank">`fast_vqa/FAST_VQA_B_1_4.pth`</a> [↑](#categories)

- **Used by**: `rqvqa`
- **Task**: RQ-VQA FAST-VQA 768-D feature encoder

### <a href="https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/motion_smoothness/flownet.pkl" target="_blank">`flownet.pkl`</a> [↑](#categories)

- **Used by**: `motion_smoothness`
- **Task**: Bundled RIFE HD v3 interpolation weights

### <a href="https://zenodo.org/records/3987831/files/Wavegram_Logmel_Cnn14_mAP%3D0.439.pth" target="_blank">`kad/Wavegram_Logmel_Cnn14_mAP=0.439.pth`</a> [↑](#categories)

- **Used by**: `kad`
- **Task**: PANNs Wavegram-Logmel audio embedding checkpoint

### <a href="https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/nisqa/nisqa.tar" target="_blank">`nisqa.tar`</a> [↑](#categories)

- **Used by**: `audio_nisqa`
- **Task**: NISQAv2 multidimensional speech quality (MIT)
- **Notes**: ~1 MB; vendored source at ayase/third_party/nisqa/

### <a href="https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/dover/onnx_dover.onnx" target="_blank">`onnx_dover.onnx`</a> [↑](#categories)

- **Used by**: `dover`
- **Task**: Optional ONNX DOVER backend

### <a href="https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/rqvqa/LIQE.pt" target="_blank">`rqvqa/LIQE.pt`</a> [↑](#categories)

- **Used by**: `rqvqa`
- **Task**: RQ-VQA LIQE feature encoder

### <a href="https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/rqvqa/SLOWFAST_8x8_R50.pyth" target="_blank">`rqvqa/SLOWFAST_8x8_R50.pyth`</a> [↑](#categories)

- **Used by**: `rqvqa`
- **Task**: RQ-VQA SlowFast-R50 motion encoder

### <a href="https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/rqvqa/Swin_b_384_in22k_SlowFast_Fast_LLM_LIQE_FASTVQA_BoT_NTIREVideo_plcc_NR_v0_epoch_4_SRCC_0.905999.pth" target="_blank">`rqvqa/Swin_b_384_in22k_SlowFast_Fast_LLM_LIQE_FASTVQA_BoT_NTIREVideo_plcc_NR_v0_epoch_4_SRCC_0.905999.pth`</a> [↑](#categories)

- **Used by**: `rqvqa`
- **Task**: RQ-VQA released ensemble fold

### <a href="https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/rqvqa/Swin_b_384_in22k_SlowFast_Fast_LLM_LIQE_FASTVQA_BoT_NTIREVideo_plcc_NR_v0_epoch_9_SRCC_0.885692.pth" target="_blank">`rqvqa/Swin_b_384_in22k_SlowFast_Fast_LLM_LIQE_FASTVQA_BoT_NTIREVideo_plcc_NR_v0_epoch_9_SRCC_0.885692.pth`</a> [↑](#categories)

- **Used by**: `rqvqa`
- **Task**: RQ-VQA released ensemble fold

### <a href="https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/rqvqa/Swin_b_384_in22k_SlowFast_Fast_LLM_LIQE_FASTVQA_BoT_NTIREVideo_plcc_NR_v1_epoch_19_SRCC_0.923127.pth" target="_blank">`rqvqa/Swin_b_384_in22k_SlowFast_Fast_LLM_LIQE_FASTVQA_BoT_NTIREVideo_plcc_NR_v1_epoch_19_SRCC_0.923127.pth`</a> [↑](#categories)

- **Used by**: `rqvqa`
- **Task**: RQ-VQA released ensemble fold

### <a href="https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/rqvqa/Swin_b_384_in22k_SlowFast_Fast_LLM_LIQE_FASTVQA_BoT_NTIREVideo_plcc_NR_v1_epoch_22_SRCC_0.894115.pth" target="_blank">`rqvqa/Swin_b_384_in22k_SlowFast_Fast_LLM_LIQE_FASTVQA_BoT_NTIREVideo_plcc_NR_v1_epoch_22_SRCC_0.894115.pth`</a> [↑](#categories)

- **Used by**: `rqvqa`
- **Task**: RQ-VQA released ensemble fold

### <a href="https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/rqvqa/Swin_b_384_in22k_SlowFast_Fast_LLM_LIQE_FASTVQA_BoT_NTIREVideo_plcc_NR_v2_epoch_21_SRCC_0.924423.pth" target="_blank">`rqvqa/Swin_b_384_in22k_SlowFast_Fast_LLM_LIQE_FASTVQA_BoT_NTIREVideo_plcc_NR_v2_epoch_21_SRCC_0.924423.pth`</a> [↑](#categories)

- **Used by**: `rqvqa`
- **Task**: RQ-VQA released ensemble fold

### <a href="https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/rqvqa/Swin_b_384_in22k_SlowFast_Fast_LLM_LIQE_FASTVQA_BoT_NTIREVideo_plcc_NR_v2_epoch_25_SRCC_0.913571.pth" target="_blank">`rqvqa/Swin_b_384_in22k_SlowFast_Fast_LLM_LIQE_FASTVQA_BoT_NTIREVideo_plcc_NR_v2_epoch_25_SRCC_0.913571.pth`</a> [↑](#categories)

- **Used by**: `rqvqa`
- **Task**: RQ-VQA released ensemble fold

### <a href="https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/rqvqa/Swin_b_384_in22k_SlowFast_Fast_LLM_LIQE_FASTVQA_BoT_NTIREVideo_plcc_NR_v3_epoch_16_SRCC_0.901800.pth" target="_blank">`rqvqa/Swin_b_384_in22k_SlowFast_Fast_LLM_LIQE_FASTVQA_BoT_NTIREVideo_plcc_NR_v3_epoch_16_SRCC_0.901800.pth`</a> [↑](#categories)

- **Used by**: `rqvqa`
- **Task**: RQ-VQA released ensemble fold

### <a href="https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/rqvqa/Swin_b_384_in22k_SlowFast_Fast_LLM_LIQE_FASTVQA_BoT_NTIREVideo_plcc_NR_v3_epoch_8_SRCC_0.896798.pth" target="_blank">`rqvqa/Swin_b_384_in22k_SlowFast_Fast_LLM_LIQE_FASTVQA_BoT_NTIREVideo_plcc_NR_v3_epoch_8_SRCC_0.896798.pth`</a> [↑](#categories)

- **Used by**: `rqvqa`
- **Task**: RQ-VQA released ensemble fold

### <a href="https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/rqvqa/Swin_b_384_in22k_SlowFast_Fast_LLM_LIQE_FASTVQA_BoT_NTIREVideo_plcc_NR_v4_epoch_14_SRCC_0.904949.pth" target="_blank">`rqvqa/Swin_b_384_in22k_SlowFast_Fast_LLM_LIQE_FASTVQA_BoT_NTIREVideo_plcc_NR_v4_epoch_14_SRCC_0.904949.pth`</a> [↑](#categories)

- **Used by**: `rqvqa`
- **Task**: RQ-VQA released ensemble fold

### <a href="https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/rqvqa/Swin_b_384_in22k_SlowFast_Fast_LLM_LIQE_FASTVQA_BoT_NTIREVideo_plcc_NR_v4_epoch_6_SRCC_0.905095.pth" target="_blank">`rqvqa/Swin_b_384_in22k_SlowFast_Fast_LLM_LIQE_FASTVQA_BoT_NTIREVideo_plcc_NR_v4_epoch_6_SRCC_0.905095.pth`</a> [↑](#categories)

- **Used by**: `rqvqa`
- **Task**: RQ-VQA released ensemble fold

### <a href="https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/rtmpose_fidelity/rtmpose_m.onnx" target="_blank">`rtmpose_m.onnx`</a> [↑](#categories)

- **Used by**: `object_integrity`
- **Task**: RTMPose keypoint estimator (rtmlib backend)
- **Notes**: Shared with rtmpose_fidelity

### <a href="https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/sama/SAMA-baseline_val-ltest_s_dev_v0.0.pth" target="_blank">`sama/SAMA-baseline_val-ltest_s_dev_v0.0.pth`</a> [↑](#categories)

- **Used by**: `sama`
- **Task**: SAMA LSVQ baseline video quality checkpoint

### `song_eval/model.safetensors` [↑](#categories)

- **Used by**: `song_eval`
- **Task**: SongEval Generator aesthetic head weights

### `syncnet_v2.model` [↑](#categories)

- **Used by**: `verse_bench`
- **Task**: SyncNet v2 lip-sync model

### <a href="https://raw.githubusercontent.com/google/uvq/811b6b1b7c085a9ac59ee5e3a03c560be18fe91c/uvq1p5_pytorch/checkpoints/aggregation_net.pth" target="_blank">`uvq1p5/aggregation_net.pth`</a> [↑](#categories)

- **Used by**: `uvq`
- **Task**: Google UVQ 1.5 aggregation network
- **Disk**: 0.3 MB
- **Notes**: Apache-2.0; pinned to google/uvq commit 811b6b1b7c085a9ac59ee5e3a03c560be18fe91c

### <a href="https://raw.githubusercontent.com/google/uvq/811b6b1b7c085a9ac59ee5e3a03c560be18fe91c/uvq1p5_pytorch/checkpoints/content_net.pth" target="_blank">`uvq1p5/content_net.pth`</a> [↑](#categories)

- **Used by**: `uvq`
- **Task**: Google UVQ 1.5 content network
- **Disk**: 15.3 MB
- **Notes**: Apache-2.0; pinned to google/uvq commit 811b6b1b7c085a9ac59ee5e3a03c560be18fe91c

### <a href="https://raw.githubusercontent.com/google/uvq/811b6b1b7c085a9ac59ee5e3a03c560be18fe91c/uvq1p5_pytorch/checkpoints/distortion_net.pth" target="_blank">`uvq1p5/distortion_net.pth`</a> [↑](#categories)

- **Used by**: `uvq`
- **Task**: Google UVQ 1.5 distortion network
- **Disk**: 15.3 MB
- **Notes**: Apache-2.0; pinned to google/uvq commit 811b6b1b7c085a9ac59ee5e3a03c560be18fe91c

### <a href="https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/rtmpose_fidelity/yolox_m.onnx" target="_blank">`yolox_m.onnx`</a> [↑](#categories)

- **Used by**: `object_integrity`
- **Task**: YOLOX person detector (rtmlib backend)
- **Notes**: Shared with rtmpose_fidelity

## pyiqa Metrics (55)

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
| `clipiqa+` | CLIP-based image quality assessment | MIT (pyiqa) | Yes | `clip_iqa` |
| `cnniqa` | CNN-based blind image quality | research | — | `cnniqa` |
| `compare2score` | Comparative-to-absolute quality scoring | research | — | `compare2score` |
| `contrique` | Contrastive image quality representation | research | — | `contrique` |
| `conviqt` | Contrastive NR-VQA | research | — | `conviqt` |
| `creativity` | Creative quality assessment | research | — | `creativity` |
| `cw_ssim` | Complex wavelet SSIM | MIT (pyiqa) | Yes | `cw_ssim` |
| `dbcnn` | Deep bilinear CNN for blind IQA | research | — | `dbcnn` |
| `deepdc` | Deep distribution conformance | research | — | `deepdc` |
| `deepwsd` | Deep Wasserstein distance IQA | research | — | `deepwsd` |
| `dmm` | Detail model metric FR-IQA | research | — | `dmm` |
| `dover` | pyiqa DOVER fallback metric | MIT (pyiqa) | Yes | `dover` |
| `face_iqa` | TOPIQ face-specific quality | research | — | `face_iqa` |
| `hyperiqa` | Adaptive hypernetwork NR-IQA | research | — | `hyperiqa` |
| `ilniqe` | Integrated local NIQE | BSD-2-Clause | Yes | `ilniqe` |
| `laion_aes` | LAION aesthetic scoring (CLIP-based) | MIT | Yes | `creativity`, `laion_aesthetic` |
| `laion_aesthetic` | LAION Aesthetics V2 predictor | research | — | `laion_aesthetic` |
| `liqe` | Learned image quality evaluator (multi-task) | research | — | `liqe` |
| `maclip` | Multi-attribute CLIP quality scoring | research | — | `maclip` |
| `mad` | Most apparent distortion FR-IQA | research | — | `mad` |
| `maniqa` | Multi-dimension attention NR-IQA | Apache-2.0 | Yes | `maniqa` |
| `mdtvsfa` | Multi-dimensional temporal-spatial VQA | research | — | `mdtvsfa` |
| `mouth_quality` | IQA | research | — | `mouth_quality` |
| `msswd` | Multi-scale sliced Wasserstein distance | research | — | `msswd` |
| `musiq` | Multi-scale image quality transformer | Apache-2.0 (Google) | Yes | `mouth_quality`, `musiq` |
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
| `ssimc` | Complex wavelet SSIM-C FR | MIT (pyiqa) | Yes | `ssimc` |
| `topiq` | TOPIQ transformer quality | research | — | `topiq` |
| `topiq_fr` | Transformer-based FR image quality | MIT (pyiqa) | Yes | `topiq_fr` |
| `topiq_nr` | Transformer-based NR image quality | MIT (pyiqa) | Yes | `topiq` |
| `topiq_nr-face` | TOPIQ face-specific quality | MIT (pyiqa) | Yes | `face_iqa` |
| `tres` | Transformer relative quality estimation | research | — | `tres` |
| `unique` | Unified NR-IQA with contrastive learning | research | — | `unique` |
| `wadiqam` | Weighted average deep IQA | research | — | `wadiqam` |
| `wadiqam_fr` | Weighted average deep FR-IQA | research | — | `wadiqam_fr` |
| `wadiqam_nr` | Weighted average deep NR-IQA | research | — | `wadiqam` |

## torchvision Models

Bundled with `pip install torchvision`. Weights download on first use.

### `torchvision/idefics2` [↑](#categories)
> torchvision

- **Used by**: `videoscore`

### `torchvision/imagebind_model` [↑](#categories)
> torchvision

- **Used by**: `imagebind_score`

### `torchvision/inception_v3` [↑](#categories)
> torchvision · BSD-3-Clause

- **Used by**: `fid`, `inception_score`, `kid`, `sfid`
- **VRAM**: ~200 MB · **Disk**: ~100 MB

### `torchvision/llama` [↑](#categories)
> torchvision

- **Used by**: `mj_video`

### `torchvision/qwen2` [↑](#categories)
> torchvision

- **Used by**: `worldmodelbench`

### `torchvision/r3d_18` [↑](#categories)
> torchvision · BSD-3-Clause

- **Used by**: `fvd`
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

- **Used by**: `grafiqs`, `modularbvqa`, `vsfa`, `watermark_classifier`

### `torchvision/inception_v3` [↑](#categories)
> torchvision · BSD-3-Clause

- **Used by**: `fid`, `kid`, `sfid`

### `torchvision/r3d_18` [↑](#categories)
> torchvision · BSD-3-Clause

- **Used by**: `fvd`, `kvd`

### `torchvision/vggt` [↑](#categories)
> torchvision

- **Used by**: `camera_trajectory`

### `torchvision/zip` [↑](#categories)
> torchvision

- **Used by**: `vbench2`

## CLIP / OpenCLIP

### `CLIP ViT-B-32` [↑](#categories)

- **Used by**: `i2v_similarity`

### `open_clip/ViT-B-32` [↑](#categories)

- **Used by**: `semantic_alignment`

## torch.hub

### `facebookresearch/co-tracker` [↑](#categories)
> torch.hub · Apache-2.0

- **Used by**: `chronomagic`, `dynamics_controllability`, `physics`

### `facebookresearch/dinov2` [↑](#categories)
> torch.hub · Apache-2.0

- **Used by**: `dino_face_identity`, `opens2v`, `spectral_complexity`, `world_consistency`

### `intel-isl/MiDaS` [↑](#categories)
> torch.hub · MIT

- **Used by**: `depth_consistency`, `depth_map_quality`
- **VRAM**: ~400 MB · **Disk**: ~400 MB

### `sarulab-speech/UTMOSv2` [↑](#categories)
> torch.hub

- **Used by**: `audio_utmos_v2`

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

### `ArcFace` [↑](#categories)

- **Used by**: `face_cross_similarity`
- **Install**: `pip install deepface`

### `aesthetic-predictor-v2-5` [↑](#categories)
> Aesthetic Predictor V2.5 (SigLIP)

- **Used by**: `aesthetic`
- **Install**: `pip install aesthetic-predictor-v2-5`

### `audiobox-aesthetics` [↑](#categories)

- **Used by**: `verse_bench`
- **Install**: `pip install audiobox_aesthetics`

### `cleanfid` [↑](#categories)

- **Used by**: `kid`
- **Install**: `pip install clean-fid`

### `cvvdp` [↑](#categories)
> MIT; calibration and display-model data ship in the package

- **Used by**: `cvvdp`
- **Install**: `pip install 'cvvdp>=0.5.6,<0.6'`

### `cyclereward` [↑](#categories)

- **Used by**: `cycle_reward`
- **Install**: `pip install cyclereward==0.1.7`

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

### `faster-whisper` [↑](#categories)

- **Used by**: `asr_transcribe`
- **Install**: `pip install faster-whisper`

### `fasttext` [↑](#categories)
> FastText (text classification)

- **Used by**: `nemo_curator`
- **Install**: `pip install fasttext`

### `frechet_audio_distance` [↑](#categories)

- **Used by**: `fad`
- **Install**: `pip install frechet_audio_distance`

### `hear21passt` [↑](#categories)

- **Used by**: `audio_isc`, `fad`, `verse_bench`
- **Install**: `pip install hear21passt`

### `hpsv2` [↑](#categories)

- **Used by**: `hpsv2`
- **Install**: `pip install hpsv2`

### `insightface` [↑](#categories)
> InsightFace (face recognition)

- **Used by**: `active_speaker`, `adaface`, `concept_presence`, `dino_face_identity`, `entitybench`, `face_cross_similarity`, `grafiqs`, `identity_loss`, `magface`, `multi_subject_identity`
- **Install**: `pip install insightface`

### `joblib` [↑](#categories)
> Joblib (serialized model storage)

- **Used by**: `brightrate`, `chipqa`, `hdr_chipqa`, `hdrmax`, `tlvqm`, `videval`
- **Install**: `pip install joblib`

### `jxlpy` [↑](#categories)
> JPEG XL codec library

- **Used by**: `butteraugli`
- **Install**: `pip install jxlpy`

### `kadtk` [↑](#categories)

- **Used by**: `kad`
- **Install**: `pip install kadtk`

### `lpips` [↑](#categories)

- **Used by**: `i2i_learned`, `image_lpips`
- **Install**: `pip install lpips`

### `mad_metric` [↑](#categories)

- **Used by**: `mauve_audio_divergence`
- **Install**: `pip install mad_metric`

### `mediapipe` [↑](#categories)
> MediaPipe (face/pose/hand detection)

- **Used by**: `anatomy_check`, `concept_presence`, `face_fidelity`, `face_landmark_quality`, `human_fidelity`
- **Install**: `pip install mediapipe`

### `muq` [↑](#categories)

- **Used by**: `muq_eval`
- **Install**: `pip install muq`

### `onnxruntime` [↑](#categories)
> ONNX Runtime (model inference)

- **Used by**: `dover`, `nima_onnx`
- **Install**: `pip install onnxruntime`

### `openai-whisper` [↑](#categories)

- **Used by**: `asr_transcribe`
- **Install**: `pip install openai-whisper`

### `panns_cnn14` [↑](#categories)

- **Used by**: `audio_kl`
- **Install**: `pip install panns_inference`

### `panns_inference` [↑](#categories)

- **Used by**: `audio_isc`, `fad`
- **Install**: `pip install panns_inference`

### `passt_s_kd_p16_128_ap486` [↑](#categories)

- **Used by**: `audio_kl`
- **Install**: `pip install hear21passt`

### `piq` [↑](#categories)
> piq (PyTorch Image Quality)

- **Used by**: `dists`, `perceptual_fr`, `vif`
- **Install**: `pip install piq`

### `ptlflow` [↑](#categories)
> ptlflow optical flow models

- **Used by**: `ptlflow_motion`
- **Install**: `pip install ptlflow`

### `rife_model` [↑](#categories)

- **Used by**: `motion_smoothness`
- **Install**: `pip install rife-model`

### `silero-vad` [↑](#categories)

- **Used by**: `silent_lip_stability`
- **Install**: `pip install silero-vad`

### `stlpips-pytorch` [↑](#categories)
> ST-LPIPS spatiotemporal perceptual

- **Used by**: `st_lpips`
- **Install**: `pip install stlpips-pytorch`

### `torch-fidelity` [↑](#categories)

- **Used by**: `kid`
- **Install**: `pip install torch-fidelity`

### `torchmetrics[audio]` [↑](#categories)
> TorchMetrics (DNSMOS, etc.)

- **Used by**: `dnsmos`
- **Install**: `pip install torchmetrics[audio]`

### `ultralytics` [↑](#categories)
> YOLOv8 object detection

- **Used by**: `geneval`, `object_detection`, `object_integrity`, `object_permanence`, `opens2v`, `rtmpose_fidelity`, `t2v_compbench`, `vbench2`
- **Install**: `pip install ultralytics`

### `vebench==1.0.0` [↑](#categories)
> MIT; AAAI 2025; CUDA required

- **Used by**: `vebench`
- **Install**: `pip install 'vebench==1.0.0'`

### `vendi_score` [↑](#categories)

- **Used by**: `vendi`
- **Install**: `pip install vendi-score`

## Quick Install Guide

Install Ayase with the bundled runtime dependencies:

```bash
pip install ayase
```
