# Ayase Metrics Reference

> **Version 0.1.17** · Generated 2026-03-21 02:16 · **312 modules** · **341 metrics**
>
> `ayase modules docs -o METRICS.md` to regenerate
>
> Tests: `pytest tests/` (light) · `pytest tests/ --full` (with ML models)

## Summary

![Summary Dashboard](docs/chart_summary.png)

### Modules by Category

![Module Distribution by Category](docs/chart_categories.png)

### By Input Type

![Input Type Distribution](docs/chart_input_types.png)

### Speed Tiers

![Speed Tiers](docs/chart_speed.png)

### Backend Usage

![Backend Usage](docs/chart_backends.png)

### Top Required Packages

![Top Required Packages](docs/chart_packages.png)
---

## No-Reference Quality (96 metrics)

| Metric | Dir | Range | Module | Input | Speed | GPU | Backend | Source | Test | Description |
|--------|-----|-------|--------|-------|-------|-----|---------|--------|------|-------------|
| `adadqa_score` | ↑ higher=better | higher=better | `adadqa` | img/vid | ⚡ fast |  | heuristic → native | — | — | Ada-DQA adaptive diverse quality feature VQA (ACM MM 2023) |
| `afine_score` | ↑ higher=better | — | `afine` | img/vid | ⏱️ medium | ✓ | — | — | — | A-FINE adaptive fidelity-naturalness IQA (CVPR 2025) |
| `aigcvqa_aesthetic` | — | — | `aigcvqa` | img/vid +cap | ⚡ fast |  | heuristic → native | — | — | AIGC-VQA holistic 3-branch AIGC perception (CVPRW 2024) |
| `aigcvqa_technical` | — | — | `aigcvqa` | img/vid +cap | ⚡ fast |  | heuristic → native | — | — | AIGC-VQA holistic 3-branch AIGC perception (CVPRW 2024) |
| `aigv_static` | — | — | `aigv_assessor` | vid | ⏱️ medium | ✓ | heuristic → aigv_assessor → clip_heuristic | [HF](https://huggingface.co/wangjiarui153/AIGV-Assessor) | — | AI-generated video quality (AIGV-Assessor model, CLIP+heuristic, or OpenCV fallback) |
| `aigvqa_score` | ↑ higher=better | higher=better | `aigvqa` | img/vid | ⚡ fast |  | heuristic → native | — | — | AIGVQA multi-dimensional AIGC VQA (ICCVW 2025) |
| `arniqa_score` | ↑ higher=better | higher=better | `arniqa` | img/vid | ⏱️ medium | ✓ | — | — | — | ARNIQA no-reference image quality assessment |
| `brisque` | ↓ lower=better | 0-100, lower=better | `brisque` | img/vid | ⏱️ medium |  | — | — | — | BRISQUE no-reference image quality (lower=better) |
| `bvqi_score` | ↑ higher=better | higher=better | `bvqi` | img/vid | ⏱️ medium |  | heuristic → native → pyiqa | — | — | BVQI zero-shot blind video quality index (ICME 2023) |
| `clifvqa_score` | ↑ higher=better | higher=better | `clifvqa` | img/vid | ⚡ fast |  | heuristic → native | — | — | CLiF-VQA human feelings VQA via CLIP (2024) |
| `clip_iqa_score` | ↑ higher=better | 0-1, higher=better | `clip_iqa` | img/vid | ⏱️ medium |  | — | — | — | CLIP-based no-reference image quality assessment |
| `clipvqa_score` | ↑ higher=better | higher=better | `clipvqa` | img/vid | ⏱️ medium | ✓ | heuristic → native → clip | [HF](https://huggingface.co/openai/clip-vit-base-patch32) | — | CLIPVQA CLIP-based spatiotemporal VQA (TIP 2024) |
| `cnniqa_score` | ↑ higher=better | — | `cnniqa` | img/vid | ⏱️ medium | ✓ | — | — | — | CNNIQA blind CNN-based image quality assessment |
| `compare2score` | ↑ higher=better | — | `compare2score` | img/vid | ⏱️ medium | ✓ | — | — | — | Compare2Score comparison-based NR image quality |
| `contrique_score` | ↑ higher=better | higher=better | `contrique` | img/vid | ⏱️ medium |  | — | — | — | Contrastive no-reference IQA |
| `conviqt_score` | ↑ higher=better | higher=better | `conviqt` | img/vid | ⏱️ medium |  | heuristic → native → pyiqa | — | — | CONVIQT contrastive self-supervised NR-VQA (TIP 2023) |
| `cover_score` | ↑ higher=better | higher=better | `cover` | img/vid | ⏱️ medium | ✓ | cover → dover | — | — | COVER 3-branch comprehensive video quality (semantic + aesthetic + technical) |
| `cover_technical` | — | — | `cover` | img/vid | ⏱️ medium | ✓ | cover → dover | — | — | COVER 3-branch comprehensive video quality (semantic + aesthetic + technical) |
| `crave_score` | ↑ higher=better | higher=better | `crave` | vid | ⚡ fast |  | heuristic → native | — | — | CRAVE content-rich AIGC video evaluator (2025) |
| `dbcnn_score` | ↑ higher=better | higher=better | `dbcnn` | img/vid | ⏱️ medium | ✓ | — | — | — | DBCNN deep bilinear CNN for no-reference IQA |
| `deepdc_score` | ↓ lower=better | lower=better | `deepdc` | img/vid | ⏱️ medium |  | heuristic → pyiqa | — | — | DeepDC distribution conformance NR-IQA via pyiqa (2024, lower=better) |
| `discovqa_score` | ↑ higher=better | higher=better | `discovqa` | img/vid | ⚡ fast |  | heuristic → native | — | — | DisCoVQA temporal distortion-content VQA (2023) |
| `dover_score` | ↑ higher=better | higher=better | `dover` | vid | ⏱️ medium | ✓ | heuristic → native → onnx → pyiqa | [GitHub](https://github.com/VQAssessment/DOVER.git) · [HF](https://huggingface.co/dover/DOVER.pth) | — | DOVER disentangled technical + aesthetic VQA (ICCV 2023) |
| `dover_score` | ↑ higher=better | higher=better | `unified_vqa` | img/vid +ref | ⚡ fast |  | heuristic → native | — | — | Unified-VQA FR+NR multi-task quality assessment (2025) |
| `dover_technical` | — | — | `dover` | vid | ⏱️ medium | ✓ | heuristic → native → onnx → pyiqa | [GitHub](https://github.com/VQAssessment/DOVER.git) · [HF](https://huggingface.co/dover/DOVER.pth) | — | DOVER disentangled technical + aesthetic VQA (ICCV 2023) |
| `fast_vqa_score` | ↑ higher=better | — | `fast_vqa` | vid | ⏱️ medium | ✓ | — | — | — | Deep Learning Video Quality Assessment (FAST-VQA) |
| `faver_score` | ↑ higher=better | higher=better | `faver` | vid | ⚡ fast |  | heuristic → native | — | — | FAVER blind VQA for variable frame rate videos (2024) |
| `finevq_score` | ↑ higher=better | — | `finevq` | img/vid | ⏱️ medium | ✓ | heuristic → finevq → topiq_handcrafted | [HF](https://huggingface.co/IntMeGroup/FineVQ_score) | — | Fine-grained video quality (FineVQ model, TOPIQ+handcrafted, or heuristic fallback) |
| `gamival_score` | ↑ higher=better | higher=better | `gamival` | img/vid | ⚡ fast |  | heuristic → native | — | — | GAMIVAL cloud gaming NR-VQA with NSS + CNN features (2023) |
| `hyperiqa_score` | ↑ higher=better | — | `hyperiqa` | img/vid | ⏱️ medium | ✓ | — | — | — | HyperIQA adaptive hypernetwork NR image quality |
| `ilniqe` | ↓ lower=better | lower=better | `ilniqe` | img/vid | ⏱️ medium |  | — | — | — | IL-NIQE integrated local no-reference quality (lower=better) |
| `internvqa_score` | ↑ higher=better | higher=better | `internvqa` | vid | ⚡ fast |  | heuristic → native | — | — | InternVQA lightweight compressed video quality (2025) |
| `kvq_score` | ↑ higher=better | — | `kvq` | img/vid | ⏱️ medium | ✓ | heuristic → kvq → topiq_saliency | [HF](https://huggingface.co/qyp2000/KVQ) | — | Saliency-guided video quality (KVQ model, TOPIQ+saliency, or heuristic fallback) |
| `liqe_score` | ↑ higher=better | higher=better | `liqe` | img/vid | ⏱️ medium |  | — | — | — | LIQE lightweight no-reference IQA |
| `lmmvqa_score` | ↑ higher=better | higher=better | `lmmvqa` | img/vid | ⚡ fast |  | heuristic → native | — | — | LMM-VQA spatiotemporal LMM VQA (IEEE 2024) |
| `maclip_score` | ↑ higher=better | higher=better | `maclip` | img/vid | ⏱️ medium |  | — | — | — | MACLIP multi-attribute CLIP no-reference quality (higher=better) |
| `maniqa_score` | ↑ higher=better | higher=better | `maniqa` | img/vid | ⏱️ medium | ✓ | — | — | — | MANIQA multi-dimension attention no-reference IQA |
| `maxvqa_score` | ↑ higher=better | higher=better | `maxvqa` | img/vid | ⏱️ medium | ✓ | heuristic → native → clip | [HF](https://huggingface.co/openai/clip-vit-base-patch32) | — | MaxVQA explainable language-prompted VQA (ACM MM 2023) |
| `mc360iqa_score` | ↑ higher=better | higher=better | `mc360iqa` | img/vid | ⚡ fast |  | heuristic → native | — | — | MC360IQA blind 360 IQA (2019) |
| `mdtvsfa_score` | ↑ higher=better | higher=better | `mdtvsfa` | img/vid | ⏱️ medium |  | — | — | — | Multi-Dimensional fragment-based VQA |
| `mdvqa_distortion` | ↑ higher=better | higher=better | `mdvqa` | img/vid | ⚡ fast |  | heuristic → native | — | — | MD-VQA multi-dimensional UGC live VQA (CVPR 2023) |
| `mdvqa_motion` | ↑ higher=better | higher=better | `mdvqa` | img/vid | ⚡ fast |  | heuristic → native | — | — | MD-VQA multi-dimensional UGC live VQA (CVPR 2023) |
| `mdvqa_semantic` | ↑ higher=better | higher=better | `mdvqa` | img/vid | ⚡ fast |  | heuristic → native | — | — | MD-VQA multi-dimensional UGC live VQA (CVPR 2023) |
| `memoryvqa_score` | ↑ higher=better | higher=better | `memoryvqa` | img/vid | ⚡ fast |  | heuristic → native | — | — | Memory-VQA human memory system VQA (Neurocomputing 2025) |
| `mm_pcqa_score` | ↑ higher=better | higher=better | `mm_pcqa` | img/vid | ⚡ fast |  | heuristic → native | — | — | MM-PCQA multi-modal point cloud QA (IJCAI 2023) |
| `modularbvqa_score` | ↑ higher=better | higher=better | `modularbvqa` | img/vid | ⚡ fast |  | heuristic → native | — | — | ModularBVQA resolution/framerate-aware blind VQA (CVPR 2024) |
| `musiq_score` | ↑ higher=better | higher=better | `musiq` | img/vid | ⏱️ medium |  | — | — | — | Multi-Scale Image Quality Transformer (no-reference) |
| `naturalness_score` | ↑ higher=better | — | `naturalness` | img/vid | ⏱️ medium |  | — | — | — | Measures naturalness of content (natural vs synthetic) |
| `niqe` | ↓ lower=better | lower=better | `niqe` | img/vid | ⏱️ medium |  | — | — | — | Natural Image Quality Evaluator (no-reference) |
| `nr_gvqm_score` | ↑ higher=better | higher=better | `nr_gvqm` | img/vid | ⚡ fast |  | heuristic | — | — | NR-GVQM no-reference gaming video quality (ISM 2018, 9 features) |
| `nrqm` | ↑ higher=better | higher=better | `nrqm` | img/vid | ⏱️ medium |  | — | — | — | NRQM no-reference quality metric for super-resolution (higher=better) |
| `paq2piq_score` | ↑ higher=better | — | `paq2piq` | img/vid | ⏱️ medium | ✓ | — | — | — | PaQ-2-PiQ patch-to-picture NR quality (CVPR 2020) |
| `pi_score` | ↓ lower=better | PIRM challenge, lower=better | `pi` | img/vid | ⏱️ medium |  | — | — | — | Perceptual Index (PIRM challenge metric, lower=better) |
| `piqe` | ↓ lower=better | lower=better | `piqe` | img/vid | ⏱️ medium |  | — | — | — | PIQE perception-based no-reference quality (lower=better) |
| `presresq_score` | ↑ higher=better | higher=better | `presresq` | img/vid | ⚡ fast |  | heuristic → native | — | — | PreResQ-R1 rank+score VQA (2025) |
| `promptiqa_score` | ↑ higher=better | — | `promptiqa` | img/vid | ⏱️ medium |  | none → promptiqa → topiq_nr | — | — | Prompt-guided NR-IQA (PromptIQA via pyiqa, TOPIQ-NR, or CLIP-IQA+ fallback) |
| `provqa_score` | ↑ higher=better | higher=better | `provqa` | img/vid | ⚡ fast |  | heuristic → native | — | — | ProVQA progressive blind 360 VQA (2022) |
| `ptmvqa_score` | ↑ higher=better | higher=better | `ptmvqa` | img/vid | ⚡ fast |  | heuristic → native | — | — | PTM-VQA multi-PTM fusion VQA (CVPR 2024) |
| `qalign_quality` | ↑ higher=better | 1-5, higher=better | `q_align` | img/vid | 🐌 slow | ✓ | — | [HF](https://huggingface.co/q-future/one-align) | — | Q-Align unified quality + aesthetic assessment (ICML 2024) |
| `qclip_score` | ↑ higher=better | higher=better | `qclip` | img/vid | ⚡ fast |  | heuristic → native | — | — | Q-CLIP VLM-based VQA (2025) |
| `qcn_score` | ↑ higher=better | — | `qcn` | img/vid | ⏱️ medium |  | none → qcn → hyperiqa | — | — | Blind IQA (QCN via pyiqa, or HyperIQA fallback) |
| `qualiclip_score` | ↑ higher=better | higher=better | `qualiclip` | img/vid | ⏱️ medium | ✓ | — | — | — | QualiCLIP opinion-unaware CLIP-based no-reference IQA |
| `rapique_score` | ↑ higher=better | higher=better | `rapique` | img/vid | ⚡ fast |  | heuristic → native | — | — | RAPIQUE rapid NR-VQA via bandpass NSS + CNN features (IEEE OJSP 2021) |
| `rqvqa_score` | ↑ higher=better | — | `rqvqa` | img/vid | ⏱️ medium | ✓ | heuristic → rqvqa → clipiqa | [HF](https://huggingface.co/sunwei925/RQ-VQA) | — | Multi-attribute video quality (RQ-VQA model, CLIP-IQA+, or heuristic fallback) |
| `sama_score` | ↑ higher=better | higher=better | `sama` | img/vid | ⚡ fast |  | heuristic → native | — | — | SAMA scaling+masking VQA (2024) |
| `siamvqa_score` | ↑ higher=better | higher=better | `siamvqa` | img/vid | ⚡ fast |  | heuristic → native | — | — | SiamVQA Siamese high-resolution VQA (2025) |
| `simplevqa_score` | ↑ higher=better | higher=better | `simplevqa` | img/vid | ⚡ fast |  | heuristic → native | — | — | SimpleVQA Swin+SlowFast blind VQA (2022) |
| `spectral_entropy` | — | — | `spectral_complexity` | vid | ⏱️ medium | ✓ | — | [HF](https://huggingface.co/facebookresearch/dinov2) | — | Analyzes spectral complexity (Effective Rank) of video features (DINOv2) |
| `spectral_rank` | — | — | `spectral_complexity` | vid | ⏱️ medium | ✓ | — | [HF](https://huggingface.co/facebookresearch/dinov2) | — | Analyzes spectral complexity (Effective Rank) of video features (DINOv2) |
| `speedqa_score` | ↑ higher=better | higher=better | `speedqa` | vid | ⚡ fast |  | heuristic → native | — | — | SpEED-QA spatial efficient entropic differencing NR-VQA (Bampis 2017) |
| `sqi_score` | ↑ higher=better | — | `sqi` | vid | ⚡ fast |  | — | — | — | SQI streaming quality index (2016) |
| `sr4kvqa_score` | ↑ higher=better | higher=better | `sr4kvqa` | img/vid | ⚡ fast |  | heuristic → native | — | — | SR4KVQA super-resolution 4K quality (2024) |
| `stablevqa_score` | ↑ higher=better | higher=better | `stablevqa` | vid | ⚡ fast |  | heuristic → native | — | — | StableVQA video stability quality assessment (ACM MM 2023) |
| `t2v_quality` | ↑ higher=better | — | `t2v_score` | vid | ⏱️ medium | ✓ | — | [HF](https://huggingface.co/TIGER-Lab/T2VScore) | — | Text-to-Video alignment and quality scoring |
| `thqa_score` | ↑ higher=better | higher=better | `thqa` | vid | ⚡ fast |  | thqa | — | — | THQA talking head quality assessment (ICIP 2024) |
| `tlvqm_score` | ↑ higher=better | — | `tlvqm` | img/vid | ⏱️ medium | ✓ | handcrafted → cnn → cnn_svr → cnn_pretrained | [GitHub](https://github.com/jarikorhonen/cnn-tlvqm) | — | Two-level video quality model (CNN-TLVQM or handcrafted fallback) |
| `topiq_score` | ↑ higher=better | higher=better | `topiq` | img/vid | ⏱️ medium | ✓ | — | — | — | TOPIQ transformer-based no-reference IQA |
| `tres_score` | ↑ higher=better | — | `tres` | img/vid | ⏱️ medium | ✓ | — | — | — | TReS transformer-based NR image quality (WACV 2022) |
| `uciqe_score` | ↑ higher=better | higher=better | `uciqe` | img/vid | ⚡ fast |  | — | — | — | UCIQE underwater color image quality evaluation (2015) |
| `ugvq_score` | ↑ higher=better | higher=better | `ugvq` | img/vid | ⚡ fast |  | heuristic → native | — | — | UGVQ unified generated video quality (TOMM 2024) |
| `uiqm_score` | ↑ higher=better | higher=better | `uiqm` | img/vid | ⚡ fast |  | — | — | — | UIQM underwater image quality measure (Panetta et al. 2016) |
| `unique_score` | ↑ higher=better | — | `unique` | img/vid | ⏱️ medium | ✓ | — | — | — | UNIQUE unified NR image quality (TIP 2021) |
| `vader_score` | ↑ higher=better | — | `vader` | img/vid | ⚡ fast |  | heuristic → native | — | — | VADER reward gradient alignment (ICLR 2025) |
| `vbliinds_score` | ↑ higher=better | higher=better | `vbliinds` | img/vid | ⚡ fast |  | heuristic → native | — | — | V-BLIINDS blind NR-VQA via DCT-domain NSS (Saad 2013) |
| `video_atlas_score` | ↑ higher=better | — | `video_atlas` | vid | ⚡ fast |  | heuristic → native | — | — | Video ATLAS temporal artifacts+stalls assessment (2018) |
| `video_memorability` | — | — | `video_memorability` | img/vid | ⏱️ medium | ✓ | heuristic → clip → dinov2 | [HF](https://huggingface.co/openai/clip-vit-base-patch32) | — | Content memorability approximation (CLIP/DINOv2 feature statistics, not a trained predictor) |
| `videoreward_vq` | — | — | `videoreward` | vid +cap | ⚡ fast |  | heuristic → native | — | — | VideoReward Kling multi-dim reward model (NeurIPS 2025) |
| `videoscore_visual` | ↑ higher=better | — | `videoscore` | img/vid | 🐌 slow | ✓ | — | [HF](https://huggingface.co/TIGER-Lab/VideoScore) | — | VideoScore 5-dimensional video quality assessment (1-4 scale) |
| `videval_score` | ↑ higher=better | — | `videval` | img/vid | ⚡ fast |  | heuristic → svr | [GitHub](https://github.com/vztu/VIDEVAL) | — | Feature-fusion NR-VQA (VIDEVAL-style SVR or heuristic linear mapping) |
| `viideo_score` | ↓ lower=better | lower=better | `viideo` | vid | ⚡ fast |  | heuristic → native | — | — | VIIDEO blind NR-VQA via natural video statistics (Mittal 2016, lower=better) |
| `vqa2_score` | ↑ higher=better | higher=better | `vqa2` | img/vid | ⚡ fast |  | heuristic → native | — | — | VQA² LMM video quality assessment (MM 2025) |
| `vqathinker_score` | ↑ higher=better | higher=better | `vqathinker` | img/vid | ⚡ fast |  | heuristic → native | — | — | VQAThinker RL-based explainable VQA (2025) |
| `vqinsight_score` | ↑ higher=better | higher=better | `vqinsight` | img/vid | ⚡ fast |  | heuristic → native | — | — | VQ-Insight ByteDance multi-dim AIGC scoring (AAAI 2026) |
| `vsfa_score` | ↑ higher=better | higher=better | `vsfa` | img/vid | ⚡ fast |  | heuristic → native | — | — | VSFA quality-aware feature aggregation with GRU (ACMMM 2019) |
| `wadiqam_score` | ↑ higher=better | higher=better | `wadiqam` | img/vid | ⏱️ medium | ✓ | — | — | — | WaDIQaM-NR weighted averaging deep image quality mapper |
| `zoomvqa_score` | ↑ higher=better | higher=better | `zoomvqa` | img/vid | ⚡ fast |  | heuristic → native | — | — | Zoom-VQA multi-level patch/frame/clip VQA (CVPRW 2023) |

## Full-Reference Quality (57 metrics)

| Metric | Dir | Range | Module | Input | Speed | GPU | Backend | Source | Test | Description |
|--------|-----|-------|--------|-------|-------|-----|---------|--------|------|-------------|
| `ahiq` | ↑ higher=better | higher=better | `ahiq` | img/vid +ref | ⏱️ medium | ✓ | — | — | — | Attention-based Hybrid IQA full-reference (higher=better) |
| `artfid_score` | ↓ lower=better | lower=better | `artfid` | img/vid +ref | ⚡ fast |  | — | — | — | ArtFID style transfer quality (FR, 2022, lower=better) |
| `avqt_score` | ↑ higher=better | higher=better | `avqt` | img/vid +ref | ⚡ fast |  | heuristic → cli | — | — | Apple AVQT perceptual video quality (full-reference) |
| `butteraugli` | ↓ lower=better | lower=better | `butteraugli` | img/vid +ref | ⚡ fast |  | jxlpy → butteraugli → approx | — | — | Butteraugli perceptual distance (Google/JPEG XL, lower=better) |
| `c3dvqa_score` | ↑ higher=better | — | `c3dvqa` | vid | ⏱️ medium | ✓ | — | — | — | 3D CNN spatiotemporal video quality assessment |
| `cgvqm` | ↑ higher=better | higher=better | `cgvqm` | img/vid +ref | ⚡ fast |  | cgvqm → approx | — | — | CGVQM gaming/rendering quality metric (Intel, higher=better) |
| `ciede2000` | ↓ lower=better | lower=better | `ciede2000` | img/vid +ref | ⚡ fast |  | — | — | — | CIEDE2000 perceptual color difference (lower=better) |
| `ckdn_score` | ↑ higher=better | — | `ckdn` | img/vid +ref | ⏱️ medium | ✓ | — | — | — | CKDN knowledge distillation FR image quality |
| `compressed_vqa_hdr` | ↑ higher=better | higher=better | `compressed_vqa_hdr` | img/vid +ref | ⚡ fast |  | — | — | — | CompressedVQA-HDR FR quality (ICME 2025) |
| `cpp_psnr` | ↑ higher=better | dB, higher=better | `spherical_psnr` | img/vid +ref | ⚡ fast |  | — | — | — | S-PSNR/WS-PSNR/CPP-PSNR spherical PSNR (MPEG/JVET) |
| `cw_ssim` | ↑ higher=better | 0-1, higher=better | `cw_ssim` | img/vid +ref | ⏱️ medium | ✓ | — | — | — | Complex Wavelet SSIM full-reference metric (0-1, higher=better) |
| `deepvqa_score` | ↑ higher=better | higher=better | `deepvqa` | img/vid +ref | ⚡ fast |  | heuristic → native | — | — | DeepVQA spatiotemporal masking FR-VQA (ECCV 2018) |
| `deepwsd_score` | ↓ lower=better | — | `deepwsd` | img/vid +ref | ⏱️ medium | ✓ | — | — | — | DeepWSD Wasserstein distance FR image quality |
| `dists` | ↓ lower=better | 0-1, lower=more similar | `dists` | img/vid +ref | ⏱️ medium | ✓ | — | — | — | Deep Image Structure and Texture Similarity (full-reference) |
| `dmm` | ↑ higher=better | higher=better | `dmm` | img/vid +ref | ⏱️ medium | ✓ | — | — | — | DMM detail model metric full-reference (higher=better) |
| `dreamsim` | ↓ lower=better | lower=more similar | `dreamsim` | img/vid +ref | ⏱️ medium |  | — | — | — | DreamSim foundation model perceptual similarity (CLIP+DINO ensemble) |
| `erqa_score` | ↑ higher=better | 0-1, higher=better | `erqa` | img/vid +ref | ⚡ fast |  | — | — | — | ERQA edge restoration quality assessment (FR, 2022) |
| `flip_score` | ↓ lower=better | 0-1, lower=better | `flip` | img/vid +ref | ⏱️ medium |  | flip_evaluator → flip_torch → approx | — | — | NVIDIA FLIP perceptual difference (0-1, lower=better) |
| `flolpips` | — | — | `flolpips` | vid | ⏱️ medium | ✓ | farneback_mse → raft_lpips → farneback_lpips | — | — | Flow-compensated perceptual distance (RAFT+LPIPS, Farneback+LPIPS, or MSE fallback) |
| `fsim` | ↑ higher=better | 0-1, higher=better | `perceptual_fr` | img/vid +ref | ⏱️ medium | ✓ | — | — | — | FSIM + GMSD + VSI full-reference perceptual metrics |
| `funque_score` | ↑ higher=better | — | `funque` | img/vid +ref | ⚡ fast |  | heuristic_nr → funque → heuristic_fr | — | — | Fused quality evaluator (FUNQUE package, handcrafted FR, or NR fallback) |
| `gmsd` | ↓ lower=better | lower=better | `perceptual_fr` | img/vid +ref | ⏱️ medium | ✓ | — | — | — | FSIM + GMSD + VSI full-reference perceptual metrics |
| `graphsim_score` | ↑ higher=better | higher=better | `graphsim` | img/vid +ref | ⚡ fast |  | — | — | — | GraphSIM graph gradient point cloud quality (2020) |
| `mad` | ↓ lower=better | lower=better | `mad` | img/vid +ref | ⏱️ medium | ✓ | — | — | — | Most Apparent Distortion full-reference metric (lower=better) |
| `movie_score` | ↑ higher=better | — | `movie` | img/vid +ref | ⚡ fast |  | — | — | — | Video quality via spatiotemporal Gabor decomposition (FR or NR fallback) |
| `ms_ssim` | — | 0-1 | `ms_ssim` | vid +ref | ⏱️ medium | ✓ | — | — | — | Multi-Scale SSIM perceptual similarity metric (full-reference) |
| `nlpd` | ↓ lower=better | lower=better | `nlpd` | img/vid +ref | ⏱️ medium | ✓ | — | — | — | Normalized Laplacian Pyramid Distance full-reference (lower=better) |
| `pc_d1_psnr` | — | dB | `pc_psnr` | img/vid +ref | ⚡ fast |  | — | — | — | D1/D2 MPEG point cloud PSNR |
| `pc_d2_psnr` | — | dB | `pc_psnr` | img/vid +ref | ⚡ fast |  | — | — | — | D1/D2 MPEG point cloud PSNR |
| `pcqm_score` | ↑ higher=better | higher=better | `pcqm` | img/vid +ref | ⚡ fast |  | — | — | — | PCQM geometry+color point cloud quality (2020) |
| `pieapp` | ↓ lower=better | lower=better | `pieapp` | img/vid +ref | ⏱️ medium | ✓ | — | — | — | PieAPP full-reference perceptual error via pairwise preference (lower=better) |
| `pointssim_score` | ↑ higher=better | higher=better | `pointssim` | img/vid +ref | ⚡ fast |  | — | — | — | PointSSIM structural similarity for point clouds (2020) |
| `psnr99` | ↑ higher=better | dB, higher=better | `psnr99` | img/vid +ref | ⚡ fast |  | — | — | — | PSNR99 worst-case region quality for super-resolution (FR, 2025) |
| `psnr_div` | ↑ higher=better | dB, higher=better | `psnr_div` | img/vid +ref | ⚡ fast |  | — | — | — | PSNR_DIV motion-weighted PSNR for frame interpolation (ICIP 2025, FR) |
| `psnr_hvs` | ↑ higher=better | dB, higher=better | `psnr_hvs` | img/vid +ref | ⚡ fast |  | dct | — | — | PSNR-HVS + PSNR-HVS-M perceptually weighted PSNR (dB, higher=better) |
| `psnr_hvs_m` | ↑ higher=better | dB, higher=better | `psnr_hvs` | img/vid +ref | ⚡ fast |  | dct | — | — | PSNR-HVS + PSNR-HVS-M perceptually weighted PSNR (dB, higher=better) |
| `pvmaf_score` | ↑ higher=better | 0-100 | `pvmaf` | img/vid +ref | ⚡ fast |  | heuristic → native | — | — | Predictive VMAF ~35x faster via bitstream+pixel features (2024, 0-100) |
| `rankdvqa_score` | ↑ higher=better | higher=better | `rankdvqa` | img/vid +ref | ⚡ fast |  | — | — | — | RankDVQA ranking-based FR VQA (WACV 2024) |
| `s_psnr` | ↑ higher=better | dB, higher=better | `spherical_psnr` | img/vid +ref | ⚡ fast |  | — | — | — | S-PSNR/WS-PSNR/CPP-PSNR spherical PSNR (MPEG/JVET) |
| `ssimc` | ↑ higher=better | higher=better | `ssimc` | img/vid +ref | ⏱️ medium | ✓ | — | — | — | SSIM-C complex wavelet structural similarity FR (higher=better) |
| `ssimulacra2` | ↓ lower=better | 0-100, lower=better, JPEG XL standard | `ssimulacra2` | img/vid +ref | ⚡ fast |  | — | — | — | SSIMULACRA 2 perceptual distance (JPEG XL standard, lower=better) |
| `st_greed_score` | ↑ higher=better | — | `st_greed` | vid +ref | ⚡ fast |  | — | — | — | Spatial-temporal entropic quality (FR entropic difference or NR heuristic fallback) |
| `st_lpips` | — | — | `st_lpips` | vid | ⏱️ medium | ✓ | heuristic → stlpips → lpips | — | — | Spatiotemporal perceptual video quality (ST-LPIPS model, LPIPS, or heuristic fallback) |
| `st_mad` | ↓ lower=better | lower=better | `st_mad` | img/vid +ref | ⚡ fast |  | — | — | — | ST-MAD spatiotemporal MAD (TIP 2012) |
| `strred` | ↓ lower=better | lower=better | `strred` | img/vid +ref | ⚡ fast |  | skvideo → approx | — | — | STRRED reduced-reference temporal quality (ITU, lower=better) |
| `topiq_fr` | ↑ higher=better | higher=better | `topiq_fr` | img/vid +ref | ⏱️ medium | ✓ | — | — | — | TOPIQ full-reference top-down semantics-to-distortion IQA (higher=better) |
| `vfips_score` | ↓ lower=better | lower=better | `vfips` | img/vid +ref | ⚡ fast |  | — | — | — | VFIPS frame interpolation perceptual similarity (ECCV 2022, FR) |
| `vif` | — | — | `vif` | img/vid +ref | ⏱️ medium | ✓ | — | — | — | Visual Information Fidelity metric (full-reference) |
| `vmaf` | ↑ higher=better | 0-100, higher=better | `vmaf` | vid +ref | ⚡ fast |  | — | — | — | VMAF perceptual video quality metric (full-reference) |
| `vmaf_4k` | ↑ higher=better | 0-100, higher=better | `vmaf_4k` | vid +ref | ⚡ fast |  | — | — | — | VMAF 4K model for UHD content (0-100, higher=better) |
| `vmaf_neg` | ↑ higher=better | no enhancement gain, 0-100, higher=better | `vmaf_neg` | vid +ref | ⚡ fast |  | — | — | — | VMAF NEG no-enhancement-gain variant (0-100, higher=better) |
| `vmaf_phone` | ↑ higher=better | 0-100, higher=better | `vmaf_phone` | vid +ref | ⚡ fast |  | — | — | — | VMAF phone model for mobile viewing (0-100, higher=better) |
| `vsi_score` | ↑ higher=better | 0-1, higher=better | `perceptual_fr` | img/vid +ref | ⏱️ medium | ✓ | — | — | — | FSIM + GMSD + VSI full-reference perceptual metrics |
| `wadiqam_fr` | ↑ higher=better | higher=better | `wadiqam_fr` | img/vid +ref | ⏱️ medium | ✓ | — | — | — | WaDIQaM full-reference deep quality metric (higher=better) |
| `ws_psnr` | ↑ higher=better | dB, higher=better | `spherical_psnr` | img/vid +ref | ⚡ fast |  | — | — | — | S-PSNR/WS-PSNR/CPP-PSNR spherical PSNR (MPEG/JVET) |
| `ws_ssim` | ↑ higher=better | 0-1, higher=better | `ws_ssim` | img/vid +ref | ⚡ fast |  | — | — | — | WS-SSIM weighted spherical SSIM |
| `xpsnr` | ↑ higher=better | dB, higher=better | `xpsnr` | img/vid +ref | ⚡ fast |  | — | — | — | XPSNR perceptually weighted PSNR (Fraunhofer, dB, higher=better) |

## Text-Video Alignment (26 metrics)

| Metric | Dir | Range | Module | Input | Speed | GPU | Backend | Source | Test | Description |
|--------|-----|-------|--------|-------|-------|-----|---------|--------|------|-------------|
| `aigcvqa_alignment` | — | — | `aigcvqa` | img/vid +cap | ⚡ fast |  | heuristic → native | — | — | AIGC-VQA holistic 3-branch AIGC perception (CVPRW 2024) |
| `aigv_alignment` | — | — | `aigv_assessor` | vid | ⏱️ medium | ✓ | heuristic → aigv_assessor → clip_heuristic | [HF](https://huggingface.co/wangjiarui153/AIGV-Assessor) | — | AI-generated video quality (AIGV-Assessor model, CLIP+heuristic, or OpenCV fallback) |
| `blip_bleu` | — | — | `captioning` | img/vid | ⏱️ medium | ✓ | — | [HF](https://huggingface.co/Salesforce/blip-image-captioning-base) | — | Generates captions using BLIP + computes BLEU score (EvalCrafter blip_bleu) |
| `clip_score` | ↑ higher=better | — | `semantic_alignment` | vid +cap | ⏱️ medium | ✓ | — | [HF](https://huggingface.co/openai/clip-vit-base-patch32) | — | Checks alignment between video and caption (CLIP Score) |
| `compbench_action` | — | 0-1 | `t2v_compbench` | vid | ⏱️ medium | ✓ | heuristic → yolo_depth → clip | [HF](https://huggingface.co/openai/clip-vit-base-patch32) | — | T2V-CompBench compositional metrics (YOLO+Depth+CLIP / CLIP / heuristic) |
| `compbench_attribute` | — | 0-1 | `t2v_compbench` | vid | ⏱️ medium | ✓ | heuristic → yolo_depth → clip | [HF](https://huggingface.co/openai/clip-vit-base-patch32) | — | T2V-CompBench compositional metrics (YOLO+Depth+CLIP / CLIP / heuristic) |
| `compbench_numeracy` | — | 0-1 | `t2v_compbench` | vid | ⏱️ medium | ✓ | heuristic → yolo_depth → clip | [HF](https://huggingface.co/openai/clip-vit-base-patch32) | — | T2V-CompBench compositional metrics (YOLO+Depth+CLIP / CLIP / heuristic) |
| `compbench_object_rel` | — | 0-1 | `t2v_compbench` | vid | ⏱️ medium | ✓ | heuristic → yolo_depth → clip | [HF](https://huggingface.co/openai/clip-vit-base-patch32) | — | T2V-CompBench compositional metrics (YOLO+Depth+CLIP / CLIP / heuristic) |
| `compbench_overall` | — | 0-1 | `t2v_compbench` | vid | ⏱️ medium | ✓ | heuristic → yolo_depth → clip | [HF](https://huggingface.co/openai/clip-vit-base-patch32) | — | T2V-CompBench compositional metrics (YOLO+Depth+CLIP / CLIP / heuristic) |
| `compbench_scene` | — | 0-1 | `t2v_compbench` | vid | ⏱️ medium | ✓ | heuristic → yolo_depth → clip | [HF](https://huggingface.co/openai/clip-vit-base-patch32) | — | T2V-CompBench compositional metrics (YOLO+Depth+CLIP / CLIP / heuristic) |
| `compbench_spatial` | — | 0-1 | `t2v_compbench` | vid | ⏱️ medium | ✓ | heuristic → yolo_depth → clip | [HF](https://huggingface.co/openai/clip-vit-base-patch32) | — | T2V-CompBench compositional metrics (YOLO+Depth+CLIP / CLIP / heuristic) |
| `dsg_score` | ↑ higher=better | higher=better | `dsg` | img/vid +cap | ⚡ fast |  | heuristic → native | — | — | DSG Davidsonian Scene Graph faithfulness (ICLR 2024, Google) |
| `sd_score` | ↑ higher=better | 0-1 | `sd_reference` | img/vid | ⏱️ medium | ✓ | — | [HF](https://huggingface.co/openai/clip-vit-base-patch32) | — | SD Score — CLIP similarity between video frames and SDXL-generated reference images |
| `t2v_alignment` | — | — | `t2v_score` | vid | ⏱️ medium | ✓ | — | [HF](https://huggingface.co/TIGER-Lab/T2VScore) | — | Text-to-Video alignment and quality scoring |
| `t2v_score` | ↑ higher=better | — | `t2v_score` | vid | ⏱️ medium | ✓ | — | [HF](https://huggingface.co/TIGER-Lab/T2VScore) | — | Text-to-Video alignment and quality scoring |
| `t2veval_score` | ↑ higher=better | higher=better | `t2veval` | img/vid | ⚡ fast |  | heuristic → native | — | — | T2VEval text-video consistency+realness (2025) |
| `tifa_score` | ↑ higher=better | 0-1, higher=better | `tifa` | img/vid +cap | ⏱️ medium | ✓ | vilt → clip → heuristic | [HF](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa) | — | TIFA text-to-image faithfulness via VQA question answering (ICCV 2023) |
| `umtscore` | ↑ higher=better | — | `umtscore` | img/vid | ⏱️ medium |  | heuristic → native → clip | [HF](https://huggingface.co/openai/clip-vit-base-patch32) | — | UMTScore video-text alignment via UMT features |
| `video_reward_score` | ↑ higher=better | — | `video_reward` | img/vid | ⏱️ medium | ✓ | — | [HF](https://huggingface.co/KlingTeam/VideoAlign-Reward) | — | VideoAlign human preference reward model (NeurIPS 2025) |
| `video_text_score` | ↑ higher=better | 0-1 | `video_text_matching` | img/vid | ⏱️ medium | ✓ | — | [HF](https://huggingface.co/openai/clip-vit-base-patch32) | — | ViCLIP / X-CLIP (Temporal alignment) or Frame-averaged CLIP |
| `videoreward_ta` | — | — | `videoreward` | vid +cap | ⚡ fast |  | heuristic → native | — | — | VideoReward Kling multi-dim reward model (NeurIPS 2025) |
| `videoscore_alignment` | ↑ higher=better | — | `videoscore` | img/vid | 🐌 slow | ✓ | — | [HF](https://huggingface.co/TIGER-Lab/VideoScore) | — | VideoScore 5-dimensional video quality assessment (1-4 scale) |
| `videoscore_factual` | ↑ higher=better | — | `videoscore` | img/vid | 🐌 slow | ✓ | — | [HF](https://huggingface.co/TIGER-Lab/VideoScore) | — | VideoScore 5-dimensional video quality assessment (1-4 scale) |
| `vqa_a_score` | ↑ higher=better | — | `aesthetic` | img/vid | ⏱️ medium | ✓ | — | — | — | Estimates aesthetic quality using Aesthetic Predictor V2.5 |
| `vqa_score_alignment` | ↑ higher=better | — | `vqa_score` | img/vid +cap | ⏱️ medium | ✓ | — | [HF](https://huggingface.co/ViT-B/32) | — | VQAScore text-visual alignment via VQA probability (0-1, higher=better) |
| `vqa_t_score` | ↑ higher=better | — | `basic_quality` | img/vid | ⚡ fast |  | — | — | — | Comprehensive technical quality assessment (blur, noise, artifacts, contrast) |

## Temporal Consistency (24 metrics)

| Metric | Dir | Range | Module | Input | Speed | GPU | Backend | Source | Test | Description |
|--------|-----|-------|--------|-------|-------|-----|---------|--------|------|-------------|
| `aigv_temporal` | — | — | `aigv_assessor` | vid | ⏱️ medium | ✓ | heuristic → aigv_assessor → clip_heuristic | [HF](https://huggingface.co/wangjiarui153/AIGV-Assessor) | — | AI-generated video quality (AIGV-Assessor model, CLIP+heuristic, or OpenCV fallback) |
| `background_consistency` | ↑ higher=better | — | `background_consistency` | vid | ⏱️ medium | ✓ | — | [HF](https://huggingface.co/openai/clip-vit-base-patch32) | — | Background consistency using CLIP (all pairwise frame similarity) |
| `cdc_score` | ↓ lower=better | lower=better | `cdc` | vid | ⚡ fast |  | — | — | — | CDC color distribution consistency for video colorization (2024) |
| `chronomagic_ch_score` | ↓ lower=better | 0-1, lower=fewer | `chronomagic` | vid | ⏱️ medium | ✓ | heuristic → clip | [HF](https://huggingface.co/openai/clip-vit-base-patch32) | — | ChronoMagic-Bench MTScore + CHScore (CLIP / heuristic) |
| `chronomagic_mt_score` | ↑ higher=better | 0-1, higher=better | `chronomagic` | vid | ⏱️ medium | ✓ | heuristic → clip | [HF](https://huggingface.co/openai/clip-vit-base-patch32) | — | ChronoMagic-Bench MTScore + CHScore (CLIP / heuristic) |
| `clip_temp` | — | — | `clip_temporal` | vid | ⏱️ medium | ✓ | — | [HF](https://huggingface.co/openai/clip-vit-base-patch32) | — | CLIP temporal consistency + face/identity consistency (EvalCrafter clip_temp & face_consistency) |
| `davis_f` | ↑ higher=better | higher=better | `davis_jf` | img/vid +ref | ⚡ fast |  | — | — | — | DAVIS J&F video segmentation quality (FR, 2016) |
| `davis_j` | ↑ higher=better | higher=better | `davis_jf` | img/vid +ref | ⚡ fast |  | — | — | — | DAVIS J&F video segmentation quality (FR, 2016) |
| `depth_temporal_consistency` | ↑ higher=better | higher=better | `depth_consistency` | vid | ⏱️ medium | ✓ | — | [HF](https://huggingface.co/intel-isl/MiDaS) | — | Monocular depth temporal consistency |
| `flicker_score` | ↓ lower=better | lower=better | `flicker_detection` | vid | ⚡ fast |  | — | — | — | Detects temporal luminance flicker |
| `flow_coherence` | — | 0-1 | `flow_coherence` | vid | ⚡ fast |  | — | — | — | Bidirectional optical flow consistency (0-1, higher=coherent) |
| `judder_score` | ↓ lower=better | lower=better | `judder_stutter` | vid | ⚡ fast |  | — | — | — | Detects judder (uneven cadence) and stutter (duplicate frames) |
| `jump_cut_score` | ↑ higher=better | 0-1, 1=no cuts | `jump_cut` | vid | ⚡ fast |  | — | — | — | Jump cut / abrupt transition detection (0-1, 1=no cuts) |
| `lse_c` | ↑ higher=better | higher=better | `lip_sync` | audio | ⚡ fast |  | syncnet | — | — | LSE-D/LSE-C lip sync error (SyncNet/Wav2Lip, 2020) |
| `lse_d` | ↓ lower=better | lower=better | `lip_sync` | audio | ⚡ fast |  | syncnet | — | — | LSE-D/LSE-C lip sync error (SyncNet/Wav2Lip, 2020) |
| `object_permanence_score` | ↑ higher=better | — | `object_permanence` | vid | ⚡ fast |  | — | — | — | Object tracking consistency (ID switches, disappearances) |
| `scene_stability` | — | — | `scene_detection` | vid | ⚡ fast |  | — | — | — | Scene stability metric — penalises rapid cuts (0-1, higher=more stable) |
| `semantic_consistency` | ↑ higher=better | higher=better | `semantic_segmentation_consistency` | vid | ⏱️ medium | ✓ | — | [HF](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512) | — | Temporal stability of semantic segmentation |
| `stutter_score` | ↓ lower=better | lower=better | `judder_stutter` | vid | ⚡ fast |  | — | — | — | Detects judder (uneven cadence) and stutter (duplicate frames) |
| `subject_consistency` | ↑ higher=better | 0-1, higher=better | `subject_consistency` | vid | ⏱️ medium | ✓ | — | [HF](https://huggingface.co/facebook/dinov2-base) | — | Subject consistency using DINOv2-base (all pairwise frame similarity) |
| `video_text_temporal` | — | 0-1 | `video_text_matching` | img/vid | ⏱️ medium | ✓ | — | [HF](https://huggingface.co/openai/clip-vit-base-patch32) | — | ViCLIP / X-CLIP (Temporal alignment) or Frame-averaged CLIP |
| `videoscore_temporal` | ↑ higher=better | — | `videoscore` | img/vid | 🐌 slow | ✓ | — | [HF](https://huggingface.co/TIGER-Lab/VideoScore) | — | VideoScore 5-dimensional video quality assessment (1-4 scale) |
| `warping_error` | ↓ lower=better | — | `temporal_flickering` | vid | ⏱️ medium | ✓ | — | — | — | Warping Error using RAFT optical flow with occlusion masking |
| `world_consistency_score` | ↑ higher=better | higher=better | `world_consistency` | vid | ⚡ fast |  | heuristic → native | — | — | World Consistency Score: object permanence + causal compliance (2025) |

## Motion & Dynamics (19 metrics)

| Metric | Dir | Range | Module | Input | Speed | GPU | Backend | Source | Test | Description |
|--------|-----|-------|--------|-------|-------|-----|---------|--------|------|-------------|
| `aigv_dynamic` | — | — | `aigv_assessor` | vid | ⏱️ medium | ✓ | heuristic → aigv_assessor → clip_heuristic | [HF](https://huggingface.co/wangjiarui153/AIGV-Assessor) | — | AI-generated video quality (AIGV-Assessor model, CLIP+heuristic, or OpenCV fallback) |
| `bas_score` | ↑ higher=better | higher=better | `beat_alignment` | audio | ⚡ fast |  | heuristic → librosa | — | — | BAS beat alignment score — audio-motion sync (EDGE/CVPR 2023) |
| `camera_jitter_score` | ↓ lower=better | 0-1, 1=stable | `camera_jitter` | vid | ⚡ fast |  | — | — | — | Camera jitter/shake detection (0-1, 1=stable) |
| `camera_motion_score` | ↑ higher=better | — | `camera_motion` | vid | ⚡ fast |  | — | — | — | Analyzes camera motion stability (VMBench) using Homography |
| `dynamics_controllability` | — | — | `dynamics_controllability` | vid | ⏱️ medium | ✓ | farneback → cotracker | [HF](https://huggingface.co/facebookresearch/co-tracker) | — | Assesses motion controllability based on text-motion alignment |
| `dynamics_range` | — | — | `dynamics_range` | vid | ⚡ fast |  | — | — | — | Measures extent of motion and content variation (DEVIL protocol) |
| `flow_score` | ↑ higher=better | — | `advanced_flow` | vid | ⏱️ medium | ✓ | — | — | — | RAFT optical flow: flow_score (all consecutive pairs) |
| `motion_ac_score` | ↑ higher=better | — | `motion_amplitude` | vid | ⏱️ medium | ✓ | — | — | — | Motion amplitude classification vs caption (motion_ac_score via RAFT) |
| `motion_score` | ↑ higher=better | — | `motion` | vid | ⚡ fast |  | — | — | — | Analyzes motion dynamics (optical flow, flickering) |
| `motion_smoothness` | ↑ higher=better | 0-1, higher=better | `motion_smoothness` | vid | ⏱️ medium | ✓ | — | [HF](https://huggingface.co/rife/flownet.pkl) | — | Motion smoothness via RIFE VFI reconstruction error (VBench) |
| `physics_score` | ↑ higher=better | 0-1, higher=better | `physics` | vid | ⏱️ medium | ✓ | heuristic → cotracker → lk | [HF](https://huggingface.co/facebookresearch/co-tracker) | — | Physics plausibility via trajectory analysis (CoTracker / LK / heuristic) |
| `playback_speed_score` | ↑ higher=better | — | `playback_speed` | vid | ⚡ fast |  | — | — | — | Playback speed normality detection (1.0=normal) |
| `ptlflow_motion_score` | ↑ higher=better | — | `ptlflow_motion` | vid | ⏱️ medium | ✓ | — | — | — | ptlflow optical flow motion scoring (dpflow model) |
| `raft_motion_score` | ↑ higher=better | — | `raft_motion` | vid | ⏱️ medium | ✓ | — | — | — | RAFT optical flow motion scoring (torchvision) |
| `stabilized_camera_score` | ↑ higher=better | — | `stabilized_motion` | vid | ⚡ fast |  | — | — | — | Calculates motion scores with camera stabilization (ORB+Homography) |
| `stabilized_motion_score` | ↑ higher=better | — | `stabilized_motion` | vid | ⚡ fast |  | — | — | — | Calculates motion scores with camera stabilization (ORB+Homography) |
| `trajan_score` | ↑ higher=better | — | `trajan` | vid | ⏱️ medium | ✓ | lk → cotracker | [HF](https://huggingface.co/facebookresearch/co-tracker) | — | Motion consistency via point tracking (CoTracker or Lucas-Kanade fallback) |
| `videoreward_mq` | — | — | `videoreward` | vid +cap | ⚡ fast |  | heuristic → native | — | — | VideoReward Kling multi-dim reward model (NeurIPS 2025) |
| `videoscore_dynamic` | ↑ higher=better | — | `videoscore` | img/vid | 🐌 slow | ✓ | — | [HF](https://huggingface.co/TIGER-Lab/VideoScore) | — | VideoScore 5-dimensional video quality assessment (1-4 scale) |

## Basic Visual Quality (15 metrics)

| Metric | Dir | Range | Module | Input | Speed | GPU | Backend | Source | Test | Description |
|--------|-----|-------|--------|-------|-------|-----|---------|--------|------|-------------|
| `artifacts_score` | ↑ higher=better | — | `basic_quality` | img/vid | ⚡ fast |  | — | — | — | Comprehensive technical quality assessment (blur, noise, artifacts, contrast) |
| `blur_score` | ↑ higher=better | — | `basic_quality` | img/vid | ⚡ fast |  | — | — | — | Comprehensive technical quality assessment (blur, noise, artifacts, contrast) |
| `brightness` | — | — | `basic_quality` | img/vid | ⚡ fast |  | — | — | — | Comprehensive technical quality assessment (blur, noise, artifacts, contrast) |
| `compression_artifacts` | — | 0-100 | `compression_artifacts` | vid | ⚡ fast |  | — | — | — | Detects compression artifacts (blocking, ringing, mosquito noise) |
| `contrast` | — | — | `basic_quality` | img/vid | ⚡ fast |  | — | — | — | Comprehensive technical quality assessment (blur, noise, artifacts, contrast) |
| `cpbd_score` | ↑ higher=better | 0-1, higher=sharper | `cpbd` | img/vid | ⚡ fast |  | — | — | — | Cumulative Probability of Blur Detection (Perceptual Blur) |
| `imaging_artifacts_score` | ↑ higher=better | 0-1, higher=cleaner | `imaging_quality` | img/vid | ⚡ fast |  | — | — | — | Assesses technical quality (Noise, Blockiness) - Proxy for MUSIQ/DOVER |
| `imaging_noise_score` | ↑ higher=better | 0-1, higher=cleaner | `imaging_quality` | img/vid | ⚡ fast |  | — | — | — | Assesses technical quality (Noise, Blockiness) - Proxy for MUSIQ/DOVER |
| `letterbox_ratio` | — | 0-1, 0=no borders | `letterbox` | img/vid | ⚡ fast |  | — | — | — | Border/letterbox detection (0-1, 0=no borders) |
| `noise_score` | ↑ higher=better | — | `basic_quality` | img/vid | ⚡ fast |  | — | — | — | Comprehensive technical quality assessment (blur, noise, artifacts, contrast) |
| `saturation` | — | — | `basic_quality` | img/vid | ⚡ fast |  | — | — | — | Comprehensive technical quality assessment (blur, noise, artifacts, contrast) |
| `spatial_information` | — | higher=more detail | `ti_si` | vid | ⚡ fast |  | — | — | — | ITU-T P.910 Temporal & Spatial Information |
| `technical_score` | ↑ higher=better | — | `basic_quality` | img/vid | ⚡ fast |  | — | — | — | Comprehensive technical quality assessment (blur, noise, artifacts, contrast) |
| `temporal_information` | — | higher=more motion | `ti_si` | vid | ⚡ fast |  | — | — | — | ITU-T P.910 Temporal & Spatial Information |
| `tonal_dynamic_range` | — | 0-100 | `tonal_dynamic_range` | img/vid | ⚡ fast |  | — | — | — | Luminance histogram tonal range (0-100) |

## Aesthetics (9 metrics)

| Metric | Dir | Range | Module | Input | Speed | GPU | Backend | Source | Test | Description |
|--------|-----|-------|--------|-------|-------|-----|---------|--------|------|-------------|
| `aesthetic_mlp_score` | ↑ higher=better | 1-10 | `aesthetic_scoring` | img/vid | ⏱️ medium | ✓ | — | [GitHub](https://github.com/christophschuhmann/improved-aesthetic-predictor) · [HF](https://huggingface.co/openai/clip-vit-large-patch14) | — | Calculates aesthetic score (1-10) using LAION-Aesthetics MLP |
| `aesthetic_score` | ↑ higher=better | — | `aesthetic` | img/vid | ⏱️ medium | ✓ | — | — | — | Estimates aesthetic quality using Aesthetic Predictor V2.5 |
| `cover_aesthetic` | — | — | `cover` | img/vid | ⏱️ medium | ✓ | cover → dover | — | — | COVER 3-branch comprehensive video quality (semantic + aesthetic + technical) |
| `cover_semantic` | — | — | `cover` | img/vid | ⏱️ medium | ✓ | cover → dover | — | — | COVER 3-branch comprehensive video quality (semantic + aesthetic + technical) |
| `creativity_score` | ↑ higher=better | 0-1, higher=better | `creativity` | img/vid | 🐌 slow | ✓ | heuristic → vlm → clip | [HF](https://huggingface.co/llava-hf/llava-1.5-7b-hf) | — | Artistic novelty assessment (VLM / CLIP / heuristic) |
| `dover_aesthetic` | — | — | `dover` | vid | ⏱️ medium | ✓ | heuristic → native → onnx → pyiqa | [GitHub](https://github.com/VQAssessment/DOVER.git) · [HF](https://huggingface.co/dover/DOVER.pth) | — | DOVER disentangled technical + aesthetic VQA (ICCV 2023) |
| `laion_aesthetic` | — | 0-10 | `laion_aesthetic` | img/vid | ⏱️ medium | ✓ | — | — | — | LAION Aesthetics V2 predictor (0-10, industry standard) |
| `nima_score` | ↑ higher=better | 1-10, higher=better | `nima` | img/vid | ⏱️ medium | ✓ | — | — | — | NIMA aesthetic and technical image quality (1-10 scale) |
| `qalign_aesthetic` | ↑ higher=better | 1-5, higher=better | `q_align` | img/vid | 🐌 slow | ✓ | — | [HF](https://huggingface.co/q-future/one-align) | — | Q-Align unified quality + aesthetic assessment (ICML 2024) |

## Audio Quality (15 metrics)

| Metric | Dir | Range | Module | Input | Speed | GPU | Backend | Source | Test | Description |
|--------|-----|-------|--------|-------|-------|-----|---------|--------|------|-------------|
| `audiobox_enjoyment` | — | — | `audiobox_aesthetics` | audio | ⚡ fast |  | audiobox | — | — | Meta Audiobox Aesthetics audio quality (2025) |
| `audiobox_production` | — | — | `audiobox_aesthetics` | audio | ⚡ fast |  | audiobox | — | — | Meta Audiobox Aesthetics audio quality (2025) |
| `av_sync_offset` | — | — | `av_sync` | audio | ⚡ fast |  | — | — | — | Audio-video synchronisation offset detection |
| `dnsmos_bak` | ↑ higher=better | 1-5, higher=better | `dnsmos` | audio | ⏱️ medium |  | torchmetrics | — | — | DNSMOS non-intrusive audio quality (Microsoft, 1-5 MOS) |
| `dnsmos_overall` | ↑ higher=better | 1-5, higher=better | `dnsmos` | audio | ⏱️ medium |  | torchmetrics | — | — | DNSMOS non-intrusive audio quality (Microsoft, 1-5 MOS) |
| `dnsmos_sig` | ↑ higher=better | 1-5, higher=better | `dnsmos` | audio | ⏱️ medium |  | torchmetrics | — | — | DNSMOS non-intrusive audio quality (Microsoft, 1-5 MOS) |
| `estoi_score` | ↑ higher=better | 0-1, higher=better | `audio_estoi` | audio +ref | ⚡ fast |  | — | — | — | ESTOI speech intelligibility (full-reference) |
| `lpdist_score` | ↓ lower=better | lower=better | `audio_lpdist` | audio +ref | ⚡ fast |  | — | — | — | Log-Power Spectral Distance (full-reference audio) |
| `mcd_score` | ↓ lower=better | dB, lower=better | `audio_mcd` | audio +ref | ⚡ fast |  | — | — | — | Mel Cepstral Distortion for TTS/VC quality (full-reference) |
| `oavqa_score` | ↑ higher=better | higher=better | `oavqa` | img/vid | ⚡ fast |  | heuristic → native | — | — | OAVQA omnidirectional audio-visual QA (2024) |
| `p1203_mos` | — | 1-5 | `p1203` | vid | ⚡ fast |  | official → parametric | — | — | ITU-T P.1203 streaming QoE estimation (1-5 MOS) |
| `pesq_score` | ↑ higher=better | -0.5 to 4.5, higher=better | `audio_pesq` | audio +ref | ⚡ fast |  | — | — | — | PESQ speech quality (full-reference, ITU-T P.862) |
| `si_sdr_score` | ↑ higher=better | dB, higher=better | `audio_si_sdr` | audio +ref | ⚡ fast |  | — | — | — | Scale-Invariant SDR for audio quality (full-reference) |
| `utmos_score` | ↑ higher=better | 1-5, higher=better | `audio_utmos` | audio | ⏱️ medium | ✓ | — | — | — | UTMOS no-reference MOS prediction for speech quality |
| `visqol` | ↑ higher=better | 1-5, higher=better | `visqol` | img/vid +ref | ⚡ fast |  | python → cli | [GitHub](https://github.com/google/visqol) | — | ViSQOL audio quality MOS (Google, 1-5, higher=better) |

## Face & Identity (14 metrics)

| Metric | Dir | Range | Module | Input | Speed | GPU | Backend | Source | Test | Description |
|--------|-----|-------|--------|-------|-------|-----|---------|--------|------|-------------|
| `celebrity_id_score` | ↑ higher=better | — | `celebrity_id` | img/vid | ⚡ fast |  | — | — | — | Face identity verification using DeepFace (EvalCrafter celebrity_id_score) |
| `crfiqa_score` | ↑ higher=better | higher=better | `crfiqa` | img/vid | ⚡ fast |  | heuristic → native | — | — | CR-FIQA face quality via classifiability (CVPR 2023) |
| `face_consistency` | ↑ higher=better | — | `clip_temporal` | vid | ⏱️ medium | ✓ | — | [HF](https://huggingface.co/openai/clip-vit-base-patch32) | — | CLIP temporal consistency + face/identity consistency (EvalCrafter clip_temp & face_consistency) |
| `face_count` | — | — | `face_fidelity` | img/vid | ⚡ fast |  | — | — | — | Face detection and per-face quality assessment |
| `face_expression_smoothness` | — | — | `face_landmark_quality` | vid | ⚡ fast |  | — | — | — | Facial landmark jitter, expression smoothness, identity consistency |
| `face_identity_consistency` | ↑ higher=better | 0-1 | `face_landmark_quality` | vid | ⚡ fast |  | — | — | — | Facial landmark jitter, expression smoothness, identity consistency |
| `face_iqa_score` | ↑ higher=better | higher=better | `face_iqa` | img/vid | ⏱️ medium | ✓ | — | — | — | Face-specific IQA via TOPIQ-face (GFIQA-trained, higher=better) |
| `face_landmark_jitter` | ↓ lower=better | lower=better | `face_landmark_quality` | vid | ⚡ fast |  | — | — | — | Facial landmark jitter, expression smoothness, identity consistency |
| `face_quality_score` | ↑ higher=better | higher=better | `face_fidelity` | img/vid | ⚡ fast |  | — | — | — | Face detection and per-face quality assessment |
| `face_recognition_score` | ↑ higher=better | 0-1, higher=better | `identity_loss` | img/vid +ref | ⚡ fast |  | insightface → deepface → mediapipe | — | — | Face identity preservation metric (cosine distance/similarity vs reference) |
| `grafiqs_score` | ↑ higher=better | higher=better | `grafiqs` | img/vid | ⚡ fast |  | heuristic → native | — | — | GraFIQs gradient face quality (CVPRW 2024) |
| `identity_loss` | ↓ lower=better | 0-1, lower=better | `identity_loss` | img/vid +ref | ⚡ fast |  | insightface → deepface → mediapipe | — | — | Face identity preservation metric (cosine distance/similarity vs reference) |
| `magface_score` | ↑ higher=better | higher=better | `magface` | img/vid | ⚡ fast |  | heuristic → native | — | — | MagFace face magnitude quality (CVPR 2021) |
| `serfiq_score` | ↑ higher=better | higher=better | `serfiq` | img/vid | ⚡ fast |  | heuristic → native | — | — | SER-FIQ face quality via embedding robustness (2020) |

## Scene & Content (14 metrics)

| Metric | Dir | Range | Module | Input | Speed | GPU | Backend | Source | Test | Description |
|--------|-----|-------|--------|-------|-------|-----|---------|--------|------|-------------|
| `action_confidence` | — | 0-100 | `action_recognition` | vid +cap | ⏱️ medium | ✓ | — | [HF](https://huggingface.co/MCG-NJU/videomae-large-finetuned-kinetics) | — | Recognizes human actions (VideoMAE / UMT) - Supports Heavy Models |
| `action_score` | ↑ higher=better | 0-100 | `action_recognition` | vid +cap | ⏱️ medium | ✓ | — | [HF](https://huggingface.co/MCG-NJU/videomae-large-finetuned-kinetics) | — | Recognizes human actions (VideoMAE / UMT) - Supports Heavy Models |
| `avg_scene_duration` | — | — | `scene_detection` | vid | ⚡ fast |  | — | — | — | Scene stability metric — penalises rapid cuts (0-1, higher=more stable) |
| `color_score` | ↑ higher=better | — | `color_consistency` | img/vid +cap | ⚡ fast |  | — | — | — | Verifies color attributes in prompt vs video content |
| `commonsense_score` | ↑ higher=better | 0-1, higher=better | `commonsense` | img/vid | 🐌 slow | ✓ | heuristic → vlm → vilt | [HF](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa) | — | Common sense adherence (VLM / ViLT VQA / heuristic) |
| `count_score` | ↑ higher=better | — | `object_detection` | img/vid | ⏱️ medium | ✓ | — | — | — | Detects objects (GRiT / YOLOv8) - Supports Heavy Models |
| `detection_diversity` | — | — | `object_detection` | img/vid | ⏱️ medium | ✓ | — | — | — | Detects objects (GRiT / YOLOv8) - Supports Heavy Models |
| `detection_score` | ↑ higher=better | — | `object_detection` | img/vid | ⏱️ medium | ✓ | — | — | — | Detects objects (GRiT / YOLOv8) - Supports Heavy Models |
| `gradient_detail` | — | 0-100 | `basic_quality` | img/vid | ⚡ fast |  | — | — | — | Comprehensive technical quality assessment (blur, noise, artifacts, contrast) |
| `human_fidelity_score` | ↑ higher=better | 0-1, higher=better | `human_fidelity` | img/vid | ⚡ fast |  | heuristic → dwpose → mediapipe | — | — | Human body/hand/face fidelity (DWPose / MediaPipe / heuristic) |
| `ram_tags` | — | — | `ram_tagging` | img/vid | ⏱️ medium | ✓ | — | [HF](https://huggingface.co/xinyu1205/recognize-anything-plus-model) | — | RAM (Recognize Anything Model) auto-tagging for video frames |
| `scene_complexity` | — | — | `scene_complexity` | vid | ⚡ fast |  | — | — | — | Spatial and temporal scene complexity analysis |
| `video_type` | — | — | `video_type_classifier` | img/vid | ⏱️ medium | ✓ | — | [HF](https://huggingface.co/openai/clip-vit-base-patch32) | — | CLIP zero-shot video content type classification |
| `video_type_confidence` | — | — | `video_type_classifier` | img/vid | ⏱️ medium | ✓ | — | [HF](https://huggingface.co/openai/clip-vit-base-patch32) | — | CLIP zero-shot video content type classification |

## HDR & Color (10 metrics)

| Metric | Dir | Range | Module | Input | Speed | GPU | Backend | Source | Test | Description |
|--------|-----|-------|--------|-------|-------|-----|---------|--------|------|-------------|
| `delta_ictcp` | ↓ lower=better | lower=better | `delta_ictcp` | img/vid +ref | ⚡ fast |  | — | — | — | Delta ICtCp HDR perceptual color difference (lower=better) |
| `hdr_quality` | ↑ higher=better | — | `hdr_sdr_vqa` | vid | ⚡ fast |  | — | — | — | HDR/SDR-aware video quality assessment |
| `hdr_technical_score` | ↑ higher=better | 0-1 | `4k_vqa` | vid | ⚡ fast |  | — | — | — | Memory-efficient quality assessment for 4K+ videos |
| `hdr_vdp` | ↑ higher=better | higher=better | `hdr_vdp` | img/vid +ref | ⚡ fast |  | python → approx | — | — | HDR-VDP visual difference predictor (higher=better) |
| `hdr_vqm` | — | — | `hdr_vqm` | img/vid +ref | ⚡ fast |  | gamma_heuristic → pu21_wavelet | — | — | HDR-aware video quality (PU21+wavelet FR or gamma heuristic fallback) |
| `max_cll` | — | — | `hdr_metadata` | vid | ⚡ fast |  | — | — | — | MaxFALL + MaxCLL HDR static metadata analysis |
| `max_fall` | — | — | `hdr_metadata` | vid | ⚡ fast |  | — | — | — | MaxFALL + MaxCLL HDR static metadata analysis |
| `pu_psnr` | ↑ higher=better | dB, higher=better | `pu_metrics` | img/vid +ref | ⚡ fast |  | — | — | — | PU-PSNR + PU-SSIM for HDR content (perceptually uniform) |
| `pu_ssim` | ↑ higher=better | 0-1, higher=better | `pu_metrics` | img/vid +ref | ⚡ fast |  | — | — | — | PU-PSNR + PU-SSIM for HDR content (perceptually uniform) |
| `sdr_quality` | ↑ higher=better | — | `hdr_sdr_vqa` | vid | ⚡ fast |  | — | — | — | HDR/SDR-aware video quality assessment |

## Codec & Technical (5 metrics)

| Metric | Dir | Range | Module | Input | Speed | GPU | Backend | Source | Test | Description |
|--------|-----|-------|--------|-------|-------|-----|---------|--------|------|-------------|
| `cambi` | ↓ lower=better | 0-24, lower=better | `cambi` | vid | ⚡ fast |  | — | — | — | CAMBI banding/contouring detector (Netflix, 0-24, lower=better) |
| `codec_artifacts` | ↓ lower=better | lower=better | `codec_specific_quality` | vid | ⚡ fast |  | — | [HF](https://huggingface.co/30/1) | — | Codec-level efficiency, GOP quality, and artifact detection |
| `codec_efficiency` | ↑ higher=better | higher=better | `codec_specific_quality` | vid | ⚡ fast |  | — | [HF](https://huggingface.co/30/1) | — | Codec-level efficiency, GOP quality, and artifact detection |
| `gop_quality` | ↑ higher=better | higher=better | `codec_specific_quality` | vid | ⚡ fast |  | — | [HF](https://huggingface.co/30/1) | — | Codec-level efficiency, GOP quality, and artifact detection |
| `p1204_mos` | — | 1-5 | `p1204` | vid | ⚡ fast |  | heuristic → native | — | — | ITU-T P.1204.3 bitstream NR quality (2020) |

## Depth & Spatial (5 metrics)

| Metric | Dir | Range | Module | Input | Speed | GPU | Backend | Source | Test | Description |
|--------|-----|-------|--------|-------|-------|-----|---------|--------|------|-------------|
| `depth_anything_consistency` | ↑ higher=better | — | `depth_anything` | img/vid | ⏱️ medium | ✓ | — | [HF](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) | — | Depth Anything V2 monocular depth estimation and consistency |
| `depth_anything_score` | ↑ higher=better | — | `depth_anything` | img/vid | ⏱️ medium | ✓ | — | [HF](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) | — | Depth Anything V2 monocular depth estimation and consistency |
| `depth_quality` | ↑ higher=better | higher=better | `depth_map_quality` | img/vid | ⏱️ medium | ✓ | — | [HF](https://huggingface.co/intel-isl/MiDaS) | — | Monocular depth map quality (sharpness, completeness, edge alignment) |
| `multiview_consistency` | ↑ higher=better | higher=better | `multi_view_consistency` | vid | ⚡ fast |  | — | — | — | Geometric multi-view consistency via epipolar analysis |
| `stereo_comfort_score` | ↑ higher=better | higher=better | `stereoscopic_quality` | vid | ⚡ fast |  | — | — | — | Stereo 3D comfort and quality assessment |

## Production Quality (5 metrics)

| Metric | Dir | Range | Module | Input | Speed | GPU | Backend | Source | Test | Description |
|--------|-----|-------|--------|-------|-------|-----|---------|--------|------|-------------|
| `banding_severity` | ↓ lower=better | lower=better | `production_quality` | img/vid | ⚡ fast |  | — | — | — | Professional production quality (colour, exposure, focus, banding) |
| `color_grading_score` | ↑ higher=better | — | `production_quality` | img/vid | ⚡ fast |  | — | — | — | Professional production quality (colour, exposure, focus, banding) |
| `exposure_consistency` | ↑ higher=better | — | `production_quality` | img/vid | ⚡ fast |  | — | — | — | Professional production quality (colour, exposure, focus, banding) |
| `focus_quality` | ↑ higher=better | — | `production_quality` | img/vid | ⚡ fast |  | — | — | — | Professional production quality (colour, exposure, focus, banding) |
| `white_balance_score` | ↑ higher=better | — | `production_quality` | img/vid | ⚡ fast |  | — | — | — | Professional production quality (colour, exposure, focus, banding) |

## OCR & Text (7 metrics)

| Metric | Dir | Range | Module | Input | Speed | GPU | Backend | Source | Test | Description |
|--------|-----|-------|--------|-------|-------|-----|---------|--------|------|-------------|
| `auto_caption` | — | — | `captioning` | img/vid | ⏱️ medium | ✓ | — | [HF](https://huggingface.co/Salesforce/blip-image-captioning-base) | — | Generates captions using BLIP + computes BLEU score (EvalCrafter blip_bleu) |
| `ocr_area_ratio` | — | — | `text_detection` | img/vid | ⚡ fast |  | — | — | — | Detects text/watermarks using OCR (PaddleOCR / Tesseract) |
| `ocr_cer` | ↓ lower=better | 0-1, lower=better | `ocr_fidelity` | img/vid | ⚡ fast |  | — | — | — | Checks whether text requested in the caption actually appears in video frames (EvalCrafter OCR) |
| `ocr_fidelity` | ↑ higher=better | 0-100, higher=better | `ocr_fidelity` | img/vid | ⚡ fast |  | — | — | — | Checks whether text requested in the caption actually appears in video frames (EvalCrafter OCR) |
| `ocr_score` | ↑ higher=better | — | `ocr_fidelity` | img/vid | ⚡ fast |  | — | — | — | Checks whether text requested in the caption actually appears in video frames (EvalCrafter OCR) |
| `ocr_wer` | ↓ lower=better | 0-1, lower=better | `ocr_fidelity` | img/vid | ⚡ fast |  | — | — | — | Checks whether text requested in the caption actually appears in video frames (EvalCrafter OCR) |
| `text_overlay_score` | ↑ higher=better | 0-1 | `text_overlay` | img/vid | ⚡ fast |  | — | — | — | Text overlay / subtitle detection in video frames |

## Safety & Ethics (7 metrics)

| Metric | Dir | Range | Module | Input | Speed | GPU | Backend | Source | Test | Description |
|--------|-----|-------|--------|-------|-------|-----|---------|--------|------|-------------|
| `ai_generated_probability` | — | — | `watermark_classifier` | img/vid | ⏱️ medium | ✓ | — | [HF](https://huggingface.co/umm-maybe/AI-image-detector) | — | Classifies video for watermarks using a pretrained model or custom ResNet-50 weights |
| `bias_score` | ↑ higher=better | — | `bias_detection` | img/vid | ⚡ fast |  | — | — | — | Demographic representation analysis (face count, age distribution) |
| `deepfake_probability` | — | — | `deepfake_detection` | img/vid | ⏱️ medium | ✓ | — | [HF](https://huggingface.co/openai/clip-vit-base-patch32) | — | Synthetic media / deepfake likelihood estimation |
| `harmful_content_score` | ↑ higher=better | — | `harmful_content` | img/vid | ⏱️ medium | ✓ | — | [HF](https://huggingface.co/openai/clip-vit-base-patch32) | — | Violence, gore, and disturbing content detection |
| `nsfw_score` | ↑ higher=better | — | `nsfw` | img/vid | ⏱️ medium | ✓ | — | [HF](https://huggingface.co/Falconsai/nsfw_image_detection) | — | Detects NSFW (adult/violent) content using ViT |
| `watermark_probability` | — | — | `watermark_classifier` | img/vid | ⏱️ medium | ✓ | — | [HF](https://huggingface.co/umm-maybe/AI-image-detector) | — | Classifies video for watermarks using a pretrained model or custom ResNet-50 weights |
| `watermark_strength` | — | — | `watermark_robustness` | img/vid | ⚡ fast |  | — | — | — | Invisible watermark detection and strength estimation |

## Image-to-Video Reference (4 metrics)

| Metric | Dir | Range | Module | Input | Speed | GPU | Backend | Source | Test | Description |
|--------|-----|-------|--------|-------|-------|-----|---------|--------|------|-------------|
| `i2v_clip` | — | 0-1 | `i2v_similarity` | vid +ref | ⏱️ medium | ✓ | — | [GitHub](https://github.com/richzhang/PerceptualSimilarity) · [HF](https://huggingface.co/lpips/alex.pth) | — | Image-to-Video reference similarity using CLIP, DINOv2, and LPIPS (sliding window) |
| `i2v_dino` | — | 0-1 | `i2v_similarity` | vid +ref | ⏱️ medium | ✓ | — | [GitHub](https://github.com/richzhang/PerceptualSimilarity) · [HF](https://huggingface.co/lpips/alex.pth) | — | Image-to-Video reference similarity using CLIP, DINOv2, and LPIPS (sliding window) |
| `i2v_lpips` | ↓ lower=better | 0-1, lower=better | `i2v_similarity` | vid +ref | ⏱️ medium | ✓ | — | [GitHub](https://github.com/richzhang/PerceptualSimilarity) · [HF](https://huggingface.co/lpips/alex.pth) | — | Image-to-Video reference similarity using CLIP, DINOv2, and LPIPS (sliding window) |
| `i2v_quality` | ↑ higher=better | 0-100 | `i2v_similarity` | vid +ref | ⏱️ medium | ✓ | — | [GitHub](https://github.com/richzhang/PerceptualSimilarity) · [HF](https://huggingface.co/lpips/alex.pth) | — | Image-to-Video reference similarity using CLIP, DINOv2, and LPIPS (sliding window) |

## Meta & Curation (6 metrics)

| Metric | Dir | Range | Module | Input | Speed | GPU | Backend | Source | Test | Description |
|--------|-----|-------|--------|-------|-------|-----|---------|--------|------|-------------|
| `confidence_score` | ↑ higher=better | — | `unqa` | img/vid | ⚡ fast |  | heuristic → native | — | — | UNQA unified no-reference quality for audio/image/video (2024) |
| `llm_qa_score` | ↑ higher=better | 0-1 | `llm_descriptive_qa` | img/vid | 🐌 slow | ✓ | — | [HF](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf) | — | LMM-based interpretable quality assessment with explanations |
| `nemo_quality_label` | ↑ higher=better | — | `nemo_curator` | img/vid +cap | ⏱️ medium | ✓ | deberta → fasttext → heuristic | — | — | Caption text quality scoring (DeBERTa/FastText/heuristic) |
| `nemo_quality_score` | ↑ higher=better | 0-1 | `nemo_curator` | img/vid +cap | ⏱️ medium | ✓ | deberta → fasttext → heuristic | — | — | Caption text quality scoring (DeBERTa/FastText/heuristic) |
| `usability_rate` | — | — | `usability_rate` | img/vid | ⚡ fast |  | — | — | — | Computes percentage of usable frames based on quality thresholds |
| `vtss` | — | 0-1 | `vtss` | img/vid | ⚡ fast |  | — | — | — | Video Training Suitability Score (0-1, meta-metric) |

## Utility & Validation (46 modules)

Modules that perform validation, embedding, deduplication, or dataset-level analysis without writing individual QualityMetrics fields.

| Module | Input | Speed | GPU | Description |
|--------|-------|-------|-----|-------------|
| `audio` | vid | ⚡ fast |  | Validates audio stream quality and presence |
| `audio_text_alignment` | audio +cap | ⏱️ medium | ✓ | Multimodal alignment check (Audio-Text) using CLAP |
| `background_diversity` | img/vid | ⚡ fast |  | Checks background complexity (entropy) to detect concept bleeding |
| `bd_rate` | img/vid | ⚡ fast |  | BD-Rate codec comparison (dataset-level, negative%=better) |
| `codec_compatibility` | vid | ⚡ fast |  | Validates codec, pixel format, and container for ML dataloader compatibility |
| `dataset_analytics` | img/vid | ⏱️ medium | ✓ | Dataset-level diversity, coverage, outliers, duplicates |
| `decoder_stress` | vid | ⚡ fast |  | Random access decoder stress test |
| `dedup` | img/vid | ⚡ fast |  | Detects duplicates using Perceptual Hashing (pHash) |
| `deduplication` | img/vid | ⚡ fast |  | Detects duplicates using Perceptual Hashing (pHash) |
| `diversity` | img/vid | ⚡ fast |  | Flags redundant samples using embedding similarity (Deduplication) |
| `diversity_selection` | img/vid | ⚡ fast |  | Flags redundant samples using embedding similarity (Deduplication) |
| `embedding` | img/vid | ⏱️ medium | ✓ | Calculates X-CLIP embeddings for similarity search |
| `exposure` | img/vid | ⚡ fast |  | Checks for overexposure, underexposure, and low contrast using histograms |
| `fad` | audio | ⚡ fast |  | Frechet Audio Distance for audio generation (batch metric, 2019) |
| `fgd` | vid | ⚡ fast |  | Frechet Gesture Distance for motion generation (batch metric, 2020) |
| `fmd` | vid | ⚡ fast |  | Frechet Motion Distance for motion generation (batch metric, 2022) |
| `fvd` | vid +ref | ⏱️ medium | ✓ | Fréchet Video Distance for video generation evaluation (batch metric) |
| `fvmd` | vid | ⚡ fast |  | Fréchet Video Motion Distance for motion quality evaluation (batch metric) |
| `generative_distribution` | img/vid | ⏱️ medium | ✓ | Precision / Recall / Coverage / Density (batch metric) |
| `generative_distribution_metrics` | img/vid | ⚡ fast |  | Precision / Recall / Coverage / Density (batch metric) |
| `jedi` | vid | ⏱️ medium | ✓ | JEDi distribution metric (V-JEPA + MMD, ICLR 2025) |
| `jedi_metric` | vid | ⚡ fast |  | JEDi distribution metric (V-JEPA + MMD, ICLR 2025) |
| `kandinsky_motion` | vid | ⏱️ medium | ✓ | Video/Camera Motion Analysis using Kandinsky Video Tools (VideoMAE-V2) |
| `knowledge_graph` | img/vid | ⚡ fast |  | Generates a conceptual knowledge graph of the video dataset |
| `kvd` | vid | ⏱️ medium | ✓ | Kernel Video Distance using Maximum Mean Discrepancy (batch metric) |
| `llm_advisor` | img/vid | 🐌 slow |  | Rule-based improvement recommendations derived from quality metrics (no LLM used) |
| `metadata` | img/vid | ⚡ fast |  | Checks video/image metadata (resolution, FPS, duration, integrity) |
| `msswd` | img/vid | ⏱️ medium |  | MSSWD multi-scale sliced Wasserstein distance via pyiqa (batch, lower=better) |
| `multiple_objects` | img/vid +cap | ⚡ fast |  | Verifies object count matches caption (VBench multiple_objects dimension) |
| `paranoid_decoder` | vid | ⚡ fast |  | Deep bitstream validation using FFmpeg (Paranoid Mode) |
| `resolution_bucketing` | img/vid | ⚡ fast |  | Validates resolution/aspect-ratio fit for training buckets |
| `scene` | vid | ⚡ fast |  | Detects scene cuts and shots using PySceneDetect |
| `scene_tagging` | img/vid | ⏱️ medium | ✓ | Tags scene context (Proxy for Tag2Text/RAM using CLIP) |
| `semantic_selection` | img/vid | ⚡ fast |  | Selects diverse samples based on VLM-extracted semantic traits |
| `sfid` | img/vid | ⏱️ medium |  | SFID spatial Fréchet Inception Distance via pyiqa (batch, lower=better) |
| `spatial_relationship` | img/vid +cap | ⚡ fast |  | Verifies spatial relations (left/right/top/bottom) in prompt vs detections |
| `spectral_upscaling` | img/vid | ⚡ fast |  | Detection of upscaled/fake high-resolution content |
| `stream_metric` | img/vid | ⚡ fast |  | STREAM spatial/temporal generation eval (ICLR 2024) |
| `structural` | vid | ⚡ fast |  | Checks structural integrity (scene cuts, black bars) |
| `style_consistency` | vid | ⚡ fast |  | Appearance Style verification (Gram Matrix Consistency) |
| `temporal_style` | vid | ⚡ fast |  | Analyzes temporal style (Slow Motion, Timelapse, Speed) |
| `umap_projection` | img/vid | ⏱️ medium | ✓ | UMAP/t-SNE/PCA 2-D projection with spread & coverage |
| `vendi` | img/vid | ⚡ fast |  | Vendi Score dataset diversity (NeurIPS 2022, batch metric) |
| `vfr_detection` | vid | ⚡ fast |  | Variable Frame Rate (VFR) and jitter detection |
| `vlm_judge` | img/vid | 🐌 slow | ✓ | Advanced semantic verification using VLM (e.g. LLaVA) |
| `worldscore` | vid | ⚡ fast |  | WorldScore world generation evaluation (ICCV 2025) |
