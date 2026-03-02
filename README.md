# Ayase

Modular media quality metrics toolkit.

## Overview

- 225 quality metrics across visual, temporal, audio, perceptual, and safety categories.
- Modular pipeline - modules compute raw values, downstream apps decide what to do with them.
- CLI and Python API.
- Profile-based pipeline configuration.

## Installation

Core (no ML models, metadata/structural checks only):

```bash
pip install ayase
```

With ML-based quality metrics:

```bash
pip install ayase[ml]           # Everything
pip install ayase[v-perceptual] # CLIP, LPIPS, open-clip, timm
pip install ayase[v-iqa]        # PyIQA, BRISQUE, NIQE, scikit-video
pip install ayase[v-motion]     # RAFT optical flow, decord
pip install ayase[v-ocr]        # PaddleOCR text recognition
pip install ayase[v-face]       # MediaPipe face detection
pip install ayase[v-audio]      # librosa audio analysis
```

Development:

```bash
pip install ayase[dev]          # pytest, black, ruff, mypy
pip install ayase[all]          # Everything including dev + TUI
```

See [MODELS.md](MODELS.md) for the complete inventory of all pretrained weights used by every module. Models are downloaded and cached automatically on first use via HuggingFace Hub, Torch Hub, and PyIQA.

## Metrics

| # | Metric | Title | Tested |
|---:|---|---|---|
| 1 | blur_score | Blur Score | Yes |
| 2 | compression_score | Compression Score | Yes |
| 3 | aesthetic_score | Aesthetic Score | Yes |
| 4 | clip_score | CLIP Score | Yes |
| 5 | brightness | Brightness | Yes |
| 6 | contrast | Contrast | Yes |
| 7 | saturation | Saturation | Yes |
| 8 | fast_vqa_score | Fast VQA Score | Yes |
| 9 | motion_score | Motion Score | Yes |
| 10 | camera_motion_score | Camera Motion Score | Yes |
| 11 | temporal_consistency | Temporal Consistency | Yes |
| 12 | technical_score | Technical Score | Yes |
| 13 | noise_score | Noise Score | Yes |
| 14 | artifacts_score | Artifacts Score | Yes |
| 15 | watermark_probability | Watermark Probability | Yes |
| 16 | ocr_area_ratio | OCR Area Ratio | Yes |
| 17 | face_count | Face Count | Yes |
| 18 | nsfw_score | NSFW Score | Yes |
| 19 | audio_quality_score | Audio Quality Score | Yes |
| 20 | perceptual_hash | Perceptual Hash | Yes |
| 21 | depth_score | Depth Score | Yes |
| 22 | auto_caption | Auto Caption | Yes |
| 23 | vqa_a_score | VQA A Score | Yes |
| 24 | vqa_t_score | VQA T Score | Yes |
| 25 | is_score | IS Score | Yes |
| 26 | sd_score | SD Score | Yes |
| 27 | gradient_detail | Gradient Detail | Yes |
| 28 | blip_bleu | BLIP BLEU | Yes |
| 29 | detection_score | Detection Score | Yes |
| 30 | count_score | Count Score | Yes |
| 31 | color_score | Color Score | Yes |
| 32 | celebrity_id_score | Celebrity ID Score | Yes |
| 33 | ocr_score | OCR Score | Yes |
| 34 | ocr_fidelity | OCR Fidelity | Yes |
| 35 | i2v_clip | I2V CLIP | Yes |
| 36 | i2v_dino | I2V DINO | Yes |
| 37 | i2v_lpips | I2V LPIPS | Yes |
| 38 | i2v_quality | I2V Quality | Yes |
| 39 | action_score | Action Score | Yes |
| 40 | action_confidence | Action Confidence | Yes |
| 41 | flow_score | Flow Score | Yes |
| 42 | motion_ac_score | Motion AC Score | Yes |
| 43 | warping_error | Warping Error | Yes |
| 44 | clip_temp | CLIP Temporal | Yes |
| 45 | face_consistency | Face Consistency | Yes |
| 46 | psnr | PSNR | Yes |
| 47 | ssim | SSIM | Yes |
| 48 | lpips | LPIPS | Yes |
| 49 | spectral_entropy | Spectral Entropy | Yes |
| 50 | spectral_rank | Spectral Rank | Yes |
| 51 | fvd | FVD | Yes |
| 52 | kvd | KVD | Yes |
| 53 | fvmd | FVMD | Yes |
| 54 | vmaf | VMAF | Yes |
| 55 | ms_ssim | MS-SSIM | Yes |
| 56 | vif | VIF | Yes |
| 57 | niqe | NIQE | Yes |
| 58 | t2v_score | T2V Score | Yes |
| 59 | t2v_alignment | T2V Alignment | Yes |
| 60 | t2v_quality | T2V Quality | Yes |
| 61 | dynamics_range | Dynamics Range | Yes |
| 62 | dynamics_controllability | Dynamics Controllability | Yes |
| 63 | scene_complexity | Scene Complexity | Yes |
| 64 | compression_artifacts | Compression Artifacts | Yes |
| 65 | naturalness_score | Naturalness Score | Yes |
| 66 | video_memorability | Video Memorability | Yes |
| 67 | usability_rate | Usability Rate | Yes |
| 68 | confidence_score | Confidence Score | Yes |
| 69 | human_preference_score | Human Preference Score | Yes |
| 70 | engagement_score | Engagement Score | Yes |
| 71 | usability_score | Usability Score | Yes |
| 72 | hdr_quality | HDR Quality | Yes |
| 73 | sdr_quality | SDR Quality | Yes |
| 74 | temporal_information | Temporal Information | Yes |
| 75 | spatial_information | Spatial Information | Yes |
| 76 | flicker_score | Flicker Score | Yes |
| 77 | judder_score | Judder Score | Yes |
| 78 | stutter_score | Stutter Score | Yes |
| 79 | dists | DISTS | Yes |
| 80 | fsim | FSIM | Yes |
| 81 | gmsd | GMSD | Yes |
| 82 | vsi_score | VSI Score | Yes |
| 83 | brisque | BRISQUE | Yes |
| 84 | pesq_score | PESQ Score | Yes |
| 85 | av_sync_offset | A/V Sync Offset | Yes |
| 86 | dover_score | DOVER Score | Yes |
| 87 | dover_technical | DOVER Technical | Yes |
| 88 | dover_aesthetic | DOVER Aesthetic | Yes |
| 89 | topiq_score | TOPIQ Score | Yes |
| 90 | liqe_score | LIQE Score | Yes |
| 91 | clip_iqa_score | CLIP-IQA Score | Yes |
| 92 | color_grading_score | Color Grading Score | Yes |
| 93 | white_balance_score | White Balance Score | Yes |
| 94 | exposure_consistency | Exposure Consistency | Yes |
| 95 | focus_quality | Focus Quality | Yes |
| 96 | banding_severity | Banding Severity | Yes |
| 97 | qalign_quality | Q-Align Quality | Yes |
| 98 | qalign_aesthetic | Q-Align Aesthetic | Yes |
| 99 | face_quality_score | Face Quality Score | Yes |
| 100 | face_identity_consistency | Face Identity Consistency | Yes |
| 101 | face_expression_smoothness | Face Expression Smoothness | Yes |
| 102 | face_landmark_jitter | Face Landmark Jitter | Yes |
| 103 | object_permanence_score | Object Permanence Score | Yes |
| 104 | semantic_consistency | Semantic Consistency | Yes |
| 105 | depth_temporal_consistency | Depth Temporal Consistency | Yes |
| 106 | subject_consistency | Subject Consistency | Yes |
| 107 | background_consistency | Background Consistency | Yes |
| 108 | motion_smoothness | Motion Smoothness | Yes |
| 109 | codec_efficiency | Codec Efficiency | Yes |
| 110 | gop_quality | GOP Quality | Yes |
| 111 | codec_artifacts | Codec Artifacts | Yes |
| 112 | deepfake_probability | Deepfake Probability | Yes |
| 113 | ai_generated_probability | AI-Generated Probability | Yes |
| 114 | harmful_content_score | Harmful Content Score | Yes |
| 115 | watermark_strength | Watermark Strength | Yes |
| 116 | bias_score | Bias Score | Yes |
| 117 | depth_quality | Depth Quality | Yes |
| 118 | multiview_consistency | Multiview Consistency | Yes |
| 119 | stereo_comfort_score | Stereo Comfort Score | Yes |
| 120 | musiq_score | MUSIQ Score | Yes |
| 121 | contrique_score | CONTRIQUE Score | Yes |
| 122 | mdtvsfa_score | MDTVSFA Score | Yes |
| 123 | nima_score | NIMA Score | Yes |
| 124 | dbcnn_score | DBCNN Score | Yes |
| 125 | wadiqam_score | WaDIQaM Score | Yes |
| 126 | maniqa_score | MANIQA Score | Yes |
| 127 | arniqa_score | ARNIQA Score | Yes |
| 128 | qualiclip_score | QualiCLIP Score | Yes |
| 129 | pieapp | PieAPP | Yes |
| 130 | cw_ssim | CW-SSIM | Yes |
| 131 | nlpd | NLPD | Yes |
| 132 | mad | MAD | Yes |
| 133 | ahiq | AHIQ | Yes |
| 134 | topiq_fr | TOPIQ-FR | Yes |
| 135 | dreamsim | DreamSim | Yes |
| 136 | cover_score | Cover Score | Yes |
| 137 | cover_technical | Cover Technical | Yes |
| 138 | cover_aesthetic | Cover Aesthetic | Yes |
| 139 | cover_semantic | Cover Semantic | Yes |
| 140 | vqa_score_alignment | VQAScore Alignment | Yes |
| 141 | videoscore_visual | VideoScore Visual | Yes |
| 142 | videoscore_temporal | VideoScore Temporal | Yes |
| 143 | videoscore_dynamic | VideoScore Dynamic | Yes |
| 144 | videoscore_alignment | VideoScore Alignment | Yes |
| 145 | videoscore_factual | VideoScore Factual | Yes |
| 146 | face_iqa_score | Face IQA Score | Yes |
| 147 | scene_stability | Scene Stability | Yes |
| 148 | avg_scene_duration | Average Scene Duration | Yes |
| 149 | raft_motion_score | RAFT Motion Score | Yes |
| 150 | ram_tags | RAM Tags | Yes |
| 151 | depth_anything_score | Depth Anything Score | Yes |
| 152 | depth_anything_consistency | Depth Anything Consistency | Yes |
| 153 | video_type | Video Type | Yes |
| 154 | video_type_confidence | Video Type Confidence | Yes |
| 155 | jedi | JEDi | Yes |
| 156 | trajan_score | TRAJAN Score | Yes |
| 157 | promptiqa_score | PromptIQA Score | Yes |
| 158 | aigv_static | AIGV Static | Yes |
| 159 | aigv_temporal | AIGV Temporal | Yes |
| 160 | aigv_dynamic | AIGV Dynamic | Yes |
| 161 | aigv_alignment | AIGV Alignment | Yes |
| 162 | video_reward_score | Video Reward Score | Yes |
| 163 | text_overlay_score | Text Overlay Score | Yes |
| 164 | ptlflow_motion_score | PTLFlow Motion Score | Yes |
| 165 | qcn_score | QCN Score | Yes |
| 166 | finevq_score | FineVQ Score | Yes |
| 167 | kvq_score | KVQ Score | Yes |
| 168 | rqvqa_score | RQ-VQA Score | Yes |
| 169 | videval_score | VIDEVAL Score | Yes |
| 170 | tlvqm_score | TLVQM Score | Yes |
| 171 | funque_score | FUNQUE Score | Yes |
| 172 | movie_score | MOVIE Score | Yes |
| 173 | st_greed_score | ST-GREED Score | Yes |
| 174 | c3dvqa_score | C3DVQA Score | Yes |
| 175 | flolpips | FloLPIPS | Yes |
| 176 | hdr_vqm | HDR-VQM | Yes |
| 177 | st_lpips | ST-LPIPS | Yes |
| 178 | camera_jitter_score | Camera Jitter Score | Yes |
| 179 | jump_cut_score | Jump Cut Score | Yes |
| 180 | playback_speed_score | Playback Speed Score | Yes |
| 181 | flow_coherence | Flow Coherence | Yes |
| 182 | letterbox_ratio | Letterbox Ratio | Yes |
| 183 | vtss | VTSS | Yes |
| 184 | cnniqa_score | CNNIQA Score | Yes |
| 185 | hyperiqa_score | HyperIQA Score | Yes |
| 186 | paq2piq_score | PaQ2PiQ Score | Yes |
| 187 | tres_score | TReS Score | Yes |
| 188 | unique_score | Unique Score | Yes |
| 189 | laion_aesthetic | LAION Aesthetic | Yes |
| 190 | compare2score | Compare2Score | Yes |
| 191 | afine_score | A-FINE Score | Yes |
| 192 | ckdn_score | CKDN Score | Yes |
| 193 | deepwsd_score | DeepWSD Score | Yes |
| 194 | ssimulacra2 | SSIMULACRA2 | Yes |
| 195 | butteraugli | Butteraugli | Yes |
| 196 | flip_score | Flip Score | Yes |
| 197 | vmaf_neg | VMAF-NEG | Yes |
| 198 | ilniqe | ILNIQE | Yes |
| 199 | nrqm | NRQM | Yes |
| 200 | pi_score | PI Score | Yes |
| 201 | piqe | PIQE | Yes |
| 202 | maclip_score | MACLIP Score | Yes |
| 203 | dmm | DMM | Yes |
| 204 | wadiqam_fr | WaDIQaM-FR | Yes |
| 205 | ssimc | SSIM-C | Yes |
| 206 | cambi | CAMBI | Yes |
| 207 | xpsnr | XPSNR | Yes |
| 208 | vmaf_phone | VMAF-Phone | Yes |
| 209 | vmaf_4k | VMAF-4K | Yes |
| 210 | visqol | ViSQOL | Yes |
| 211 | dnsmos_overall | DNSMOS Overall | Yes |
| 212 | dnsmos_sig | DNSMOS Signal | Yes |
| 213 | dnsmos_bak | DNSMOS Background | Yes |
| 214 | pu_psnr | PU-PSNR | Yes |
| 215 | pu_ssim | PU-SSIM | Yes |
| 216 | max_fall | Max FALL | Yes |
| 217 | max_cll | Max CLL | Yes |
| 218 | hdr_vdp | HDR-VDP | Yes |
| 219 | delta_ictcp | Delta ICtCp | Yes |
| 220 | ciede2000 | CIEDE2000 | Yes |
| 221 | psnr_hvs | PSNR-HVS | Yes |
| 222 | psnr_hvs_m | PSNR-HVS-M | Yes |
| 223 | cgvqm | CGVQM | Yes |
| 224 | strred | STRRED | Yes |
| 225 | p1203_mos | P.1203 MOS | Yes |

## Quick Start

### CLI

```bash
# Scan a dataset and get a report
ayase scan ./my_dataset

# Scan with specific modules
ayase scan ./my_dataset --modules metadata,basic_quality,motion

# List all available modules
ayase modules list

# Check which modules can be loaded (dependencies installed)
ayase modules check

# Filter dataset by quality score
ayase filter ./my_dataset --min-score 70 --output ./filtered
```

### Python API (recommended)

```python
from ayase import AyasePipeline

ayase = AyasePipeline(modules=["basic"])
results = ayase.run("./my_dataset")

for path, sample in results.items():
    if sample.quality_metrics:
        print(f"{sample.path.name}: technical={sample.quality_metrics.technical_score}")

print(f"Total: {ayase.stats.total_samples}, Valid: {ayase.stats.valid_samples}")
ayase.export("report.json")
```

`AyasePipeline` accepts three ways to configure modules:

```python
# By module names
ayase = AyasePipeline(modules=["metadata", "basic_quality", "motion"])

# By profile dict
ayase = AyasePipeline(profile={
    "name": "my_check",
    "modules": ["basic", "aesthetic"],
    "module_config": {
        "aesthetic": {"model_name": "openai/clip-vit-large-patch14"},
    },
})

# By profile file
ayase = AyasePipeline(profile="my_profile.toml")

# With custom config
from ayase.config import AyaseConfig
ayase = AyasePipeline(config=AyaseConfig(general={"parallel_jobs": 16}), modules=["basic"])
```

### Low-level Pipeline API

```python
import asyncio
from pathlib import Path
from ayase.pipeline import Pipeline, ModuleRegistry
from ayase.scanner import scan_dataset

ModuleRegistry.discover_modules()
module_names = ["metadata", "basic_quality", "semantic_alignment"]
modules = [ModuleRegistry.get_module(n)() for n in module_names]

pipeline = Pipeline(modules)
pipeline.start()

samples = scan_dataset(Path("./my_dataset"), recursive=True)
for sample in samples:
    processed = asyncio.run(pipeline.process_sample(sample))

pipeline.stop()
pipeline.export_report("report.json", format="json")
```

### Profile-based pipelines

```python
from ayase import load_profile, instantiate_profile_modules

profile = load_profile("my_profile.toml")
modules = instantiate_profile_modules(profile)
# modules is a list of PipelineModule instances ready for Pipeline()
```

## Configuration

Create `ayase.toml` in your project root:

```toml
[general]
parallel_jobs = 8
cache_enabled = true

[quality]
enable_blur_detection = true
blur_threshold = 100.0

[pipeline]
dataset_path = "./my_dataset"
modules = ["metadata", "basic_quality", "motion"]
plugin_folders = ["plugins"]

[output]
default_format = "markdown"
artifacts_dir = "reports"
artifacts_format = "json"

[filter]
default_mode = "list"
min_score_threshold = 60
```

Ayase looks for config in: `./ayase.toml` -> `~/.config/ayase/config.toml` -> built-in defaults.

## Writing Plugins

Create a `.py` file in your `plugins/` folder:

```python
from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

class MyCustomCheck(PipelineModule):
    name = "my_check"
    description = "Custom quality check"
    default_config = {"threshold": 0.5}

    def process(self, sample: Sample) -> Sample:
        # Your logic here
        if some_score < self.config["threshold"]:
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message="Quality below threshold",
                )
            )
        return sample
```

Then run:

```bash
ayase scan ./data --modules metadata,my_check
```

## Development

```bash
git clone <repo-url>
cd ayase
pip install -e ".[dev]"

# Run tests
pytest

# Lint and format
ruff check src/ tests/
black src/ tests/

# Type check
mypy src/ayase
```

## License

MIT -- see [LICENSE](LICENSE).
