# Migration guide — Ayase 0.1.80

0.1.80 is the metric provenance release. Metrics that computed something
other than their published definition were renamed to honest names, stub
modules were removed, and unsupported fallback tiers were dropped. This
document covers downstream-visible changes and the compatibility
shims that keep existing integrations working while you migrate.

## Compatibility shims (read-only, deprecated)

**Module names.** Old registry names still resolve through
`ModuleRegistry.get_module()` — `--modules ttsds2`, `modules=["jedi"]`,
and profile lists keep working. Resolution emits a `DeprecationWarning`;
the aliases are not listed in `list_modules()` output.

**Field names.** Reading a renamed field on `QualityMetrics`/`DatasetStats`
returns the new field's value with a `DeprecationWarning`. Constructing
with legacy keys (`QualityMetrics(vqa_a_score=0.5)`) maps onto the new
names. Fields dropped with stub modules read as `None` (the only value
they ever carried) instead of raising `AttributeError`.

Structured output is **not** aliased: `model_dump()`, `non_null_metrics()`,
report JSON, and `to_grouped_dict()` contain only the new names. Code that
reads serialized metrics dicts by key must migrate now.

## Corrected metric semantics in 0.1.80

Fields below kept their names but their computations changed. Where Ayase
retains a different model, preprocessing, sampling, or aggregation, the field
is classified as `adapted` with the deviation recorded in `METRICS.md`.
Values are **not comparable** with what the same field produced before.

`fid`, `sfid`, `is_score`, PRDC precision/recall/density/coverage, `vendi`,
`p1203_mos`, `i2i_dinov2_cls_similarity`, `i2i_clip_similarity`,
`i2i_lpips_alex`, and `psnr99` are conservatively
classified as `adapted`. Distribution metrics select or pool video frame
features through Ayase-defined protocols; P.1203 receives a synthesized video-only session;
DINO and CLIP similarities permit configurable encoders; CLIP reports a single
image-pair cosine rather than the cited benchmark's aggregate protocol, and
LPIPS resizes both images to 256×256. These image-to-image computations retain
their numerical behavior, but require explicit selection under the corrected
provenance classification. PSNR99 also supports a custom
video sampling and aggregation protocol. Source-matching image subprotocols
remain available, but a shared field cannot promise published equivalence for
every supported input. Explicit module selection or adapted-provenance opt-in
is required to emit these fields. FID and sFID leave scores unset when their
required matrix-square-root computation fails; they do not emit a substitute
distance formula.

| Field | Before | Now |
|---|---|---|
| `aesthetic_v25_score`, `aesthetic_v25_dup` (legacy reads: `aesthetic_score`, `vqa_a_score`) | rescaled to 0-100 | native Aesthetic Predictor V2.5 scale **1-10**; video = mean over 5 frames |
| `ocr_fidelity`, `ocr_score` | `(1-NED)·100` of the best frame, higher = better | mean over **all** frames of `(NED+CER+WER)/3`, **lower = better** |
| `ocr_cer`, `ocr_wer` | not emitted | per-video CER/WER means (lower = better) |
| `motion_ac_score` | continuous magnitude score, RAFT-Small on ≤150 frames downscaled to 512 | binary **0/1** match: RAFT things-weights, every frame, native resolution, `large` when mean flow > 5; expected class comes from `expected_motion` config or caption keywords |
| `warping_error` | RAFT-Small, ≤300 strided frames, Farneback fallback | RAFT things-weights, every consecutive pair, replicate-padded; no fallback — unset when RAFT is unavailable |
| `clip_temp` | mean consecutive-pair cosine over ≤32 sampled frames | same formula over **every** frame pair |
| `face_consistency` | mean consecutive-pair cosine (duplicate of `clip_temp`) | mean cosine of frames [1:] to the **first** frame (EvalCrafter Face Consistency) |
| `background_consistency` | all-pairs cosine over 16 sampled frames | per-frame `(max(0, sim_prev) + max(0, sim_first))/2` over all frames (VBench) |
| `subject_consistency` | all-pairs cosine, DINOv2-base, 16 frames | per-frame `(max(0, sim_prev) + max(0, sim_first))/2`, DINO ViT-B/16, all frames (VBench) |
| `peaq_odg`, `peaq_di` | unsupported binary flags and a published-metric classification | documented peaqb-fast BASIC interface with 48 kHz mono 16-bit preprocessing; classified as `adapted`; failed executions and unsupported advanced modes leave values unset |

`ocr_fidelity`, `ocr_score`, `ocr_cer`, `ocr_wer`, and `warping_error` are
now registered as lower-is-better for `ayase filter --min-score`.

## Audio, alignment and reference metric changes

| Field | Before | Now |
|---|---|---|
| `imagebind_score` | `(cos+1)/2` remap | raw ImageBind cosine similarity |
| `imagebind_av_score` | same remap | raw cosine (JavisBench `sim_av`) |
| `human_clap_score` | `(cos+1)/2` over first 10 s, LAION CLAP | raw cosine over the whole clip, official `sarulab-speech/human-clap-wsce-mae` checkpoint |
| `pam_score` | LAION-CLAP sigmoid blend over custom prompt lists | MS-CLAP softmax probability of the published positive prompt (requires `msclap`; unset otherwise) |
| `motion_smoothness` | adjacent-frame triples over 64 sampled frames, `1 - mean_L1` | VBench even/odd reconstruction `(255 - absdiff)/255` over all frames (values shift upward) |
| `mcd_score` | librosa MFCC truncation distance | pymcd's 13-dimensional MFCC/FastDTW distance with coefficient 0 excluded; adapted, not SPTK mel-cepstral MCD (unset without `pymcd`) |
| `dnsmos_*` | wrong output ordering; no P.808 | official Microsoft ONNX outputs in correct order, plus new `dnsmos_p808` |
| `mouth_quality` | MUSIQ-KonIQ checkpoint | MUSIQ-SPAQ (the THEval backbone) via pyiqa |
| `blip_bleu` | hand-rolled BLEU over BLIP captions | official `blip2-opt-2.7b` + pycocoevalcap BLEU (unset without the scorer package) |
| `sd_reference` | 8 sampled frames | every frame (EvalCrafter SD-score protocol) |
| `advanced_flow` | RAFT C_T_SKHT_V2, 512-px downscale, ≤150 frames | RAFT things-equivalent (`C_T_V1`), native resolution, all pairs, 20 flow updates |
| `asr_cer`, `asr_wer` | custom tokenization, clipped to [0,1] | Whisper normalizer (jiwer when installed), unclipped — rates >1.0 now reported |
| `face_recognition_score`, `identity_loss` | whichever of InsightFace/DeepFace loaded first (incomparable distances) | InsightFace only (DeepFace branch removed) |
| `aqascore_score` | generated numeric answer, audio never passed to the processor | P("Yes") of the first generated token, audio passed to Qwen2.5-Omni (opt-in) |
| `lpips_diversity` | random pairs across the whole dataset | pairs within one conditioning input (`reference_path`) — the published protocol |
| `artfid_score` | one reference used as both style and content sets, 8-frame FID | separate style (`style_reference_path`) and content (`reference_path`) sets, all frames; unset when either is missing |

- **Removed:** `nima_onnx` (unverified third-party ONNX export). The module
  name aliases to `nima` (NIMA via pyiqa, adapted for video) and `nima_onnx_score`
  reads as `nima_score`; both emit a deprecation warning.
- **`i2v_similarity`:** declared `own` (was falsely attributed to
  VBench/EvalCrafter). Field names unchanged.
- **New `Sample` field:** `style_reference_path` — second reference channel
  for metrics that need separate style/content sets (`artfid`).

## Distribution and perceptual metric changes

| Field | Before | Now |
|---|---|---|
| `camera_rot_error` | degrees | radians (published CamI2V unit) |
| `camera_trajectory` module | VGGT poses | canonical GLOMAP+COLMAP reconstruction, left-relative poses, `normalize_t` scaling |
| `grafiqs_score` | normalized index, higher=better | raw \|grad\| sum over the FR backbone's BN statistics (official GraFIQs; lower=better) |
| `eyebrow_dynamics_score` | mean \|Δ\| of brow distance | across-frame std per THEval Eq.28 (values are smaller) |
| `is_score` | per-sample score | dataset-level Inception Score (torch-fidelity, 10 splits) written to `DatasetStats` |
| `cdc_score` | invented temporal-consistency proxy | published CDC — mean consecutive-frame JS divergence of per-channel RGB histograms |
| `davis_jf` | fixed-tolerance boundary F, capped frames | official `db_eval_boundary` (adaptive dilation tolerance), all frames |
| `pc_psnr` | one-direction Hausdorff | MPEG D1/D2 symmetric max over both directions |
| `uiqm` | custom component formulas | published UICM/UISM/UConM coefficients and weighting |
| `i2i_gradient_similarity_mean` | plain Sobel cosine | true GMSM (Prewitt maps, ×2 downsample, c=170) |
| `audio_log_f0_dtw` | librosa pYIN + own DTW cost | ESPnet protocol: pyworld.harvest F0, mel-cepstra fastdtw, natural-log RMSE (emitted in cents) |
| `beat_alignment_score` | audio-onset-vs-motion beats | Bailando Eq.15: pose-kinematic beats vs music beats, exp kernel σ=3 |
| `qwen_image_bench_*` | own checklist + own prompts | verbatim official `checklists.py` rubrics, system/user prompts, 0/1/2→0/60/100 scoring |
| `pose_heat_ssim` | single-person frames only | all detected persons per frame (published P×J×2 protocol), SIM3/Umeyama alignment, σ=4 heatmaps |
| `object_integrity_score` | RTMLib YOLOX+RTMPose-m, first 120 frames | canonical mmdet RTMDet-m + mmpose RTMPose-m (body8) over all frames; RTMLib demoted to opt-in |
| `opens2v_nexus_score` | GroundingDINO + CLIP/DINOv2 text match | canonical YOLO-World image-prompt detection + GME-Qwen2-VL embeddings, official gates/thresholds |
| `opens2v_natural_score` | LLaVA judge | verbatim GPT-4o rubric (16 frames @512px, 1–5, ×3 runs); LLaVA demoted to opt-in |
| `video_reward_score` | raw VLM head output | official VideoVLMRewardInference: verbatim `detailed_special` prompt, fps=2/max_pixels=200704 sampling, `use_norm` normalization |
| `cover` module | own 3-branch heads | official COVER weights + cross-gating heads, verbatim `spatial_temporal_view_decomposition` views |
| `vsfa_score` | custom CNN+attention | official VSFA architecture (res5c mean+std → ANN+GRU → TP pooling), upstream checkpoint |
| `flolpips_score` | own flow-weighted LPIPS | official FloLPIPS: per-layer LPIPS-alex diffs weighted by flow error, all consecutive pairs, native res |
| `tifa_score` | single VQA pass on generated media | official tifascore pipeline: LLaMA-2 question generation → UnifiedQA filtering → mPLUG MC-VQA |
| `worldmodelbench_*` | paraphrased prompts + own parsing | verbatim upstream templates/questions and `"no" in pred` parsing (parse failure → 0, as upstream) |
| `camerabench_*` | argmax over custom label set | verbatim 15-primitive binary questions, per-primitive P("Yes") |
| `video_edit_motion_fidelity` | grid 30, ≤60 frames @512px | MTBench protocol: grid 50, whole video, native resolution |
| `face_iqa` | full-frame TOPIQ-face | retinaface detection + canonical 5-landmark alignment (GFIQA protocol) |
| `hdr_metadata` nits | SDR-range luma estimate | true ST.2084 PQ decode via ffmpeg rgb48le, per-pixel max(R,G,B) |
| `dice_edit_*` | paraphrased judge prompts | verbatim official detector/coherence prompts and message order |
| `fad_*` | custom log-mel embeddings | fadtk frame-level embeddings per model family; FAD∞ = linear fit of FAD vs 1/n |
| `afine` | NR variant | published FR variant via pyiqa |

## Renamed modules

| 0.1.79 name | 0.1.80 name |
|---|---|
| `audio_lpdist` | `audio_logmel_dist` |
| `clifvqa` | `clip_feel` |
| `dynamics_range` | `content_variation` |
| `entitybench` | `entity_consistency` |
| `finevq` | `finevq_raw` |
| `geneval` | `clip_prompt_check` |
| `graphsim` | `local_std_uniformity` |
| `hdr_vqm` | `hdr_subband_flicker_score` |
| `jedi` | `mmd_selfsplit` |
| `jedi_metric` | `mmd_selfsplit_metric` |
| `magface` | `face_emb_norm` |
| `modularbvqa` | `clip_slowfast_vq` |
| `movie` | `gabor_flow_vq` |
| `multiple_objects` | `object_count_check` |
| `naturalness` | `brisque_inverted` |
| `pcqm` | `mse_color` |
| `pointssim` | `chamfer_sim` |
| `psnr_div` | `psnr_grad` |
| `psnr_hvs` | `psnr_hvs_approx` |
| `pu_metrics` | `log_metrics` |
| `spherical_psnr` | `erp_psnr` |
| `st_greed` | `mscn_entropy` |
| `st_lpips` | `stlpips_selfdist` |
| `t2v_score` | `t2v_generic_score` |
| `tc_bench` | `clip_event_order` |
| `tlvqm` | `resnet_svr_vq` |
| `ttsds2` | `tts_system_dist` |
| `vader` | `hpsv2_const` |
| `videophy` | `vlm_phy` |
| `videval` | `svr60_vq` |

## Renamed fields (`QualityMetrics`)

| 0.1.79 name | 0.1.80 name |
|---|---|
| `active_speaker_best_lse_c` | `active_speaker_best_conf` |
| `aesthetic_score` | `aesthetic_v25_score` |
| `aigv_alignment` | `aigv_alignment_est` |
| `aigv_dynamic` | `aigv_dynamic_est` |
| `aigv_static` | `aigv_static_est` |
| `aigv_temporal` | `aigv_temporal_est` |
| `audio_f0_voicing_error` | `audio_f0_voiced_mismatch` |
| `celebrity_id_score` | `face_id_similarity` |
| `clifvqa_score` | `clip_feel_score` |
| `cpp_psnr` | `psnr_cosw` |
| `dynamics_range` | `content_variation` |
| `entitybench_appearance_consistency` | `entity_appearance_cos` |
| `entitybench_identity_consistency` | `entity_identity_cos` |
| `finevq_score` | `finevq_raw_mean` |
| `geneval_color_attribution` | `clipchk_color_attribution` |
| `geneval_colors` | `clipchk_colors` |
| `geneval_counting` | `clipchk_counting` |
| `geneval_overall` | `clipchk_overall` |
| `geneval_position` | `clipchk_position` |
| `geneval_single_object` | `clipchk_single_object` |
| `geneval_two_object` | `clipchk_two_object` |
| `graphsim_score` | `local_std_uniformity_score` |
| `hdr_vqm` | `hdr_subband_flicker_score` |
| `i2v_clip` | `i2v_clip_winmed` |
| `i2v_dino` | `i2v_dino_winmed` |
| `i2v_lpips` | `i2v_lpips_winmed` |
| `i2v_quality` | `i2v_quality_blend` |
| `long_form_event_fulfillment` | `clipord_event_fulfillment` |
| `lpdist_score` | `logmel_rmse_db` |
| `magface_score` | `face_emb_norm` |
| `modularbvqa_score` | `clip_slowfast_vq_score` |
| `movie_score` | `gabor_flow_score` |
| `naturalness_score` | `brisque_inverted` |
| `pcqm_score` | `mse_color_score` |
| `physics_iq_score` | `physics_iq_neutral_score` |
| `pointssim_score` | `chamfer_sim_score` |
| `psnr_div` | `psnr_grad` |
| `psnr_hvs` | `psnr_hvs_approx` |
| `psnr_hvs_m` | `psnr_acmask` |
| `pu_psnr` | `log_psnr` |
| `pu_ssim` | `plain_ssim` |
| `s_psnr` | `psnr_sphw` |
| `st_greed_score` | `mscn_entropy_score` |
| `st_lpips` | `stlpips_selfdist` |
| `t2v_alignment` | `t2v_generic_alignment` |
| `t2v_quality` | `t2v_generic_quality` |
| `t2v_score` | `t2v_generic_score` |
| `tcbench_attribute_score` | `clipord_attribute_score` |
| `tcbench_background_score` | `clipord_background_score` |
| `tcbench_object_score` | `clipord_object_score` |
| `tcbench_overall` | `clipord_overall` |
| `tlvqm_score` | `resnet_svr_score` |
| `ttsds2_score` | `tts_system_dist_score` |
| `unified_reward_2_score` | `unified_reward_2_mean` |
| `vader_score` | `hpsv2_const_quality` |
| `video_text_score` | `video_text_logit` |
| `video_text_temporal` | `video_text_consistency` |
| `videophy_pc_score` | `vlm_pc_likert` |
| `videophy_sa_score` | `vlm_sa_likert` |
| `videval_score` | `svr60_score` |
| `vqa_a_score` | `aesthetic_v25_dup` |
| `ws_psnr` | `ws_psnr_linw` |

## Renamed fields (`DatasetStats`)

| 0.1.79 name | 0.1.80 name |
|---|---|
| `jedi` | `mmd_selfsplit` |
| `verse_bench_breakdown` | `verse_bench_breakdown_est` |
| `verse_bench_overall` | `verse_bench_overall_est` |

## Removed modules

Seventeen modules were non-functional stubs and were removed entirely
(requesting one now fails with "Unknown module"): `adadqa`, `aigcvqa`,
`clipvqa`, `fgd`, `fmd`, `presresq`, `qclip`, `sqi`, `t2v_compbench`,
`t2veval`, `thqa`, `ugvq`, `umtscore`, `unified_vqa`, `unqa`,
`video_atlas`, `videoreward`.

Their fields were removed with them. Reads still return `None` via the
removed-field shim:

- `QualityMetrics`: `adadqa_score`, `aigcvqa_aesthetic`,
  `aigcvqa_alignment`, `aigcvqa_technical`, `clipvqa_score`,
  `compbench_action`, `compbench_attribute`, `compbench_numeracy`,
  `compbench_object_rel`, `compbench_overall`, `compbench_scene`,
  `compbench_spatial`, `confidence_score`, `presresq_score`,
  `qclip_score`, `sqi_score`, `t2veval_score`, `thqa_score`,
  `ugvq_score`, `umtscore`, `unified_vqa_score`, `video_atlas_score`,
  `videoreward_mq`, `videoreward_ta`, `videoreward_vq`
- `DatasetStats`: `fgd`, `fmd`, `fvd_content_debiased` (the mean-subtraction
  left the Fréchet distance mathematically identical to `fvd`),
  `fvd_dinov2` (unverified attribution)

## Removed own/heuristic fields

Ayase-defined fields without a published definition were removed from the
schema (reads still return `None` with a `DeprecationWarning`; constructors
drop them silently):

- `QualityMetrics`: `artifacts_score`, `audio_f0_joint_coverage`,
  `face_motion_blink_f1`, `face_motion_ear_correlation`, `gradient_detail`,
  `i2i_blue_bias`, `i2i_chroma_cb_mae`, `i2i_chroma_cr_mae`,
  `i2i_colorfulness_delta`, `i2i_edge_f1`, `i2i_exact_match_ratio`,
  `i2i_green_bias`, `i2i_hist_bhattacharyya_blue`,
  `i2i_hist_bhattacharyya_green`, `i2i_hist_bhattacharyya_red`,
  `i2i_hue_mae_degrees`, `i2i_luminance_mae`, `i2i_mean_bias`,
  `i2i_mutual_information`, `i2i_red_bias`, `i2i_spectral_cosine`,
  `noise_score`,
  `ref4d_overall_score`, `technical_score`, `temporal_risk_rate`,
  `unified_reward_edit_score`, `voice_identity_max`
- `DatasetStats`: `stream_spatial` (own harmonic composite — upstream
  STREAM-S reports fidelity and diversity separately)

Fields split to match their published sources:

| Was | Now |
|---|---|
| `stream_spatial` (harmonic mean) | `stream_F` + `stream_D` on `DatasetStats` |
| `vqinsight_score` (AIGC mean) | `vqinsight_spatial` + `vqinsight_temporal` + `vqinsight_consistency`; `vqinsight_score` remains for `video_type="natural"` |
| `unified_reward_edit_score` (mixed aggregate) | `unified_reward_edit_success_score`, `unified_reward_edit_overediting_score`, `unified_reward_edit_image_1_score`, `unified_reward_edit_image_2_score`, `unified_reward_edit_winner` |
| `ref4d_overall_score` (mean of available dimensions) | the four imported `ref4d_*_score` dimensions only |
| `i2i_*` diagnostic fields | `i2i_mse`, `i2i_mae`, `i2i_gradient_similarity_mean` (published GMSM) remain |

Internal readers now use silent dict access (`model_dump().get`) so retired
names no longer emit warnings inside `ayase scan`/`filter`/TUI output. The
`dedup` default `priority_metric` moved from `technical_score` to
`blur_score`, and the `filter` CLI default `--metric` likewise.

## Provenance gating

Every module now declares a `provenance` class per output field:
`published`, `adapted`, `own`, or `utility`. The default module selection
(balanced/`--quick`/`--deep`, `stats`, `filter`) runs only `published`
and `utility` outputs.

Explicitly naming a module is itself the opt-in — these paths run
`adapted`/`own` modules without further flags:

- `ayase scan --modules a,b` and `ayase run --pipeline "a,b"`
- `AyasePipeline(modules=[...])` and `AyasePipeline(profile=...)`
- `instantiate_profile_modules(profile)`
- `[pipeline] modules = [...]` in `ayase.toml`
- modules selected in the TUI

To opt in on a *default* selection, set `[pipeline] allow_provenance =
["adapted", "own"]` or pass `--allow-provenance adapted,own`.
`Pipeline(modules, allow_provenance=[...])` remains available for
programmatic pipelines that construct module instances directly.

Per-field provenance is exposed on `QualityMetrics.metric_provenance`
and `DatasetStats.metric_provenance`, and `metric_backends` records the
backend that produced each field.

## Behavior changes

- **Distribution metrics without a reference set now report nothing.**
  `fid`, `fad`, `fvd`, `kvd`, `sfid`, `kid`, `cmmd`, `audio_kl`,
  `prdc_dinov2`, `generative_distribution_metrics` previously split the
  evaluated set in half and compared it against itself. That silently
  produced misleading values; without an explicit reference set they now
  skip the dataset metric. Supply a real reference set to keep values.
- **Removed fallback backends.** Substitute tiers that measured a
  different quantity than the declared metric were dropped. Affected
  modules now emit no value when their real backend is unavailable
  instead of a number from a different computation. Notable ones:
  - `audio_visual_sync` requires Synchformer (the energy-based fallback
    is gone)
  - `temporal_flickering` `warping_error` requires RAFT (the Farneback
    fallback is gone)
  - `prdc_dinov2` requires DINOv2 (the handcrafted-feature fallback is
    gone)
  - `dover` runs the official DOVER model only (the pyiqa approximation
    is gone — DOVER scores shift to true DOVER values; re-check any
    thresholds you tuned against the approximation)
  - `deepfake_detection`, `clip_prompt_check`, `compression_artifacts`
    and similar blends now leave fields unset when a required backend is
    missing rather than reporting `0.0`/`0.5`/`1.0` constants
- **Restored as opt-in:** `expression_similarity` (provenance `own`) is
  back but does not run on default selections.

## Integration actions

- Update serialized dictionary keys using the rename tables. Attribute aliases
  do not add legacy keys to JSON or CSV.
- Replace aesthetic percentage thresholds with thresholds calibrated for the
  native model scale. OCR thresholds must use an upper error bound; a larger
  value now means worse text fidelity.
- Treat binary `motion_ac_score` as class agreement, not motion magnitude.
  Use separately declared flow measurements when magnitude is needed.
- Recompute baselines when the model, feature space, reference set, sampling or
  aggregation changes. A rename alias does not convert historical values.
- Select adapted metrics explicitly or set `allow_provenance` when an application
  needs video-frame averages of image metrics or other documented adaptations.
  These fields retain their names; their classification corrects an earlier
  reproduction claim and does not itself change the score calculation.
- Supply `bd_rate.reference_curve` as at least four `[bitrate, quality]` pairs
  for a baseline codec. Candidate points use sample bitrate and the configured
  quality metric (`vmaf` by default; the former `psnr` default did not name a
  declared field). Both curves must use the same higher-is-better quality axis.
  Missing curves and non-overlapping quality ranges yield no score.
- Keep backend availability distinct from provenance: external-backend fields
  are reserved schema slots and do not imply an executable scoring backend.
- Check `run_status.complete` and `availability_excluded` before accepting an
  evaluation. A requested external backend that is unavailable makes the run
  incomplete; provenance exclusion is reported separately.
- Interpret `id_reveal_distance` as squared L2, not Euclidean distance. Its
  reference extraction protocol is adapted; recalibrate thresholds and use
  complete, matching reference sets. Old extraction caches are invalidated.
- Record the Ayase version, module configuration, field provenance and backend
  with evaluation results. Do not pool values across incompatible protocols.
