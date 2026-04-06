"""Data models for Ayase using Pydantic."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationIssue(BaseModel):
    """A validation issue found in a sample."""

    severity: ValidationSeverity
    message: str
    details: Optional[Dict[str, Any]] = None
    recommendation: Optional[str] = None  # Suggestion for fixing the issue


class VideoMetadata(BaseModel):
    """Metadata extracted from a video file."""

    width: int
    height: int
    frame_count: int
    fps: float
    duration: float  # seconds
    codec: Optional[str] = None
    bitrate: Optional[int] = None  # bits per second
    file_size: int  # bytes
    scene_changes: List[float] = Field(default_factory=list)  # List of timestamps (seconds)


class ImageMetadata(BaseModel):
    """Metadata extracted from an image file."""

    width: int
    height: int
    channels: int
    format: str
    file_size: int  # bytes


class AudioMetadata(BaseModel):
    """Metadata extracted from an audio stream."""

    sample_rate: int
    channels: int
    bitrate: Optional[int] = None
    codec: str
    duration: float  # seconds
    language: Optional[str] = None


class CaptionMetadata(BaseModel):
    """Metadata about associated caption/text."""

    text: str
    length: int
    language: Optional[str] = None
    source_file: Optional[Path] = None


class QualityMetrics(BaseModel):
    """Quality assessment metrics for a sample.

    All fields are Optional — modules populate only the metrics they compute.
    Use ``to_grouped_dict()`` for a structured view, or ``non_null_metrics()``
    to get only the metrics that were actually computed.
    """

    model_config = ConfigDict(extra="forbid")

    # -- Field grouping registry (field name → category) ------------------
    _FIELD_GROUPS: dict = {
        # Basic visual quality
        "blur_score": "basic",
        "brightness": "basic",
        "contrast": "basic",
        "saturation": "basic",
        "noise_score": "basic",
        "artifacts_score": "basic",
        "technical_score": "basic",
        "compression_artifacts": "basic",
        "temporal_information": "basic",
        "spatial_information": "basic",
        "letterbox_ratio": "basic",
        "tonal_dynamic_range": "basic",
        "cpbd_score": "basic",
        "imaging_noise_score": "basic",
        "imaging_artifacts_score": "basic",
        # Aesthetics
        "aesthetic_score": "aesthetic",
        "laion_aesthetic": "aesthetic",
        "aesthetic_mlp_score": "aesthetic",
        "nima_score": "aesthetic",
        "qalign_aesthetic": "aesthetic",
        "dover_aesthetic": "aesthetic",
        "cover_aesthetic": "aesthetic",
        "cover_semantic": "aesthetic",
        # Text-video alignment
        "clip_score": "alignment",
        "blip_bleu": "alignment",
        "t2v_score": "alignment",
        "t2v_alignment": "alignment",
        "vqa_score_alignment": "alignment",
        "videoscore_alignment": "alignment",
        "videoscore2_alignment": "alignment",
        "aigv_alignment": "alignment",
        "sd_score": "alignment",
        "videoscore_factual": "alignment",
        "videoscore2_physical": "alignment",
        "vqa_a_score": "alignment",
        "vqa_t_score": "alignment",
        "video_text_score": "alignment",
        "video_reward_score": "alignment",
        "tifa_score": "alignment",
        "image_reward_score": "alignment",
        "pickscore_score": "alignment",
        "hpsv3_score": "alignment",
        "chipqa_score": "nr_quality",
        # Motion & dynamics
        "motion_score": "motion",
        "camera_motion_score": "motion",
        "flow_score": "motion",
        "motion_ac_score": "motion",
        "motion_smoothness": "motion",
        "raft_motion_score": "motion",
        "ptlflow_motion_score": "motion",
        "dynamics_range": "motion",
        "dynamics_controllability": "motion",
        "trajan_score": "motion",
        "camera_jitter_score": "motion",
        "videoscore_dynamic": "motion",
        "aigv_dynamic": "motion",
        "stabilized_motion_score": "motion",
        "stabilized_camera_score": "motion",
        "playback_speed_score": "motion",
        # Temporal consistency
        "warping_error": "temporal",
        "clip_temp": "temporal",
        "flicker_score": "temporal",
        "judder_score": "temporal",
        "stutter_score": "temporal",
        "subject_consistency": "temporal",
        "background_consistency": "temporal",
        "scene_stability": "temporal",
        "flow_coherence": "temporal",
        "object_permanence_score": "temporal",
        "semantic_consistency": "temporal",
        "depth_temporal_consistency": "temporal",
        "videoscore_temporal": "temporal",
        "aigv_temporal": "temporal",
        "video_text_temporal": "temporal",
        "jump_cut_score": "temporal",
        # Perceptual quality (no-reference)
        "fast_vqa_score": "nr_quality",
        "dover_score": "nr_quality",
        "dover_technical": "nr_quality",
        "topiq_score": "nr_quality",
        "liqe_score": "nr_quality",
        "clip_iqa_score": "nr_quality",
        "brisque": "nr_quality",
        "niqe": "nr_quality",
        "musiq_score": "nr_quality",
        "maniqa_score": "nr_quality",
        "qalign_quality": "nr_quality",
        "t2v_quality": "nr_quality",
        "cover_score": "nr_quality",
        "cover_technical": "nr_quality",
        "naturalness_score": "nr_quality",
        "video_memorability": "nr_quality",
        "contrique_score": "nr_quality",
        "mdtvsfa_score": "nr_quality",
        "dbcnn_score": "nr_quality",
        "wadiqam_score": "nr_quality",
        "arniqa_score": "nr_quality",
        "qualiclip_score": "nr_quality",
        "promptiqa_score": "nr_quality",
        "aigv_static": "nr_quality",
        "qcn_score": "nr_quality",
        "finevq_score": "nr_quality",
        "kvq_score": "nr_quality",
        "rqvqa_score": "nr_quality",
        "videval_score": "nr_quality",
        "tlvqm_score": "nr_quality",
        "cnniqa_score": "nr_quality",
        "hyperiqa_score": "nr_quality",
        "paq2piq_score": "nr_quality",
        "tres_score": "nr_quality",
        "unique_score": "nr_quality",
        "compare2score": "nr_quality",
        "afine_score": "nr_quality",
        "ilniqe": "nr_quality",
        "nrqm": "nr_quality",
        "pi_score": "nr_quality",
        "piqe": "nr_quality",
        "internvqa_score": "nr_quality",
        "nr_gvqm_score": "nr_quality",
        "maclip_score": "nr_quality",
        "cgvqm": "fr_quality",
        "spectral_entropy": "nr_quality",
        "spectral_rank": "nr_quality",
        "videoscore_visual": "nr_quality",
        "videoscore2_visual": "nr_quality",
        # Perceptual similarity (full-reference)
        "vmaf": "fr_quality",
        "ms_ssim": "fr_quality",
        "vif": "fr_quality",
        "dists": "fr_quality",
        "fsim": "fr_quality",
        "dreamsim": "fr_quality",
        "gmsd": "fr_quality",
        "vsi_score": "fr_quality",
        "pieapp": "fr_quality",
        "cw_ssim": "fr_quality",
        "nlpd": "fr_quality",
        "mad": "fr_quality",
        "ahiq": "fr_quality",
        "topiq_fr": "fr_quality",
        "funque_score": "fr_quality",
        "movie_score": "fr_quality",
        "st_greed_score": "fr_quality",
        "c3dvqa_score": "fr_quality",
        "flolpips": "fr_quality",
        "st_lpips": "fr_quality",
        "ssimulacra2": "fr_quality",
        "butteraugli": "fr_quality",
        "flip_score": "fr_quality",
        "vmaf_neg": "fr_quality",
        "vmaf_phone": "fr_quality",
        "vmaf_4k": "fr_quality",
        "dmm": "fr_quality",
        "wadiqam_fr": "fr_quality",
        "ssimc": "fr_quality",
        "ckdn_score": "fr_quality",
        "deepwsd_score": "fr_quality",
        "xpsnr": "fr_quality",
        "ciede2000": "fr_quality",
        "psnr_hvs": "fr_quality",
        "psnr_hvs_m": "fr_quality",
        "strred": "fr_quality",
        # Face
        "face_count": "face",
        "face_consistency": "face",
        "face_quality_score": "face",
        "face_identity_consistency": "face",
        "face_expression_smoothness": "face",
        "face_landmark_jitter": "face",
        "face_iqa_score": "face",
        "celebrity_id_score": "face",
        "identity_loss": "face",
        "face_recognition_score": "face",
        "face_cross_similarity": "face",
        "face_identity_count": "face",
        # OCR & text
        "ocr_area_ratio": "text",
        "ocr_score": "text",
        "ocr_fidelity": "text",
        "ocr_cer": "text",
        "ocr_wer": "text",
        "text_overlay_score": "text",
        "auto_caption": "text",
        # I2V reference
        "i2v_clip": "i2v",
        "i2v_dino": "i2v",
        "i2v_lpips": "i2v",
        "i2v_quality": "i2v",
        # Safety & ethics
        "nsfw_score": "safety",
        "watermark_probability": "safety",
        "deepfake_probability": "safety",
        "ai_generated_probability": "safety",
        "harmful_content_score": "safety",
        "watermark_strength": "safety",
        "bias_score": "safety",
        # Audio
        "pesq_score": "audio",
        "estoi_score": "audio",
        "mcd_score": "audio",
        "si_sdr_score": "audio",
        "lpdist_score": "audio",
        "utmos_score": "audio",
        "av_sync_offset": "audio",
        "visqol": "audio",
        "dnsmos_overall": "audio",
        "dnsmos_sig": "audio",
        "dnsmos_bak": "audio",
        "p1203_mos": "audio",
        # Scene & content
        "action_score": "scene",
        "action_confidence": "scene",
        "detection_score": "scene",
        "detection_diversity": "scene",
        "count_score": "scene",
        "color_score": "scene",
        "scene_complexity": "scene",
        "gradient_detail": "scene",
        "avg_scene_duration": "scene",
        "ram_tags": "scene",
        "video_type": "scene",
        "video_type_confidence": "scene",
        # Distribution (dataset-level, per-sample placeholder)
        "is_score": "distribution",
        # HDR
        "hdr_quality": "hdr",
        "sdr_quality": "hdr",
        "pu_psnr": "hdr",
        "pu_ssim": "hdr",
        "max_fall": "hdr",
        "max_cll": "hdr",
        "hdr_vdp": "hdr",
        "delta_ictcp": "hdr",
        "hdr_technical_score": "hdr",
        "hdr_vqm": "hdr",
        "hdr_chipqa_score": "hdr",
        "hdrmax_score": "hdr",
        "brightrate_score": "hdr",
        # Codec
        "codec_efficiency": "codec",
        "gop_quality": "codec",
        "codec_artifacts": "codec",
        "cambi": "codec",
        # Depth & spatial
        "depth_quality": "spatial",
        "multiview_consistency": "spatial",
        "stereo_comfort_score": "spatial",
        "depth_anything_score": "spatial",
        "depth_anything_consistency": "spatial",
        # Production quality
        "color_grading_score": "production",
        "white_balance_score": "production",
        "exposure_consistency": "production",
        "focus_quality": "production",
        "banding_severity": "production",
        # Meta / curation
        "usability_rate": "meta",
        "confidence_score": "meta",
        "human_preference_score": "meta",
        "engagement_score": "meta",
        "vtss": "meta",
        "perceptual_hash": "meta",
        "nemo_quality_score": "meta",
        "llm_qa_score": "meta",
        "nemo_quality_label": "meta",
        # VBench-2.0 faithfulness
        "human_fidelity_score": "scene",
        "physics_score": "motion",
        "commonsense_score": "scene",
        "creativity_score": "aesthetic",
        # ChronoMagic-Bench
        "chronomagic_mt_score": "temporal",
        "chronomagic_ch_score": "temporal",
        # T2V-CompBench
        "compbench_attribute": "alignment",
        "compbench_object_rel": "alignment",
        "compbench_action": "alignment",
        "compbench_spatial": "alignment",
        "compbench_numeracy": "alignment",
        "compbench_scene": "alignment",
        "compbench_overall": "alignment",
        # NR-VQA (new models)
        "rapique_score": "nr_quality",
        "conviqt_score": "nr_quality",
        "stablevqa_score": "nr_quality",
        "maxvqa_score": "nr_quality",
        "bvqi_score": "nr_quality",
        "modularbvqa_score": "nr_quality",
        "ptmvqa_score": "nr_quality",
        "clipvqa_score": "nr_quality",
        "discovqa_score": "nr_quality",
        "zoomvqa_score": "nr_quality",
        "faver_score": "nr_quality",
        "siamvqa_score": "nr_quality",
        "memoryvqa_score": "nr_quality",
        "sama_score": "nr_quality",
        "clifvqa_score": "nr_quality",
        "simplevqa_score": "nr_quality",
        "adadqa_score": "nr_quality",
        "mdvqa_semantic": "nr_quality",
        "mdvqa_distortion": "nr_quality",
        "mdvqa_motion": "nr_quality",
        # FR-VQA (new models)
        "rankdvqa_score": "fr_quality",
        "compressed_vqa_hdr": "fr_quality",
        "deepvqa_score": "fr_quality",
        "st_mad": "fr_quality",
        "avqt_score": "fr_quality",
        "pvmaf_score": "fr_quality",
        "sr4kvqa_score": "nr_quality",  # NR: evaluates SR artifacts without reference
        # AIGC-specific VQA
        "crave_score": "nr_quality",
        "aigcvqa_technical": "nr_quality",
        "aigcvqa_aesthetic": "nr_quality",
        "aigcvqa_alignment": "alignment",
        "ugvq_score": "nr_quality",
        "aigvqa_score": "nr_quality",
        "t2veval_score": "alignment",
        "world_consistency_score": "temporal",
        # LLM/VLM-based VQA
        "vqa2_score": "nr_quality",
        "lmmvqa_score": "nr_quality",
        "vqinsight_score": "nr_quality",
        "vqathinker_score": "nr_quality",
        "qclip_score": "nr_quality",
        "presresq_score": "nr_quality",
        # Distribution-level
        "umtscore": "alignment",
        # Video reward
        "videoreward_vq": "nr_quality",
        "videoreward_mq": "motion",
        "videoreward_ta": "alignment",
        "vader_score": "nr_quality",
        # 360/VR spherical
        "s_psnr": "fr_quality",
        "ws_psnr": "fr_quality",
        "cpp_psnr": "fr_quality",
        "ws_ssim": "fr_quality",
        "mc360iqa_score": "nr_quality",
        "provqa_score": "nr_quality",
        # Point cloud
        "pc_d1_psnr": "fr_quality",
        "pc_d2_psnr": "fr_quality",
        "pcqm_score": "fr_quality",
        "graphsim_score": "fr_quality",
        "pointssim_score": "fr_quality",
        "mm_pcqa_score": "nr_quality",
        # Streaming QoE
        "p1204_mos": "codec",
        "sqi_score": "nr_quality",
        "video_atlas_score": "nr_quality",
        # Face quality (recognition-aware)
        "serfiq_score": "face",
        "crfiqa_score": "face",
        "magface_score": "face",
        "grafiqs_score": "face",
        # Niche domains
        "uiqm_score": "nr_quality",
        "uciqe_score": "nr_quality",
        "oavqa_score": "audio",
        # Classic NR-VQA
        "viideo_score": "nr_quality",
        "vbliinds_score": "nr_quality",
        "vsfa_score": "nr_quality",
        "speedqa_score": "nr_quality",
        "gamival_score": "nr_quality",
        # Task-specific FR
        "erqa_score": "fr_quality",
        "vfips_score": "fr_quality",
        "artfid_score": "fr_quality",
        "psnr_div": "fr_quality",
        "psnr99": "fr_quality",
        # pyiqa built-ins
        "deepdc_score": "nr_quality",
        # Distribution/generation
        # Audio aesthetics
        "audiobox_production": "audio",
        "audiobox_enjoyment": "audio",
        "song_eval_coherence": "audio",
        "song_eval_musicality": "audio",
        "song_eval_memorability": "audio",
        "song_eval_clarity": "audio",
        "song_eval_naturalness": "audio",
        # Talking head / lip sync
        "thqa_score": "nr_quality",
        "lse_d": "temporal",
        "lse_c": "temporal",
        # Video segmentation
        "davis_j": "temporal",
        "davis_f": "temporal",
        # Video colorization
        "cdc_score": "temporal",
        # Dance/motion
        "bas_score": "motion",
        # Scene graph
        "dsg_score": "alignment",
        # Image LPIPS (FR perceptual distance)
        "image_lpips": "fr_quality",
        # Concept presence detection
        "concept_presence": "scene",
        "concept_count": "scene",
        "concept_face_count": "face",
    }

    def non_null_metrics(self) -> dict[str, object]:
        """Return only the metrics that were actually computed (non-None)."""
        return {k: v for k, v in self.model_dump().items() if v is not None}

    def non_null_count(self) -> int:
        """Count how many metrics were actually computed."""
        return sum(1 for v in self.model_dump().values() if v is not None)

    def to_grouped_dict(self) -> dict[str, dict[str, object]]:
        """Return non-null metrics organized by category.

        Returns a dict like::

            {
                "alignment": {"clip_score": 0.82, "blip_bleu": 0.45},
                "motion": {"flow_score": 3.2},
                "nr_quality": {"dover_score": 0.71},
                ...
            }

        Fields not mapped to a group appear under ``"other"``.
        """
        result: dict[str, dict[str, object]] = {}
        for field_name, value in self.model_dump().items():
            if value is None:
                continue
            group = self._FIELD_GROUPS.get(field_name, "other")
            result.setdefault(group, {})[field_name] = value
        return result

    def summary(self) -> str:
        """One-line summary: count of non-null metrics per group."""
        grouped = self.to_grouped_dict()
        parts = [f"{grp}={len(fields)}" for grp, fields in sorted(grouped.items())]
        total = self.non_null_count()
        return f"{total} metrics ({', '.join(parts)})" if parts else "0 metrics"

    # -- Fields -----------------------------------------------------------

    blur_score: Optional[float] = None  # Laplacian variance
    aesthetic_score: Optional[float] = None  # 0-100, normalized from aesthetic predictor
    clip_score: Optional[float] = None  # Caption-image alignment
    brightness: Optional[float] = None
    contrast: Optional[float] = None
    saturation: Optional[float] = None

    # Advanced metrics
    fast_vqa_score: Optional[float] = None  # 0-100
    motion_score: Optional[float] = None  # Scene motion intensity
    camera_motion_score: Optional[float] = None  # Camera motion intensity
    stabilized_motion_score: Optional[float] = None  # Stabilized scene motion (camera-invariant)
    stabilized_camera_score: Optional[float] = None  # Stabilized camera motion estimate
    technical_score: Optional[float] = None  # Composite technical score
    noise_score: Optional[float] = None
    artifacts_score: Optional[float] = None
    cpbd_score: Optional[float] = None  # CPBD perceptual blur detection (0-1, higher=sharper)
    imaging_noise_score: Optional[float] = None  # Imaging noise level (0-1, higher=cleaner)
    imaging_artifacts_score: Optional[float] = None  # Imaging edge-density artifacts (0-1, higher=cleaner)
    watermark_probability: Optional[float] = None  # 0-1
    ocr_area_ratio: Optional[float] = None  # 0-1
    face_count: Optional[int] = None
    nsfw_score: Optional[float] = None  # 0-1, likelihood of being NSFW
    perceptual_hash: Optional[str] = None  # dHash or similar
    auto_caption: Optional[str] = None  # Generated caption
    vqa_a_score: Optional[float] = None
    vqa_t_score: Optional[float] = None
    is_score: Optional[float] = None
    detection_diversity: Optional[float] = None  # Object detection category entropy
    sd_score: Optional[float] = None  # SD-reference similarity (0-1)
    gradient_detail: Optional[float] = None  # Sobel gradient detail (0-100)
    blip_bleu: Optional[float] = None
    detection_score: Optional[float] = None
    count_score: Optional[float] = None
    color_score: Optional[float] = None
    celebrity_id_score: Optional[float] = None
    identity_loss: Optional[float] = None  # Face identity cosine distance (0-1, lower=better)
    face_recognition_score: Optional[float] = None  # Face identity cosine similarity (0-1, higher=better)
    face_cross_similarity: Optional[float] = None  # Avg pairwise face similarity (0-1, higher=more consistent)
    face_identity_count: Optional[int] = None  # Number of unique identities detected
    ocr_score: Optional[float] = None
    ocr_fidelity: Optional[float] = None  # OCR text accuracy vs caption (0-100, higher=better)
    ocr_cer: Optional[float] = None  # Character Error Rate (0-1, lower=better)
    ocr_wer: Optional[float] = None  # Word Error Rate (0-1, lower=better)

    # Image-to-Video reference similarity (sliding-window)
    i2v_clip: Optional[float] = None  # CLIP image-video similarity (0-1)
    i2v_dino: Optional[float] = None  # DINOv2 image-video similarity (0-1)
    i2v_lpips: Optional[float] = None  # LPIPS image-video distance (0-1, lower=better)
    i2v_quality: Optional[float] = None  # Aggregated I2V quality (0-100)

    action_score: Optional[float] = None  # Caption-action fidelity (0-100)
    action_confidence: Optional[float] = None  # Top-1 action confidence (0-100)
    flow_score: Optional[float] = None
    motion_ac_score: Optional[float] = None
    warping_error: Optional[float] = None
    clip_temp: Optional[float] = None
    video_text_score: Optional[float] = None  # Video-text alignment via X-CLIP/CLIP (0-1)
    video_text_temporal: Optional[float] = None  # Video-text temporal consistency (0-1)
    face_consistency: Optional[float] = None
    spectral_entropy: Optional[float] = None  # DINOv2 spectral entropy
    spectral_rank: Optional[float] = None  # DINOv2 effective rank ratio

    # Enhanced perceptual metrics
    vmaf: Optional[float] = None  # VMAF (0-100, higher=better)
    ms_ssim: Optional[float] = None  # Multi-Scale SSIM (0-1)
    vif: Optional[float] = None  # Visual Information Fidelity
    niqe: Optional[float] = None  # Natural Image Quality Evaluator (lower=better)

    # Text-to-Video alignment
    t2v_score: Optional[float] = None  # T2VScore alignment + quality
    t2v_alignment: Optional[float] = None  # Text-video semantic alignment
    t2v_quality: Optional[float] = None  # Video production quality

    # Dynamics and motion
    dynamics_range: Optional[float] = None  # Extent of content variation
    dynamics_controllability: Optional[float] = None  # Motion control fidelity

    # Content quality
    scene_complexity: Optional[float] = None  # Visual complexity score
    compression_artifacts: Optional[float] = None  # Artifact severity (0-100)
    naturalness_score: Optional[float] = None  # Natural scene statistics
    video_memorability: Optional[float] = None  # Memorability prediction

    # Meta quality
    usability_rate: Optional[float] = None  # Percentage of usable frames
    confidence_score: Optional[float] = None  # Prediction confidence
    llm_qa_score: Optional[float] = None  # LMM descriptive quality rating (0-1)
    human_preference_score: Optional[float] = None
    engagement_score: Optional[float] = None

    # Format-specific
    hdr_quality: Optional[float] = None  # HDR-specific quality
    sdr_quality: Optional[float] = None  # SDR-specific quality

    # ITU-T P.910 temporal/spatial information
    temporal_information: Optional[float] = None  # ITU-T P.910 TI (higher=more motion)
    spatial_information: Optional[float] = None  # ITU-T P.910 SI (higher=more detail)

    # Temporal stability
    flicker_score: Optional[float] = None  # Flicker severity 0-100 (lower=better)
    judder_score: Optional[float] = None  # Judder severity 0-100 (lower=better)
    stutter_score: Optional[float] = None  # Duplicate/dropped frames 0-100 (lower=better)

    # Deep perceptual similarity (full-reference)
    dists: Optional[float] = None  # DISTS (0-1, lower=more similar)
    fsim: Optional[float] = None  # Feature Similarity Index (0-1, higher=better)
    gmsd: Optional[float] = None  # Gradient Magnitude Similarity Deviation (lower=better)
    vsi_score: Optional[float] = None  # Visual Saliency Index (0-1, higher=better)

    # No-reference perceptual quality
    brisque: Optional[float] = None  # BRISQUE (0-100, lower=better)

    # Audio quality metrics
    pesq_score: Optional[float] = None  # PESQ (-0.5 to 4.5, higher=better)
    estoi_score: Optional[float] = None  # ESTOI intelligibility (0-1, higher=better)
    mcd_score: Optional[float] = None  # Mel Cepstral Distortion (dB, lower=better)
    si_sdr_score: Optional[float] = None  # Scale-Invariant SDR (dB, higher=better)
    lpdist_score: Optional[float] = None  # Log-Power Spectral Distance (lower=better)
    utmos_score: Optional[float] = None  # UTMOS predicted MOS (1-5, higher=better)
    av_sync_offset: Optional[float] = None  # Audio-video sync offset in ms

    # SOTA no-reference VQA
    dover_score: Optional[float] = None  # DOVER overall (higher=better)
    dover_technical: Optional[float] = None  # DOVER technical quality
    dover_aesthetic: Optional[float] = None  # DOVER aesthetic quality
    internvqa_score: Optional[float] = None  # InternVQA video quality (higher=better)
    topiq_score: Optional[float] = None  # TOPIQ transformer-based IQA (higher=better)
    liqe_score: Optional[float] = None  # LIQE lightweight IQA (higher=better)
    clip_iqa_score: Optional[float] = None  # CLIP-IQA semantic quality (0-1, higher=better)

    # Professional production quality
    color_grading_score: Optional[float] = None  # Colour consistency 0-100
    white_balance_score: Optional[float] = None  # White balance accuracy 0-100
    exposure_consistency: Optional[float] = None  # Exposure stability 0-100
    focus_quality: Optional[float] = None  # Sharpness/focus quality 0-100
    banding_severity: Optional[float] = None  # Colour banding 0-100 (lower=better)

    # Vision-language quality assessment
    qalign_quality: Optional[float] = None  # Q-Align technical quality (1-5, higher=better)
    qalign_aesthetic: Optional[float] = None  # Q-Align aesthetic quality (1-5, higher=better)

    # Face quality metrics
    face_quality_score: Optional[float] = None  # Composite face quality 0-100 (higher=better)
    face_identity_consistency: Optional[float] = None  # Temporal face identity stability (0-1)
    face_expression_smoothness: Optional[float] = (
        None  # Expression smoothness 0-100 (higher=better)
    )
    face_landmark_jitter: Optional[float] = None  # Landmark jitter 0-100 (lower=better)

    # Semantic consistency metrics
    object_permanence_score: Optional[float] = (
        None  # Object tracking consistency 0-100 (higher=better)
    )
    semantic_consistency: Optional[float] = None  # Segmentation temporal IoU 0-1 (higher=better)
    depth_temporal_consistency: Optional[float] = None  # Depth map correlation 0-1 (higher=better)
    subject_consistency: Optional[float] = None  # Subject identity consistency (0-1, higher=better)
    background_consistency: Optional[float] = (
        None  # Background stability consistency (0-1, higher=better)
    )
    motion_smoothness: Optional[float] = None  # Motion smoothness (0-1, higher=better)

    # Codec-specific metrics
    codec_efficiency: Optional[float] = None  # Quality-per-bit efficiency 0-100 (higher=better)
    gop_quality: Optional[float] = None  # GOP structure appropriateness 0-100 (higher=better)
    codec_artifacts: Optional[float] = None  # Block artifact severity 0-100 (lower=better)

    # Ethical and safety metrics
    deepfake_probability: Optional[float] = None  # Synthetic/deepfake likelihood 0-1
    ai_generated_probability: Optional[float] = None  # AI-generated content likelihood 0-1
    harmful_content_score: Optional[float] = None  # Violence/gore severity 0-1
    watermark_strength: Optional[float] = None  # Invisible watermark strength 0-1
    bias_score: Optional[float] = None  # Representation imbalance indicator 0-1

    # 3D and spatial quality
    depth_quality: Optional[float] = None  # Depth map quality 0-100 (higher=better)
    multiview_consistency: Optional[float] = None  # Geometric consistency 0-1 (higher=better)
    stereo_comfort_score: Optional[float] = None  # Stereo viewing comfort 0-100 (higher=better)

    # Additional SOTA IQA/VQA
    musiq_score: Optional[float] = None  # MUSIQ multi-scale IQA (higher=better)
    contrique_score: Optional[float] = None  # CONTRIQUE contrastive IQA (higher=better)
    mdtvsfa_score: Optional[float] = None  # MDTVSFA fragment-based VQA (higher=better)

    # pyiqa NR-IQA (no-reference)
    nima_score: Optional[float] = None  # NIMA aesthetic+technical (1-10, higher=better)
    dbcnn_score: Optional[float] = None  # DBCNN bilinear CNN (higher=better)
    wadiqam_score: Optional[float] = None  # WaDIQaM-NR (higher=better)
    maniqa_score: Optional[float] = None  # MANIQA multi-attention (higher=better)
    arniqa_score: Optional[float] = None  # ARNIQA (higher=better)
    qualiclip_score: Optional[float] = None  # QualiCLIP opinion-unaware (higher=better)

    # pyiqa FR-IQA (full-reference)
    pieapp: Optional[float] = None  # PieAPP pairwise preference (lower=better)
    cw_ssim: Optional[float] = None  # Complex Wavelet SSIM (0-1, higher=better)
    nlpd: Optional[float] = None  # Normalized Laplacian Pyramid Distance (lower=better)
    mad: Optional[float] = None  # Most Apparent Distortion (lower=better)
    ahiq: Optional[float] = None  # Attention Hybrid IQA (higher=better)
    topiq_fr: Optional[float] = None  # TOPIQ full-reference (higher=better)

    # Foundation model perceptual similarity
    dreamsim: Optional[float] = None  # DreamSim CLIP+DINO similarity (lower=more similar)

    # Comprehensive video quality (COVER)
    cover_score: Optional[float] = None  # COVER overall (higher=better)
    cover_technical: Optional[float] = None  # COVER technical branch
    cover_aesthetic: Optional[float] = None  # COVER aesthetic branch
    cover_semantic: Optional[float] = None  # COVER semantic branch

    # Text-visual alignment
    vqa_score_alignment: Optional[float] = (
        None  # VQAScore text-visual alignment (0-1, higher=better)
    )

    # VideoScore multi-dimensional (1-4 scale)
    videoscore_visual: Optional[float] = None  # VideoScore visual quality
    videoscore_temporal: Optional[float] = None  # VideoScore temporal consistency
    videoscore_dynamic: Optional[float] = None  # VideoScore dynamic degree
    videoscore_alignment: Optional[float] = None  # VideoScore text-video alignment
    videoscore_factual: Optional[float] = None  # VideoScore factual consistency
    videoscore2_visual: Optional[float] = None  # VideoScore2 visual quality
    videoscore2_alignment: Optional[float] = None  # VideoScore2 text-video alignment
    videoscore2_physical: Optional[float] = None  # VideoScore2 physical/common-sense consistency

    # Face-specific IQA
    face_iqa_score: Optional[float] = None  # TOPIQ-face face quality (higher=better)

    # Scene stability (TransNetV2 / heuristic)
    scene_stability: Optional[float] = (
        None  # Scene stability score (0-1, 1=single continuous scene)
    )
    avg_scene_duration: Optional[float] = None  # Average scene duration in seconds

    # RAFT optical flow (Data-Juicer)
    raft_motion_score: Optional[float] = None  # RAFT optical flow magnitude

    # RAM tagging (Data-Juicer)
    ram_tags: Optional[str] = None  # Comma-separated RAM auto-tags

    # Depth Anything (Data-Juicer)
    depth_anything_score: Optional[float] = None  # Monocular depth quality
    depth_anything_consistency: Optional[float] = None  # Temporal depth consistency

    # Video type classifier (NVIDIA Curator)
    video_type: Optional[str] = None  # Content type (real, animated, game, etc.)
    video_type_confidence: Optional[float] = None  # Classification confidence

    # TRAJAN (ICLR 2025)
    trajan_score: Optional[float] = None  # Point track motion consistency

    # PromptIQA (ECCV 2024)
    promptiqa_score: Optional[float] = None  # Few-shot NR-IQA score

    # AIGV-Assessor (CVPR 2025)
    aigv_static: Optional[float] = None  # AI video static quality
    aigv_temporal: Optional[float] = None  # AI video temporal smoothness
    aigv_dynamic: Optional[float] = None  # AI video dynamic degree
    aigv_alignment: Optional[float] = None  # AI video text-video alignment

    # VideoAlign reward (NeurIPS 2025)
    video_reward_score: Optional[float] = None  # Human preference reward

    # TIFA (ICCV 2023) — Text-to-Image Faithfulness Assessment
    tifa_score: Optional[float] = None  # VQA faithfulness (0-1, higher=better)

    # ImageReward (human preference for text-to-image)
    image_reward_score: Optional[float] = None  # Human preference reward (-2..+2, higher=better)
    pickscore_score: Optional[float] = None  # PickScore prompt-image preference score (higher=better)
    hpsv3_score: Optional[float] = None  # HPSv3 human preference reward mu (higher=better)
    chipqa_score: Optional[float] = None  # ChipQA space-time-chip NR-VQA (higher=better)

    # Text overlay (NVIDIA Curator)
    text_overlay_score: Optional[float] = None  # Text overlay severity (0-1)

    # ptlflow optical flow (Data-Juicer)
    ptlflow_motion_score: Optional[float] = None  # ptlflow optical flow magnitude

    # QCN (CVPR 2024)
    qcn_score: Optional[float] = None  # Geometric order blind IQA

    # Video-native VQA
    finevq_score: Optional[float] = None  # FineVQ fine-grained UGC VQA (CVPR 2025)
    kvq_score: Optional[float] = None  # KVQ saliency-guided VQA (CVPR 2025)
    rqvqa_score: Optional[float] = None  # RQ-VQA rich quality-aware (CVPR 2024 winner)
    videval_score: Optional[float] = None  # VIDEVAL 60-feature fusion NR-VQA
    tlvqm_score: Optional[float] = None  # TLVQM two-level video quality
    funque_score: Optional[float] = None  # FUNQUE unified quality (beats VMAF)
    movie_score: Optional[float] = None  # MOVIE motion trajectory FR
    st_greed_score: Optional[float] = None  # ST-GREED variable frame rate FR
    c3dvqa_score: Optional[float] = None  # C3DVQA 3D CNN spatiotemporal FR
    flolpips: Optional[float] = None  # FloLPIPS flow-based perceptual FR
    hdr_vqm: Optional[float] = None  # HDR-VQM HDR video quality FR
    hdr_chipqa_score: Optional[float] = None  # HDR-ChipQA HDR NR-VQA (higher=better)
    hdrmax_score: Optional[float] = None  # HDRMAX / HDR-VMAF family score (higher=better)
    brightrate_score: Optional[float] = None  # BrightRate HDR UGC NR-VQA (higher=better)
    st_lpips: Optional[float] = None  # ST-LPIPS spatiotemporal perceptual FR

    # Video curation signals
    camera_jitter_score: Optional[float] = None  # Camera stability (0-1, 1=stable)
    jump_cut_score: Optional[float] = None  # Jump cut absence (0-1, 1=no cuts)
    playback_speed_score: Optional[float] = None  # Normal speed (1.0=normal)
    flow_coherence: Optional[float] = None  # Bidirectional optical flow consistency (0-1)
    letterbox_ratio: Optional[float] = None  # Border/letterbox fraction (0-1, 0=no borders)
    tonal_dynamic_range: Optional[float] = None  # Luminance histogram span (0-100)
    vtss: Optional[float] = None  # Video Training Suitability Score (0-1)

    # Image IQA (keyframe-level)
    cnniqa_score: Optional[float] = None  # CNNIQA blind CNN IQA
    hyperiqa_score: Optional[float] = None  # HyperIQA adaptive NR-IQA
    paq2piq_score: Optional[float] = None  # PaQ-2-PiQ patch-to-picture (CVPR 2020)
    tres_score: Optional[float] = None  # TReS transformer IQA (WACV 2022)
    unique_score: Optional[float] = None  # UNIQUE unified NR-IQA (TIP 2021)
    laion_aesthetic: Optional[float] = None  # LAION Aesthetics V2 (0-10)
    aesthetic_mlp_score: Optional[float] = None  # LAION Aesthetics MLP (1-10)
    compare2score: Optional[float] = None  # Compare2Score comparison-based
    afine_score: Optional[float] = None  # A-FINE fidelity-naturalness (CVPR 2025)
    ckdn_score: Optional[float] = None  # CKDN knowledge distillation FR
    deepwsd_score: Optional[float] = None  # DeepWSD Wasserstein distance FR

    # Compression/rendering perceptual metrics
    ssimulacra2: Optional[float] = None  # SSIMULACRA 2 (0-100, lower=better, JPEG XL standard)
    butteraugli: Optional[float] = None  # Butteraugli perceptual distance (lower=better)
    flip_score: Optional[float] = None  # NVIDIA FLIP perceptual metric (0-1, lower=better)
    vmaf_neg: Optional[float] = None  # VMAF NEG (no enhancement gain, 0-100, higher=better)

    # pyiqa NR-IQA (no-reference)
    ilniqe: Optional[float] = None  # IL-NIQE Integrated Local NIQE (lower=better)
    nrqm: Optional[float] = None  # NRQM No-Reference Quality Metric (higher=better)
    pi_score: Optional[float] = None  # Perceptual Index (PIRM challenge, lower=better)
    piqe: Optional[float] = None  # PIQE perception-based NR-IQA (lower=better)
    maclip_score: Optional[float] = None  # MACLIP multi-attribute CLIP NR-IQA (higher=better)

    # pyiqa FR-IQA (full-reference)
    dmm: Optional[float] = None  # DMM Detail Model Metric FR (higher=better)
    wadiqam_fr: Optional[float] = None  # WaDIQaM full-reference (higher=better)
    ssimc: Optional[float] = None  # Complex Wavelet SSIM-C FR (higher=better)

    # FFmpeg-based metrics
    cambi: Optional[float] = None  # CAMBI banding index (0-24, lower=better)
    xpsnr: Optional[float] = None  # XPSNR perceptual PSNR (dB, higher=better)
    vmaf_phone: Optional[float] = None  # VMAF phone model (0-100, higher=better)
    vmaf_4k: Optional[float] = None  # VMAF 4K model (0-100, higher=better)

    # Audio quality
    visqol: Optional[float] = None  # ViSQOL audio quality MOS (1-5, higher=better)
    dnsmos_overall: Optional[float] = None  # DNSMOS overall MOS (1-5, higher=better)
    dnsmos_sig: Optional[float] = None  # DNSMOS signal quality (1-5, higher=better)
    dnsmos_bak: Optional[float] = None  # DNSMOS background quality (1-5, higher=better)

    # HDR metrics
    pu_psnr: Optional[float] = None  # PU-PSNR perceptually uniform HDR (dB, higher=better)
    pu_ssim: Optional[float] = None  # PU-SSIM perceptually uniform HDR (0-1, higher=better)
    max_fall: Optional[float] = None  # MaxFALL frame average light level (nits)
    max_cll: Optional[float] = None  # MaxCLL content light level (nits)
    hdr_vdp: Optional[float] = None  # HDR-VDP visual difference predictor (higher=better)
    delta_ictcp: Optional[float] = None  # Delta ICtCp HDR color difference (lower=better)
    hdr_technical_score: Optional[float] = None  # HDR/SDR-aware technical quality (0-1)

    # Color, codec, gaming, streaming
    ciede2000: Optional[float] = None  # CIEDE2000 perceptual color difference (lower=better)
    psnr_hvs: Optional[float] = None  # PSNR-HVS perceptually weighted (dB, higher=better)
    psnr_hvs_m: Optional[float] = None  # PSNR-HVS-M with masking (dB, higher=better)
    cgvqm: Optional[float] = None  # CGVQM gaming quality (higher=better)
    strred: Optional[float] = None  # STRRED reduced-reference temporal (lower=better)
    p1203_mos: Optional[float] = None  # ITU-T P.1203 streaming QoE MOS (1-5)

    # NeMo Curator text quality
    nemo_quality_score: Optional[float] = None  # Caption text quality (0-1)
    nemo_quality_label: Optional[str] = None  # Quality label (Low/Medium/High)

    # VBench-2.0 faithfulness (arXiv:2503.21755)
    human_fidelity_score: Optional[float] = None  # Body/hand/face quality (0-1, higher=better)
    physics_score: Optional[float] = None  # Physics plausibility (0-1, higher=better)
    commonsense_score: Optional[float] = None  # Common sense adherence (0-1, higher=better)
    creativity_score: Optional[float] = None  # Artistic novelty (0-1, higher=better)

    # ChronoMagic-Bench (NeurIPS 2024, arXiv:2406.18522)
    chronomagic_mt_score: Optional[float] = None  # Metamorphic temporal (0-1, higher=better)
    chronomagic_ch_score: Optional[float] = None  # Chrono-hallucination (0-1, lower=fewer)

    # T2V-CompBench (CVPR 2025)
    compbench_attribute: Optional[float] = None  # Attribute binding (0-1)
    compbench_object_rel: Optional[float] = None  # Object relationship (0-1)
    compbench_action: Optional[float] = None  # Action binding (0-1)
    compbench_spatial: Optional[float] = None  # Spatial relationship (0-1)
    compbench_numeracy: Optional[float] = None  # Generative numeracy (0-1)
    compbench_scene: Optional[float] = None  # Scene composition (0-1)
    compbench_overall: Optional[float] = None  # Overall composition (0-1)

    # NR-VQA (new models, 2023-2025)
    rapique_score: Optional[float] = None  # RAPIQUE bandpass+CNN NR-VQA (higher=better)
    conviqt_score: Optional[float] = None  # CONVIQT contrastive NR-VQA (higher=better)
    stablevqa_score: Optional[float] = None  # StableVQA video stability (higher=better)
    maxvqa_score: Optional[float] = None  # MaxVQA explainable quality (higher=better)
    bvqi_score: Optional[float] = None  # BVQI zero-shot blind VQA (higher=better)
    modularbvqa_score: Optional[float] = None  # ModularBVQA resolution-aware (higher=better)
    ptmvqa_score: Optional[float] = None  # PTM-VQA multi-PTM fusion (higher=better)
    clipvqa_score: Optional[float] = None  # CLIPVQA CLIP-based VQA (higher=better)
    discovqa_score: Optional[float] = None  # DisCoVQA distortion-content (higher=better)
    zoomvqa_score: Optional[float] = None  # Zoom-VQA multi-level (higher=better)
    faver_score: Optional[float] = None  # FAVER variable frame rate (higher=better)
    siamvqa_score: Optional[float] = None  # SiamVQA Siamese high-res (higher=better)
    memoryvqa_score: Optional[float] = None  # Memory-VQA human memory (higher=better)
    sama_score: Optional[float] = None  # SAMA scaling+masking (higher=better)
    clifvqa_score: Optional[float] = None  # CLiF-VQA human feelings (higher=better)
    simplevqa_score: Optional[float] = None  # SimpleVQA Swin+SlowFast (higher=better)
    adadqa_score: Optional[float] = None  # Ada-DQA adaptive diverse (higher=better)
    mdvqa_semantic: Optional[float] = None  # MD-VQA semantic quality (higher=better)
    mdvqa_distortion: Optional[float] = None  # MD-VQA distortion quality (higher=better)
    mdvqa_motion: Optional[float] = None  # MD-VQA motion quality (higher=better)

    # FR-VQA (new models)
    rankdvqa_score: Optional[float] = None  # RankDVQA ranking-based FR (higher=better)
    compressed_vqa_hdr: Optional[float] = None  # CompressedVQA-HDR (higher=better)
    deepvqa_score: Optional[float] = None  # DeepVQA spatiotemporal FR (higher=better)
    st_mad: Optional[float] = None  # ST-MAD spatiotemporal MAD (lower=better)
    avqt_score: Optional[float] = None  # Apple AVQT perceptual (higher=better)
    pvmaf_score: Optional[float] = None  # pVMAF predictive VMAF (0-100)
    sr4kvqa_score: Optional[float] = None  # SR4KVQA super-resolution 4K (higher=better)

    # AIGC-specific VQA
    crave_score: Optional[float] = None  # CRAVE next-gen AIGC (higher=better)
    aigcvqa_technical: Optional[float] = None  # AIGC-VQA technical branch
    aigcvqa_aesthetic: Optional[float] = None  # AIGC-VQA aesthetic branch
    aigcvqa_alignment: Optional[float] = None  # AIGC-VQA text-video alignment
    ugvq_score: Optional[float] = None  # UGVQ unified generated VQ (higher=better)
    aigvqa_score: Optional[float] = None  # AIGVQA multi-dimensional (higher=better)
    t2veval_score: Optional[float] = None  # T2VEval consistency+realness (higher=better)
    world_consistency_score: Optional[float] = None  # WCS object permanence (higher=better)

    # LLM/VLM-based VQA
    vqa2_score: Optional[float] = None  # VQA² LMM quality (higher=better)
    lmmvqa_score: Optional[float] = None  # LMM-VQA spatiotemporal (higher=better)
    vqinsight_score: Optional[float] = None  # VQ-Insight ByteDance (higher=better)
    vqathinker_score: Optional[float] = None  # VQAThinker GRPO (higher=better)
    qclip_score: Optional[float] = None  # Q-CLIP VLM-based (higher=better)
    presresq_score: Optional[float] = None  # PreResQ-R1 rank+score (higher=better)

    umtscore: Optional[float] = None  # UMTScore video-text alignment

    # Video reward models
    videoreward_vq: Optional[float] = None  # VideoReward visual quality
    videoreward_mq: Optional[float] = None  # VideoReward motion quality
    videoreward_ta: Optional[float] = None  # VideoReward text alignment
    vader_score: Optional[float] = None  # VADER reward alignment

    # 360/VR spherical metrics
    s_psnr: Optional[float] = None  # Spherical PSNR (dB, higher=better)
    ws_psnr: Optional[float] = None  # Weighted Spherical PSNR (dB, higher=better)
    cpp_psnr: Optional[float] = None  # Craster Parabolic PSNR (dB, higher=better)
    ws_ssim: Optional[float] = None  # Weighted Spherical SSIM (0-1, higher=better)
    mc360iqa_score: Optional[float] = None  # MC360IQA blind 360 (higher=better)
    provqa_score: Optional[float] = None  # ProVQA progressive 360 (higher=better)

    # Point cloud quality
    pc_d1_psnr: Optional[float] = None  # Point-to-point PSNR (dB)
    pc_d2_psnr: Optional[float] = None  # Point-to-plane PSNR (dB)
    pcqm_score: Optional[float] = None  # PCQM geometry+color (higher=better)
    graphsim_score: Optional[float] = None  # GraphSIM gradient (higher=better)
    pointssim_score: Optional[float] = None  # PointSSIM structural (higher=better)
    mm_pcqa_score: Optional[float] = None  # MM-PCQA multi-modal (higher=better)

    # Streaming QoE
    p1204_mos: Optional[float] = None  # ITU-T P.1204.3 bitstream MOS (1-5)
    sqi_score: Optional[float] = None  # SQI streaming quality index
    video_atlas_score: Optional[float] = None  # Video ATLAS temporal artifacts

    # Face quality (recognition-aware)
    serfiq_score: Optional[float] = None  # SER-FIQ embedding robustness (higher=better)
    crfiqa_score: Optional[float] = None  # CR-FIQA classifiability (higher=better)
    magface_score: Optional[float] = None  # MagFace magnitude quality (higher=better)
    grafiqs_score: Optional[float] = None  # GraFIQs gradient-based (higher=better)

    # Niche domains
    uiqm_score: Optional[float] = None  # UIQM underwater quality (higher=better)
    uciqe_score: Optional[float] = None  # UCIQE underwater color (higher=better)
    oavqa_score: Optional[float] = None  # OAVQA omnidirectional AV (higher=better)

    # Classic NR-VQA (missed from first round)
    viideo_score: Optional[float] = None  # VIIDEO blind natural video statistics (lower=better)
    vbliinds_score: Optional[float] = None  # V-BLIINDS DCT-domain NSS (higher=better)
    vsfa_score: Optional[float] = None  # VSFA quality-aware feature aggregation (higher=better)
    speedqa_score: Optional[float] = None  # SpEED-QA entropic differencing (higher=better)
    gamival_score: Optional[float] = None  # GAMIVAL cloud gaming NR-VQA (higher=better)
    nr_gvqm_score: Optional[float] = None  # NR-GVQM cloud gaming VQA (higher=better)

    # Task-specific FR metrics
    erqa_score: Optional[float] = None  # ERQA edge restoration quality (0-1, higher=better)
    vfips_score: Optional[float] = None  # VFIPS frame interpolation perceptual (lower=better)
    artfid_score: Optional[float] = None  # ArtFID style transfer quality (lower=better)
    psnr_div: Optional[float] = None  # PSNR_DIV motion-weighted PSNR (dB, higher=better)
    psnr99: Optional[float] = None  # PSNR99 worst-case region quality (dB, higher=better)

    # pyiqa built-ins
    deepdc_score: Optional[float] = None  # DeepDC distribution conformance (lower=better)

    # Audio aesthetics
    audiobox_production: Optional[float] = None  # Audiobox production quality
    audiobox_enjoyment: Optional[float] = None  # Audiobox content enjoyment
    song_eval_coherence: Optional[float] = None  # SongEval overall coherence (1-5, higher=better)
    song_eval_musicality: Optional[float] = None  # SongEval overall musicality (1-5, higher=better)
    song_eval_memorability: Optional[float] = None  # SongEval memorability (1-5, higher=better)
    song_eval_clarity: Optional[float] = None  # SongEval clarity of song structure (1-5, higher=better)
    song_eval_naturalness: Optional[float] = None  # SongEval vocal breathing/phrasing naturalness (1-5, higher=better)

    # Talking head / lip sync
    thqa_score: Optional[float] = None  # THQA talking head quality (higher=better)
    lse_d: Optional[float] = None  # LSE-D lip sync error distance (lower=better)
    lse_c: Optional[float] = None  # LSE-C lip sync error confidence (higher=better)

    # Video segmentation
    davis_j: Optional[float] = None  # DAVIS J region similarity IoU (higher=better)
    davis_f: Optional[float] = None  # DAVIS F boundary accuracy (higher=better)

    # Video colorization
    cdc_score: Optional[float] = None  # CDC color distribution consistency (lower=better)

    # Dance/motion generation
    bas_score: Optional[float] = None  # BAS beat alignment score (higher=better)

    # Scene graph faithfulness
    dsg_score: Optional[float] = None  # DSG Davidsonian Scene Graph (higher=better)

    # Image LPIPS (FR perceptual distance)
    image_lpips: Optional[float] = None  # LPIPS perceptual distance vs reference (0-1, lower=more similar)

    # Concept presence detection
    concept_presence: Optional[float] = None  # Concept presence confidence (0-1, higher=more confident)
    concept_count: Optional[int] = None  # Number of detected instances of target concept
    concept_face_count: Optional[int] = None  # Number of faces detected


class Sample(BaseModel):
    """A single sample (video/image) in the dataset."""

    path: Path
    is_video: bool
    reference_path: Optional[Path] = None
    video_metadata: Optional[VideoMetadata] = None
    image_metadata: Optional[ImageMetadata] = None
    audio_metadata: Optional[AudioMetadata] = None
    caption: Optional[CaptionMetadata] = None
    quality_metrics: Optional[QualityMetrics] = None
    validation_issues: List[ValidationIssue] = Field(default_factory=list)
    detections: List[Dict[str, Any]] = Field(
        default_factory=list
    )  # [{'label': 'person', 'box': [x,y,w,h], 'conf': 0.9}, ...]
    embedding: Optional[List[float]] = None  # X-CLIP embedding

    @property
    def is_valid(self) -> bool:
        """Check if sample has no ERROR-level issues."""
        return not any(
            issue.severity == ValidationSeverity.ERROR for issue in self.validation_issues
        )

    @property
    def width(self) -> Optional[int]:
        """Get width from appropriate metadata."""
        if self.video_metadata:
            return self.video_metadata.width
        if self.image_metadata:
            return self.image_metadata.width
        return None

    @property
    def height(self) -> Optional[int]:
        """Get height from appropriate metadata."""
        if self.video_metadata:
            return self.video_metadata.height
        if self.image_metadata:
            return self.image_metadata.height
        return None

    @property
    def aspect_ratio(self) -> Optional[float]:
        """Calculate aspect ratio."""
        if self.width is not None and self.height is not None:
            return self.width / self.height
        return None

    def load_image(self) -> Any:
        """Load an image array for either image files or a representative video frame."""
        import cv2

        if self.is_video:
            cap = cv2.VideoCapture(str(self.path))
            if not cap.isOpened():
                return None
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
            ok, frame = cap.read()
            cap.release()
            return frame if ok else None

        return cv2.imread(str(self.path))


class DatasetStats(BaseModel):
    """Aggregated statistics for the entire dataset."""

    total_samples: int
    valid_samples: int
    invalid_samples: int
    total_size: int  # bytes

    # Aggregated metrics (averages)
    avg_technical_score: Optional[float] = None
    avg_aesthetic_score: Optional[float] = None
    avg_motion_score: Optional[float] = None
    usability_ratio: Optional[float] = None
    size_distribution: Optional[Dict[str, int]] = None
    duration_distribution: Optional[Dict[str, int]] = None

    # Issue distribution
    issues_by_type: Dict[str, int] = Field(default_factory=dict)
    severity_distribution: Dict[str, int] = Field(default_factory=dict)
    resolution_distribution: Dict[str, int] = Field(default_factory=dict)
    aspect_ratio_distribution: Dict[str, int] = Field(default_factory=dict)
    format_distribution: Dict[str, int] = Field(default_factory=dict)
    avg_quality_score: Optional[float] = None
    quality_score_distribution: Dict[str, int] = Field(default_factory=dict)

    # Distribution metrics (dataset-level)
    fvd: Optional[float] = None  # Fréchet Video Distance
    kvd: Optional[float] = None  # Kernel Video Distance
    fvmd: Optional[float] = None  # Fréchet Video Motion Distance
    fid: Optional[float] = None  # Fréchet Inception Distance
    jedi: Optional[float] = None  # JEDi (V-JEPA + MMD, ICLR 2025)
    kid: Optional[float] = None  # Kernel Inception Distance (lower=better)
    kid_std: Optional[float] = None  # KID standard deviation

    # Generative distribution metrics (dataset-level)
    precision: Optional[float] = None  # Quality of generated samples (0-1)
    recall: Optional[float] = None  # Coverage of real distribution (0-1)
    coverage: Optional[float] = None  # Diversity of generated samples (0-1)
    density: Optional[float] = None  # Concentration around real samples

    # Dataset-level analytics
    diversity_score: Optional[float] = None  # Visual diversity 0-1 (higher=more diverse)
    semantic_coverage: Optional[float] = None  # Embedding space coverage 0-1
    outlier_count: Optional[int] = None  # Number of statistical outliers
    class_balance_score: Optional[float] = None  # Category balance 0-1 (higher=balanced)
    duplicate_pairs: Optional[int] = None  # Count of near-duplicate pairs

    # Face cross-similarity (dataset-level)
    face_similarity_matrix: Optional[List[List[float]]] = None  # NxN pairwise similarity
    avg_face_cross_similarity: Optional[float] = None  # Dataset-level average
    identity_cluster_count: Optional[int] = None  # Number of identity clusters

    # UMAP projection (dataset-level)
    umap_spread: Optional[float] = None  # UMAP projection spread
    umap_coverage: Optional[float] = None  # UMAP projection coverage (0-1)

    # Batch distribution metrics (dataset-level)
    fad: Optional[float] = None  # Frechet Audio Distance (lower=better)
    fgd: Optional[float] = None  # Frechet Gesture Distance (lower=better)
    fmd: Optional[float] = None  # Frechet Motion Distance (lower=better)
    msswd: Optional[float] = None  # Multi-Scale Sliced Wasserstein (lower=better)
    sfid: Optional[float] = None  # Spatial FID (lower=better)
    vendi: Optional[float] = None  # Vendi Score diversity (higher=better)
    stream_spatial: Optional[float] = None  # STREAM spatial fidelity+diversity
    stream_temporal: Optional[float] = None  # STREAM temporal naturalness
    worldscore: Optional[float] = None  # WorldScore generation quality

    # Codec comparison (dataset-level)
    bd_rate: Optional[float] = None  # BD-Rate compression efficiency (%, negative=better)
    bd_psnr: Optional[float] = None  # BD-PSNR quality delta (dB, positive=better)

    # Image LPIPS diversity (dataset-level)
    lpips_diversity: Optional[float] = None  # Average pairwise LPIPS across dataset (higher=more diverse)

    # Verse-Bench benchmark (dataset-level)
    verse_bench_overall: Optional[float] = None  # Official Verse-Bench final score
    verse_bench_metrics: Optional[Dict[str, float]] = None  # Raw Verse-Bench component metrics
    verse_bench_breakdown: Optional[Dict[str, float]] = None  # Verse-Bench subscores and overall


