"""Data models for Ayase using Pydantic."""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationIssue(BaseModel):
    """A validation issue found in a sample."""

    severity: ValidationSeverity
    message: str
    # Structured, machine-stable category for aggregation (e.g. "module_error",
    # "too_dark"). Preferred over parsing ``message`` for ``issues_by_type``.
    # Optional for backward compatibility; falls back to a safe message prefix.
    issue_type: Optional[str] = None
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

    # -- Field grouping registry (field name -> category) ------------------
    # Empty by design: every metric's category is declared on its producing
    # module via a `metric_groups` attribute and folded in here by
    # register_field_groups() at discovery time. Adding a self-describing
    # module needs no edit to this file.
    _FIELD_GROUPS: ClassVar[Dict[str, str]] = {}

    # Fields that carry provenance/bookkeeping rather than a computed metric.
    # They are excluded from the metric-view helpers (counts, grouping, summary)
    # so adding them does not inflate metric statistics.
    _NON_METRIC_FIELDS: ClassVar[frozenset] = frozenset({"metric_backends"})

    def non_null_metrics(self) -> dict[str, object]:
        """Return only the metrics that were actually computed (non-None)."""
        return {
            k: v
            for k, v in self.model_dump().items()
            if v is not None and k not in self._NON_METRIC_FIELDS
        }

    def non_null_count(self) -> int:
        """Count how many metrics were actually computed."""
        return sum(
            1
            for k, v in self.model_dump().items()
            if v is not None and k not in self._NON_METRIC_FIELDS
        )

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
            if value is None or field_name in self._NON_METRIC_FIELDS:
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

    @classmethod
    def register_field_groups(cls, mapping: Dict[str, str]) -> None:
        """Merge module-declared metric→group mappings into the registry.

        Modules own the grouping of the metrics they produce by declaring a
        ``metric_groups`` class attribute; ``ModuleRegistry.discover_modules``
        calls this for each one so the built-in ``_FIELD_GROUPS`` table need
        not be touched when a self-describing module is added. Module
        declarations win over the defaults, and a conflicting re-map is logged.
        """
        for field, group in mapping.items():
            existing = cls._FIELD_GROUPS.get(field)
            if existing is not None and existing != group:
                logger.warning(
                    "metric group for %r overridden: %r -> %r", field, existing, group
                )
            cls._FIELD_GROUPS[field] = group

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
    kandinsky_camera_motion_score: Optional[float] = None  # Kandinsky camera motion prediction
    kandinsky_object_motion_score: Optional[float] = None  # Kandinsky object motion prediction
    kandinsky_dynamics_score: Optional[float] = None  # Kandinsky dynamics prediction
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
    auto_caption: Optional[str] = None  # Generated caption
    vqa_a_score: Optional[float] = None
    vqa_t_score: Optional[float] = None
    is_score: Optional[float] = None
    detection_diversity: Optional[float] = None  # Object detection category entropy
    sd_score: Optional[float] = None  # SD-reference similarity (0-1)
    gradient_detail: Optional[float] = None  # Sobel gradient detail (0-100)
    blip_score: Optional[float] = None  # BLIP image-text matching score (0-1, higher=better)
    blip_bleu: Optional[float] = None
    detection_score: Optional[float] = None
    count_score: Optional[float] = None
    person_count: Optional[int] = None  # Peak number of 'person' detections in a single frame (crowd size)
    person_count_score: Optional[float] = None  # Normalized crowd/person-count score (0-100, saturates at 10/frame)
    color_score: Optional[float] = None
    celebrity_id_score: Optional[float] = None
    identity_loss: Optional[float] = None  # Face identity cosine distance (0-1, lower=better)
    face_recognition_score: Optional[float] = None  # Face identity cosine similarity (0-1, higher=better)
    clip_image_similarity: Optional[float] = None  # CLIP image-to-image cosine similarity vs reference (0-1, higher=better)
    face_cross_similarity: Optional[float] = None  # Avg pairwise face similarity (0-1, higher=more consistent)
    face_identity_count: Optional[int] = None  # Number of unique identities detected
    dino_face_identity: Optional[float] = None  # DINOv2 face identity cosine similarity (0-1, higher=better)
    dino_face_identity_max: Optional[float] = None  # Max DINOv2 face identity across frames (0-1, higher=better)
    adaface_identity_similarity: Optional[float] = None  # AdaFace cosine similarity vs reference face (0-1, higher=better)
    expression_following: Optional[float] = None  # Driver-expression fidelity (0-1, higher=better)
    expression_following_distance: Optional[float] = None  # Mean blendshape L1 distance (0-1, lower=better)
    expression_following_coverage: Optional[float] = None  # Joint valid-face coverage (0-1)
    expression_similarity: Optional[float] = None  # Time-free expression-manner similarity (0-1, higher=better)
    expression_similarity_distribution: Optional[float] = None  # Expression-repertoire agreement (0-1)
    expression_similarity_coactivation: Optional[float] = None  # Correlation-structure agreement (0-1)
    expression_similarity_dynamics: Optional[float] = None  # Change-rate agreement (0-1)
    expression_similarity_range_ratio: Optional[float] = None  # Expressive spread, sample/reference (1.0=equal)
    expression_similarity_coverage: Optional[float] = None  # Lower per-video valid-face coverage (0-1)
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
    utmos_v2_score: Optional[float] = None  # UTMOSv2 predicted MOS (1-5, higher=better)
    asr_cer: Optional[float] = None  # ASR character error rate vs reference text (0-1, lower=better)
    asr_wer: Optional[float] = None  # ASR word error rate vs reference text (0-1, lower=better)
    scoreq_score: Optional[float] = None  # SCOREQ speech naturalness score (0-1, higher=better)
    ttsds2_score: Optional[float] = None  # TTSDS2 speech quality score (0-1, higher=better)
    human_clap_score: Optional[float] = None  # Human-CLAP audio-text relevance (0-1, higher=better)
    laion_clap_score: Optional[float] = None  # LAION-CLAP audio-text relevance (0-1, higher=better)
    ms_clap_score: Optional[float] = None  # Microsoft CLAP audio-text relevance (0-1, higher=better)
    clap_score: Optional[float] = None  # Generic CLAP audio-text relevance (0-1, higher=better)
    imagebind_score: Optional[float] = None  # ImageBind audio-text relevance (0-1, higher=better)
    pam_score: Optional[float] = None  # PAM anti-prompt perceptual audio quality (0-1, higher=better)
    aqascore_score: Optional[float] = None  # AQAScore audio question-answering alignment (0-1)
    av_sync_offset: Optional[float] = None  # Audio-video sync offset in ms

    # No-reference VQA
    dover_score: Optional[float] = None  # DOVER overall (higher=better)
    uvq1p5_score: Optional[float] = None  # Google UVQ 1.5 MOS (1-5, higher=better)
    unified_vqa_score: Optional[float] = None  # Unified-VQA FR/NR quality (0-1, higher=better)
    dover_technical: Optional[float] = None  # DOVER technical quality
    dover_aesthetic: Optional[float] = None  # DOVER aesthetic quality
    internvqa_score: Optional[float] = None  # InternVQA video quality (higher=better)
    topiq_score: Optional[float] = None  # TOPIQ transformer-based IQA (higher=better)
    liqe_score: Optional[float] = None  # LIQE lightweight IQA (higher=better)
    clip_iqa_score: Optional[float] = None  # CLIP-IQA semantic quality (0-1, higher=better)
    nima_onnx_score: Optional[float] = None  # NIMA ONNX aesthetic score (1-10, higher=better)

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

    # Additional IQA/VQA
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
    mj_video_overall_score: Optional[float] = None  # MJ-Video learned preference reward
    mj_video_alignment_score: Optional[float] = None  # MJ-Video prompt alignment aspect
    mj_video_safety_score: Optional[float] = None  # MJ-Video safety aspect
    mj_video_fineness_score: Optional[float] = None  # MJ-Video fine-detail aspect
    mj_video_coherence_score: Optional[float] = None  # MJ-Video coherence/consistency aspect
    mj_video_fairness_score: Optional[float] = None  # MJ-Video bias/fairness aspect

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
    hpsv2_score: Optional[float] = None  # HPSv2 prompt-image preference score (higher=better)
    hpsv3_score: Optional[float] = None  # HPSv3 human preference reward mu (higher=better)
    cycle_reward_score: Optional[float] = None  # CycleReward-Combo alignment (higher=better)
    chipqa_score: Optional[float] = None  # ChipQA space-time-chip NR-VQA (higher=better)
    evoquality_score: Optional[float] = None  # EvoQuality self-evolving VLM NR-IQA (1-5, higher=better)

    # Text overlay (NVIDIA Curator)
    text_overlay_score: Optional[float] = None  # Text overlay severity (0-1)

    # ptlflow optical flow (Data-Juicer)
    ptlflow_motion_score: Optional[float] = None  # ptlflow optical flow magnitude

    # QCN (CVPR 2024)
    qcn_score: Optional[float] = None  # Geometric order blind IQA

    # Video-native VQA
    finevq_score: Optional[float] = None  # FineVQ fine-grained UGC VQA (CVPR 2025)
    kvq_score: Optional[float] = None  # KVQ saliency-guided VQA (CVPR 2025)
    rqvqa_score: Optional[float] = None  # RQ-VQA raw regression score (higher=better)
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
    cvvdp_score: Optional[float] = None  # ColorVideoVDP quality in JOD units (max 10)

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
    chronomagic_ch_score: Optional[float] = None  # CHScore = 1/TSI_sum (unbounded, higher=more coherent)

    # GenEval T2I compositional (NeurIPS 2024, arXiv:2310.11513) — image-only, 0-1, higher=better
    geneval_single_object: Optional[float] = None  # Single-object presence
    geneval_two_object: Optional[float] = None  # Two-object co-presence
    geneval_counting: Optional[float] = None  # Counting accuracy
    geneval_colors: Optional[float] = None  # Color attribute match
    geneval_position: Optional[float] = None  # Spatial position relation
    geneval_color_attribution: Optional[float] = None  # Color↔object binding
    geneval_overall: Optional[float] = None  # Mean of activated sub-scores

    # UnifiedReward 2.0 T2I reward (1-5, higher=better)
    unified_reward_2_score: Optional[float] = None  # Mean alignment/coherence/style score
    unified_reward_2_alignment_score: Optional[float] = None  # Prompt-image alignment
    unified_reward_2_coherence_score: Optional[float] = None  # Logical/visual coherence
    unified_reward_2_style_score: Optional[float] = None  # Aesthetic style quality

    # Qwen-Image-Bench T2I judge (0-100, higher=better)
    qwen_image_bench_quality: Optional[float] = None  # Quality L1 score
    qwen_image_bench_aesthetics: Optional[float] = None  # Aesthetics L1 score
    qwen_image_bench_alignment: Optional[float] = None  # Prompt-image alignment L1 score
    qwen_image_bench_real_world_fidelity: Optional[float] = None  # Real-world fidelity L1
    qwen_image_bench_creative_generation: Optional[float] = None  # Creative generation L1
    qwen_image_bench_overall: Optional[float] = None  # Mean of Qwen-Image-Bench L1 scores

    # UnifiedReward Edit (instruction-guided image editing)
    unified_reward_edit_score: Optional[float] = None  # Primary edit quality score
    unified_reward_edit_success_score: Optional[float] = None  # Instruction success (0-25)
    unified_reward_edit_overediting_score: Optional[float] = None  # Edit preservation (0-25)
    unified_reward_edit_image_1_score: Optional[float] = None  # Pairwise edit image 1 score
    unified_reward_edit_image_2_score: Optional[float] = None  # Pairwise edit image 2 score
    unified_reward_edit_winner: Optional[float] = None  # 0=tie, 1=image1, 2=image2
    dice_edit_coherence_score: Optional[float] = None  # DICE coherent localized changes (0-1)
    vebench_score: Optional[float] = None  # Comparative instruction-guided video-edit quality

    # TC-Bench temporal compositionality (T2V, 0-1, higher=better)
    tcbench_attribute_score: Optional[float] = None  # Time-ordered attribute changes
    tcbench_object_score: Optional[float] = None  # Time-ordered object appearance
    tcbench_background_score: Optional[float] = None  # Time-ordered background changes
    tcbench_overall: Optional[float] = None  # Mean TC-Bench score

    # VideoPhy-2 VLM-based physics adherence (0-1, higher=better)
    videophy_pc_score: Optional[float] = None  # Physical commonsense
    videophy_sa_score: Optional[float] = None  # Semantic adherence

    # EntityBench cross-shot identity persistence (0-1, higher=better; batch metric)
    entitybench_identity_consistency: Optional[float] = None  # Face/identity persistence across shots
    entitybench_appearance_consistency: Optional[float] = None  # Overall appearance persistence across shots

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
    mdvqa_score: Optional[float] = None  # MD-VQA fused quality (0-1, higher=better)

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

    # NISQA multidimensional speech quality (arXiv:2104.09494, 1-5 MOS, higher=better)
    nisqa_mos: Optional[float] = None  # Overall predicted MOS
    nisqa_noisiness: Optional[float] = None  # Noisiness sub-score
    nisqa_coloration: Optional[float] = None  # Coloration sub-score
    nisqa_discontinuity: Optional[float] = None  # Discontinuity sub-score
    nisqa_loudness: Optional[float] = None  # Loudness sub-score

    # PEAQ reference-based audio codec quality (ITU-R BS.1387)
    peaq_odg: Optional[float] = None  # Objective Difference Grade (-4..0, higher=better)
    peaq_di: Optional[float] = None  # Distortion Index (higher=better)

    # Audio aesthetics
    audiobox_production: Optional[float] = None  # Audiobox production quality (PQ)
    audiobox_enjoyment: Optional[float] = None  # Audiobox content enjoyment (CE)
    audiobox_pc: Optional[float] = None  # Audiobox production complexity (PC)
    audiobox_cu: Optional[float] = None  # Audiobox content usefulness (CU)
    song_eval_coherence: Optional[float] = None  # SongEval overall coherence (1-5, higher=better)
    song_eval_musicality: Optional[float] = None  # SongEval overall musicality (1-5, higher=better)
    song_eval_memorability: Optional[float] = None  # SongEval memorability (1-5, higher=better)
    song_eval_clarity: Optional[float] = None  # SongEval clarity of song structure (1-5, higher=better)
    song_eval_naturalness: Optional[float] = None  # SongEval vocal breathing/phrasing naturalness (1-5, higher=better)
    muq_eval_mi_score: Optional[float] = None  # MuQ-Eval musical impression MOS (1-5, higher=better)

    # Talking head / lip sync
    thqa_score: Optional[float] = None  # THQA talking head quality (higher=better)
    lse_d: Optional[float] = None  # LSE-D lip sync error distance (lower=better)
    lse_c: Optional[float] = None  # LSE-C lip sync error confidence (higher=better)
    silent_lip_stability: Optional[float] = None  # THEval silent-mouth lip-opening MAD (lower=better)
    lip_dynamics_score: Optional[float] = None  # THEval mouth-shape distance variation (higher=more dynamic)
    eyebrow_dynamics_score: Optional[float] = None  # THEval normalized brow-motion intensity (higher=more dynamic)
    head_motion_dynamics_score: Optional[float] = None  # THEval pose/translation complexity (higher=more dynamic)
    mouth_quality_score: Optional[float] = None  # THEval MUSIQ on mouth crops (higher=better)
    video_edit_motion_fidelity: Optional[float] = None  # Source/edit trajectory-motion similarity (higher=better)

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

    # Fine-grained preference reward (VisionReward, AAAI 2026)
    vision_reward_score: Optional[float] = None  # VisionReward weighted judgment score (higher=better)

    # Physics-IQ reference-based physical understanding (ICCV 2025)
    physics_iq_score: Optional[float] = None  # Combined Physics-IQ score (0-100, higher=better)
    physics_iq_spatial_iou: Optional[float] = None  # Spatial IoU vs real continuation (0-1)
    physics_iq_spatiotemporal_iou: Optional[float] = None  # Spatiotemporal IoU vs real continuation (0-1)
    physics_iq_weighted_spatial_iou: Optional[float] = None  # Weighted spatial IoU vs real continuation (0-1)
    physics_iq_mse: Optional[float] = None  # MSE vs real continuation (lower=better)
    physics_iq_verified_score: Optional[float] = None  # Two-real-take verified score (0-100)
    physics_iq_verified_spatial_score: Optional[float] = None  # Variance-normalized spatial IoU
    physics_iq_verified_spatiotemporal_score: Optional[float] = None  # Variance-normalized ST-IoU
    physics_iq_verified_weighted_spatial_score: Optional[float] = None  # Normalized weighted IoU
    physics_iq_verified_mse_score: Optional[float] = None  # Inverse variance-normalized MSE

    # 2025-2026 reference video-generation evaluator result adapters
    love_perception_score: Optional[float] = None  # LOVE raw perception regressor score
    love_correspondence_score: Optional[float] = None  # LOVE raw prompt correspondence score
    ref4d_semantic_score: Optional[float] = None  # Ref4D semantic score (0-100)
    ref4d_event_score: Optional[float] = None  # Ref4D event-temporal score (0-100)
    ref4d_motion_score: Optional[float] = None  # Ref4D motion-dynamics score (0-100)
    ref4d_world_score: Optional[float] = None  # Ref4D world-knowledge score
    ref4d_overall_score: Optional[float] = None  # Mean of available Ref4D dimensions
    phyground_spatial_alignment_score: Optional[float] = None  # SA judge score (1-5)
    phyground_prompt_temporal_validity_score: Optional[float] = None  # PTV judge score (1-5)
    phyground_persistence_score: Optional[float] = None  # Persistence judge score (1-5)
    phyground_general_score: Optional[float] = None  # Mean general judge score (1-5)
    phyground_physical_score: Optional[float] = None  # Mean applicable-law score (1-5)
    phyground_physical_coverage: Optional[float] = None  # Fraction of laws scored (0-1)

    # Image-to-image fidelity diagnostics
    i2i_mse: Optional[float] = None
    i2i_mae: Optional[float] = None
    i2i_mean_bias: Optional[float] = None
    i2i_exact_match_ratio: Optional[float] = None
    i2i_red_bias: Optional[float] = None
    i2i_green_bias: Optional[float] = None
    i2i_blue_bias: Optional[float] = None
    i2i_luminance_mae: Optional[float] = None
    i2i_chroma_cr_mae: Optional[float] = None
    i2i_chroma_cb_mae: Optional[float] = None
    i2i_hue_mae_degrees: Optional[float] = None
    i2i_colorfulness_delta: Optional[float] = None
    i2i_hist_bhattacharyya_red: Optional[float] = None
    i2i_hist_bhattacharyya_green: Optional[float] = None
    i2i_hist_bhattacharyya_blue: Optional[float] = None
    i2i_gradient_similarity_mean: Optional[float] = None
    i2i_edge_f1: Optional[float] = None
    i2i_spectral_cosine: Optional[float] = None
    i2i_mutual_information: Optional[float] = None
    i2i_dinov2_cls_similarity: Optional[float] = None
    i2i_dinov2_patch_similarity: Optional[float] = None
    i2i_clip_similarity: Optional[float] = None
    i2i_siglip_similarity: Optional[float] = None
    i2i_lpips_alex: Optional[float] = None

    # Camera trajectory adherence (CamI2V-style pose errors)
    camera_rot_error: Optional[float] = None  # RotErr: rotation error vs target trajectory (deg, lower=better)
    camera_trans_error: Optional[float] = None  # TransErr: translation error vs target trajectory (lower=better)
    camera_traj_consistency: Optional[float] = None  # CamMC: camera motion consistency (lower=better)

    # Camera motion taxonomy (CameraBench)
    camera_motion_class_confidence: Optional[float] = None  # Confidence of predicted camera-motion class (0-1)

    # Audio-visual generation sync
    desync_score: Optional[float] = None  # Synchformer predicted AV offset (seconds, lower=better)
    av_align_score: Optional[float] = None  # AV-Align onset/flow-peak IoU (0-1, higher=better)

    # Subject-driven generation consistency (OpenS2V-Eval)
    opens2v_nexus_score: Optional[float] = None  # NexusScore detected-subject-crop consistency (higher=better)
    opens2v_natural_score: Optional[float] = None  # NaturalScore VLM naturalness (higher=better)

    # Human anatomy plausibility
    anatomy_score: Optional[float] = None  # Keypoint-based limb-count/anatomy plausibility (0-1, higher=better)

    # RTMPose pose/gesture plausibility
    rtmpose_score: Optional[float] = None  # RTMPose keypoint-confidence pose plausibility (0-1, higher=better)
    pose_driver_fidelity: Optional[float] = None  # Body-pose fidelity to a driving video, PCK over normalised skeletons (0-1, higher=better)
    pose_driver_fidelity_min: Optional[float] = None  # Worst matched moment of the same measure (0-1, higher=better)
    pose_driver_fidelity_coverage: Optional[float] = None  # Share of compared moments where both skeletons were found (0-1)
    multi_subject_identity_worst: Optional[float] = None  # Lowest per-subject identity similarity in a multi-person clip (higher=better)
    multi_subject_identity_mean: Optional[float] = None  # Mean per-subject identity similarity in a multi-person clip (higher=better)
    multi_subject_identity_coverage: Optional[float] = None  # Share of sampled frames covered by the assigned face tracks (0-1)
    multi_subject_identity_tracks: Optional[float] = None  # Number of face tracks the assignment was built from
    active_speaker_margin: Optional[float] = None  # Lip-sync confidence gap between the best-synced face and the runner-up (higher=cleaner)
    active_speaker_best_lse_c: Optional[float] = None  # Lip-sync confidence of the best-synced face (higher=better)
    active_speaker_silent_faces: Optional[float] = None  # Faces for which no talking mouth was detected
    object_permanence_interior_vanish: Optional[float] = None  # Tracks that ended away from the frame border (disappearance, not exit)
    object_permanence_border_exit: Optional[float] = None  # Tracks that ended at the frame border (a legitimate exit)
    object_permanence_occlusion_share: Optional[float] = None  # Share of frames with overlapping boxes; how far the two counts above can be trusted

    # VMBench Object Integrity Score (human bone-length/joint-angle temporal integrity)
    object_integrity_score: Optional[float] = None  # VMBench OIS (0-1, higher=better)

    # VMBench Motion Smoothness Score (Q-Align per-frame quality jump detection)
    vmbench_mss: Optional[float] = None  # VMBench MSS (0-1, higher=smoother)

    # VMBench Perceptible Amplitude Score (subject-vs-background motion magnitude)
    perceptible_amplitude_score: Optional[float] = None  # VMBench PAS (0-1, subject motion degree)

    # VMBench Temporal Coherence Score (implausible object vanish/emerge)
    temporal_coherence_score: Optional[float] = None  # VMBench TCS (0-1, higher=more coherent)

    # VMBench Commonsense Adherence Score (physical-commonsense plausibility)
    commonsense_adherence_score: Optional[float] = None  # VMBench CAS (0-1, higher=more plausible)

    # Layout artifacts
    grid_layout_score: Optional[float] = None  # Split-screen/grid-collage likelihood (0-1, higher=more likely)

    # -- Provenance (not a metric) ----------------------------------------
    # Maps ``module.name`` -> the backend/tier that produced its metrics for
    # this sample (e.g. "pyiqa", "proxy", "heuristic"). Populated automatically
    # by the pipeline from each module's ``_backend`` attribute; modules need
    # not touch it. Excluded from metric counts/grouping via _NON_METRIC_FIELDS.
    metric_backends: Dict[str, str] = Field(default_factory=dict)


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
    # Names of modules that raised (or returned invalid output) while processing
    # this sample. A non-empty list marks the sample as INCOMPLETE so the
    # pipeline reprocesses it on the next run / resume instead of serving a
    # partial result from cache. Populated by the pipeline; empty by default so
    # legacy state files load cleanly.
    failed_modules: List[str] = Field(default_factory=list)
    # Free-form, structured annotations a module attaches to a sample when the
    # information is not a numeric metric field — e.g. grid_layout stores the
    # detected layout string ("2x2") and camerabench the predicted camera-motion
    # class. Serialised with the sample state; empty by default so legacy state
    # files load cleanly.
    metadata: Dict[str, Any] = Field(default_factory=dict)

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
    fvd_content_debiased: Optional[float] = None  # Content-Debiased FVD (Ge et al. CVPR 2024, lower=better)
    fvd_dinov2: Optional[float] = None  # FVD with DINOv2 spatial backbone (rFVD, lower=better)
    kvd: Optional[float] = None  # Kernel Video Distance
    fvmd: Optional[float] = None  # Fréchet Video Motion Distance
    fid: Optional[float] = None  # Fréchet Inception Distance
    cmmd: Optional[float] = None  # CLIP Maximum Mean Discrepancy (lower=better)
    jedi: Optional[float] = None  # JEDi (V-JEPA + MMD, ICLR 2025)
    kid: Optional[float] = None  # Kernel Inception Distance (lower=better)
    kid_std: Optional[float] = None  # KID standard deviation
    prdc_precision: Optional[float] = None  # PRDC precision in DINOv2 space (0-1)
    prdc_recall: Optional[float] = None  # PRDC recall in DINOv2 space (0-1)
    prdc_density: Optional[float] = None  # PRDC density in DINOv2 space
    prdc_coverage: Optional[float] = None  # PRDC coverage in DINOv2 space (0-1)

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
    fad_infinity: Optional[float] = None  # FAD extrapolated to infinite sample size (lower=better)
    fad_vggish: Optional[float] = None  # Frechet Audio Distance with VGGish backbone (lower=better)
    fad_vggish_infinity: Optional[float] = None  # VGGish FAD extrapolated to infinite sample size (lower=better)
    fad_panns: Optional[float] = None  # Frechet Audio Distance with PANNs CNN14 backbone (lower=better)
    fad_panns_infinity: Optional[float] = None  # PANNs FAD extrapolated to infinite sample size (lower=better)
    fad_passt: Optional[float] = None  # Frechet Audio Distance with PaSST backbone (lower=better)
    fad_passt_infinity: Optional[float] = None  # PaSST FAD extrapolated to infinite sample size (lower=better)
    audio_isc_mean: Optional[float] = None  # Inception Score for Audio mean (higher=better)
    audio_isc_std: Optional[float] = None  # Inception Score for Audio standard deviation
    audio_kl: Optional[float] = None  # Audio classifier distribution KL divergence (lower=better)
    mauve_audio_divergence: Optional[float] = None  # MAD -log(MAUVE), lower=better
    kad: Optional[float] = None  # Kernel Audio Distance (lower=better)
    fgd: Optional[float] = None  # Frechet Gesture Distance (lower=better)
    fmd: Optional[float] = None  # Frechet Motion Distance (lower=better)
    msswd: Optional[float] = None  # Multi-Scale Sliced Wasserstein (lower=better)
    sfid: Optional[float] = None  # Spatial FID (lower=better)
    vendi: Optional[float] = None  # Vendi Score diversity (higher=better)
    stream_spatial: Optional[float] = None  # STREAM spatial fidelity+diversity
    stream_temporal: Optional[float] = None  # STREAM temporal naturalness
    worldscore: Optional[float] = None  # WorldScore generation quality

    # Reference VBench 2.0 intrinsic-faithfulness suite (dataset-level)
    vbench2_human_anatomy: Optional[float] = None
    vbench2_human_identity: Optional[float] = None
    vbench2_human_clothes: Optional[float] = None
    vbench2_diversity: Optional[float] = None
    vbench2_composition: Optional[float] = None
    vbench2_dynamic_spatial_relationship: Optional[float] = None
    vbench2_dynamic_attribute: Optional[float] = None
    vbench2_motion_order_understanding: Optional[float] = None
    vbench2_human_interaction: Optional[float] = None
    vbench2_complex_landscape: Optional[float] = None
    vbench2_complex_plot: Optional[float] = None
    vbench2_camera_motion: Optional[float] = None
    vbench2_motion_rationality: Optional[float] = None
    vbench2_instance_preservation: Optional[float] = None
    vbench2_mechanics: Optional[float] = None
    vbench2_thermotics: Optional[float] = None
    vbench2_material: Optional[float] = None
    vbench2_multiview_consistency: Optional[float] = None
    vbench2_creativity_score: Optional[float] = None
    vbench2_commonsense_score: Optional[float] = None
    vbench2_controllability_score: Optional[float] = None
    vbench2_human_fidelity_score: Optional[float] = None
    vbench2_physics_score: Optional[float] = None
    vbench2_total_score: Optional[float] = None

    # WorldModelBench (CVPR 2025 workshop, dataset-level; higher=better)
    worldmodelbench_instruction_score: Optional[float] = None  # Range 0-3
    worldmodelbench_newton_adherence: Optional[float] = None  # Fraction without violation
    worldmodelbench_mass_solid_adherence: Optional[float] = None
    worldmodelbench_fluid_adherence: Optional[float] = None
    worldmodelbench_penetration_adherence: Optional[float] = None
    worldmodelbench_gravity_adherence: Optional[float] = None
    worldmodelbench_aesthetics_adherence: Optional[float] = None
    worldmodelbench_temporal_adherence: Optional[float] = None
    worldmodelbench_physical_score: Optional[float] = None  # Sum of five adherence rates, 0-5
    worldmodelbench_common_sense_score: Optional[float] = None  # Sum of two rates, 0-2
    worldmodelbench_total_score: Optional[float] = None  # Raw total, 0-10

    # Codec comparison (dataset-level)
    bd_rate: Optional[float] = None  # BD-Rate compression efficiency (%, negative=better)
    bd_psnr: Optional[float] = None  # BD-PSNR quality delta (dB, positive=better)

    # Image LPIPS diversity (dataset-level)
    lpips_diversity: Optional[float] = None  # Average pairwise LPIPS across dataset (higher=more diverse)

    # Verse-Bench benchmark (dataset-level)
    verse_bench_overall: Optional[float] = None  # Verse-Bench final score
    verse_bench_metrics: Optional[Dict[str, float]] = None  # Raw Verse-Bench component metrics
    verse_bench_breakdown: Optional[Dict[str, float]] = None  # Verse-Bench subscores and overall


