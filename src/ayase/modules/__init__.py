"""Ayase metric modules.

All modules are auto-discovered at runtime by ``ModuleRegistry.discover_modules()``.
This file exports the most commonly-used modules for direct import convenience::

    from ayase.modules import DOVERModule, VMAFModule, VideoScoreModule

Modules whose optional dependencies are not installed are silently skipped.
"""

import importlib as _importlib
import logging as _logging

_logger = _logging.getLogger(__name__)

# (attribute_name, module_path) pairs for lazy, fault-tolerant imports.
_IMPORTS = [
    # --- Core / basic ---
    ("BasicQualityModule", ".basic"),
    ("MetadataModule", ".metadata"),
    ("StructuralModule", ".structural"),
    ("ExposureModule", ".exposure"),
    ("CompressionArtifactsModule", ".compression_artifacts"),
    ("TonalDynamicRangeModule", ".tonal_dynamic_range"),
    # --- Aesthetics ---
    ("AestheticModule", ".aesthetic"),
    ("AestheticScoringModule", ".aesthetic_scoring"),
    ("NIMAModule", ".nima"),
    ("NIMALegacyONNXModule", ".nima_legacy_onnx"),
    ("LAIONAestheticModule", ".laion_aesthetic"),
    # --- Text / OCR ---
    ("TextDetectionModule", ".text"),
    ("OCRFidelityModule", ".ocr_fidelity"),
    ("CaptioningModule", ".captioning"),
    ("ASRTranscribeModule", ".asr_transcribe"),
    ("ASRCERModule", ".asr_cer"),
    ("ASRWERModule", ".asr_wer"),
    # --- Motion & flow ---
    ("MotionModule", ".motion"),
    ("MotionSmoothnessModule", ".motion_smoothness"),
    ("MotionAmplitudeModule", ".motion_amplitude"),
    ("AdvancedFlowModule", ".advanced_flow"),
    ("CameraMotionModule", ".camera_motion"),
    ("KandinskyMotionModule", ".kandinsky_motion"),
    # --- Temporal consistency ---
    ("TemporalFlickeringModule", ".temporal_flickering"),
    ("CLIPTemporalModule", ".clip_temporal"),
    ("SubjectConsistencyModule", ".subject_consistency"),
    ("BackgroundConsistencyModule", ".background_consistency"),
    ("ObjectPermanenceModule", ".object_permanence"),
    ("ColorConsistencyModule", ".color_consistency"),
    ("TemporalStyleModule", ".temporal_style"),
    ("StyleConsistencyModule", ".style_consistency"),
    # --- Alignment ---
    ("SemanticAlignmentModule", ".semantic_alignment"),
    ("VideoTextMatchingModule", ".video_text_matching"),
    ("T2VScoreModule", ".t2v_score"),
    ("VQAScoreModule", ".vqa_score"),
    ("BLIPScoreModule", ".blip_score"),
    ("TIFAModule", ".tifa"),
    ("DSGModule", ".dsg"),
    ("PickScoreModule", ".pickscore"),
    ("ImageRewardModule", ".image_reward"),
    ("HPSv2Module", ".hpsv2"),
    ("HPSv3Module", ".hpsv3"),
    ("UnifiedReward2Module", ".unified_reward_2"),
    ("QwenImageBenchModule", ".qwen_image_bench"),
    ("UnifiedRewardEditModule", ".unified_reward_edit"),
    # --- No-reference quality ---
    ("DOVERModule", ".dover"),
    ("FastVQAModule", ".fast_vqa"),
    ("QAlignModule", ".q_align"),
    ("TOPIQModule", ".topiq"),
    ("MUSIQModule", ".musiq"),
    ("MANIQAModule", ".maniqa"),
    ("BRISQUEModule", ".brisque"),
    ("NIQEModule", ".niqe"),
    ("CLIPIQAModule", ".clip_iqa"),
    ("ImagingQualityModule", ".imaging_quality"),
    ("ChipQAModule", ".chipqa"),
    ("EvoQualityModule", ".evoquality"),
    # --- Full-reference quality ---
    ("VMAFModule", ".vmaf"),
    ("DISTSModule", ".dists"),
    ("PerceptualFRModule", ".perceptual_fr"),
    # --- SOTA video quality (CVPR/NeurIPS/EMNLP 2024-2025) ---
    ("VideoScoreModule", ".videoscore"),
    ("VideoScore2Module", ".videoscore2"),
    ("OfficialVBench2Module", ".vbench2_official"),
    ("WorldModelBenchModule", ".worldmodelbench"),
    ("MJVideoModule", ".mj_video"),
    ("RTMPoseFidelityModule", ".rtmpose_fidelity"),
    ("VideoRewardModule", ".video_reward"),
    ("RQVQAModule", ".rqvqa"),
    ("AIGVAssessorModule", ".aigv_assessor"),
    ("FineVQModule", ".finevq"),
    ("KVQModule", ".kvq"),
    ("JEDiModule", ".jedi_metric"),
    ("COVERModule", ".cover"),
    ("VIDEVALModule", ".videval"),
    ("UNQAModule", ".unqa"),
    ("InternVQAModule", ".internvqa"),
    ("NRGVQMModule", ".nr_gvqm"),
    # --- Generation metrics ---
    ("KIDModule", ".kid"),
    ("FIDModule", ".fid"),
    ("CMMDModule", ".cmmd"),
    ("PRDCDINOv2Module", ".prdc_dinov2"),
    ("FVDModule", ".fvd"),
    ("FVMDModule", ".fvmd"),
    ("InceptionScoreModule", ".inception_score"),
    ("I2VSimilarityModule", ".i2v_similarity"),
    ("SDReferenceModule", ".sd_reference"),
    # --- Face & human ---
    ("HumanFidelityModule", ".human_fidelity"),
    ("FaceFidelityModule", ".face_fidelity"),
    ("FaceLandmarkQualityModule", ".face_landmark_quality"),
    ("FaceIQAModule", ".face_iqa"),
    ("IdentityLossModule", ".identity_loss"),
    ("FaceCrossSimilarityModule", ".face_cross_similarity"),
    # --- Scene & content ---
    ("SceneModule", ".scene"),
    ("SceneDetectionModule", ".scene_detection"),
    ("SceneTaggingModule", ".scene_tagging"),
    ("ObjectDetectionModule", ".object_detection"),
    ("ActionRecognitionModule", ".action_recognition"),
    ("SpatialRelationshipModule", ".spatial_relationship"),
    ("PhysicsModule", ".physics"),
    ("CommonsenseModule", ".commonsense"),
    ("MultipleObjectsModule", ".multiple_objects"),
    # --- Safety & ethics ---
    ("NSFWModule", ".nsfw"),
    ("DeepfakeDetectionModule", ".deepfake_detection"),
    ("HarmfulContentModule", ".harmful_content"),
    ("WatermarkClassificationModule", ".watermark_classifier"),
    ("BiasDetectionModule", ".bias_detection"),
    # --- Audio ---
    ("AudioModule", ".audio"),
    ("AudioPESQModule", ".audio_pesq"),
    ("AudioUTMOSv2Module", ".audio_utmos_v2"),
    ("AudioISCModule", ".audio_isc"),
    ("AudioKLModule", ".audio_kl"),
    ("SCOREQModule", ".scoreq"),
    ("TTSDS2Module", ".ttsds2"),
    ("KADModule", ".kad"),
    ("HumanCLAPModule", ".human_clap"),
    ("LAIONCLAPScoreModule", ".clap_score"),
    ("MSCLAPScoreModule", ".clap_score"),
    ("GenericCLAPScoreModule", ".clap_score"),
    ("ImageBindScoreModule", ".imagebind_score"),
    ("PAMModule", ".pam"),
    ("AQAScoreModule", ".aqascore"),
    ("DNSMOSModule", ".dnsmos"),
    ("BeatAlignmentModule", ".beat_alignment"),
    ("SongEvalModule", ".song_eval"),
    # --- HDR / codec ---
    ("HDRMetadataModule", ".hdr_metadata"),
    ("HDRChipQAModule", ".hdr_chipqa"),
    ("HDRMAXModule", ".hdrmax"),
    ("BrightRateModule", ".brightrate"),
    ("ProductionQualityModule", ".production_quality"),
    # --- Dataset operations ---
    ("DeduplicationModule", ".dedup"),
    ("EmbeddingModule", ".embedding"),
    ("DiversitySelectionModule", ".diversity_selection"),
    ("DatasetAnalyticsModule", ".dataset_analytics"),
    ("UMAPProjectionModule", ".umap_projection"),
    ("ResolutionBucketingModule", ".resolution_bucketing"),
    ("LLMAdvisorModule", ".llm_advisor"),
    # --- Utility ---
    ("CPBDModule", ".cpbd"),
    ("SpectralComplexityModule", ".spectral"),
    ("BackgroundDiversityModule", ".background_diversity"),
    ("VideoMemorabilityModule", ".video_memorability"),
    ("NemoCuratorModule", ".nemo_curator"),
    # --- VBench-2.0 / Benchmarks ---
    ("CreativityModule", ".creativity"),
    ("ChronoMagicModule", ".chronomagic"),
    ("T2VCompBenchModule", ".t2v_compbench"),
    ("VerseBenchModule", ".verse_bench"),
    # --- Image LPIPS & Concept Presence ---
    ("ImageLPIPSModule", ".image_lpips"),
    ("ConceptPresenceModule", ".concept_presence"),
]

# Perform imports, silently skipping modules with missing optional deps.
_available = {}
for _attr, _mod in _IMPORTS:
    try:
        _m = _importlib.import_module(_mod, __name__)
        _obj = getattr(_m, _attr)
        _available[_attr] = _obj
        globals()[_attr] = _obj
    except Exception:
        pass

__all__ = list(_available.keys())
