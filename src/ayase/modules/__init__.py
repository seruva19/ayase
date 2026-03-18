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
    ("LAIONAestheticModule", ".laion_aesthetic"),
    # --- Text / OCR ---
    ("TextDetectionModule", ".text"),
    ("OCRFidelityModule", ".ocr_fidelity"),
    ("CaptioningModule", ".captioning"),
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
    ("TIFAModule", ".tifa"),
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
    # --- Full-reference quality ---
    ("VMAFModule", ".vmaf"),
    ("DISTSModule", ".dists"),
    ("PerceptualFRModule", ".perceptual_fr"),
    # --- SOTA video quality (CVPR/NeurIPS/EMNLP 2024-2025) ---
    ("VideoScoreModule", ".videoscore"),
    ("VideoRewardModule", ".video_reward"),
    ("RQVQAModule", ".rqvqa"),
    ("AIGVAssessorModule", ".aigv_assessor"),
    ("FineVQModule", ".finevq"),
    ("KVQModule", ".kvq"),
    ("JEDiModule", ".jedi_metric"),
    ("COVERModule", ".cover"),
    ("VIDEVALModule", ".videval"),
    # --- Generation metrics ---
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
    ("DNSMOSModule", ".dnsmos"),
    # --- HDR / codec ---
    ("HDRMetadataModule", ".hdr_metadata"),
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
