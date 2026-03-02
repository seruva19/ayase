"""Ayase metric modules.

All modules are auto-discovered at runtime by ``ModuleRegistry.discover_modules()``.
This file exports the most commonly-used modules for direct import convenience::

    from ayase.modules import DOVERModule, VMAFModule, VideoScoreModule
"""

# --- Core / basic ---
from .basic import BasicQualityModule
from .metadata import MetadataModule
from .structural import StructuralModule
from .exposure import ExposureModule
from .compression_artifacts import CompressionArtifactsModule

# --- Aesthetics ---
from .aesthetic import AestheticModule
from .aesthetic_scoring import AestheticScoringModule
from .nima import NIMAModule
from .laion_aesthetic import LAIONAestheticModule

# --- Text / OCR ---
from .text import TextDetectionModule
from .ocr_fidelity import OCRFidelityModule
from .captioning import CaptioningModule

# --- Motion & flow ---
from .motion import MotionModule
from .motion_smoothness import MotionSmoothnessModule
from .motion_amplitude import MotionAmplitudeModule
from .advanced_flow import AdvancedFlowModule
from .camera_motion import CameraMotionModule
from .kandinsky_motion import KandinskyMotionModule

# --- Temporal consistency ---
from .temporal_flickering import TemporalFlickeringModule
from .clip_temporal import CLIPTemporalModule
from .subject_consistency import SubjectConsistencyModule
from .background_consistency import BackgroundConsistencyModule
from .object_permanence import ObjectPermanenceModule
from .color_consistency import ColorConsistencyModule
from .temporal_style import TemporalStyleModule
from .style_consistency import StyleConsistencyModule

# --- Alignment ---
from .semantic_alignment import SemanticAlignmentModule
from .video_text_matching import VideoTextMatchingModule
from .t2v_score import T2VScoreModule
from .vqa_score import VQAScoreModule

# --- No-reference quality ---
from .dover import DOVERModule
from .fast_vqa import FastVQAModule
from .q_align import QAlignModule
from .topiq import TOPIQModule
from .musiq import MUSIQModule
from .maniqa import MANIQAModule
from .brisque import BRISQUEModule
from .niqe import NIQEModule
from .clip_iqa import CLIPIQAModule
from .imaging_quality import ImagingQualityModule

# --- Full-reference quality ---
from .vmaf import VMAFModule
from .dists import DISTSModule
from .perceptual_fr import PerceptualFRModule

# --- SOTA video quality (CVPR/NeurIPS/EMNLP 2024-2025) ---
from .videoscore import VideoScoreModule
from .video_reward import VideoRewardModule
from .rqvqa import RQVQAModule
from .aigv_assessor import AIGVAssessorModule
from .finevq import FineVQModule
from .kvq import KVQModule
from .jedi_metric import JEDiModule
from .cover import COVERModule
from .videval import VIDEVALModule

# --- Generation metrics ---
from .fvd import FVDModule
from .fvmd import FVMDModule
from .inception_score import InceptionScoreModule
from .i2v_similarity import I2VSimilarityModule
from .sd_reference import SDReferenceModule

# --- Face & human ---
from .human_fidelity import HumanFidelityModule
from .face_fidelity import FaceFidelityModule
from .face_landmark_quality import FaceLandmarkQualityModule
from .face_iqa import FaceIQAModule

# --- Scene & content ---
from .scene import SceneModule
from .scene_detection import SceneDetectionModule
from .scene_tagging import SceneTaggingModule
from .object_detection import ObjectDetectionModule
from .action_recognition import ActionRecognitionModule
from .spatial_relationship import SpatialRelationshipModule
from .physics import PhysicsModule
from .commonsense import CommonsenseModule
from .multiple_objects import MultipleObjectsModule

# --- Safety & ethics ---
from .nsfw import NSFWModule
from .deepfake_detection import DeepfakeDetectionModule
from .harmful_content import HarmfulContentModule
from .watermark_classifier import WatermarkClassificationModule
from .bias_detection import BiasDetectionModule

# --- Audio ---
from .audio import AudioModule
from .audio_pesq import AudioPESQModule
from .dnsmos import DNSMOSModule

# --- HDR / codec ---
from .hdr_metadata import HDRMetadataModule
from .production_quality import ProductionQualityModule

# --- Dataset operations ---
from .dedup import DeduplicationModule
from .embedding import EmbeddingModule
from .diversity_selection import DiversitySelectionModule
from .dataset_analytics import DatasetAnalyticsModule
from .resolution_bucketing import ResolutionBucketingModule
from .llm_advisor import LLMAdvisorModule

# --- Utility ---
from .cpbd import CPBDModule
from .spectral import SpectralComplexityModule
from .background_diversity import BackgroundDiversityModule
from .video_memorability import VideoMemorabilityModule


__all__ = [
    # Core
    "BasicQualityModule",
    "MetadataModule",
    "StructuralModule",
    "ExposureModule",
    "CompressionArtifactsModule",
    # Aesthetics
    "AestheticModule",
    "AestheticScoringModule",
    "NIMAModule",
    "LAIONAestheticModule",
    # Text / OCR
    "TextDetectionModule",
    "OCRFidelityModule",
    "CaptioningModule",
    # Motion
    "MotionModule",
    "MotionSmoothnessModule",
    "MotionAmplitudeModule",
    "AdvancedFlowModule",
    "CameraMotionModule",
    "KandinskyMotionModule",
    # Temporal
    "TemporalFlickeringModule",
    "CLIPTemporalModule",
    "SubjectConsistencyModule",
    "BackgroundConsistencyModule",
    "ObjectPermanenceModule",
    "ColorConsistencyModule",
    "TemporalStyleModule",
    "StyleConsistencyModule",
    # Alignment
    "SemanticAlignmentModule",
    "VideoTextMatchingModule",
    "T2VScoreModule",
    "VQAScoreModule",
    # NR quality
    "DOVERModule",
    "FastVQAModule",
    "QAlignModule",
    "TOPIQModule",
    "MUSIQModule",
    "MANIQAModule",
    "BRISQUEModule",
    "NIQEModule",
    "CLIPIQAModule",
    "ImagingQualityModule",
    # FR quality
    "VMAFModule",
    "DISTSModule",
    "PerceptualFRModule",
    # SOTA video quality
    "VideoScoreModule",
    "VideoRewardModule",
    "RQVQAModule",
    "AIGVAssessorModule",
    "FineVQModule",
    "KVQModule",
    "JEDiModule",
    "COVERModule",
    "VIDEVALModule",
    # Generation
    "FVDModule",
    "FVMDModule",
    "InceptionScoreModule",
    "I2VSimilarityModule",
    "SDReferenceModule",
    # Face & human
    "HumanFidelityModule",
    "FaceFidelityModule",
    "FaceLandmarkQualityModule",
    "FaceIQAModule",
    # Scene & content
    "SceneModule",
    "SceneDetectionModule",
    "SceneTaggingModule",
    "ObjectDetectionModule",
    "ActionRecognitionModule",
    "SpatialRelationshipModule",
    "PhysicsModule",
    "CommonsenseModule",
    "MultipleObjectsModule",
    # Safety
    "NSFWModule",
    "DeepfakeDetectionModule",
    "HarmfulContentModule",
    "WatermarkClassificationModule",
    "BiasDetectionModule",
    # Audio
    "AudioModule",
    "AudioPESQModule",
    "DNSMOSModule",
    # HDR / codec
    "HDRMetadataModule",
    "ProductionQualityModule",
    # Dataset ops
    "DeduplicationModule",
    "EmbeddingModule",
    "DiversitySelectionModule",
    "DatasetAnalyticsModule",
    "ResolutionBucketingModule",
    "LLMAdvisorModule",
    # Utility
    "CPBDModule",
    "SpectralComplexityModule",
    "BackgroundDiversityModule",
    "VideoMemorabilityModule",
]
