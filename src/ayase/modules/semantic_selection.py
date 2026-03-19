import logging
import numpy as np
from typing import List, Dict, Set
from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

class SemanticSelectionModule(PipelineModule):
    """
    Identifies a diverse subset of samples based on VLM-extracted semantic traits.
    This module uses the 'style_traits' detection data produced by the VLMJudgeModule.
    """
    name = "semantic_selection"
    description = "Selects diverse samples based on VLM-extracted semantic traits"
    default_config = {
        "num_to_select": 10,
        "uniqueness_weight": 0.7,
        "quality_weight": 0.3,
    }

    def process(self, sample: Sample) -> Sample:
        # Cross-sample analysis happens in post_process
        return sample

    def post_process(self, all_samples: List[Sample]) -> None:
        """
        Implements greedy selection for semantic diversity.
        """
        if not all_samples:
            return

        # 1. Extract traits from detections
        video_traits: Dict[str, Dict[str, float]] = {}
        video_scores: Dict[str, float] = {}
        
        for sample in all_samples:
            path_str = str(sample.path)
            traits = {}
            for det in sample.detections:
                if det.get("type") == "style_traits":
                    traits.update(det.get("data", {}))
            
            if traits:
                video_traits[path_str] = traits
                qm = sample.quality_metrics
                video_scores[path_str] = (getattr(qm, "aesthetic_score", None) or 0.5) if qm else 0.5

        if not video_traits:
            logger.info("No style traits found. Skipping semantic selection.")
            return

        # 2. Identify key traits for diversity
        trait_frequency: Dict[str, int] = {}
        for traits in video_traits.values():
            for trait in traits:
                trait_frequency[trait] = trait_frequency.get(trait, 0) + 1

        total_processed = len(video_traits)
        # Traits that are neither too rare nor too common are "distinctive"
        important_traits = [
            t for t, freq in trait_frequency.items() 
            if 0.05 * total_processed <= freq <= 0.6 * total_processed
        ]

        # 3. Greedy selection
        num_to_select = self.config.get("num_to_select", 10)
        u_weight = self.config.get("uniqueness_weight", 0.7)
        q_weight = self.config.get("quality_weight", 0.3)
        
        selected_paths: List[str] = []
        covered_traits: Set[str] = set()
        remaining_paths = list(video_traits.keys())

        while len(selected_paths) < num_to_select and remaining_paths:
            best_path = None
            best_combined_score = -1.0

            for path in remaining_paths:
                traits = set(video_traits[path].keys())
                new_traits = traits - covered_traits
                
                # Uniqueness: fraction of traits that are new to the selection
                uniqueness_score = len(new_traits) / max(1, len(traits))
                quality_score = video_scores[path]
                
                combined_score = (u_weight * uniqueness_score) + (q_weight * quality_score)

                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    best_path = path

            if best_path:
                selected_paths.append(best_path)
                covered_traits.update(video_traits[best_path].keys())
                remaining_paths.remove(best_path)
            else:
                break

        # 4. Flag selected samples
        selected_set = set(selected_paths)
        for sample in all_samples:
            path_str = str(sample.path)
            if path_str in selected_set:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message="Top Selection (Semantic Diversity)",
                        details={"selection_rank": selected_paths.index(path_str) + 1}
                    )
                )

        logger.info(f"Semantic Selection: Picked {len(selected_paths)} diverse samples.")
