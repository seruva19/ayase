"""Cross-sample redundancy detection using embedding cosine similarity.

Identifies semantically redundant samples by comparing normalized embeddings.
Keeps the highest-quality sample in each similarity cluster. Runs in post_process phase."""

import logging
import numpy as np
from typing import List
from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

class DiversitySelectionModule(PipelineModule):
    """
    Identifies and flags redundant samples based on embedding similarity.
    This module performs cross-sample analysis after the main pipeline run.
    """
    name = "diversity"
    description = "Flags redundant samples using embedding similarity (Deduplication)"
    default_config = {
        "similarity_threshold": 0.95,
        "priority_metric": "aesthetic_score" # aesthetic_score, fast_vqa_score, technical_score
    }

    def process(self, sample: Sample) -> Sample:
        # Cross-sample analysis happens in post_process
        return sample

    def post_process(self, all_samples: List[Sample]) -> None:
        """
        Calculates similarity matrix and flags redudant samples.
        Keeps the highest quality sample in each cluster.
        """
        if not all_samples:
            return

        # 1. Filter samples that have embeddings
        valid_samples = [s for s in all_samples if s.embedding is not None]
        if len(valid_samples) < 2:
            logger.info("Not enough samples with embeddings for diversity analysis.")
            return

        # 2. Sort samples by quality (to keep the best ones when duplicates are found)
        metric = self.config.get("priority_metric", "aesthetic_score")

        def get_score(s):
            if s.quality_metrics:
                score = getattr(s.quality_metrics, metric, 0.0)
                return score if score is not None else 0.0
            return 0.0

        # Sort descending (highest quality first)
        valid_samples.sort(key=get_score, reverse=True)

        selected_embeddings = []
        similarity_threshold = self.config.get("similarity_threshold", 0.95)

        redundant_count = 0
        for sample in valid_samples:
            emb = np.array(sample.embedding)
            
            is_redundant = False
            max_sim = 0.0
            if selected_embeddings:
                # Calculate cosine similarities
                # Embeddings from EmbeddingModule are already normalized
                sims = np.dot(selected_embeddings, emb)
                max_sim = np.max(sims)
                
                if max_sim > similarity_threshold:
                    is_redundant = True

            if is_redundant:
                redundant_count += 1
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"Redundant Sample (Similarity: {max_sim:.3f})",
                        details={"max_similarity": float(max_sim), "is_redundant": True},
                        recommendation="Consider removal if high dataset diversity is required."
                    )
                )
            else:
                selected_embeddings.append(emb)

        logger.info(f"Diversity Selection: Kept {len(selected_embeddings)} unique samples, flagged {redundant_count} as redundant.")


class DiversitySelectionCompatModule(DiversitySelectionModule):
    """Compatibility alias matching filename-based discovery."""

    name = "diversity_selection"
