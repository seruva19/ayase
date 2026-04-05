"""Rule-based quality advisor that generates actionable recommendations from metric values.

Analyzes blur, brightness, contrast, noise, aesthetics, CLIP alignment, motion,
NSFW, watermarks, OCR overlays, and VQA scores. No ML model used despite the name."""

import logging
from typing import List, Optional

from ayase.models import Sample, ValidationIssue, ValidationSeverity, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class LLMAdvisorModule(PipelineModule):
    name = "llm_advisor"
    description = "Rule-based improvement recommendations derived from quality metrics (no LLM used)"
    default_config = {
        "severity_level": "INFO",
    }

    def process(self, sample: Sample) -> Sample:
        """Analyze all available metrics and generate actionable recommendations."""
        if not sample.quality_metrics:
            return sample

        m = sample.quality_metrics
        recommendations: List[str] = []

        # --- Sharpness / Blur ---
        if m.blur_score is not None and m.blur_score < 100:
            recommendations.append(
                f"Blurry (Laplacian={m.blur_score:.0f}). "
                "Re-encode from a sharper source or discard."
            )

        # --- Brightness ---
        if m.brightness is not None:
            if m.brightness < 40:
                recommendations.append(
                    f"Underexposed (brightness={m.brightness:.0f}). "
                    "Check if artistic intent; otherwise discard or apply gamma correction."
                )
            elif m.brightness > 220:
                recommendations.append(
                    f"Overexposed (brightness={m.brightness:.0f}). "
                    "Blown-out highlights degrade generation quality."
                )

        # --- Contrast ---
        if m.contrast is not None and m.contrast < 20:
            recommendations.append(
                f"Low contrast ({m.contrast:.0f}). "
                "Flat images train poorly; consider histogram equalization or discard."
            )

        # --- Noise ---
        if m.noise_score is not None and m.noise_score < 0.5:
            recommendations.append(
                f"High noise (score={m.noise_score:.2f}). "
                "Apply denoising (NLMeans) or source a cleaner copy."
            )

        # --- Aesthetic score ---
        if m.aesthetic_score is not None:
            if m.aesthetic_score < 3.0:
                recommendations.append(
                    f"Low aesthetic quality ({m.aesthetic_score:.1f}/10). "
                    "Consider excluding from training to avoid degrading generation style."
                )
            elif m.aesthetic_score >= 7.0:
                recommendations.append(
                    f"High aesthetic quality ({m.aesthetic_score:.1f}/10). "
                    "Good candidate for priority inclusion or upweighting."
                )

        # --- CLIP alignment ---
        if m.clip_score is not None and m.clip_score < 0.18:
            recommendations.append(
                f"Poor caption-visual alignment (CLIP={m.clip_score:.2f}). "
                "Rewrite caption or regenerate with a captioning model."
            )

        # --- Motion ---
        if m.motion_score is not None:
            if m.motion_score < 0.5:
                recommendations.append(
                    "Near-static video. May act as a duplicate of a single image; "
                    "consider converting to image sample or discarding."
                )
            elif m.motion_score > 80:
                recommendations.append(
                    f"Excessive motion ({m.motion_score:.0f}). "
                    "Likely camera shake or fast cuts; may produce flickering artifacts in generation."
                )

        # --- Temporal consistency (CLIP) ---
        if m.clip_temp is not None and m.clip_temp < 0.85:
            recommendations.append(
                f"Low temporal consistency (CLIP_temp={m.clip_temp:.2f}). "
                "Scene cuts or flickering detected; split into single-scene clips."
            )

        # --- NSFW ---
        if m.nsfw_score is not None and m.nsfw_score > 0.7:
            recommendations.append(
                f"NSFW content likely ({m.nsfw_score:.2f}). "
                "Exclude from general-purpose training datasets."
            )

        # --- Watermark ---
        if m.watermark_probability is not None and m.watermark_probability > 0.5:
            recommendations.append(
                f"Watermark detected ({m.watermark_probability:.2f}). "
                "Watermarked samples teach the model to reproduce watermarks; discard."
            )

        # --- OCR / text overlay ---
        if m.ocr_area_ratio is not None and m.ocr_area_ratio > 0.05:
            recommendations.append(
                f"Significant text overlay ({m.ocr_area_ratio:.1%} of frame). "
                "Burnt-in text confuses generation; crop or discard."
            )

        # --- Warping error (flickering) ---
        if m.warping_error is not None and m.warping_error > 15.0:
            recommendations.append(
                f"High warping error ({m.warping_error:.1f}). "
                "Indicates temporal flickering or lighting strobes."
            )

        # --- VQA scores ---
        if m.fast_vqa_score is not None and m.fast_vqa_score < 30:
            recommendations.append(
                f"Low perceptual quality (FAST-VQA={m.fast_vqa_score:.0f}/100). "
                "Deep model confirms poor quality; strongly consider discarding."
            )

        # --- Compression artifacts ---
        if m.compression_artifacts is not None and m.compression_artifacts > 70:
            recommendations.append(
                f"Heavy compression artifacts ({m.compression_artifacts:.0f}/100). "
                "Source a higher-bitrate version if available."
            )

        # --- Technical composite ---
        if m.technical_score is not None and m.technical_score < 30:
            recommendations.append(
                f"Very low technical score ({m.technical_score:.0f}/100). "
                "Multiple technical issues present; discard unless rare content."
            )

        # --- Cross-metric interaction: dark + low contrast = underexposed ---
        if (
            m.brightness is not None
            and m.contrast is not None
            and m.brightness < 60
            and m.contrast < 30
        ):
            recommendations.append(
                "Combined underexposure + low contrast suggests a severely underlit scene. "
                "Unlikely to train useful features."
            )

        if not recommendations:
            return sample

        combined = " | ".join(recommendations)
        sample.validation_issues.append(
            ValidationIssue(
                severity=ValidationSeverity.INFO,
                message=f"Advisor: {combined}",
                details={"source": "metric_advisor", "recommendation_count": len(recommendations)},
            )
        )

        return sample
