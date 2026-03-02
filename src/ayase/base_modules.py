"""Base classes for specialized pipeline modules."""

from abc import abstractmethod
from pathlib import Path
from typing import List, Optional

from ayase.pipeline import PipelineModule
from ayase.models import Sample, QualityMetrics


class ReferenceBasedModule(PipelineModule):
    """Base class for full-reference quality metrics.

    Full-reference metrics require a reference video/image for comparison.
    Examples: VMAF, MS-SSIM, VIF, PSNR

    The reference should be specified in sample metadata as 'reference_path'.
    If no reference is available, the module should skip processing gracefully.
    """

    @abstractmethod
    def compute_reference_score(self, sample_path: Path, reference_path: Path) -> Optional[float]:
        """Compute quality score by comparing sample against reference.

        Args:
            sample_path: Path to the sample video/image
            reference_path: Path to the reference video/image

        Returns:
            Quality score, or None if computation failed
        """
        pass

    # Subclasses can set this to the QualityMetrics field name for automatic storage.
    # e.g. metric_field = "vmaf"
    metric_field: Optional[str] = None

    def process(self, sample: Sample) -> Sample:
        """Process sample with reference-based metric.

        Checks for reference_path in sample metadata. If not found, skips processing.
        Subclasses should either override this method or set ``metric_field`` to
        automatically store the computed score.
        """
        reference = getattr(sample, "reference_path", None)
        if reference is None:
            return sample

        if not isinstance(reference, Path):
            reference = Path(reference)

        if not reference.exists():
            return sample

        score = self.compute_reference_score(sample.path, reference)

        if score is not None and self.metric_field:
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            setattr(sample.quality_metrics, self.metric_field, score)

        return sample


class BatchMetricModule(PipelineModule):
    """Base class for batch/distribution metrics (FVD, KVD, FID, etc.).

    These metrics compare distributions of features between two sets of samples,
    rather than evaluating individual samples. They accumulate features during
    processing and compute the final metric after all samples are processed.

    Dataset-level metrics are stored in pipeline stats, not individual samples.
    """

    def __init__(self, config=None):
        """Initialize batch metric module."""
        super().__init__(config)
        self._feature_cache: List = []  # Accumulate features across samples
        self._reference_cache: List = []  # Reference features if available
        self.pipeline = None  # Will be set by Pipeline during initialization

    @abstractmethod
    def extract_features(self, sample: Sample) -> Optional[object]:
        """Extract features from a sample for distribution comparison.

        Args:
            sample: Sample to extract features from

        Returns:
            Feature representation (e.g., embedding vector), or None if extraction failed
        """
        pass

    @abstractmethod
    def compute_distribution_metric(
        self, features: List, reference_features: Optional[List] = None
    ) -> float:
        """Compute distribution metric between feature sets.

        Args:
            features: List of feature representations from samples
            reference_features: Optional list of reference features for comparison.
                              If None, assumes synthetic vs real comparison or self-comparison.

        Returns:
            Distribution metric score (e.g., FVD, KVD)
        """
        pass

    def on_mount(self) -> None:
        """Called when module is loaded. Register as batch module."""
        super().on_mount()

        # Register with pipeline if available
        if hasattr(self, "pipeline") and self.pipeline:
            if hasattr(self.pipeline, "register_batch_module"):
                self.pipeline.register_batch_module(self)

    def process(self, sample: Sample) -> Sample:
        """Extract and cache features from sample.

        Does not modify the sample directly. Features are accumulated for
        batch computation in on_dispose().
        """
        features = self.extract_features(sample)
        if features is not None:
            self._feature_cache.append(features)

        # Check if sample has reference for paired comparison
        reference_path = getattr(sample, "reference_path", None)
        if reference_path is not None:
            try:
                reference_path = Path(reference_path)
                if reference_path.exists():
                    reference_sample = Sample(
                        path=reference_path,
                        is_video=sample.is_video,
                        video_metadata=sample.video_metadata,
                        image_metadata=sample.image_metadata,
                        audio_metadata=sample.audio_metadata,
                        caption=sample.caption,
                    )
                    extractor = getattr(self, "extract_reference_features", self.extract_features)
                    ref_features = extractor(reference_sample)
                    if ref_features is not None:
                        self._reference_cache.append(ref_features)
            except Exception:
                # Reference features are optional for many batch metrics.
                pass

        return sample

    def on_dispose(self) -> None:
        """Compute batch metric after all samples processed.

        This is called when the pipeline finishes processing all samples.
        The computed metric should be stored in pipeline-level stats.
        """
        if len(self._feature_cache) < 2:
            # Not enough samples for distribution metric
            self._feature_cache = []
            self._reference_cache = []
            return

        try:
            score = self.compute_distribution_metric(
                self._feature_cache, self._reference_cache if self._reference_cache else None
            )

            # Store in pipeline stats (implementation will be added in pipeline.py)
            if hasattr(self, "pipeline") and self.pipeline:
                # Pipeline will have a method to store dataset-level metrics
                if hasattr(self.pipeline, "add_dataset_metric"):
                    metric_name = getattr(self, "name", self.__class__.__name__)
                    self.pipeline.add_dataset_metric(metric_name, score)

        except Exception as e:
            # Graceful failure
            import logging

            logging.warning(f"Failed to compute batch metric: {e}")

        finally:
            # Clean up cache
            self._feature_cache = []
            self._reference_cache = []


class NoReferenceModule(PipelineModule):
    """Base class for no-reference quality metrics.

    No-reference (NR) metrics assess quality without needing a reference.
    Examples: NIQE, BRISQUE, Scene Complexity, Naturalness

    These are typically the most practical for real-world datasets where
    references are not available.
    """

    # Subclasses can set this to the QualityMetrics field name for automatic storage.
    # e.g. metric_field = "niqe"
    metric_field: Optional[str] = None

    @abstractmethod
    def compute_nr_score(self, sample_path: Path) -> Optional[float]:
        """Compute no-reference quality score.

        Args:
            sample_path: Path to the sample video/image

        Returns:
            Quality score, or None if computation failed
        """
        pass

    def process(self, sample: Sample) -> Sample:
        """Process sample with no-reference metric.

        Subclasses should either override this method or set ``metric_field`` to
        automatically store the computed score.
        """
        score = self.compute_nr_score(sample.path)

        if score is not None and self.metric_field:
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            setattr(sample.quality_metrics, self.metric_field, score)

        return sample
