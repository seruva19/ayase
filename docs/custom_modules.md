# Custom modules and plugins

[Back to README](../README.md)

Subclass `PipelineModule` to add a check. This example operates on still images
and writes the existing `blur_score` field:

```python
from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule
import cv2

class BlurCheck(PipelineModule):
    name = "blur_check"
    description = "Flag blurry frames via Laplacian variance"
    default_config = {"threshold": 100.0}

    def process(self, sample: Sample) -> Sample:
        img = cv2.imread(str(sample.path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return sample
        score = float(cv2.Laplacian(img, cv2.CV_64F).var())
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.blur_score = score
        if score < self.config.get("threshold", 100.0):
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Blurry ({score:.0f})",
                )
            )
        return sample
```

Modules auto-register via `__init_subclass__`. Config is available as `self.config`.

For an external plugin, save the module in a `.py` file under a configured
plugin folder (the default is `plugins`). Ayase discovers it automatically;
no source-package registration is needed. Always return the sample, including
when input cannot be processed, and log recoverable failures.

New output fields must be declared in `QualityMetrics` and its field groups.
See [the contributor guide](../CONTRIBUTING.md) for built-in modules and tests.
