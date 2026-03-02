"""Example Ayase plugin — drop .py files like this into a plugin folder.

Ayase auto-discovers any PipelineModule subclass with a unique ``name``.
Place this file (or your own) in a folder listed under
``pipeline.plugin_folders`` in ayase.toml, or in the default ``plugins/``
directory next to your project.

Usage:
    ayase scan ./my_dataset --modules metadata,example
"""

from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule


class ExamplePlugin(PipelineModule):
    name = "example"
    description = "Example plugin that logs sample paths (template for custom plugins)"
    default_config = {
        "log_valid": True,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._count = 0

    def process(self, sample: Sample) -> Sample:
        self._count += 1

        if not sample.path.exists():
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"File not found: {sample.path}",
                )
            )
            sample.is_valid = False

        return sample

    def post_process(self, all_samples):
        if self.config.get("log_valid"):
            valid = sum(1 for s in all_samples if s.is_valid)
            print(f"[example plugin] Processed {self._count} samples, {valid} valid")
