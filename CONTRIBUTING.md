# Contributing to Ayase

Thank you for your interest in contributing to Ayase! This guide covers everything you need to get started.

## Prerequisites

- **Python 3.9+**
- **git**

## Development Setup

1. Fork the repository and clone your fork:

   ```bash
   git clone https://github.com/<your-username>/ayase.git
   cd ayase
   ```

2. Install the package in editable mode with development dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

3. Verify the installation by running the test suite:

   ```bash
   pytest tests/
   ```

## Code Style

This project enforces consistent code style through the following tools:

- **Black** with `line-length=100`
- **Ruff** for linting
- **MyPy** for type checking on core files

Before submitting a pull request, make sure your code passes all checks:

```bash
black --check --line-length=100 src/
ruff check src/
mypy src/ayase/config.py src/ayase/models.py src/ayase/pipeline.py src/ayase/profile.py --ignore-missing-imports
```

## Running Tests

Run the full test suite with:

```bash
pytest tests/
```

To run a specific test file or test case:

```bash
pytest tests/test_specific.py
pytest tests/test_specific.py::test_function_name
```

## Writing a New Module

To add a new pipeline module:

1. Create a new file under the appropriate package directory.
2. Inherit from `PipelineModule`.
3. Implement the `process()` method with your module's logic.
4. Register the module in the package's `__init__.py`.

Example skeleton:

```python
import logging
from ayase.pipeline import PipelineModule
from ayase.models import Sample, QualityMetrics

logger = logging.getLogger(__name__)


class MyNewModule(PipelineModule):
    name = "my_new_module"
    description = "Brief description of what this module does"
    default_config = {}

    def process(self, sample: Sample) -> Sample:
        # Your processing logic here
        return sample
```

The module is auto-registered by setting `name`. Optionally, add it to `modules/__init__.py` for convenience imports.

## Writing a Plugin

Plugins follow a simpler workflow than built-in modules:

1. Create a `.py` file in the `plugins/` folder.
2. Subclass `PipelineModule` and set `name` -- Ayase discovers and registers it automatically.

```python
# plugins/my_plugin.py
from ayase.pipeline import PipelineModule
from ayase.models import Sample


class MyPlugin(PipelineModule):
    name = "my_plugin"
    description = "Custom plugin"

    def process(self, sample: Sample) -> Sample:
        # Your logic here
        return sample
```

## Pull Request Process

1. **Fork** the repository on GitHub.
2. **Create a branch** from `main` for your changes:
   ```bash
   git checkout -b my-feature main
   ```
3. Make your changes, ensuring code style checks and tests pass.
4. **Test** thoroughly:
   ```bash
   pytest tests/
   ```
5. Push your branch and open a **Pull Request against `main`**.
6. Fill in the PR description with a summary of changes and any relevant context.
7. Address review feedback promptly.

## Commit Message Convention

- Use **imperative mood** in the subject line (e.g., "Add module for X" not "Added module for X").
- Keep the subject line concise (72 characters or fewer).
- Optionally include a body separated by a blank line for additional context.

Examples:

```
Add CLIP-based similarity module

Implement a new pipeline module that computes CLIP similarity
scores between generated and reference images.
```

```
Fix edge case in BRISQUE metric for grayscale input
```

---

If you have questions or need help, feel free to open an issue or start a discussion. We appreciate every contribution!
