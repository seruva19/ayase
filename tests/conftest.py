"""Root test configuration for Ayase.

Two test modes:
    pytest tests/              # Light mode (default): heuristic backends only, fast
    pytest tests/ --full       # Full mode: loads real ML models, slow but thorough

Light mode sets AYASE_TEST_MODE=1 globally so all modules skip heavy
ML model downloads and use heuristic fallbacks. Full mode leaves it off
so modules load real weights (requires GPU + model downloads).
"""

import pytest

from ayase.pipeline import PipelineModule


def pytest_addoption(parser):
    parser.addoption(
        "--full",
        action="store_true",
        default=False,
        help="Run in full mode with real ML model loading (slow). Default is light/heuristic mode.",
    )


@pytest.fixture(autouse=True, scope="session")
def _set_test_mode(request):
    """Enable test_mode globally unless --full is passed."""
    full = request.config.getoption("--full")
    if not full:
        PipelineModule.set_test_mode(True)
    yield
    PipelineModule.set_test_mode(False)
