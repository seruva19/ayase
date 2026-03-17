"""Configuration management for Ayase."""

from pathlib import Path
from typing import Any, Dict, Optional, List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


import logging

_log = logging.getLogger(__name__)


def download_model_file(relative_path: str, url: str, models_dir: str = "models") -> Path:
    """Download a model file to ``models_dir/relative_path`` if not present.

    Returns the local ``Path`` to the downloaded (or already cached) file.
    """
    base = Path(models_dir).resolve()
    dest = (base / relative_path).resolve()
    if not str(dest).startswith(str(base)):
        raise ValueError(f"Path traversal detected: {relative_path!r} escapes {models_dir!r}")
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)

    _log.info("Downloading %s → %s", url, dest)
    import urllib.request
    import shutil

    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=300) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f)
        tmp.rename(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    _log.info("Downloaded %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


def resolve_model_path(model_name: str, models_dir: str = "models") -> str:
    """Resolve a HuggingFace model name to a local path if available.

    Checks ``models_dir/model_name`` and the ``--``-delimited variant
    (e.g. ``models/openai--clip-vit-base-patch32``).  Falls back to the
    original *model_name* so that ``transformers`` downloads from the Hub.
    """
    base = Path(models_dir)
    # Direct subpath: models/openai/clip-vit-base-patch32
    local = base / model_name
    if local.is_dir():
        return str(local)
    # HF cache style: models/openai--clip-vit-base-patch32
    flat = base / model_name.replace("/", "--")
    if flat.is_dir():
        return str(flat)
    # Not cached locally — return original name for Hub download
    return model_name


class GeneralConfig(BaseSettings):
    """General configuration settings."""

    parallel_jobs: int = 8
    cache_enabled: bool = True
    cache_dir: Path = Path.home() / ".cache" / "ayase"
    models_dir: Path = Path("models")


class QualityConfig(BaseSettings):
    """Quality assessment configuration."""

    enable_blur_detection: bool = True
    blur_threshold: float = 100.0
    enable_compression_check: bool = True
    min_caption_length: int = 10
    max_caption_length: int = 1000


class OutputConfig(BaseSettings):
    """Output configuration."""

    default_format: str = "markdown"
    show_progress: bool = True
    color_output: bool = True
    artifacts_dir: Path = Path("reports")
    artifacts_format: str = "json"


class PipelineConfig(BaseSettings):
    """Pipeline configuration."""

    dataset_path: Optional[Path] = None
    modules: List[str] = Field(default_factory=list)
    plugin_folders: List[Path] = Field(default_factory=lambda: [Path("plugins")])


class FilterConfig(BaseSettings):
    """Filter configuration."""

    default_mode: str = "list"
    min_score_threshold: int = 60


class AyaseConfig(BaseSettings):
    """Main Ayase configuration."""

    model_config = SettingsConfigDict(
        env_prefix="AYASE_",
        env_nested_delimiter="__",
    )

    general: GeneralConfig = Field(default_factory=GeneralConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    filter: FilterConfig = Field(default_factory=FilterConfig)

    @staticmethod
    def _load_toml(path: Path) -> Dict[str, Any]:
        """Read a TOML file and return its contents as a dict."""
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib

        with open(path, "rb") as f:
            return tomllib.load(f)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "AyaseConfig":
        """Load configuration from file or defaults."""
        if config_path and config_path.exists():
            return cls(**cls._load_toml(config_path))

        # Try default locations
        default_paths = [
            Path("ayase.toml"),
            Path.home() / ".config" / "ayase" / "config.toml",
        ]

        for path in default_paths:
            if path.exists():
                return cls(**cls._load_toml(path))

        # Return default config
        return cls()

    def save(self, config_path: Path) -> None:
        """Save configuration to TOML file."""
        import tomli_w

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "wb") as f:
            tomli_w.dump(self.model_dump(), f)
