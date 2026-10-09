"""Configuration management for Ayase."""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, cast
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


import logging

_log = logging.getLogger(__name__)

DEFAULT_ASSETS_REPO = "AkaneTendo25/ayase-assets"
_DEFAULT_ASSETS_RESOLVE_PREFIX = f"https://huggingface.co/{DEFAULT_ASSETS_REPO}/resolve/"
_HF_REPO_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _normalize_assets_repo(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("assets_repo must be a Hugging Face repository id or URL")

    candidate = value
    if candidate.startswith("https://"):
        parsed = urlsplit(candidate)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "huggingface.co"
            or parsed.port is not None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("assets_repo URL must identify a Hugging Face repository")
        candidate = parsed.path.strip("/")
    elif "://" in candidate:
        raise ValueError("assets_repo only supports https://huggingface.co URLs")

    parts = candidate.split("/")
    if (
        len(parts) != 2
        or len(candidate) > 96
        or any(part in ("", ".", "..") for part in parts)
        or any(not _HF_REPO_COMPONENT.fullmatch(part) for part in parts)
        or any(part.startswith(("-", ".")) for part in parts)
        or any(part.endswith(("-", ".")) for part in parts)
        or any("--" in part or ".." in part for part in parts)
    ):
        raise ValueError("assets_repo must be an owner/repository Hugging Face id")
    return "/".join(parts)


def resolve_assets_repo(config: Optional[Mapping[str, Any]] = None) -> str:
    """Return the configured Ayase asset repository as an ``owner/repo`` id."""

    value = (
        DEFAULT_ASSETS_REPO if config is None else config.get("assets_repo", DEFAULT_ASSETS_REPO)
    )
    return _normalize_assets_repo(value)


def resolve_assets_url(url: str, config: Optional[Mapping[str, Any]] = None) -> str:
    """Point a default Ayase asset URL at the configured Hugging Face repo.

    Only the canonical default repository's ``/resolve/`` prefix is replaced.
    The revision, file path, and query string remain byte-for-byte unchanged.
    """

    if not url.startswith(_DEFAULT_ASSETS_RESOLVE_PREFIX):
        return url
    repo_id = resolve_assets_repo(config)
    return (
        f"https://huggingface.co/{repo_id}/resolve/" f"{url[len(_DEFAULT_ASSETS_RESOLVE_PREFIX):]}"
    )


def download_model_file(relative_path: str, url: str, models_dir: str = "models") -> Path:
    """Download a model file to ``models_dir/relative_path`` if not present.

    Returns the local ``Path`` to the downloaded (or already cached) file.
    """
    base = Path(models_dir).resolve()
    dest = (base / relative_path).resolve()
    try:
        dest.resolve().relative_to(base.resolve())
    except ValueError:
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


def download_torch_hub_checkpoint(filename: str, url: str, models_dir: str = "models") -> Path:
    """Populate torch.hub's checkpoint cache from an Ayase-controlled URL."""
    if Path(filename).name != filename:
        raise ValueError(f"Torch Hub checkpoint must be a basename: {filename!r}")
    return download_model_file(f"hub/checkpoints/{filename}", url, models_dir)


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


def download_hf_snapshot(
    repo_id: str,
    models_dir: str = "models",
    *,
    revision: Optional[str] = None,
    allow_patterns: Optional[List[str]] = None,
    ignore_patterns: Optional[List[str]] = None,
) -> Path:
    """Download a Hugging Face repository into Ayase's model directory.

    The stable, human-readable destination is ``models_dir/org--model``.  An
    already complete snapshot is reused by ``huggingface_hub``; interrupted
    downloads resume through its normal local-directory metadata.
    """
    if not repo_id or repo_id.startswith((".", "/", "\\")) or ".." in repo_id.split("/"):
        raise ValueError(f"Invalid Hugging Face repository id: {repo_id!r}")
    destination = (Path(models_dir).resolve() / repo_id.replace("/", "--")).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    _log.info("Resolving Hugging Face snapshot %s → %s", repo_id, destination)
    resolved = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(destination),
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )
    return Path(resolved).resolve()


class GeneralConfig(BaseModel):
    """General configuration settings.

    Plain :class:`BaseModel` (not ``BaseSettings``) so that building this
    section via ``default_factory`` never reads bare, unprefixed environment
    variables (e.g. ``DEVICE``, ``PARALLEL_JOBS``). The only environment
    channel is the ``AYASE_*`` allowlist handled by
    :meth:`AyaseConfig._load_env_overrides`.
    """

    parallel_jobs: int = 8
    cache_enabled: bool = True
    cache_dir: Path = Path.home() / ".cache" / "ayase"
    models_dir: Path = Path("models")
    assets_repo: str = DEFAULT_ASSETS_REPO
    device: str = "auto"
    dtype: str = "auto"
    amp_enabled: bool = True
    attention_backend: str = "auto"
    frame_cache_enabled: bool = True
    timing_enabled: bool = True
    sample_batch_size: int = 1
    max_clip_images_per_forward: int = 64

    @field_validator("assets_repo", mode="before")
    @classmethod
    def normalize_assets_repo(cls, value: Any) -> str:
        """Normalize supported Hugging Face repository URLs to repository ids."""

        return _normalize_assets_repo(value)


class QualityConfig(BaseModel):
    """Quality assessment configuration."""

    enable_blur_detection: bool = True
    blur_threshold: float = 100.0
    enable_compression_check: bool = True
    min_caption_length: int = 10
    max_caption_length: int = 1000


class OutputConfig(BaseModel):
    """Output configuration."""

    default_format: str = "markdown"
    show_progress: bool = True
    color_output: bool = True
    artifacts_dir: Path = Path("reports")
    artifacts_format: str = "json"


class PipelineConfig(BaseModel):
    """Pipeline configuration."""

    dataset_path: Optional[Path] = None
    modules: List[str] = Field(default_factory=list)
    plugin_folders: List[Path] = Field(default_factory=lambda: [Path("plugins")])
    # Provenance classes a pipeline is allowed to run beyond the defaults
    # ("published", "utility"). Set e.g. ["adapted", "own"] to opt in to
    # metrics that deviate from their published source or are Ayase-specific.
    allow_provenance: List[str] = Field(default_factory=list)


class FilterConfig(BaseModel):
    """Filter configuration."""

    default_mode: str = "list"
    min_score_threshold: float = 60.0


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
            return cast(Dict[str, Any], tomllib.load(f))

    @classmethod
    def _load_env_overrides(cls) -> Dict[str, Any]:
        """Read ``AYASE_*`` env vars into a nested dict understood by Pydantic.

        Only variables targeting real top-level config sections are considered so
        runtime env vars like ``AYASE_TEST_MODE`` do not get treated as config.
        """
        valid_sections = set(cls.model_fields.keys())
        nested: Dict[str, Any] = {}

        for key, value in os.environ.items():
            if not key.startswith("AYASE_"):
                continue

            parts = key[len("AYASE_") :].lower().split("__")
            if not parts or parts[0] not in valid_sections:
                continue

            cursor: Dict[str, Any] = nested
            for part in parts[:-1]:
                child = cursor.get(part)
                if not isinstance(child, dict):
                    child = {}
                    cursor[part] = child
                cursor = child
            parsed: Any = value
            stripped = value.lstrip()
            if stripped.startswith("[") or stripped.startswith("{"):
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    parsed = value
            cursor[parts[-1]] = parsed

        return nested

    @staticmethod
    def _merge_nested(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge nested dicts, preferring override values."""
        merged = dict(base)
        for key, value in overrides.items():
            current = merged.get(key)
            if isinstance(current, dict) and isinstance(value, dict):
                merged[key] = AyaseConfig._merge_nested(current, value)
            else:
                merged[key] = value
        return merged

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "AyaseConfig":
        """Load configuration from file or defaults.

        An explicitly supplied ``config_path`` that does not exist is a hard
        error: silently falling back to defaults would run the whole pipeline
        misconfigured after a typo'd ``--config``. The implicit ``./ayase.toml``
        default location may still be absent and fall back to built-in defaults.
        """
        file_data: Dict[str, Any] = {}
        if config_path is not None:
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            file_data = cls._load_toml(config_path)
        else:
            # Try default locations
            default_paths = [
                Path("ayase.toml"),
                Path.home() / ".config" / "ayase" / "config.toml",
            ]

            for path in default_paths:
                if path.exists():
                    file_data = cls._load_toml(path)
                    break

        merged = cls._merge_nested(file_data, cls._load_env_overrides())

        # Validate explicit data only; defaults are filled by the model itself.
        return cast("AyaseConfig", cls.model_validate(merged))

    @staticmethod
    def _toml_safe(value: Any) -> Any:
        """Recursively convert a value into something ``tomli_w`` can serialize.

        ``tomli_w`` cannot serialize ``Path`` objects or ``None`` (TOML has no
        null type). Paths are stringified and ``None`` values are dropped —
        omitted keys simply fall back to their model defaults on reload, so a
        round-trip of the default config is stable.
        """
        if isinstance(value, dict):
            result: Dict[str, Any] = {}
            for key, val in value.items():
                if val is None:
                    continue
                result[key] = AyaseConfig._toml_safe(val)
            return result
        if isinstance(value, (list, tuple)):
            return [AyaseConfig._toml_safe(item) for item in value if item is not None]
        if isinstance(value, Path):
            return str(value)
        return value

    def save(self, config_path: Path) -> None:
        """Save configuration to TOML file."""
        import tomli_w

        config_path.parent.mkdir(parents=True, exist_ok=True)
        data = self._toml_safe(self.model_dump())
        with open(config_path, "wb") as f:
            tomli_w.dump(data, f)
