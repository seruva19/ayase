"""Dataset scanning and file discovery."""

import json
import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

from .models import CaptionMetadata, Sample

logger = logging.getLogger(__name__)

# Supported file extensions
VIDEO_EXTENSIONS = {".mp4", ".webm", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
CAPTION_EXTENSIONS = {".txt", ".caption", ".json"}


class DatasetScanner:
    """Scanner for discovering files in a dataset."""

    def __init__(
        self,
        dataset_path: Path,
        include_videos: bool = True,
        include_images: bool = True,
        recursive: bool = True,
    ):
        """Initialize the scanner.

        Args:
            dataset_path: Root path of the dataset
            include_videos: Whether to include video files
            include_images: Whether to include image files
            recursive: Whether to scan recursively
        """
        self.dataset_path = dataset_path
        self.include_videos = include_videos
        self.include_images = include_images
        self.recursive = recursive

    def scan(self) -> Iterator[Sample]:
        """Scan the dataset and yield samples.

        Yields:
            Sample objects for each discovered file
        """
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")

        if not self.dataset_path.is_dir():
            raise NotADirectoryError(f"Dataset path is not a directory: {self.dataset_path}")

        logger.info(f"Scanning dataset: {self.dataset_path}")

        # Collect all media files
        media_files = self._discover_media_files()
        caption_map, caption_stem_map = self._build_caption_map()

        for media_file in media_files:
            is_video = media_file.suffix.lower() in VIDEO_EXTENSIONS

            # Look for associated caption
            caption_path = self._find_caption(media_file, caption_map, caption_stem_map)
            caption = self._load_caption(caption_path) if caption_path else None

            sample = Sample(
                path=media_file,
                is_video=is_video,
                caption=caption,
            )

            yield sample

    def _discover_media_files(self) -> List[Path]:
        """Discover all media files in the dataset.

        Returns:
            List of media file paths
        """
        allowed_extensions: Set[str] = set()

        if self.include_videos:
            allowed_extensions.update(VIDEO_EXTENSIONS)
        if self.include_images:
            allowed_extensions.update(IMAGE_EXTENSIONS)

        media_files = []
        pattern = "**/*" if self.recursive else "*"

        for file_path in self.dataset_path.glob(pattern):
            if file_path.is_symlink():
                continue
            if file_path.is_file() and file_path.suffix.lower() in allowed_extensions:
                media_files.append(file_path)

        logger.info(f"Discovered {len(media_files)} media files")
        return sorted(media_files)

    def _build_caption_map(self) -> Tuple[Dict[str, Path], Dict[str, List[Path]]]:
        """Build a map of media files to caption files.

        Returns:
            Tuple of:
            - exact map of dataset-relative stem path to caption file
            - fallback map of filename stem to caption file candidates
        """
        caption_map: Dict[str, Path] = {}
        caption_stem_map: Dict[str, List[Path]] = {}
        pattern = "**/*" if self.recursive else "*"

        for file_path in self.dataset_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in CAPTION_EXTENSIONS:
                # Primary key: full dataset-relative path without extension.
                rel_stem = file_path.relative_to(self.dataset_path).with_suffix("").as_posix()
                caption_map[rel_stem] = file_path

                # Fallback key: basename stem for legacy/simple layouts.
                stem = file_path.stem
                caption_stem_map.setdefault(stem, []).append(file_path)

        logger.debug(f"Found {len(caption_map)} caption files")
        return caption_map, caption_stem_map

    def _find_caption(
        self,
        media_file: Path,
        caption_map: Dict[str, Path],
        caption_stem_map: Dict[str, List[Path]],
    ) -> Optional[Path]:
        """Find caption file for a media file.

        Args:
            media_file: Path to media file
            caption_map: Map of dataset-relative stems to caption paths
            caption_stem_map: Fallback map of filename stem to caption candidates

        Returns:
            Path to caption file if found, None otherwise
        """
        rel_stem = media_file.relative_to(self.dataset_path).with_suffix("").as_posix()
        if rel_stem in caption_map:
            return caption_map[rel_stem]

        stem = media_file.stem
        candidates = caption_stem_map.get(stem, [])
        if len(candidates) == 1:
            return candidates[0]
        return None

    def _load_caption(self, caption_path: Path) -> Optional[CaptionMetadata]:
        """Load caption metadata from text/caption/json sidecar."""
        try:
            suffix = caption_path.suffix.lower()
            text = ""

            if suffix in {".txt", ".caption"}:
                text = caption_path.read_text(encoding="utf-8").strip()
            elif suffix == ".json":
                data = json.loads(caption_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    for key in ("text", "caption", "prompt", "description"):
                        value = data.get(key)
                        if isinstance(value, str) and value.strip():
                            text = value.strip()
                            break
                elif isinstance(data, str):
                    text = data.strip()

            if not text:
                return None

            return CaptionMetadata(
                text=text,
                length=len(text),
                source_file=caption_path,
            )
        except Exception as e:
            logger.warning(f"Failed to parse caption file {caption_path}: {e}")
            return None


def scan_dataset(
    dataset_path: Path,
    include_videos: bool = True,
    include_images: bool = True,
    recursive: bool = True,
) -> List[Sample]:
    """Convenience function to scan a dataset.

    Args:
        dataset_path: Root path of the dataset
        include_videos: Whether to include video files
        include_images: Whether to include image files
        recursive: Whether to scan recursively

    Returns:
        List of discovered samples
    """
    scanner = DatasetScanner(
        dataset_path=dataset_path,
        include_videos=include_videos,
        include_images=include_images,
        recursive=recursive,
    )

    return list(scanner.scan())
