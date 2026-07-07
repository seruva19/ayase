"""ArtFID — Artistic Style Transfer FID (2022).

Full-reference metric for evaluating style transfer quality. Combines content
fidelity (LPIPS) with style similarity (FID over Inception features).

Requires the ``art-fid`` package (``pip install art-fid``). ArtFID is a
distribution/FID-based metric, so it is evaluated over the *sets* of frames
sampled from the sample and its reference. When ``art-fid`` is not installed the
metric is left unset — there is no heuristic approximation.

artfid_score — lower = better (combined content + style distance).
"""

import logging
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from ayase.image import sample_frames
from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)


class ArtFIDModule(ReferenceBasedModule):
    name = "artfid"
    description = "ArtFID style transfer quality (FR, 2022, lower=better; requires art-fid)"
    metric_field = "artfid_score"
    default_config = {"subsample": 8}
    metric_groups = {
        "artfid_score": "fr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._art_fid = None
        self._ml_available = False
        self._backend = "unavailable"
        self._device = "cpu"
        self.subsample = self.config.get("subsample", 8)

    def setup(self) -> None:
        from ayase.runtime import resolve_torch_device

        self._device = resolve_torch_device(self.config.get("device", "auto"))
        try:
            import art_fid

            self._art_fid = art_fid
            self._ml_available = True
            self._backend = "package:art_fid"
            logger.info("ArtFID initialised (art-fid package) on %s", self._device)
        except ImportError:
            logger.warning(
                "ArtFID unavailable: the 'art-fid' package is not installed "
                "(pip install art-fid); artfid_score will be left unset."
            )
        except Exception as e:
            logger.warning("ArtFID setup failed: %s", e)

    def compute_reference_score(self, sample_path: Path, reference_path: Path) -> Optional[float]:
        if not self._ml_available or self._art_fid is None:
            return None

        tmp_root: Optional[Path] = None
        try:
            styl_frames = self._load_frames(sample_path)
            ref_frames = self._load_frames(reference_path)
            if not styl_frames or not ref_frames:
                return None

            tmp_root = Path(tempfile.mkdtemp(prefix="ayase_artfid_"))
            styl_dir = tmp_root / "stylized"
            style_dir = tmp_root / "style"
            content_dir = tmp_root / "content"
            self._dump_frames(styl_frames, styl_dir)
            # Only a single reference is available; ArtFID expects a content
            # source and a style source, so the reference frame set is used for
            # both (a reference-based interpretation of ArtFID).
            self._dump_frames(ref_frames, style_dir)
            self._dump_frames(ref_frames, content_dir)

            score = self._art_fid.compute_art_fid(
                str(styl_dir),
                str(style_dir),
                str(content_dir),
                device=str(self._device),
            )
            return float(score)
        except Exception as e:
            logger.warning("ArtFID computation failed: %s", e)
            return None
        finally:
            if tmp_root is not None:
                shutil.rmtree(tmp_root, ignore_errors=True)

    def _load_frames(self, path: Path) -> List[np.ndarray]:
        try:
            return sample_frames(path, max_frames=self.subsample, color="rgb")
        except Exception as e:
            logger.debug("ArtFID frame load failed for %s: %s", path, e)
            return []

    def _dump_frames(self, frames: List[np.ndarray], out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            # frames are RGB read-only views; make a writable BGR copy for cv2.
            bgr = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_dir / f"{i:05d}.png"), bgr)
