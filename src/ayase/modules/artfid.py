"""ArtFID — Artistic Style Transfer FID (Wright & Ommer, 2022).

Full-reference metric for evaluating style transfer quality. Combines content
fidelity (LPIPS against the content set) with style similarity (FID between
stylized outputs and the style set), as implemented by the official
``art-fid`` package.

The published protocol needs two *separate* reference sets: the content source
(``sample.reference_path``) and the style source
(``sample.style_reference_path``). If either is missing the metric is left
unset — reusing one set for both roles does not measure ArtFID.

Requires ``pip install art-fid``. Frames are taken from the full sample /
reference videos by default (``subsample`` is an explicit cap; FID over a
handful of frames is statistically meaningless).

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
    provenance = "published"
    sources = {
        "artfid_score": "ArtFID (Wright & Ommer, 2022), art-fid package — https://github.com/matthias-wright/art-fid",
    }
    description = "ArtFID style transfer quality (FR, 2022, lower=better; requires art-fid)"
    metric_field = "artfid_score"
    default_config = {"subsample": 0}
    metric_groups = {
        "artfid_score": "fr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._art_fid = None
        self._ml_available = False
        self._backend = "unavailable"
        self._device = "cpu"
        self.subsample = self.config.get("subsample", 0)

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

    def process(self, sample):
        # ArtFID requires distinct content and style references; the published
        # protocol cannot run with a single shared reference.
        if not self._ml_available:
            return sample
        reference = getattr(sample, "reference_path", None)
        style = getattr(sample, "style_reference_path", None)
        if reference is None or style is None:
            return sample
        if not Path(reference).exists() or not Path(style).exists():
            return sample

        score = self._compute_art_fid(sample.path, Path(reference), Path(style))
        if score is not None and self.metric_field:
            from ayase.models import QualityMetrics

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            setattr(sample.quality_metrics, self.metric_field, score)
        return sample

    def compute_reference_score(self, sample_path: Path, reference_path: Path) -> Optional[float]:
        # Kept for the ReferenceBasedModule interface; ArtFID also needs a
        # style reference, so the real entry point is process().
        return None

    def _compute_art_fid(
        self, sample_path: Path, content_path: Path, style_path: Path
    ) -> Optional[float]:
        tmp_root: Optional[Path] = None
        try:
            styl_frames = self._load_frames(sample_path)
            cnt_frames = self._load_frames(content_path)
            sty_frames = self._load_frames(style_path)
            if not styl_frames or not cnt_frames or not sty_frames:
                return None

            tmp_root = Path(tempfile.mkdtemp(prefix="ayase_artfid_"))
            styl_dir = tmp_root / "stylized"
            style_dir = tmp_root / "style"
            content_dir = tmp_root / "content"
            self._dump_frames(styl_frames, styl_dir)
            self._dump_frames(sty_frames, style_dir)
            self._dump_frames(cnt_frames, content_dir)

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
            # subsample <= 0 means all frames: the FID statistics need the full
            # distribution, so the cap is effectively unbounded.
            max_frames = self.subsample if self.subsample > 0 else 10**9
            return sample_frames(path, max_frames=max_frames, color="rgb")
        except Exception as e:
            logger.debug("ArtFID frame load failed for %s: %s", path, e)
            return []

    def _dump_frames(self, frames: List[np.ndarray], out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            # frames are RGB read-only views; make a writable BGR copy for cv2.
            bgr = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_dir / f"{i:05d}.png"), bgr)
