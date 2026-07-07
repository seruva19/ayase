"""Split-screen / grid-collage detector.

The curation filter used by the HunyuanVideo-1.5 and Cosmos-Curate data
pipelines that ayase lacked: it rejects frames that are actually a grid /
collage / side-by-side composite rather than a single continuous shot.

Honest algorithmic detector (``_backend = "algorithmic"``) — no ML. A grid is
detected from two persistent, corroborating signals:

  1. **Straight internal dividers** — column/row projections of a Sobel edge
     map show a sharp, prominent peak (a full-length vertical or horizontal
     line), and that line covers most of the frame's height/width.
  2. **Content discontinuity** — the two cells on either side of the candidate
     divider differ strongly in mean colour and texture.

Both must hold, and the divider must recur at a consistent position across the
sampled frames (persistence). Letterbox / pillarbox black bars are NOT grids:
frame-boundary bars are cropped out first (same rule as ``letterbox``) and a
margin near the interior boundary is excluded, so only *internal* dividers are
considered.

``grid_layout_score`` (0-1, higher = more likely a grid). When the score is
high the detected layout (e.g. ``"2x2"``, ``"1x2"``) is stored in
``sample.metadata["grid_layout"]``.
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Detection parameters                                                         #
# --------------------------------------------------------------------------- #
BORDER_THRESHOLD = 16     # luminance below this = black bar (matches letterbox.py)
MARGIN_FRAC = 0.10        # ignore candidate dividers within this fraction of an interior edge
EDGE_THRESHOLD = 40.0     # |Sobel| above this counts as an edge pixel along a line
PEAK_PROMINENCE = 3.0     # a divider's edge-profile value must exceed median * this
PEAK_STD_K = 4.0          # ...and also exceed median + K * std of the profile
COVERAGE_MIN = 0.50       # fraction of the line length that must be a continuous edge
DISC_MIN = 0.12           # min normalised cross-boundary content difference
STRIP_FRAC = 0.05         # half-width (fraction of interior dim) of the content-compare strip
POS_TOL = 0.03            # normalised position tolerance for cross-frame persistence
PERSIST_FRAC = 0.60       # a divider must appear in >= this fraction of sampled frames
HIGH_SCORE = 0.50         # store the layout in metadata when score >= this
MIN_INTERIOR = 24         # skip interiors smaller than this (px) on either dimension


def _border_extents(gray: np.ndarray, thr: float) -> Tuple[int, int, int, int]:
    """Return (top, bottom, left, right) black-bar thickness, like letterbox.py."""
    h, w = gray.shape[:2]
    top = 0
    for r in range(h // 2):
        if float(gray[r, :].mean()) < thr:
            top = r + 1
        else:
            break
    bottom = 0
    for r in range(h - 1, h // 2, -1):
        if float(gray[r, :].mean()) < thr:
            bottom = h - r
        else:
            break
    left = 0
    for c in range(w // 2):
        if float(gray[:, c].mean()) < thr:
            left = c + 1
        else:
            break
    right = 0
    for c in range(w - 1, w // 2, -1):
        if float(gray[:, c].mean()) < thr:
            right = w - c
        else:
            break
    return top, bottom, left, right


def _find_peaks(profile: np.ndarray, margin: int) -> List[int]:
    """Local maxima of an edge profile that clear the prominence gates.

    Positions within ``margin`` of either end (residual border / interior
    boundary) are ignored so only internal dividers survive. Nearby peaks are
    suppressed within POS_TOL so one thick line yields a single candidate.
    """
    n = len(profile)
    lo, hi = margin, n - margin
    if hi - lo < 3:
        return []
    interior = profile[lo:hi]
    med = float(np.median(interior))
    std = float(np.std(interior))
    thr = max(med * PEAK_PROMINENCE, med + PEAK_STD_K * std)

    cands: List[Tuple[int, float]] = []
    for i in range(lo, hi):
        v = float(profile[i])
        if v <= thr:
            continue
        if v >= float(profile[i - 1]) and v >= float(profile[i + 1]):
            cands.append((i, v))

    cands.sort(key=lambda t: -t[1])
    tol = max(1, int(POS_TOL * n))
    kept: List[int] = []
    for i, _v in cands:
        if all(abs(i - j) > tol for j in kept):
            kept.append(i)
    return kept


def _line_coverage(abs_grad_line: np.ndarray) -> float:
    """Fraction of a candidate divider line whose gradient exceeds EDGE_THRESHOLD."""
    if abs_grad_line.size == 0:
        return 0.0
    return float(np.mean(abs_grad_line > EDGE_THRESHOLD))


def _content_discontinuity(cell_a: np.ndarray, cell_b: np.ndarray) -> float:
    """Normalised mean colour + texture difference between two adjacent cells (0-1)."""
    if cell_a.size == 0 or cell_b.size == 0:
        return 0.0
    a = cell_a.reshape(-1, cell_a.shape[-1]).astype(np.float32)
    b = cell_b.reshape(-1, cell_b.shape[-1]).astype(np.float32)
    color = float(np.abs(a.mean(axis=0) - b.mean(axis=0)).mean()) / 128.0
    texture = float(abs(a.std() - b.std())) / 64.0
    return float(min(1.0, color + 0.3 * texture))


class GridLayoutModule(PipelineModule):
    name = "grid_layout"
    description = "Split-screen/grid-collage detector (0-1, higher=more likely a grid)"
    default_config = {
        "subsample": 4,               # frames sampled per video
        "border_threshold": BORDER_THRESHOLD,
        "warn_threshold": HIGH_SCORE,  # emit a WARNING issue at/above this score
    }
    metric_info = {
        "grid_layout_score": "Split-screen/grid-collage likelihood (0-1, higher=more likely)",
    }
    metric_groups = {
        "grid_layout_score": "basic",
    }

    def __init__(self, config=None):
        super().__init__(config)
        # Pure edge-projection geometry — no ML backend.
        self._backend = "algorithmic"

    # ------------------------------------------------------------------ #
    # Per-frame analysis                                                  #
    # ------------------------------------------------------------------ #

    def _analyze_frame(self, frame_bgr: np.ndarray):
        """Return (vertical_splits, horizontal_splits) as lists of (pos_norm, strength)."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        thr = self.config.get("border_threshold", BORDER_THRESHOLD)
        top, bottom, left, right = _border_extents(gray, thr)

        h, w = gray.shape[:2]
        y0, y1 = top, h - bottom
        x0, x1 = left, w - right
        if (y1 - y0) < MIN_INTERIOR or (x1 - x0) < MIN_INTERIOR:
            return [], []

        interior_gray = gray[y0:y1, x0:x1].astype(np.float32)
        interior_bgr = frame_bgr[y0:y1, x0:x1]
        ih, iw = interior_gray.shape

        gx = np.abs(cv2.Sobel(interior_gray, cv2.CV_32F, 1, 0, ksize=3))  # vertical edges
        gy = np.abs(cv2.Sobel(interior_gray, cv2.CV_32F, 0, 1, ksize=3))  # horizontal edges
        col_profile = gx.mean(axis=0)  # length iw -> vertical dividers
        row_profile = gy.mean(axis=1)  # length ih -> horizontal dividers

        strip_v = max(2, int(STRIP_FRAC * iw))
        strip_h = max(2, int(STRIP_FRAC * ih))

        v_splits: List[Tuple[float, float]] = []
        for i in _find_peaks(col_profile, max(1, int(MARGIN_FRAC * iw))):
            coverage = _line_coverage(gx[:, i])
            if coverage < COVERAGE_MIN:
                continue
            a0, a1 = max(0, i - strip_v), i
            b0, b1 = i, min(iw, i + strip_v)
            disc = _content_discontinuity(interior_bgr[:, a0:a1], interior_bgr[:, b0:b1])
            if disc < DISC_MIN:
                continue
            v_splits.append((i / max(1, iw - 1), coverage * disc))

        h_splits: List[Tuple[float, float]] = []
        for i in _find_peaks(row_profile, max(1, int(MARGIN_FRAC * ih))):
            coverage = _line_coverage(gy[i, :])
            if coverage < COVERAGE_MIN:
                continue
            a0, a1 = max(0, i - strip_h), i
            b0, b1 = i, min(ih, i + strip_h)
            disc = _content_discontinuity(interior_bgr[a0:a1, :], interior_bgr[b0:b1, :])
            if disc < DISC_MIN:
                continue
            h_splits.append((i / max(1, ih - 1), coverage * disc))

        return v_splits, h_splits

    @staticmethod
    def _persistent_splits(per_frame: List[List[Tuple[float, float]]], n_frames: int):
        """Cluster per-frame dividers by position; keep those recurring across frames."""
        items: List[Tuple[float, float, int]] = []
        for fi, splits in enumerate(per_frame):
            for pos, strength in splits:
                items.append((pos, strength, fi))
        items.sort(key=lambda t: -t[1])  # strongest first anchors a cluster

        clusters: List[dict] = []
        for pos, strength, fi in items:
            placed = False
            for c in clusters:
                if abs(pos - c["pos"]) <= POS_TOL:
                    c["frames"].add(fi)
                    c["strengths"].append(strength)
                    placed = True
                    break
            if not placed:
                clusters.append({"pos": pos, "frames": {fi}, "strengths": [strength]})

        need = max(1, int(np.ceil(PERSIST_FRAC * max(1, n_frames))))
        out: List[Tuple[float, float]] = []
        for c in clusters:
            if len(c["frames"]) >= need:
                out.append((c["pos"], float(np.mean(c["strengths"]))))
        return out

    def _compute(self, frames: List[np.ndarray]):
        v_per_frame: List[List[Tuple[float, float]]] = []
        h_per_frame: List[List[Tuple[float, float]]] = []
        for frame in frames:
            v, h = self._analyze_frame(frame)
            v_per_frame.append(v)
            h_per_frame.append(h)

        n = len(frames)
        v_persist = self._persistent_splits(v_per_frame, n)
        h_persist = self._persistent_splits(h_per_frame, n)

        v_strength = max((s for _p, s in v_persist), default=0.0)
        h_strength = max((s for _p, s in h_persist), default=0.0)
        score = max(v_strength, h_strength)

        n_cols = len(v_persist) + 1
        n_rows = len(h_persist) + 1
        layout = f"{n_rows}x{n_cols}"
        return float(min(1.0, max(0.0, score))), layout, n_rows, n_cols

    def _load_frames(self, sample: Sample) -> List[np.ndarray]:
        from ayase.image import sample_frames

        max_frames = self.config.get("subsample", 4) if sample.is_video else 1
        return list(sample_frames(sample.path, max_frames=max_frames, color="bgr"))

    @staticmethod
    def _store_layout(sample: Sample, layout: str) -> None:
        """Attach the detected layout to ``sample.metadata['grid_layout']``."""
        sample.metadata["grid_layout"] = layout

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()

        try:
            frames = self._load_frames(sample)
            if not frames:
                return sample

            score, layout, n_rows, n_cols = self._compute(frames)
            sample.quality_metrics.grid_layout_score = float(score)

            if score >= self.config.get("warn_threshold", HIGH_SCORE) and (n_rows > 1 or n_cols > 1):
                self._store_layout(sample, layout)
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Split-screen/grid layout detected ({layout}), score={score:.2f}",
                        details={"grid_layout_score": score, "grid_layout": layout},
                        recommendation=(
                            "Frame appears to be a grid/collage/side-by-side composite "
                            "rather than a single continuous shot."
                        ),
                    )
                )
        except Exception as e:
            logger.warning("grid_layout detection failed: %s", e)

        return sample
