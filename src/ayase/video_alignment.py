"""Timestamp alignment for comparing an edited video to its source."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Iterable
from math import isfinite
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


def _timestamps(values: Iterable[float]) -> tuple[float, ...]:
    times = tuple(float(value) for value in values)
    if not all(isfinite(time) for time in times) or any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError("frame timestamps must be finite and strictly increasing")
    if not times:
        return ()
    elapsed = tuple(time - times[0] for time in times)
    if not isfinite(elapsed[-1]):
        raise ValueError("frame timestamp span must be finite")
    return elapsed


class VideoAlignment(BaseModel):
    """Configure how edited frame timestamps map to the source video."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["timestamp", "normalized", "anchors"] = "timestamp"
    anchors: tuple[tuple[float, float], ...] = ()

    @model_validator(mode="after")
    def _check_anchors(self) -> VideoAlignment:
        if self.mode != "anchors":
            if self.anchors:
                raise ValueError("anchors require mode='anchors'")
            return self
        if len(self.anchors) < 2 or any(
            not isfinite(time) or time < 0 for pair in self.anchors for time in pair
        ):
            raise ValueError("at least two finite, nonnegative time anchors are required")
        if any(a[0] >= b[0] or a[1] > b[1] for a, b in zip(self.anchors, self.anchors[1:])):
            raise ValueError("edited anchors must increase and source anchors must not decrease")
        return self

    def source_indices(
        self, edited_pts: Iterable[float], source_pts: Iterable[float]
    ) -> list[int | None]:
        """Map each edited frame to the nearest source frame within coverage.

        Times are seconds relative to each video's first frame. Return None
        outside the source range or outside the explicit anchor range.
        """
        edited = _timestamps(edited_pts)
        source = _timestamps(source_pts)
        if not edited or not source:
            return [None] * len(edited)

        mapped: Iterable[float | None] = edited
        if self.mode == "normalized":
            duration = edited[-1]
            mapped = (time / duration * source[-1] for time in edited) if duration else (0.0,)
        elif self.mode == "anchors":
            mapped = self._interpolate(edited)

        result: list[int | None] = []
        for time in mapped:
            if time is None or time < 0 or time > source[-1]:
                result.append(None)
                continue
            right = bisect_left(source, time)
            if right > 0 and time - source[right - 1] <= source[right] - time:
                result.append(right - 1)
            else:
                result.append(right)
        return result

    def _interpolate(self, times: Iterable[float]) -> Iterable[float | None]:
        edited = tuple(pair[0] for pair in self.anchors)
        for time in times:
            if time < edited[0] or time > edited[-1]:
                yield None
                continue
            upper = min(bisect_right(edited, time), len(edited) - 1)
            start, first = self.anchors[upper - 1]
            end, last = self.anchors[upper]
            yield first + (last - first) * ((time - start) / (end - start))
