"""Licences of the vendored components, and the notice a module owes its user.

Ayase itself is MIT. Some metrics run research code vendored under
``ayase.vendor``, and three of those trees are not permissive: two are
non-commercial, one carries no licence file at all, and one pulls in copyleft. A
result produced by such a metric is governed by that component's terms, not by
Ayase's, and the user has to learn this before the number lands in their report -
not from a lawyer afterwards.

So every affected module declares what it runs, and the first setup of that module
in a process writes the notice to the log. The notice is not a substitute for
fixing the situation: the plan for 1.0 is to replace these components with
implementations Ayase can license itself.
"""

import logging
from typing import Dict, Iterable, NamedTuple, Set

logger = logging.getLogger(__name__)


class VendorLicense(NamedTuple):
    """A vendored component and the terms it imposes on the caller."""

    component: str
    license: str
    permissive: bool
    note: str


#: Components whose licence differs from Ayase's own MIT.
VENDOR_LICENSES: Dict[str, VendorLicense] = {
    "cotracker": VendorLicense(
        component="CoTracker (facebookresearch/co-tracker)",
        license="CC BY-NC 4.0",
        permissive=False,
        note="non-commercial use only",
    ),
    "imagebind": VendorLicense(
        component="ImageBind (facebookresearch/ImageBind)",
        license="CC BY-NC-SA 4.0",
        permissive=False,
        note="non-commercial use only, share-alike",
    ),
    "mj_video": VendorLicense(
        component="MJ-Video (aiming-lab/MJ-Video)",
        license="no licence file in the upstream snapshot",
        permissive=False,
        note="no redistribution or use grant is stated upstream",
    ),
    "vbench": VendorLicense(
        component="VBench 2.0 (Vchitect/VBench) with its vendored YOLO-World and CoTracker",
        license="Apache-2.0, but includes GPL-3.0 (mmyolo) and CC BY-NC 4.0 (CoTracker)",
        permissive=False,
        note="copyleft and non-commercial terms reach the result",
    ),
}

#: Trees whose terms match Ayase's own and need no notice.
PERMISSIVE_COMPONENTS = {
    "q_align",
    "s2wrapper",
    "sam",
    "sam2",
    "t2v_metrics",
    "verse_bench",
    "videomae",
    "vila",
    "vmbench",
    "vqa2",
}

_announced: Set[str] = set()


def announce(components: Iterable[str]) -> None:
    """Log the licence notice for components that impose non-MIT terms.

    Called from a module's ``setup``. Each component is announced once per
    process: the notice must be visible, and repeating it per sample would train
    the reader to filter it out.

    Args:
        components (Iterable[str]): Keys of ``VENDOR_LICENSES``.
    """
    for key in components:
        entry = VENDOR_LICENSES.get(key)
        if entry is None or entry.permissive or key in _announced:
            continue
        _announced.add(key)
        logger.warning(
            "%s runs vendored %s under %s (%s). Ayase is MIT, but this metric is "
            "not: its result is governed by that licence.",
            "This metric",
            entry.component,
            entry.license,
            entry.note,
        )


def report() -> Dict[str, VendorLicense]:
    """Return the non-permissive components, for documentation and tests.

    Returns:
        Dict[str, VendorLicense]: Mapping of vendor tree to its terms.
    """
    return {k: v for k, v in VENDOR_LICENSES.items() if not v.permissive}
