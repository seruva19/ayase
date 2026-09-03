# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from ayase.vendor.cotracker.models.core.cotracker.cotracker import CoTracker2
from ayase.vendor.cotracker.models.core.cotracker.cotracker3_offline import (
    CoTrackerThreeOffline,
)


def build_cotracker(checkpoint=None, offline=True, window_len=60, v2=False):
    """Build a vendored CoTracker model.

    Two configurations are vendored, both offline: CoTracker3 (``v2=False``,
    ``window_len=60``), which matches the ``scaled_offline.pth`` checkpoint, and
    CoTracker2 (``v2=True``, ``window_len=8``), which matches ``cotracker2.pth``.
    The online models are not vendored.

    Argument defaults follow the upstream builder, so a caller that passes the
    checkpoint and the matching ``window_len`` gets the same network the authors'
    hub entrypoint builds.
    """
    if not offline:
        raise ValueError("This vendored build only supports the offline models.")
    if v2:
        cotracker = CoTracker2(stride=4, window_len=window_len)
    else:
        cotracker = CoTrackerThreeOffline(
            stride=4, corr_radius=3, window_len=window_len
        )

    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
        cotracker.load_state_dict(state_dict)
    return cotracker
