# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from ayase.vendor.cotracker.models.core.cotracker.cotracker3_offline import (
    CoTrackerThreeOffline,
)


def build_cotracker(checkpoint=None, offline=True, window_len=60, v2=False):
    """Build the offline CoTracker3 model.

    Only the offline CoTracker3 configuration is vendored (``v2=False``,
    ``offline=True``), which matches the ``scaled_offline.pth`` checkpoint.
    """
    if v2 or not offline:
        raise ValueError(
            "This vendored build only supports the offline CoTracker3 model "
            "(v2=False, offline=True)."
        )
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
