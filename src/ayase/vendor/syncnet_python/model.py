"""SyncNet (port of ``SyncNetModel.py`` from joonson/syncnet_python, MIT).

Layer names are unchanged: ``syncnet_v2.model`` is loaded by them.
"""

from pathlib import Path

import torch
from torch import nn

EMBEDDING_SIZE = 1024
# The 2019 checkpoint predates BatchNorm counters; they do not affect inference.
_OPTIONAL_STATE_SUFFIX = "num_batches_tracked"


class SyncNetModel(nn.Module):
    """Two-stream SyncNet: MFCC audio windows and mouth-region video windows."""

    def __init__(self, num_layers_in_fc_layers: int = EMBEDDING_SIZE) -> None:
        super().__init__()

        self.netcnnaud = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2)),
            nn.Conv2d(192, 384, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            nn.Conv2d(256, 512, kernel_size=(5, 4), padding=(0, 0)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.netfcaud = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_layers_in_fc_layers),
        )

        self.netfclip = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_layers_in_fc_layers),
        )

        self.netcnnlip = nn.Sequential(
            nn.Conv3d(3, 96, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=0),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            nn.Conv3d(96, 256, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            nn.Conv3d(256, 512, kernel_size=(1, 6, 6), padding=0),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        )

    def forward_aud(self, x: torch.Tensor) -> torch.Tensor:
        """Encode MFCC windows ``(N, 1, 13, 20)`` into ``(N, embedding)``."""
        mid = self.netcnnaud(x)
        mid = mid.view((mid.size()[0], -1))
        return self.netfcaud(mid)

    def forward_lip(self, x: torch.Tensor) -> torch.Tensor:
        """Encode frame windows ``(N, 3, 5, 224, 224)`` into ``(N, embedding)``."""
        mid = self.netcnnlip(x)
        mid = mid.view((mid.size()[0], -1))
        return self.netfclip(mid)


def load_syncnet(weights_path: Path, device: torch.device) -> SyncNetModel:
    """Build SyncNet, load ``syncnet_v2.model`` and switch to ``eval`` on ``device``.

    Raises ``RuntimeError`` when the checkpoint keys do not match the architecture.
    """
    model = SyncNetModel()
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    result = model.load_state_dict(state, strict=False)
    missing = [key for key in result.missing_keys if not key.endswith(_OPTIONAL_STATE_SUFFIX)]
    if missing or result.unexpected_keys:
        raise RuntimeError(
            "SyncNet weights do not match the model: "
            f"missing {missing}, unexpected {list(result.unexpected_keys)}"
        )
    model.to(device)
    model.eval()
    return model
