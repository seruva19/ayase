# This source code is licensed under the S-Lab License 1.0 found in the
# LICENSE file in the current directory's parent directory.
"""
The code has been adopted from FAST-VQA-and-FasterVQA
(https://github.com/VQAssessment/FAST-VQA-and-FasterVQA/blob/dev/fastvqa/models/head.py)
"""

import torch
import torch.nn as nn


class VQAHead(nn.Module):
    """
    MLP Head for VQA.
    """

    def __init__(
        self,
        in_channels=768,
        hidden_channels=64,
        dropout_ratio=0.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.dropout_ratio = dropout_ratio

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_hid = nn.Conv3d(in_channels, hidden_channels, 1, 1, 0)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_ratio)
        self.fc_last = nn.Conv3d(hidden_channels, 1, 1, 1, 0)

    def forward(self, x):
        if x.ndim == 5:
            x = self.avg_pool(x)
        x = self.fc_hid(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc_last(x)
        x = x.flatten(1)
        return x


class VARHead(nn.Module):
    """
    MLP Head for VQA.
    """

    def __init__(
        self,
        in_channels=768,
        hidden_channels=64,
        dropout_ratio=0.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.dropout_ratio = dropout_ratio

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_ratio)
        self.linear2 = nn.Linear(hidden_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.ndim == 5:
            x = self.avg_pool(x)
            x = x.flatten(1)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x


class IQAHead(nn.Module):
    """
    MLP Head for IQA.
    """

    def __init__(
        self,
        in_channels=768,
        hidden_channels=64,
        dropout_ratio=0.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.dropout_ratio = dropout_ratio

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_ratio)
        self.linear2 = nn.Linear(hidden_channels, 1)

    def forward(self, x):
        if x.ndim == 4:
            x = self.avg_pool(x)
            x = x.flatten(1)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
