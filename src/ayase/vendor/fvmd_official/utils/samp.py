"""Minimal sampling helper used by the official FVMD PIPs++ runtime.

Modified for Ayase from upstream ``fvmd/utils/samp.py``: only the PIPs++
inference dependency is retained.
"""

import torch


def bilinear_sample2d(im, x, y, return_inbounds=False):
    """Sample ``im`` at pixel-space ``x``/``y`` coordinates."""
    batch, channels, height, width = list(im.shape)
    point_count = list(x.shape)[1]

    x = x.float()
    y = y.float()
    max_y = height - 1
    max_x = width - 1

    x0 = torch.floor(x).int()
    x1 = x0 + 1
    y0 = torch.floor(y).int()
    y1 = y0 + 1

    x0_clip = torch.clamp(x0, 0, max_x)
    x1_clip = torch.clamp(x1, 0, max_x)
    y0_clip = torch.clamp(y0, 0, max_y)
    y1_clip = torch.clamp(y1, 0, max_y)
    base = torch.arange(0, batch, dtype=torch.int64, device=x.device) * (width * height)
    base = base.reshape(batch, 1).repeat(1, point_count)

    base_y0 = base + y0_clip * width
    base_y1 = base + y1_clip * width
    idx_y0_x0 = base_y0 + x0_clip
    idx_y0_x1 = base_y0 + x1_clip
    idx_y1_x0 = base_y1 + x0_clip
    idx_y1_x1 = base_y1 + x1_clip

    im_flat = im.permute(0, 2, 3, 1).reshape(batch * height * width, channels)
    i_y0_x0 = im_flat[idx_y0_x0.long()]
    i_y0_x1 = im_flat[idx_y0_x1.long()]
    i_y1_x0 = im_flat[idx_y1_x0.long()]
    i_y1_x1 = im_flat[idx_y1_x1.long()]

    x0_f = x0.float()
    x1_f = x1.float()
    y0_f = y0.float()
    y1_f = y1.float()
    w_y0_x0 = ((x1_f - x) * (y1_f - y)).unsqueeze(2)
    w_y0_x1 = ((x - x0_f) * (y1_f - y)).unsqueeze(2)
    w_y1_x0 = ((x1_f - x) * (y - y0_f)).unsqueeze(2)
    w_y1_x1 = ((x - x0_f) * (y - y0_f)).unsqueeze(2)

    output = (
        w_y0_x0 * i_y0_x0
        + w_y0_x1 * i_y0_x1
        + w_y1_x0 * i_y1_x0
        + w_y1_x1 * i_y1_x1
    )
    output = output.view(batch, -1, channels).permute(0, 2, 1)

    if return_inbounds:
        x_valid = (x > -0.5) & (x < width - 0.5)
        y_valid = (y > -0.5) & (y < height - 0.5)
        return output, (x_valid & y_valid).float().reshape(batch, point_count)
    return output
