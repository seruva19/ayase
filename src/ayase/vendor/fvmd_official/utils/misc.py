"""Minimal positional embedding helper used by official FVMD PIPs++.

Modified for Ayase from upstream ``fvmd/utils/misc.py``: parameter-reporting
and training utilities were omitted to avoid PrettyTable and stdout side effects.
"""

import torch


def posemb_sincos_2d_xy(xy, channels, temperature=10000, dtype=torch.float32, cat_coords=False):
    """Build the upstream 2-D sine/cosine coordinate embedding."""
    del dtype  # Upstream keeps this argument but uses the input tensor dtype.
    device = xy.device
    input_dtype = xy.dtype
    batch, sequence, dims = xy.shape
    assert dims == 2
    assert channels % 4 == 0, "feature dimension must be multiple of 4 for sincos emb"

    x = xy[:, :, 0]
    y = xy[:, :, 1]
    omega = torch.arange(channels // 4, device=device) / (channels // 4 - 1)
    omega = 1.0 / (temperature**omega)
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    embedding = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    embedding = embedding.reshape(batch, sequence, channels).to(input_dtype)
    if cat_coords:
        embedding = torch.cat([embedding, xy], dim=2)
    return embedding
