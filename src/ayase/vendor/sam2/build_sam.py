# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Hydra-free builder for the SAM 2.1 Hiera-Large video predictor.

Upstream SAM 2 instantiates its models from Hydra/OmegaConf YAML configs. To
keep this vendored copy free of the ``hydra-core``/``omegaconf`` dependencies,
the exact architecture described by ``configs/sam2.1/sam2.1_hiera_l.yaml`` is
reproduced here as explicit Python constructor calls. The scalar values below
are copied verbatim from that config, plus the default "apply_postprocessing"
overrides that upstream ``build_sam2_video_predictor`` adds.
"""

import logging

import torch

from ayase.vendor.sam2.modeling.backbones.hieradet import Hiera
from ayase.vendor.sam2.modeling.backbones.image_encoder import FpnNeck, ImageEncoder
from ayase.vendor.sam2.modeling.memory_attention import (
    MemoryAttention,
    MemoryAttentionLayer,
)
from ayase.vendor.sam2.modeling.memory_encoder import (
    CXBlock,
    Fuser,
    MaskDownSampler,
    MemoryEncoder,
)
from ayase.vendor.sam2.modeling.position_encoding import PositionEmbeddingSine
from ayase.vendor.sam2.modeling.sam.transformer import RoPEAttention
from ayase.vendor.sam2.sam2_video_predictor import SAM2VideoPredictor


def _build_image_encoder() -> ImageEncoder:
    """Hiera-L image encoder + FPN neck (config: image_encoder)."""
    trunk = Hiera(
        embed_dim=144,
        num_heads=2,
        stages=(2, 6, 36, 4),
        global_att_blocks=(23, 33, 43),
        window_pos_embed_bkg_spatial_size=(7, 7),
        window_spec=(8, 4, 16, 8),
    )
    neck = FpnNeck(
        position_encoding=PositionEmbeddingSine(
            num_pos_feats=256,
            normalize=True,
            scale=None,
            temperature=10000,
        ),
        d_model=256,
        backbone_channel_list=[1152, 576, 288, 144],
        fpn_top_down_levels=[2, 3],
        fpn_interp_model="nearest",
    )
    return ImageEncoder(trunk=trunk, neck=neck, scalp=1)


def _build_memory_attention() -> MemoryAttention:
    """Memory attention stack (config: memory_attention)."""
    self_attention = RoPEAttention(
        rope_theta=10000.0,
        feat_sizes=(64, 64),
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
    )
    cross_attention = RoPEAttention(
        rope_theta=10000.0,
        feat_sizes=(64, 64),
        rope_k_repeat=True,
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
        kv_in_dim=64,
    )
    layer = MemoryAttentionLayer(
        activation="relu",
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        self_attention=self_attention,
        d_model=256,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        cross_attention=cross_attention,
    )
    return MemoryAttention(
        d_model=256,
        pos_enc_at_input=True,
        layer=layer,
        num_layers=4,
    )


def _build_memory_encoder() -> MemoryEncoder:
    """Memory encoder (config: memory_encoder)."""
    mask_downsampler = MaskDownSampler(kernel_size=3, stride=2, padding=1)
    fuser = Fuser(
        layer=CXBlock(
            dim=256,
            kernel_size=7,
            padding=3,
            layer_scale_init_value=1e-6,
            use_dwconv=True,
        ),
        num_layers=2,
    )
    return MemoryEncoder(
        out_dim=64,
        position_encoding=PositionEmbeddingSine(
            num_pos_feats=64,
            normalize=True,
            scale=None,
            temperature=10000,
        ),
        mask_downsampler=mask_downsampler,
        fuser=fuser,
    )


# Scalar SAM2Base flags from sam2.1_hiera_l.yaml (model: section, minus the
# three submodules built above).
_SAM2_BASE_KWARGS = dict(
    num_maskmem=7,
    image_size=1024,
    sigmoid_scale_for_mem_enc=20.0,
    sigmoid_bias_for_mem_enc=-10.0,
    use_mask_input_as_output_without_sam=True,
    directly_add_no_mem_embed=True,
    no_obj_embed_spatial=True,
    use_high_res_features_in_sam=True,
    multimask_output_in_sam=True,
    iou_prediction_use_sigmoid=True,
    use_obj_ptrs_in_encoder=True,
    add_tpos_enc_to_obj_ptrs=True,
    proj_tpos_enc_in_obj_ptrs=True,
    use_signed_tpos_enc_to_obj_ptrs=True,
    only_obj_ptrs_in_the_past_for_eval=True,
    pred_obj_scores=True,
    pred_obj_scores_mlp=True,
    fixed_no_obj_ptr=True,
    multimask_output_for_tracking=True,
    use_multimask_token_for_obj_ptr=True,
    multimask_min_pt_num=0,
    multimask_max_pt_num=1,
    use_mlp_for_obj_ptr_proj=True,
    compile_image_encoder=False,
)

# The default apply_postprocessing overrides that upstream
# build_sam2_video_predictor injects when apply_postprocessing=True.
_POSTPROCESSING_KWARGS = dict(
    sam_mask_decoder_extra_args=dict(
        dynamic_multimask_via_stability=True,
        dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98,
    ),
    binarize_mask_from_pts_for_mem_enc=True,
    fill_hole_area=8,
)


def build_sam2_video_predictor(ckpt_path, device="cuda", mode="eval"):
    """Build the SAM 2.1 Hiera-L video predictor and load its checkpoint.

    Args:
        ckpt_path: local path to ``sam2.1_hiera_large.pt``.
        device: torch device string.
        mode: ``"eval"`` puts the model in eval mode.

    Returns:
        A ``(model, load_result)`` tuple. ``load_result`` is the
        ``(missing_keys, unexpected_keys)`` returned by ``load_state_dict``.
    """
    model = SAM2VideoPredictor(
        image_encoder=_build_image_encoder(),
        memory_attention=_build_memory_attention(),
        memory_encoder=_build_memory_encoder(),
        **_SAM2_BASE_KWARGS,
        **_POSTPROCESSING_KWARGS,
    )
    load_result = _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model, load_result


def _load_checkpoint(model, ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
    missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)
    if missing_keys:
        logging.warning("SAM2 missing keys: %d", len(missing_keys))
    if unexpected_keys:
        logging.warning("SAM2 unexpected keys: %d", len(unexpected_keys))
    return missing_keys, unexpected_keys
