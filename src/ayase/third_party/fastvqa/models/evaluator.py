# This source code is licensed under the S-Lab License 1.0 found in the
# LICENSE file in the current directory's parent directory.
"""
The code has been adopted from FAST-VQA-and-FasterVQA
(https://github.com/VQAssessment/FAST-VQA-and-FasterVQA/blob/dev/fastvqa/models/evaluator.py)
"""

import torch
import torch.nn as nn
import time
from .swin_backbone import SwinTransformer3D as SwinBackbone
from .conv_backbone import convnext_3d_tiny, convnext_3d_small
from .xclip_backbone import build_x_clip_model
from .head import VQAHead, IQAHead, VARHead


class BaseEvaluator(nn.Module):
    def __init__(
        self,
        backbone=dict(),
        head=dict(),
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        feats = self.extract_feature(x)
        if isinstance(self.head, dict):
            return {k: self.head[k](feats[k]) for k in feats}
        if isinstance(feats, dict):
            import torch
            feat_list = [feats[k] for k in sorted(feats.keys())]
            feats = torch.cat(feat_list, 1)
        return self.head(feats)

    def extract_feature(self, x):
        if isinstance(self.backbone, dict):
            if not isinstance(x, dict):
                raise ValueError("Expected dict input for multi-branch backbone.")
            return {k: self.backbone[k](x[k]) for k in self.backbone}
        return self.backbone(x)


class DiViDeAddEvaluator(BaseEvaluator):
    def __init__(
        self,
        backbone_size="tiny",
        backbone_preserve_keys="fragments,resize",
        multi=False,
        layer=-1,
        backbone=dict(resize={"window_size": (4, 4, 4)}, fragments={"window_size": (4, 4, 4)}),
        divide_head=False,
        vqa_head=dict(in_channels=768),
        var=False,
    ):
        self.backbone_preserve_keys = backbone_preserve_keys.split(",")
        self.multi = multi
        self.layer = layer
        self.divide_head = divide_head
        self.var = var
        super().__init__(
            backbone,
            vqa_head,
        )

        def _branch_cfg(branch_key: str) -> dict:
            cfg = self.backbone.get(branch_key)
            if cfg is None:
                cfg = {}
            if not isinstance(cfg, dict):
                cfg = {}
            return cfg

        if backbone_size in {"swin_tiny", "swin_tiny_grpb"}:
            for branch_key in self.backbone_preserve_keys:
                cfg = _branch_cfg(branch_key)
                kwargs = dict(
                    pretrained=cfg.get("pretrained"),
                    patch_size=(2, 4, 4),
                    embed_dim=96,
                    depths=[2, 2, 6, 2],
                    num_heads=[3, 6, 12, 24],
                    window_size=cfg.get("window_size", (8, 7, 7)),
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    qk_scale=None,
                    drop_rate=0.0,
                    attn_drop_rate=0.0,
                    drop_path_rate=0.1,
                    patch_norm=True,
                )
                if backbone_size == "swin_tiny_grpb":
                    kwargs["start_group_attn"] = 2
                backbone_module = SwinBackbone(**kwargs)
                self.backbone[branch_key] = backbone_module
                self.add_module(f"{branch_key}_backbone", backbone_module)
        elif backbone_size == "conv_tiny":
            for branch_key in self.backbone_preserve_keys:
                backbone_module = convnext_3d_tiny(pretrained=True)
                self.backbone[branch_key] = backbone_module
                self.add_module(f"{branch_key}_backbone", backbone_module)
        elif backbone_size == "xclip":
            for branch_key in self.backbone_preserve_keys:
                cfg = _branch_cfg(branch_key)
                backbone_module = build_x_clip_model(**cfg)
                self.backbone[branch_key] = backbone_module
                self.add_module(f"{branch_key}_backbone", backbone_module)
        else:
            raise NotImplementedError

        if self.divide_head:
            for key in self.backbone_preserve_keys:
                self.head[key] = VQAHead(in_channels=vqa_head["in_channels"], hidden_channels=64)
                self.add_module(key + "_head", self.head[key])
        else:
            self.vqa_head = VQAHead(
                in_channels=vqa_head["in_channels"] * len(self.backbone_preserve_keys),
                hidden_channels=64,
            )
        if self.var:
            self.var_head = VARHead(
                in_channels=vqa_head["in_channels"] * len(self.backbone_preserve_keys),
                hidden_channels=64,
            )

    def forward(self, vclips, return_pooled_feats=False, reduce_scores=False):
        # This is a modified forward for checking
        # inference time and similarity between two branches

        feats = {}
        for key in self.backbone_preserve_keys:
            feat = self.backbone[key](vclips[key], multi=self.multi, layer=self.layer)
            if self.multi:
                feat = feat.mean(2).mean(2).mean(2)
            feats[key] = feat

        if return_pooled_feats:
            return feats

        if self.divide_head:
            seperate_scores = {}
            for key in self.backbone_preserve_keys:
                seperate_scores[key] = self.head[key](feats[key])
            if reduce_scores:
                return sum(seperate_scores.values())
            return seperate_scores
        else:
            # feats_concat = torch.concat([feats[key] for key in self.backbone_preserve_keys], 1)
            feats_concat = torch.cat([feats[key] for key in self.backbone_preserve_keys], 1)
            if self.var:
                return self.vqa_head(feats_concat), self.var_head(feats_concat)
            return self.vqa_head(feats_concat)
