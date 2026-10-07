from ayase.third_party.dover.models.conv_backbone import (
    convnext_3d_small,
    convnext_3d_tiny,
    convnextv2_3d_femto,
    convnextv2_3d_pico,
)

from .clipiqa_arch import CLIPIQA


def clip_vitL14(pretrained, **kwargs):
    model = CLIPIQA(
        model_type="clipiqa+_vitL14_512", backbone="ViT-L/14", pretrained=pretrained
    )
    return model
