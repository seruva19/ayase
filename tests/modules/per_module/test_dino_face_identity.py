"""Tests for dino_face_identity module."""

from ..conftest import _test_module_basics


def test_dino_face_identity_basics():
    from ayase.modules.dino_face_identity import DINOFaceIdentityModule

    _test_module_basics(DINOFaceIdentityModule, "dino_face_identity")


def test_dino_face_identity_image_without_setup(image_sample):
    from ayase.modules.dino_face_identity import DINOFaceIdentityModule

    module = DINOFaceIdentityModule()
    result = module.process(image_sample)
    assert result is image_sample


def test_dino_face_identity_video_without_setup(video_sample):
    from ayase.modules.dino_face_identity import DINOFaceIdentityModule

    module = DINOFaceIdentityModule()
    result = module.process(video_sample)
    assert result is video_sample
