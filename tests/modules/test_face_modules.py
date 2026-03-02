def test_face_fidelity_basics():
    from ayase.modules.face_fidelity import FaceFidelityModule
    from .conftest import _test_module_basics

    _test_module_basics(FaceFidelityModule, "face_fidelity")


def test_face_fidelity_setup():
    from ayase.modules.face_fidelity import FaceFidelityModule

    m = FaceFidelityModule()
    m.setup()
    assert m._ml_available


def test_face_fidelity_image_no_faces(image_sample):
    from ayase.modules.face_fidelity import FaceFidelityModule

    m = FaceFidelityModule()
    m.setup()
    result = m.process(image_sample)
    assert result.quality_metrics is not None
    assert result.quality_metrics.face_count == 0


def test_face_landmark_quality_basics():
    from ayase.modules.face_landmark_quality import FaceLandmarkQualityModule
    from .conftest import _test_module_basics

    _test_module_basics(FaceLandmarkQualityModule, "face_landmark_quality")
