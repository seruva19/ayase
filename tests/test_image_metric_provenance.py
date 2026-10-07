"""Video aggregation must not claim an unchanged published image protocol."""

import pytest

from ayase.pipeline import ModuleRegistry


IMAGE_VIDEO_WRAPPERS = """
aesthetic_scoring ahiq arniqa brisque butteraugli ciede2000 ckdn clip_iqa cnniqa
compare2score cpbd cw_ssim dbcnn deepdc dists dmm erqa evoquality hpsv2 hpsv3
hyperiqa ilniqe image_reward laion_aesthetic liqe maclip mad maniqa mc360iqa
mouth_quality musiq nima niqe nlpd nrqm paq2piq pickscore pieapp piqe pi qcn
qualiclip semantic_alignment serfiq ssimc topiq topiq_fr tres uciqe unique
vfips wadiqam wadiqam_fr
""".split()


@pytest.fixture(scope="module", autouse=True)
def discover_wrappers():
    ModuleRegistry.discover_modules()


@pytest.mark.parametrize("name", IMAGE_VIDEO_WRAPPERS)
def test_video_image_score_is_declared_adapted(name):
    cls = ModuleRegistry.get_module(name)
    assert cls is not None
    metadata = cls.get_metadata()
    fields = set(metadata["output_fields"]) | set(metadata["dataset_output_fields"])
    assert fields, name
    for field in fields:
        classification = metadata["provenance"].get(field)
        assert classification in {"adapted", "own", "utility"}, (name, field)
        if classification == "adapted":
            assert metadata["sources"].get(field), (name, field)
            assert "video" in metadata["deviations"].get(field, "").lower(), (name, field)


def test_duplicate_aesthetic_value_has_same_provenance():
    from ayase.modules.aesthetic import AestheticModule

    metadata = AestheticModule.get_metadata()
    for field in ("aesthetic_v25_score", "aesthetic_v25_dup"):
        assert metadata["provenance"][field] == "adapted"
        assert metadata["sources"][field] == metadata["sources"]["aesthetic_v25_score"]


def test_csim_substitute_recognizer_is_not_published():
    from ayase.modules.csim import CSIMModule

    metadata = CSIMModule.get_metadata()
    assert metadata["provenance"]["csim"] == "adapted"
    assert metadata["provenance"]["csim_face_frames"] == "utility"
    assert "buffalo_l" in metadata["deviations"]["csim"]
