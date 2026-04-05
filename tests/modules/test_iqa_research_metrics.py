from ayase.models import QualityMetrics


def test_nima_basics():
    from ayase.modules.nima import NIMAModule
    from .conftest import _test_module_basics

    _test_module_basics(NIMAModule, "nima")


def test_dbcnn_basics():
    from ayase.modules.dbcnn import DBCNNModule
    from .conftest import _test_module_basics

    _test_module_basics(DBCNNModule, "dbcnn")


def test_wadiqam_basics():
    from ayase.modules.wadiqam import WaDIQaMModule
    from .conftest import _test_module_basics

    _test_module_basics(WaDIQaMModule, "wadiqam")


def test_maniqa_basics():
    from ayase.modules.maniqa import MANIQAModule
    from .conftest import _test_module_basics

    _test_module_basics(MANIQAModule, "maniqa")


def test_arniqa_basics():
    from ayase.modules.arniqa import ARNIQAModule
    from .conftest import _test_module_basics

    _test_module_basics(ARNIQAModule, "arniqa")


def test_qualiclip_basics():
    from ayase.modules.qualiclip import QualiCLIPModule
    from .conftest import _test_module_basics

    _test_module_basics(QualiCLIPModule, "qualiclip")


def test_pieapp_basics():
    from ayase.modules.pieapp import PieAPPModule
    from .conftest import _test_module_basics

    _test_module_basics(PieAPPModule, "pieapp")


def test_cw_ssim_basics():
    from ayase.modules.cw_ssim import CWSSIMModule
    from .conftest import _test_module_basics

    _test_module_basics(CWSSIMModule, "cw_ssim")


def test_nlpd_basics():
    from ayase.modules.nlpd_metric import NLPDModule
    from .conftest import _test_module_basics

    _test_module_basics(NLPDModule, "nlpd")


def test_mad_basics():
    from ayase.modules.mad_metric import MADModule
    from .conftest import _test_module_basics

    _test_module_basics(MADModule, "mad")


def test_ahiq_basics():
    from ayase.modules.ahiq import AHIQModule
    from .conftest import _test_module_basics

    _test_module_basics(AHIQModule, "ahiq")


def test_topiq_fr_basics():
    from ayase.modules.topiq_fr import TOPIQFRModule
    from .conftest import _test_module_basics

    _test_module_basics(TOPIQFRModule, "topiq_fr")


def test_dreamsim_basics():
    from ayase.modules.dreamsim_metric import DreamSimModule
    from .conftest import _test_module_basics

    _test_module_basics(DreamSimModule, "dreamsim")


def test_cover_basics():
    from ayase.modules.cover import COVERModule
    from .conftest import _test_module_basics

    _test_module_basics(COVERModule, "cover")


def test_vqa_score_basics():
    from ayase.modules.vqa_score import VQAScoreModule
    from .conftest import _test_module_basics

    _test_module_basics(VQAScoreModule, "vqa_score")


def test_videoscore_basics():
    from ayase.modules.videoscore import VideoScoreModule
    from .conftest import _test_module_basics

    _test_module_basics(VideoScoreModule, "videoscore")


def test_videoscore2_basics():
    from ayase.modules.videoscore2 import VideoScore2Module
    from .conftest import _test_module_basics

    _test_module_basics(VideoScore2Module, "videoscore2")


def test_face_iqa_basics():
    from ayase.modules.face_iqa import FaceIQAModule
    from .conftest import _test_module_basics

    _test_module_basics(FaceIQAModule, "face_iqa")


def test_research_qualitymetrics_fields():
    qm = QualityMetrics()
    fields = [
        "nima_score",
        "dbcnn_score",
        "wadiqam_score",
        "maniqa_score",
        "arniqa_score",
        "qualiclip_score",
        "pieapp",
        "cw_ssim",
        "nlpd",
        "mad",
        "ahiq",
        "topiq_fr",
        "dreamsim",
        "cover_score",
        "cover_technical",
        "cover_aesthetic",
        "cover_semantic",
        "vqa_score_alignment",
        "videoscore_visual",
        "videoscore_temporal",
        "videoscore_dynamic",
        "videoscore_alignment",
        "videoscore_factual",
        "videoscore2_visual",
        "videoscore2_alignment",
        "videoscore2_physical",
        "face_iqa_score",
    ]
    for field in fields:
        assert hasattr(qm, field)
