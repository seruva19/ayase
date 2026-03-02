from ayase.models import QualityMetrics


def test_cnniqa_basics():
    from ayase.modules.cnniqa import CNNIQAModule
    from .conftest import _test_module_basics

    _test_module_basics(CNNIQAModule, "cnniqa")


def test_cnniqa_no_ml(image_sample):
    from ayase.modules.cnniqa import CNNIQAModule

    image_sample.quality_metrics = QualityMetrics()
    m = CNNIQAModule()
    result = m.process(image_sample)
    assert result.quality_metrics.cnniqa_score is None


def test_hyperiqa_basics():
    from ayase.modules.hyperiqa import HyperIQAModule
    from .conftest import _test_module_basics

    _test_module_basics(HyperIQAModule, "hyperiqa")


def test_paq2piq_basics():
    from ayase.modules.paq2piq import PaQ2PiQModule
    from .conftest import _test_module_basics

    _test_module_basics(PaQ2PiQModule, "paq2piq")


def test_tres_basics():
    from ayase.modules.tres import TReSModule
    from .conftest import _test_module_basics

    _test_module_basics(TReSModule, "tres")


def test_unique_basics():
    from ayase.modules.unique_iqa import UNIQUEModule
    from .conftest import _test_module_basics

    _test_module_basics(UNIQUEModule, "unique")


def test_laion_aesthetic_basics():
    from ayase.modules.laion_aesthetic import LAIONAestheticModule
    from .conftest import _test_module_basics

    _test_module_basics(LAIONAestheticModule, "laion_aesthetic")


def test_compare2score_basics():
    from ayase.modules.compare2score import Compare2ScoreModule
    from .conftest import _test_module_basics

    _test_module_basics(Compare2ScoreModule, "compare2score")


def test_afine_basics():
    from ayase.modules.afine import AFINEModule
    from .conftest import _test_module_basics

    _test_module_basics(AFINEModule, "afine")


def test_ckdn_basics():
    from ayase.modules.ckdn import CKDNModule
    from .conftest import _test_module_basics

    _test_module_basics(CKDNModule, "ckdn")


def test_deepwsd_basics():
    from ayase.modules.deepwsd import DeepWSDModule
    from .conftest import _test_module_basics

    _test_module_basics(DeepWSDModule, "deepwsd")


def test_image_iqa_fields():
    qm = QualityMetrics()
    fields = [
        "cnniqa_score",
        "hyperiqa_score",
        "paq2piq_score",
        "tres_score",
        "unique_score",
        "laion_aesthetic",
        "compare2score",
        "afine_score",
        "ckdn_score",
        "deepwsd_score",
    ]
    for field in fields:
        assert hasattr(qm, field)
