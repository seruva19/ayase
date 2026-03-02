def test_dists_basics():
    from ayase.modules.dists import DISTSModule
    from .conftest import _test_module_basics

    _test_module_basics(DISTSModule, "dists")


def test_dists_disabled(image_sample):
    from ayase.modules.dists import DISTSModule

    m = DISTSModule({})
    result = m.process(image_sample)
    assert result is image_sample


def test_brisque_basics():
    from ayase.modules.brisque import BRISQUEModule
    from .conftest import _test_module_basics

    _test_module_basics(BRISQUEModule, "brisque")


def test_dover_basics():
    from ayase.modules.dover import DOVERModule
    from .conftest import _test_module_basics

    _test_module_basics(DOVERModule, "dover")


def test_topiq_basics():
    from ayase.modules.topiq import TOPIQModule
    from .conftest import _test_module_basics

    _test_module_basics(TOPIQModule, "topiq")


def test_liqe_basics():
    from ayase.modules.liqe import LIQEModule
    from .conftest import _test_module_basics

    _test_module_basics(LIQEModule, "liqe")


def test_clip_iqa_basics():
    from ayase.modules.clip_iqa import CLIPIQAModule
    from .conftest import _test_module_basics

    _test_module_basics(CLIPIQAModule, "clip_iqa")


def test_perceptual_fr_basics():
    from ayase.modules.perceptual_fr import PerceptualFRModule
    from .conftest import _test_module_basics

    _test_module_basics(PerceptualFRModule, "perceptual_fr")


def test_q_align_basics():
    from ayase.modules.q_align import QAlignModule
    from .conftest import _test_module_basics

    _test_module_basics(QAlignModule, "q_align")


def test_musiq_basics():
    from ayase.modules.musiq import MUSIQModule
    from .conftest import _test_module_basics

    _test_module_basics(MUSIQModule, "musiq")


def test_contrique_basics():
    from ayase.modules.contrique import CONTRIQUEModule
    from .conftest import _test_module_basics

    _test_module_basics(CONTRIQUEModule, "contrique")


def test_mdtvsfa_basics():
    from ayase.modules.mdtvsfa import MDTVSFAModule
    from .conftest import _test_module_basics

    _test_module_basics(MDTVSFAModule, "mdtvsfa")


def test_audio_pesq_basics():
    from ayase.modules.audio_pesq import AudioPESQModule
    from .conftest import _test_module_basics

    _test_module_basics(AudioPESQModule, "audio_pesq")


def test_audio_visual_sync_basics():
    from ayase.modules.audio_visual_sync import AudioVisualSyncModule
    from .conftest import _test_module_basics

    _test_module_basics(AudioVisualSyncModule, "av_sync")
