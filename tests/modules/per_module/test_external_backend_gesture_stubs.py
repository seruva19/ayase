"""Tests for the published-protocol external-backend stubs (group: gesture,
lip, identity)."""

import pytest

from ayase.models import QualityMetrics, Sample


@pytest.mark.parametrize("module_name,cls_name,field", [
    ("fgd", "FGDModule", "fgd"),
    ("lip_reading_wer", "LipReadingWERModule", "lip_reading_wer"),
    ("acc_emo", "AccEmoModule", "acc_emo"),
    ("fdd", "FDDModule", "fdd"),
    ("lve", "LVEModule", "lve"),
    ("mouth_opening_distance", "MouthOpeningDistanceModule", "mod"),
    ("srgr", "SRGRModule", "srgr"),
    ("poi_forensics", "POIForensicsModule", "poi_forensics_score"),
    ("manner_correlations", "MannerCorrelationsModule", "manner_correlation"),
    ("face_gesture_correlations", "FaceGestureCorrelationsModule",
     "face_gesture_correlation"),
    ("behavioral_embedding", "BehavioralEmbeddingModule",
     "behavioral_embedding_distance"),
])
def test_stub_registered_external_backend(module_name, cls_name, field):
    import importlib

    mod = importlib.import_module(f"ayase.modules.{module_name}")
    cls = getattr(mod, cls_name)
    assert cls.name == module_name
    assert cls.requires_external_backend is True
    meta = cls.get_metadata()
    prov = meta["provenance"]
    assert prov.get(field) == "published" if isinstance(prov, dict) else prov == "published"


def test_stub_process_passthrough(tmp_path):
    from ayase.modules.acc_emo import AccEmoModule
    m = AccEmoModule()
    sample = Sample(path=tmp_path / "v.mp4", is_video=True)
    sample.quality_metrics = QualityMetrics()
    out = m.process(sample)
    assert out is sample
    assert out.quality_metrics.acc_emo is None
