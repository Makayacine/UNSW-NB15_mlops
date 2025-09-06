# tests/test_bundle.py
import json
from pathlib import Path

BUNDLE = Path("notebooks/ids_unsw/models/bundle_xgb")

def _read_features(p: Path):
    obj = json.loads(p.read_text())
    if isinstance(obj, dict) and "features" in obj:
        return obj["features"]
    return obj

def test_bundle_dir_exists():
    assert BUNDLE.exists()

def test_feature_names_count_is_34():
    feats = _read_features(BUNDLE / "feature_names.json")
    assert isinstance(feats, list)
    assert len(feats) == 34

def test_metadata_has_threshold():
    meta = json.loads((BUNDLE / "metadata.json").read_text())
    assert "threshold" in meta

def test_onnx_is_loadable():
    import onnx
    onnx.load(str(BUNDLE / "xgb.onnx"))
