import json
from pathlib import Path
import onnx

BUNDLE = Path("notebooks/ids_unsw/models/bundle_xgb")

def test_bundle_dir_exists():
    assert BUNDLE.exists(), f"Missing bundle dir: {BUNDLE}"

def test_feature_names_count_is_34():
    feat = json.loads((BUNDLE / "feature_names.json").read_text())
    assert isinstance(feat, list) and len(feat) == 34

def test_metadata_has_threshold():
    meta = json.loads((BUNDLE / "metadata.json").read_text())
    assert "threshold" in meta

def test_onnx_is_loadable():
    m = onnx.load(str(BUNDLE / "xgb.onnx"))
    onnx.checker.check_model(m)
