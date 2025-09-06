# ids_unsw/experiments/xgb_to_onnx.py
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

def load_features(p: Path):
    obj = json.loads(p.read_text())
    return obj["features"] if isinstance(obj, dict) and "features" in obj else obj

def parse_args():
    p = argparse.ArgumentParser("Export best_xgboost_model.pkl to ONNX")
    p.add_argument("--base", default="notebooks/ids_unsw", help="Base with data/ and models/")
    p.add_argument("--features-json", default=None, help="Path to feature_names.json (optional)")
    p.add_argument("--xgb-pkl", default=None, help="Path to best_xgboost_model.pkl (optional)")
    p.add_argument("--onnx-out", default=None, help="Output ONNX path (default models/xgb.onnx)")
    p.add_argument("--opset", type=int, default=17)
    return p.parse_args()

def main():
    args = parse_args()
    base = Path(args.base)
    model_dir = base / "models"

    features_path = Path(args.features_json) if args.features_json else (model_dir / "feature_names.json")
    model_path    = Path(args.xgb_pkl)       if args.xgb_pkl       else (model_dir / "best_xgboost_model.pkl")
    onnx_out      = Path(args.onnx_out)      if args.onnx_out      else (model_dir / "xgb.onnx")

    feats = load_features(features_path)
    n_features = len(feats)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # ---- Try onnxmltools first (best for XGBoost) ----
    onnx_model = None
    try:
        # Some versions expose the function in different places
        try:
            from onnxmltools import convert_xgboost
        except Exception:
            from onnxmltools.convert import convert_xgboost  # older/newer layouts

        from onnxmltools.convert.common.data_types import FloatTensorType
        initial_types = [("input", FloatTensorType([None, n_features]))]

        # Newer onnxmltools supports options={'zipmap': False}; older ones don't.
        try:
            onnx_model = convert_xgboost(
                model, initial_types=initial_types, target_opset=args.opset,
                options={"zipmap": False}
            )
        except TypeError:
            onnx_model = convert_xgboost(
                model, initial_types=initial_types, target_opset=args.opset
            )

        # Save (polish if utils available)
        try:
            import onnxmltools
            onnx_model = onnxmltools.utils.polish_model(onnx_model)
            onnxmltools.utils.save_model(onnx_model, str(onnx_out))
        except Exception:
            with open(onnx_out, "wb") as f:
                f.write(onnx_model.SerializeToString())

        print(f"⯑ Exported with onnxmltools → {onnx_out}")
        return
    except Exception as e:
        print(f"onnxmltools path failed ({type(e).__name__}: {e}). Trying skl2onnx…")

    # ---- Fallback: skl2onnx ----
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        onnx_model = convert_sklearn(
            model,
            initial_types=[("input", FloatTensorType([None, n_features]))],
            target_opset=args.opset,
            options={type(model): {"zipmap": False}},
        )
        with open(onnx_out, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"⯑ Exported with skl2onnx → {onnx_out}")
    except Exception as e:
        raise SystemExit(
            "Failed to export to ONNX. If you see import errors, install converters:\n"
            "  pip install onnxmltools onnxconverter-common skl2onnx onnx onnxruntime\n"
            f"Converter error: {type(e).__name__}: {e}"
        )

if __name__ == "__main__":
    main()
