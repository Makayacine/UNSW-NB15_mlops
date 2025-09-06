# ids_unsw/experiments/register_onnx.py
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import mlflow
import mlflow.onnx
import onnx

def load_features(p: Path):
    obj = json.loads(p.read_text())
    return obj["features"] if isinstance(obj, dict) and "features" in obj else obj

def parse_args():
    p = argparse.ArgumentParser("Register xgb.onnx into MLflow Model Registry")
    p.add_argument("--base", default="notebooks/ids_unsw", help="Base path with models/")
    p.add_argument("--mlflow-uri", required=True, help="MLflow tracking URI")
    p.add_argument("--mlflow-exp", default="unsw-nb15", help="Experiment name")
    p.add_argument("--name", default="unsw_xgb_ids_onnx", help="Registered model base name")
    return p.parse_args()

def main():
    args = parse_args()
    base = Path(args.base)
    model_dir = base / "models"

    onnx_path = model_dir / "xgb.onnx"
    feats_path = model_dir / "feature_names.json"
    meta_path  = model_dir / "metadata.json"

    assert onnx_path.exists(), f"Missing {onnx_path}"
    features = load_features(feats_path) if feats_path.exists() else list(range(34))
    n_features = len(features)

    # MLflow setup
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.mlflow_exp)

    # Pick a unique registry name if needed
    client = mlflow.tracking.MlflowClient()
    reg_name = args.name
    try:
        existing = client.search_registered_models(filter_string=f"name='{reg_name}'")
        if existing:
            reg_name = f"{reg_name}_{time.strftime('%Y%m%d-%H%M%S')}"
    except Exception:
        pass

    input_example = np.zeros((1, n_features), dtype=np.float32)

    # Load ONNX *model object* (not a path string)
    onnx_model = onnx.load(str(onnx_path))

    with mlflow.start_run(run_name="register_xgb_onnx") as run:
        # Prefer the newer API that takes `name=`; fall back to artifact_path if needed
        try:
            mlflow.onnx.log_model(
                onnx_model=onnx_model,
                name="xgb",
                registered_model_name=reg_name,
                input_example=input_example,
            )
        except TypeError:
            mlflow.onnx.log_model(
                onnx_model=onnx_model,
                artifact_path="xgb",
                registered_model_name=reg_name,
                input_example=input_example,
            )

        # Extra helpful artifacts
        if feats_path.exists():
            mlflow.log_artifact(str(feats_path), artifact_path="xgb")
        if meta_path.exists():
            mlflow.log_artifact(str(meta_path), artifact_path="xgb")

        print(f"🏃 View run register_xgb_onnx at: {mlflow.get_tracking_uri()}/#/experiments/{mlflow.get_experiment_by_name(args.mlflow_exp).experiment_id}/runs/{run.info.run_id}")

    print(f"⯑ Registered ONNX as `{reg_name}`. Open your MLflow UI to see it in the Model Registry.")

if __name__ == "__main__":
    main()
