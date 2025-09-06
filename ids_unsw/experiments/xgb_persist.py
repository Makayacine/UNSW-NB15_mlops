# ids_unsw/experiments/xgb_persist.py
from __future__ import annotations
import argparse, json, pickle, os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler
import mlflow

def load_features(p: Path):
    obj = json.loads(p.read_text())
    return obj["features"] if isinstance(obj, dict) and "features" in obj else obj

def choose_threshold(y, proba, recall_min=0.95):
    grid = np.linspace(0.50, 0.95, 15)
    qs = np.quantile(proba, np.linspace(0.50, 0.99, 12))
    thresholds = np.unique(np.clip(np.concatenate([grid, qs]), 0, 1))
    rows = []
    for t in thresholds:
        y_hat = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, y_hat, labels=[0,1]).ravel()
        prec, rec, f1, _ = precision_recall_fscore_support(y, y_hat, average="binary", zero_division=0)
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        rows.append((t, prec, rec, f1, fpr))
    df = pd.DataFrame(rows, columns=["threshold","precision","recall","f1","FPR"]).sort_values("threshold")
    elig = df[df["recall"] >= float(recall_min)]
    if elig.empty:
        chosen = df.sort_values(["recall","threshold"], ascending=[False, True]).iloc[0]
    else:
        chosen = elig.sort_values(["FPR","precision","threshold"], ascending=[True, False, True]).iloc[0]
    return float(chosen["threshold"])

def parse_args():
    p = argparse.ArgumentParser("Persist XGB threshold + metrics to models/metadata.json and MLflow (Cell 8)")
    p.add_argument("--base", default="notebooks/ids_unsw", help="Base with data/ and models/")
    p.add_argument("--features-json", default=None)
    p.add_argument("--xgb-pkl", default=None)
    p.add_argument("--scaler-pkl", default=None)
    p.add_argument("--test-parquet", default=None)
    p.add_argument("--threshold", type=float, default=None, help="If omitted, will sweep with --recall-min")
    p.add_argument("--recall-min", type=float, default=0.95)
    p.add_argument("--mlflow-uri", default="http://host.docker.internal:5000")
    p.add_argument("--mlflow-exp", default="unsw-nb15")
    return p.parse_args()

def main():
    args = parse_args()
    base = Path(args.base)
    data_dir, model_dir = base / "data", base / "models"

    features_path = Path(args.features_json) if args.features_json else (model_dir / "feature_names.json")
    model_path    = Path(args.xgb_pkl)       if args.xgb_pkl       else (model_dir / "best_xgboost_model.pkl")
    scaler_path   = Path(args.scaler_pkl)    if args.scaler_pkl    else (model_dir / "scaler.pkl")
    test_path     = Path(args.test_parquet)  if args.test_parquet  else (data_dir / "UNSW_NB15_test_clean.parquet")

    feats = load_features(features_path)
    with open(model_path, "rb") as f: model = pickle.load(f)
    with open(scaler_path, "rb") as f: scaler: StandardScaler = pickle.load(f)

    df_test = pd.read_parquet(test_path)
    label = next((c for c in ["label","y","is_attack","target"] if c in df_test.columns), None)
    assert label is not None, "Label column not found."

    X_np = df_test[feats].to_numpy(np.float32, copy=False)
    X = scaler.transform(X_np).astype(np.float32, copy=False)
    y = df_test[label].astype(int).to_numpy()

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:,1]
    else:
        import xgboost as xgb
        proba = model.predict(xgb.DMatrix(X))
    proba = np.asarray(proba, dtype=np.float32).ravel()

    thr = float(args.threshold) if args.threshold is not None else choose_threshold(y, proba, args.recall_min)
    y_pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0,1]).ravel()
    prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred, average="binary", zero_division=0)
    auc = roc_auc_score(y, proba)
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    # --- save/update metadata.json
    meta_path = model_dir / "metadata.json"
    meta = {}
    if meta_path.exists():
        try: meta = json.loads(meta_path.read_text())
        except Exception: meta = {}
    meta.update({
        "champion": "xgboost",
        "threshold": float(thr),
        "n_features": int(X.shape[1]),
        "metrics_at_threshold": {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "roc_auc": float(auc),
            "FPR": float(fpr),
            "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        },
    })
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"⯑ Saved threshold + metrics to {meta_path}")

    # --- log to MLflow
    os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.mlflow_exp)

    with mlflow.start_run(run_name=f"xgb_threshold@{thr:.4f}"):
        mlflow.log_params({
            "model": "xgboost",
            "threshold": float(thr),
            "n_features": int(X.shape[1]),
        })
        mlflow.log_metrics({
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "roc_auc": float(auc),
            "FPR": float(fpr),
            "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        })
        # optional artifacts if present
        for name in ("xgb.onnx", "feature_names.json", "metadata.json"):
            p = model_dir / name
            if p.exists():
                mlflow.log_artifact(str(p), artifact_path="xgb")

    print("⯑ Logged to MLflow.")

if __name__ == "__main__":
    main()
