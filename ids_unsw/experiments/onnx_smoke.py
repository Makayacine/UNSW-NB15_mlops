# ids_unsw/experiments/onnx_smoke.py
from __future__ import annotations
import argparse, json, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, precision_recall_fscore_support
)

def load_features(p: Path):
    """Return a list of feature names from feature_names.json in any of these shapes:
       - ["f1","f2",...]
       - {"features": ["f1","f2",...]}
       - {"0":"f1","1":"f2",...} or {"f1":0,"f2":1,...} (we normalize to a list)
    """
    obj = json.loads(p.read_text())
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        if "features" in obj and isinstance(obj["features"], list):
            return obj["features"]
        # dict of index->name
        if all(k.isdigit() for k in obj.keys()):
            # sort by integer key
            return [v for _, v in sorted(((int(k), v) for k, v in obj.items()))]
        # dict of name->index
        if all(isinstance(v, int) for v in obj.values()):
            return [k for k, _ in sorted(obj.items(), key=lambda kv: kv[1])]
        # last resort: keys as list
        return list(obj.keys())
    raise ValueError(f"Unrecognized feature file format at {p}")

def parse_args():
    p = argparse.ArgumentParser("ONNX smoke test against test parquet")
    p.add_argument("--base", default="notebooks/ids_unsw", help="Base with data/ and models/")
    p.add_argument("--n", type=int, default=10, help="How many rows to preview")
    return p.parse_args()

def main():
    args = parse_args()
    base = Path(args.base)
    data_dir = base / "data"
    models_dir = base / "models"
    bundle = models_dir / "bundle_xgb"

    # ---- load bits
    features = load_features(bundle / "feature_names.json")
    meta = json.loads((bundle / "metadata.json").read_text())
    thr = float(meta.get("threshold", 0.5))

    df = pd.read_parquet(data_dir / "UNSW_NB15_test_clean.parquet")

    # find label column robustly
    label_col = next((c for c in ["label", "y", "is_attack", "target"] if c in df.columns), None)
    assert label_col is not None, f"Could not find label column in {df.columns.tolist()}"

    with open(models_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # ---- prep inputs
    X = df[features].to_numpy(dtype=np.float32, copy=False)
    X = scaler.transform(X).astype(np.float32, copy=False)
    y = df[label_col].astype(int).to_numpy()

    # ---- ONNX runtime (CPU)
    sess = ort.InferenceSession(str(bundle / "xgb.onnx"), providers=["CPUExecutionProvider"])

    def _probs_from_onnx(outs):
        # Many XGBoost-ONNX exports return two outputs: [labels, probabilities]
        if isinstance(outs, list) and len(outs) > 1 and getattr(outs[1], "ndim", 1) == 2:
            return outs[1][:, 1]
        arr = outs[0]
        return arr[:, 1] if getattr(arr, "ndim", 1) == 2 and arr.shape[1] == 2 else arr.ravel()

    # preview N rows
    outs_preview = sess.run(None, {"input": X[:args.n]})
    probs_preview = _probs_from_onnx(outs_preview)
    preds_preview = (probs_preview >= thr).astype(int)

    print("probs[:5] =", np.round(probs_preview[:5], 4))
    print("preds[:5] =", preds_preview[:5], "| thr =", thr)

    # full-set quick metrics at chosen threshold
    outs_full = sess.run(None, {"input": X})
    proba = _probs_from_onnx(outs_full)
    y_pred = (proba >= thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred, average="binary", zero_division=0)
    auc = roc_auc_score(y, proba)

    print(f"Confusion: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print({"precision": float(prec), "recall": float(rec), "f1": float(f1), "roc_auc": float(auc)})
    print("\nClassification report:")
    print(classification_report(y, y_pred, digits=4))

if __name__ == "__main__":
    main()
