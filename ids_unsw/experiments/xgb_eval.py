# ids_unsw/experiments/xgb_threshold.py
from __future__ import annotations
import argparse, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, precision_recall_fscore_support, roc_auc_score
)
from sklearn.preprocessing import StandardScaler

def parse_args():
    p = argparse.ArgumentParser("XGB-only threshold sweep (Cells 5–7)")
    p.add_argument("--base", default="notebooks/ids_unsw", help="Base dir with data/ and models/")
    p.add_argument("--features-json", default=None)
    p.add_argument("--xgb-pkl", default=None)
    p.add_argument("--scaler-pkl", default=None)
    p.add_argument("--test-parquet", default=None)
    p.add_argument("--recall-min", type=float, default=0.95, help="Min recall constraint for operating point")
    return p.parse_args()

def load_features(path: Path):
    obj = json.loads(path.read_text())
    return obj["features"] if isinstance(obj, dict) and "features" in obj else obj

def main():
    args = parse_args()
    base = Path(args.base)
    data_dir, model_dir = base / "data", base / "models"

    features_path = Path(args.features_json) if args.features_json else (model_dir / "feature_names.json")
    xgb_pkl_path  = Path(args.xgb_pkl)       if args.xgb_pkl       else (model_dir / "best_xgboost_model.pkl")
    scaler_path   = Path(args.scaler_pkl)    if args.scaler_pkl    else (model_dir / "scaler.pkl")
    test_path     = Path(args.test_parquet)  if args.test_parquet  else (data_dir / "UNSW_NB15_test_clean.parquet")

    # Load features + model
    feats = load_features(features_path)
    with open(xgb_pkl_path, "rb") as f:
        model = pickle.load(f)

    # Load test + scaler
    df_test = pd.read_parquet(test_path)
    label_col = next((c for c in ["label","y","is_attack","target"] if c in df_test.columns), None)
    assert label_col is not None, "Could not find label column."
    with open(scaler_path, "rb") as f:
        scaler: StandardScaler = pickle.load(f)

    X_test_np = df_test.loc[:, feats].to_numpy(dtype=np.float32, copy=False)
    X_test = scaler.transform(X_test_np).astype(np.float32, copy=False)
    y_test = df_test[label_col].astype(int).to_numpy()

    # Predict probabilities
    if hasattr(model, "predict_proba"):           # XGBClassifier
        proba = model.predict_proba(X_test)[:, 1]
    else:                                         # Booster
        import xgboost as xgb
        proba = model.predict(xgb.DMatrix(X_test))
    proba = np.asarray(proba, dtype=np.float32).ravel()

    # ---- Cell 5: sweep thresholds
    grid = np.linspace(0.50, 0.95, 15)
    qs = np.quantile(proba, np.linspace(0.50, 0.99, 12))
    thresholds = np.unique(np.clip(np.concatenate([grid, qs]), 0, 1))

    rows = []
    P = int((y_test == 1).sum()); N = int((y_test == 0).sum())
    for t in thresholds:
        y_hat = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_hat, labels=[0,1]).ravel()
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_hat, average="binary", zero_division=0)
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        rows.append(dict(
            threshold=float(t), precision=float(prec), recall=float(rec), f1=float(f1),
            FPR=float(fpr), TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn)
        ))
    sweep = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)

    print("Top of sweep:\n", sweep.head(), "\n")
    print("Bottom of sweep:\n", sweep.tail(), "\n")

    # ---- Cell 6: choose operating threshold (replicates your notebook)
    eligible = sweep[sweep["recall"] >= float(args.recall_min)].copy()
    if eligible.empty:
        print(f"No threshold reaches recall ≥ {args.recall_min:.2f} — falling back to highest recall row.")
        chosen = sweep.sort_values(["recall","threshold"], ascending=[False, True]).iloc[0]
    else:
        # minimize FPR; tie-breaker: maximize precision; then prefer the higher threshold? 
        # Notebook used ascending=[True, False, True]; we keep that to match.
        chosen = eligible.sort_values(["FPR","precision","threshold"], ascending=[True, False, True]).iloc[0]

    thr = float(chosen["threshold"])
    print(f"\nChosen threshold = {thr:.4f}")
    print(chosen.to_frame().T.to_string(index=False))

    # ---- Cell 7: final metrics at chosen threshold (+ ROC AUC reference)
    y_pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    auc = roc_auc_score(y_test, proba)
    fpr = fp / (fp + tn)
    print("\nConfusion: TN={}, FP={}, FN={}, TP={}".format(tn, fp, fn, tp))
    print({
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc),
        "FPR": float(fpr),
        "threshold": float(thr),
    })

if __name__ == "__main__":
    main()
