"""
mfcc_classic_model.py
Baseline Classical Models:
- RandomForest
- XGBoost
- SHAP Global + Per-Class Analysis

Expected CSV format (mfcc_features.csv):
---------------------------------------------------------
meta_genre     raw_genre     instrument
wav_path       json_path     mid_path
feat_path      (optional mel_path)
---------------------------------------------------------
feat_path → points to *.npy MFCC+spectral 1D vector (~75–200 dims)
"""

import os
import json
import joblib
import shap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Dict, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# ---------------------------------------------------------------------
RANDOM_STATE = 42
plt.rcParams["figure.dpi"] = 140
sns.set()
# ---------------------------------------------------------------------

OUTPUT_ROOT = r"C:\Users\anjen\Desktop\project\anjenn\genre_classification\4_rq1_outputs"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

def out(path):
    return os.path.join(OUTPUT_ROOT, path)

# =========================================================
# 0) FEATURE NAMES (Optional but recommended)
# =========================================================
def generate_feature_names():
    """
    Maps your exact 75-dim design to interpretable names.
    """

    names = []

    # MFCC mean/std (40)
    for i in range(20): names.append(f"mfcc_mean_{i}")
    for i in range(20): names.append(f"mfcc_std_{i}")

    # Spectral features (10)
    names += [
        "spec_cent_mean", "spec_cent_std",
        "spec_bw_mean", "spec_bw_std",
        "spec_roll_mean", "spec_roll_std",
        "zcr_mean", "zcr_std",
        "rms_mean", "rms_std",
    ]

    # Chroma (24)
    for i in range(12): names.append(f"chroma_mean_{i}")
    for i in range(12): names.append(f"chroma_std_{i}")

    # Tempo (1)
    names.append("tempo")

    assert len(names) == 75, f"Expected 75 names, got {len(names)}"
    return names

# =========================================================
# 2) LOADING FEATURES FROM A SPLIT CSV
# =========================================================
def load_split_csv(csv_path: str, label_col="meta_genre"):
    df = pd.read_csv(csv_path)
    assert label_col in df.columns, f"{label_col} missing in {csv_path}"

    X = np.stack([np.load(p) for p in df["feat_path"]])
    y = df[label_col].astype(str).values

    return df, X, y

# =========================================================
# 3) MODEL MAKERS
# =========================================================

def make_rf():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            class_weight='balanced',
            n_jobs=-1,
            random_state=RANDOM_STATE
        ))
    ])

def make_xgb(num_classes: int):
    """
    Bare XGBClassifier; class weights will be applied via sample_weight at fit time.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ))
    ])

# =========================================================
# 4) TRAIN + EVAL + CONFUSION MATRIX
# =========================================================
def compute_confusion(y_test, y_pred, class_names, save_prefix):
    """
    y_pred already inverse-transformed (strings).
    """
    # ----- Save predictions -----
    pred_csv = out(save_prefix + "_predictions.csv")
    pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred
    }).to_csv(pred_csv, index=False)

    # ----- Report -----
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    report_json = out(save_prefix + "_report.json")
    with open(report_json, "w") as f:
        json.dump(report, f, indent=4)

    print(f"\n===== {save_prefix} Classification Report =====")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # ----- Confusion Matrix -----
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    cm_json = out(save_prefix + "_confusion.json")
    with open(cm_json, "w") as f:
        json.dump(cm.tolist(), f, indent=4)

    # ----- Plot -----
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{save_prefix} – Normalized Confusion Matrix")
    plt.tight_layout()
    cm_png = out(save_prefix + "_cm.png")
    plt.savefig(cm_png)
    plt.close()

    return cm, report

# =========================================================
# 5) SHAP FUNCTIONS
# =========================================================
def compute_shap(pipeline, X_train, X_test):
    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled)

    return shap_values, X_test_scaled


def plot_global_shap(shap_values, X_scaled, feature_names, out_prefix):
    shap_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    mean_abs = shap_abs.mean(axis=0)
    order = np.argsort(mean_abs)[::-1]

    top = order[:20]

    plt.figure(figsize=(6, 8))
    plt.barh(range(len(top)), mean_abs[top][::-1])
    plt.yticks(range(len(top)), [feature_names[i] for i in top][::-1])
    plt.xlabel("Mean |SHAP|")
    plt.title("Global Feature Importance")
    plt.tight_layout()
    plt.savefig(out(f"{out_prefix}_shap_global.png"))
    plt.close()


def compute_shap_per_class(shap_values, y_test, class_names):
    result = {}
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    for c in class_names:
        ci = class_to_idx[c]
        mask = (y_test == c)
        if mask.sum() == 0:
            continue

        v = np.abs(shap_values[ci][mask])  # (Nc, F)
        result[c] = v.mean(axis=0)

    return result


def plot_per_class_shap(per_class_dict, feature_names, out_prefix, max_display=10):
    for c, vec in per_class_dict.items():
        order = np.argsort(vec)[::-1][:max_display]
        feats = [feature_names[i] for i in order]
        vals = vec[order]

        plt.figure(figsize=(6, 4))
        plt.barh(range(max_display), vals[::-1])
        plt.yticks(range(max_display), feats[::-1])
        plt.title(f"Per-Class SHAP – {c}")
        plt.tight_layout()
        plt.savefig(out(f"{out_prefix}_shap_{c}.png"))
        plt.close()

# =========================================================
# SHAP EXPORTS (global + per-class)
# =========================================================
def save_shap_outputs(shap_values, feature_names, y_test, class_names, save_prefix):
    """
    Stable SHAP saver:
    - ensures consistent feature counts across classes
    - trims ragged shapes
    - exports global & per-class SHAP cleanly
    """

    # Convert raw SHAP list → array (C, N, F)
    # Some classes may have different F due to pipeline/SHAP quirks
    shapes = [sv.shape[1] for sv in shap_values]
    F_min = min(shapes)   # e.g., 75 or 76

    # Trim all SHAP[class] to F_min
    trimmed = [sv[:, :F_min] for sv in shap_values]

    # Trim feature_names also
    feature_names = feature_names[:F_min]

    # ========== Global SHAP ==========
    # Concatenate classes → (C*N, F_min)
    concat = np.vstack(trimmed)
    global_abs = np.mean(np.abs(concat), axis=0)

    df_global = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": global_abs
    }).sort_values("mean_abs_shap", ascending=False)

    df_global.to_csv(out(save_prefix + "_shap_global.csv"), index=False)

    # ========== Per-class SHAP ==========
    y_test_arr = np.array(y_test)
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    per_class_dict = {}
    for c in class_names:
        class_idx = class_to_idx[c]
        mask = (y_test_arr == c)
        if mask.sum() == 0:
            continue

        values = trimmed[class_idx][mask]      # (Nc, F_min)
        per_class_dict[c] = np.mean(np.abs(values), axis=0).tolist()

    with open(out(save_prefix + "_shap_per_class.json"), "w") as f:
        json.dump(per_class_dict, f, indent=4)

    # ========== Raw SHAP ==========
    np.save(out(save_prefix + "_shap_raw.npy"), trimmed)

    print(f"[SHAP] Saved global + per-class SHAP using {F_min} features.")


# =========================================================
# 6) MAIN RUNNER (per dataset)
# =========================================================
def run_dataset_splits(base_dir: str, dataset_name: str):
    print(f"\n================ {dataset_name} ================")

    train_csv = os.path.join(base_dir, f"{dataset_name}_train.csv")
    val_csv   = os.path.join(base_dir, f"{dataset_name}_val.csv")
    test_csv  = os.path.join(base_dir, f"{dataset_name}_test.csv")
    weights_json = os.path.join(base_dir, f"{dataset_name}_class_weights.json")

    df_train, X_train, y_train = load_split_csv(train_csv)
    df_val,   X_val,   y_val   = load_split_csv(val_csv)
    df_test,  X_test,  y_test  = load_split_csv(test_csv)

    class_names = sorted(pd.unique(y_train))
    feature_cols = generate_feature_names()

    # Load class weights
    with open(weights_json, "r") as f:
        class_weights = json.load(f)

    # Label encoder (for XGB)
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc   = le.transform(y_val)
    y_test_enc  = le.transform(y_test)

    # =========================================================
    # RandomForest (string labels)
    # =========================================================
    rf = make_rf()
    rf.fit(X_train, y_train)

    rf_preds = rf.predict(X_test)

    compute_confusion(
        y_test, rf_preds, class_names,
        save_prefix=f"{dataset_name}_RF"
    )

    # =========================================================
    # XGBoost (encoded labels + per-sample class weights)
    # =========================================================
    xgb = make_xgb(len(class_names))

    # Build per-sample weights from class_weights dict
    # class_weights comes from *_class_weights.json, keys are labels like "Blues-like"
    sample_weight_train = np.array([class_weights[label] for label in y_train])

    # When using a Pipeline, pass weights to the final estimator with 'model__'
    xgb.fit(
        X_train,
        y_train_enc,
        model__sample_weight=sample_weight_train
    )

    xgb_preds_enc = xgb.predict(X_test)
    xgb_preds = le.inverse_transform(xgb_preds_enc)

    compute_confusion(
        y_test, xgb_preds, class_names,
        save_prefix=f"{dataset_name}_XGB"
    )

    # ===============================
    # FIXED SHAP for XGB + Pipeline
    # ===============================
    # ---------- SHAP normalization ----------
    print(f"[SHAP] Computing SHAP for {dataset_name}...")

    scaler = xgb.named_steps["scaler"]
    model  = xgb.named_steps["model"]

    X_test_scaled = scaler.transform(X_test)
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    raw = explainer.shap_values(X_test_scaled)

    # Convert raw to numpy array
    raw = np.array(raw)

    num_classes = len(class_names)

    clean = []

    # CASE A – XGB returns class-wise list: [ (N,F), (N,F), ... ] length C
    if isinstance(raw, list):
        for sv in raw:
            sv = np.array(sv)
            if sv.ndim != 2:
                raise ValueError(f"[SHAP] Unexpected shape inside list: {sv.shape}")
            clean.append(sv)

    # CASE B – XGB returns (C, N, F)
    elif raw.ndim == 3 and raw.shape[0] == num_classes:
        for ci in range(num_classes):
            clean.append(raw[ci])

    # CASE C – XGB returns (N, C, F)
    elif raw.ndim == 3 and raw.shape[1] == num_classes:
        for ci in range(num_classes):
            clean.append(raw[:, ci, :])

    # CASE D – XGB returns (N, F, C)   ← YOUR ACTUAL CASE
    elif raw.ndim == 3 and raw.shape[2] == num_classes:
        for ci in range(num_classes):
            clean.append(raw[:, :, ci])

    else:
        raise ValueError(f"[SHAP] Unrecognized SHAP output shape: {raw.shape}")

    # clean = list of C arrays shaped (N,F)




# =========================================================
# ENTRYPOINT
# =========================================================
def run_classical_rq1():

    ROOT = r"C:\Users\anjen\Desktop\project\anjenn\genre_classification\rq1_combined"

    run_dataset_splits(
        base_dir=os.path.join(ROOT, "combined", "split"),
        dataset_name="combined"
    )

    run_dataset_splits(
        base_dir=os.path.join(ROOT, "gtzan", "split"),
        dataset_name="gtzan"
    )

    run_dataset_splits(
        base_dir=os.path.join(ROOT, "korean", "split"),
        dataset_name="korean"
    )


if __name__ == "__main__":
    run_classical_rq1()
    print("\n=== ALL DONE ===")