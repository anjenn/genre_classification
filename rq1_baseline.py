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
from xgboost import XGBClassifier

# ---------------------------------------------------------------------
RANDOM_STATE = 42
plt.rcParams["figure.dpi"] = 140
sns.set()
# ---------------------------------------------------------------------

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

# # =========================================================
# # 1) LOAD DATASET
# # =========================================================
# def load_feature_dataset(csv_path: str, label_col: str = "meta_genre"):
#     df = pd.read_csv(csv_path)
#     assert label_col in df.columns, f"{label_col} must exist in CSV."

#     print(f"\n[LOAD] {csv_path}")

#     # Load feature vectors
#     X_list = []
#     for p in df["feat_path"]:
#         x = np.load(p)
#         X_list.append(x)

#     X = np.stack(X_list)
#     y = df[label_col].astype(str).values
#     class_names = sorted(pd.unique(y))

#     print(f"[INFO] Samples={X.shape[0]}  FeatureDim={X.shape[1]}  Classes={len(class_names)}")

#     # Prefer interpretable names
#     if X.shape[1] == 75:
#         feature_cols = generate_feature_names()
#     else:
#         feature_cols = [f"f{i}" for i in range(X.shape[1])]

#     return df, X, y, feature_cols, class_names


# # =========================================================
# # 2) SPLITTING
# # =========================================================
# def train_val_test_split(X, y, test_size=0.2, val_size=0.1):
#     X_train, X_tmp, y_train, y_tmp = train_test_split(
#         X, y,
#         test_size=(test_size + val_size),
#         stratify=y,
#         random_state=RANDOM_STATE
#     )

#     # 70/10/20 split
#     val_ratio = val_size / (test_size + val_size)

#     X_val, X_test, y_val, y_test = train_test_split(
#         X_tmp, y_tmp,
#         test_size=(1 - val_ratio),
#         stratify=y_tmp,
#         random_state=RANDOM_STATE
#     )

#     print(f"[SPLIT] Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")
#     return X_train, X_val, X_test, y_train, y_val, y_test

# =========================================================
# LOADING FEATURES FROM A SPLIT CSV
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
            n_jobs=-1
        ))
    ])

# =========================================================
# 4) TRAIN + EVAL + CONFUSION MATRIX
# =========================================================
def compute_confusion(model, X_test, y_test, class_names, title="Model"):
    y_pred = model.predict(X_test)

    print(f"\n===== {title} – Classification Report =====")
    print(classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{title} – Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}_cm.png")
    plt.close()

    return cm, y_pred

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
    plt.savefig(f"{out_prefix}_shap_global.png")
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
        plt.savefig(f"{out_prefix}_shap_{c}.png")
        plt.close()


# =========================================================
# 6) MAIN RUNNER (per dataset)
# =========================================================
def run_dataset_splits(base_dir: str, dataset_name: str):
    """
    base_dir should contain:
        {dataset}_train.csv
        {dataset}_val.csv
        {dataset}_test.csv
    """
    print(f"\n================ {dataset_name} ================")

    train_csv = os.path.join(base_dir, f"{dataset_name}_train.csv")
    val_csv   = os.path.join(base_dir, f"{dataset_name}_val.csv")
    test_csv  = os.path.join(base_dir, f"{dataset_name}_test.csv")

    df_train, X_train, y_train = load_split_csv(train_csv)
    df_val,   X_val,   y_val   = load_split_csv(val_csv)
    df_test,  X_test,  y_test  = load_split_csv(test_csv)

    class_names = sorted(pd.unique(y_train))
    feature_cols = generate_feature_names()

    # ---------------- RandomForest ----------------
    rf = make_rf()
    rf.fit(X_train, y_train)
    compute_confusion(
        rf, X_test, y_test, class_names,
        title=f"{dataset_name} RF"
    )

    # ---------------- XGB ----------------
    xgb = make_xgb(num_classes=len(class_names))
    xgb.fit(X_train, y_train)
    compute_confusion(
        xgb, X_test, y_test, class_names,
        title=f"{dataset_name} XGB"
    )

    # ---------------- SHAP ----------------
    print(f"[SHAP] Computing SHAP for {dataset_name} XGB…")
    shap_values, X_test_scaled = compute_shap(xgb, X_train, X_test)

    plot_global_shap(shap_values, X_test_scaled, feature_cols, dataset_name)
    shap_dict = compute_shap_per_class(shap_values, y_test, class_names)
    plot_per_class_shap(shap_dict, feature_cols, dataset_name)

    print(f"[DONE] {dataset_name} finished.\n")
    return {
        "train": df_train,
        "test": df_test,
        "xgb": xgb,
        "shap_values": shap_values
    }


# =========================================================
# ENTRYPOINT
# =========================================================
def run_classical_rq1():

    ROOT = r"C:\Users\anjen\Desktop\project\anjenn\genre_classification\rq1_combined"

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