"""
cross_cultural_zero_shot.py

E5 – Zero-shot Cross-Cultural Evaluation (light RQ3)

- Train RF + XGB on GTZAN (meta-genre)
- Evaluate on:
    1) GTZAN test (source baseline)
    2) Korean test (zero-shot cross-cultural)
- Compute accuracy / macro-F1 drop
- Compare SHAP global importance between GTZAN-test vs Korean-test

Assumes pre-split CSVs from split_datasets.py exist in:

C:/Users/anjen/Desktop/project/anjenn/genre_classification/rq1_combined/
    gtzan/split/gtzan_train.csv
    gtzan/split/gtzan_test.csv
    korean/split/korean_test.csv
"""

import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier

RANDOM_STATE = 42
plt.rcParams["figure.dpi"] = 140
sns.set()


# =========================================================
# Feature name helper (same as mfcc_classic_model.py)
# =========================================================
def generate_feature_names():
    names = []
    for i in range(20): names.append(f"mfcc_mean_{i}")
    for i in range(20): names.append(f"mfcc_std_{i}")
    names += [
        "spec_cent_mean", "spec_cent_std",
        "spec_bw_mean", "spec_bw_std",
        "spec_roll_mean", "spec_roll_std",
        "zcr_mean", "zcr_std",
        "rms_mean", "rms_std",
    ]
    for i in range(12): names.append(f"chroma_mean_{i}")
    for i in range(12): names.append(f"chroma_std_{i}")
    names.append("tempo")
    assert len(names) == 75, f"Expected 75 names, got {len(names)}"
    return names


# =========================================================
# Loading helpers
# =========================================================
def load_split_csv(csv_path: str, label_col="meta_genre"):
    df = pd.read_csv(csv_path)
    assert label_col in df.columns, f"{label_col} missing in {csv_path}"
    X = np.stack([np.load(p) for p in df["feat_path"]])
    y = df[label_col].astype(str).values
    return df, X, y


def make_rf():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            class_weight="balanced",
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
            n_jobs=-1,
            random_state=RANDOM_STATE
        ))
    ])


# =========================================================
# Evaluation
# =========================================================
def eval_model(model, X, y, class_names, title=""):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average="macro")

    print(f"\n===== {title} =====")
    print(classification_report(y, y_pred, target_names=class_names))
    print(f"Accuracy    : {acc:.4f}")
    print(f"Macro F1    : {macro_f1:.4f}")

    cm = confusion_matrix(y, y_pred, labels=class_names)
    return acc, macro_f1, cm, y_pred


def print_confusion(cm, class_names, title):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(title.replace(" ", "_") + "_cm.png")
    plt.close()


# =========================================================
# SHAP (for XGB)
# =========================================================
def compute_shap_for_dataset(xgb_pipeline, X_train_src, X_test_target, out_prefix):
    scaler = xgb_pipeline.named_steps["scaler"]
    model = xgb_pipeline.named_steps["model"]

    X_train_scaled = scaler.transform(X_train_src)
    X_target_scaled = scaler.transform(X_test_target)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_target_scaled)  # list[num_classes][N,F]

    feature_names = generate_feature_names()

    # Global mean |SHAP|
    shap_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0)  # [N,F]
    mean_abs = shap_abs.mean(axis=0)  # [F]
    order = np.argsort(mean_abs)[::-1][:20]

    plt.figure(figsize=(6, 8))
    plt.barh(range(len(order)), mean_abs[order][::-1])
    plt.yticks(range(len(order)), [feature_names[i] for i in order][::-1])
    plt.xlabel("Mean |SHAP|")
    plt.title(f"Global Feature Importance – {out_prefix}")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_shap_global.png")
    plt.close()

    return shap_values, X_target_scaled


# =========================================================
# MAIN: Zero-shot Cross-Cultural Evaluation
# =========================================================
def main():
    ROOT = r"C:\Users\anjen\Desktop\project\anjenn\genre_classification\rq1_combined"

    # 1) Load GTZAN splits (source)
    gtzan_dir = os.path.join(ROOT, "gtzan", "split")
    _, Xg_train, yg_train = load_split_csv(os.path.join(gtzan_dir, "gtzan_train.csv"))
    _, Xg_test, yg_test   = load_split_csv(os.path.join(gtzan_dir, "gtzan_test.csv"))

    # 2) Load Korean test (target)
    korean_dir = os.path.join(ROOT, "korean", "split")
    _, Xk_test, yk_test = load_split_csv(os.path.join(korean_dir, "korean_test.csv"))

    # Use GTZAN training labels as class reference
    class_names = sorted(np.unique(yg_train))
    print("\n[CLASSES] Meta genres (GTZAN train):", class_names)
    print("[INFO] Assuming Korean meta_genre labels use the same set / subset.")

    # =====================================================
    # Train RF and XGB on GTZAN only
    # =====================================================
    rf = make_rf()
    xgb = make_xgb(num_classes=len(class_names))

    print("\n[TRAIN] Training RF on GTZAN train...")
    rf.fit(Xg_train, yg_train)

    print("[TRAIN] Training XGB on GTZAN train...")
    xgb.fit(Xg_train, yg_train)

    # =====================================================
    # Evaluate on GTZAN test (source baseline)
    # =====================================================
    rf_acc_src, rf_f1_src, rf_cm_src, _ = eval_model(
        rf, Xg_test, yg_test, class_names, title="RF – GTZAN test (source)"
    )
    xgb_acc_src, xgb_f1_src, xgb_cm_src, _ = eval_model(
        xgb, Xg_test, yg_test, class_names, title="XGB – GTZAN test (source)"
    )

    print_confusion(rf_cm_src, class_names, "RF GTZAN Test")
    print_confusion(xgb_cm_src, class_names, "XGB GTZAN Test")

    # =====================================================
    # Evaluate on Korean test (zero-shot target)
    # =====================================================
    rf_acc_tgt, rf_f1_tgt, rf_cm_tgt, _ = eval_model(
        rf, Xk_test, yk_test, class_names, title="RF – Korean test (zero-shot)"
    )
    xgb_acc_tgt, xgb_f1_tgt, xgb_cm_tgt, _ = eval_model(
        xgb, Xk_test, yk_test, class_names, title="XGB – Korean test (zero-shot)"
    )

    print_confusion(rf_cm_tgt, class_names, "RF Korean Test")
    print_confusion(xgb_cm_tgt, class_names, "XGB Korean Test")

    # =====================================================
    # Compute cross-cultural drops
    # =====================================================
    print("\n===== Cross-Cultural Performance Drop (GTZAN → Korean) =====")
    print(f"RF   ΔAccuracy = {rf_acc_src - rf_acc_tgt:.4f}")
    print(f"RF   ΔMacroF1  = {rf_f1_src  - rf_f1_tgt:.4f}")
    print(f"XGB  ΔAccuracy = {xgb_acc_src - xgb_acc_tgt:.4f}")
    print(f"XGB  ΔMacroF1  = {xgb_f1_src  - xgb_f1_tgt:.4f}")

    # =====================================================
    # SHAP comparison for XGB
    # =====================================================
    print("\n[SHAP] GTZAN test (source) feature importance...")
    shap_gtzan, _ = compute_shap_for_dataset(
        xgb, Xg_train, Xg_test, out_prefix="cross_gtzan"
    )

    print("[SHAP] Korean test (target) feature importance...")
    shap_korean, _ = compute_shap_for_dataset(
        xgb, Xg_train, Xk_test, out_prefix="cross_korean"
    )

    print("\n[DONE] Zero-shot cross-cultural evaluation complete.")


if __name__ == "__main__":
    main()



# This script gives you:

# RF + XGB zero-shot GTZAN → Korean

# Accuracy & macro F1 drop

# Confusion matrices (4 PNGs)

# Global SHAP plots for GTZAN vs Korean (2 PNGs)

# You can now discuss:

# Which features remain important cross-culturally

# Which ones “break” when moving to Korean loops (e.g., chroma / MFCCs)
