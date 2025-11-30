"""
split_datasets.py
Creates stratified 70/15/15 splits for:
- Korean dataset
- GTZAN dataset
- Combined (Korean + GTZAN)

Outputs:
    {dataset}_train.csv
    {dataset}_val.csv
    {dataset}_test.csv

Also prints and saves class weights (balanced).
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ============================================================
# Utility: Stratified 70/15/15 Split
# ============================================================
def stratified_split(df, label_col="meta_genre", seed=42):
    """
    Returns: train_df, val_df, test_df
    Ensures stratified sampling by label_col.
    """

    assert label_col in df.columns, f"{label_col} not found in dataframe."

    # 70% train
    train_df, tmp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df[label_col],
        random_state=seed
    )

    # tmp → 15% val, 15% test
    val_df, test_df = train_test_split(
        tmp_df,
        test_size=0.50,
        stratify=tmp_df[label_col],
        random_state=seed
    )

    print(f"[SPLIT] Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df

# ============================================================
# Utility: Save splits + class weights
# ============================================================
def save_splits(train_df, val_df, test_df, out_dir, dataset_name, label_col="meta_genre"):
    os.makedirs(out_dir, exist_ok=True)

    # File paths
    train_path = os.path.join(out_dir, f"{dataset_name}_train.csv")
    val_path   = os.path.join(out_dir, f"{dataset_name}_val.csv")
    test_path  = os.path.join(out_dir, f"{dataset_name}_test.csv")
    weights_path = os.path.join(out_dir, f"{dataset_name}_class_weights.json")

    # Save dataframes
    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    # Compute class weights
    classes = np.unique(train_df[label_col])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df[label_col]
    )
    class_weights = {cls: float(w) for cls, w in zip(classes, weights)}

    print(f"\n=== {dataset_name.upper()} CLASS WEIGHTS ===")
    print(class_weights)

    # Save weights to JSON
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(class_weights, f, indent=4)

    return class_weights

# # ============================================================
# # 1. Korean Dataset
# # ============================================================
# korean_df = pd.read_csv(r"C:\Users\anjen\Desktop\project\anjenn\genre_classification\k-music\Training\labels.csv")

# print("\n===== Splitting Korean Dataset =====")
# k_train, k_val, k_test = stratified_split(korean_df)

# k_weights = save_splits(
#     k_train, k_val, k_test,
#     out_dir=os.path.join(BASE, "korean", "split"),
#     dataset_name="korean"
# )

# ============================================================
# Helper: Validate essential columns before splitting
# ============================================================
def validate_input_df(df, name):
    required = ["meta_genre", "feat_path"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"[ERROR] Column '{col}' missing in {name} dataframe.")

    print(f"[OK] {name}: Columns validated.")


# ============================================================
# MAIN EXECUTION
# ============================================================
def main():
    
    # Base directory where split folders will be created
    BASE = r"C:\Users\anjen\Desktop\project\anjenn\genre_classification\rq1_combined"

    # ---------------------------
    # 1. Korean Dataset
    # ---------------------------
    korean_csv = r"C:\Users\anjen\Desktop\project\anjenn\genre_classification\k-music\Training\labels.csv"
    korean_df = pd.read_csv(korean_csv)
    validate_input_df(korean_df, "Korean")

    print("\n===== Splitting Korean Dataset =====")
    k_train, k_val, k_test = stratified_split(korean_df)

    save_splits(
        k_train, k_val, k_test,
        out_dir=os.path.join(BASE, "korean", "split"),
        dataset_name="korean"
    )

    # ---------------------------
    # 2. GTZAN Dataset
    # ---------------------------
    gtzan_csv = r"C:\Users\anjen\Desktop\project\anjenn\genre_classification\GTZAN\labels.csv"
    gtzan_df = pd.read_csv(gtzan_csv)
    validate_input_df(gtzan_df, "GTZAN")

    print("\n===== Splitting GTZAN Dataset =====")
    g_train, g_val, g_test = stratified_split(gtzan_df)

    save_splits(
        g_train, g_val, g_test,
        out_dir=os.path.join(BASE, "gtzan", "split"),
        dataset_name="gtzan"
    )

    # ---------------------------
    # 3. Combined Dataset
    # ---------------------------
    combined_df = pd.concat([korean_df, gtzan_df], ignore_index=True)

    print("\n===== Splitting Combined Dataset =====")
    c_train, c_val, c_test = stratified_split(combined_df)

    save_splits(
        c_train, c_val, c_test,
        out_dir=os.path.join(BASE, "combined", "split"),
        dataset_name="combined"
    )

    print("\nDONE — All datasets split + class weights saved.")

if __name__ == "__main__":
    main()