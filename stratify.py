from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("labels.csv")

# First split: train vs temp
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["meta_genre"],
    random_state=42
)

# Then split temp → val/test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["meta_genre"],
    random_state=42
)

print("Train:", train_df["meta_genre"].value_counts())
print("Val:", val_df["meta_genre"].value_counts())
print("Test:", test_df["meta_genre"].value_counts())

train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)


# ⚖️ D. For Training: How to Fix Imbalance

# Even with stratified splits, the training will still be skewed.

# Recommended solutions:
# 1. Class-Weighted Loss (Best)

# Use weights proportional to inverse frequency:
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

classes = df["meta_genre"].unique()
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=df["meta_genre"]
)

print(dict(zip(classes, weights)))
# + Add augmentation for minorities