import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load SHAP per-class JSON
# -----------------------------
json_path = r"C:\Users\anjen\Desktop\project\anjenn\genre_classification\4_rq1_outputs\gtzan_XGB_shap_per_class.json"

with open(json_path, "r") as f:
    data = json.load(f)

# -----------------------------
# 2. Feature names (75-dim)
# -----------------------------
def generate_feature_names():
    names = []
    for i in range(20): names.append(f"mfcc_mean_{i}")
    for i in range(20): names.append(f"mfcc_std_{i}")

    names += [
        "spec_cent_mean", "spec_cent_std",
        "spec_bw_mean", "spec_bw_std",
        "spec_roll_mean", "spec_roll_std",
        "zcr_mean", "zcr_std",
        "rms_mean", "rms_std"
    ]

    for i in range(12): names.append(f"chroma_mean_{i}")
    for i in range(12): names.append(f"chroma_std_{i}")

    names.append("tempo")
    return names

feature_names = generate_feature_names()

# Verify 75 features
assert len(feature_names) == 75, f"Expected 75 features, got {len(feature_names)}"


# -----------------------------
# 3. Convert dict → matrix
# -----------------------------
genres = list(data.keys())
mat = np.array([data[g] for g in genres])

df = pd.DataFrame(mat, index=genres, columns=feature_names)

# Optionally save to CSV
df.to_csv("shap_per_class_matrix.csv")

# -----------------------------
# 4. Draw heatmap
# -----------------------------
plt.figure(figsize=(20, 6))
sns.heatmap(df, cmap="magma", xticklabels=True, yticklabels=True)

plt.xlabel("Acoustic Features", fontsize=12)
plt.ylabel("Genre", fontsize=12)
plt.title("Per-Class SHAP Values (XGBoost, 75 Features)", fontsize=14)

plt.xticks(rotation=90)
plt.tight_layout()

plt.savefig("shap_per_class_heatmap.png", dpi=300)
plt.show()
