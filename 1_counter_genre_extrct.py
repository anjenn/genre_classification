import pandas as pd

# Load CSV
df = pd.read_csv(r"C:\Users\anjen\Desktop\project\anjenn\genre_classification\k-music\Training\labels.csv")
# df = pd.read_csv(r"C:\Users\anjen\Desktop\project\anjenn\genre_classification\GTZAN\labels.csv")


print("=== RAW GENRE COUNTS ===")
print(df["raw_genre"].value_counts(), "\n")

print("=== META GENRE COUNTS ===")
print(df["meta_genre"].value_counts(), "\n")


# Instruments selection


# ◆ “What should I do with this distribution?”

# Use stratified split so all splits preserve the proportions.
# Otherwise val/test sets will be too small for minority classes.

# ◆ For training stability on imbalanced meta-genres:

# Keep stratified splits
# Then apply:
# class weights (softmax-weighted cross entropy)
# or weighted sampler (torch WeightedRandomSampler)
# or augment the minority classes (pitch-shift, time-stretch, EQ)


# 200 vs 1932  → Pop/Dance/RnB
# 200 vs 1020  → Rock/Metal
# 100 vs 325   → HipHop/Rap
# 100 vs 183   → Blues-like

# Option D → Train GTZAN and K-Music separately, then compare
# Best for your RQ3 (cross-cultural degradation) experiment.

