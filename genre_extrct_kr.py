import pandas as pd

# Load CSV
# df = pd.read_csv("C:\Users\anjen\Desktop\project\anjenn\genre_classification\k-music\Training\labels.csv")
df = pd.read_csv(r"C:\Users\anjen\Desktop\project\anjenn\genre_classification\k-music\Training\labels.csv")



print("=== RAW GENRE COUNTS ===")
print(df["raw_genre"].value_counts(), "\n")

print("=== META GENRE COUNTS ===")
print(df["meta_genre"].value_counts(), "\n")


Instruments selection


# Pop/Dance/RnB dominates → 57% of all data
# Smallest class (Blues-like) is only 5.9%
# Imbalance ratio (max/min) ≈ 9.55 : 1


# ◆ “What should I do with this distribution?”

# Use stratified split so all splits preserve the proportions.
# Otherwise val/test sets will be too small for minority classes.

# ◆ For training stability on imbalanced meta-genres:

# Keep stratified splits
# Then apply:
# class weights (softmax-weighted cross entropy)
# or weighted sampler (torch WeightedRandomSampler)
# or augment the minority classes (pitch-shift, time-stretch, EQ)