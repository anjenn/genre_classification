import pandas as pd
import os
import re

df = pd.read_csv(r"C:\Users\anjen\Desktop\project\anjenn\genre_classification\k-music\Training\labels.csv")

# Extract the folder names from wav path
def extract_instrument_info(path):
    folder = os.path.basename(os.path.dirname(path))  
    # => "TS_03.Piano_02.Bright Acoustic Piano"

    # Split on first '.' → ["TS_03", "Piano_02.Bright Acoustic Piano"]
    try:
        _, rest = folder.split(".", 1)
    except ValueError:
        return None, None

    # rest → "Piano_02.Bright Acoustic Piano"
    # Split into "Piano" and "Bright Acoustic Piano"
    try:
        inst_family, inst_sub = rest.split("_", 1)
        inst_sub = inst_sub.split(".", 1)[1]  # remove "02."
    except:
        return inst_family, None

    # Clean underscores if needed
    inst_family = inst_family.strip()
    inst_sub = inst_sub.strip()

    return inst_family, inst_sub

# Apply to dataframe
df["instrument"], df["subinstrument"] = zip(*df["wav"].map(extract_instrument_info))

# Save updated CSV (optional)
df.to_csv("labels_with_instruments.csv", index=False, encoding="utf-8-sig")

# print(df[["instrument", "subinstrument"]].head())

unique_instruments = df["instrument"].unique().tolist()
unique_subinstruments = df["subinstrument"].unique().tolist()

print("Instruments:", unique_instruments)
print("Sub-instruments:", unique_subinstruments)

KEEP_INSTRUMENTS = ["Piano", "Guitar", "Bass", "Strings"]

df_filtered = df[df["instrument"].isin(KEEP_INSTRUMENTS)]

df_filtered.to_csv("labels_instrument_filtered.csv", index=False, encoding="utf-8-sig")
