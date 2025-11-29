import os
import json
import librosa
import numpy as np
import pandas as pd
import cv2

# -----------------------------------------
# Meta-genre mapping
# -----------------------------------------
META_MAP = {
    "Folk&Blues": "Blues-like",
    "Rock&Metal": "Rock/Metal",
    "Rap&Hiphop": "HipHop/Rap",
    "Dance": "Pop/Dance/RnB",
    "RnB&Soul": "Pop/Dance/RnB",
}

KEEP_META = set(META_MAP.values())

Instruments selection

# ------------------------------------------------------------
# USER PATHS (EDIT THESE ONLY)
# ------------------------------------------------------------
ROOT = r"C:\Users\anjen\Desktop\project\anjenn\genre_classification\k-music\Training"

SRC_DIR = os.path.join(ROOT, "01.원천데이터")
LBL_DIR = os.path.join(ROOT, "02.라벨링데이터")

OUTPUT_MELS = os.path.join(ROOT, "processed_mels")
os.makedirs(OUTPUT_MELS, exist_ok=True)

CSV_PATH = os.path.join(ROOT, "labels.csv")

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def extract_loop_index(filename):
    """
    Given filename like:
    Vocal&Melody_Voice Aahs_0092224_Folk&Blues_100BPM.wav
    Return: "0092224"
    """
    parts = filename.split("_")
    # Identify the part that is the loop index (always digits)
    for p in parts:
        if p.isdigit():
            return p
    return None


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Fix length to make it 5 secs
def load_audio_fixed(wav_path, target_sr=22050, target_len_sec=5):
    y, sr = librosa.load(wav_path, sr=target_sr, mono=True)
    # NEW: use keyword argument for size
    y = librosa.util.fix_length(y, size=target_len_sec * target_sr)
    return y, sr

# Generating melspec
def make_melspec(y, sr, size=128):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=size, n_fft=2048, hop_length=512
    )
    S = librosa.power_to_db(S, ref=np.max)

    # Resize to 128×128 (time axis depends on hop_length)
    S_resized = cv2.resize(S, (size, size))
    return S_resized


# ------------------------------------------------------------
# Scan folders & collect file paths
# ------------------------------------------------------------
def collect_files(root, extensions):
    """
    Walks root and returns {loopIndex: full_path} for each file type.
    """
    result = {}
    for folder, _, files in os.walk(root):
        for f in files:
            if any(f.lower().endswith(ext) for ext in extensions):
                idx = extract_loop_index(f)
                if idx:
                    result.setdefault(idx, []).append(os.path.join(folder, f))
    return result


print("Scanning source WAV files...")
wav_files = collect_files(SRC_DIR, [".wav"])

print("Scanning label JSON/MID files...")
label_files = collect_files(LBL_DIR, [".json", ".mid"])


# ------------------------------------------------------------
# Match WAV ↔ JSON by loop index
# ------------------------------------------------------------
matched = []

for idx, wavs in wav_files.items():
    if idx not in label_files:
        print(f"[WARN] No label found for loopIndex {idx}")
        continue

    json_path = None
    mid_path = None

    # classify json vs mid
    for fp in label_files[idx]:
        if fp.lower().endswith(".json"):
            json_path = fp
        elif fp.lower().endswith(".mid"):
            mid_path = fp

    if json_path is None:
        print(f"[WARN] No JSON for {idx}; skipping.")
        continue

    # usually there is exactly one wav
    for wav_path in wavs:
        matched.append((idx, wav_path, json_path, mid_path))


print(f"Matched samples: {len(matched)}")


# ------------------------------------------------------------
# Process all matched samples
# ------------------------------------------------------------
records = []

for idx, wav_path, json_path, mid_path in matched:
    try:
        # Load JSON → extract original genre
        meta = load_json(json_path)
        raw_genre = meta["dataSet"]["loopInfo"]["genre"]  # e.g. Folk&Blues

        # Skip if raw genre not recognized
        if raw_genre not in META_MAP:
            print(f"[SKIP] {idx} rawGenre={raw_genre} not in target groups")
            continue

        # Map to meta-genre
        meta_genre = META_MAP[raw_genre]

        # Load audio → fixed 5 sec
        y, sr = load_audio_fixed(wav_path)

        # Make mel-spectrogram 128×128
        mel = make_melspec(y, sr, size=128)

        # Save .npy
        out_path = os.path.join(OUTPUT_MELS, f"{idx}.npy")
        np.save(out_path, mel)

        # Add record
        records.append([
            idx,
            raw_genre,
            meta_genre,
            wav_path,
            json_path,
            mid_path,
            out_path
        ])

        print(f"[OK] {idx} → {meta_genre}")

    except Exception as e:
        print(f"[ERROR] idx={idx} | {e}")

# ------------------------------------------------------------
# Save CSV
# ------------------------------------------------------------
df = pd.DataFrame(records, columns=[
    "loopIndex",
    "raw_genre",
    "meta_genre",
    "wav",
    "json",
    "mid",
    "mel_path"
])

df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

print("Done!")
print(f"Saved CSV → {CSV_PATH}")
print(f"Saved mels → {OUTPUT_MELS}")