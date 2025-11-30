import os
import json
import librosa
import numpy as np
import pandas as pd
import cv2

# ------------------------------------------------------------
# 1. Meta-genre mapping (unchanged)
# ------------------------------------------------------------
META_MAP = {
    "Folk&Blues": "Blues-like",
    "Rock&Metal": "Rock/Metal",
    "Rap&Hiphop": "HipHop/Rap",
    "Dance": "Pop/Dance/RnB",
    "RnB&Soul": "Pop/Dance/RnB",
}

KEEP_META = set(META_MAP.values())

# ------------------------------------------------------------
# 2. OPTIONAL: Instrument selection
# ------------------------------------------------------------
# KEEP_INSTRUMENTS = None  # Keep all instruments
KEEP_INSTRUMENTS = ["Piano", "Guitar", "Bass", "Strings"]  # Recommended option

# ------------------------------------------------------------
# 3. Paths
# ------------------------------------------------------------
ROOT = r"C:\Users\anjen\Desktop\project\anjenn\genre_classification\k-music\Training"

SRC_DIR = os.path.join(ROOT, "01.원천데이터")
LBL_DIR = os.path.join(ROOT, "02.라벨링데이터")

OUTPUT_MELS = os.path.join(ROOT, "processed_mels")
os.makedirs(OUTPUT_MELS, exist_ok=True)

CSV_PATH = os.path.join(ROOT, "labels.csv")

# ------------------------------------------------------------
# 4. Helpers
# ------------------------------------------------------------
def extract_loop_index(filename):
    """Extracts digits like 0092224 from filename."""
    parts = filename.split("_")
    for p in parts:
        if p.isdigit():
            return p
    return None


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_audio_fixed(wav_path, target_sr=22050, target_len_sec=5):
    """Load & force 5-second audio."""
    y, sr = librosa.load(wav_path, sr=target_sr, mono=True)
    y = librosa.util.fix_length(y, size=target_len_sec * target_sr)
    return y, sr


def make_melspec(y, sr, size=128):
    """128×128 mel-spectrogram."""
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=size, n_fft=2048, hop_length=512
    )
    S = librosa.power_to_db(S, ref=np.max)
    return cv2.resize(S, (size, size))


# ------------------------------------------------------------
# 5. Instrument parser
# ------------------------------------------------------------
def extract_instrument_info(wav_path):
    """
    Extracts:
        instrument      → Piano, Guitar, Bass, ...
        subinstrument   → Bright Acoustic Piano, Voice Aahs, etc.
    """
    folder = os.path.basename(os.path.dirname(wav_path))
    # Example: TS_03.Piano_02.Bright Acoustic Piano

    try:
        _, rest = folder.split(".", 1)           # "Piano_02.Bright Acoustic Piano"
        inst_family, inst_sub = rest.split("_", 1)
        inst_sub = inst_sub.split(".", 1)[1]     # remove "02."
        return inst_family.strip(), inst_sub.strip()
    except:
        return None, None


# ------------------------------------------------------------
# 6. Scan folders
# ------------------------------------------------------------
def collect_files(root, extensions):
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
# 7. Match WAV ↔ JSON
# ------------------------------------------------------------
matched = []

for idx, wavs in wav_files.items():
    if idx not in label_files:
        print(f"[WARN] No label found for loopIndex {idx}")
        continue

    json_path = None
    mid_path = None

    for fp in label_files[idx]:
        if fp.lower().endswith(".json"):
            json_path = fp
        elif fp.lower().endswith(".mid"):
            mid_path = fp

    if json_path is None:
        print(f"[WARN] No JSON for {idx}; skipping.")
        continue

    for wav_path in wavs:
        matched.append((idx, wav_path, json_path, mid_path))


print(f"Matched samples: {len(matched)}")


# ------------------------------------------------------------
# 8. Process samples
# ------------------------------------------------------------
records = []

for idx, wav_path, json_path, mid_path in matched:
    try:
        meta = load_json(json_path)
        raw_genre = meta["dataSet"]["loopInfo"]["genre"]

        # --- META-GENRE FILTER ---
        if raw_genre not in META_MAP:
            print(f"[SKIP] {idx}: genre {raw_genre} not in target list")
            continue

        meta_genre = META_MAP[raw_genre]

        # --- INSTRUMENT EXTRACTION ---
        instrument, subinstrument = extract_instrument_info(wav_path)

        # --- INSTRUMENT FILTER (if enabled) ---
        if KEEP_INSTRUMENTS is not None:
            if instrument not in KEEP_INSTRUMENTS:
                print(f"[SKIP] {idx}: instrument {instrument} not allowed")
                continue

        # --- Mel-spectrogram ---
        y, sr = load_audio_fixed(wav_path)
        mel = make_melspec(y, sr)

        # --- Save .npy ---
        out_path = os.path.join(OUTPUT_MELS, f"{idx}.npy")
        np.save(out_path, mel)

        # --- Record ---
        records.append([
            idx,
            raw_genre,
            meta_genre,
            instrument,
            subinstrument,
            wav_path,
            json_path,
            mid_path,
            out_path
        ])

        print(f"[OK] {idx} → {meta_genre}, {instrument}/{subinstrument}")

    except Exception as e:
        print(f"[ERROR] {idx} | {e}")


# ------------------------------------------------------------
# 9. Save labels.csv
# ------------------------------------------------------------
df = pd.DataFrame(records, columns=[
    "loopIndex",
    "raw_genre",
    "meta_genre",
    "instrument",
    "subinstrument",
    "wav",
    "json",
    "mid",
    "mel_path"
])

df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

print("\nDone!")
print(f"Saved CSV → {CSV_PATH}")
print(f"Saved mels → {OUTPUT_MELS}")
