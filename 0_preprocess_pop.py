import os
import librosa
import numpy as np
import pandas as pd
import cv2

# --------------------------------------------------------------------
# PATHS (EDIT THESE)
# --------------------------------------------------------------------
ROOT = r"C:\Users\anjen\Desktop\project\anjenn\genre_classification\GTZAN"
GENRE_DIR = os.path.join(ROOT, "genres_original")
OUTPUT_MELS = os.path.join(ROOT, "processed_mels")
os.makedirs(OUTPUT_MELS, exist_ok=True)

CSV_PATH = os.path.join(ROOT, "labels.csv")

# --------------------------------------------------------------------
# META-GENRE MAP: ONLY KEEP THESE 6 GENRES
# Everything else is excluded.
# --------------------------------------------------------------------
META_MAP = {
    "blues": "Blues-like",

    "rock": "Rock/Metal",
    "metal": "Rock/Metal",

    "hiphop": "HipHop/Rap",

    "pop": "Pop/Dance/RnB",
    "disco": "Pop/Dance/RnB"
}
# excluded: classical, jazz, country, reggae

# --------------------------------------------------------------------
# AUDIO HELPERS
# --------------------------------------------------------------------
def load_audio_fixed(path, target_sr=22050, target_len_sec=30):
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    y = librosa.util.fix_length(y, size=target_len_sec * target_sr)
    return y, sr


def make_melspec(y, sr, size=128):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_mels=size,
        n_fft=2048,
        hop_length=512
    )
    S = librosa.power_to_db(S, ref=np.max)
    return cv2.resize(S, (size, size))


# --------------------------------------------------------------------
# MAIN PREPROCESSING LOOP
# --------------------------------------------------------------------
records = []

for genre in os.listdir(GENRE_DIR):
    genre_folder = os.path.join(GENRE_DIR, genre)
    if not os.path.isdir(genre_folder):
        continue

    # Skip entire genre folder if not in meta-map
    if genre not in META_MAP:
        print(f"[SKIP GENRE] {genre} (not in META_MAP)")
        continue

    meta_genre = META_MAP[genre]
    print(f"[PROCESS] Genre: {genre} → {meta_genre}")

    for fname in os.listdir(genre_folder):
        if not fname.lower().endswith(".wav"):
            continue

        wav_path = os.path.join(genre_folder, fname)
        file_id = os.path.splitext(fname)[0]

        try:
            # Load & mel
            y, sr = load_audio_fixed(wav_path)
            mel = make_melspec(y, sr)

            # Save mel
            mel_out = os.path.join(OUTPUT_MELS, f"{file_id}.npy")
            np.save(mel_out, mel)

            # Log info
            records.append([
                file_id,
                genre,
                meta_genre,
                wav_path,
                mel_out
            ])

            print(f"[OK] {fname} → {meta_genre}")

        except Exception as e:
            print(f"[ERROR] {fname} | {e}")


# --------------------------------------------------------------------
# SAVE LABELS CSV
# --------------------------------------------------------------------
df = pd.DataFrame(records, columns=[
    "id",
    "raw_genre",
    "meta_genre",
    "wav_path",
    "mel_path"
])

df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

print("\n=== DONE ===")
print(f"Saved CSV → {CSV_PATH}")
print(f"Processed mel files → {OUTPUT_MELS}")
