import pandas as pd
from tqdm import tqdm
import numpy as np
import os
import librosa

SR = 22050
N_FFT = 2048
HOP = 512
N_MFCC = 20
TARGET_LEN_SEC = 5

# MFCC mean (20)
# MFCC std (20)
# Spectral centroid mean/std (2)
# Spectral bandwidth mean/std (2)
# Spectral rolloff mean/std (2)
# Zero-crossing mean/std (2)
# RMS mean/std (2)
# Chroma mean (12)
# Chroma std (12)
# Tempo (1)
# --------------------------------
# Total = 20+20+2+2+2+2+2+12+12+1 = 75 dims

# Feature Extraction 

def extract_features(wav_path):
    # Load audio
    y, _ = librosa.load(wav_path, sr=SR, mono=True)
    y = librosa.util.fix_length(y, size=SR * TARGET_LEN_SEC)

    # ---------------------
    # MFCC (20 dims)
    # ---------------------
    mfcc = librosa.feature.mfcc(
        y=y, sr=SR, n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=HOP
    )
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    # ---------------------
    # Spectral features
    # ---------------------
    centroid = librosa.feature.spectral_centroid(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)[0]

    feat_cent_mean, feat_cent_std = centroid.mean(), centroid.std()
    feat_band_mean, feat_band_std = bandwidth.mean(), bandwidth.std()
    feat_roll_mean, feat_roll_std = rolloff.mean(), rolloff.std()
    feat_zcr_mean, feat_zcr_std  = zcr.mean(), zcr.std()
    feat_rms_mean, feat_rms_std  = rms.mean(), rms.std()

    # ---------------------
    # Chroma (12)
    # ---------------------
    chroma = librosa.feature.chroma_stft(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP
    )
    chroma_mean = chroma.mean(axis=1)
    chroma_std = chroma.std(axis=1)

    # ---------------------
    # Tempo (scalar)
    # ---------------------
    tempo, _ = librosa.beat.beat_track(y=y, sr=SR)
    tempo = float(tempo)

    # Concatenate all features
    features = np.concatenate([
        mfcc_mean, mfcc_std,           # 40
        [feat_cent_mean, feat_cent_std],
        [feat_band_mean, feat_band_std],
        [feat_roll_mean, feat_roll_std],
        [feat_zcr_mean, feat_zcr_std],
        [feat_rms_mean, feat_rms_std],
        chroma_mean, chroma_std,       # 24
        [tempo],                       # 1
    ])

    return features

# Apply Features

def extract_feature_csv(label_csv, out_csv, out_npy_dir):
    df = pd.read_csv(label_csv)

    os.makedirs(out_npy_dir, exist_ok=True)

    feature_vectors = []
    feature_paths = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        wav_path = row["wav"] if "wav" in row else row["wav_path"]

        feats = extract_features(wav_path)
        feats = feats.astype(np.float32)

        # Save feature vector
        out_path = os.path.join(out_npy_dir, f"{row['loopIndex'] if 'loopIndex' in row else row['id']}.npy")
        np.save(out_path, feats)

        feature_vectors.append(feats)
        feature_paths.append(out_path)

    df["feat_path"] = feature_paths
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("Saved:", out_csv)

# Usage for korean
extract_feature_csv(
    label_csv=r"C:\Users\anjen\Desktop\project\anjenn\genre_classification\k-music\Training\labels.csv",
    out_csv=r"C:\Users\anjen\Desktop\project\anjenn\genre_classification\k-music\Training\mfcc_features.csv",
    out_npy_dir=r"C:\Users\anjen\Desktop\project\anjenn\genre_classification\k-music\Training\mfcc_npy"
)

# Usage for GTZAN:
extract_feature_csv(
    label_csv=r"C:\Users\anjen\Desktop\project\anjenn\genre_classification\GTZAN\labels.csv",
    out_csv=r"C:\Users\anjen\Desktop\project\anjenn\genre_classification\GTZAN\mfcc_features.csv",
    out_npy_dir=r"C:\Users\anjen\Desktop\project\anjenn\genre_classification\GTZAN\mfcc_npy"
)

# 📦 3. Combined Dataset MFCC Extraction
df_k = pd.read_csv(
    r"C:\Users\anjen\Desktop\project\anjenn\genre_classification\k-music\Training\mfcc_features.csv"
)

df_g = pd.read_csv(
    r"C:\Users\anjen\Desktop\project\anjenn\genre_classification\GTZAN\mfcc_features.csv"
)

df_combined = pd.concat([df_k, df_g], ignore_index=True)

df_combined.to_csv(
    r"C:\Users\anjen\Desktop\project\anjenn\genre_classification\combined_mfcc_features.csv",
    index=False,
    encoding="utf-8-sig"
)

## Both datasets (Korean + GTZAN):

# Same sample rate (22050 Hz)

# Same FFT & hop length (2048 / 512)

# Same 5-second padded/chopped length

# Same MFCC, chroma, spectral feature settings

# Same feature dimensionality (~75)

# One consistent feat_path per file

# Works with your meta-genre + instrument-enhanced CSVs

# Ready for classical ML (RF/XGB/SVM/MLP) and SHAP explainability