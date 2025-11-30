"""
cnn_cross_cultural_zero_shot.py

Cross-Cultural Zero-Shot Evaluation for CNN (RQ3, E3/E5)

- Train a CNN on GTZAN mel-spectrogram segments (meta-genre labels)
- Evaluate on:
    1) GTZAN test (source)
    2) Korean test (zero-shot cross-cultural)
- Compute accuracy / macro-F1 and report cross-cultural drop
- Run Grad-CAM on a few GTZAN vs Korean examples to compare focus

Assumes:
C:/Users/anjen/Desktop/project/anjenn/genre_classification/rq1_combined/
    gtzan/mel/gtzan_mel_train.csv
    gtzan/mel/gtzan_mel_val.csv
    gtzan/mel/gtzan_mel_test.csv

    korean/mel/korean_mel_test.csv

Each mel CSV must contain:
    mel_path   -> .npy file with shape (n_mels, time)
    meta_genre -> label string in the same space as GTZAN meta-genres
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, f1_score, classification_report
from typing import Dict, List

from cnn_model import GenreCNN  # Your CNN model definition

sns.set()
plt.rcParams["figure.dpi"] = 140

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
RANDOM_STATE = 42


# =========================================================
# Dataset + Loader
# =========================================================
class MelSpecDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label2idx: Dict[str, int]):
        self.df = df.reset_index(drop=True)
        self.label2idx = label2idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mel_path = row["mel_path"]
        mel = np.load(mel_path).astype("float32")  # (n_mels, time)
        mel = mel[None, :, :]  # (1, H, W)

        label_str = row["meta_genre"]
        label_idx = self.label2idx[label_str]

        x = torch.tensor(mel)           # (1, H, W)
        y = torch.tensor(label_idx)     # scalar
        return x, y


def make_loader(df: pd.DataFrame, label2idx: Dict[str, int], batch_size=BATCH_SIZE, shuffle=False):
    ds = MelSpecDataset(df, label2idx)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# =========================================================
# Training / Evaluation
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / total, total_correct / total


@torch.no_grad()
def eval_model(model, loader, class_names: List[str], title: str = ""):
    model.eval()
    all_preds = []
    all_targets = []

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(y.cpu().numpy().tolist())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")

    print(f"\n===== {title} =====")
    print("Accuracy   :", f"{acc:.4f}")
    print("Macro F1   :", f"{macro_f1:.4f}")
    print(classification_report(
        all_targets,
        all_preds,
        target_names=class_names
    ))

    return acc, macro_f1, all_targets, all_preds


# =========================================================
# Grad-CAM
# =========================================================
class GradCAM:
    def __init__(self, model: nn.Module, target_layer_name: str):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        # Find the layer object from its name
        module_dict = dict(self.model.named_modules())
        assert target_layer_name in module_dict, f"{target_layer_name} not found in model."
        self.target_layer = module_dict[target_layer_name]

        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, x: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        x: (1, 1, H, W) tensor
        returns: (H, W) numpy CAM normalized 0–1
        """
        self.model.zero_grad()
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        score = logits[:, class_idx]
        score.backward(retain_graph=True)

        grads = self.gradients       # (B, C, H, W)
        acts = self.activations      # (B, C, H, W)

        weights = grads.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = (weights * acts).sum(dim=1)               # (B, H, W)
        cam = F.relu(cam)
        cam = cam[0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return cam


def save_gradcam_example(model, df: pd.DataFrame, label2idx: Dict[str, int],
                         class_names: List[str], out_prefix: str,
                         target_layer_name: str = "conv3", num_examples: int = 3):
    """
    Take a few rows from df, run Grad-CAM, and save overlay images.
    """
    gradcam = GradCAM(model, target_layer_name=target_layer_name)

    for i in range(min(num_examples, len(df))):
        row = df.iloc[i]
        mel = np.load(row["mel_path"]).astype("float32")  # (H, W)
        mel_tensor = torch.tensor(mel[None, None, :, :]).to(DEVICE)  # (1,1,H,W)

        with torch.no_grad():
            logits = model(mel_tensor)
            pred_idx = logits.argmax(dim=1).item()

        cam = gradcam.generate(mel_tensor, class_idx=pred_idx)

        true_label = row["meta_genre"]
        pred_label = class_names[pred_idx]

        # Plot original mel + CAM overlay
        fig, ax = plt.subplots(1, 2, figsize=(8, 3))

        ax[0].imshow(mel, origin="lower", aspect="auto")
        ax[0].set_title(f"Mel – True: {true_label}")

        ax[1].imshow(mel, origin="lower", aspect="auto")
        ax[1].imshow(cam, origin="lower", aspect="auto", alpha=0.5)
        ax[1].set_title(f"Grad-CAM – Pred: {pred_label}")

        for a in ax:
            a.set_xticks([])
            a.set_yticks([])

        plt.tight_layout()
        fname = f"{out_prefix}_example_{i}.png"
        plt.savefig(fname)
        plt.close()
        print(f"[Grad-CAM] Saved: {fname}")


# =========================================================
# Main zero-shot pipeline
# =========================================================
def main():
    ROOT = r"C:\Users\anjen\Desktop\project\anjenn\genre_classification\rq1_combined"

    # ---------- Load GTZAN mel splits ----------
    gtzan_mel_dir = os.path.join(ROOT, "gtzan", "mel")
    g_train = pd.read_csv(os.path.join(gtzan_mel_dir, "gtzan_mel_train.csv"))
    g_val   = pd.read_csv(os.path.join(gtzan_mel_dir, "gtzan_mel_val.csv"))
    g_test  = pd.read_csv(os.path.join(gtzan_mel_dir, "gtzan_mel_test.csv"))

    # Label vocabulary from GTZAN train
    class_names = sorted(g_train["meta_genre"].unique().tolist())
    label2idx = {g: i for i, g in enumerate(class_names)}
    print("[CLASSES] GTZAN meta-genres:", class_names)

    # ---------- Load Korean mel test ----------
    korean_mel_dir = os.path.join(ROOT, "korean", "mel")
    k_test = pd.read_csv(os.path.join(korean_mel_dir, "korean_mel_test.csv"))

    # Check that Korean labels are subset of / equal to GTZAN mapping
    korean_labels = sorted(k_test["meta_genre"].unique().tolist())
    print("[INFO] Korean meta-genres (test):", korean_labels)
    missing = [c for c in korean_labels if c not in label2idx]
    if missing:
        raise ValueError(f"Some Korean labels not in GTZAN label2idx: {missing}")

    # ---------- DataLoaders ----------
    train_loader = make_loader(g_train, label2idx, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = make_loader(g_val,   label2idx, batch_size=BATCH_SIZE, shuffle=False)
    gtzan_test_loader  = make_loader(g_test,  label2idx, batch_size=BATCH_SIZE, shuffle=False)
    korean_test_loader = make_loader(k_test,  label2idx, batch_size=BATCH_SIZE, shuffle=False)

    # ---------- Model ----------
    num_classes = len(class_names)
    model = GenreCNN(n_classes=num_classes).to(DEVICE)

    # Either load pre-trained weights OR train here.
    # Recommended: pre-train with a separate script `train_cnn.py`
    # and save to a path like below.
    weights_path = os.path.join(ROOT, "gtzan", "cnn", "gtzan_cnn_best.pth")

    if os.path.exists(weights_path):
        print(f"[LOAD] Loading pre-trained weights from {weights_path}")
        state = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(state)
    else:
        print("[WARN] No pre-trained weights found. Training CNN from scratch on GTZAN.")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        best_val_acc = 0.0
        for epoch in range(1, 31):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            val_acc, val_f1, _, _ = eval_model(model, val_loader, class_names, title=f"Val Epoch {epoch}")

            print(f"[Epoch {epoch:02d}] train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                os.makedirs(os.path.dirname(weights_path), exist_ok=True)
                torch.save(model.state_dict(), weights_path)
                print(f"[SAVE] Best CNN weights -> {weights_path}")

        print(f"[INFO] Finished training. Best val_acc={best_val_acc:.3f}")

    # If we just trained, ensure best weights are loaded
    if not model.training and os.path.exists(weights_path):
        # Already loaded earlier
        pass
    else:
        # Ensure we load best checkpoint after training
        if os.path.exists(weights_path):
            state = torch.load(weights_path, map_location=DEVICE)
            model.load_state_dict(state)

    # ---------- Evaluation: GTZAN test (source) ----------
    src_acc, src_f1, _, _ = eval_model(model, gtzan_test_loader, class_names,
                                       title="CNN – GTZAN test (source)")

    # ---------- Evaluation: Korean test (zero-shot) ----------
    tgt_acc, tgt_f1, _, _ = eval_model(model, korean_test_loader, class_names,
                                       title="CNN – Korean test (zero-shot)")

    print("\n===== Cross-Cultural Performance Drop (CNN, GTZAN → Korean) =====")
    print(f"ΔAccuracy = {src_acc - tgt_acc:.4f}")
    print(f"ΔMacro F1 = {src_f1  - tgt_f1:.4f}")

    # ---------- Grad-CAM Visualizations ----------
    # A few examples from GTZAN test (source)
    print("\n[Grad-CAM] GTZAN examples...")
    save_gradcam_example(
        model, g_test.sample(n=min(3, len(g_test)), random_state=RANDOM_STATE),
        label2idx, class_names,
        out_prefix="cnn_gradcam_gtzan"
    )

    # A few examples from Korean test (target)
    print("\n[Grad-CAM] Korean examples...")
    save_gradcam_example(
        model, k_test.sample(n=min(3, len(k_test)), random_state=RANDOM_STATE),
        label2idx, class_names,
        out_prefix="cnn_gradcam_korean"
    )

    print("\n[DONE] CNN cross-cultural zero-shot evaluation complete.")


if __name__ == "__main__":
    main()
