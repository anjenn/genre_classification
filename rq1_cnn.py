"""
rq1_cnn.py
RQ1 — CNN Explainability on Mel-Spectrograms

Trains a 3-block CNN on GTZAN mel-spectrogram segments.
Outputs:
- Test accuracy, macro F1
- Confusion matrix
- Grad-CAM for correctly classified + misclassified samples
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 32
EPOCHS = 50
LR = 1e-4
PATIENCE = 7  # for early stopping

plt.rcParams["figure.dpi"] = 140
sns.set()

# ============================================================
# 1. CNN ARCHITECTURE (RQ1 spec)
# ============================================================
class GenreCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(128, 256)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 64×64
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 32×32
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 16×16

        x = self.gap(x)                                # (B,128,1,1)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


# ============================================================
# 2. Dataset Loader
# ============================================================
class MelDataset(Dataset):
    """
    Works for both GTZAN and Korean loops.
    Expected columns:
        - meta_genre
        - mel_path
    All other columns are ignored.
    """
    def __init__(self, df: pd.DataFrame, label2idx: dict):
        assert "mel_path" in df.columns, "mel_path column missing"
        assert "meta_genre" in df.columns, "meta_genre column missing"

        self.df = df.reset_index(drop=True)
        self.label2idx = label2idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        mel = np.load(row["mel_path"]).astype("float32")  # (128,128)
        mel = mel[None, :, :]  # (1,H,W)

        label = self.label2idx[row["meta_genre"]]
        return torch.tensor(mel), torch.tensor(label)


def make_loader(df, label2idx, shuffle=False):
    return DataLoader(
        MelDataset(df, label2idx),
        batch_size=BATCH,
        shuffle=shuffle,
        num_workers=4
    )


# ============================================================
# 3. Training / Evaluation
# ============================================================
def train_epoch(model, loader, opt, criterion):
    model.train()
    losses, correct, total = 0, 0, 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()

        losses += loss.item() * x.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return losses / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    losses, correct, total = 0, 0, 0

    all_preds, all_y = [], []

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = criterion(logits, y)

        losses += loss.item() * x.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += x.size(0)

        all_preds.append(preds.cpu().numpy())
        all_y.append(y.cpu().numpy())

    return (
        losses / total,
        correct / total,
        np.concatenate(all_y),
        np.concatenate(all_preds)
    )


# ============================================================
# 4. Grad-CAM
# ============================================================
class GradCAM:
    def __init__(self, model, layer_name="conv3"):
        self.model = model
        self.model.eval()

        modules = dict(model.named_modules())
        self.layer = modules[layer_name]

        self.gradients = None
        self.activations = None

        self.layer.register_forward_hook(self._fwd_hook)
        self.layer.register_backward_hook(self._bwd_hook)

    def _fwd_hook(self, m, inp, out):
        self.activations = out.detach()

    def _bwd_hook(self, m, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, x, class_idx=None):
        self.model.zero_grad()
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(1).item()

        target = logits[:, class_idx]
        target.backward()

        grads = self.gradients              # (B,C,H,W)
        acts = self.activations             # (B,C,H,W)
        weights = grads.mean(dim=(2,3))     # (B,C)

        cam = (acts * weights[:,:,None,None]).sum(1)
        cam = F.relu(cam)

        cam = cam[0].cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-6
        return cam


def save_cam_example(model, gradcam, df, label2idx, class_names, prefix):
    os.makedirs("rq1_gradcam", exist_ok=True)

    for i, (_, row) in enumerate(df.iterrows()):
        if i >= 2: break
        mel = np.load(row["mel_path"]).astype("float32")
        x = torch.tensor(mel[None, None]).to(DEVICE)

        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(1).item()

        cam = gradcam.generate(x, class_idx=pred)

        # Save image
        fig, ax = plt.subplots(1, 2, figsize=(7,3))
        ax[0].imshow(mel, aspect='auto', origin='lower')
        ax[1].imshow(mel, aspect='auto', origin='lower')
        ax[1].imshow(cam, alpha=0.5, cmap='jet', aspect='auto', origin='lower')

        ax[0].set_title(f"GT: {row['meta_genre']}")
        ax[1].set_title(f"Pred: {class_names[pred]}")
        for a in ax: a.set_xticks([]); a.set_yticks([])

        fname = f"rq1_gradcam/{prefix}_{i}.png"
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
        print("[Grad-CAM saved]", fname)


# ============================================================
# 5. Main RQ1 Pipeline
# ============================================================
def validate_csv(df, name):
    required = ["meta_genre", "mel_path"]
    for r in required:
        if r not in df.columns:
            raise ValueError(f"[{name}] Missing required column: {r}")
    print(f"[{name}] CSV OK — {len(df)} samples")

def main():
    ROOT = r"C:\Users\anjen\Desktop\project\anjenn\genre_classification\rq1_combined"

    csv_train = os.path.join(ROOT, "gtzan", "mel", "gtzan_mel_train.csv")
    csv_val   = os.path.join(ROOT, "gtzan", "mel", "gtzan_mel_val.csv")
    csv_test  = os.path.join(ROOT, "gtzan", "mel", "gtzan_mel_test.csv")

    df_train = pd.read_csv(csv_train)
    df_val   = pd.read_csv(csv_val)
    df_test  = pd.read_csv(csv_test)

    validate_csv(df_train, "Train")
    validate_csv(df_val, "Val")
    validate_csv(df_test, "Test")

    class_names = sorted(df_train["meta_genre"].unique())
    label2idx = {c:i for i,c in enumerate(class_names)}

    train_loader = make_loader(df_train, label2idx, shuffle=True)
    val_loader   = make_loader(df_val,   label2idx)
    test_loader  = make_loader(df_test,  label2idx)

    # --------------------
    # Model + Training
    # --------------------
    model = GenreCNN(len(class_names)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val = 0
    patience = PATIENCE

    for epoch in range(1, EPOCHS+1):
        tr_loss, tr_acc = train_epoch(model, train_loader, opt, criterion)
        val_loss, val_acc, _, _ = eval_epoch(model, val_loader, criterion)

        print(f"[Epoch {epoch:03d}] "
              f"train_acc={tr_acc:.3f} val_acc={val_acc:.3f}")

        if val_acc > best_val:
            best_val = val_acc
            patience = PATIENCE
            torch.save(model.state_dict(), "rq1_cnn_best.pth")
        else:
            patience -= 1
            if patience == 0:
                print("[Early Stop]")
                break

    model.load_state_dict(torch.load("rq1_cnn_best.pth"))

    # --------------------
    # Test Evaluation
    # --------------------
    _, _, y_true, y_pred = eval_epoch(model, test_loader, criterion)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print("\n=== RQ1 CNN Test Results ===")
    print("Accuracy :", acc)
    print("Macro F1 :", macro_f1)
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(7,6))
    sns.heatmap(cm_n, annot=True, fmt=".2f",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("RQ1 CNN – Normalized Confusion Matrix")
    plt.savefig("rq1_cnn_cm.png")
    plt.close()

    # --------------------
    # Grad-CAM Analysis
    # --------------------
    gradcam = GradCAM(model, "conv3")

    # Correctly classified examples
    correct_df = df_test[(y_true == y_pred)]
    save_cam_example(model, gradcam, correct_df, label2idx, class_names,
                     prefix="correct")

    # Misclassified examples
    wrong_df = df_test[(y_true != y_pred)]
    save_cam_example(model, gradcam, wrong_df, label2idx, class_names,
                     prefix="wrong")

    print("\n[Done] RQ1 CNN complete.")


if __name__ == "__main__":
    main()
