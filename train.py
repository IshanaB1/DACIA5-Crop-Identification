"""
=======================================================================
DACIA5 – Subtask 2 / Challenge 1: Crop Identification (Past vs Present)
=======================================================================
Dataset  : Sentinel-2 patches (32×32×12 bands), years 2020-2023 = train,
           2024 = test.
Classes  : Wheat(0) Corn(1) Peas(2) Rapeseed(3) Potato(4) Sugarbeet(5) Alfalfa(6)
Metric   : Q1 = 0.5×AA + 0.5×OA  (target > 80)
Model    : SpectralSpatial CNN + Channel Attention (SE) + multi-scale
Outputs  : best_model.pth | confusion_matrix.png | per_class_accuracy.png
=======================================================================
pip install torch torchvision rasterio numpy scikit-learn matplotlib seaborn tqdm scipy
=======================================================================
"""

import os, random, glob, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import scipy.io as sio

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("[WARN] rasterio not found – set USE_MAT=True to use .mat files")

warnings.filterwarnings("ignore")

# ──────────────────────────────── CONFIG ───────────────────────────────────
CFG = dict(
    # Data
    data_root       = "./dacia5_data",   # root folder of unzipped DACIA5
    use_mat         = False,             # True → load .mat, False → load .tif
    patch_size      = 32,
    num_bands       = 12,                # Sentinel-2 bands
    num_classes     = 7,
    # Class mapping (merge original DACIA5 labels → 7 challenge classes)
    # Original label names vary by dataset file; adjust key strings if needed.
    label_map       = {
        "winter wheat": 0, "spring wheat": 0,   # → Wheat
        "corn": 1,         "corn silage": 1,     # → Corn
        "peas": 2,                               # → Peas
        "winter rapeseed": 3,                    # → Rapeseed
        "late potato": 4,  "other potato": 4,   # → Potato
        "sugar beet": 5,                         # → Sugarbeet
        "alfalfa": 6,                            # → Alfalfa
        # soybean is NOT in test → exclude from training too (set to -1)
        "soybean": -1,
    },
    train_years     = [2020, 2021, 2022, 2023],
    test_years      = [2024],
    val_split       = 0.15,              # fraction of train years used for val
    # Model
    base_ch         = 64,
    se_reduction    = 8,
    dropout         = 0.3,
    # Training
    epochs          = 80,
    batch_size      = 128,
    lr              = 3e-4,
    weight_decay    = 1e-4,
    label_smoothing = 0.05,
    mixup_alpha     = 0.3,
    warmup_pct      = 0.05,
    grad_clip       = 1.0,
    # Misc
    seed            = 42,
    num_workers     = 4,
    save_dir        = "checkpoints",
    plot_dir        = "plots",
)

CLASS_NAMES = ["Wheat", "Corn", "Peas", "Rapeseed", "Potato", "Sugarbeet", "Alfalfa"]

os.makedirs(CFG["save_dir"], exist_ok=True)
os.makedirs(CFG["plot_dir"], exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {DEVICE}")


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True

set_seed(CFG["seed"])


# ──────────────────────────────── DATASET ──────────────────────────────────
def load_mat_patch(path):
    """Load a 32×32×12 patch from .mat file → numpy (12, 32, 32)."""
    mat = sio.loadmat(path)
    # key is usually 'patch' or the filename stem – try common keys
    for key in ["patch", "data", "x", "X"]:
        if key in mat:
            arr = mat[key].astype(np.float32)
            break
    else:
        # fallback: pick first non-meta key
        arr = list(v for k, v in mat.items()
                   if not k.startswith("__"))[0].astype(np.float32)
    # ensure shape (C, H, W)
    if arr.ndim == 3:
        if arr.shape[2] == 12:           # (H, W, C)
            arr = arr.transpose(2, 0, 1)
    elif arr.ndim == 2:
        arr = arr[np.newaxis]            # (1, H, W)
    return arr                           # (12, 32, 32)


def load_tif_patch(path):
    """Load a GeoTIFF patch → numpy (C, 32, 32)."""
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)   # (C, H, W)
    return arr


def discover_patches(data_root, years, label_map, use_mat=False):
    """
    Walk data_root to find all patches for the given years.
    Expected directory layout (adapt to actual DACIA5 structure):

        data_root/
          patches/
            optical/
              2020/
                <crop_label>/
                  patch_0001.mat  (or .tif)

    Returns list of (file_path, int_label) tuples.
    """
    ext = ".mat" if use_mat else ".tif"
    samples = []
    optical_root = os.path.join(data_root, "patches", "optical")

    if not os.path.isdir(optical_root):
        # Fallback: search anywhere under data_root
        optical_root = data_root
        print(f"[WARN] Expected 'patches/optical' folder not found; "
              f"scanning {data_root} recursively.")

    for year in years:
        year_dir = os.path.join(optical_root, str(year))
        if not os.path.isdir(year_dir):
            print(f"[WARN] Year folder not found: {year_dir}")
            continue
        # Each sub-folder is a crop label
        for crop_dir in os.scandir(year_dir):
            if not crop_dir.is_dir():
                continue
            crop_name = crop_dir.name.lower().strip()
            label = label_map.get(crop_name, None)
            if label is None:
                print(f"[WARN] Unknown crop label '{crop_name}' – skipping")
                continue
            if label == -1:              # soybean excluded
                continue
            for fpath in glob.glob(os.path.join(crop_dir.path, f"*{ext}")):
                samples.append((fpath, label))

    print(f"[INFO] Found {len(samples)} patches for years {years}")
    return samples


class DACIA5Dataset(Dataset):
    """
    Sentinel-2 patch dataset for DACIA5 crop identification.

    Applies per-channel normalization using pre-computed stats (or
    estimates them from the provided scaler).
    """

    def __init__(self, samples, scaler=None, augment=False, use_mat=False):
        self.samples  = samples
        self.scaler   = scaler
        self.augment  = augment
        self.use_mat  = use_mat

    def __len__(self):
        return len(self.samples)

    def _load(self, path):
        if self.use_mat:
            return load_mat_patch(path)
        return load_tif_patch(path)

    def _normalize(self, x):
        """
        Percentile clipping + standardization.
        x: (C, H, W)
        """
        # Sentinel-2 reflectance is typically in [0, 10000] (DN)
        x = np.clip(x, 0, 10000) / 10000.0   # → [0, 1]
        if self.scaler is not None:
            C, H, W = x.shape
            flat = x.reshape(C, -1).T         # (H*W, C)
            flat = self.scaler.transform(flat)
            x = flat.T.reshape(C, H, W)
        return x

    def _augment(self, x):
        """Simple spatial augmentation for (C, H, W) numpy array."""
        # Random horizontal flip
        if random.random() > 0.5:
            x = x[:, :, ::-1].copy()
        # Random vertical flip
        if random.random() > 0.5:
            x = x[:, ::-1, :].copy()
        # Random 90° rotation
        k = random.randint(0, 3)
        if k:
            x = np.rot90(x, k, axes=(1, 2)).copy()
        # Additive Gaussian noise
        if random.random() > 0.5:
            x = x + np.random.randn(*x.shape).astype(np.float32) * 0.01
        return x

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        x = self._load(path)          # (C, H, W)
        x = self._normalize(x)
        if self.augment:
            x = self._augment(x)
        x = torch.from_numpy(x.astype(np.float32))
        return x, label


def compute_scaler(samples, use_mat=False, max_samples=3000):
    """
    Fit a per-channel StandardScaler on a subset of training patches.
    """
    print("[INFO] Fitting channel-wise scaler …")
    subset = random.sample(samples, min(max_samples, len(samples)))
    pixels = []
    for path, _ in tqdm(subset, desc="Scaler fit"):
        if use_mat:
            x = load_mat_patch(path)
        else:
            x = load_tif_patch(path)
        x = np.clip(x, 0, 10000) / 10000.0    # (C, H, W)
        C, H, W = x.shape
        pixels.append(x.reshape(C, -1).T)     # (H*W, C)
    pixels = np.concatenate(pixels, axis=0)   # (N_pixels, C)
    scaler = StandardScaler().fit(pixels)
    return scaler


def build_weighted_sampler(samples):
    labels = [s[1] for s in samples]
    class_counts = np.bincount(labels, minlength=CFG["num_classes"])
    weights = 1.0 / (class_counts[labels] + 1e-6)
    return WeightedRandomSampler(torch.DoubleTensor(weights), len(weights))


# ──────────────────────────────── MODEL ────────────────────────────────────
class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, ch // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch // reduction, ch, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        s = self.se(x).view(x.size(0), -1, 1, 1)
        return x * s


class ConvBnRelu(nn.Sequential):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, groups=1):
        super().__init__(
            nn.Conv2d(in_c, out_c, k, s, p, groups=groups, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )


class ResBlock(nn.Module):
    def __init__(self, ch, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            ConvBnRelu(ch, ch),
            nn.Dropout2d(dropout),
            ConvBnRelu(ch, ch),
            SEBlock(ch),
        )

    def forward(self, x):
        return x + self.block(x)


class MultiScaleEncoder(nn.Module):
    """
    Multi-scale spatial feature extractor for (C=12, 32, 32) patches.
    Branch 1: 3×3 convolutions (local texture)
    Branch 2: 5×5 convolutions (medium-range patterns)
    Branch 3: spectral mixing (1×1, captures cross-band relationships)
    """
    def __init__(self, in_ch=12, base=64, dropout=0.3):
        super().__init__()
        half = base // 2

        # Branch 1 – 3×3
        self.b1 = nn.Sequential(
            ConvBnRelu(in_ch, base, 3, 1, 1),
            ResBlock(base, dropout),
            nn.MaxPool2d(2),              # 16×16
            ConvBnRelu(base, base * 2, 3, 1, 1),
            ResBlock(base * 2, dropout),
            nn.MaxPool2d(2),              # 8×8
            ConvBnRelu(base * 2, base * 2, 3, 1, 1),
            ResBlock(base * 2, dropout),
            nn.AdaptiveAvgPool2d(1),      # (B, base*2, 1, 1)
        )

        # Branch 2 – 5×5
        self.b2 = nn.Sequential(
            ConvBnRelu(in_ch, half, 5, 1, 2),
            ResBlock(half, dropout),
            nn.MaxPool2d(2),
            ConvBnRelu(half, base, 5, 1, 2),
            ResBlock(base, dropout),
            nn.AdaptiveAvgPool2d(1),
        )

        # Branch 3 – spectral (1×1)
        self.b3 = nn.Sequential(
            ConvBnRelu(in_ch, half, 1, 1, 0),
            ConvBnRelu(half, half, 1, 1, 0),
            nn.AdaptiveAvgPool2d(1),
        )

        self.out_dim = base * 2 + base + half   # concat all branches

    def forward(self, x):
        f1 = self.b1(x).flatten(1)   # (B, base*2)
        f2 = self.b2(x).flatten(1)   # (B, base)
        f3 = self.b3(x).flatten(1)   # (B, half)
        return torch.cat([f1, f2, f3], dim=1)  # (B, out_dim)


class DACIA5Classifier(nn.Module):
    """
    Full classifier:
      MultiScaleEncoder → dropout → FC head (BN + residual)
    """
    def __init__(self, in_ch=12, base=64, num_classes=7, dropout=0.3):
        super().__init__()
        self.encoder  = MultiScaleEncoder(in_ch, base, dropout)
        feat_dim      = self.encoder.out_dim

        self.head = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        feat = self.encoder(x)
        return self.head(feat)


# ─────────────────────────────── LOSSES ────────────────────────────────────
class LabelSmoothCE(nn.Module):
    def __init__(self, smoothing=0.05, num_classes=7):
        super().__init__()
        self.s = smoothing
        self.K = num_classes

    def forward(self, logits, targets):
        log_p = F.log_softmax(logits, dim=1)
        # one-hot
        with torch.no_grad():
            smooth = torch.full_like(log_p, self.s / (self.K - 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1 - self.s)
        return -(smooth * log_p).sum(dim=1).mean()


def mixup_batch(x, y, alpha=0.3):
    """Returns mixed inputs + two targets + lambda."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


# ─────────────────────────────── METRICS ───────────────────────────────────
def compute_q1(preds, labels, num_classes=7):
    """Q1 = 0.5 × AA + 0.5 × OA"""
    preds  = np.array(preds)
    labels = np.array(labels)
    OA = (preds == labels).mean() * 100
    per_class_acc = []
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        per_class_acc.append((preds[mask] == labels[mask]).mean() * 100)
    AA = np.mean(per_class_acc)
    Q1 = 0.5 * AA + 0.5 * OA
    return Q1, AA, OA, per_class_acc


# ─────────────────────────────── PLOTS ─────────────────────────────────────
def plot_confusion_matrix(labels, preds, save_path):
    cm = confusion_matrix(labels, preds, labels=list(range(CFG["num_classes"])))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.patch.set_facecolor("#0d1117")

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Confusion Matrix (counts)", "Confusion Matrix (normalized)"],
        ["d", ".2f"],
    ):
        ax.set_facecolor("#161b22")
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="YlOrRd",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=ax, linewidths=0.5, linecolor="#30363d",
            annot_kws={"size": 11, "weight": "bold"},
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted", color="#8b949e", fontsize=12)
        ax.set_ylabel("True", color="#8b949e", fontsize=12)
        ax.tick_params(colors="white", labelsize=10)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    fig.suptitle("DACIA5 – Crop Identification: Past vs Present",
                 color="white", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[INFO] Saved: {save_path}")


def plot_per_class_accuracy(per_class_acc, oa, aa, q1, save_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(CLASS_NAMES)))
    bars = ax.bar(CLASS_NAMES, per_class_acc, color=colors,
                  edgecolor="#30363d", linewidth=0.8, zorder=3)

    # Value labels on bars
    for bar, val in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8, f"{val:.1f}%",
                ha="center", va="bottom", color="white",
                fontsize=11, fontweight="bold")

    # Reference lines
    for val, label, ls in [(80, "80% target", "--"), (aa, f"AA={aa:.1f}%", "-.")]:
        ax.axhline(val, color="#58a6ff", linewidth=1.5, linestyle=ls,
                   label=label, zorder=2)

    ax.set_ylim(0, 105)
    ax.set_xlabel("Crop Class", color="#8b949e", fontsize=12)
    ax.set_ylabel("Accuracy (%)", color="#8b949e", fontsize=12)
    ax.tick_params(colors="white", labelsize=11)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.yaxis.grid(True, color="#21262d", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    legend = ax.legend(loc="lower right", facecolor="#161b22",
                       edgecolor="#30363d", labelcolor="white", fontsize=10)

    title = (f"Per-Class Accuracy — DACIA5 Crop ID\n"
             f"OA={oa:.1f}%  |  AA={aa:.1f}%  |  Q1={q1:.1f}")
    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[INFO] Saved: {save_path}")


# ─────────────────────────────── TRAIN LOOP ────────────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler, criterion, scaler_amp):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for x, y in tqdm(loader, desc="  Train", leave=False, ncols=80):
        x, y = x.to(DEVICE), y.to(DEVICE)

        # Mixup
        x_mix, ya, yb, lam = mixup_batch(x, y, CFG["mixup_alpha"])

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler_amp is not None):
            logits = model(x_mix)
            loss   = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)

        if scaler_amp:
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item() * x.size(0)
        preds       = logits.argmax(1)
        correct    += (preds == ya).sum().item()
        total      += x.size(0)

    return total_loss / total, correct / total * 100


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for x, y in tqdm(loader, desc="  Val  ", leave=False, ncols=80):
        x = x.to(DEVICE)
        preds = model(x).argmax(1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(y.numpy().tolist())
    return all_preds, all_labels


# ─────────────────────────────── MAIN ──────────────────────────────────────
def main():
    print("=" * 65)
    print("  DACIA5 – Crop Identification: Past vs Present")
    print("=" * 65)

    # ── 1. Discover patches ────────────────────────────────────────────────
    all_train = discover_patches(
        CFG["data_root"], CFG["train_years"], CFG["label_map"], CFG["use_mat"]
    )
    all_test  = discover_patches(
        CFG["data_root"], CFG["test_years"],  CFG["label_map"], CFG["use_mat"]
    )

    if len(all_train) == 0:
        raise RuntimeError(
            f"No training patches found under '{CFG['data_root']}'.\n"
            "Please set CFG['data_root'] to the root of the unzipped DACIA5 dataset.\n"
            "Expected layout:  data_root/patches/optical/<year>/<crop_label>/*.tif"
        )

    # ── 2. Train / Val split ───────────────────────────────────────────────
    random.shuffle(all_train)
    n_val = int(len(all_train) * CFG["val_split"])
    val_samples   = all_train[:n_val]
    train_samples = all_train[n_val:]
    print(f"[INFO] Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(all_test)}")

    # ── 3. Scaler ──────────────────────────────────────────────────────────
    scaler = compute_scaler(train_samples, CFG["use_mat"])

    # ── 4. DataLoaders ────────────────────────────────────────────────────
    train_ds = DACIA5Dataset(train_samples, scaler, augment=True,  use_mat=CFG["use_mat"])
    val_ds   = DACIA5Dataset(val_samples,   scaler, augment=False, use_mat=CFG["use_mat"])
    test_ds  = DACIA5Dataset(all_test,      scaler, augment=False, use_mat=CFG["use_mat"])

    train_sampler = build_weighted_sampler(train_samples)

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                              sampler=train_sampler,
                              num_workers=CFG["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"] * 2,
                              shuffle=False,
                              num_workers=CFG["num_workers"], pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=CFG["batch_size"] * 2,
                              shuffle=False,
                              num_workers=CFG["num_workers"], pin_memory=True)

    # ── 5. Model, optimizer, scheduler ────────────────────────────────────
    model = DACIA5Classifier(
        in_ch=CFG["num_bands"],
        base=CFG["base_ch"],
        num_classes=CFG["num_classes"],
        dropout=CFG["dropout"],
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Parameters: {total_params:,}")

    criterion = LabelSmoothCE(CFG["label_smoothing"], CFG["num_classes"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"]
    )
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer, max_lr=CFG["lr"],
        steps_per_epoch=steps_per_epoch,
        epochs=CFG["epochs"],
        pct_start=CFG["warmup_pct"],
        anneal_strategy="cos",
    )
    amp_scaler = torch.cuda.amp.GradScaler() if DEVICE.type == "cuda" else None

    # ── 6. Training ────────────────────────────────────────────────────────
    best_q1      = 0.0
    best_path    = os.path.join(CFG["save_dir"], "best_model.pth")
    history      = dict(train_loss=[], train_acc=[], val_q1=[], val_oa=[], val_aa=[])

    print(f"\n[INFO] Training for {CFG['epochs']} epochs …\n")
    for epoch in range(1, CFG["epochs"] + 1):
        t_loss, t_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, amp_scaler
        )
        v_preds, v_labels = evaluate(model, val_loader)
        q1, aa, oa, pca   = compute_q1(v_preds, v_labels)

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_q1"].append(q1)
        history["val_oa"].append(oa)
        history["val_aa"].append(aa)

        marker = " ★" if q1 > best_q1 else ""
        print(f"Epoch {epoch:03d}/{CFG['epochs']}  "
              f"loss={t_loss:.4f}  train_acc={t_acc:.1f}%  "
              f"OA={oa:.1f}%  AA={aa:.1f}%  Q1={q1:.1f}{marker}")

        if q1 > best_q1:
            best_q1 = q1
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "q1": q1, "oa": oa, "aa": aa,
                "scaler": scaler,
            }, best_path)

    print(f"\n[INFO] Best val Q1 = {best_q1:.1f}  (saved → {best_path})")

    # ── 7. Final evaluation on validation set (best checkpoint) ───────────
    ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    print(f"\n[INFO] Loaded best checkpoint (epoch {ckpt['epoch']}, Q1={ckpt['q1']:.1f})")

    v_preds, v_labels = evaluate(model, val_loader)
    q1, aa, oa, pca   = compute_q1(v_preds, v_labels)
    print(f"\n── Validation Results (best model) ──")
    print(f"  Overall Accuracy (OA) : {oa:.2f}%")
    print(f"  Average Accuracy (AA) : {aa:.2f}%")
    print(f"  Q1                    : {q1:.2f}")
    print(f"  Per-class:")
    for name, acc in zip(CLASS_NAMES, pca):
        print(f"    {name:<12} {acc:.1f}%")

    # ── 8. Plots ───────────────────────────────────────────────────────────
    cm_path  = os.path.join(CFG["plot_dir"], "confusion_matrix.png")
    pca_path = os.path.join(CFG["plot_dir"], "per_class_accuracy.png")
    plot_confusion_matrix(v_labels, v_preds, cm_path)
    plot_per_class_accuracy(pca, oa, aa, q1, pca_path)

    # ── 9. Test inference (no labels available) ────────────────────────────
    if len(all_test) > 0:
        print("\n[INFO] Running inference on test set …")
        test_preds, _ = evaluate(model, test_loader)
        pred_path = os.path.join(CFG["save_dir"], "test_predictions.npy")
        np.save(pred_path, np.array(test_preds))
        print(f"[INFO] Test predictions saved → {pred_path}")

    print("\n[DONE] All outputs saved.")
    return model, scaler


# ─────────────────────────────── ENTRY ─────────────────────────────────────
if __name__ == "__main__":
    main()
