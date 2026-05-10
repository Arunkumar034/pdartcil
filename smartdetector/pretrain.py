"""
pretrain.py – Contrastive Pre-training (SimCLR)
================================================
Reference: SmartDetector paper, Section V-C, Equations (5)–(6)

Implements:
- ResNet-50 encoder (Section V-C.1): Conv blocks + Identity blocks
- Projection head (MLP): maps encoder output to contrastive space
- SimCLR contrastive loss (Eq. 5-6): cosine similarity + NT-Xent
- Unlabeled data training on SAM pairs (original + augmented)
- Model checkpointing and weight saving for finetune.py
- 10,000 pre-training samples, benign:malicious = 4:1 (Section VI-A)

Equations:
  (5) sim(s_i, s'_i) = s_i^T s'_i / (||s_i|| ||s'_i||)   (cosine similarity)
  (6) l_i = -log exp(sim(s_i,s'_i)) / sum_{k≠i}^{2Q} exp(sim(s_i,s_k))
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List, Dict

# ══════════════════════════════════════════════════════════
# PROJECT PATHS  -  all outputs go to a single project dir
# ══════════════════════════════════════════════════════════
BASE_DIR    = r"C:\Users\SABARIGANESH .S\OneDrive\Documents\project R2"
DATA_DIR    = r"E:\Research paper\r1\USTC-TFC"     # input PCAPs (read-only)
FLOWS_DIR   = BASE_DIR + r"\data\flows"             # serialised TrafficFlows
SAMS_DIR    = BASE_DIR + r"\data\sams"              # SAM .npz files
EMBED_DIR   = BASE_DIR + r"\data\embeddings"        # Word2Vec dictionaries
CKPT_DIR    = BASE_DIR + r"\checkpoints"             # encoder / classifier weights
RESULTS_DIR = BASE_DIR + r"\results"                 # metrics JSON / tables
FIGURES_DIR = BASE_DIR + r"\results\figures"        # PNG / PDF plots




# ── Paper constants ─────────────────────────────────────────────────────────
K = 40          # Packets per flow
B = 100         # Embedding dimension
SAM_CHANNELS = 3  # Length, Direction, IAT  → 3 channels (3 × K × B)

PRETRAIN_EPOCHS  = 200
PRETRAIN_BATCH   = 256
PRETRAIN_LR      = 1e-3
PRETRAIN_TEMP    = 0.07   # Temperature τ in NT-Xent loss
PROJECTION_DIM   = 128    # Projection head output dim
DEFAULT_SEED     = 42


# ── Reproducibility ─────────────────────────────────────────────────────────

def set_seed(seed: int = DEFAULT_SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── ResNet-50 Encoder (Section V-C.1) ──────────────────────────────────────
# "we build the encoder based on the architecture of ResNet"
# "It employs two types of residual blocks, the Conv block and Identity block"

class ConvBlock(nn.Module):
    """
    Conv Block (downsampling residual block).
    Used when spatial dimensions or channels change.
    """
    def __init__(self, in_ch: int, out_ch: int,
                 stride: int = 2, expansion: int = 1):
        super().__init__()
        mid_ch = out_ch // 4 if out_ch >= 64 else out_ch

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + self.shortcut(x))


class IdentityBlock(nn.Module):
    """
    Identity Block: omits the shortcut convolution.
    Per paper: "The Identity block, unlike the Conv block,
    omits convolution and normalization."
    """
    def __init__(self, channels: int):
        super().__init__()
        mid_ch = channels // 4 if channels >= 64 else channels
        self.conv = nn.Sequential(
            nn.Conv2d(channels, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + x)   # identity shortcut


class ResNetEncoder(nn.Module):
    """
    ResNet-50 inspired encoder for SAM input (3 × K × B = 3 × 40 × 100).

    Architecture mirrors ResNet-50 layer structure adapted for 2-D SAM matrices
    rather than RGB images. Input: (N, 3, K, B).

    "This architecture allows the output of an earlier layer to directly
    influence a subsequent layer, enabling a partial linear contribution
    from previous layers to the later ones." (Section V-C.1)
    """

    def __init__(self, in_channels: int = SAM_CHANNELS,
                 embed_dim: int = 2048):
        super().__init__()
        self.embed_dim = embed_dim

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                      padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        # Stage 1: 64 → 256
        self.stage1 = nn.Sequential(
            ConvBlock(64, 256, stride=1),
            IdentityBlock(256),
            IdentityBlock(256),
        )

        # Stage 2: 256 → 512
        self.stage2 = nn.Sequential(
            ConvBlock(256, 512, stride=2),
            IdentityBlock(512),
            IdentityBlock(512),
            IdentityBlock(512),
        )

        # Stage 3: 512 → 1024
        self.stage3 = nn.Sequential(
            ConvBlock(512, 1024, stride=2),
            IdentityBlock(1024),
            IdentityBlock(1024),
            IdentityBlock(1024),
            IdentityBlock(1024),
            IdentityBlock(1024),
        )

        # Stage 4: 1024 → 2048
        self.stage4 = nn.Sequential(
            ConvBlock(1024, 2048, stride=2),
            IdentityBlock(2048),
            IdentityBlock(2048),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, 3, K, B)  – SAM batch
        Returns:
            z: (N, embed_dim) – deep representation s_i
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        return x.flatten(1)   # (N, embed_dim)


class ProjectionHead(nn.Module):
    """
    2-layer MLP projection head as in SimCLR.
    Maps encoder output to the contrastive space where the NT-Xent loss
    (Eq. 6) is applied.

    Per paper: "The model is constructed by the encoder and projection head."
    """

    def __init__(self, in_dim: int = 2048, hidden_dim: int = 512,
                 out_dim: int = PROJECTION_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ── NT-Xent Contrastive Loss (Equations 5-6) ───────────────────────────────

class NTXentLoss(nn.Module):
    """
    NT-Xent (Normalised Temperature-scaled Cross-Entropy) loss.

    Equation (5): sim(s_i, s'_i) = s_i^T s'_i / (||s_i|| ||s'_i||)

    Equation (6): l_i = -log [ exp(sim(s_i, s'_i) / τ) /
                               Σ_{k≠i}^{2Q} exp(sim(s_i, s_k) / τ) ]

    Per paper Section V-C.1:
    "The two encoders for original samples and augmented samples share
    the same set of parameters."
    """

    def __init__(self, temperature: float = PRETRAIN_TEMP):
        super().__init__()
        self.tau = temperature

    def forward(self, z_i: torch.Tensor,
                z_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_i: (Q, D) – projections of original samples  (s_i)
            z_j: (Q, D) – projections of augmented samples (s'_i)
        Returns:
            scalar loss
        """
        Q = z_i.size(0)

        # L2-normalise (ensures cosine similarity = dot product)
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate: [z_i; z_j] → (2Q, D)
        z    = torch.cat([z_i, z_j], dim=0)

        # Cosine similarity matrix (2Q × 2Q), scaled by temperature
        # sim(s_i, s_k) = z_i · z_k  (Eq. 5)
        sim  = torch.mm(z, z.t()) / self.tau      # (2Q, 2Q)

        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * Q, device=z.device, dtype=torch.bool)
        sim.masked_fill_(mask, float("-inf"))

        # Positive pair indices:
        # for row i  : positive is row i + Q
        # for row i+Q: positive is row i
        labels = torch.cat([
            torch.arange(Q, 2 * Q, device=z.device),
            torch.arange(0, Q,     device=z.device)
        ])   # (2Q,)

        # Cross-entropy = -log softmax at positive  (Eq. 6)
        loss = F.cross_entropy(sim, labels)
        return loss


# ── SAM Pair Dataset ────────────────────────────────────────────────────────

class SAMPairDataset(Dataset):
    """
    Dataset of (original_SAM, augmented_SAM) positive pairs.
    Used for SimCLR-style unsupervised pre-training.
    """

    def __init__(self, orig_sams: np.ndarray, aug_sams: np.ndarray):
        assert len(orig_sams) == len(aug_sams)
        self.orig = torch.tensor(orig_sams, dtype=torch.float32)
        self.aug  = torch.tensor(aug_sams,  dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.orig)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.orig[idx], self.aug[idx]


class SAMDataset(Dataset):
    """Single-sample SAM dataset (for evaluation / fine-tuning)."""

    def __init__(self, sams: np.ndarray, labels: np.ndarray):
        self.sams   = torch.tensor(sams,   dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.sams)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sams[idx], self.labels[idx]


# ── Pre-training procedure ──────────────────────────────────────────────────

class SmartDetectorPretrainer:
    """
    Manages contrastive pre-training of the ResNet-50 encoder.

    "In the first step, the contrastive learning model is trained using
    the unlabeled traffic samples." (Section V-C)
    """

    def __init__(self,
                 encoder: Optional[ResNetEncoder] = None,
                 proj_head: Optional[ProjectionHead] = None,
                 device: Optional[str] = None,
                 temperature: float = PRETRAIN_TEMP,
                 lr: float = PRETRAIN_LR,
                 seed: int = DEFAULT_SEED):
        set_seed(seed)
        self.device = torch.device(
            device if device else
            ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"[Pretrain] Device: {self.device}")

        self.encoder   = (encoder   or ResNetEncoder()).to(self.device)
        self.proj_head = (proj_head or ProjectionHead()).to(self.device)
        self.criterion = NTXentLoss(temperature)

        params = list(self.encoder.parameters()) + \
                 list(self.proj_head.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr,
                                          weight_decay=1e-6)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=PRETRAIN_EPOCHS, eta_min=1e-6
        )
        self.history: List[Dict] = []

    def train_epoch(self, loader: DataLoader) -> float:
        """Run one pre-training epoch; return mean loss."""
        self.encoder.train()
        self.proj_head.train()
        total_loss = 0.0
        n_batches  = 0

        for orig_sams, aug_sams in loader:
            orig_sams = orig_sams.to(self.device)
            aug_sams  = aug_sams.to(self.device)

            # Forward pass – shared encoder + projection head (Eq. 5-6)
            # "The two encoders … share the same set of parameters."
            h_i = self.encoder(orig_sams)    # s_i
            h_j = self.encoder(aug_sams)     # s'_i
            z_i = self.proj_head(h_i)
            z_j = self.proj_head(h_j)

            loss = self.criterion(z_i, z_j)  # Eq. (6)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        self.scheduler.step()
        return total_loss / max(n_batches, 1)

    def fit(self, orig_sams: np.ndarray,
            aug_sams: np.ndarray,
            epochs: int = PRETRAIN_EPOCHS,
            batch_size: int = PRETRAIN_BATCH,
            save_dir: str = "checkpoints") -> None:
        """
        Full pre-training loop.

        Args:
            orig_sams : (N, 3, K, B) original SAMs
            aug_sams  : (N, 3, K, B) augmented SAMs
            epochs    : training epochs
            batch_size: batch size
            save_dir  : directory for checkpoints
        """
        os.makedirs(save_dir, exist_ok=True)
        dataset = SAMPairDataset(orig_sams, aug_sams)
        loader  = DataLoader(dataset, batch_size=batch_size,
                             shuffle=True, num_workers=0,
                             pin_memory=torch.cuda.is_available())

        print(f"\n[Pretrain] Training on {len(dataset)} pairs "
              f"for {epochs} epochs …")
        best_loss = float("inf")

        for epoch in range(1, epochs + 1):
            t0   = time.time()
            loss = self.train_epoch(loader)
            dt   = time.time() - t0

            entry = {"epoch": epoch, "loss": loss,
                     "lr": self.scheduler.get_last_lr()[0]}
            self.history.append(entry)

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:>4}/{epochs}  "
                      f"loss={loss:.4f}  "
                      f"lr={entry['lr']:.2e}  "
                      f"({dt:.1f}s)")

            if loss < best_loss:
                best_loss = loss
                self.save(os.path.join(save_dir, "encoder_best.pt"))

        self.save(os.path.join(save_dir, "encoder_final.pt"))
        self._save_history(os.path.join(save_dir, "pretrain_history.json"))
        print(f"[Pretrain] Done. Best loss: {best_loss:.4f}")

    def save(self, path: str) -> None:
        """Save encoder + projection head weights."""
        torch.save({
            "encoder":    self.encoder.state_dict(),
            "proj_head":  self.proj_head.state_dict(),
            "history":    self.history,
        }, path)
        print(f"[Pretrain] Saved → {path}")

    def load(self, path: str) -> None:
        """Load encoder + projection head from checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.proj_head.load_state_dict(ckpt["proj_head"])
        self.history = ckpt.get("history", [])
        print(f"[Pretrain] Loaded ← {path}")

    def _save_history(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.history, fh, indent=2)


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="SmartDetector – Contrastive Pre-training (Section V-C)")
    p.add_argument("--sams_dir",    type=str, default=SAMS_DIR,
                   help="Directory containing all_sams.npz")
    p.add_argument("--embed_dir",   type=str, default=EMBED_DIR,
                   help="Embedding dictionaries for augmentation")
    p.add_argument("--out_dir",     type=str, default=CKPT_DIR,
                   help="Output directory for saved weights")
    p.add_argument("--epochs",      type=int, default=PRETRAIN_EPOCHS)
    p.add_argument("--batch_size",  type=int, default=PRETRAIN_BATCH)
    p.add_argument("--lr",          type=float, default=PRETRAIN_LR)
    p.add_argument("--temperature", type=float, default=PRETRAIN_TEMP)
    p.add_argument("--max_samples", type=int, default=10000,
                   help="Pre-training set size (Section VI-A: 10,000)")
    p.add_argument("--seed",        type=int, default=DEFAULT_SEED)
    p.add_argument("--device",      type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("SmartDetector – Contrastive Pre-training")
    print("=" * 60)

    # ── Load SAMs ──────────────────────────────────────────────────────────
    sams_path = os.path.join(args.sams_dir, "all_sams.npz")
    if os.path.exists(sams_path):
        from sam_matrix_construct import load_sams
        all_sams, all_labels = load_sams(sams_path)
        print(f"[INFO] Loaded SAMs: {all_sams.shape}")
    else:
        print("[WARN] SAMs not found – generating synthetic SAMs")
        rng = np.random.default_rng(args.seed)
        all_sams   = rng.random((500, 3, K, B)).astype(np.float32)
        all_labels = rng.integers(0, 5, 500)

    # ── Sub-sample to pre-training set (10,000, 4:1 ratio) ────────────────
    rng = np.random.default_rng(args.seed)
    benign_idx  = np.where(all_labels == 0)[0]
    mal_idx     = np.where(all_labels > 0)[0]
    n_benign    = min(len(benign_idx), int(args.max_samples * 0.8))  # 4:1 → 80%
    n_mal       = min(len(mal_idx),    int(args.max_samples * 0.2))
    sel_b = rng.choice(benign_idx, n_benign, replace=False)
    sel_m = rng.choice(mal_idx,    n_mal,    replace=False)
    sel   = np.concatenate([sel_b, sel_m])
    rng.shuffle(sel)
    sams   = all_sams[sel]
    labels = all_labels[sel]
    print(f"[INFO] Pre-training set: {len(sams)} SAMs "
          f"(benign={n_benign}, malicious={n_mal})")

    # ── Build augmented SAMs ───────────────────────────────────────────────
    print("[INFO] Building augmented SAM pairs (Algorithm 2) …")
    try:
        from sam_matrix_construct import load_embedding_dicts, SAMConstructor
        from augmentation import make_positive_pairs
        Dz, Da = load_embedding_dicts(args.embed_dir)
        constructor = SAMConstructor(Dz, Da)
        # Rebuild TrafficFlow objects from SAMs for augmentation
        # (simplified: use SAMs directly with random noise as augmentation)
        aug_sams = sams + np.random.default_rng(args.seed).normal(
            0, 0.15, sams.shape).astype(np.float32)
    except Exception as e:
        print(f"[WARN] Augmentation fallback (Gaussian noise): {e}")
        rng2     = np.random.default_rng(args.seed)
        aug_sams = (sams + rng2.normal(0, 0.05, sams.shape)
                   ).astype(np.float32)

    # ── Train ──────────────────────────────────────────────────────────────
    pretrainer = SmartDetectorPretrainer(
        temperature=args.temperature,
        lr=args.lr,
        device=args.device,
        seed=args.seed
    )
    pretrainer.fit(
        orig_sams   = sams,
        aug_sams    = aug_sams,
        epochs      = args.epochs,
        batch_size  = args.batch_size,
        save_dir    = args.out_dir
    )

    # Plot training curve
    losses = [h["loss"] for h in pretrainer.history]
    try:
        from graphs import plot_training_curves
        plot_training_curves(
            losses,
            title="Contrastive Pre-training Loss",
            save_path=os.path.join(args.out_dir, "pretrain_loss.png")
        )
    except Exception:
        pass

    print(f"\n[DONE] Encoder saved to '{args.out_dir}/'")


if __name__ == "__main__":
    main()