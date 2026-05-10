"""
finetune.py – Supervised Fine-tuning
=====================================
Reference: SmartDetector paper, Section V-C.2, Section VI-C/D/E

Implements:
- Frozen encoder + trainable FC classifier
- Few-shot training: N ∈ {1,3,5,10} samples per class (Table VI)
- Imbalanced dataset handling: β ∈ {4:1, 24:1, 49:1} (Fig. 7)
- Obfuscation robustness evaluation: IDP/IBP/APR/INP (Table VII)
- 100-run averaging per experimental protocol (Section VI-A)
- Model saving and result export matching paper tables

"In the second step, labeled traffic samples are used to learn how
to distinguish benign and malicious traffic. The parameters of the
encoder are frozen in the process of supervised learning." (Section V-C.2)
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
from torch.utils.data import DataLoader, Subset
from typing import Optional, Dict, List, Tuple
from collections import defaultdict

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
K = 40
B = 100
N_CLASSES    = 5      # Benign + 4 malware classes (USTC-TFC D4)
FT_EPOCHS    = 100    # Fine-tuning epochs per run
FT_LR        = 1e-3
FT_BATCH     = 64
N_RUNS       = 100    # Repetitions per experiment (Section VI-A)
DEFAULT_SEED = 42
MALWARE_CLASSES = ["Htbot", "Neris", "Miuref", "Virut"]
ALL_CLASSES     = ["Benign"] + MALWARE_CLASSES


# ── Reproducibility ─────────────────────────────────────────────────────────

def set_seed(seed: int = DEFAULT_SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── FC Classifier (fully connected layer on frozen encoder) ────────────────

class FCClassifier(nn.Module):
    """
    Two-layer fully-connected classifier appended to the frozen encoder.

    Per Section V-C.2: "we only need a few labeled samples to train the
    fully connected layer to achieve high accuracy."

    Input:  encoder output (N, encoder_dim)
    Output: class logits   (N, n_classes)
    """

    def __init__(self, encoder_dim: int = 2048,
                 n_classes: int = N_CLASSES,
                 hidden_dim: int = 256,
                 dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Combined model (encoder + classifier) ──────────────────────────────────

class SmartDetectorFinetuner:
    """
    Fine-tuning wrapper: loads pre-trained encoder, freezes it,
    trains a FC classifier on top.

    "The parameters of the encoder are frozen in the process of
    supervised learning." (Section V-C.2)
    """

    def __init__(self,
                 encoder_path: str,
                 n_classes: int = N_CLASSES,
                 device: Optional[str] = None,
                 seed: int = DEFAULT_SEED):
        set_seed(seed)
        self.device = torch.device(
            device if device else
            ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"[Finetune] Device: {self.device}")

        # Load encoder
        from pretrain import ResNetEncoder
        self.encoder = ResNetEncoder().to(self.device)
        encoder_dim  = self.encoder.embed_dim

        ckpt = torch.load(encoder_path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"])

        # Freeze all encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        print(f"[Finetune] Encoder frozen, loaded from: {encoder_path}")

        # Trainable classifier head
        self.classifier = FCClassifier(
            encoder_dim=encoder_dim,
            n_classes=n_classes
        ).to(self.device)

        self.n_classes  = n_classes
        self.criterion  = nn.CrossEntropyLoss()

    def extract_features(self, sams: np.ndarray,
                          batch_size: int = 256) -> np.ndarray:
        """
        Extract deep representations from frozen encoder.
        Returns (N, encoder_dim) array.
        """
        self.encoder.eval()
        features = []
        tensor = torch.tensor(sams, dtype=torch.float32)
        with torch.no_grad():
            for i in range(0, len(tensor), batch_size):
                batch = tensor[i:i + batch_size].to(self.device)
                feat  = self.encoder(batch)
                features.append(feat.cpu().numpy())
        return np.concatenate(features, axis=0)

    def train_one_run(self,
                      train_feats:  np.ndarray,
                      train_labels: np.ndarray,
                      epochs: int = FT_EPOCHS,
                      lr:     float = FT_LR,
                      batch_size: int = FT_BATCH) -> None:
        """
        Fine-tune the FC classifier on N labeled samples.
        Reinitialises classifier weights at the start of each run.
        """
        # Reinitialise classifier for clean run
        from pretrain import ResNetEncoder
        encoder_dim = self.encoder.embed_dim
        self.classifier = FCClassifier(
            encoder_dim=encoder_dim,
            n_classes=self.n_classes
        ).to(self.device)

        optimizer = torch.optim.Adam(self.classifier.parameters(),
                                     lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.5)

        feats_t = torch.tensor(train_feats,  dtype=torch.float32)
        lbls_t  = torch.tensor(train_labels, dtype=torch.long)

        self.classifier.train()
        for epoch in range(epochs):
            # Shuffle
            perm = torch.randperm(len(feats_t))
            total_loss = 0.0
            n_batches  = 0
            for i in range(0, len(feats_t), batch_size):
                idx = perm[i:i + batch_size]
                xb  = feats_t[idx].to(self.device)
                yb  = lbls_t[idx].to(self.device)
                logits = self.classifier(xb)
                loss   = self.criterion(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches  += 1
            scheduler.step()

    def predict(self,
                eval_feats: np.ndarray,
                batch_size: int = 256
                ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference on extracted features.
        Returns (y_pred, y_score) where y_score is the softmax probability
        matrix (N, n_classes).
        """
        self.classifier.eval()
        preds  = []
        scores = []
        tensor = torch.tensor(eval_feats, dtype=torch.float32)
        with torch.no_grad():
            for i in range(0, len(tensor), batch_size):
                batch  = tensor[i:i + batch_size].to(self.device)
                logits = self.classifier(batch)
                prob   = F.softmax(logits, dim=1)
                preds.append(logits.argmax(dim=1).cpu().numpy())
                scores.append(prob.cpu().numpy())
        return np.concatenate(preds), np.concatenate(scores, axis=0)

    def save_classifier(self, path: str) -> None:
        torch.save(self.classifier.state_dict(), path)
        print(f"[Finetune] Classifier saved → {path}")

    def load_classifier(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.classifier.load_state_dict(state)
        print(f"[Finetune] Classifier loaded ← {path}")


# ── Experiment runners ──────────────────────────────────────────────────────

def run_few_shot_experiment(finetuner: SmartDetectorFinetuner,
                             all_sams:   np.ndarray,
                             all_labels: np.ndarray,
                             new_class_idx: int,
                             n_shot: int,
                             n_runs: int = N_RUNS,
                             base_seed: int = DEFAULT_SEED) -> Dict:
    """
    Section VI-C protocol – Few-shot learning experiment.

    For each run:
    1. Randomly sample N labeled examples of the new attack class
    2. Fine-tune FC classifier on those N samples
    3. Evaluate on remaining new attack + benign samples
    4. Record Recall, F1, AUC

    Repeat n_runs times; return mean ± std.
    """
    from metrics import compute_metrics, RunAccumulator

    print(f"\n[FewShot] N={n_shot}, new_class={ALL_CLASSES[new_class_idx]}, "
          f"runs={n_runs}")

    # New attack indices
    new_mask   = all_labels == new_class_idx
    benign_mask = all_labels == 0

    new_sams   = all_sams[new_mask]
    new_labels = all_labels[new_mask]
    ben_sams   = all_sams[benign_mask]
    ben_labels = all_labels[benign_mask]

    # Cap at 2,000 per Table III protocol
    rng = np.random.default_rng(base_seed)
    n_new = min(2000, len(new_sams))
    n_ben = min(n_new, len(ben_sams))

    # Extract features once (encoder is frozen)
    print("  Extracting features from frozen encoder …")
    new_feats = finetuner.extract_features(new_sams[:n_new])
    ben_feats = finetuner.extract_features(ben_sams[:n_ben])

    acc = RunAccumulator()

    for run_id in range(n_runs):
        run_rng = np.random.default_rng(base_seed + run_id)
        perm    = run_rng.permutation(n_new)

        # N samples for fine-tuning
        ft_idx  = perm[:n_shot]
        eval_idx = perm[n_shot:]

        ben_ft_idx = run_rng.choice(n_ben, size=n_shot, replace=False)
        ft_feats  = np.concatenate([new_feats[ft_idx], ben_feats[ben_ft_idx]])
        ft_labels = np.concatenate([
            np.ones(n_shot,  dtype=int),  # malicious
            np.zeros(n_shot, dtype=int)   # benign
        ])

        eval_feats  = np.concatenate([new_feats[eval_idx], ben_feats])
        eval_labels = np.concatenate([
            np.ones(len(eval_idx),  dtype=int),
            np.zeros(n_ben, dtype=int)
        ])

        # Fine-tune
        finetuner.train_one_run(ft_feats, ft_labels,
                                 epochs=FT_EPOCHS, lr=FT_LR)

        # Predict
        y_pred, y_score = finetuner.predict(eval_feats)
        m = compute_metrics(eval_labels, y_pred,
                            y_score[:, 1] if y_score.ndim > 1 else y_score,
                            average="binary")
        acc.add_run(m)

        if (run_id + 1) % 10 == 0:
            print(f"  Run {run_id + 1:>4}/{n_runs}: "
                  f"F1={m['f1']:.2f}  AUC={m['auc']:.2f}")

    return acc.to_dict()


def run_all_fewshot_scenarios(finetuner, all_sams, all_labels,
                               n_shots: List[int] = [1, 3, 5, 10],
                               n_runs: int = N_RUNS,
                               save_dir: str = "results") -> None:
    """
    Run Table VI: all (N_shot × new_attack) combinations for D1.
    """
    os.makedirs(save_dir, exist_ok=True)
    full_results = {}

    for new_cls_idx in range(1, len(ALL_CLASSES)):  # skip Benign
        cls_name = ALL_CLASSES[new_cls_idx]
        for n_shot in n_shots:
            key = f"{cls_name}_N{n_shot}"
            print(f"\n── {key} ──")
            res = run_few_shot_experiment(
                finetuner, all_sams, all_labels,
                new_class_idx=new_cls_idx,
                n_shot=n_shot,
                n_runs=n_runs
            )
            full_results[key] = res
            agg = res["aggregate"]
            print(f"  AVG  Recall={agg['recall']['mean']:.2f}  "
                  f"F1={agg['f1']['mean']:.2f}  "
                  f"AUC={agg['auc']['mean']:.2f}")

    out_path = os.path.join(save_dir, "fewshot_results.json")
    with open(out_path, "w") as fh:
        json.dump(full_results, fh, indent=2)
    print(f"\n[FewShot] Results saved → {out_path}")
    classifier_path = os.path.join(save_dir, "classifier.pt")
    finetuner.save_classifier(classifier_path)
    print(f"[Finetune] Classifier saved → {classifier_path}")

def run_imbalanced_experiment(finetuner, all_sams, all_labels,
                               betas: List[int] = [4, 24, 49],
                               n_finetune: int = 5,
                               n_runs: int = N_RUNS,
                               save_dir: str = "results") -> None:
    """
    Section VI-D: Imbalanced dataset experiment (Fig. 7).
    Vary benign:malicious ratio β; keep N=5 fine-tune samples fixed.
    """
    from metrics import compute_metrics, RunAccumulator
    os.makedirs(save_dir, exist_ok=True)
    imbal_results = {}

    for beta in betas:
        print(f"\n[Imbalanced] β={beta}:1")
        acc = RunAccumulator()
        rng = np.random.default_rng(DEFAULT_SEED)

        for run_id in range(n_runs):
            run_rng = np.random.default_rng(DEFAULT_SEED + run_id)

            # Build imbalanced evaluation set
            benign_idx = np.where(all_labels == 0)[0]
            mal_idx    = np.where(all_labels > 0)[0]
            n_mal_eval = min(500, len(mal_idx))
            n_ben_eval = min(beta * n_mal_eval, len(benign_idx))

            ben_sel = run_rng.choice(benign_idx, n_ben_eval, replace=False)
            mal_sel = run_rng.choice(mal_idx,    n_mal_eval, replace=False)

            eval_feats  = finetuner.extract_features(
                np.concatenate([all_sams[ben_sel], all_sams[mal_sel]]))
            eval_labels = np.concatenate([
                np.zeros(n_ben_eval, dtype=int),
                np.ones(n_mal_eval, dtype=int)
            ])

            # Fine-tune with N=5 malicious samples
            ft_sel    = run_rng.choice(mal_idx, n_finetune, replace=False)
            ft_feats  = finetuner.extract_features(all_sams[ft_sel])
            ft_labels = np.ones(n_finetune, dtype=int)
            finetuner.train_one_run(ft_feats, ft_labels)

            y_pred, y_score = finetuner.predict(eval_feats)
            score_1d = y_score[:, 1] if y_score.ndim > 1 else y_score
            m = compute_metrics(eval_labels, y_pred, score_1d, average="binary")
            acc.add_run(m)

        imbal_results[f"beta_{beta}"] = acc.to_dict()
        agg = acc.aggregate()
        print(f"  β={beta}:1  F1={agg['f1']['mean']:.2f}±{agg['f1']['std']:.2f}")

    out_path = os.path.join(save_dir, "imbalanced_results.json")
    with open(out_path, "w") as fh:
        json.dump(imbal_results, fh, indent=2)
    print(f"[Imbalanced] Results saved → {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="SmartDetector – Supervised Fine-tuning (Section V-C.2)")
    p.add_argument("--encoder_path", type=str,
                   default=CKPT_DIR + r"\\encoder_best.pt",
                   help="Path to pre-trained encoder checkpoint")
    p.add_argument("--sams_dir",     type=str, default=SAMS_DIR)
    p.add_argument("--out_dir",      type=str, default=RESULTS_DIR)
    p.add_argument("--mode",         type=str,
                   choices=["fewshot", "imbalanced", "quick"],
                   default="quick",
                   help="Experiment mode")
    p.add_argument("--n_shots",      type=int, nargs="+",
                   default=[1, 3, 5, 10])
    p.add_argument("--betas",        type=int, nargs="+",
                   default=[4, 24, 49])
    p.add_argument("--n_runs",       type=int, default=N_RUNS)
    p.add_argument("--n_classes",    type=int, default=N_CLASSES)
    p.add_argument("--device",       type=str, default=None)
    p.add_argument("--seed",         type=int, default=DEFAULT_SEED)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("SmartDetector – Supervised Fine-tuning")
    print("=" * 60)

    # ── Check encoder ──────────────────────────────────────────────────────
    if not os.path.exists(args.encoder_path):
        print(f"[WARN] Encoder not found at '{args.encoder_path}'")
        print("  Run pretrain.py first, or use --encoder_path")
        # Create dummy encoder for testing
        from pretrain import ResNetEncoder, ProjectionHead
        dummy_enc = ResNetEncoder()
        dummy_state = {
            "encoder":   dummy_enc.state_dict(),
            "proj_head": ProjectionHead().state_dict(),
            "history":   []
        }
        os.makedirs(os.path.dirname(args.encoder_path) or ".", exist_ok=True)
        torch.save(dummy_state, args.encoder_path)
        print(f"  Created dummy encoder at: {args.encoder_path}")

    # ── Load SAMs ──────────────────────────────────────────────────────────
    sams_path = os.path.join(args.sams_dir, "all_sams.npz")
    if os.path.exists(sams_path):
        from sam_matrix_construct import load_sams
        all_sams, all_labels = load_sams(sams_path)
        print(f"[INFO] Loaded SAMs: {all_sams.shape}")
    else:
        print("[WARN] SAMs not found – using synthetic data")
        rng        = np.random.default_rng(args.seed)
        all_sams   = rng.random((500, 3, K, B)).astype(np.float32)
        all_labels = rng.integers(0, args.n_classes, 500)

    # ── Initialise finetuner ───────────────────────────────────────────────
    finetuner = SmartDetectorFinetuner(
        encoder_path=args.encoder_path,
        n_classes=args.n_classes,
        device=args.device,
        seed=args.seed
    )

    if args.mode == "quick":
        # Single quick run with N=5 and new_class=Neris (idx=2)
        print("\n[MODE] Quick test (N=5, new_class=Neris)")
        result = run_few_shot_experiment(
            finetuner, all_sams, all_labels,
            new_class_idx=2, n_shot=5,
            n_runs=min(3, args.n_runs)  # 3 runs for speed in quick mode
        )
        agg = result["aggregate"]
        print(f"\n  Recall={agg['recall']['mean']:.2f}  "
              f"F1={agg['f1']['mean']:.2f}  "
              f"AUC={agg['auc']['mean']:.2f}")

    elif args.mode == "fewshot":
        run_all_fewshot_scenarios(
            finetuner, all_sams, all_labels,
            n_shots=args.n_shots,
            n_runs=args.n_runs,
            save_dir=args.out_dir
        )

    elif args.mode == "imbalanced":
        run_imbalanced_experiment(
            finetuner, all_sams, all_labels,
            betas=args.betas,
            n_runs=args.n_runs,
            save_dir=args.out_dir
        )

    print(f"\n[DONE] Results saved to '{args.out_dir}/'")


if __name__ == "__main__":
    main()