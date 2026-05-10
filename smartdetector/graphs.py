"""
graphs.py – Visualisation Suite
================================
Reference: SmartDetector paper Figures 2, 5, 6, 7, 8 and Tables V, VI-XI

Implements:
- KDE plots (Section IV-A, Fig. 2): Packet Length, Down/Up Ratio, IAT
- t-SNE embeddings (Section V.C.1, Fig. 5): before/after encoder
- ROC curves (Section VI-E, Fig. 8): four obfuscation strategies
- Euclidean distance bar charts (Table V, Fig. 6): representations × obfuscation
- Imbalanced dataset F1 bar chart (Fig. 7)
- Ablation study plots (Tables VIII, XI)
"""

import os
import json
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from typing import Dict, List, Optional, Tuple
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde

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



# ── Style configuration ─────────────────────────────────────────────────────
COLORS = {
    "benign":       "#2196F3",   # Blue
    "malicious":    "#F44336",   # Red
    "SmartDetector":"#4CAF50",   # Green
    "FC-Net":       "#FF9800",   # Orange
    "TF":           "#9C27B0",   # Purple
    "ST-Graph":     "#795548",   # Brown
    "DFR":          "#607D8B",   # Blue-grey
}

CLASS_COLORS = [
    "#2196F3", "#F44336", "#FF9800", "#4CAF50", "#9C27B0", "#795548"
]

METHOD_STYLES = {
    "FC-Net":       ("--",  "o"),
    "TF":           ("-.",  "s"),
    "ST-Graph":     (":",   "^"),
    "SmartDetector":("-",   "D"),
}

OBFS_LABELS = {
    "idp": "IDP",
    "ibp": "IBP",
    "apr": "APR",
    "inp": "INP",
}

# ── Fig. 2: KDE plots (Section IV-A) ───────────────────────────────────────

def plot_kde_features(benign_feats: Dict[str, np.ndarray],
                       malicious_feats: Dict[str, np.ndarray],
                       dataset_name: str = "CIC-IDS-2017",
                       save_path: Optional[str] = None) -> None:
    """
    Reproduce Fig. 2: KDE curves for Packet Length, Down/Up Ratio, IAT.

    Args:
        benign_feats   : dict { 'length': array, 'ratio': array, 'iat': array }
        malicious_feats: same structure for malicious traffic
        dataset_name   : title annotation
        save_path      : if given, save PNG to this path
    """
    feature_keys   = ["length", "ratio", "iat"]
    feature_labels = ["Packet Length [×100B]",
                      "Down/Up Ratio",
                      "IAT [×100 μs]"]
    x_scales       = [100.0, 1.0, 1e4]   # unit conversions matching paper

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    fig.suptitle(dataset_name, fontsize=11, fontweight="bold")

    for ax, fkey, flabel, xscale in zip(axes, feature_keys,
                                         feature_labels, x_scales):
        b_data = np.asarray(benign_feats.get(fkey, []))
        m_data = np.asarray(malicious_feats.get(fkey, []))

        b_scaled = b_data / xscale if len(b_data) > 1 else np.array([0.0])
        m_scaled = m_data / xscale if len(m_data) > 1 else np.array([0.0])

        x_min = min(b_scaled.min(), m_scaled.min()) * 0.9
        x_max = max(b_scaled.max(), m_scaled.max()) * 1.1
        xs = np.linspace(x_min, x_max, 300)

        for data, color, label in [(b_scaled, COLORS["benign"],     "Benign"),
                                    (m_scaled, COLORS["malicious"], "Malicious")]:
            if len(data) > 1:
                try:
                    kde = gaussian_kde(data, bw_method="scott")
                    ax.plot(xs, kde(xs), color=color, lw=1.8, label=label)
                    ax.fill_between(xs, kde(xs), alpha=0.15, color=color)
                except Exception:
                    pass

        ax.set_xlabel(flabel, fontsize=9)
        ax.set_ylabel("Probability Density", fontsize=8)
        ax.legend(fontsize=8, framealpha=0.7)
        ax.set_xlim(left=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    _save_or_show(fig, save_path)


# ── Fig. 5: t-SNE visualisation (Section V.C.1) ────────────────────────────

def plot_tsne(representations: np.ndarray,
              labels: np.ndarray,
              class_names: List[str],
              title: str = "Encoder Output",
              include_obfuscated: bool = False,
              obf_representations: Optional[np.ndarray] = None,
              save_path: Optional[str] = None) -> None:
    """
    Reproduce Fig. 5: t-SNE 2-D projection of encoder representations.

    Per Section V.C.1: "The feature visualization is shown in Fig. 5,
    with dimensionality reduction using the t-SNE algorithm and displayed
    on two-dimensional axes."

    Args:
        representations     : shape (N, D) – encoder deep representations
        labels              : shape (N,)   – integer class labels
        class_names         : list of class name strings
        include_obfuscated  : if True, also plot obf_representations
        obf_representations : shape (N, D) – obfuscated sample reps
    """
    print(f"[t-SNE] Running dimensionality reduction on {len(representations)} samples …")
    all_reps = representations
    all_lbls = labels.copy()
    n_orig   = len(representations)

    if include_obfuscated and obf_representations is not None:
        all_reps = np.concatenate([representations, obf_representations], 0)
        all_lbls = np.concatenate([labels, labels + len(class_names)], 0)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30,
                n_iter=1000, verbose=0)
    z = tsne.fit_transform(all_reps.astype(np.float32))

    fig, ax = plt.subplots(figsize=(7, 6))
    n_classes = len(class_names)

    for cls_idx, cls_name in enumerate(class_names):
        mask = all_lbls[:n_orig] == cls_idx
        if mask.sum() == 0:
            continue
        col = CLASS_COLORS[cls_idx % len(CLASS_COLORS)]
        ax.scatter(z[:n_orig][mask, 0], z[:n_orig][mask, 1],
                   c=col, s=10, alpha=0.6, label=cls_name)

        if include_obfuscated and obf_representations is not None:
            obf_mask = all_lbls[n_orig:] == cls_idx
            ax.scatter(z[n_orig:][obf_mask, 0], z[n_orig:][obf_mask, 1],
                       c=col, s=10, alpha=0.6, marker="x",
                       label=f"{cls_name}-obf")

    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7, ncol=2, framealpha=0.8, loc="upper right")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ── Fig. 6: Euclidean distance bar chart ───────────────────────────────────

def plot_euclidean_distances(distance_data: Dict[str, Dict[str, float]],
                              save_path: Optional[str] = None) -> None:
    """
    Reproduce Fig. 6: Euclidean distance per representation under
    four obfuscation strategies (IDP, IBP, APR, INP).

    distance_data: {
        'IDP': { 'Color Image': 0.47, 'Direction Sequence': 0.46,
                 'Traffic Graph': 0.51, 'SAM': 0.60 },
        'IBP': { ... },
        ...
    }
    """
    strategies = ["IDP", "IBP", "APR", "INP"]
    reps       = ["Color Image", "Direction Sequence", "Traffic Graph", "SAM"]
    rep_colors = ["#90CAF9", "#A5D6A7", "#FFCC80", "#EF9A9A"]
    rep_hatches= ["///", "\\\\\\", "xxx", "|||"]

    x = np.arange(len(strategies))
    width = 0.18
    offsets = np.linspace(-(1.5 * width), 1.5 * width, 4)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for i, (rep, col, hatch) in enumerate(zip(reps, rep_colors, rep_hatches)):
        vals = [distance_data.get(s, {}).get(rep, 0.0) for s in strategies]
        bars = ax.bar(x + offsets[i], vals, width,
                      label=rep, color=col, hatch=hatch,
                      edgecolor="black", linewidth=0.6)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Obfuscation Strategy", fontsize=10)
    ax.set_ylabel("Euclidean Distance", fontsize=10)
    ax.set_title("Fig. 6 – Feature Distance Under Obfuscation", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, max(
        v for s in distance_data.values() for v in s.values()
    ) * 1.25 + 0.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ── Fig. 7: Imbalanced dataset F1 bar chart (Section VI-D) ─────────────────

def plot_imbalanced_f1(f1_data: Dict[str, Dict[str, float]],
                        save_path: Optional[str] = None) -> None:
    """
    Reproduce Fig. 7: F1 scores under different benign:malicious ratios.

    f1_data: {
        'FC-Net':        { '4:1': 86.31, '24:1': 84.99, '49:1': 43.51 },
        'TF':            { '4:1': 97.20, '24:1': 84.42, '49:1': 77.14 },
        'ST-Graph':      { '4:1': 96.65, '24:1': 79.54, '49:1': 79.54 },
        'SmartDetector': { '4:1': 97.90, '24:1': 96.65, '49:1': 96.16 },
    }
    """
    ratios  = ["4:1", "24:1", "49:1"]
    methods = ["FC-Net", "TF", "ST-Graph", "SmartDetector"]
    colors  = [COLORS[m] for m in methods]
    hatches = ["///", "\\\\\\", "xxx", ""]

    x = np.arange(len(ratios))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, 4)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (m, col, hatch) in enumerate(zip(methods, colors, hatches)):
        vals = [f1_data.get(m, {}).get(r, 0.0) for r in ratios]
        bars = ax.bar(x + offsets[i], vals, width,
                      label=m, color=col, hatch=hatch,
                      edgecolor="black", linewidth=0.6, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom",
                    fontsize=6.5, rotation=0)

    ax.set_xlabel("Benign-to-Malicious Ratio (Times)", fontsize=10)
    ax.set_ylabel("F1-Score (%)", fontsize=10)
    ax.set_title("Fig. 7 – F1 under Imbalanced Datasets", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(ratios)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ── Fig. 8: ROC curves (Section VI-E) ──────────────────────────────────────

def plot_roc_curves(roc_data: Dict[str, Dict[str, Tuple]],
                    obfs_strategy: str = "IDP-30%",
                    save_path: Optional[str] = None) -> None:
    """
    Reproduce Fig. 8: ROC curves for four obfuscation strategies.

    roc_data: {
        'method_name': { 'fpr': array, 'tpr': array, 'auc': float }
    }
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)

    for method, data in roc_data.items():
        fpr = np.asarray(data.get("fpr", [0, 1]))
        tpr = np.asarray(data.get("tpr", [0, 1]))
        auc = data.get("auc", 0.0)
        style, marker = METHOD_STYLES.get(method, ("-", "o"))
        ax.plot(fpr, tpr, linestyle=style,
                color=COLORS.get(method, "grey"),
                lw=1.8, label=f"{method} (AUC={auc:.2f})")

    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title(f"ROC – {obfs_strategy}", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_all_roc_panels(panel_data: Dict[str, Dict],
                         save_path: Optional[str] = None) -> None:
    """
    Reproduce Fig. 8 (a)-(d): 2×2 panel of ROC curves for
    IDP-30%, IBP-50%, APR, INP strategies.
    """
    strategies = ["IDP-30%", "IBP-50%", "APR", "INP"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    for ax, strat in zip(axes, strategies):
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
        strat_data = panel_data.get(strat, {})
        for method, data in strat_data.items():
            fpr = np.asarray(data.get("fpr", [0, 1]))
            tpr = np.asarray(data.get("tpr", [0, 1]))
            auc = data.get("auc", 0.0)
            ls, _ = METHOD_STYLES.get(method, ("-", "o"))
            ax.plot(fpr, tpr, linestyle=ls,
                    color=COLORS.get(method, "grey"),
                    lw=1.6, label=f"{method}")
        ax.set_title(strat, fontsize=9)
        ax.set_xlabel("FPR", fontsize=8)
        ax.set_ylabel("TPR", fontsize=8)
        ax.legend(fontsize=7)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Fig. 8 – ROC Curves Under Obfuscation Strategies",
                 fontsize=11)
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ── Ablation bar chart ──────────────────────────────────────────────────────

def plot_ablation(ablation_data: Dict[str, List[float]],
                   x_labels: List[str],
                   title: str = "Ablation Study (F1 Score)",
                   save_path: Optional[str] = None) -> None:
    """
    Generic ablation bar chart matching paper's Table VIII / XI format.

    ablation_data: { 'variant_name': [f1_N1, f1_N3, f1_N5, f1_N10] }
    x_labels: e.g. ['N=1', 'N=3', 'N=5', 'N=10']
    """
    variants = list(ablation_data.keys())
    x = np.arange(len(x_labels))
    width = 0.8 / len(variants)
    offsets = np.linspace(-(len(variants) - 1) * width / 2,
                           (len(variants) - 1) * width / 2,
                           len(variants))

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (var, col) in enumerate(zip(variants,
                                        CLASS_COLORS[:len(variants)])):
        vals = ablation_data[var]
        ax.bar(x + offsets[i], vals, width, label=var, color=col,
               edgecolor="black", linewidth=0.5, alpha=0.8)

    ax.set_xlabel("Number of Training Samples per Class", fontsize=10)
    ax.set_ylabel("F1 Score (%)", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ── Training curve ──────────────────────────────────────────────────────────

def plot_training_curves(train_losses: List[float],
                          val_metrics: Optional[Dict[str, List[float]]] = None,
                          title: str = "Pre-training Loss",
                          save_path: Optional[str] = None) -> None:
    """Plot contrastive pre-training loss curve and optional val metrics."""
    fig, axes = plt.subplots(1, 2 if val_metrics else 1, figsize=(10, 4))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    axes[0].plot(train_losses, color=COLORS["SmartDetector"], lw=1.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Contrastive Loss")
    axes[0].set_title(title)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    if val_metrics and len(axes) > 1:
        for metric_name, vals in val_metrics.items():
            axes[1].plot(vals, label=metric_name, lw=1.6)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Score (%)")
        axes[1].set_title("Validation Metrics")
        axes[1].legend()
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)

    plt.tight_layout()
    _save_or_show(fig, save_path)


# ── Internal helper ─────────────────────────────────────────────────────────

def _save_or_show(fig: plt.Figure, path: Optional[str]) -> None:
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Graphs] Saved → {path}")
    else:
        plt.show()
    plt.close(fig)


# ── CLI – generate all demo plots ──────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="SmartDetector – Graphs module")
    p.add_argument("--out_dir",  type=str, default=FIGURES_DIR)
    p.add_argument("--data_dir", type=str, default=SAMS_DIR,
                   help="Directory with SAMs/embeddings for t-SNE")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(42)

    print("=" * 60)
    print("SmartDetector – Graphs Module (Demo)")
    print("=" * 60)

    # ── Fig. 2: KDE demo ───────────────────────────────────────────────────
    print("\n[1] KDE plots (Fig. 2) …")
    benign_feats = {
        "length": rng.uniform(50,   1400, 2000),
        "ratio":  rng.normal(1.0,   0.2,  2000),
        "iat":    rng.exponential(1e4, 2000),
    }
    mal_feats = {
        "length": rng.normal(200,  30, 2000).clip(10, 1500),
        "ratio":  np.concatenate([rng.normal(0.1, 0.05, 1000),
                                   rng.normal(3.0, 0.3,  1000)]),
        "iat":    rng.exponential(200, 2000),
    }
    plot_kde_features(benign_feats, mal_feats,
                      "CIC-IDS-2017 (Demo)",
                      save_path=os.path.join(args.out_dir, "fig2_kde.png"))

    # ── Fig. 5: t-SNE demo ─────────────────────────────────────────────────
    print("[2] t-SNE plots (Fig. 5) …")
    n = 300
    class_names = ["Normal", "FTP-Patator", "DoS", "SSH-Patator"]
    reps  = np.vstack([rng.normal([i * 3, i * 2], 0.8, (n, 2))
                       for i in range(len(class_names))])
    lbls  = np.repeat(np.arange(len(class_names)), n)
    plot_tsne(reps, lbls, class_names,
              title="(a) Input to Encoder",
              save_path=os.path.join(args.out_dir, "fig5_tsne_input.png"))

    reps2 = np.vstack([rng.normal([i * 8, i * 6], 0.5, (n, 2))
                       for i in range(len(class_names))])
    plot_tsne(reps2, lbls, class_names,
              title="(b) Output of Encoder",
              save_path=os.path.join(args.out_dir, "fig5_tsne_output.png"))

    # ── Fig. 6: Euclidean distance demo ───────────────────────────────────
    print("[3] Euclidean distance chart (Fig. 6) …")
    dist_data = {
        "IDP": {"Color Image": 0.47, "Direction Sequence": 0.46,
                "Traffic Graph": 0.51, "SAM": 0.60},
        "IBP": {"Color Image": 0.40, "Direction Sequence": 0.34,
                "Traffic Graph": 0.34, "SAM": 0.59},
        "APR": {"Color Image": 0.41, "Direction Sequence": 0.55,
                "Traffic Graph": 0.39, "SAM": 0.64},
        "INP": {"Color Image": 0.38, "Direction Sequence": 0.42,
                "Traffic Graph": 0.38, "SAM": 0.55},
    }
    plot_euclidean_distances(dist_data,
        save_path=os.path.join(args.out_dir, "fig6_distances.png"))

    # ── Fig. 7: Imbalanced dataset demo ───────────────────────────────────
    print("[4] Imbalanced dataset F1 chart (Fig. 7) …")
    f1_data = {
        "FC-Net":        {"4:1": 86.31, "24:1": 84.99, "49:1": 43.51},
        "TF":            {"4:1": 97.20, "24:1": 84.42, "49:1": 77.14},
        "ST-Graph":      {"4:1": 96.65, "24:1": 79.58, "49:1": 79.54},
        "SmartDetector": {"4:1": 97.90, "24:1": 96.65, "49:1": 96.16},
    }
    plot_imbalanced_f1(f1_data,
        save_path=os.path.join(args.out_dir, "fig7_imbalanced_f1.png"))

    # ── Fig. 8: ROC curves demo ────────────────────────────────────────────
    print("[5] ROC curves (Fig. 8) …")
    fpr_b = np.linspace(0, 1, 100)
    roc_panel = {}
    for strat, tpr_shift in [("IDP-30%", 0.85), ("IBP-50%", 0.80),
                               ("APR", 0.90), ("INP", 0.92)]:
        roc_panel[strat] = {}
        for method, quality in [("FC-Net", 0.65), ("TF", 0.70),
                                  ("ST-Graph", 0.72), ("SmartDetector", 0.96)]:
            tpr = np.clip(fpr_b ** (1.0 / (quality * 3)), 0, 1)
            roc_panel[strat][method] = {
                "fpr": fpr_b.tolist(),
                "tpr": tpr.tolist(),
                "auc": float(np.trapz(tpr, fpr_b))
            }
    plot_all_roc_panels(roc_panel,
        save_path=os.path.join(args.out_dir, "fig8_roc_panels.png"))

    # ── Ablation demo ──────────────────────────────────────────────────────
    print("[6] Ablation study chart …")
    abl_data = {
        "w/o Encoder":    [0.0,  0.0,  0.0,  0.0],
        "w/o Embedding":  [60.8, 63.9, 72.8, 74.1],
        "SmartDetector-T":[48.3, 65.9, 59.2, 52.7],
        "SmartDetector":  [90.4, 95.8, 97.9, 98.6],
    }
    plot_ablation(abl_data,
                  x_labels=["N=1", "N=3", "N=5", "N=10"],
                  title="Table VIII – Ablation Study",
                  save_path=os.path.join(args.out_dir, "ablation_f1.png"))

    print(f"\n[DONE] All figures saved to '{args.out_dir}/'")


if __name__ == "__main__":
    main()