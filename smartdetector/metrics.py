"""
metrics.py – Evaluation Metrics and Statistical Reporting
==========================================================
Reference: SmartDetector paper, Section VI-A (Performance Metrics)

Implements:
- Recall, F1 Score, AUC/ROC (Section VI-A, used in Tables VI-XI)
- 100-run averaging with standard deviation (Section VI-A: "each experiment
  is repeated 100 times")
- Per-run and aggregate reporting
- Min-max normalisation for Table V distance analysis
- Metrics tables formatted to match paper output (Tables VI, VII, VIII, XI)
"""

import os
import json
import argparse
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.metrics import (

    recall_score, f1_score, roc_auc_score,
    precision_score, confusion_matrix,
    roc_curve
)

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




# ── Single-run metrics ──────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_score: Optional[np.ndarray] = None,
                    average: str = "macro",
                    pos_label: int = 1) -> Dict[str, float]:
    """
    Compute Recall, F1, and AUC for one evaluation run.

    Per Section VI-A: "We select Recall, F1 score and the area under
    ROC curve (AUC) as the performance metrics."

    Args:
        y_true   : ground-truth integer labels, shape (N,)
        y_pred   : predicted integer labels,    shape (N,)
        y_score  : predicted probabilities or scores for positive class,
                   shape (N,) or (N, C).  Required for AUC.
        average  : sklearn averaging mode for multi-class ('macro'/'binary')
        pos_label: positive class index for binary AUC

    Returns:
        dict with keys: 'recall', 'f1', 'auc', 'precision'
    """
    results: Dict[str, float] = {}

    # ── Recall (Section VI-A, Table VI) ───────────────────────────────────
    results["recall"] = float(recall_score(
        y_true, y_pred,
        average=average,
        zero_division=0
    ) * 100.0)

    # ── F1 Score ───────────────────────────────────────────────────────────
    results["f1"] = float(f1_score(
        y_true, y_pred,
        average=average,
        zero_division=0
    ) * 100.0)

    # ── AUC (ROC) ──────────────────────────────────────────────────────────
    if y_score is not None:
        try:
            if y_score.ndim == 1:
                # Binary case
                results["auc"] = float(roc_auc_score(y_true, y_score) * 100.0)
            else:
                # Multi-class: one-vs-rest
                results["auc"] = float(roc_auc_score(
                    y_true, y_score,
                    multi_class="ovr",
                    average=average
                ) * 100.0)
        except (ValueError, TypeError):
            results["auc"] = 0.0
    else:
        results["auc"] = 0.0

    # ── Precision (for completeness) ──────────────────────────────────────
    results["precision"] = float(precision_score(
        y_true, y_pred,
        average=average,
        zero_division=0
    ) * 100.0)

    return results


def compute_roc_data(y_true: np.ndarray,
                      y_score: np.ndarray,
                      pos_label: int = 1
                      ) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute ROC curve data for plotting (Section VI-E, Fig. 8).

    Returns (fpr, tpr, auc_score).
    """
    # For multi-class, binarise: positive = any malicious class
    if y_score.ndim > 1:
        # Sum probabilities of all malicious classes
        score_1d = y_score[:, 1:].sum(axis=1)
        y_bin = (y_true > 0).astype(int)
    else:
        score_1d = y_score
        y_bin = y_true

    fpr, tpr, _ = roc_curve(y_bin, score_1d, pos_label=1)
    auc_val = float(roc_auc_score(y_bin, score_1d))
    return fpr, tpr, auc_val


# ── Multi-run aggregation (100 runs) ───────────────────────────────────────

class RunAccumulator:
    """
    Accumulates metrics across multiple runs and computes
    mean ± std (Section VI-A: "each experiment is repeated 100 times").
    """

    def __init__(self):
        self._runs: List[Dict[str, float]] = []

    def add_run(self, metrics: Dict[str, float]) -> None:
        self._runs.append(metrics)

    @property
    def n_runs(self) -> int:
        return len(self._runs)

    def aggregate(self) -> Dict[str, Dict[str, float]]:
        """
        Returns dict: { metric_name: { 'mean': ..., 'std': ...,
                                        'min':  ..., 'max': ... } }
        """
        if not self._runs:
            return {}
        keys = list(self._runs[0].keys())
        result = {}
        for k in keys:
            vals = np.array([r[k] for r in self._runs])
            result[k] = {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals)),
                "min":  float(np.min(vals)),
                "max":  float(np.max(vals)),
            }
        return result

    def summary_string(self) -> str:
        agg = self.aggregate()
        lines = [f"[Metrics over {self.n_runs} runs]"]
        for k, v in agg.items():
            lines.append(
                f"  {k:<12}: {v['mean']:>6.2f} ± {v['std']:>5.2f}  "
                f"[{v['min']:.2f}, {v['max']:.2f}]"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "n_runs":    self.n_runs,
            "runs":      self._runs,
            "aggregate": self.aggregate()
        }

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    def load(self, path: str) -> None:
        with open(path, "r") as fh:
            data = json.load(fh)
        self._runs = data["runs"]


# ── Table formatting (matching paper Tables VI-XI) ─────────────────────────

def format_table_vi(results: Dict[str, Dict[str, float]],
                     n_shot: int,
                     dataset_id: str = "D1") -> str:
    """
    Format results as Table VI (Few-shot learning results).

    columns: Recall | F1 | AUC  per method.
    """
    header = (f"\nTable VI (N={n_shot}, Dataset={dataset_id})\n"
              f"{'Method':<20} {'Recall':>8} {'F1':>8} {'AUC':>8}")
    rows = [header, "-" * 50]
    for method, m in results.items():
        rec = m.get("recall", 0.0)
        f1  = m.get("f1",     0.0)
        auc = m.get("auc",    0.0)
        if isinstance(rec, dict):
            rec = rec["mean"]
        if isinstance(f1, dict):
            f1 = f1["mean"]
        if isinstance(auc, dict):
            auc = auc["mean"]
        rows.append(f"{method:<20} {rec:>8.2f} {f1:>8.2f} {auc:>8.2f}")
    return "\n".join(rows)


def format_table_vii(results: Dict[str, Dict]) -> str:
    """
    Format results as Table VII (Obfuscation strategy detection).
    """
    header = ("\nTable VII – Detection under Obfuscation Strategies\n"
              f"{'Method':<16} {'Obfs':>6} {'p':>6} "
              f"{'Recall':>8} {'F1':>8} {'AUC':>8}")
    rows = [header, "-" * 60]
    for key, m in results.items():
        method, obfs, p = key
        rows.append(f"{method:<16} {obfs:>6} {str(p):>6} "
                    f"{m.get('recall',0):>8.2f} "
                    f"{m.get('f1',0):>8.2f} "
                    f"{m.get('auc',0):>8.2f}")
    return "\n".join(rows)


def format_table_xi(results: Dict[str, Dict]) -> str:
    """
    Format results as Table XI (Ablation study F1 scores).
    """
    header = ("\nTable XI – Ablation Study (F1 Scores)\n"
              f"{'Variant':<20} {'No Obfs':>8} {'IDP':>8} "
              f"{'IBP':>8} {'APR':>8} {'INP':>8} {'Full':>8}")
    rows = [header, "-" * 70]
    for variant, scenario_f1s in results.items():
        scores = [f"{scenario_f1s.get(s, 0):>8.2f}"
                  for s in ["no_obfs", "idp", "ibp", "apr", "inp", "full"]]
        rows.append(f"{variant:<20} {''.join(scores)}")
    return "\n".join(rows)


# ── Min-max normalisation (Table V) ────────────────────────────────────────

def minmax_normalise(values: Dict[str, float]) -> Dict[str, float]:
    """
    Per Section VI-B: "we take the average of all distances and use
    min-max normalization."
    """
    if not values:
        return {}
    arr = np.array(list(values.values()), dtype=np.float64)
    v_min, v_max = arr.min(), arr.max()
    if v_max == v_min:
        return {k: 0.0 for k in values}
    return {k: float((v - v_min) / (v_max - v_min))
            for k, v in values.items()}


def format_table_v(distances: Dict[str, Dict[str, float]]) -> str:
    """
    Format Table V: Euclidean distance between malicious and benign traffic
    under different representations.

    distances: { attack_class: { representation: distance } }
    """
    reps = ["Color Image", "Direction Sequence", "Traffic Graph", "SAM"]
    short = ["Color\nImage", "Direction\nSequence", "Traffic\nGraph", "SAM"]
    header = f"\nTable V – Euclidean Distance (normalised)\n"
    header += f"{'Attack':<16}" + "".join(f"{r:>12}" for r in reps)
    rows = [header, "-" * 64]
    all_vals = {r: [] for r in reps}

    for attack, rep_dist in distances.items():
        row = f"{attack:<16}"
        for r in reps:
            d = rep_dist.get(r, 0.0)
            all_vals[r].append(d)
            bold = "**" if r == "SAM" else "  "
            row += f"{bold}{d:>8.2f}{bold}"
        rows.append(row)

    # Average row
    avg_row = f"{'Average':<16}"
    for r in reps:
        avg = np.mean(all_vals[r]) if all_vals[r] else 0.0
        avg_row += f"  {avg:>8.2f}  "
    rows.append("-" * 64)
    rows.append(avg_row)
    return "\n".join(rows)


# ── Report saving ──────────────────────────────────────────────────────────

def save_metrics_report(results: dict, path: str) -> None:
    """Save full metrics report as JSON."""
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[Metrics] Report saved → {path}")


def load_metrics_report(path: str) -> dict:
    with open(path, "r") as fh:
        return json.load(fh)


# ── CLI – standalone evaluation ────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="SmartDetector – Metrics module (Section VI-A)")
    p.add_argument("--preds_path", type=str, default=None,
                   help="Path to .npz with y_true, y_pred, y_score arrays")
    p.add_argument("--n_runs",     type=int, default=100)
    p.add_argument("--out_dir",    type=str, default=RESULTS_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("SmartDetector – Metrics Module")
    print("=" * 60)

    if args.preds_path and os.path.exists(args.preds_path):
        data = np.load(args.preds_path)
        y_true  = data["y_true"]
        y_pred  = data["y_pred"]
        y_score = data["y_score"] if "y_score" in data else None
        m = compute_metrics(y_true, y_pred, y_score)
        print("\nSingle-run metrics:")
        for k, v in m.items():
            print(f"  {k:<12}: {v:.2f}%")
    else:
        # Demo: simulate 100 runs with synthetic data
        print("\n[DEMO] Simulating 100 runs with random predictions …")
        rng = np.random.default_rng(42)
        n_classes = 5
        n_samples = 2000
        acc = RunAccumulator()

        for run in range(args.n_runs):
            y_true  = rng.integers(0, n_classes, n_samples)
            y_pred  = rng.integers(0, n_classes, n_samples)
            y_score = rng.random((n_samples, n_classes))
            y_score /= y_score.sum(axis=1, keepdims=True)
            m = compute_metrics(y_true, y_pred, y_score, average="macro")
            acc.add_run(m)

        print(acc.summary_string())
        acc.save(os.path.join(args.out_dir, "demo_100runs.json"))

        # Demo table formatting
        demo_results = {
            "SmartDetector": {"recall": 95.25, "f1": 98.35, "auc": 95.72},
            "ST-Graph":       {"recall": 88.11, "f1": 73.49, "auc": 75.51},
            "FC-Net":         {"recall": 75.60, "f1": 80.83, "auc": 92.93},
            "TF":             {"recall": 85.92, "f1": 83.71, "auc": 90.44},
        }
        print(format_table_vi(demo_results, n_shot=1, dataset_id="D1"))

    print(f"\n[DONE] Reports saved to '{args.out_dir}/'")


if __name__ == "__main__":
    main()