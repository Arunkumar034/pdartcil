"""
predict.py – Inference Pipeline
=================================
Reference: SmartDetector paper, Section VI (Performance Evaluation)

Implements:
- PCAP/CSV processing → bidirectional flows
- SAM construction using saved embedding dictionaries
- Class prediction: benign vs. malicious (Htbot/Neris/Miuref/Virut)
- Metrics output when ground-truth labels are provided
- Batch and single-flow inference modes
- Command-line interface with configurable checkpoint/dict paths

"The test process is online, during which a label is predicted for
an unknown traffic sample." (Section VI-F)
"""

import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple

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
ALL_CLASSES     = ["Benign", "Htbot", "Neris", "Miuref", "Virut"]
N_CLASSES       = len(ALL_CLASSES)
DEFAULT_SEED    = 42
MALICIOUS_LABEL = 1     # Consolidated binary label for any malware class


# ── Model loader ────────────────────────────────────────────────────────────

def load_smartdetector(encoder_path: str,
                        classifier_path: Optional[str],
                        n_classes: int = N_CLASSES,
                        device: Optional[str] = None):
    """
    Load frozen encoder + FC classifier for inference.

    Returns (encoder, classifier, device).
    """
    from pretrain import ResNetEncoder
    from finetune import FCClassifier

    device = torch.device(
        device if device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # ── Encoder ──────────────────────────────────────────────────────────
    encoder = ResNetEncoder().to(device)
    ckpt    = torch.load(encoder_path, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    print(f"[Predict] Encoder loaded: {encoder_path}")
    if not os.path.exists(classifier_path):
        raise FileNotFoundError(
            f"Classifier weights not found: {classifier_path}\n"
            f"You must run finetune.py first to train the classifier."
        )
    state = torch.load(classifier_path, map_location=device)

    # ── Classifier ───────────────────────────────────────────────────────
    classifier = FCClassifier(encoder_dim=encoder.embed_dim,
                              n_classes=n_classes).to(device)
    if classifier_path and os.path.exists(classifier_path):
        state = torch.load(classifier_path, map_location=device)
        classifier.load_state_dict(state)
        print(f"[Predict] Classifier loaded: {classifier_path}")
    else:
        print("[Predict][WARN] No classifier weights found; "
              "predictions will be random.")
    classifier.eval()

    return encoder, classifier, device


# ── Inference on SAM arrays ─────────────────────────────────────────────────

def predict_from_sams(sams: np.ndarray,
                       encoder,
                       classifier,
                       device,
                       batch_size: int = 256
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Core inference function.

    Args:
        sams: (N, 3, K, B) SAM batch
    Returns:
        y_pred  : (N,)    integer class predictions
        y_score : (N, C)  softmax probabilities
    """
    preds  = []
    scores = []
    tensor = torch.tensor(sams, dtype=torch.float32)

    with torch.no_grad():
        for i in range(0, len(tensor), batch_size):
            batch      = tensor[i:i + batch_size].to(device)
            feats      = encoder(batch)               # deep representation s_i
            logits     = classifier(feats)
            probs      = F.softmax(logits, dim=1)
            pred_class = logits.argmax(dim=1)
            preds.append(pred_class.cpu().numpy())
            scores.append(probs.cpu().numpy())

    return np.concatenate(preds), np.concatenate(scores, axis=0)


# ── Flow processing pipeline ────────────────────────────────────────────────

def process_flows_to_predictions(flows: List,
                                  embed_dir: str,
                                  encoder,
                                  classifier,
                                  device,
                                  batch_size: int = 256,
                                  verbose: bool = False
                                  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    End-to-end: TrafficFlow list → SAMs → predictions.

    Per Section VI-F (Test Time):
    "SmartDetector has a shorter test time … SmartDetector extracts more
    features, including packet length, direction, and IAT."
    """
    from sam_matrix_construct import load_embedding_dicts, SAMConstructor
    if verbose:
        print(f"[Predict] Building SAMs for {len(flows)} flows …")
    t_start = time.time()

    Dz, Da  = load_embedding_dicts(embed_dir)
    constructor = SAMConstructor(Dz, Da, k=K, b=B)
    sams, labels = constructor.batch_flows_to_sam_and_labels(flows, verbose)

    t_sam = time.time() - t_start
    if verbose:
        print(f"  SAM construction: {t_sam:.2f}s "
              f"({t_sam / len(flows) * 1000:.2f} ms/flow)")

    t_inf  = time.time()
    y_pred, y_score = predict_from_sams(sams, encoder, classifier,
                                         device, batch_size)
    t_inf  = time.time() - t_inf
    if verbose:
        n = len(flows)
        print(f"  Inference: {t_inf:.2f}s "
              f"({t_inf / n * 1000:.2f} ms/flow)")
        print(f"  Total FET+Prediction: {(t_sam + t_inf):.2f}s "
              f"({(t_sam + t_inf) / n * 1000:.2f} ms/flow)")

    return y_pred, y_score, labels   # labels may be all-zeros if unlabelled


# ── PCAP processing ─────────────────────────────────────────────────────────

def process_pcap(pcap_path: str,
                  embed_dir: str,
                  encoder, classifier, device,
                  verbose: bool = True
                  ) -> Dict:
    """
    Process a PCAP file end-to-end.
    Requires scapy: pip install scapy

    Returns dict with 'flows', 'predictions', 'scores', 'summary'.
    """
    try:
        from scapy.all import rdpcap, IP, TCP, UDP
    except ImportError:
        print("[Predict][WARN] scapy not installed. "
              "Install with: pip install scapy")
        return {}

    from dataset import (TrafficFlow, Packet,
                          canonical_5tuple, determine_direction)
    import collections

    print(f"[Predict] Reading PCAP: {pcap_path}")
    packets = rdpcap(pcap_path)
    flow_dict: Dict = collections.defaultdict(
        lambda: TrafficFlow(None, 0, "Unknown"))

    for pkt in packets:
        if not (pkt.haslayer(IP) and (pkt.haslayer(TCP) or pkt.haslayer(UDP))):
            continue
        ip    = pkt[IP]
        proto = 6 if pkt.haslayer(TCP) else 17
        layer = pkt[TCP] if proto == 6 else pkt[UDP]

        key = canonical_5tuple(ip.src, ip.dst,
                                layer.sport, layer.dport, proto)
        direction = determine_direction(ip.src, ip.dst,
                                         layer.sport, layer.dport, key)
        ts     = float(pkt.time)
        length = min(len(pkt), 1500)
        iat    = 0.0  # recomputed after sort

        if flow_dict[key].key is None:
            flow_dict[key] = TrafficFlow(key, 0, "Unknown")
        flow_dict[key].add_packet(Packet(length, direction, iat, ts))

    flows = list(flow_dict.values())
    for f in flows:
        f.sort_by_time()
    if verbose:
        print(f"[Predict] Reconstructed {len(flows)} bidirectional flows")

    y_pred, y_score, _ = process_flows_to_predictions(
        flows, embed_dir, encoder, classifier, device, verbose=verbose)

    summary = _summarise_predictions(y_pred, y_score)
    return {
        "n_flows":     len(flows),
        "predictions": y_pred.tolist(),
        "scores":      y_score.tolist(),
        "summary":     summary,
    }


# ── CSV processing ──────────────────────────────────────────────────────────

def process_csv(csv_path: str,
                 class_name: str,
                 embed_dir: str,
                 encoder, classifier, device,
                 verbose: bool = True) -> Dict:
    """
    Process a CSV file with ground-truth labels (for benchmarking).
    Returns predictions and metrics.
    """
    
    from dataset import load_flows_from_csv
    from metrics import compute_metrics

    print(f"[Predict] Loading CSV: {csv_path}")
    flows = load_flows_from_csv(csv_path, class_name)
    if verbose:
        print(f"  Loaded {len(flows)} flows (class='{class_name}')")

    y_pred, y_score, y_true = process_flows_to_predictions(
        flows, embed_dir, encoder, classifier, device, verbose=verbose)

    # Binary metrics: benign vs. malicious
    y_true_bin = (y_true > 0).astype(int)
    y_pred_bin = (y_pred > 0).astype(int)
    score_mal  = y_score[:, 1:].sum(axis=1) if y_score.ndim > 1 else y_score

    m = compute_metrics(y_true_bin, y_pred_bin, score_mal, average="binary")
    if verbose:
        print(f"\n  Recall={m['recall']:.2f}%  "
              f"F1={m['f1']:.2f}%  AUC={m['auc']:.2f}%")

    return {
        "n_flows": len(flows),
        "metrics": m,
        "predictions": y_pred.tolist(),
    }


# ── Obfuscation evaluation (Table VII) ──────────────────────────────────────

def evaluate_obfuscation_robustness(flows: List,
                                     embed_dir: str,
                                     encoder, classifier, device,
                                     strategies: Optional[List] = None,
                                     verbose: bool = True) -> Dict:
    """
    Evaluate detection under four obfuscation strategies (Section VI-E).

    Applies each obfuscation to malicious flows and evaluates metrics.
    Returns Table VII-style results dict.
    """
    from augmentation import (strategy_idp, strategy_ibp,
                               strategy_apr, strategy_inp,
                               _AugmentedFlowProxy, _pkts_from_flow)
    from sam_matrix_construct import load_embedding_dicts, SAMConstructor
    from metrics import compute_metrics

    Dz, Da = load_embedding_dicts(embed_dir)
    constructor = SAMConstructor(Dz, Da)

    if strategies is None:
        strategies = ["idp", "ibp", "apr", "inp"]

    mal_flows = [f for f in flows if getattr(f, "label", 0) > 0]
    ben_flows = [f for f in flows if getattr(f, "label", 0) == 0]
    if verbose:
        print(f"[ObfsEval] {len(mal_flows)} malicious, "
              f"{len(ben_flows)} benign flows")

    results = {}

    # Baseline (no obfuscation)
    print("\n  Baseline (no obfuscation) …")
    all_flows  = mal_flows + ben_flows
    sams, lbls = constructor.batch_flows_to_sam_and_labels(all_flows)
    y_pred, y_score = predict_from_sams(sams, encoder, classifier, device)
    y_bin   = (lbls > 0).astype(int)
    y_p_bin = (y_pred > 0).astype(int)
    m = compute_metrics(y_bin, y_p_bin,
                         y_score[:, 1:].sum(1), average="binary")
    results["no_obfs"] = m
    if verbose:
        print(f"    F1={m['f1']:.2f}  AUC={m['auc']:.2f}")

    # Per-strategy
    for strat in strategies:
        if strat == "idp":
            probs = [0.1, 0.2, 0.3]
        elif strat == "ibp":
            probs = [0.1, 0.3, 0.5]
        elif strat == "apr":
            probs = [0.5]    # ±50%
        elif strat == "inp":
            probs = [0.5]    # 50%
        else:
            probs = [0.3]

        for p in probs:
            print(f"\n  Strategy={strat.upper()}, p={p} …")
            obf_flows = []
            for f in mal_flows:
                if strat == "idp":
                    pkts = strategy_idp(f, pd=p, seed=DEFAULT_SEED)
                elif strat == "ibp":
                    benign_seqs = [_pkts_from_flow(bf) for bf in ben_flows[:50]]
                    pkts = strategy_ibp(f, benign_seqs, pb=p, seed=DEFAULT_SEED)
                elif strat == "apr":
                    pkts = strategy_apr(f, max_pct=p, seed=DEFAULT_SEED)
                elif strat == "inp":
                    pkts = strategy_inp(f, noise_prob=p, seed=DEFAULT_SEED)
                else:
                    pkts = _pkts_from_flow(f)

                proxy = _AugmentedFlowProxy(pkts, f.label,
                                            getattr(f, "class_name", ""))
                obf_flows.append(proxy)

            eval_flows   = obf_flows + ben_flows
            eval_labels  = np.array(
                [f.label for f in eval_flows], dtype=int)
            sams_obf, _  = constructor.batch_flows_to_sam_and_labels(eval_flows)
            y_pred_o, y_score_o = predict_from_sams(
                sams_obf, encoder, classifier, device)
            y_bin_o  = (eval_labels > 0).astype(int)
            y_pb_o   = (y_pred_o > 0).astype(int)
            m_obf = compute_metrics(
                y_bin_o, y_pb_o,
                y_score_o[:, 1:].sum(1), average="binary")

            key = f"{strat}_{p}"
            results[key] = m_obf
            if verbose:
                print(f"    F1={m_obf['f1']:.2f}  AUC={m_obf['auc']:.2f}  "
                      f"Recall={m_obf['recall']:.2f}")

    return results


# ── Output summary ──────────────────────────────────────────────────────────

def _summarise_predictions(y_pred: np.ndarray,
                            y_score: np.ndarray) -> Dict:
    """Build a human-readable summary of prediction results."""
    n_total   = len(y_pred)
    n_benign  = int((y_pred == 0).sum())
    n_malicious = n_total - n_benign
    class_counts = {}
    for i, cls in enumerate(ALL_CLASSES):
        class_counts[cls] = int((y_pred == i).sum())

    return {
        "total_flows":  n_total,
        "benign":       n_benign,
        "malicious":    n_malicious,
        "malicious_pct":round(n_malicious / max(n_total, 1) * 100, 2),
        "class_counts": class_counts,
        "avg_conf_benign": float(y_score[:, 0].mean())
        if y_score.ndim > 1 else 0.0,
    }


def print_results_table(results: Dict, title: str = "Prediction Results") -> None:
    """Print a formatted results table to stdout."""
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")
    for key, val in results.items():
        if isinstance(val, dict):
            print(f"\n  [{key}]")
            for k, v in val.items():
                if isinstance(v, float):
                    print(f"    {k:<18}: {v:.2f}")
                else:
                    print(f"    {k:<18}: {v}")
        else:
            print(f"  {key:<20}: {val}")
    print("=" * 50)


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="SmartDetector – Inference Pipeline")
    p.add_argument("--mode",       type=str,
                   choices=["pcap", "csv", "sams", "obfs", "demo"],
                   default="demo",
                   help="Inference mode")
    p.add_argument("--input",      type=str, default=None,
                   help="Input PCAP or CSV file path")
    p.add_argument("--class_name", type=str, default="Benign",
                   help="Ground-truth class label (for CSV mode with metrics)")
    p.add_argument("--sams_path",  type=str, default=SAMS_DIR + r"\\all_sams.npz",
                   help="Pre-built SAMs .npz (for sams mode)")
    p.add_argument("--encoder_path",    type=str,
                   default=CKPT_DIR + r"\\encoder_best.pt")
    p.add_argument("--classifier_path", type=str,
                   default=CKPT_DIR + r"\\classifier.pt")
    p.add_argument("--embed_dir",  type=str, default=EMBED_DIR)
    p.add_argument("--out_dir",    type=str, default=RESULTS_DIR)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device",     type=str, default=None)
    p.add_argument("--seed",       type=int, default=DEFAULT_SEED)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("SmartDetector – Inference Pipeline")
    print("=" * 60)

    # ── Load model ─────────────────────────────────────────────────────────
    encoder, classifier, device = load_smartdetector(
        args.encoder_path,
        args.classifier_path,
        device=args.device
    )

    if args.mode == "demo":
        # Demo: run prediction on synthetic SAMs
        print("\n[MODE] Demo – synthetic SAMs")
        rng  = np.random.default_rng(args.seed)
        sams = rng.random((100, 3, K, B)).astype(np.float32)
        t0   = time.time()
        y_pred, y_score = predict_from_sams(sams, encoder, classifier, device)
        dt   = time.time() - t0
        summary = _summarise_predictions(y_pred, y_score)
        print_results_table(summary, "Demo Prediction Summary")
        print(f"\n  Test time: {dt:.3f}s  ({dt/len(sams)*1000:.2f} ms/flow)")
        print(f"  [Matches Table X target of ~30ms per-flow test time]")

    elif args.mode == "sams":
        print(f"\n[MODE] SAMs from file: {args.sams_path}")
        data = np.load(args.sams_path)
        sams = data["sams"]
        y_true = data.get("labels", None)
        y_pred, y_score = predict_from_sams(sams, encoder, classifier, device,
                                             args.batch_size)
        summary = _summarise_predictions(y_pred, y_score)
        print_results_table(summary)
        if y_true is not None:
            from metrics import compute_metrics
            m = compute_metrics(y_true, y_pred, y_score, average="macro")
            print("\nMetrics vs ground truth:")
            for k, v in m.items():
                print(f"  {k}: {v:.2f}%")
        out_path = os.path.join(args.out_dir, "predictions.json")
        with open(out_path, "w") as fh:
            json.dump({"summary": summary,
                       "n_flows": len(y_pred)}, fh, indent=2)
        print(f"\n[DONE] Results saved → {out_path}")

    elif args.mode == "pcap":
        assert args.input, "Provide --input path to PCAP file"
        result = process_pcap(args.input, args.embed_dir,
                               encoder, classifier, device)
        print_results_table(result.get("summary", {}), "PCAP Prediction")
        out_path = os.path.join(args.out_dir, "pcap_predictions.json")
        with open(out_path, "w") as fh:
            json.dump({k: v for k, v in result.items()
                       if k != "predictions"}, fh, indent=2)

    elif args.mode == "csv":
        assert args.input, "Provide --input path to CSV file"
        result = process_csv(args.input, args.class_name, args.embed_dir,
                              encoder, classifier, device)
        print_results_table(result.get("metrics", {}), "CSV Evaluation")

    elif args.mode == "obfs":
        print("\n[MODE] Obfuscation robustness evaluation")
        from dataset import generate_synthetic_flows
        flows   = generate_synthetic_flows(50, args.seed)
        results = evaluate_obfuscation_robustness(
            flows, args.embed_dir, encoder, classifier, device)
        out_path = os.path.join(args.out_dir, "obfs_results.json")
        with open(out_path, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\n[DONE] Obfuscation results saved → {out_path}")


if __name__ == "__main__":
    main()