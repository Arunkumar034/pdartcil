"""
dataset.py – Flow Construction and Dataset Management
=======================================================
Reference: SmartDetector paper, Section IV-A, Section VI-A
Dataset: USTC-TFC (D4) – Htbot, Neris, Miuref, Virut (malware) + Benign

Input format: raw PCAP files organised per class:
    <data_dir>/
        Benign/   *.pcap  (or *.pcapng)
        Htbot/    *.pcap
        Neris/    *.pcap
        Miuref/   *.pcap
        Virut/    *.pcap

Implements:
- 5-tuple bidirectional flow identification (src_ip, dst_ip, src_port, dst_port, proto)
- Bidirectional flow merging (A→B and B→A in same flow)
- Timestamp sorting per flow
- K=40 packet truncation/padding (Section VI-B)
- Few-shot splits: N ∈ {1,3,5,10} (Section VI-C, Table VI)
- Imbalanced splits: β ∈ {4:1, 24:1, 49:1} (Section VI-D, Fig. 7)
- Cross-dataset generalization support
- Reproducibility via seed control (100-run averaging, Section VI-A)

Requires: scapy  (pip install scapy)
"""

import os
import json
import random
import argparse
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

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




# ── Paper constants (Section VI-B) ─────────────────────────────────────────
K = 40          # Number of packets extracted per flow
BENIGN_LABEL = 0
MALWARE_CLASSES = ["Htbot", "Neris", "Miuref", "Virut"]
ALL_CLASSES = ["Benign"] + MALWARE_CLASSES
CLASS_TO_IDX = {cls: i for i, cls in enumerate(ALL_CLASSES)}

# Pre-training set size (Section VI-A): 10,000 samples, benign:malicious = 4:1
PRETRAIN_TOTAL = 10_000
PRETRAIN_BENIGN = 8_000
PRETRAIN_PER_ATTACK = 500   # 500 × known attack types

DEFAULT_SEED = 42


# ── Utility ────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Fix random seeds for full reproducibility across all runs."""
    random.seed(seed)
    np.random.seed(seed)


def canonical_5tuple(src_ip: str, dst_ip: str,
                     src_port: int, dst_port: int,
                     proto: int) -> Tuple:
    """
    Return a canonical (sorted) 5-tuple key for bidirectional flow merging.

    Per Section IV-A: "A bidirectional flow represents a traffic entity
    that includes all packets with the same 5-tuple (source and destination
    IP addresses, source and destination ports, and transport protocol)."

    Both (A→B) and (B→A) packets are merged into the same flow by
    normalising the IP/port pair so the lexicographically smaller
    (ip, port) always comes first.
    """
    ep1 = (src_ip, src_port)
    ep2 = (dst_ip, dst_port)
    if ep1 <= ep2:
        return (src_ip, dst_ip, src_port, dst_port, proto)
    else:
        return (dst_ip, src_ip, dst_port, src_port, proto)


def determine_direction(src_ip: str, dst_ip: str,
                        src_port: int, dst_port: int,
                        canonical_key: Tuple) -> int:
    """
    Assign direction label: +1 (upstream) or -1 (downstream).

    Per Algorithm 1 (line 15-16): direction d ∈ {+1, −1}.
    The "first" endpoint in the canonical key is treated as the
    upstream (client) side.
    """
    # canonical_key = (ip_a, ip_b, port_a, port_b, proto)
    if src_ip == canonical_key[0] and src_port == canonical_key[2]:
        return 1    # upstream: matches the canonical "first" endpoint
    else:
        return -1   # downstream


# ── Packet & Flow data structures ──────────────────────────────────────────

class Packet:
    """
    Single packet representation (before SAM embedding).
    Stores the three raw features used in SAM (Section IV-B).
    """
    __slots__ = ("length", "direction", "iat", "timestamp")

    def __init__(self, length: int, direction: int,
                 iat: float, timestamp: float):
        self.length = length        # z  (bytes)
        self.direction = direction  # d  ∈ {+1, -1}
        self.iat = iat              # a  (seconds)
        self.timestamp = timestamp


class TrafficFlow:
    """
    A bidirectional flow with its K-packet window and class label.
    Corresponds to one row in the dataset fed to SAM construction.
    """
    def __init__(self, key: Tuple, label: int, class_name: str):
        self.key = key              # canonical 5-tuple
        self.label = label          # integer class index
        self.class_name = class_name
        self.packets: List[Packet] = []

    def add_packet(self, pkt: Packet) -> None:
        self.packets.append(pkt)

    def sort_by_time(self) -> None:
        """Sort packets by timestamp and recompute IATs."""
        self.packets.sort(key=lambda p: p.timestamp)
        for i in range(1, len(self.packets)):
            self.packets[i].iat = max(
                0.0,
                self.packets[i].timestamp - self.packets[i - 1].timestamp
            )
        if self.packets:
            self.packets[0].iat = 0.0

    def truncate_or_pad(self, k: int = K) -> List[Packet]:
        """
        Return exactly K packets.
        Per Algorithm 1 (lines 1-4): truncate if L ≥ K, pad with zeros if L < K.
        """
        if len(self.packets) >= k:
            return self.packets[:k]
        else:
            pad_count = k - len(self.packets)
            zero_pkt = Packet(length=0, direction=0, iat=0.0, timestamp=0.0)
            return self.packets + [zero_pkt] * pad_count

    def to_raw_matrix(self, k: int = K) -> np.ndarray:
        """
        Build Per-Packet Feature Matrix M ∈ ℝ^{3 × K} (Algorithm 1, lines 6-10).
        Row 0: packet lengths, Row 1: directions, Row 2: IATs.
        """
        pkts = self.truncate_or_pad(k)
        M = np.zeros((3, k), dtype=np.float32)
        for j, pkt in enumerate(pkts):
            M[0, j] = pkt.length    # z_j
            M[1, j] = pkt.direction # d_j
            M[2, j] = pkt.iat       # a_j
        return M


# ── PCAP Loader ────────────────────────────────────────────────────────────

def _parse_pcap_packet(pkt, class_name: str,
                        flows: Dict[Tuple, "TrafficFlow"]) -> None:
    """
    Parse one scapy packet and add it to the appropriate bidirectional flow.

    Supports IPv4/IPv6 over TCP, UDP, and ICMP.
    Non-IP or fragmented packets are silently skipped.
    """
    try:
        from scapy.layers.inet  import IP,  TCP,  UDP,  ICMP
        from scapy.layers.inet6 import IPv6
    except ImportError:
        raise ImportError("scapy is required: pip install scapy")

    # ── Extract IP layer ───────────────────────────────────────────────────
    if pkt.haslayer(IP):
        ip_layer = pkt[IP]
        src_ip, dst_ip = ip_layer.src, ip_layer.dst
        proto_num = ip_layer.proto          # 6=TCP, 17=UDP, 1=ICMP
    elif pkt.haslayer(IPv6):
        ip_layer = pkt[IPv6]
        src_ip, dst_ip = ip_layer.src, ip_layer.dst
        proto_num = ip_layer.nh
    else:
        return  # non-IP packet – skip

    # ── Extract transport ports ────────────────────────────────────────────
    if pkt.haslayer(TCP):
        src_port = int(pkt[TCP].sport)
        dst_port = int(pkt[TCP].dport)
    elif pkt.haslayer(UDP):
        src_port = int(pkt[UDP].sport)
        dst_port = int(pkt[UDP].dport)
    else:
        # ICMP or other: use type/code as pseudo-ports so flows group correctly
        src_port, dst_port = 0, 0

    # ── Packet length and timestamp ────────────────────────────────────────
    length = min(len(pkt), 1500)            # network MTU cap (Section V-B)
    ts     = float(pkt.time)               # seconds (float with μs precision)

    # ── Bidirectional flow key (canonical 5-tuple) ─────────────────────────
    key       = canonical_5tuple(src_ip, dst_ip, src_port, dst_port, proto_num)
    direction = determine_direction(src_ip, dst_ip, src_port, dst_port, key)
    label     = CLASS_TO_IDX.get(class_name, -1)

    if key not in flows:
        flows[key] = TrafficFlow(key, label, class_name)

    flows[key].add_packet(Packet(length, direction, iat=0.0, timestamp=ts))


def load_flows_from_pcap(pcap_path: str,
                          class_name: str,
                          max_flows: Optional[int] = None) -> List[TrafficFlow]:
    """
    Load bidirectional traffic flows from a single PCAP or PCAPNG file.

    Process:
    1. Read all packets with scapy (supports .pcap and .pcapng)
    2. Group by canonical 5-tuple → bidirectional flows
    3. Sort each flow's packets by timestamp
    4. Recompute IAT between consecutive packets
    5. Truncate to max_flows if specified

    Args:
        pcap_path  : absolute path to the .pcap / .pcapng file
        class_name : one of ALL_CLASSES (Benign/Htbot/Neris/Miuref/Virut)
        max_flows  : optional cap on number of flows returned

    Returns:
        List of TrafficFlow objects ready for SAM construction
    """
    try:
        from scapy.all import rdpcap, PcapReader
    except ImportError:
        raise ImportError("scapy is required: pip install scapy")

    flows: Dict[Tuple, TrafficFlow] = {}

    # Use PcapReader (streaming) to avoid loading the whole file into RAM
    try:
        with PcapReader(pcap_path) as reader:
            for pkt in reader:
                _parse_pcap_packet(pkt, class_name, flows)
                # Early-stop at 10× max_flows packets to keep RAM bounded
                if max_flows and len(flows) >= max_flows * 10:
                    break
    except Exception as e:
        # Fallback: rdpcap for malformed/truncated files
        print(f"  [WARN] PcapReader failed ({e}), falling back to rdpcap …")
        try:
            pkts = rdpcap(pcap_path)
            for pkt in pkts:
                _parse_pcap_packet(pkt, class_name, flows)
        except Exception as e2:
            print(f"  [ERROR] Could not read {pcap_path}: {e2}")
            return []

    # Finalise flows: sort by timestamp, recompute IATs
    result: List[TrafficFlow] = []
    for flow in flows.values():
        if len(flow.packets) == 0:
            continue
        flow.sort_by_time()
        result.append(flow)
        if max_flows and len(result) >= max_flows:
            break

    return result


def load_flows_from_directory(data_dir: str,
                               class_names: Optional[List[str]] = None,
                               max_per_class: int = 2000) -> List[TrafficFlow]:
    """
    Load all flows from the USTC-TFC PCAP directory structure.

    Expected layout:
        <data_dir>/
            Benign/   *.pcap  (also matches *.pcapng)
            Htbot/    *.pcap
            Neris/    *.pcap
            Miuref/   *.pcap
            Virut/    *.pcap

    Per Section VI-A / Table III:
    - Up to 2,000 flows per malicious class
    - Multiple PCAP files per class are processed in sorted order
    - Loading stops once max_per_class flows are collected for that class

    Args:
        data_dir      : root path, e.g. r"E:/Research paper/r1/USTC-TFC"
        class_names   : subset of ALL_CLASSES to load (default: all five)
        max_per_class : maximum flows collected per class

    Returns:
        Combined list of TrafficFlow objects from all classes
    """
    if class_names is None:
        class_names = ALL_CLASSES

    all_flows: List[TrafficFlow] = []

    for cls in class_names:
        cls_dir = Path(data_dir) / cls
        if not cls_dir.exists():
            print(f"[WARN] Class directory not found: {cls_dir}")
            continue

        # Collect .pcap and .pcapng files, sorted for reproducibility
        pcap_files = sorted(
            list(cls_dir.glob("*.pcap")) +
            list(cls_dir.glob("*.pcapng"))
        )

        if not pcap_files:
            print(f"[WARN] No PCAP files found in: {cls_dir}")
            continue

        cls_flows: List[TrafficFlow] = []
        for pcap_file in pcap_files:
            remaining = max_per_class - len(cls_flows)
            if remaining <= 0:
                break
            print(f"  Reading {pcap_file.name} ({cls}) …", end=" ", flush=True)
            new_flows = load_flows_from_pcap(str(pcap_file), cls, remaining)
            cls_flows.extend(new_flows)
            print(f"{len(new_flows)} flows  [total {len(cls_flows)}]")

        print(f"  ✓ {cls:<12}: {len(cls_flows):>5} flows loaded")
        all_flows.extend(cls_flows)

    return all_flows


# ── Synthetic / dummy data generator (for unit-testing without real data) ──

def generate_synthetic_flows(n_per_class: int = 200,
                              seed: int = DEFAULT_SEED,
                              k: int = K) -> List[TrafficFlow]:
    """
    Generate synthetic TrafficFlow objects for testing the pipeline
    without real PCAP/CSV data.

    Length/IAT distributions roughly match Fig. 2 of the paper:
    - Malicious: concentrated packet lengths, sparse IATs
    - Benign: dispersed lengths, smooth IATs
    """
    set_seed(seed)
    flows: List[TrafficFlow] = []

    for cls_idx, cls_name in enumerate(ALL_CLASSES):
        is_malicious = cls_idx > 0
        for _ in range(n_per_class):
            key = (f"192.168.{cls_idx}.{random.randint(1,254)}",
                   f"10.0.{cls_idx}.{random.randint(1,254)}",
                   random.randint(1024, 65535),
                   random.choice([80, 443, 8080]),
                   6)
            flow = TrafficFlow(key, cls_idx, cls_name)
            ts = 0.0
            for j in range(k):
                if is_malicious:
                    # Malicious: concentrated lengths (Fig. 2)
                    length = int(np.random.normal(200, 30))
                    iat = abs(float(np.random.exponential(0.01)))  # sparse
                else:
                    # Benign: dispersed lengths
                    length = int(np.random.uniform(50, 1400))
                    iat = abs(float(np.random.exponential(0.1)))   # smooth

                length = max(1, min(length, 1500))
                direction = random.choice([1, -1])
                ts += iat
                flow.add_packet(Packet(length, direction, iat, ts))
            flow.sort_by_time()
            flows.append(flow)

    random.shuffle(flows)
    return flows


# ── Dataset split utilities ────────────────────────────────────────────────

def split_few_shot(flows: List[TrafficFlow],
                   new_attack_class: str,
                   n_shot: int,
                   seed: int = DEFAULT_SEED) -> Dict[str, List[TrafficFlow]]:
    """
    Few-shot split: one attack type held out as "new" (Section VI-C).

    Protocol (Section VI-A, Validation):
    - known attacks + benign  → pre-training set (10,000 total, 4:1)
    - new attack samples      → N randomly selected for fine-tuning
    - remaining new attack    → evaluation set
    - repeated 100 times per N

    Returns dict with keys: 'pretrain', 'finetune', 'eval'
    """
    set_seed(seed)

    by_class: Dict[str, List[TrafficFlow]] = defaultdict(list)
    for f in flows:
        by_class[f.class_name].append(f)

    # Known attacks = all malware classes except the held-out one
    known_malware = [c for c in MALWARE_CLASSES if c != new_attack_class]

    # ── Pre-training set: 10,000 samples, benign:malicious = 4:1 ──────────
    pretrain_flows: List[TrafficFlow] = []
    benign_pool = random.sample(by_class["Benign"],
                                min(PRETRAIN_BENIGN, len(by_class["Benign"])))
    pretrain_flows.extend(benign_pool)

    per_known = PRETRAIN_PER_ATTACK
    for cls in known_malware:
        pool = random.sample(by_class[cls],
                             min(per_known, len(by_class[cls])))
        pretrain_flows.extend(pool)
    random.shuffle(pretrain_flows)

    # ── New attack split ───────────────────────────────────────────────────
    new_pool = by_class[new_attack_class].copy()
    # Cap at 2,000 per Table III protocol
    new_pool = random.sample(new_pool, min(2000, len(new_pool)))
    random.shuffle(new_pool)
    finetune_samples = new_pool[:n_shot]
    eval_samples     = new_pool[n_shot:]

    return {
        "pretrain": pretrain_flows,
        "finetune": finetune_samples,
        "eval":     eval_samples,
        "new_class": new_attack_class,
        "n_shot": n_shot
    }


def split_imbalanced(flows: List[TrafficFlow],
                     beta: int,
                     n_finetune: int = 5,
                     seed: int = DEFAULT_SEED) -> Dict[str, List[TrafficFlow]]:
    """
    Imbalanced dataset split with benign:malicious ratio β (Section VI-D).

    β ∈ {4, 24, 49} → pre-training ratio.
    Fine-tune: N=5 samples per class (fixed, per Section VI-D).
    """
    set_seed(seed)
    by_class: Dict[str, List[TrafficFlow]] = defaultdict(list)
    for f in flows:
        by_class[f.class_name].append(f)

    n_malicious_per_class = PRETRAIN_PER_ATTACK  # 500
    n_benign = beta * n_malicious_per_class * len(MALWARE_CLASSES)
    n_benign = min(n_benign, len(by_class["Benign"]))

    pretrain: List[TrafficFlow] = []
    pretrain.extend(random.sample(by_class["Benign"], n_benign))
    for cls in MALWARE_CLASSES:
        pretrain.extend(random.sample(by_class[cls],
                                      min(n_malicious_per_class,
                                          len(by_class[cls]))))
    random.shuffle(pretrain)

    # Fine-tune: N=5 per class (malware only, as in paper)
    finetune: List[TrafficFlow] = []
    eval_set: List[TrafficFlow] = []
    for cls in MALWARE_CLASSES:
        pool = random.sample(by_class[cls],
                             min(len(by_class[cls]), 2000))
        finetune.extend(pool[:n_finetune])
        eval_set.extend(pool[n_finetune:])

    return {
        "pretrain": pretrain,
        "finetune": finetune,
        "eval":     eval_set,
        "beta":     beta
    }


def build_multi_run_splits(flows: List[TrafficFlow],
                            n_shot: int,
                            new_attack_class: str,
                            n_runs: int = 100,
                            base_seed: int = DEFAULT_SEED
                            ) -> List[Dict]:
    """
    Generate 100 independent splits for robust averaging (Section VI-A).
    Each run uses a different seed for sample selection randomness.
    """
    splits = []
    for run_id in range(n_runs):
        seed = base_seed + run_id
        split = split_few_shot(flows, new_attack_class, n_shot, seed)
        split["run_id"] = run_id
        splits.append(split)
    return splits


# ── Persistence ────────────────────────────────────────────────────────────

def save_flows(flows: List[TrafficFlow], path: str) -> None:
    """Pickle a list of TrafficFlow objects for reuse."""
    with open(path, "wb") as fh:
        pickle.dump(flows, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[INFO] Saved {len(flows)} flows → {path}")


def load_flows(path: str) -> List[TrafficFlow]:
    """Load pickled TrafficFlow list."""
    with open(path, "rb") as fh:
        flows = pickle.load(fh)
    print(f"[INFO] Loaded {len(flows)} flows ← {path}")
    return flows


# ── CLI ────────────────────────────────────────────────────────────────────



def parse_args():
    p = argparse.ArgumentParser(
        description="SmartDetector – Dataset builder (Section IV-A / VI-A)")
    p.add_argument("--data_dir",    type=str,  default=DATA_DIR,
                   help="Root directory of USTC-TFC PCAPs")
    p.add_argument("--out_dir",     type=str,  default=FLOWS_DIR,
                   help="Output directory for serialised flows")
    p.add_argument("--mode",        type=str,
                   choices=["load", "synthetic"],
                   default="load",
                   help="'load' reads real PCAPs from data_dir; 'synthetic' generates dummy data")
    p.add_argument("--n_synthetic", type=int,  default=500,
                   help="Flows per class in synthetic mode")
    p.add_argument("--max_per_class", type=int, default=2000,
                   help="Max flows per class when loading real PCAP data")
    p.add_argument("--seed",        type=int,  default=DEFAULT_SEED)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("SmartDetector – Dataset Module")
    print(f"Output dir : {args.out_dir}")
    print("=" * 60)

    if args.mode == "synthetic":
        print(f"\n[MODE] Synthetic data ({args.n_synthetic} flows/class)")
        flows = generate_synthetic_flows(args.n_synthetic, args.seed)
    else:
        print(f"\n[MODE] Loading real PCAPs from: {args.data_dir}")
        flows = load_flows_from_directory(args.data_dir,
                                          max_per_class=args.max_per_class)

    # Summary
    by_class: Dict[str, List] = defaultdict(list)
    for f in flows:
        by_class[f.class_name].append(f)
    print("\n── Class distribution ──")
    for cls in ALL_CLASSES:
        print(f"  {cls:<12}: {len(by_class[cls]):>5} flows")

    # Save all flows
    save_flows(flows, os.path.join(args.out_dir, "all_flows.pkl"))

    # Example: few-shot split with N=5, Neris as new attack
    print("\n── Few-shot split example (N=5, new=Neris) ──")
    split = split_few_shot(flows, "Neris", n_shot=5, seed=args.seed)
    print(f"  Pre-train : {len(split['pretrain'])} flows")
    print(f"  Fine-tune : {len(split['finetune'])} flows")
    print(f"  Eval      : {len(split['eval'])} flows")

    # Save split
    with open(os.path.join(args.out_dir, "split_fewshot_N5_Neris.json"), "w") as fh:
        # Save only indices into all_flows for reproducibility
        all_keys = {id(f): i for i, f in enumerate(flows)}
        json.dump({
            "pretrain_size": len(split["pretrain"]),
            "finetune_size": len(split["finetune"]),
            "eval_size":     len(split["eval"]),
            "new_class":     split["new_class"],
            "n_shot":        split["n_shot"],
            "seed":          args.seed
        }, fh, indent=2)

    print(f"\n[DONE] Outputs saved to '{args.out_dir}/'")


if __name__ == "__main__":
    main()