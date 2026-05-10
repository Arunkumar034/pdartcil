"""
augmentation.py – Traffic Flow Augmentation (Algorithm 2)
==========================================================
Reference: SmartDetector paper, Section V-B, Algorithm 2
Figure 4: "Traffic Flow Augmentation" module

Implements four obfuscation strategies (Section VI-E):
  1. IDP – Inserting Dummy Packets        pd ∈ {10%, 20%, 30%}
  2. IBP – Inserting Benign Packets       pb ∈ {10%, 30%, 50%}
  3. APR – Altering Packet Rate           ±50% IAT variation
  4. INP – Inserting Noise Into Each Packet  50% noise probability

Algorithm 2 (pre-training augmentation):
  - Dummy packet insertion  probability q = 0.5
  - Random delay injection  probability r = 0.5
  - Packets ≤ 1500 bytes (MTU preservation)
  - Used to build (original SAM, augmented SAM) positive pairs for SimCLR
"""

import os
import copy
import random
import argparse
import numpy as np
from typing import List, Optional, Tuple
import sys
# ... other imports ...

# Import TrafficFlow and Packet classes needed for unpickling
try:
    from dataset import TrafficFlow, Packet
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dataset import TrafficFlow, Packet
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
K = 40          # Packets per flow  (Section VI-B)
MAX_PKT_LEN = 1500  # Network MTU (Section V-B: "packets generally ≤ 1500 bytes")
MAX_IAT_DELTA = 0.2  # Maximum random delay added (Algorithm 2, line 10)

# Algorithm 2 default probabilities
DEFAULT_Q = 0.5   # dummy packet insertion probability
DEFAULT_R = 0.5   # random delay injection probability

DEFAULT_SEED = 42


# ── Packet stub (mirrors dataset.Packet without circular import) ───────────

class PacketStub:
    """Lightweight packet representation for augmentation."""
    __slots__ = ("length", "direction", "iat", "timestamp")

    def __init__(self, length: int, direction: int,
                 iat: float, timestamp: float = 0.0):
        self.length    = length
        self.direction = direction
        self.iat       = iat
        self.timestamp = timestamp

    def copy(self) -> "PacketStub":
        return PacketStub(self.length, self.direction,
                          self.iat, self.timestamp)


def _pkts_from_flow(flow) -> List[PacketStub]:
    """Extract raw packet list from a TrafficFlow (or PacketStub list)."""
    if isinstance(flow, list):
        return [p.copy() for p in flow]
    pkts = flow.truncate_or_pad(K) if hasattr(flow, "truncate_or_pad") \
           else flow.packets
    return [PacketStub(p.length, p.direction, p.iat,
                       getattr(p, "timestamp", 0.0)) for p in pkts]


# ── Algorithm 2: Pre-training augmentation (Section V-B) ───────────────────

def augment_flow_algorithm2(flow,
                             q: float = DEFAULT_Q,
                             r: float = DEFAULT_R,
                             seed: Optional[int] = None) -> List[PacketStub]:
    """
    Algorithm 2 – Traffic Flow Augmentation.

    Input : flow T = {pkt_1, pkt_2, ..., pkt_L},
            dummy packet insertion probability q,
            random delay insertion probability r.
    Output: augmented flow T' of PacketStub objects.

    Lines 1-14 (Algorithm 2):
      For each packet pkt_i in T:
        (a) q' ~ U(0,1); if q' ≤ q → insert dummy packet p̂kt  (lines 3-7)
        (b) r' ~ U(0,1); if r' ≤ r → add random delay δ to IAT (lines 8-12)
        (c) Append pkt_i to T'  (line 13)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    orig_pkts = _pkts_from_flow(flow)
    T_prime: List[PacketStub] = []

    for pkt_i in orig_pkts:
        # ── (a) Dummy packet insertion  (Algorithm 2, lines 3-7) ──────────
        q_prime = random.uniform(0, 1)
        if q_prime <= q:
            dummy_len = random.randint(0, MAX_PKT_LEN)   # z' ∈ [0, 1500]
            dummy_dir = random.choice([-1, 1])            # d' ∈ {-1, +1}
            dummy_iat = random.uniform(0.0, 0.2)          # a' ∈ [0, 0.2]
            dummy_ts  = pkt_i.timestamp - dummy_iat
            T_prime.append(PacketStub(dummy_len, dummy_dir,
                                      dummy_iat, dummy_ts))

        # ── (b) Random delay injection  (Algorithm 2, lines 8-12) ─────────
        pkt_copy = pkt_i.copy()
        r_prime = random.uniform(0, 1)
        if r_prime <= r:
            delta = random.uniform(0.0, MAX_IAT_DELTA)   # δ ~ U(0, 0.2)
            pkt_copy.iat = pkt_copy.iat + delta            # a_i ← a_i + δ

        # ── (c) Append pkt_i (with possible delay) ─────────────────────────
        T_prime.append(pkt_copy)

    return T_prime   # Return T' (may be longer than K; truncation in SAM)


# ── Evasion Attack Strategies (Section VI-E, Table VII) ────────────────────

def strategy_idp(flow, pd: float = 0.1,
                 seed: Optional[int] = None) -> List[PacketStub]:
    """
    Strategy 1: Inserting Dummy Packets (IDP).

    Section VI-E: "the attacker can add a random dummy packet before
    each packet with probability pd ∈ {10%, 20%, 30%}."
    Dummy packets added to malicious traffic only.
    """
    if seed is not None:
        random.seed(seed)

    orig_pkts = _pkts_from_flow(flow)
    T_prime: List[PacketStub] = []

    for pkt in orig_pkts:
        if random.random() < pd:
            # Insert dummy packet before pkt
            dummy_len = random.randint(0, MAX_PKT_LEN)
            dummy_dir = random.choice([-1, 1])
            dummy_iat = random.uniform(0.0, 0.2)
            T_prime.append(PacketStub(dummy_len, dummy_dir,
                                      dummy_iat, pkt.timestamp))
        T_prime.append(pkt.copy())

    return T_prime


def strategy_ibp(flow, benign_pool: List[List[PacketStub]],
                 pb: float = 0.1,
                 seed: Optional[int] = None) -> List[PacketStub]:
    """
    Strategy 2: Inserting Benign Packets (IBP).

    Section VI-E: "the attackers insert a sequence of benign traffic
    packets into the malicious traffic. pb = ratio of inserted benign
    packets to total packets."

    benign_pool: list of benign packet sequences to sample from.
    """
    if seed is not None:
        random.seed(seed)

    orig_pkts = _pkts_from_flow(flow)
    total_original = len(orig_pkts)
    n_insert = max(1, int(total_original * pb / (1 - pb)))  # so ratio ≈ pb

    T_prime = [p.copy() for p in orig_pkts]
    if benign_pool:
        for _ in range(n_insert):
            benign_seq = random.choice(benign_pool)
            if benign_seq:
                benign_pkt = random.choice(benign_seq).copy()
                pos = random.randint(0, len(T_prime))
                T_prime.insert(pos, benign_pkt)

    return T_prime


def strategy_apr(flow, max_pct: float = 0.5,
                 seed: Optional[int] = None) -> List[PacketStub]:
    """
    Strategy 3: Altering Packet Rate (APR).

    Section VI-E: "we randomly adjust the IAT of each packet by either
    increasing or decreasing the delay by up to 50%."
    Only IAT (timing) is modified; packet length and direction unchanged.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    pkts = _pkts_from_flow(flow)
    for p in pkts:
        # ±max_pct of original IAT
        factor = random.uniform(1.0 - max_pct, 1.0 + max_pct)
        p.iat = max(0.0, p.iat * factor)
    return pkts


def strategy_inp(flow, noise_prob: float = 0.5,
                 seed: Optional[int] = None) -> List[PacketStub]:
    """
    Strategy 4: Inserting Noise Into Each Packet (INP).

    Section VI-E: "attackers insert random noise into each packet with
    probability 50%. The attackers append a random byte sequence to each
    packet, while ensuring that the length of the padded packets mimics
    the distribution of benign traffic."

    We simulate noise by randomly perturbing packet length to stay ≤ 1500.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    pkts = _pkts_from_flow(flow)
    for p in pkts:
        if random.random() < noise_prob:
            # Append random bytes: new_len = old_len + U(0, 1500 - old_len)
            max_add = max(0, MAX_PKT_LEN - int(p.length))
            noise_bytes = random.randint(0, max_add)
            p.length = int(p.length) + noise_bytes   # stays ≤ 1500
    return pkts


# ── Batch augmentation for contrastive learning (SimCLR positive pairs) ────

def make_positive_pairs(flows: List,
                         sam_constructor,
                         q: float = DEFAULT_Q,
                         r: float = DEFAULT_R,
                         seed: int = DEFAULT_SEED
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (original SAM, augmented SAM) positive pairs for SimCLR pre-training.

    Per Section V-A, Figure 4:
    - Original flow  → SAM  → encoder → s_i
    - Augmented flow → SAM  → encoder → s'_i
    - Contrastive loss maximises sim(s_i, s'_i)

    Returns two arrays, each of shape (N, 3, K, B):
      orig_sams:  SAMs of original flows
      aug_sams:   SAMs of augmented flows
    """
    rng = np.random.default_rng(seed)
    orig_sams = []
    aug_sams  = []

    for i, flow in enumerate(flows):
        run_seed = int(rng.integers(0, 2**31))

        # Original SAM
        orig_sam = sam_constructor.flow_to_sam(flow)
        orig_sams.append(orig_sam)

        # Augmented flow → augmented SAM
        aug_pkts = augment_flow_algorithm2(flow, q=q, r=r, seed=run_seed)

        # Convert augmented packet list to a pseudo-flow for SAM construction
        aug_flow = _AugmentedFlowProxy(aug_pkts, flow.label,
                                        getattr(flow, "class_name", ""))
        aug_sam = sam_constructor.flow_to_sam(aug_flow)
        aug_sams.append(aug_sam)

    return np.stack(orig_sams, 0), np.stack(aug_sams, 0)


class _AugmentedFlowProxy:
    """Minimal wrapper so augmented packet lists work with SAMConstructor."""
    def __init__(self, packets: List[PacketStub], label: int, class_name: str):
        self._packets = packets
        self.label = label
        self.class_name = class_name

    def truncate_or_pad(self, k: int = K) -> List[PacketStub]:
        if len(self._packets) >= k:
            return self._packets[:k]
        pad = PacketStub(0, 0, 0.0, 0.0)
        return self._packets + [pad] * (k - len(self._packets))

    @property
    def packets(self):
        return self._packets


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="SmartDetector – Traffic Augmentation (Section V-B, Algorithm 2)")
    p.add_argument("--flows_path", type=str, default=FLOWS_DIR + r"\all_flows.pkl")
    p.add_argument("--out_dir",    type=str, default=SAMS_DIR + r"\augmented")
    p.add_argument("--strategy",   type=str,
                   choices=["algorithm2", "idp", "ibp", "apr", "inp"],
                   default="algorithm2",
                   help="Augmentation strategy to apply")
    p.add_argument("--q",          type=float, default=DEFAULT_Q,
                   help="Dummy packet insertion probability (Algorithm 2)")
    p.add_argument("--r",          type=float, default=DEFAULT_R,
                   help="Random delay injection probability (Algorithm 2)")
    p.add_argument("--pd",         type=float, default=0.3,
                   help="IDP: dummy packet insertion probability")
    p.add_argument("--pb",         type=float, default=0.5,
                   help="IBP: benign packet insertion ratio")
    p.add_argument("--noise_prob", type=float, default=0.5,
                   help="INP: noise injection probability")
    p.add_argument("--seed",       type=int,   default=DEFAULT_SEED)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("SmartDetector – Traffic Augmentation Module")
    print("=" * 60)

    import pickle
    if os.path.exists(args.flows_path):
        with open(args.flows_path, "rb") as fh:
            flows = pickle.load(fh)
    else:
        print("[WARN] flows_path not found – using synthetic data")
        from dataset import generate_synthetic_flows
        flows = generate_synthetic_flows(50, args.seed)

    print(f"\n[INFO] Strategy: {args.strategy.upper()}")
    print(f"[INFO] Input flows: {len(flows)}")

    if args.strategy == "algorithm2":
        print(f"  q={args.q}, r={args.r}")
        augmented = [augment_flow_algorithm2(f, args.q, args.r, args.seed)
                     for f in flows[:20]]
        print(f"  Sample original length : {len(_pkts_from_flow(flows[0]))}")
        print(f"  Sample augmented length: {len(augmented[0])}")

    elif args.strategy == "idp":
        print(f"  pd={args.pd}")
        augmented = [strategy_idp(f, args.pd, args.seed) for f in flows[:20]]
        avg_increase = np.mean([len(a) - len(_pkts_from_flow(f))
                                for a, f in zip(augmented, flows[:20])])
        print(f"  Avg packet increase: {avg_increase:.1f}")

    elif args.strategy == "apr":
        augmented = [strategy_apr(f, seed=args.seed) for f in flows[:20]]
        print("  APR applied (IAT-only modification)")

    elif args.strategy == "inp":
        print(f"  noise_prob={args.noise_prob}")
        augmented = [strategy_inp(f, args.noise_prob, args.seed)
                     for f in flows[:20]]

    elif args.strategy == "ibp":
        # For IBP, we need benign flow packets
        benign_pool = [_pkts_from_flow(f) for f in flows
                       if getattr(f, "label", -1) == 0][:50]
        print(f"  pb={args.pb}, benign pool size: {len(benign_pool)}")
        augmented = [strategy_ibp(f, benign_pool, args.pb, args.seed)
                     for f in flows[:20]]

    print(f"\n[DONE] Augmentation complete. Example output stored in memory.")
    print("  (Use make_positive_pairs() in pretrain.py for full batch processing)")


if __name__ == "__main__":
    main()