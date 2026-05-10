"""
sam_matrix_construct.py – Semantic Attribute Matrix (SAM) Construction
=======================================================================
Reference: SmartDetector paper, Section IV-B, Algorithm 1
Equations: Eq. (1)–(4) – Word2Vec CBOW embedding

Implements:
- Feature extraction: packet length (z), direction (d), IAT (a)
- Word2Vec CBOW embedding for length and IAT (Eq. 1-4)
- Direct tiling B-dimensional vector for direction (Algorithm 1, line 15-16)
- Nearest-neighbor lookup for out-of-vocabulary values (Algorithm 1, line 20-21)
- Output: Semantic Attribute Matrix R ∈ ℝ^{3 × K × B}  (K=40, B=100)
- Embedding dictionaries Dz (length) and Da (IAT) built from background traffic
"""

import os
import pickle
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from gensim.models import Word2Vec


import os
import sys
import pickle
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from gensim.models import Word2Vec

# Import TrafficFlow and Packet classes needed for unpickling
try:
    from dataset import TrafficFlow, Packet
except ImportError:
    # If running from a different directory, add current dir to path
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
K = 40      # Number of packets per flow  (Section VI-B)
B = 100     # Embedding dimension          (Section VI-B)
CBOW_WINDOW = 5     # Context window size for CBOW
CBOW_MIN_COUNT = 1  # Minimum frequency for vocabulary entry
CBOW_EPOCHS = 100    # Training iterations for Word2Vec
DEFAULT_SEED = 42

# IAT quantisation: bin edges in seconds
# Covers 0 to 2s in fine steps matching Fig. 2 distributions
IAT_BIN_EDGES = (
    list(np.round(np.arange(0.0, 0.02, 0.001), 4)) +   # 0–0.02s  (fine)
    list(np.round(np.arange(0.02, 0.2,  0.01 ), 3)) +   # 0.02–0.2s
    list(np.round(np.arange(0.2,  2.0,  0.1  ), 2)) +   # 0.2–2s
    [2.0, 5.0, 10.0, 30.0, 60.0]                        # coarse tail
)
IAT_BIN_EDGES = sorted(set(IAT_BIN_EDGES))


def _quantise_iat(iat: float) -> float:
    """
    Map raw IAT to the nearest bin edge value.
    This reduces the vocabulary size for Word2Vec (Algorithm 1, lines 17-21).
    """
    if iat <= IAT_BIN_EDGES[0]:
        return IAT_BIN_EDGES[0]
    for edge in IAT_BIN_EDGES:
        if iat <= edge:
            return edge
    return IAT_BIN_EDGES[-1]


# ── Embedding Dictionary ────────────────────────────────────────────────────

class EmbeddingDictionary:
    """
    Attribute value dictionary D = {(S_1, W_1), ..., (S_V, W_V)}.
    Maps scalar attribute values to B-dimensional embedding vectors.

    Per Section IV-B: "we collect different attribute values of all
    packet lengths and construct attribute value dictionary Dz".
    """

    def __init__(self, embedding_dim: int = B):
        self.embedding_dim = embedding_dim
        self._keys: List[float] = []        # S_1 … S_V
        self._vectors: np.ndarray = None    # shape (V, B)

    @property
    def size(self) -> int:
        return len(self._keys)

    def build_from_word2vec(self, model: Word2Vec,
                             all_values: List[float]) -> None:
        """
        Extract embedding vectors from a trained Word2Vec model.
        Each attribute value S_k maps to embedding vector W_k (the
        weight matrix W row in Eq. 1).
        """
        self._keys = sorted(set(all_values))
        rows = []
        for val in self._keys:
            token = str(val)
            if token in model.wv:
                rows.append(model.wv[token])
            else:
                # Fallback: zero vector for unseen values
                rows.append(np.zeros(self.embedding_dim, dtype=np.float32))
        self._vectors = np.array(rows, dtype=np.float32)

    def lookup(self, value: float) -> np.ndarray:
        """
        Algorithm 1, lines 17-21: exact match, else nearest-neighbor.
        Returns embedding vector W_m for attribute value S_m.
        """
        if len(self._keys) == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        # Exact match
        token = str(value)
        try:
            idx = self._keys.index(value)
            return self._vectors[idx]
        except ValueError:
            pass

        # Nearest-neighbor: argmin |S_k - value|  (Algorithm 1, line 20)
        keys_arr = np.array(self._keys, dtype=np.float64)
        idx = int(np.argmin(np.abs(keys_arr - value)))
        return self._vectors[idx]

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump({"keys": self._keys,
                         "vectors": self._vectors,
                         "dim": self.embedding_dim}, fh)

    def load(self, path: str) -> None:
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        self._keys = data["keys"]
        self._vectors = data["vectors"]
        self.embedding_dim = data["dim"]


# ── Word2Vec CBOW training (Eq. 1–4) ───────────────────────────────────────

def _flows_to_token_sequences(flows,
                               feature: str) -> List[List[str]]:
    """
    Convert a list of TrafficFlow objects to Word2Vec token sequences.
    Each flow yields one sequence of tokens (attribute values as strings).

    Per Section IV-B: "we integrate the features from multiple flows
    into a set and employ Word2Vec for feature embedding."

    feature: one of 'length', 'iat'
    """
    sequences = []
    for flow in flows:
        pkts = flow.truncate_or_pad() if hasattr(flow, "truncate_or_pad") \
               else flow.packets
        seq = []
        for pkt in pkts:
            if feature == "length":
                seq.append(str(int(pkt.length)))
            elif feature == "iat":
                seq.append(str(_quantise_iat(pkt.iat)))
        sequences.append(seq)
    return sequences


def train_embedding_dictionaries(background_flows,
                                  embedding_dim: int = B,
                                  window: int = CBOW_WINDOW,
                                  min_count: int = CBOW_MIN_COUNT,
                                  epochs: int = CBOW_EPOCHS,
                                  seed: int = DEFAULT_SEED,
                                  save_dir: Optional[str] = None
                                  ) -> Tuple[EmbeddingDictionary,
                                             EmbeddingDictionary]:
    """
    Train Word2Vec CBOW models on background traffic and build Dz and Da.

    Per Section IV-B: "we collect 3,989,459 flows from the gateway of a
    campus network as background traffic."

    Returns (Dz, Da) – embedding dictionaries for length and IAT.

    Word2Vec CBOW: Eq. (1) h = (1/C) W^T * sum(x_i)
                   Eq. (2) u = W'^T * h
                   Eq. (3) y_k = exp(u_k) / sum_j exp(u_j)
                   Eq. (4) L = -sum_k t_k * log(y_k)
    Gensim's Word2Vec sg=0 implements exactly this CBOW objective.
    """
    print(f"[SAM] Training Word2Vec CBOW on {len(background_flows)} flows …")

    # ── Length embedding ───────────────────────────────────────────────────
    len_sequences = _flows_to_token_sequences(background_flows, "length")
    all_lengths = [float(tok) for seq in len_sequences for tok in seq]

    print(f"  Length vocabulary size: {len(set(all_lengths))}")
    model_len = Word2Vec(
        sentences=len_sequences,
        vector_size=embedding_dim,
        window=window,
        min_count=min_count,
        sg=0,          # CBOW (sg=0), Skip-gram (sg=1)
        seed=seed,
        workers=4,
        epochs=epochs
    )
    Dz = EmbeddingDictionary(embedding_dim)
    Dz.build_from_word2vec(model_len, all_lengths)
    print(f"  Dz built: {Dz.size} length entries")

    # ── IAT embedding ──────────────────────────────────────────────────────
    iat_sequences = _flows_to_token_sequences(background_flows, "iat")
    all_iats = [float(tok) for seq in iat_sequences for tok in seq]

    print(f"  IAT vocabulary size: {len(set(all_iats))}")
    model_iat = Word2Vec(
        sentences=iat_sequences,
        vector_size=embedding_dim,
        window=window,
        min_count=min_count,
        sg=0,
        seed=seed,
        workers=4,
        epochs=epochs
    )
    Da = EmbeddingDictionary(embedding_dim)
    Da.build_from_word2vec(model_iat, all_iats)
    print(f"  Da built: {Da.size} IAT entries")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        Dz.save(os.path.join(save_dir, "dict_length.pkl"))
        Da.save(os.path.join(save_dir, "dict_iat.pkl"))
        print(f"  Dictionaries saved to '{save_dir}/'")

    return Dz, Da


# ── SAM Construction (Algorithm 1) ─────────────────────────────────────────

class SAMConstructor:
    """
    Converts a TrafficFlow into a Semantic Attribute Matrix R ∈ ℝ^{3 × K × B}.

    Per Fig. 3 and Algorithm 1:
    - Step 1 (Feature Extraction): extract z, d, a from first K packets
    - Step 2 (Feature Embedding):
        * d  → direct tiling: [d] × B  (Algorithm 1, lines 15-16)
        * z  → Dz lookup / nearest-neighbor  (lines 17-21)
        * a  → Da lookup / nearest-neighbor  (lines 17-21)
    - Concatenate across features → R of shape (3, K, B)
    """

    def __init__(self,
                 Dz: EmbeddingDictionary,
                 Da: EmbeddingDictionary,
                 k: int = K,
                 b: int = B):
        self.Dz = Dz
        self.Da = Da
        self.k = k
        self.b = b

    # ── Algorithm 1 main procedure ─────────────────────────────────────────
    def flow_to_sam(self, flow) -> np.ndarray:
        """
        Algorithm 1: Feature Extraction and Embedding.

        Input : TrafficFlow with L packets, K, B, Dz, Da
        Output: R ∈ ℝ^{3 × K × B}  (Semantic Attribute Matrix)

        Lines 1-4  : truncate (L ≥ K) or pad with zeros (L < K)
        Lines 6-10 : extract per-packet (z_j, d_j, a_j)
        Lines 11-24: embed each feature type
        Line  24   : append R_i to R
        """
        # Step 1: truncate / pad  (Algorithm 1, lines 1-4)
        pkts = flow.truncate_or_pad(self.k) \
               if hasattr(flow, "truncate_or_pad") else flow.packets[:self.k]

        # Step 2: Per-packet Feature Matrix M ∈ ℝ^{3 × K}  (lines 6-10)
        M = np.zeros((3, self.k), dtype=np.float32)
        for j, pkt in enumerate(pkts):
            M[0, j] = float(pkt.length)
            M[1, j] = float(pkt.direction)
            M[2, j] = _quantise_iat(float(pkt.iat))

        # Step 3: Feature Embedding  (lines 11-24)
        R = np.zeros((3, self.k, self.b), dtype=np.float32)  # initialise R ← {}

        # i = d (direction): binary tiling  (lines 15-16)
        # "the binary value is directly tiled into a B-dimensional vector"
        for j in range(self.k):
            d_val = M[1, j]  # +1 or -1 or 0 (pad)
            R[1, j, :] = np.full(self.b, d_val)   # R_d[j] = [d] * B

        # i = z (packet length): Dz lookup  (lines 17-21)
        for j in range(self.k):
            z_val = M[0, j]
            R[0, j, :] = self.Dz.lookup(z_val)    # R_z[j] = W_{z}

        # i = a (IAT): Da lookup  (lines 17-21)
        for j in range(self.k):
            a_val = M[2, j]
            R[2, j, :] = self.Da.lookup(a_val)    # R_a[j] = W_{a}

        return R   # shape: (3, K, B)

    def batch_flows_to_sam(self,
                            flows: List,
                            verbose: bool = False) -> np.ndarray:
        """
        Convert a list of TrafficFlow objects to a batch of SAMs.
        Returns np.ndarray of shape (N, 3, K, B).
        """
        sams = []
        for i, flow in enumerate(flows):
            sams.append(self.flow_to_sam(flow))
            if verbose and (i + 1) % 500 == 0:
                print(f"  Embedded {i + 1}/{len(flows)} flows …")
        return np.stack(sams, axis=0)  # (N, 3, K, B)

    def batch_flows_to_sam_and_labels(self,
                                       flows: List,
                                       verbose: bool = False
                                       ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (SAMs, labels) where SAMs.shape = (N, 3, K, B)
        and labels.shape = (N,).
        """
        sams = self.batch_flows_to_sam(flows, verbose)
        labels = np.array([f.label for f in flows], dtype=np.int64)
        return sams, labels


# ── Persistence helpers ─────────────────────────────────────────────────────

def load_embedding_dicts(embed_dir: str
                          ) -> Tuple[EmbeddingDictionary, EmbeddingDictionary]:
    """Load pre-built Dz and Da from disk."""
    Dz = EmbeddingDictionary()
    Da = EmbeddingDictionary()
    Dz.load(os.path.join(embed_dir, "Dz.pkl"))
    Da.load(os.path.join(embed_dir, "Da.pkl"))
    return Dz, Da


def save_sams(sams: np.ndarray, labels: np.ndarray, path: str) -> None:
    """Save SAM batch and labels to a compressed .npz file."""
    np.savez_compressed(path, sams=sams, labels=labels)
    print(f"[SAM] Saved {len(sams)} SAMs → {path}.npz")


def load_sams(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load SAM batch and labels from a .npz file."""
    data = np.load(path if path.endswith(".npz") else path + ".npz")
    return data["sams"], data["labels"]


# ── Euclidean distance utility (for Table V analysis) ──────────────────────

def mean_euclidean_distance(sams_cls1: np.ndarray,
                             sams_cls2: np.ndarray,
                             n_sample: int = 500,
                             seed: int = DEFAULT_SEED) -> float:
    """
    Compute mean Euclidean distance between two classes of SAMs.

    Per Section VI-B: "we use the euclidean distance to assess the
    dissimilarity between malicious and benign traffic."
    Each SAM is flattened to a 1-D vector before distance computation.
    """
    rng = np.random.default_rng(seed)
    n1 = min(n_sample, len(sams_cls1))
    n2 = min(n_sample, len(sams_cls2))
    idx1 = rng.choice(len(sams_cls1), n1, replace=False)
    idx2 = rng.choice(len(sams_cls2), n2, replace=False)
    flat1 = sams_cls1[idx1].reshape(n1, -1)
    flat2 = sams_cls2[idx2].reshape(n2, -1)

    # Pairwise Euclidean distances (take random pairing for efficiency)
    n = min(n1, n2)
    dists = np.linalg.norm(flat1[:n] - flat2[:n], axis=1)
    return float(np.mean(dists))


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="SmartDetector – SAM Construction (Section IV-B, Algorithm 1)")
    p.add_argument("--flows_path",  type=str, default=FLOWS_DIR + r"\all_flows.pkl",
                   help="Path to pickled TrafficFlow list (from dataset.py)")
    p.add_argument("--embed_dir",   type=str, default=EMBED_DIR,
                   help="Directory for embedding dictionaries")
    p.add_argument("--out_dir",     type=str, default=SAMS_DIR,
                   help="Output directory for SAM .npz files")
    p.add_argument("--k",           type=int, default=K)
    p.add_argument("--b",           type=int, default=B)
    p.add_argument("--epochs",      type=int, default=CBOW_EPOCHS)
    p.add_argument("--seed",        type=int, default=DEFAULT_SEED)
    p.add_argument("--rebuild_dict", action="store_true",
                   help="Force rebuild of embedding dictionaries")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir,   exist_ok=True)
    os.makedirs(args.embed_dir, exist_ok=True)

    print("=" * 60)
    print("SmartDetector – SAM Construction Module")
    print("=" * 60)

    # ── Load flows ─────────────────────────────────────────────────────────
    if os.path.exists(args.flows_path):
        import pickle
        with open(args.flows_path, "rb") as fh:
            all_flows = pickle.load(fh)
        print(f"[INFO] Loaded {len(all_flows)} flows from {args.flows_path}")
    else:
        print("[WARN] flows_path not found – generating synthetic flows")
        from dataset import generate_synthetic_flows
        all_flows = generate_synthetic_flows(200, args.seed)

    # ── Build / load embedding dictionaries ───────────────────────────────
    dz_path = os.path.join(args.embed_dir, "dict_length.pkl")
    da_path = os.path.join(args.embed_dir, "dict_iat.pkl")

    if os.path.exists(dz_path) and os.path.exists(da_path) \
            and not args.rebuild_dict:
        print("[INFO] Loading existing embedding dictionaries …")
        Dz, Da = load_embedding_dicts(args.embed_dir)
    else:
        print("[INFO] Training Word2Vec CBOW dictionaries …")
        Dz, Da = train_embedding_dictionaries(
            all_flows,
            embedding_dim=args.b,
            epochs=args.epochs,
            seed=args.seed,
            save_dir=args.embed_dir
        )

    # ── Build SAMs ─────────────────────────────────────────────────────────
    constructor = SAMConstructor(Dz, Da, k=args.k, b=args.b)
    print(f"\n[INFO] Building SAMs for {len(all_flows)} flows …")
    sams, labels = constructor.batch_flows_to_sam_and_labels(all_flows,
                                                              verbose=True)
    print(f"[INFO] SAM batch shape: {sams.shape}")  # (N, 3, K, B)

    save_sams(sams, labels,
              os.path.join(args.out_dir, "all_sams"))

    # ── Euclidean distance analysis (Table V) ──────────────────────────────
    from dataset import ALL_CLASSES, MALWARE_CLASSES
    benign_mask = labels == 0
    print("\n── Euclidean distances (SAM vs Benign) ──")
    distances = {}
    for cls_idx, cls in enumerate(MALWARE_CLASSES, start=1):
        cls_mask = labels == cls_idx
        if cls_mask.sum() == 0:
            continue
        dist = mean_euclidean_distance(sams[cls_mask], sams[benign_mask])
        distances[cls] = dist
        print(f"  {cls:<12}: {dist:.4f}")
    avg = np.mean(list(distances.values()))
    print(f"  {'Average':<12}: {avg:.4f}")

    print(f"\n[DONE] SAMs saved to '{args.out_dir}/'")


if __name__ == "__main__":
    main()

# (r1) E:\Project R1> python "multimodal/multimodal_predict_from_pcap.py" --pcap_dir "F:\USTC-TK2016-master\USTC-TK2016-master\3_ProcessedSession\TrimedSession\Train\voipbuster_4b-ALL"
# =======================================================
#   Multimodal Prediction from PCAP
# =======================================================
# [MultimodalFusionNetwork] Initialised:
#   Visual Branch dim    : 512
#   Semantic Branch dim  : 2048
#   Fused dim            : 2560
#   Output classes       : 16
# [Model] Loading weights from E:\Project R1\checkpoints\multimodal_best.pth
# [*] Device: cuda
# Traceback (most recent call last):
#   File "E:\Project R1\multimodal\multimodal_predict_from_pcap.py", line 394, in <module>
#     main()
#   File "E:\Project R1\multimodal\multimodal_predict_from_pcap.py", line 361, in main
#     sam_constructor = load_sam_constructor(embed_dir=args.embed_dir)
#   File "E:\Project R1\multimodal\multimodal_predict_from_pcap.py", line 122, in load_sam_constructor
#     Dz, Da = load_embedding_dicts(embed_dir)
#   File "E:\Project R1\smartdetector\sam_matrix_construct.py", line 367, in load_embedding_dicts
#     Dz.load(os.path.join(embed_dir, "Dz.pkl"))
#   File "E:\Project R1\smartdetector\sam_matrix_construct.py", line 159, in load
#     self._keys = data["keys"]
# TypeError: 'EmbeddingDictionary' object is not subscriptable

# (r1) E:\Project R1>