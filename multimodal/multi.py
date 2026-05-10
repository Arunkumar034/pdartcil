"""
multi.py – Multimodal Late / Logit-Level Fusion Network
=============================================================
Architecture (from diagram):
  NETWORK TRAFFIC SOURCE
       |                         |
  MODALITY 1 (Visual)       MODALITY 2 (Semantic)
  28×28×1 greyscale image   3×40×100 SAM matrix
       |                         |
  BRANCH 1 (CIL Paper)      BRANCH 2 (SmartDetector)
  NAS-Optimised CNN +       Expandable CIL Module
       |                         |
  N-dim Feature Vector A    N-dim Feature Vector B
              \              /
           LATE FUSION BRIDGE
           (Concatenation → Fused Vector)
                    |
          FUSION CLASSIFIER HEAD
             (Fully Connected)
                    |
         FINAL CLASSIFICATION
       (DDoS-ICMP, VPN, Benign, …)
"""
#python "multimodal/multi.py" --mode train_fusion --data_dir "C:\Users\LENOVO\Downloads\processed_multimodal_dataset" --epochs 30


import os
import sys
import json
import time
import argparse
import numpy as np
import shutil
import subprocess
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from typing import Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── Resolve parent paths so we can import from sibling packages ─────────────
_HERE   = Path(__file__).resolve().parent          # …/multimodal/
_ROOT   = _HERE.parent                             # …/Project R1/
_CIL    = _ROOT / "CIL"
_SMART  = _ROOT / "smartdetector"
_USTC   = _ROOT / "USTC-TK2016-master"

for _p in [str(_ROOT), str(_CIL), str(_SMART)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from model import PDARTSBackbone
except ImportError as e:
    raise ImportError(f"Cannot import CIL model.py from {_CIL}.\nOriginal error: {e}")

try:
    from pretrain import ResNetEncoder
except ImportError as e:
    raise ImportError(f"Cannot import pretrain.py from {_SMART}.\nOriginal error: {e}")

try:
    from sam_matrix_construct import SAMConstructor, load_embedding_dicts, train_embedding_dictionaries
    from dataset import load_flows_from_pcap
    from model import ExpandableFeatureExtractor
except ImportError as e:
    raise ImportError(f"Cannot import from {_SMART}.\nOriginal error: {e}")

# ── Constants ───────────────────────────────────────────────────────────────
IMG_SIZE   = 28
IMG_BYTES  = IMG_SIZE * IMG_SIZE
SAM_K      = 40
SAM_B      = 100
SAM_C      = 3

DEFAULT_SEED    = 42

 # Instead of PDARTSBackbone

# ════════════════════════════════════════════════════════════════════════════
# ── BRANCH 1: Visual Branch (NAS Optimised CNN + CIL) ──────────────────────
# ════════════════════════════════════════════════════════════════════════════
class VisualBranch(nn.Module):
    def __init__(self, C: int = 32, layers: int = 6):
        super().__init__()
        self.backbone  = ExpandableFeatureExtractor(C=C, layers=layers)
        self.feature_dim = self.backbone.feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.0 if x.max() > 1.5 else x
        feat = self.backbone(x)
        return F.normalize(feat, dim=1)

# ════════════════════════════════════════════════════════════════════════════
# ── BRANCH 2: Semantic Branch (SmartDetector ResNet50 Contrastive Encoder) ──
# ════════════════════════════════════════════════════════════════════════════
class SemanticBranch(nn.Module):
    def __init__(self, embed_dim: int = 2048):
        super().__init__()
        self.encoder    = ResNetEncoder(in_channels=SAM_C, embed_dim=embed_dim)
        self.feature_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        return F.normalize(feat, dim=1)

# ════════════════════════════════════════════════════════════════════════════
# ── LATE FUSION BRIDGE (Concatenation) ─────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════
class LateFusionBridge(nn.Module):
    def __init__(self, dim_a: int, dim_b: int):
        super().__init__()
        self.dim_a  = dim_a
        self.dim_b  = dim_b
        self.out_dim = dim_a + dim_b

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        return torch.cat([feat_a, feat_b], dim=1)

# ════════════════════════════════════════════════════════════════════════════
# ── FUSION CLASSIFIER HEAD (Fully Connected MLP) ───────────────────────────
# ════════════════════════════════════════════════════════════════════════════
class FusionClassifierHead(nn.Module):
    def __init__(self, fused_dim: int, n_classes: int, hidden1: int = 512, hidden2: int = 256, dropout: float = 0.4):
        super().__init__()
        self.n_classes = n_classes
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

# ════════════════════════════════════════════════════════════════════════════
# ── MULTIMODAL FUSION NETWORK (Full Model) ──────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════
class MultimodalFusionNetwork(nn.Module):
    def __init__(self, n_classes: int, visual_C: int = 32, visual_layers: int = 6,
                 semantic_embed_dim: int = 2048, fusion_hidden1: int = 512, fusion_hidden2: int = 256, dropout: float = 0.4):
        super().__init__()
        self.visual_branch   = VisualBranch(C=visual_C, layers=visual_layers)
        dim_a = self.visual_branch.feature_dim
        self.semantic_branch = SemanticBranch(embed_dim=semantic_embed_dim)
        dim_b = self.semantic_branch.feature_dim
        self.fusion_bridge   = LateFusionBridge(dim_a, dim_b)
        fused_dim = self.fusion_bridge.out_dim

        self.classifier_head = FusionClassifierHead(
            fused_dim=fused_dim, n_classes=n_classes, hidden1=fusion_hidden1, hidden2=fusion_hidden2, dropout=dropout
        )

        print(f"[MultimodalFusionNetwork] Initialised:")
        print(f"  Visual Branch dim    : {dim_a}")
        print(f"  Semantic Branch dim  : {dim_b}")
        print(f"  Fused dim            : {fused_dim}")
        print(f"  Output classes       : {n_classes}")

    def forward(self, images: torch.Tensor, sams: torch.Tensor) -> torch.Tensor:
        feat_a = self.visual_branch(images)
        feat_b = self.semantic_branch(sams)
        fused  = self.fusion_bridge(feat_a, feat_b)
        logits = self.classifier_head(fused)
        return logits

    def freeze_visual_branch(self):
        for p in self.visual_branch.parameters(): p.requires_grad = False
        print("[MultimodalFusionNetwork] Visual branch FROZEN.")

    def freeze_semantic_branch(self):
        for p in self.semantic_branch.parameters(): p.requires_grad = False
        print("[MultimodalFusionNetwork] Semantic branch FROZEN.")

    def unfreeze_all(self):
        for p in self.parameters(): p.requires_grad = True
        print("[MultimodalFusionNetwork] All parameters UNFROZEN.")

    def count_parameters(self) -> dict:
        def _c(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {"visual_branch": _c(self.visual_branch), "semantic_branch": _c(self.semantic_branch),
                "classifier_head": _c(self.classifier_head), "total": _c(self)}

# ════════════════════════════════════════════════════════════════════════════
# ── AUTOMATED PREPROCESSING PIPELINE USING USTC-TK SCRIPTS ─────────────────
# ════════════════════════════════════════════════════════════════════════════
class AutomatedPreprocessor:
    """
    Automates the multimodal dataset generation process strictly using USTC-TK2016 scripts:
      1. Copies raw pcap data into `USTC-TK2016-master/1_Pcap/`.
      2. Executes `1_Pcap2Session.ps1` to break all pcap into sessions.
      3. Executes `2_ProcessSession.ps1` to filter lengths and split Train/Test.
      4. Executes `3_Session2Png.py` to generate the 28x28 grayscale PNG Images for Train/Test.
      5. Scans the generated TrimmedSession pcaps that have corresponding PNGs in `4_Png`
         and dynamically extracts Semantic Attribute Matrices (SAM), writing them to `4_Sam`.
      6. Moves all finished datasets (`4_Png` and `4_Sam`) symmetrically to the target `data_dir`.
    """
    def __init__(self, raw_pcap_dir: str, output_base_dir: str):
        self.raw_dir = Path(raw_pcap_dir).resolve()
        self.out_base = Path(output_base_dir).resolve()
        self.ustc_dir = _ROOT / "USTC-TK2016-master"
        
        self.sam_constructor = None
        self.embed_dim = SAM_B
        
    def _execute_powershell(self, script_path: Path):
        print(f"[*] Executing PowerShell script: {script_path.name}")
        cmd = ["powershell.exe", "-ExecutionPolicy", "Bypass", "-File", str(script_path)]
        # We run it with cwd=self.ustc_dir so relative paths in the scripts work perfectly
        try:
            result = subprocess.run(cmd, cwd=str(self.ustc_dir), check=True, capture_output=True, text=True)
            print(f"[+] Subprocess successful: {script_path.name}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[!] Error executing {script_path.name}: {e.stderr}")
            return False

    def _execute_python(self, script_path: Path):
        print(f"[*] Executing Python script: {script_path.name}")
        cmd = [sys.executable, str(script_path)]
        try:
            subprocess.run(cmd, cwd=str(self.ustc_dir), check=True, capture_output=True, text=True)
            print(f"[+] Subprocess successful: {script_path.name}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[!] Error executing {script_path.name}: {e.stderr}")
            return False

    def run_pipeline(self):
        print("\n--- [Step 1] Preparing USTC-TK environment ---")
        
        # We need to structure the 1_Pcap folder for the scripts
        pcap_1_dir = self.ustc_dir / "1_Pcap"
        os.makedirs(pcap_1_dir, exist_ok=True)
        
        # Clear previous runs
        print("[*] Clearing previous outputs from USTC-TK directory...")
        for folder in ["1_Pcap", "2_Session", "3_ProcessedSession", "4_Png", "4_Sam"]:
            d = self.ustc_dir / folder
            if d.exists():
                try: shutil.rmtree(d)
                except Exception: pass
            os.makedirs(d, exist_ok=True)
        
        # Prepare 1_Pcap using mergecap to merge all pcap files in each class folder into a single file.
        print("[*] Migrating and merging raw data into 1_Pcap...")
        classes = [c.name for c in self.raw_dir.iterdir() if c.is_dir()]

        if not classes:
            # Check if there are pcaps directly in the directory
            pcap_files = list(self.raw_dir.glob("*.pcap")) + list(self.raw_dir.glob("*.pcapng"))
            if pcap_files:
                classes = [self.raw_dir.name]
                self.raw_dir = self.raw_dir.parent
        
        wireshark_dir = Path(r"C:\Program Files\Wireshark")
        mergecap_path = wireshark_dir / "mergecap.exe"
        editcap_path = wireshark_dir / "editcap.exe"
        
        for cls_name in classes:
            pcap_files = list((self.raw_dir / cls_name).glob("*.pcap")) + list((self.raw_dir / cls_name).glob("*.pcapng"))
            if not pcap_files:
                print(f"[!] No PCAP files found for class {cls_name}. Skipping.")
                continue
                
            dst = pcap_1_dir / f"{cls_name}.pcap"
            print(f"[{cls_name}] Processing {len(pcap_files)} files into {dst.name}...")
            
            # Use mergecap if Wireshark is installed
            if mergecap_path.exists():
                temp_pcap_files = []
                # Stage 1: Convert each file to standard pcap individually (Robust)
                if editcap_path.exists():
                    print(f"  -> Converting {len(pcap_files)} files to standard pcap format...")
                    for i, f in enumerate(pcap_files):
                        tmp_f = pcap_1_dir / f"tmp_{cls_name}_{i}.pcap"
                        # editcap -F pcap -T ether <src> <dst>
                        cmd = [str(editcap_path), "-F", "pcap", str(f), str(tmp_f)]
                        try:
                            subprocess.run(cmd, check=True, capture_output=True)
                            temp_pcap_files.append(str(tmp_f))
                        except subprocess.CalledProcessError as e:
                            print(f"  [!] Editcap failed for {f.name}: {e.stderr.decode(errors='replace')}")
                            # Fallback: just use it as is
                            temp_pcap_files.append(str(f))
                else:
                    print(f"  [!] Editcap not found at {editcap_path}. Skipping individual conversion.")
                    temp_pcap_files = [str(f) for f in pcap_files]

                # Stage 2: Merge the (likely converted) files
                print(f"  -> Merging files...")
                cmd = [str(mergecap_path), "-F", "pcap", "-w", str(dst)] + temp_pcap_files
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                except subprocess.CalledProcessError as e:
                    print(f"  [!] Mergecap failed for {cls_name}: {e.stderr.decode(errors='replace')}")
                    # Final fallback: copy the first one
                    if not dst.exists(): shutil.copy(pcap_files[0], dst)
                
                # Cleanup temp files
                for tmp_f in temp_pcap_files:
                    if "tmp_" in tmp_f:
                        try: os.unlink(tmp_f)
                        except: pass
            else:
                print(f"[!] Mergecap not found at {mergecap_path}. Make sure Wireshark is installed.")
                print(f"[*] Fallback: Copying only the first file {pcap_files[0].name}")
                shutil.copy(pcap_files[0], dst)

        print("\n--- [Step 2] Running SplitCap to extract sessions natively ---")
        splitcap_exe = self.ustc_dir / "0_Tool" / "SplitCap_2-1" / "SplitCap.exe"
        all_layers_dir = self.ustc_dir / "2_Session" / "AllLayers"
        os.makedirs(all_layers_dir, exist_ok=True)
        
        for cls_name in classes:
            pcap_file = pcap_1_dir / f"{cls_name}.pcap"
            if not pcap_file.exists(): continue
            
            target_out = all_layers_dir / cls_name
            os.makedirs(target_out, exist_ok=True)
            
            print(f"[{cls_name}] Splitting {pcap_file.name} into sessions...")
            cmd = [str(splitcap_exe), "-p", "50000", "-b", "50000", "-s", "session", "-r", str(pcap_file), "-o", str(target_out)]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                
                # Clean empty files (which 1_Pcap2Session.ps1 did)
                for f in target_out.glob("*"):
                    if f.is_file() and f.stat().st_size == 0:
                        try: f.unlink()
                        except: pass
            except subprocess.CalledProcessError as e:
                print(f"[!] SplitCap failed for {cls_name}: {e.stderr}")

        print("\n--- [Step 3] Running 2_ProcessSession.ps1 (Train/Test Split & Trimming) ---")
        if not self._execute_powershell(self.ustc_dir / "2_ProcessSession.ps1"): return

        print("\n--- [Step 4] Running 3_Session2Png.py (Image Generation) ---")
        if not self._execute_python(self.ustc_dir / "3_Session2Png.py"): return

        print("\n--- [Step 5] Extracting SAM Matrices iteratively mapped from Images ---")
        self._generate_sams()

        print("\n--- [Step 6] Migrating final output datasets to target directory ---")
        out_images = self.out_base / "3_images"
        out_sams   = self.out_base / "4_sams"
        if out_images.exists(): shutil.rmtree(out_images)
        if out_sams.exists(): shutil.rmtree(out_sams)
        
        shutil.copytree(self.ustc_dir / "4_Png", out_images)
        shutil.copytree(self.ustc_dir / "4_Sam", out_sams)
        
        print("\n[+] Full Automated Preprocessing Complete using USTC-TK logic.")
        print(f"[+] Processed multimodal data fully structured in: {self.out_base}")

    def _init_sam_constructor(self):
        print("\n--- Initialising SAM Constructor ---")
        embed_dir = str(_ROOT / "checkpoints" / "embeddings")
        if os.path.exists(embed_dir):
            try:
                Dz, Da = load_embedding_dicts(embed_dir)
                self.sam_constructor = SAMConstructor(Dz, Da, k=SAM_K, b=SAM_B)
                print(f"[SAM] Loaded pre-trained embeddings from {embed_dir}")
                return
            except Exception as e:
                print(f"[SAM] Warning: Could not load embeddings: {e}")
                
        print(f"[SAM] Training new Word2Vec embeddings from Train set...")
        all_train_flows = []
        train_src = self.ustc_dir / "3_ProcessedSession" / "TrimedSession" / "Train"
        if not train_src.exists():
            print("[SAM] Error: TrimedSession Train dir missing.")
            return

        for cls_dir in [d for d in train_src.iterdir() if d.is_dir()]:
            for sf in list(cls_dir.glob("*"))[:200]: 
                try:
                    flows = load_flows_from_pcap(str(sf), cls_dir.name)
                    all_train_flows.extend(flows)
                except Exception:
                    pass
        
        if len(all_train_flows) > 0:
            Dz, Da = train_embedding_dictionaries(all_train_flows, embedding_dim=SAM_B, epochs=5)
            self.sam_constructor = SAMConstructor(Dz, Da, k=SAM_K, b=SAM_B)
            
            os.makedirs(embed_dir, exist_ok=True)
            import pickle
            with open(os.path.join(embed_dir, "Dz.pkl"), "wb") as f: pickle.dump(Dz, f)
            with open(os.path.join(embed_dir, "Da.pkl"), "wb") as f: pickle.dump(Da, f)
            print(f"[SAM] Saved new embeddings to {embed_dir}")
        else:
             print("[SAM] Error: Could not parse any flows for training embeddings.")

    def _generate_sams(self):
        if not self.sam_constructor:
            self._init_sam_constructor()

        if not self.sam_constructor:
            return

        # Symmetrically iterate generated PNGs and parse exact original PCAP
        base_png_dir = self.ustc_dir / "4_Png"
        # Since TrimedSession files are 784 bytes filled with 0s at the end, Scapy might fail to parse valid network flows
        # from them because the pcap trailer/headers are potentially corrupted or cut off. 
        # Smartdetector uses the complete session files, so we load the corresponding un-trimed files from FilteredSession!
        base_session_dir = self.ustc_dir / "3_ProcessedSession" / "FilteredSession"
        base_sam_dir = self.ustc_dir / "4_Sam"
        
        for split in ["Train", "Test"]:
            png_split = base_png_dir / split
            ses_split = base_session_dir / split
            sam_split = base_sam_dir / split
            
            if not png_split.exists() or not ses_split.exists(): continue
            
            classes = [d.name for d in png_split.iterdir() if d.is_dir()]
            for cls_name in classes:
                png_cls_dir = png_split / cls_name
                ses_cls_dir = ses_split / cls_name
                sam_cls_dir = sam_split / cls_name
                os.makedirs(sam_cls_dir, exist_ok=True)
                
                png_files = list(png_cls_dir.glob("*.png"))
                print(f"[{split} - {cls_name}] Generating SAM Matrices for {len(png_files)} sessions...")
                
                for png_f in tqdm(png_files):
                    session_basename = png_f.stem  # drops the .png extension
                    # Match this against FilteredSession to get the real PCAP flow
                    # It might have .pcap extension or no extension.
                    session_file = ses_cls_dir / session_basename
                    fallback_file = ses_cls_dir / f"{session_basename}.pcap"
                    
                    target_file = session_file
                    if not session_file.exists(): target_file = fallback_file

                    if target_file.exists():
                        try:
                            flows = load_flows_from_pcap(str(target_file), cls_name)
                            if flows:
                                sam_batch = self.sam_constructor.batch_flows_to_sam(flows, verbose=False)
                                sam_matrix = sam_batch[0]
                                np.save(sam_cls_dir / f"{session_basename}.npy", sam_matrix)
                            else:
                                # Parsing failed or flow empty, discard the image to keep alignment strict!
                                png_f.unlink()
                        except Exception:
                            # If it crashes entirely, delete image
                            png_f.unlink()
                    else:
                        # Original session lost, discard image
                        png_f.unlink()

# ════════════════════════════════════════════════════════════════════════════
# ── MULTIMODAL DATASET (Offline Pre-processed) ─────────────────────────────
# ════════════════════════════════════════════════════════════════════════════
class PreprocessedMultimodalDataset(Dataset):
    """
    Dataset that loads strictly aligned pre-processed Images and SAMs from the automated pipeline outputs.
    """
    def __init__(self, images_dir: str, sams_dir: str):
        self.images_dir = Path(images_dir)
        self.sams_dir = Path(sams_dir)
        
        self.classes = sorted([d.name for d in self.images_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        self.samples = []
        for cls_name in self.classes:
            img_cls_dir = self.images_dir / cls_name
            sam_cls_dir = self.sams_dir / cls_name
            
            for img_path in img_cls_dir.glob("*.png"):
                sam_path = sam_cls_dir / f"{img_path.stem}.npy"
                if sam_path.exists():
                    self.samples.append((str(img_path), str(sam_path), self.class_to_idx[cls_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, sam_path, label = self.samples[idx]
        
        # Load Image
        img = Image.open(img_path).convert('L')
        img_arr = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.tensor(img_arr).unsqueeze(0) # (1, 28, 28)
        
        # Load SAM
        sam_arr = np.load(sam_path).astype(np.float32)
        sam_tensor = torch.tensor(sam_arr) # (3, 40, 100)
        
        return img_tensor, sam_tensor, torch.tensor(label, dtype=torch.long)

    @classmethod
    def generate_synthetic(cls, n_samples: int = 200, n_classes: int = 20, seed: int = DEFAULT_SEED):
        class SyntheticDataset(Dataset):
            def __init__(self, n_samples, n_classes, seed):
                rng = np.random.default_rng(seed)
                self.images = torch.tensor(rng.random((n_samples, 1, IMG_SIZE, IMG_SIZE)).astype(np.float32))
                self.sams = torch.tensor(rng.random((n_samples, SAM_C, SAM_K, SAM_B)).astype(np.float32))
                self.labels = torch.tensor(rng.integers(0, n_classes, n_samples, dtype=np.int64))
                
            def __len__(self): return len(self.labels)
            def __getitem__(self, idx): return self.images[idx], self.sams[idx], self.labels[idx]
        return SyntheticDataset(n_samples, n_classes, seed)


# ════════════════════════════════════════════════════════════════════════════
# ── MULTIMODAL TRAINER ──────────────────────────────────────────════════════
# ════════════════════════════════════════════════════════════════════════════
class MultimodalTrainer:
    def __init__(self, model: MultimodalFusionNetwork, device: Optional[str] = None, lr: float = 1e-3, weight_decay: float = 1e-4):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.history: list = []

    def _is_lfs_pointer(self, path: str) -> bool:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                line = f.readline()
                return "git-lfs" in line
        except: return False

    def load_visual_branch(self, checkpoint_path: str, key: str = "backbone") -> None:
        if self._is_lfs_pointer(checkpoint_path):
            print(f"[ERROR] '{checkpoint_path}' is a Git LFS pointer, not the actual weight file.")
            print("[HINT] Please download the real weights or run 'git lfs pull'.")
            return
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state = ckpt.get(key, ckpt.get("model", ckpt.get("state_dict", ckpt))) if isinstance(ckpt, dict) else ckpt
        missing, unexp = self.model.visual_branch.backbone.load_state_dict(state, strict=False)
        print(f"[Trainer] Visual branch loaded from: {checkpoint_path}")

    def load_semantic_branch(self, checkpoint_path: str, key: str = "encoder") -> None:
        if self._is_lfs_pointer(checkpoint_path):
            print(f"[ERROR] '{checkpoint_path}' is a Git LFS pointer, not the actual weight file.")
            print("[HINT] Please download the real weights or run 'git lfs pull'.")
            return
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state = ckpt.get(key, ckpt.get("model", ckpt.get("state_dict", ckpt))) if isinstance(ckpt, dict) else ckpt
        missing, unexp = self.model.semantic_branch.encoder.load_state_dict(state, strict=False)
        print(f"[Trainer] Semantic branch loaded from: {checkpoint_path}")

    def _build_optimizer(self) -> torch.optim.Optimizer:
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.Adam(trainable, lr=self.lr, weight_decay=self.weight_decay)

    def _build_scheduler(self, optimizer: torch.optim.Optimizer, epochs: int):
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    def train_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
        self.model.train()
        total_loss, n_batches = 0.0, 0
        pbar = tqdm(loader, desc="  Training", leave=False)
        for images, sams, labels in pbar:
            images, sams, labels = images.to(self.device), sams.to(self.device), labels.to(self.device)
            logits = self.model(images, sams)
            loss = self.criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=loss.item())
        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        self.model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        pbar = tqdm(loader, desc="  Evaluating", leave=False)
        for images, sams, labels in pbar:
            images, sams, labels = images.to(self.device), sams.to(self.device), labels.to(self.device)
            logits = self.model(images, sams)
            loss = self.criterion(logits, labels)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += len(labels)
            total_loss += loss.item()
        return {"loss": total_loss / max(len(loader), 1), "accuracy": total_correct / max(total_samples, 1) * 100.0}

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, epochs: int = 50, mode: str = "full", save_dir: str = str(_ROOT / "checkpoints")) -> list:
        os.makedirs(save_dir, exist_ok=True)
        if mode == "fusion_only":
            self.model.freeze_visual_branch()
            self.model.freeze_semantic_branch()
            print("[Trainer] Mode: FUSION HEAD ONLY")
        elif mode == "full":
            self.model.unfreeze_all()
            print("[Trainer] Mode: FULL END-TO-END")
        elif mode == "finetune":
            self.model.freeze_visual_branch()
            self.model.freeze_semantic_branch()
            print("[Trainer] Mode: FINETUNE (warm-up)")

        param_counts = self.model.count_parameters()
        print(f"[Trainer] Trainable params: {param_counts['total']:,}")

        optimizer = self._build_optimizer()
        scheduler = self._build_scheduler(optimizer, epochs)
        best_val_acc = 0.0

        for epoch in range(1, epochs + 1):
            if mode == "finetune" and epoch == 11:
                self.model.unfreeze_all()
                optimizer = self._build_optimizer()
                scheduler = self._build_scheduler(optimizer, epochs - 10)
                print(f"  [Epoch {epoch}] Unfreezing branches for full fine-tune.")

            t0 = time.time()
            train_loss = self.train_epoch(train_loader, optimizer)
            scheduler.step()
            dt = time.time() - t0

            entry = {"epoch": epoch, "train_loss": train_loss, "lr": scheduler.get_last_lr()[0], "time_s": round(dt, 2)}
            
            if val_loader:
                val_metrics = self.evaluate(val_loader)
                entry.update({"val_loss": val_metrics["loss"], "val_acc": val_metrics["accuracy"]})
                if val_metrics["accuracy"] > best_val_acc:
                    best_val_acc = val_metrics["accuracy"]
                    self.save(os.path.join(save_dir, "multimodal_best.pth"))
                    print(f"  [Epoch {epoch:>3}/{epochs}] Train={train_loss:.4f} | Val={val_metrics['loss']:.4f}, Acc={val_metrics['accuracy']:.2f}% (Best!) [{dt:.1f}s]")
                else:
                    print(f"  [Epoch {epoch:>3}/{epochs}] Train={train_loss:.4f} | Val={val_metrics['loss']:.4f}, Acc={val_metrics['accuracy']:.2f}% [{dt:.1f}s]")
            else:
                print(f"  [Epoch {epoch:>3}/{epochs}] Train={train_loss:.4f} [{dt:.1f}s]")
                
            self.history.append(entry)

        print(f"\n[Trainer] Complete. Best Val Acc: {best_val_acc:.2f}%")
        self.save(os.path.join(save_dir, "multimodal_final.pth"))
        return self.history

    def save(self, path: str) -> None: torch.save(self.model.state_dict(), path)


# ════════════════════════════════════════════════════════════════════════════
# ── COMMAND LINE INTERFACE (Testing / Preprocessing / Training) ────────────
# ════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Multimodal Late Fusion Network with Automated Preprocessing")
    parser.add_argument("--mode", type=str, choices=["test", "preprocess", "train", "train_fusion"], default="test")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--visual_weights", type=str, default=str(_ROOT / "checkpoints" / "base_backbone_T2.pth"))
    parser.add_argument("--semantic_weights", type=str, default=str(_ROOT / "checkpoints" / "encoder_best.pt"))
    parser.add_argument("--data_dir", type=str, default=None, help="Root dataset config depending on mode.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Multimodal Late Fusion Network ===")
    
    if args.mode == "preprocess":
        if not args.data_dir:
            print("[ERROR] --data_dir must be provided for preprocessing (e.g. C:/Users/LENOVO/Downloads/pcap_classes)")
            return
            
        out_base = str(Path(args.data_dir).parent / "processed_multimodal_dataset")
        print(f"\n[MODE] PREPROCESS")
        preprocessor = AutomatedPreprocessor(raw_pcap_dir=args.data_dir, output_base_dir=out_base)
        preprocessor.run_pipeline()
        return

    ds_train = None
    ds_val = None
    n_classes = None

    if args.mode in ["train", "train_fusion"] and args.data_dir and os.path.isdir(args.data_dir):
        base = Path(args.data_dir)
        
        # --- UPDATED: Use your actual folder names ---
        img_dir_name = "4_Png"   # Changed from "3_images"
        sam_dir_name = "4_sams"  # Changed from "4_sams" (ensure this matches your folder name)

        # Check if the folders exist
        if not (base / img_dir_name).exists():
            print(f"[ERROR] Could not find image folder: {base / img_dir_name}")
            return
        if not (base / sam_dir_name).exists():
            print(f"[ERROR] Could not find SAM folder: {base / sam_dir_name}")
            return

        print(f"[INFO] Detected Data Source: Images={img_dir_name}, SAMs={sam_dir_name}")

        train_img_dir = os.path.join(args.data_dir, img_dir_name, "Train")
        train_sam_dir = os.path.join(args.data_dir, sam_dir_name, "Train")
        test_img_dir = os.path.join(args.data_dir, img_dir_name, "Test")
        test_sam_dir = os.path.join(args.data_dir, sam_dir_name, "Test")
        
        ds_train = PreprocessedMultimodalDataset(train_img_dir, train_sam_dir)
        ds_val   = PreprocessedMultimodalDataset(test_img_dir, test_sam_dir)
        n_classes = len(ds_train.classes)
        print(f"[INFO] Detected {n_classes} classes from dataset: {ds_train.classes}")
        print(f"[INFO] Loaded {len(ds_train)} training, {len(ds_val)} validation samples.")
    else:
        # Default fallback for testing or synthetic generation if no dataset is loaded
        n_classes = 20

    # Initialize model
    model = MultimodalFusionNetwork(n_classes=n_classes).to(device)

    if args.mode == "test":
        print("\n[MODE] Smoke test with random synthetic data")
        ds = PreprocessedMultimodalDataset.generate_synthetic(n_samples=8, n_classes=n_classes)
        dl = DataLoader(ds, batch_size=4)
        imgs, sams, labels = next(iter(dl))
        print(f"Images : {imgs.shape}")
        print(f"SAMs   : {sams.shape}")
        logits = model(imgs.to(device), sams.to(device))
        print(f"Output Logits: {logits.shape}")
        print("\nSmoke test passed successfully!")

    elif args.mode in ["train", "train_fusion"]:
        print(f"\n[MODE] {args.mode.upper()}")
        trainer = MultimodalTrainer(model, device=device)
        
        if os.path.exists(args.visual_weights): trainer.load_visual_branch(args.visual_weights)
        if os.path.exists(args.semantic_weights): trainer.load_semantic_branch(args.semantic_weights)

        if not ds_train:
            print("\n[WARN] Generating small synthetic dataset for training demo.")
            ds_train = PreprocessedMultimodalDataset.generate_synthetic(n_samples=256, n_classes=n_classes)
            ds_val   = PreprocessedMultimodalDataset.generate_synthetic(n_samples=64, n_classes=n_classes)
        
        train_loader = DataLoader(ds_train, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
        val_loader   = DataLoader(ds_val, batch_size=32, num_workers=2, pin_memory=True)

        train_mode = "fusion_only" if args.mode == "train_fusion" else "finetune"
        trainer.train(train_loader, val_loader, epochs=args.epochs, mode=train_mode)

if __name__ == "__main__":
    main()
