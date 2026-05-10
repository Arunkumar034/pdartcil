import os
import shutil
import binascii
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from scipy.io import savemat

from smartdetector.sam_matrix_construct import SAMConstructor
from smartdetector.dataset import load_flows_from_pcap
from smartdetector.sam_matrix_construct import train_embedding_dictionaries, load_embedding_dicts

def getMatrixfrom_pcap(filename, width=28, max_len=784):
    with open(filename, 'rb') as f:
        content = f.read()
    hexst = binascii.hexlify(content)
    fh = np.array([int(hexst[i:i+2], 16) for i in range(0, len(hexst), 2)])
    if len(fh) > max_len:
        fh = fh[:max_len]
    else:
        fh = np.pad(fh, (0, max_len - len(fh)), 'constant')
    rn = len(fh) // width
    fh = np.reshape(fh[:rn * width], (-1, width))
    fh = np.uint8(fh)
    return fh

def build_dataset(pcap_classes_dir, output_dir="3_preprocess_session", split_ratio=0.8):
    pcap_dir = Path(pcap_classes_dir)
    out_dir = Path(output_dir)
    
    train_dir = out_dir / "train"
    test_dir = out_dir / "test"
    sam_dir = out_dir / "sams"
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(sam_dir, exist_ok=True)

    print(f"[*] Processing dataset from {pcap_classes_dir}")
    
    classes = [d.name for d in pcap_dir.iterdir() if d.is_dir()]
    print(f"[*] Found {len(classes)} classes: {classes}")
    
    # Train test split for files
    for cls in classes:
        cls_path = pcap_dir / cls
        files = list(cls_path.glob("*.pcap")) + list(cls_path.glob("*"))
        files = [f for f in files if f.is_file()]
        
        np.random.shuffle(files)
        split_idx = int(len(files) * split_ratio)
        train_files = files[:split_idx]
        test_files = files[split_idx:]
        
        os.makedirs(train_dir / cls, exist_ok=True)
        os.makedirs(test_dir / cls, exist_ok=True)
        os.makedirs(sam_dir / cls, exist_ok=True)
        
        for f in tqdm(train_files, desc=f"Train {cls}"):
             shutil.copy(f, train_dir / cls / f.name)
             
        for f in tqdm(test_files, desc=f"Test {cls}"):
             shutil.copy(f, test_dir / cls / f.name)
             
    print("[*] Processing Complete.")
