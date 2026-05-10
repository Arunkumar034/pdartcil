import torch
import struct
import numpy as np
import gzip  # Necessary for .gz files
from torch.utils.data import Dataset

class IDXDataset(Dataset):
    """
    Load IDX-format datasets.
    Updated to support both compressed (.gz) and uncompressed files.
    """
    def __init__(self, images_path, labels_path, class_filter=None, max_samples_per_class=None):
        self.images = self._load_idx_images(images_path)
        self.labels = self._load_idx_labels(labels_path)
        
        # Filter by class
        if class_filter is not None:
            mask = np.isin(self.labels, class_filter)
            self.images = self.images[mask]
            self.labels = self.labels[mask]
        
        # Limit samples per class
        if max_samples_per_class is not None:
            self._limit_samples_per_class(max_samples_per_class)
        
        self.samples = [(i, int(self.labels[i])) for i in range(len(self.labels))]
    
    def _load_idx_images(self, path):
        """Load IDX3-UBYTE format images (Supports .gz)"""
        # Determine if we need gzip or standard open
        open_func = gzip.open if path.endswith('.gz') else open
        
        with open_func(path, 'rb') as f:
            magic = struct.unpack('>I', f.read(4))[0]
            if magic != 2051:
                raise ValueError(f"Invalid magic number {magic} in {path}. Check if file is corrupted or not Gzipped.")
            
            num_images = struct.unpack('>I', f.read(4))[0]
            rows = struct.unpack('>I', f.read(4))[0]
            cols = struct.unpack('>I', f.read(4))[0]
            
            # Read the rest of the buffer
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(num_images, rows, cols)
        
        return images
    
    def _load_idx_labels(self, path):
        """Load IDX1-UBYTE format labels (Supports .gz)"""
        open_func = gzip.open if path.endswith('.gz') else open
        
        with open_func(path, 'rb') as f:
            magic = struct.unpack('>I', f.read(4))[0]
            if magic != 2049:
                raise ValueError(f"Invalid magic number {magic} in {path}")
            
            num_labels = struct.unpack('>I', f.read(4))[0]
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        
        return labels
    
    def _limit_samples_per_class(self, max_samples):
        unique_classes = np.unique(self.labels)
        keep_indices = []
        
        print(f"   Limiting samples per class to {max_samples}...")
        
        for cls in unique_classes:
            cls_indices = np.where(self.labels == cls)[0]
            original_count = len(cls_indices)
            
            if len(cls_indices) > max_samples:
                np.random.seed(42)
                selected_indices = np.random.choice(cls_indices, max_samples, replace=False)
                keep_indices.extend(selected_indices)
                print(f"     Class {cls}: {original_count} → {max_samples} samples")
            else:
                keep_indices.extend(cls_indices)
                print(f"     Class {cls}: {original_count} samples")
        
        keep_indices = sorted(keep_indices)
        self.images = self.images[keep_indices]
        self.labels = self.labels[keep_indices]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        label = int(self.labels[idx])
        # Normalization and channel add [1, H, W]
        img_tensor = torch.tensor(img / 255.0, dtype=torch.float32).unsqueeze(0)
        return img_tensor, label

class MemoryDataset(Dataset):
    def __init__(self, exemplar_list):
        self.exemplars = exemplar_list
    
    def __len__(self):
        return len(self.exemplars)
    
    def __getitem__(self, idx):
        img_tensor, label = self.exemplars[idx]
        return img_tensor, int(label)

def load_label_mapping(mapping_path):
    """
    Updated to handle 'Index: Name' format used in USTC-TK tools
    """
    label_map = {}
    try:
        with open(mapping_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Try splitting by colon first (Standard for USTC-TK output)
                if ':' in line:
                    parts = line.split(':')
                # Fallback to tab
                elif '\t' in line:
                    parts = line.split('\t')
                else:
                    continue
                    
                if len(parts) >= 2:
                    idx = parts[0].strip()
                    name = parts[1].strip()
                    label_map[int(idx)] = name
    except Exception as e:
        print(f"Error loading label mapping: {e}")
    
    return label_map