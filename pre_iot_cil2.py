import os
import random
import hashlib
import struct
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from scapy.all import IP, TCP, UDP, Ether, PcapReader

# ==========================================================
# CONFIGURATION
# ==========================================================

IMAGE_SIDE = 28
IMAGE_SIZE = IMAGE_SIDE * IMAGE_SIDE
TRAIN_RATIO = 0.9
RANDOM_SEED = 42
MAX_SAMPLES_PER_CLASS = 4000  # Limit per class

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ==========================================================
# SESSION EXTRACTION
# ==========================================================

def extract_sessions(pcap_path, max_sessions=None):
    """Extract sessions - stop early if max_sessions reached"""
    sessions = defaultdict(list)
    session_count = 0

    try:
        with PcapReader(pcap_path) as reader:
            for pkt in reader:
                if IP not in pkt:
                    continue

                # Stop early if we have enough sessions
                if max_sessions and session_count >= max_sessions:
                    break

                timestamp = float(pkt.time)
                src_ip = pkt[IP].src
                dst_ip = pkt[IP].dst
                proto = pkt[IP].proto

                src_port = 0
                dst_port = 0

                if TCP in pkt:
                    src_port = pkt[TCP].sport
                    dst_port = pkt[TCP].dport
                elif UDP in pkt:
                    src_port = pkt[UDP].sport
                    dst_port = pkt[UDP].dport

                # Bidirectional session key
                if (src_ip, src_port) <= (dst_ip, dst_port):
                    key = (src_ip, dst_ip, src_port, dst_port, proto)
                else:
                    key = (dst_ip, src_ip, dst_port, src_port, proto)

                is_new_session = key not in sessions
                sessions[key].append((timestamp, bytes(pkt)))
                
                if is_new_session:
                    session_count += 1
                
    except Exception as e:
        print(f"Error reading {os.path.basename(pcap_path)}: {e}")
        return []

    ordered_sessions = []
    for packet_list in sessions.values():
        packet_list.sort(key=lambda x: x[0])
        ordered_sessions.append([pkt for _, pkt in packet_list])

    return ordered_sessions

# ==========================================================
# ANONYMIZATION
# ==========================================================

class Anonymizer:
    def __init__(self, seed=42):
        self.ip_map = {}
        self.mac_map = {}
        self.rng = random.Random(seed)

    def anon_ip(self, ip):
        if ip not in self.ip_map:
            h = int(hashlib.md5(ip.encode()).hexdigest()[:8], 16)
            self.rng.seed(h)
            self.ip_map[ip] = (
                f"{self.rng.randint(1,254)}."
                f"{self.rng.randint(0,255)}."
                f"{self.rng.randint(0,255)}."
                f"{self.rng.randint(1,254)}"
            )
        return self.ip_map[ip]

    def anon_mac(self, mac):
        if mac not in self.mac_map:
            h = int(hashlib.md5(mac.encode()).hexdigest()[:12], 16)
            self.rng.seed(h)
            self.mac_map[mac] = ":".join(
                f"{self.rng.randint(0,255):02x}" for _ in range(6)
            )
        return self.mac_map[mac]

def anonymize_session(session_packets, anonymizer):
    """Anonymize IPs and MACs using Scapy"""
    anonymized = []
    
    for raw_pkt in session_packets:
        try:
            pkt = Ether(raw_pkt)

            if Ether in pkt:
                pkt[Ether].src = anonymizer.anon_mac(pkt[Ether].src)
                pkt[Ether].dst = anonymizer.anon_mac(pkt[Ether].dst)

            if IP in pkt:
                pkt[IP].src = anonymizer.anon_ip(pkt[IP].src)
                pkt[IP].dst = anonymizer.anon_ip(pkt[IP].dst)
                
                # Clear checksums
                del pkt[IP].chksum
                if TCP in pkt:
                    del pkt[TCP].chksum
                elif UDP in pkt:
                    del pkt[UDP].chksum

            anonymized.append(bytes(pkt))
            
        except Exception:
            anonymized.append(raw_pkt)
    
    return anonymized

# ==========================================================
# BUILD 784 SAMPLE
# ==========================================================

def build_784_sample(session_packets):
    buffer = bytearray()

    for pkt in session_packets:
        buffer.extend(pkt)
        if len(buffer) >= IMAGE_SIZE:
            break

    if len(buffer) >= IMAGE_SIZE:
        return bytes(buffer[:IMAGE_SIZE])
    else:
        buffer.extend(b"\x00" * (IMAGE_SIZE - len(buffer)))
        return bytes(buffer)

# ==========================================================
# PROCESS SINGLE PCAP
# ==========================================================

def process_single_pcap(pcap_path, label_id, anonymizer, needed_samples):
    """Process one pcap file and return only needed samples"""
    
    images = []
    labels = []
    seen_hashes = set()

    # Stop early if we already have enough
    if needed_samples <= 0:
        return images, labels

    sessions = extract_sessions(pcap_path, max_sessions=needed_samples * 2)

    for session in sessions:
        if len(images) >= needed_samples:
            break  # Stop once we have enough
            
        if not session:
            continue

        # Apply anonymization
        session = anonymize_session(session, anonymizer)
        sample = build_784_sample(session)

        # Remove empty samples
        if sample.strip(b"\x00") == b"":
            continue

        # Remove duplicates
        signature = hashlib.md5(sample).hexdigest()
        if signature in seen_hashes:
            continue

        seen_hashes.add(signature)

        # Convert to image
        img = np.frombuffer(sample, dtype=np.uint8).reshape(IMAGE_SIDE, IMAGE_SIDE)
        images.append(img)
        labels.append(label_id)

    return images, labels

# ==========================================================
# IDX SAVE
# ==========================================================

def save_idx_images(images, filepath):
    with open(filepath, "wb") as f:
        f.write(struct.pack(">I", 2051))
        f.write(struct.pack(">I", len(images)))
        f.write(struct.pack(">I", IMAGE_SIDE))
        f.write(struct.pack(">I", IMAGE_SIDE))
        f.write(np.array(images, dtype=np.uint8).tobytes())

def save_idx_labels(labels, filepath):
    with open(filepath, "wb") as f:
        f.write(struct.pack(">I", 2049))
        f.write(struct.pack(">I", len(labels)))
        f.write(np.array(labels, dtype=np.uint8).tobytes())

def save_label_mapping(label_map, filepath):
    """Save label mapping to text file"""
    with open(filepath, 'w') as f:
        for label_id in sorted(label_map.keys()):
            label_name = label_map[label_id]
            f.write(f"{label_id} {label_name}\n")
    print(f"✅ Label mapping saved to: {filepath}")

# ==========================================================
# GET CLASS LABEL FROM PATH
# ==========================================================

def get_class_label(file_path, root_dir):
    """
    Extract class name from path.
    Example: C:/pcap_classes/nonvpn_chat/pcaps/file.pcap -> nonvpn_chat
    """
    rel_path = os.path.relpath(file_path, root_dir)
    parts = rel_path.split(os.sep)
    
    # The first directory after root is the class
    if len(parts) >= 2:
        return parts[0]
    
    return "unknown"

# ==========================================================
# MAIN PROCESSOR
# ==========================================================

def process_dataset(input_root, output_prefix):

    # Collect all PCAP files grouped by class
    class_files = defaultdict(list)
    
    print(f"Scanning directory: {input_root}\n")
    
    for root, _, files in os.walk(input_root):
        for f in files:
            if f.endswith(".pcap"):
                full_path = os.path.join(root, f)
                
                # Get class label from directory structure
                label = get_class_label(full_path, input_root)
                
                # Skip if unknown
                if label == "unknown":
                    continue
                
                # Get file size
                file_size = os.path.getsize(full_path)
                
                class_files[label].append((full_path, file_size))

    if len(class_files) == 0:
        print("ERROR: No classes found!")
        print("Make sure your directory structure is:")
        print("  pcap_classes/")
        print("    nonvpn_chat/")
        print("      *.pcap")
        print("    nonvpn_email/")
        print("      *.pcap")
        return

    # Sort files by size (small first - faster processing)
    for label in class_files:
        class_files[label].sort(key=lambda x: x[1])

    # Create label map - IMPORTANT: Use sorted() for consistent ordering
    sorted_labels = sorted(class_files.keys())
    label_map = {idx: label for idx, label in enumerate(sorted_labels)}
    
    # Create reverse mapping (name -> id)
    label_to_id = {label: idx for idx, label in label_map.items()}

    print(f"Found {len(label_map)} classes:")
    for idx in sorted(label_map.keys()):
        label = label_map[idx]
        files = class_files[label]
        total_size = sum(size for _, size in files) / (1024**3)  # GB
        print(f"  [{idx}] {label}: {len(files)} files ({total_size:.2f} GB)")
    
    print(f"\nMax samples per class: {MAX_SAMPLES_PER_CLASS}")
    print(f"Processing smallest files first...\n")

    all_images = []
    all_labels = []
    
    # Create single anonymizer
    anonymizer = Anonymizer(seed=RANDOM_SEED)

    # Process each class IN SORTED ORDER
    for idx in sorted(label_map.keys()):
        label = label_map[idx]
        files = class_files[label]
        
        class_images = []
        class_labels = []
        
        print(f"\n{'='*60}")
        print(f"Processing class [{idx}]: {label}")
        print(f"{'='*60}")
        
        # Process files until we have enough samples
        with tqdm(total=MAX_SAMPLES_PER_CLASS, desc=f"{label}") as pbar:
            for pcap_path, file_size in files:
                if len(class_images) >= MAX_SAMPLES_PER_CLASS:
                    print(f"✓ Reached {MAX_SAMPLES_PER_CLASS} samples, skipping remaining files")
                    break
                
                needed = MAX_SAMPLES_PER_CLASS - len(class_images)
                
                images, labels = process_single_pcap(
                    pcap_path, 
                    idx,  # Use the sorted index
                    anonymizer, 
                    needed
                )
                
                class_images.extend(images)
                class_labels.extend(labels)
                
                pbar.update(len(images))
                pbar.set_postfix({
                    'file': os.path.basename(pcap_path)[:20],
                    'collected': len(class_images)
                })
        
        print(f"✓ Class '{label}' complete: {len(class_images)} samples\n")
        
        all_images.extend(class_images)
        all_labels.extend(class_labels)

    print(f"\n{'='*60}")
    print(f"COLLECTION SUMMARY")
    print(f"{'='*60}")
    for idx in sorted(label_map.keys()):
        label = label_map[idx]
        count = sum(1 for l in all_labels if l == idx)
        print(f"  [{idx}] {label}: {count} samples")
    print(f"\nTotal samples: {len(all_images)}")
    print(f"{'='*60}")
    
    if len(all_images) == 0:
        print("ERROR: No samples collected!")
        return
    
    print("\nShuffling...")
    combined = list(zip(all_images, all_labels))
    random.shuffle(combined)
    all_images, all_labels = zip(*combined)

    split_index = int(len(all_images) * TRAIN_RATIO)

    train_images = all_images[:split_index]
    train_labels = all_labels[:split_index]
    test_images = all_images[split_index:]
    test_labels = all_labels[split_index:]

    print("Saving IDX files...")
    save_idx_images(train_images, output_prefix + "_train_images.idx")
    save_idx_labels(train_labels, output_prefix + "_train_labels.idx")
    save_idx_images(test_images, output_prefix + "_test_images.idx")
    save_idx_labels(test_labels, output_prefix + "_test_labels.idx")
    
    # CRITICAL: Save label mapping
    label_mapping_path = output_prefix + "_label_mapping.txt"
    save_label_mapping(label_map, label_mapping_path)

    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Train samples: {len(train_images)}")
    print(f"Test samples: {len(test_images)}")
    print(f"Classes: {len(label_map)}")
    print(f"Files saved to: {output_prefix}_*.idx")
    print(f"Label mapping: {label_mapping_path}")
    print("="*60)
    
    # Print label mapping for verification
    print("\n" + "="*60)
    print("LABEL MAPPING (for verification):")
    print("="*60)
    for idx in sorted(label_map.keys()):
        print(f"  {idx} → {label_map[idx]}")
    print("="*60)

# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    INPUT_ROOT = r"C:\Users\LENOVO\Downloads\pcap_classes"
    OUTPUT_PREFIX = r"C:\Users\LENOVO\Downloads\iscx_vpn"

    process_dataset(INPUT_ROOT, OUTPUT_PREFIX)