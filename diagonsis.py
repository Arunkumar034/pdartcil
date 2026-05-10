import os
import numpy as np
import cv2
from tqdm import tqdm
from scapy.all import PcapReader, IP, TCP, UDP, ICMP, conf
from collections import defaultdict
import hashlib
import random
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
INPUT_ROOT = r"E:\Edge-IIoTset dataset"
OUTPUT_ROOT = r"E:\Project R1\Processed_Images"

IMAGE_SIDE = 28
IMAGE_SIZE = IMAGE_SIDE * IMAGE_SIDE
MAX_FLOWS_PER_PCAP = 30000

conf.use_pcap = True
conf.sniff_promisc = False


# ---------------- IP ANONYMIZER ----------------
class IPAnonymizer:
    def __init__(self, seed=42):
        self.map = {}
        random.seed(seed)

    def anon(self, ip):
        if ip not in self.map:
            h = int(hashlib.md5(ip.encode()).hexdigest()[:8], 16)
            random.seed(h)
            self.map[ip] = f"{random.randint(1,254)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
        return self.map[ip]


# ---------------- FLOW EXTRACTION ----------------
def extract_unidirectional_flows(pcap_path, max_flows):
    flows = defaultdict(lambda: {
        "packets": [],
        "bytes": 0
    })

    anonymizer = IPAnonymizer()

    with PcapReader(pcap_path) as reader:
        for pkt in reader:
            if IP not in pkt:
                continue

            src_ip = anonymizer.anon(pkt[IP].src)
            dst_ip = anonymizer.anon(pkt[IP].dst)
            proto = pkt[IP].proto

            if TCP in pkt:
                src_port, dst_port = pkt[TCP].sport, pkt[TCP].dport
            elif UDP in pkt:
                src_port, dst_port = pkt[UDP].sport, pkt[UDP].dport
            else:  # ICMP or others
                src_port, dst_port = 0, 0

            key = (src_ip, dst_ip, src_port, dst_port, proto)
            raw = bytes(pkt)

            flows[key]["packets"].append(raw)
            flows[key]["bytes"] += len(raw)

            if len(flows) >= max_flows:
                break

    return [v for v in flows.values() if v["bytes"] > 0]


# ---------------- FLOW → IMAGE ----------------
def flow_to_image(flow):
    byte_stream = bytearray()

    for pkt_bytes in flow["packets"]:
        byte_stream.extend(pkt_bytes)
        if len(byte_stream) >= IMAGE_SIZE:
            break

    if len(byte_stream) < IMAGE_SIZE:
        byte_stream.extend(b"\x00" * (IMAGE_SIZE - len(byte_stream)))
    else:
        byte_stream = byte_stream[:IMAGE_SIZE]

    return np.frombuffer(byte_stream, dtype=np.uint8).reshape(IMAGE_SIDE, IMAGE_SIDE)


# ---------------- PROCESS SINGLE PCAP ----------------
def process_pcap(args):
    pcap_path, label = args

    name = os.path.splitext(os.path.basename(pcap_path))[0]
    out_dir = os.path.join(OUTPUT_ROOT, label, name)
    os.makedirs(out_dir, exist_ok=True)

    flows = extract_unidirectional_flows(pcap_path, MAX_FLOWS_PER_PCAP)

    for i, flow in enumerate(flows):
        img = flow_to_image(flow)
        cv2.imwrite(os.path.join(out_dir, f"{i}.png"), img)

    return name, len(flows)


# ---------------- COLLECT ALL PCAPS ----------------
def collect_pcaps():
    tasks = []

    for root, _, files in os.walk(INPUT_ROOT):
        for f in files:
            if f.endswith(".pcap"):
                label = "normal" if "Normal traffic" in root else "attack"
                tasks.append((os.path.join(root, f), label))

    return tasks


# ---------------- MAIN ----------------
if __name__ == "__main__":
    multiprocessing.freeze_support()
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    pcaps = collect_pcaps()

    print("=" * 80)
    print(f"PARALLEL PAPER-ALIGNED PREPROCESSING")
    print(f"PCAP files found: {len(pcaps)}")
    print("=" * 80)

    max_workers = 2
    print(f"⚙ Using {max_workers} parallel processes\n")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_pcap, p) for p in pcaps]

        for future in as_completed(futures):
            name, count = future.result()
            print(f"✅ {name}: {count} images")

    print("\n🎉 ALL FILES PROCESSED")
