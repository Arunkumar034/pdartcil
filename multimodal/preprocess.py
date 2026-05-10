"""
preprocess.py – Standalone Multimodal Preprocessing Pipeline
=============================================================
Extracts all preprocessing steps from multimodal.py into a
self-contained script that can be run independently.

Pipeline Steps:
  1. PCAP Merging         – mergecap (Wireshark) merges all per-class PCAPs
  2. Session Splitting    – SplitCap.exe splits merged PCAPs into sessions
  3. Train/Test Split     – 2_ProcessSession.ps1 filters & splits sessions
  4. Image Generation     – 3_Session2Png.py generates 28x28 greyscale PNGs
  5. SAM Matrix Gen       – SmartDetector SAMConstructor builds 3×40×100 SAMs
  6. Dataset Migration    – Outputs organised into <output_dir>/3_images & 4_sams

Usage:
  # Activate your conda environment first, e.g.:
  #   conda activate r1
  python preprocess.py --data_dir C:/path/to/pcap_classes --output_dir C:/path/to/output

  # Optionally supply the path to cached Word2Vec embeddings (speeds up SAM gen):
  python preprocess.py --data_dir ... --output_dir ... --embed_dir E:/Project R1/checkpoints/embeddings
"""

import os
import sys
import shutil
import subprocess
import argparse
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ── Resolve project root paths ───────────────────────────────────────────────
_HERE  = Path(__file__).resolve().parent   # …/multimodal/
_ROOT  = _HERE.parent                      # …/Project R1/
_SMART = _ROOT / "smartdetector"
_USTC  = _ROOT / "USTC-TK2016-master"

for _p in [str(_ROOT), str(_SMART)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── SmartDetector imports ────────────────────────────────────────────────────
try:
    from sam_matrix_construct import SAMConstructor, load_embedding_dicts, train_embedding_dictionaries
    from dataset import load_flows_from_pcap
except ImportError as e:
    raise ImportError(
        f"Cannot import SmartDetector modules from {_SMART}.\n"
        f"Original error: {e}"
    )

# ── Constants (must match multimodal.py) ─────────────────────────────────────
SAM_K = 40    # packets per flow
SAM_B = 100   # embedding dimension / bytes
SAM_C = 3     # SAM channels (length, direction, IAT)

# ── Wireshark tools ───────────────────────────────────────────────────────────
_WIRESHARK = Path(r"C:\Program Files\Wireshark")
MERGECAP   = _WIRESHARK / "mergecap.exe"
EDITCAP    = _WIRESHARK / "editcap.exe"
SPLITCAP   = _USTC / "0_Tool" / "SplitCap_2-1" / "SplitCap.exe"


# ══════════════════════════════════════════════════════════════════════════════
# ── STEP 1: Merge raw PCAPs per class ─────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def merge_pcaps(raw_dir: Path, pcap_1_dir: Path) -> list:
    """
    For each class sub-folder in *raw_dir*, merge all .pcap/.pcapng files
    into a single ``<class>.pcap`` in *pcap_1_dir*.

    Returns the list of discovered class names.
    """
    classes = [c.name for c in raw_dir.iterdir() if c.is_dir()]
    if not classes:
        raise RuntimeError(f"[MERGE] No class sub-folders found in {raw_dir}")

    for cls_name in classes:
        pcap_files = (
            list((raw_dir / cls_name).glob("*.pcap")) +
            list((raw_dir / cls_name).glob("*.pcapng"))
        )
        if not pcap_files:
            print(f"  [!] No PCAP files for class '{cls_name}'. Skipping.")
            continue

        dst = pcap_1_dir / f"{cls_name}.pcap"
        print(f"\n[{cls_name}] Merging {len(pcap_files)} file(s) → {dst.name}")

        if MERGECAP.exists():
            temp_files = []

            # Stage 1 – convert each file to legacy pcap format via editcap
            if EDITCAP.exists():
                print(f"  → Converting {len(pcap_files)} file(s) to standard pcap…")
                for i, f in enumerate(pcap_files):
                    tmp = pcap_1_dir / f"tmp_{cls_name}_{i}.pcap"
                    cmd = [str(EDITCAP), "-F", "pcap", str(f), str(tmp)]
                    try:
                        subprocess.run(cmd, check=True, capture_output=True)
                        temp_files.append(str(tmp))
                    except subprocess.CalledProcessError as e:
                        print(f"  [!] editcap failed for {f.name}: "
                              f"{e.stderr.decode(errors='replace')}")
                        temp_files.append(str(f))
            else:
                print(f"  [!] editcap not found at {EDITCAP}. Skipping conversion.")
                temp_files = [str(f) for f in pcap_files]

            # Stage 2 – merge converted files
            print("  → Merging files…")
            cmd = [str(MERGECAP), "-F", "pcap", "-w", str(dst)] + temp_files
            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                print(f"  [!] mergecap failed for '{cls_name}': "
                      f"{e.stderr.decode(errors='replace')}")
                if not dst.exists():
                    shutil.copy(pcap_files[0], dst)

            # Cleanup temp files
            for tmp in temp_files:
                if "tmp_" in tmp:
                    try:
                        os.unlink(tmp)
                    except Exception:
                        pass
        else:
            print(f"  [!] mergecap not found at {MERGECAP}. "
                  f"Copying first file as fallback: {pcap_files[0].name}")
            shutil.copy(pcap_files[0], dst)

    return classes


# ══════════════════════════════════════════════════════════════════════════════
# ── STEP 2: Split merged PCAPs into bidirectional sessions ────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def split_sessions(classes: list, pcap_1_dir: Path, session_dir: Path) -> None:
    """
    Run SplitCap.exe on every merged PCAP to produce per-session files
    stored in *session_dir*/AllLayers/<class>/.
    """
    all_layers_dir = session_dir / "AllLayers"
    os.makedirs(all_layers_dir, exist_ok=True)

    if not SPLITCAP.exists():
        raise FileNotFoundError(
            f"[SPLIT] SplitCap.exe not found at {SPLITCAP}.\n"
            f"Please place SplitCap.exe inside 'USTC-TK2016-master/0_Tool/SplitCap_2-1/'."
        )

    for cls_name in classes:
        pcap_file = pcap_1_dir / f"{cls_name}.pcap"
        if not pcap_file.exists():
            print(f"  [!] Merged PCAP not found for '{cls_name}'. Skipping.")
            continue

        target_out = all_layers_dir / cls_name
        os.makedirs(target_out, exist_ok=True)

        print(f"\n[{cls_name}] Splitting {pcap_file.name} into sessions…")
        cmd = [
            str(SPLITCAP),
            "-p", "50000",
            "-b", "50000",
            "-s", "session",
            "-r", str(pcap_file),
            "-o", str(target_out),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            # Remove empty session files (same behaviour as 1_Pcap2Session.ps1)
            for f in target_out.glob("*"):
                if f.is_file() and f.stat().st_size == 0:
                    try:
                        f.unlink()
                    except Exception:
                        pass
        except subprocess.CalledProcessError as e:
            print(f"  [!] SplitCap failed for '{cls_name}': {e.stderr}")


# ══════════════════════════════════════════════════════════════════════════════
# ── STEP 3: Train/Test split and session trimming (PowerShell) ────────────────
# ══════════════════════════════════════════════════════════════════════════════

def run_process_session(ustc_dir: Path) -> bool:
    """Execute 2_ProcessSession.ps1 which filters lengths and creates Train/Test."""
    script = ustc_dir / "2_ProcessSession.ps1"
    print(f"\n[STEP 3] Running {script.name} …")
    cmd = ["powershell.exe", "-ExecutionPolicy", "Bypass", "-File", str(script)]
    try:
        result = subprocess.run(
            cmd, cwd=str(ustc_dir), check=True, capture_output=True, text=True
        )
        print(f"  [+] {script.name} completed successfully.")
        if result.stdout:
            print(result.stdout[-500:])   # last 500 chars to avoid wall-of-text
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [!] {script.name} failed:\n{e.stderr}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# ── STEP 4: 28×28 grayscale PNG generation ────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def run_session2png(ustc_dir: Path) -> bool:
    """Execute 3_Session2Png.py which produces 28×28 greyscale images in 4_Png/."""
    script = ustc_dir / "3_Session2Png.py"
    print(f"\n[STEP 4] Running {script.name} …")
    cmd = [sys.executable, str(script)]
    try:
        subprocess.run(
            cmd, cwd=str(ustc_dir), check=True, capture_output=True, text=True
        )
        print(f"  [+] {script.name} completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [!] {script.name} failed:\n{e.stderr}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# ── STEP 5: SAM matrix generation ────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _init_sam_constructor(ustc_dir: Path, embed_dir: str) -> object:
    """
    Load pre-trained Word2Vec embeddings from *embed_dir* if they exist,
    otherwise train new ones from the TrimedSession/Train PCAP data.
    """
    print("\n[STEP 5] Initialising SAM Constructor …")

    if embed_dir and os.path.exists(embed_dir):
        try:
            Dz, Da = load_embedding_dicts(embed_dir)
            constructor = SAMConstructor(Dz, Da, k=SAM_K, b=SAM_B)
            print(f"  [+] Loaded pre-trained embeddings from: {embed_dir}")
            return constructor
        except Exception as e:
            print(f"  [!] Could not load embeddings: {e}")

    print("  [*] Training new Word2Vec embeddings from Train sessions…")
    train_src = ustc_dir / "3_ProcessedSession" / "TrimedSession" / "Train"
    if not train_src.exists():
        print(f"  [!] TrimedSession/Train not found at {train_src}. Cannot train embeddings.")
        return None

    all_flows = []
    for cls_dir in [d for d in train_src.iterdir() if d.is_dir()]:
        session_files = list(cls_dir.glob("*"))[:200]
        for sf in session_files:
            try:
                flows = load_flows_from_pcap(str(sf), cls_dir.name)
                all_flows.extend(flows)
            except Exception:
                pass

    if not all_flows:
        print("  [!] No flows found for training embeddings. SAM step will be skipped.")
        return None

    Dz, Da = train_embedding_dictionaries(all_flows, embedding_dim=SAM_B, epochs=5)
    constructor = SAMConstructor(Dz, Da, k=SAM_K, b=SAM_B)

    if embed_dir:
        os.makedirs(embed_dir, exist_ok=True)
        with open(os.path.join(embed_dir, "Dz.pkl"), "wb") as f:
            pickle.dump(Dz, f)
        with open(os.path.join(embed_dir, "Da.pkl"), "wb") as f:
            pickle.dump(Da, f)
        print(f"  [+] Saved new embeddings to: {embed_dir}")

    return constructor


def generate_sam_matrices(ustc_dir: Path, embed_dir: str) -> None:
    """
    Walk through all PNGs generated in 4_Png/, find the corresponding session
    file in FilteredSession/, build a SAM matrix with SAMConstructor, and save
    a .npy file in 4_Sam/ at the same relative path.

    If a session file cannot be parsed, the orphan PNG is deleted so that
    (image, SAM) pairs remain strictly aligned.
    """
    constructor = _init_sam_constructor(ustc_dir, embed_dir)
    if constructor is None:
        print("  [!] SAM Constructor unavailable — skipping SAM generation.")
        return

    base_png_dir     = ustc_dir / "4_Png"
    base_session_dir = ustc_dir / "3_ProcessedSession" / "FilteredSession"
    base_sam_dir     = ustc_dir / "4_Sam"

    for split in ["Train", "Test"]:
        png_split = base_png_dir / split
        ses_split = base_session_dir / split
        sam_split = base_sam_dir / split

        if not png_split.exists() or not ses_split.exists():
            print(f"  [!] Skipping {split} — png or session dir missing.")
            continue

        classes = [d.name for d in png_split.iterdir() if d.is_dir()]
        for cls_name in classes:
            png_cls = png_split / cls_name
            ses_cls = ses_split / cls_name
            sam_cls = sam_split / cls_name
            os.makedirs(sam_cls, exist_ok=True)

            png_files = list(png_cls.glob("*.png"))
            print(f"\n  [{split} – {cls_name}] Generating SAMs for "
                  f"{len(png_files)} session(s)…")

            for png_f in tqdm(png_files, desc=f"  {cls_name}", leave=False):
                stem         = png_f.stem
                session_file = ses_cls / stem
                fallback     = ses_cls / f"{stem}.pcap"

                target = session_file if session_file.exists() else fallback

                if target.exists():
                    try:
                        flows = load_flows_from_pcap(str(target), cls_name)
                        if flows:
                            sam_batch  = constructor.batch_flows_to_sam(
                                flows, verbose=False
                            )
                            sam_matrix = sam_batch[0]   # shape (3, SAM_K, SAM_B)
                            np.save(sam_cls / f"{stem}.npy", sam_matrix)
                        else:
                            # No usable flows → discard image to keep alignment
                            png_f.unlink()
                    except Exception:
                        png_f.unlink()
                else:
                    # Session file lost → discard image
                    png_f.unlink()

    print("\n  [+] SAM matrix generation complete.")


# ══════════════════════════════════════════════════════════════════════════════
# ── STEP 6: Copy final outputs to target directory ───────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def migrate_outputs(ustc_dir: Path, output_dir: Path) -> None:
    """
    Copy 4_Png → <output_dir>/3_images
    Copy 4_Sam → <output_dir>/4_sams
    """
    src_png = ustc_dir / "4_Png"
    src_sam = ustc_dir / "4_Sam"
    dst_img = output_dir / "3_images"
    dst_sam = output_dir / "4_sams"

    print(f"\n[STEP 6] Migrating outputs to: {output_dir}")

    if dst_img.exists():
        shutil.rmtree(dst_img)
    if dst_sam.exists():
        shutil.rmtree(dst_sam)

    shutil.copytree(src_png, dst_img)
    shutil.copytree(src_sam, dst_sam)

    # Count final aligned samples
    total_img = sum(1 for _ in dst_img.rglob("*.png"))
    total_sam = sum(1 for _ in dst_sam.rglob("*.npy"))
    print(f"  [+] Copied {total_img} images  → {dst_img}")
    print(f"  [+] Copied {total_sam} SAM files → {dst_sam}")


# ══════════════════════════════════════════════════════════════════════════════
# ── FULL PIPELINE ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(raw_pcap_dir: str, output_dir: str, embed_dir: str = None) -> None:
    """
    Execute all 6 preprocessing steps end-to-end.

    Args:
        raw_pcap_dir : Root containing one sub-folder per class, each with
                       .pcap / .pcapng files.
                       e.g.  C:/data/raw/
                                Benign/  *.pcap
                                Miuref/  *.pcap  …
        output_dir   : Destination for the final dataset.
                       Will contain 3_images/Train|Test/<class>/*.png
                       and          4_sams/Train|Test/<class>/*.npy
        embed_dir    : Optional path to cached Word2Vec embeddings (Dz.pkl / Da.pkl).
                       Pass None to always train fresh embeddings.
    """
    raw_dir    = Path(raw_pcap_dir).resolve()
    out_dir    = Path(output_dir).resolve()
    ustc_dir   = _USTC
    pcap_1_dir = ustc_dir / "1_Pcap"

    print("\n" + "=" * 70)
    print("  Multimodal Preprocessing Pipeline")
    print("=" * 70)
    print(f"  Raw PCAP dir   : {raw_dir}")
    print(f"  Output dir     : {out_dir}")
    print(f"  USTC-TK dir    : {ustc_dir}")
    print(f"  Embed cache    : {embed_dir or '(will train fresh)'}")
    print("=" * 70 + "\n")

    # ── Prepare USTC-TK working directories ──────────────────────────────────
    print("[INIT] Clearing previous USTC-TK working directories…")
    for folder_name in ["1_Pcap", "2_Session", "3_ProcessedSession", "4_Png", "4_Sam"]:
        d = ustc_dir / folder_name
        if d.exists():
            try:
                shutil.rmtree(d)
            except Exception as ex:
                print(f"  [!] Could not remove {d}: {ex}")
        os.makedirs(d, exist_ok=True)

    os.makedirs(out_dir, exist_ok=True)

    # ── Step 1: Merge PCAPs ──────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("[STEP 1] Merging raw PCAPs per class…")
    print("─" * 50)
    classes = merge_pcaps(raw_dir, pcap_1_dir)
    print(f"\n  [+] Found {len(classes)} class(es): {classes}")

    # ── Step 2: Split sessions ───────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("[STEP 2] Splitting sessions with SplitCap…")
    print("─" * 50)
    split_sessions(classes, pcap_1_dir, ustc_dir / "2_Session")

    # ── Step 3: Train/Test split (PowerShell) ────────────────────────────────
    print("\n" + "─" * 50)
    print("[STEP 3] Train/Test split & session trimming…")
    print("─" * 50)
    if not run_process_session(ustc_dir):
        print("[ABORT] 2_ProcessSession.ps1 failed. Stopping pipeline.")
        return

    # ── Step 4: PNG image generation ─────────────────────────────────────────
    print("\n" + "─" * 50)
    print("[STEP 4] Generating 28×28 grayscale PNG images…")
    print("─" * 50)
    if not run_session2png(ustc_dir):
        print("[ABORT] 3_Session2Png.py failed. Stopping pipeline.")
        return

    # ── Step 5: SAM matrix generation ────────────────────────────────────────
    print("\n" + "─" * 50)
    print("[STEP 5] Generating SAM matrices (3×40×100)…")
    print("─" * 50)
    generate_sam_matrices(ustc_dir, embed_dir)

    # ── Step 6: Migrate to output directory ──────────────────────────────────
    print("\n" + "─" * 50)
    migrate_outputs(ustc_dir, out_dir)

    print("\n" + "=" * 70)
    print("  ✅  Preprocessing complete!")
    print(f"  Dataset ready at: {out_dir}")
    print("=" * 70 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# ── COMMAND LINE INTERFACE ────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Multimodal Preprocessing Pipeline\n"
            "Converts raw per-class PCAP files into aligned 28×28 PNG images\n"
            "and 3×40×100 SAM matrices ready for multimodal.py training.\n\n"
            "Example:\n"
            "  conda activate r1\n"
            "  python preprocess.py \\\\\n"
            "      --data_dir  C:/data/raw_pcaps \\\\\n"
            "      --output_dir C:/data/processed \\\\\n"
            "      --embed_dir  E:/Project R1/checkpoints/embeddings"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help=(
            "Root directory containing one sub-folder per class, each with "
            ".pcap or .pcapng files.  "
            "Example: C:/data/raw_pcaps  (must contain Benign/, Miuref/, … inside)"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Destination directory for the finished dataset "
            "(3_images/ and 4_sams/).  "
            "Defaults to <data_dir>/../processed_multimodal_dataset"
        ),
    )
    parser.add_argument(
        "--embed_dir",
        type=str,
        default=str(_ROOT / "checkpoints" / "embeddings"),
        help=(
            "Path to a folder containing pre-trained Word2Vec embeddings "
            "(Dz.pkl, Da.pkl).  If the folder exists and contains these files "
            "they are reused; otherwise new ones are trained and saved here.  "
            f"Default: {_ROOT / 'checkpoints' / 'embeddings'}"
        ),
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    data_dir   = args.data_dir
    output_dir = args.output_dir or str(Path(data_dir).parent / "processed_multimodal_dataset")
    embed_dir  = args.embed_dir

    run_pipeline(
        raw_pcap_dir=data_dir,
        output_dir=output_dir,
        embed_dir=embed_dir,
    )


if __name__ == "__main__":
    main()
