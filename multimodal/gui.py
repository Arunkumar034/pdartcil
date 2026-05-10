"""
Streamlit UI for Multimodal Network Traffic Classification
===========================================================
"""

import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt

# ── Project paths ──────────────────────────────────────────────────────────────
_HERE  = Path(__file__).resolve().parent
_ROOT  = _HERE.parent
_CIL   = _ROOT / "CIL"
_SMART = _ROOT / "smartdetector"

for _p in [str(_ROOT), str(_CIL), str(_SMART)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from multi import (
        MultimodalFusionNetwork,
        AutomatedPreprocessor,
        IMG_SIZE, SAM_K, SAM_B, SAM_C,
    )
except ImportError:
    st.error("Could not import from multi.py. Make sure it is in the Python path.")
    st.stop()

# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = str(_ROOT / "checkpoints" / "multimodal_best.pth")

# Every folder-name variant the pipeline has ever produced
_IMG_FOLDER_CANDIDATES = ["4_Png", "3_images", "4_png"]
_SAM_FOLDER_CANDIDATES = ["4_sams", "4_Sam", "4_sam", "4_Sams"]

FALLBACK_CLASS_NAMES = [
    "Benign", "Miuref", "Neris", "Virut", "Htbot",
    "Geodo", "Shifu", "Tinba", "Nsis-ay", "Zeus",
    "nonvpn_chat", "nonvpn_email", "nonvpn_file_transfer",
    "nonvpn_p2p", "nonvpn_streaming", "nonvpn_voip",
    "vpn_bittorrent", "vpn_chat", "vpn_email", "vpn_file_transfer",
]

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Network Traffic Multimodal Classifier",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header { font-size:2.5rem; font-weight:bold; color:#1f77b4;
                   text-align:center; margin-bottom:2rem; }
    .sub-header  { font-size:1.2rem; color:#666; text-align:center;
                   margin-bottom:2rem; }
    .prediction-card { background:#000000; padding:1.5rem; border-radius:10px;
                        border-left:5px solid #1f77b4; margin:1rem 0; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DIRECTORY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _find_root(base: Path, candidates: List[str]) -> Optional[Path]:
    """Return the first candidate sub-folder that exists under *base*."""
    for name in candidates:
        p = base / name
        if p.is_dir():
            return p
    return None


def _find_img_sam_roots(base: str) -> Tuple[Optional[Path], Optional[Path]]:
    b = Path(base)
    return _find_root(b, _IMG_FOLDER_CANDIDATES), _find_root(b, _SAM_FOLDER_CANDIDATES)


def _list_png_npy(directory: Path) -> Tuple[List[Path], List[Path]]:
    """Recursively collect all .png and .npy files under *directory*."""
    pngs = sorted(directory.rglob("*.png"))
    npys = sorted(directory.rglob("*.npy"))
    return pngs, npys


def scan_processed_dir(processed_dir: str) -> Tuple[List[Dict], Dict]:
    """
    Walk processed_dir and pair every .png with its matching .npy by stem.

    Returns:
        samples  – list of dicts  {image, sam, class, split, name}
        report   – diagnostic info dict

    Strategy:
      1. Find img_root (4_Png / 3_images …) and sam_root (4_sams / 4_Sam …).
      2. Build a stem→path lookup for ALL .npy files under sam_root.
      3. Walk every .png under img_root; derive class / split from the
         path relative to img_root.
      4. Look up the matching .npy by stem (case-insensitive).
      5. If no SAM found for a PNG, still record it as image-only so the
         user sees counts and understands what is missing.
    """
    img_root, sam_root = _find_img_sam_roots(processed_dir)

    report = {
        "img_root":       str(img_root) if img_root else None,
        "sam_root":       str(sam_root) if sam_root else None,
        "total_png":      0,
        "total_npy":      0,
        "paired":         0,
        "unpaired_png":   [],   # PNGs with no matching SAM
        "missing_sam_classes": set(),
    }

    if img_root is None:
        return [], report

    # Build stem → Path map for all SAMs (handle any nesting depth)
    sam_by_stem: Dict[str, Path] = {}
    if sam_root:
        for npy in sam_root.rglob("*.npy"):
            sam_by_stem[npy.stem.lower()] = npy
        report["total_npy"] = len(sam_by_stem)

    pngs = sorted(img_root.rglob("*.png"))
    report["total_png"] = len(pngs)

    samples: List[Dict] = []

    for png in pngs:
        # Derive split and class from path relative to img_root
        rel_parts = png.relative_to(img_root).parts
        # Expected layouts:
        #   Train/<class>/<stem>.png   → split=Train, class=parts[0-based index 0], or
        #   <class>/<stem>.png         → split=All
        if len(rel_parts) == 3:          # split / class / file
            split_label = rel_parts[0]
            cls_name    = rel_parts[1]
        elif len(rel_parts) == 2:        # class / file  (flat, no split)
            split_label = "All"
            cls_name    = rel_parts[0]
        else:                            # unexpected depth — skip
            continue

        stem    = png.stem.lower()
        sam_path = sam_by_stem.get(stem)

        entry = {
            "image":     str(png),
            "sam":       str(sam_path) if sam_path else None,
            "class":     cls_name,
            "split":     split_label,
            "name":      png.name,
            "has_sam":   sam_path is not None,
        }
        samples.append(entry)

        if sam_path:
            report["paired"] += 1
        else:
            report["unpaired_png"].append(png.name[:60])
            report["missing_sam_classes"].add(cls_name)

    return samples, report


def derive_class_names(processed_dir: str) -> List[str]:
    """Sorted class names derived from the image folder — matches multi.py."""
    img_root, _ = _find_img_sam_roots(processed_dir)
    if img_root is None:
        return []
    for sub in ["Train", "Test", ""]:
        target = img_root / sub if sub else img_root
        if target.is_dir():
            classes = sorted(d.name for d in target.iterdir() if d.is_dir())
            if classes:
                return classes
    return []


def debug_dir_tree(base: str, max_lines: int = 80) -> str:
    lines     = []
    base_path = Path(base)
    for root, dirs, files in os.walk(base_path):
        depth  = len(Path(root).relative_to(base_path).parts)
        indent = "  " * depth
        lines.append(f"{indent}{Path(root).name}/")
        for f in sorted(files)[:4]:
            lines.append(f"{indent}  {f}")
        if len(files) > 4:
            lines.append(f"{indent}  … ({len(files)} files)")
        dirs[:] = sorted(dirs)
        if len(lines) >= max_lines:
            lines.append("  … (truncated)")
            break
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model(checkpoint_path: str, n_classes: int, device: str
               ) -> Tuple[MultimodalFusionNetwork, str, bool]:
    """
    Returns (model, status_message, checkpoint_found).
    ALL st.* calls are intentionally kept OUTSIDE this function because
    @st.cache_resource replays cached calls in a different layout context,
    which causes the 'element called on layout block created outside' error.
    """
    model = MultimodalFusionNetwork(
        n_classes=n_classes,
        visual_C=32, visual_layers=6,
        semantic_embed_dim=2048,
        fusion_hidden1=512, fusion_hidden2=256,
        dropout=0.4,
    )
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt)
        status  = "Checkpoint loaded successfully."
        ckpt_ok = True
    else:
        status  = (f"Checkpoint not found at {checkpoint_path!r}. "
                   "Model is using random weights.")
        ckpt_ok = False
    model.eval()
    return model.to(device), status, ckpt_ok


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def load_sample_tensors(image_path: str, sam_path: Optional[str], device: str
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load one PNG and (if available) its SAM .npy.
    If SAM is missing, a zero tensor of the correct shape is used so the
    model can still run (semantic branch will output near-zero features).
    """
    img     = Image.open(image_path).convert("L")
    img_arr = np.array(img, dtype=np.float32) / 255.0
    img_t   = torch.tensor(img_arr).unsqueeze(0).unsqueeze(0)   # (1,1,28,28)

    if sam_path and os.path.exists(sam_path):
        sam_arr = np.load(sam_path).astype(np.float32)
    else:
        # Zero SAM fallback — shape must be (SAM_C, SAM_K, SAM_B) = (3, 40, 100)
        sam_arr = np.zeros((SAM_C, SAM_K, SAM_B), dtype=np.float32)

    sam_t = torch.tensor(sam_arr).unsqueeze(0)                  # (1,3,40,100)
    return img_t.to(device), sam_t.to(device)


def predict_single(model, image: torch.Tensor, sam: torch.Tensor,
                   class_names: List[str]) -> Dict:
    with torch.no_grad():
        logits = model(image, sam)
        probs  = F.softmax(logits, dim=1)
        pred_i = torch.argmax(probs, dim=1).item()
        n_out  = probs.shape[1]

        if len(class_names) != n_out:
            class_names = [str(i) for i in range(n_out)]

        all_probs = {class_names[i]: probs[0][i].item() for i in range(n_out)}
        return {
            "prediction":          class_names[pred_i],
            "confidence":          probs[0][pred_i].item(),
            "all_probabilities":   all_probs,
            "predicted_class_idx": pred_i,
        }


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def prob_chart(probabilities: Dict[str, float], predicted_class: str) -> go.Figure:
    classes = list(probabilities.keys())
    probs   = list(probabilities.values())
    colors  = ["#ff7f0e" if c == predicted_class else "#1f77b4" for c in classes]
    fig = go.Figure(go.Bar(
        x=classes, y=probs, marker_color=colors,
        text=[f"{p:.3f}" for p in probs], textposition="auto",
    ))
    fig.update_layout(
        title="Class Probability Distribution",
        xaxis_title="Traffic Class", yaxis_title="Probability",
        yaxis_range=[0, 1], template="plotly_white", height=400,
        xaxis_tickangle=-45,
    )
    return fig


def show_image_and_sam(image_path: str, sam_path: Optional[str]):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🖼️ Visual Input (28×28)")
        st.image(Image.open(image_path), caption="Network Traffic Grayscale Image",
                 use_column_width=True)
    with col2:
        st.subheader("📊 Semantic Input (SAM Matrix)")
        if sam_path and os.path.exists(sam_path):
            sam_data = np.load(sam_path)
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            for i, (ax, name) in enumerate(zip(axes, ["Size", "Time", "Direction"])):
                im = ax.imshow(sam_data[i], cmap="viridis", aspect="auto")
                ax.set_title(f"Ch {i+1}: {name}")
                ax.set_xlabel("Embedding Dim")
                ax.set_ylabel("Packet Pos")
                plt.colorbar(im, ax=ax)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.warning("⚠️ No SAM file found for this sample — zero SAM used for inference.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.markdown('<div class="main-header">🔒 Network Traffic Multimodal Classifier</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Visual + Semantic Fusion · Inference UI</div>',
                unsafe_allow_html=True)

    # ── Session state ──────────────────────────────────────────────────────────
    for key, default in [("processed_dir", None), ("model", None),
                         ("samples", []), ("class_names", []),
                         ("scan_report", {})]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── Sidebar ────────────────────────────────────────────────────────────────
    st.sidebar.header("⚙️ Configuration")

    checkpoint_path = st.sidebar.text_input(
        "Model Checkpoint Path", value=DEFAULT_CHECKPOINT)

    dev_opt = st.sidebar.selectbox("Device", ["Auto", "CUDA", "CPU"])
    device  = ("cuda" if torch.cuda.is_available() else "cpu") \
              if dev_opt == "Auto" else dev_opt.lower()
    st.sidebar.info(f"Device: **{device.upper()}**")

    st.sidebar.subheader("Class Labels Directory")
    class_dir = st.sidebar.text_input(
        "Load Classes from Path", 
        value=r"C:\Users\LENOVO\Downloads\processed_multimodal_dataset\4_Png\Train"
    )
    
    loaded_from_dir = False
    if class_dir and os.path.exists(class_dir):
        det = sorted([d.name for d in Path(class_dir).iterdir() if d.is_dir()])
        if det:
            st.session_state.class_names = det
            loaded_from_dir = True

    # Resolve class names from selected processed directory (if path above wasn't used/valid)
    if not loaded_from_dir and st.session_state.processed_dir:
        det = derive_class_names(st.session_state.processed_dir)
        if det:
            st.session_state.class_names = det

    if st.session_state.class_names:
        class_names = st.session_state.class_names
        st.sidebar.success(f"✅ {len(class_names)} classes auto-detected")
        with st.sidebar.expander("View class list"):
            for i, c in enumerate(class_names):
                st.sidebar.write(f"{i}: {c}")
    else:
        raw = st.sidebar.text_area(
            "Class Names (comma-separated)",
            value=", ".join(FALLBACK_CLASS_NAMES))
        class_names = [c.strip() for c in raw.split(",") if c.strip()]
        st.sidebar.warning("⚠️ Using fallback class names.")

    n_classes = len(class_names)

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📁 Load Dataset", "🔍 Prediction", "📊 Batch Analysis"])

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 1 — Load Dataset
    # ═══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.header("📁 Load Processed Dataset")

        st.info("""
        **How to use:**
        - Enter the path to your `processed_multimodal_dataset` folder.
        - It must contain **`4_Png`** (images) and **`4_sams`** (SAM matrices).
        - Each must have `Train/<class>/` and/or `Test/<class>/` subfolders.
        - The scanner auto-pairs `.png` files with matching `.npy` files by filename stem.
        """)

        existing_dir = st.text_input(
            "📂 Processed Dataset Directory Path",
            value=st.session_state.processed_dir or "",
            placeholder=r"C:\Users\LENOVO\Downloads\processed_multimodal_dataset",
        )

        col_scan, col_clear = st.columns([1, 1])
        do_scan  = col_scan.button("🔍 Scan & Load", type="primary")
        do_clear = col_clear.button("🗑️ Clear", type="secondary")

        if do_clear:
            st.session_state.processed_dir = None
            st.session_state.samples       = []
            st.session_state.class_names   = []
            st.session_state.scan_report   = {}
            st.rerun()

        if do_scan and existing_dir:
            ep = Path(existing_dir)
            if not ep.is_dir():
                st.error(f"❌ Directory not found: `{existing_dir}`")
            else:
                with st.spinner("Scanning directory…"):
                    samples, report = scan_processed_dir(existing_dir)

                st.session_state.processed_dir = existing_dir
                st.session_state.samples       = samples
                st.session_state.scan_report   = report

                det = derive_class_names(existing_dir)
                if det:
                    st.session_state.class_names = det
                    class_names = det
                    n_classes   = len(class_names)

        # ── Always show results if we have a loaded dir ────────────────────────
        if st.session_state.processed_dir and st.session_state.scan_report:
            rep     = st.session_state.scan_report
            samples = st.session_state.samples

            st.markdown("---")
            st.subheader("📊 Scan Results")

            # Folder detection status
            c1, c2 = st.columns(2)
            with c1:
                if rep["img_root"]:
                    st.success(f"✅ Image folder: `{Path(rep['img_root']).name}`")
                else:
                    st.error(f"❌ Image folder not found (tried: {_IMG_FOLDER_CANDIDATES})")
            with c2:
                if rep["sam_root"]:
                    st.success(f"✅ SAM folder: `{Path(rep['sam_root']).name}`")
                else:
                    st.error(f"❌ SAM folder not found (tried: {_SAM_FOLDER_CANDIDATES})")

            # Count metrics
            paired_samples   = [s for s in samples if s["has_sam"]]
            unpaired_samples = [s for s in samples if not s["has_sam"]]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total PNGs found",    rep["total_png"])
            m2.metric("Total SAMs found",    rep["total_npy"])
            m3.metric("✅ Paired (PNG+SAM)",  len(paired_samples))
            m4.metric("⚠️ PNG only (no SAM)", len(unpaired_samples))

            if len(paired_samples) == 0 and len(unpaired_samples) > 0:
                st.error(
                    "❌ **No SAM files could be matched to the PNG files.**\n\n"
                    "The SAM `.npy` filenames must match the PNG filenames exactly "
                    "(same stem, different extension). For example:\n"
                    "```\n"
                    "4_Png/Train/Benign/Benign.pcap.TCP_1-1-0-104_2941.png\n"
                    "4_sams/Train/Benign/Benign.pcap.TCP_1-1-0-104_2941.npy\n"
                    "```\n"
                    "**Tip:** Use the directory tree below to verify your SAM folder structure."
                )
                # Show SAM folder tree for debugging
                if rep["sam_root"]:
                    with st.expander("🗂️ SAM folder structure (debug)"):
                        st.code(debug_dir_tree(rep["sam_root"], 60))
                with st.expander("🗂️ Image folder structure (debug)"):
                    st.code(debug_dir_tree(rep["img_root"], 60))

            elif len(unpaired_samples) > 0:
                st.warning(
                    f"⚠️ {len(unpaired_samples)} PNGs have no matching SAM. "
                    "They will use a **zero SAM** for inference (reduced accuracy)."
                )
                with st.expander(f"Show {min(20, len(unpaired_samples))} unmatched PNG names"):
                    st.write(rep["unpaired_png"][:20])

            if samples:
                # Class breakdown
                st.subheader("Class Breakdown")
                df_all = pd.DataFrame(samples)
                breakdown = (
                    df_all.groupby(["class", "split", "has_sam"])
                    .size()
                    .reset_index(name="count")
                )
                st.dataframe(breakdown, use_container_width=True)

                # Sample preview
                with st.expander("📋 Sample preview (first 20 rows)"):
                    st.dataframe(
                        df_all[["split","class","name","has_sam","image","sam"]].head(20),
                        use_container_width=True,
                    )

                st.success(
                    f"✅ Dataset ready: **{len(paired_samples)}** fully paired samples "
                    f"+ **{len(unpaired_samples)}** image-only samples · "
                    f"**{len(set(s['class'] for s in samples))}** classes · "
                    f"splits: {sorted(set(s['split'] for s in samples))}"
                )

        st.markdown("---")
        st.subheader("🚀 Run Preprocessing (raw PCAP → images + SAMs)")
        st.caption("Only needed if you have raw PCAP files. Requires Wireshark + SplitCap.")

        input_method = st.radio("Input Method", ["Local Directory Path", "Upload ZIP"],
                                horizontal=True)
        pcap_source  = None

        if input_method == "Upload ZIP":
            uploaded = st.file_uploader("Upload PCAP ZIP", type=["zip"])
            if uploaded:
                tmpdir   = tempfile.mkdtemp()
                zip_path = os.path.join(tmpdir, "upload.zip")
                with open(zip_path, "wb") as f:
                    f.write(uploaded.getvalue())
                extract_dir = os.path.join(tmpdir, "extracted")
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(extract_dir)
                pcap_source = extract_dir
                st.success(f"✅ Extracted to `{extract_dir}`")
        else:
            pcap_source = st.text_input(
                "PCAP Directory Path",
                placeholder=r"C:\Users\LENOVO\Downloads\pcap_classes",
            )

        if pcap_source and st.button("▶️ Start Preprocessing", type="primary"):
            pb          = st.progress(0)
            output_base = tempfile.mkdtemp(prefix="multimodal_processed_")
            with st.spinner("Processing…"):
                try:
                    preprocessor = AutomatedPreprocessor(
                        raw_pcap_dir=pcap_source,
                        output_base_dir=output_base,
                    )
                    pb.progress(10, "Initialising…")
                    preprocessor.run_pipeline()
                    pb.progress(100, "Done!")
                    samples, report = scan_processed_dir(output_base)
                    st.session_state.processed_dir = output_base
                    st.session_state.samples       = samples
                    st.session_state.scan_report   = report
                    det = derive_class_names(output_base)
                    if det:
                        st.session_state.class_names = det
                    st.success(f"✅ Done! {len(samples)} samples found.")
                except Exception as e:
                    st.error(f"Preprocessing failed: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 2 — Single Sample Prediction
    # ═══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.header("🔍 Single Sample Prediction")

        # Load model
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🔄 Load Model", type="primary"):
                with st.spinner("Loading…"):
                    try:
                        # load_model() returns (model, status_msg, ckpt_ok).
                        # st.* calls are intentionally here (outside the cached
                        # function) to avoid the layout-block replay error.
                        model_obj, status_msg, ckpt_ok = load_model(
                            checkpoint_path, n_classes, device)
                        st.session_state.model = model_obj
                        if ckpt_ok:
                            st.success(f"✅ {status_msg}")
                            st.toast("✅ Checkpoint loaded successfully", icon="✅")
                        else:
                            st.warning(f"⚠️ {status_msg}")
                    except Exception as e:
                        st.error(f"Failed: {e}")
        with col2:
            if st.session_state.model is not None:
                st.info(
                    f"Model on **{device.upper()}** · {n_classes} classes · "
                    f"Labels: {', '.join(class_names[:6])}{'…' if n_classes > 6 else ''}"
                )
            else:
                st.warning("⚠️ Model not loaded yet. Click 'Load Model'.")

        input_source = st.radio("Prediction Source", ["From Loaded Dataset", "Upload Raw PCAP"], horizontal=True)
        st.markdown("---")
        
        if input_source == "From Loaded Dataset":
            if not st.session_state.samples:
                st.info("👆 Go to 'Load Dataset' tab and scan your dataset first.")
            else:
                samples = st.session_state.samples
                st.subheader("Select a Sample")

                # Filters
                c1, c2, c3 = st.columns(3)
                with c1:
                    splits    = sorted(set(s["split"] for s in samples))
                    sel_split = st.selectbox("Split", ["All splits"] + splits)
                with c2:
                    pool = samples if sel_split == "All splits" \
                           else [s for s in samples if s["split"] == sel_split]
                    classes   = sorted(set(s["class"] for s in pool))
                    sel_class = st.selectbox("Class", ["All classes"] + classes)
                with c3:
                    sam_filter = st.selectbox("SAM availability",
                                              ["All", "Has SAM only", "Missing SAM only"])

                # Apply filters
                filtered = pool
                if sel_class != "All classes":
                    filtered = [s for s in filtered if s["class"] == sel_class]
                if sam_filter == "Has SAM only":
                    filtered = [s for s in filtered if s["has_sam"]]
                elif sam_filter == "Missing SAM only":
                    filtered = [s for s in filtered if not s["has_sam"]]

                if not filtered:
                    st.warning("No samples match the current filters.")
                else:
                    sel = st.selectbox(
                        f"Sample ({len(filtered)} available)",
                        filtered,
                        format_func=lambda x: (
                            f"{'✅' if x['has_sam'] else '⚠️'}  "
                            f"[{x['split']}] {x['class']} / {x['name']}"
                        ),
                    )

                    if not sel["has_sam"]:
                        st.warning(
                            "⚠️ This sample has no matching SAM file. "
                            "Inference will use a zero SAM — accuracy may be reduced."
                        )

                    if st.session_state.model and st.button("🔮 Run Prediction", type="primary"):
                        with st.spinner("Analysing…"):
                            img_t, sam_t = load_sample_tensors(
                                sel["image"], sel["sam"], device)
                            result = predict_single(
                                st.session_state.model, img_t, sam_t, class_names)

                        st.markdown("---")
                        c1, c2 = st.columns(2)
                        with c1:
                            show_image_and_sam(sel["image"], sel["sam"])
                        with c2:
                            st.subheader("🎯 Prediction Results")
                            pred, conf = result["prediction"], result["confidence"]
                            icon   = "🟢" if conf > 0.8 else ("🟡" if conf > 0.5 else "🔴")
                            status = "High" if conf > 0.8 else ("Medium" if conf > 0.5 else "Low")
                            st.markdown(f"""
                            <div class="prediction-card">
                                <h3>{icon} Predicted: <b>{pred}</b></h3>
                                <h4>Confidence: {conf:.2%}  ({status})</h4>
                            </div>""", unsafe_allow_html=True)

                            true_cls = sel["class"]
                            match    = pred == true_cls
                            st.markdown(
                                f"**Ground Truth:** `{true_cls}`  \n"
                                f"**Result:** {'✅ Correct' if match else '❌ Incorrect'}"
                            )
                            st.plotly_chart(
                                prob_chart(result["all_probabilities"], pred),
                                use_container_width=True)

                        with st.expander("📋 Full probability table"):
                            df = pd.DataFrame([
                                {"Class": k,
                                 "Probability": round(v, 6),
                                 "Percentage": f"{v*100:.3f}%"}
                                for k, v in sorted(
                                    result["all_probabilities"].items(),
                                    key=lambda x: x[1], reverse=True)
                            ])
                            st.dataframe(df, use_container_width=True)
                    elif not st.session_state.model:
                        st.warning("⚠️ Load the model first (button above).")

        else:
            st.subheader("Upload raw PCAP for Prediction")
            uploaded_pcap = st.file_uploader("Upload a Single PCAP file", type=["pcap", "pcapng"])
            if uploaded_pcap:
                if st.session_state.model and st.button("🔮 Preprocess & Predict", type="primary"):
                    with st.spinner("Preprocessing raw PCAP and analysing..."):
                        tmpdir = tempfile.mkdtemp(prefix="single_pcap_")
                        
                        # Create dummy "Unknown" folder structure for automated preprocessor
                        raw_dir = os.path.join(tmpdir, "raw", "Unknown")
                        os.makedirs(raw_dir, exist_ok=True)
                        pcap_path = os.path.join(raw_dir, uploaded_pcap.name)
                        with open(pcap_path, "wb") as f:
                            f.write(uploaded_pcap.getvalue())
                        
                        out_dir = os.path.join(tmpdir, "processed")
                        os.makedirs(out_dir, exist_ok=True)
                        
                        try:
                            # Preprocess
                            preprocessor = AutomatedPreprocessor(
                                raw_pcap_dir=os.path.join(tmpdir, "raw"),
                                output_base_dir=out_dir
                            )
                            preprocessor.run_pipeline()
                            
                            # Scan outputs
                            samples, _ = scan_processed_dir(out_dir)
                            
                            if not samples:
                                st.error("❌ No valid sessions were extracted from the PCAP.")
                            else:
                                st.success(f"✅ Preprocessing successful! Extracted {len(samples)} session(s). Using the first mapped session.")
                                
                                # Use prediction on the first generated sample image/sam pair
                                sel = samples[0]
                                if not sel["has_sam"]:
                                    st.warning("⚠️ This session has no associated SAM file. Using a 'zero-SAM' array during inference.")
                                
                                img_t, sam_t = load_sample_tensors(sel["image"], sel["sam"], device)
                                result = predict_single(st.session_state.model, img_t, sam_t, class_names)
                                
                                st.markdown("---")
                                c1, c2 = st.columns(2)
                                with c1:
                                    show_image_and_sam(sel["image"], sel["sam"])
                                with c2:
                                    st.subheader("🎯 Prediction Results")
                                    pred, conf = result["prediction"], result["confidence"]
                                    icon   = "🟢" if conf > 0.8 else ("🟡" if conf > 0.5 else "🔴")
                                    status = "High" if conf > 0.8 else ("Medium" if conf > 0.5 else "Low")
                                    st.markdown(f"""
                                    <div class="prediction-card">
                                        <h3>{icon} Predicted: <b>{pred}</b></h3>
                                        <h4>Confidence: {conf:.2%}  ({status})</h4>
                                    </div>""", unsafe_allow_html=True)
                                    
                                    st.plotly_chart(
                                        prob_chart(result["all_probabilities"], pred),
                                        use_container_width=True)
                                
                                with st.expander("📋 Full probability table"):
                                    df = pd.DataFrame([
                                        {"Class": k,
                                         "Probability": round(v, 6),
                                         "Percentage": f"{v*100:.3f}%"}
                                        for k, v in sorted(
                                            result["all_probabilities"].items(),
                                            key=lambda x: x[1], reverse=True)
                                    ])
                                    st.dataframe(df, use_container_width=True)
                                    
                        except Exception as e:
                            st.error(f"Error during preprocessing: {e}")
                elif not st.session_state.model:
                    st.warning("⚠️ Load the model first (button above).")

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 3 — Batch Analysis
    # ═══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.header("📊 Batch Analysis")

        if not st.session_state.samples:
            st.info("👆 Load a dataset first.")
        elif st.session_state.model is None:
            st.warning("⚠️ Load the model first (go to Prediction tab).")
        else:
            samples = st.session_state.samples

            # Batch filters
            c1, c2 = st.columns(2)
            with c1:
                splits    = sorted(set(s["split"] for s in samples))
                batch_split = st.selectbox("Split to analyse",
                                           ["All splits"] + splits, key="batch_split")
            with c2:
                batch_sam = st.selectbox("SAM filter",
                                         ["All", "Has SAM only"], key="batch_sam")

            pool = samples if batch_split == "All splits" \
                   else [s for s in samples if s["split"] == batch_split]
            if batch_sam == "Has SAM only":
                pool = [s for s in pool if s["has_sam"]]

            max_n = st.slider("Max samples", 1, max(1, len(pool)),
                              min(100, len(pool)))

            if st.button("🚀 Run Batch Prediction", type="primary"):
                pb   = st.progress(0)
                todo = pool[:max_n]
                rows = []

                for i, s in enumerate(todo):
                    pb.progress((i + 1) / len(todo),
                                f"{i+1}/{len(todo)}: {s['name'][:50]}")
                    img_t, sam_t = load_sample_tensors(s["image"], s["sam"], device)
                    res = predict_single(
                        st.session_state.model, img_t, sam_t, class_names)
                    rows.append({
                        "name":            s["name"],
                        "split":           s["split"],
                        "true_class":      s["class"],
                        "predicted_class": res["prediction"],
                        "confidence":      round(res["confidence"], 4),
                        "correct":         res["prediction"] == s["class"],
                        "has_sam":         s["has_sam"],
                    })

                df  = pd.DataFrame(rows)
                acc = df["correct"].mean()

                # Summary metrics
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total",     len(rows))
                c2.metric("Accuracy",  f"{acc:.2%}")
                c3.metric("Correct",   int(df["correct"].sum()))
                c4.metric("Incorrect", len(rows) - int(df["correct"].sum()))

                # Per-SAM accuracy
                if df["has_sam"].any() and (~df["has_sam"]).any():
                    s1, s2 = st.columns(2)
                    sam_acc  = df[df["has_sam"]]["correct"].mean()
                    zero_acc = df[~df["has_sam"]]["correct"].mean()
                    s1.metric("Acc (with SAM)",   f"{sam_acc:.2%}")
                    s2.metric("Acc (zero SAM)",   f"{zero_acc:.2%}")

                st.subheader("Confusion Matrix")
                st.dataframe(
                    pd.crosstab(df["true_class"], df["predicted_class"], margins=True),
                    use_container_width=True)

                st.subheader("Per-Class Performance")
                ca = (df.groupby("true_class")["correct"]
                        .agg(["count", "sum", "mean"])
                        .reset_index())
                ca.columns = ["Class", "Total", "Correct", "Accuracy"]
                ca["Accuracy"] = ca["Accuracy"].apply(lambda x: f"{x:.2%}")
                st.dataframe(ca, use_container_width=True)

                st.subheader("Confidence Distribution")
                fig = px.histogram(
                    df, x="confidence", color="correct", nbins=20,
                    title="Prediction Confidence Distribution",
                    color_discrete_map={True: "#2ecc71", False: "#e74c3c"})
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("📋 All results"):
                    st.dataframe(df, use_container_width=True)
                    st.download_button(
                        "📥 Download CSV", df.to_csv(index=False),
                        "batch_results.csv", "text/csv")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#666;'>"
        "Multimodal Fusion Network · Inference UI · Built with Streamlit"
        "</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()