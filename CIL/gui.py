"""
gui.py - Streamlit GUI for CIL-MTC Malware Traffic Classification

Features:
- Upload PCAP files or PNG/JPG images
- Automatic PCAP to image conversion
- Real-time prediction with confidence scores
- Beautiful, user-friendly interface

Usage:
    streamlit run gui.py
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
import os
import tempfile
import cv2

# Import your model components
from model import ExpandableFeatureExtractor, ExpandableClassifier
from dataset import load_label_mapping


# ============================================================================
# PCAP TO IMAGE CONVERSION (Pure Python - No External Toolkit!)
# ============================================================================

def pcap_to_image(pcap_path, output_size=(28, 28)):
    """
    Convert PCAP file to grayscale image representation
    
    Args:
        pcap_path: Path to PCAP file
        output_size: Tuple of (height, width) for output image
    
    Returns:
        PIL Image object (grayscale)
    """
    try:
        from scapy.all import rdpcap
    except ImportError:
        st.error("❌ Scapy not installed! Install with: pip install scapy")
        return None
    
    try:
        # Read PCAP file
        packets = rdpcap(pcap_path)
        
        if len(packets) == 0:
            st.warning("⚠️ PCAP file is empty!")
            return None
        
        # Extract bytes from packets
        packet_bytes = []
        for packet in packets:
            try:
                # Get raw packet bytes
                raw_bytes = bytes(packet)
                packet_bytes.extend(raw_bytes[:784])  # Max 784 bytes per packet (28x28)
            except:
                continue
        
        if len(packet_bytes) == 0:
            st.warning("⚠️ No valid packet data found!")
            return None
        
        # Limit to 784 bytes (28x28 image)
        if len(packet_bytes) < 784:
            # Pad with zeros if too short
            packet_bytes.extend([0] * (784 - len(packet_bytes)))
        else:
            # Truncate if too long
            packet_bytes = packet_bytes[:784]
        
        # Convert to numpy array and reshape to 28x28
        img_array = np.array(packet_bytes, dtype=np.uint8).reshape(28, 28)
        
        # Convert to PIL Image
        img = Image.fromarray(img_array, mode='L')
        
        return img
        
    except Exception as e:
        st.error(f"❌ Error processing PCAP: {str(e)}")
        return None


# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_model(backbone_path, classifier_path, num_incremental_tasks, C=16, layers=3):
    """
    Load trained CIL-MTC model
    
    Args:
        backbone_path: Path to backbone weights
        classifier_path: Path to classifier weights
        num_incremental_tasks: Number of incremental tasks
        C: Initial channels
        layers: Number of layers
    
    Returns:
        Tuple of (backbone, classifier, device)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Get number of classes from classifier file
    classifier_state = torch.load(classifier_path, map_location='cpu')
    num_classes = classifier_state['fc.weight'].shape[0]
    
    # Reconstruct model architecture
    backbone = ExpandableFeatureExtractor(C=C, layers=layers)
    
    # Add branches for incremental tasks
    for i in range(num_incremental_tasks):
        backbone.add_new_task_backbone()
    
    classifier = ExpandableClassifier(backbone.out_dim, num_classes)
    
    # Load weights
    try:
        backbone.load_state_dict(torch.load(backbone_path, map_location=device, weights_only=True))
        classifier.load_state_dict(torch.load(classifier_path, map_location=device, weights_only=True))
    except:
        backbone.load_state_dict(torch.load(backbone_path, map_location=device))
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    
    backbone.to(device)
    classifier.to(device)
    
    backbone.eval()
    classifier.eval()
    
    return backbone, classifier, device


# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_image(img_pil, backbone, classifier, device, label_map):
    """
    Predict class for a PIL Image
    
    Args:
        img_pil: PIL Image (grayscale, 28x28)
        backbone: Loaded backbone model
        classifier: Loaded classifier model
        device: torch device
        label_map: Dictionary mapping class indices to names
    
    Returns:
        Tuple of (predicted_class_name, confidence_score, class_idx, all_probs)
    """
    # Ensure image is grayscale and 28x28
    if img_pil.mode != 'L':
        img_pil = img_pil.convert('L')
    
    if img_pil.size != (28, 28):
        img_pil = img_pil.resize((28, 28))
    
    # Convert to tensor
    img_array = np.array(img_pil, dtype=np.float32) / 255.0
    img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    img_tensor = img_tensor.to(device)
    
    # Predict
    with torch.no_grad():
        features = backbone(img_tensor)
        logits = classifier(features)
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
    
    pred_idx = pred_idx.item()
    confidence_score = conf.item() * 100
    
    # Get predicted label name
    if pred_idx in label_map:
        predicted_label = label_map[pred_idx]
    else:
        predicted_label = f"Class {pred_idx}"
    
    # Get all probabilities for top-k display
    all_probs = probs.cpu().numpy()[0]
    
    return predicted_label, confidence_score, pred_idx, all_probs


# ============================================================================
# STREAMLIT GUI
# ============================================================================

def main():
    # Page configuration
    st.set_page_config(
        page_title="CIL-MTC Malware Traffic Classifier",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            color: #1E88E5;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        .prediction-box {
            background-color: #f0f8ff;
            border-left: 5px solid #1E88E5;
            padding: 1.5rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        .confidence-high {
            color: #4CAF50;
            font-weight: bold;
        }
        .confidence-medium {
            color: #FF9800;
            font-weight: bold;
        }
        .confidence-low {
            color: #F44336;
            font-weight: bold;
        }
        .stProgress > div > div > div > div {
            background-color: #1E88E5;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<p class="main-header">🛡️ CIL-MTC Malware Traffic Classifier</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Class Incremental Learning for Network Traffic Classification</p>', unsafe_allow_html=True)
    
    # Sidebar - Configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Model paths
    st.sidebar.subheader("Model Settings")
    
    SCENARIO = st.sidebar.selectbox(
        "Scenario",
        ["T=2", "T=5"],
        help="Select the training scenario used"
    )
    
    backbone_path = f"checkpoints/final_backbone_{SCENARIO.replace('=', '')}.pth"
    classifier_path = f"checkpoints/final_classifier_{SCENARIO.replace('=', '')}.pth"
    label_mapping_path = "E:/Project R1/Processed_IDX/vpn/iscvpn_label_mapping.txt"
    
    # Allow custom paths
    use_custom_paths = st.sidebar.checkbox("Use custom model paths")
    
    if use_custom_paths:
        backbone_path = st.sidebar.text_input("Backbone Path", backbone_path)
        classifier_path = st.sidebar.text_input("Classifier Path", classifier_path)
        label_mapping_path = st.sidebar.text_input("Label Mapping Path", label_mapping_path)
    
    # Architecture parameters
    st.sidebar.subheader("Architecture")
    C = st.sidebar.number_input("Initial Channels (C)", value=16, min_value=8, max_value=64)
    LAYERS = st.sidebar.number_input("Layers", value=3, min_value=1, max_value=10)
    
    # Number of incremental tasks
    if SCENARIO == "T=5":
        NUM_INCREMENTAL_TASKS = 3
    else:  # T=2
        NUM_INCREMENTAL_TASKS = 2
    
    st.sidebar.info(f"📊 Incremental Tasks: {NUM_INCREMENTAL_TASKS}")
    
    # Check if model files exist
    model_exists = os.path.exists(backbone_path) and os.path.exists(classifier_path)
    
    if not model_exists:
        st.error("❌ Model files not found!")
        st.error(f"Expected:\n- {backbone_path}\n- {classifier_path}")
        st.info("💡 Please train the model first using base.py and incmodel.py")
        return
    
    # Load label mapping
    if os.path.exists(label_mapping_path):
        label_map = load_label_mapping(label_mapping_path)
        st.sidebar.success(f"✅ Loaded {len(label_map)} classes")
    else:
        st.sidebar.warning("⚠️ Label mapping not found. Using class indices.")
        label_map = {}
    
    # Load model
    with st.spinner("🔄 Loading model..."):
        try:
            backbone, classifier, device = load_model(
                backbone_path, 
                classifier_path, 
                NUM_INCREMENTAL_TASKS,
                C=C,
                layers=LAYERS
            )
            st.sidebar.success(f"✅ Model loaded on {device}")
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            return
    
    # Main content
    st.markdown("---")
    
    # Input method selection
    input_method = st.radio(
        "📥 Select Input Method:",
        ["Upload PCAP File", "Upload Image (PNG/JPG)", "Use Sample Image"],
        horizontal=True
    )
    
    st.markdown("---")
    
    # Initialize variables
    img_to_predict = None
    input_source = None
    
    # OPTION 1: Upload PCAP
    if input_method == "Upload PCAP File":
        st.subheader("📂 Upload PCAP File")
        
        uploaded_pcap = st.file_uploader(
            "Choose a PCAP file",
            type=['pcap', 'pcapng', 'cap'],
            help="Upload network traffic capture file"
        )
        
        if uploaded_pcap is not None:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pcap') as tmp_file:
                tmp_file.write(uploaded_pcap.getvalue())
                tmp_pcap_path = tmp_file.name
            
            st.info(f"📦 File: {uploaded_pcap.name} ({uploaded_pcap.size} bytes)")
            
            # Convert PCAP to image
            with st.spinner("🔄 Converting PCAP to image..."):
                img_to_predict = pcap_to_image(tmp_pcap_path)
            
            # Clean up temp file
            os.unlink(tmp_pcap_path)
            
            if img_to_predict is not None:
                st.success("✅ PCAP converted successfully!")
                input_source = "PCAP"
            else:
                st.error("❌ Failed to convert PCAP to image")
    
    # OPTION 2: Upload Image
    elif input_method == "Upload Image (PNG/JPG)":
        st.subheader("🖼️ Upload Traffic Image")
        
        uploaded_image = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a 28x28 grayscale traffic image"
        )
        
        if uploaded_image is not None:
            img_to_predict = Image.open(uploaded_image)
            st.info(f"📦 File: {uploaded_image.name}")
            st.success("✅ Image loaded successfully!")
            input_source = "Image"
    
    # OPTION 3: Sample Image
    else:
        st.subheader("🎯 Use Sample Image")
        st.info("Using a random sample image for demonstration")
        
        # Create a random sample image
        sample_array = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
        img_to_predict = Image.fromarray(sample_array, mode='L')
        input_source = "Sample"
    
    # Display and predict
    if img_to_predict is not None:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📷 Input Image")
            
            # Display image (scaled up for visibility)
            display_img = img_to_predict.resize((224, 224), Image.NEAREST)
            st.image(display_img, caption="28x28 Traffic Image", use_container_width=True)
            
            # Image info
            st.caption(f"Mode: {img_to_predict.mode} | Size: {img_to_predict.size}")
        
        with col2:
            st.subheader("🎯 Prediction Results")
            
            # Predict button
            if st.button("🚀 Classify Traffic", type="primary", use_container_width=True):
                with st.spinner("🔮 Analyzing traffic pattern..."):
                    try:
                        # Make prediction
                        pred_label, confidence, class_idx, all_probs = predict_image(
                            img_to_predict,
                            backbone,
                            classifier,
                            device,
                            label_map
                        )
                        
                        # Determine confidence level
                        if confidence >= 90:
                            conf_class = "confidence-high"
                            conf_emoji = "🟢"
                        elif confidence >= 70:
                            conf_class = "confidence-medium"
                            conf_emoji = "🟡"
                        else:
                            conf_class = "confidence-low"
                            conf_emoji = "🔴"
                        
                        # Display prediction
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h2 style="margin-top: 0;">Predicted Class</h2>
                            <h1 style="color: #1E88E5; margin: 0.5rem 0;">🏷️ {pred_label}</h1>
                            <p style="font-size: 1.2rem; margin: 0;">
                                {conf_emoji} Confidence: <span class="{conf_class}">{confidence:.2f}%</span>
                            </p>
                            <p style="color: #666; margin-top: 0.5rem;">Class Index: {class_idx}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence bar
                        st.progress(float(confidence / 100))
                        
                        # Top-5 predictions
                        st.subheader("📊 Top-5 Predictions")
                        
                        top_k = 5
                        top_indices = np.argsort(all_probs)[::-1][:top_k]
                        
                        for rank, idx in enumerate(top_indices, 1):
                            prob = all_probs[idx] * 100
                            class_name = label_map.get(idx, f"Class {idx}")
                            
                            # Color code based on rank
                            if rank == 1:
                                emoji = "🥇"
                            elif rank == 2:
                                emoji = "🥈"
                            elif rank == 3:
                                emoji = "🥉"
                            else:
                                emoji = f"{rank}️⃣"
                            
                            col_a, col_b = st.columns([3, 1])
                            with col_a:
                                st.write(f"{emoji} **{class_name}**")
                            with col_b:
                                st.write(f"{prob:.2f}%")
                            
                            st.progress(float(prob / 100))
                        
                        # Download results
                        st.markdown("---")
                        st.subheader("💾 Export Results")
                        
                        results_text = f"""
Classification Results
======================
Input Source: {input_source}
Predicted Class: {pred_label}
Class Index: {class_idx}
Confidence: {confidence:.2f}%
Model: {SCENARIO}
Device: {device}

Top-5 Predictions:
"""
                        for rank, idx in enumerate(top_indices, 1):
                            prob = all_probs[idx] * 100
                            class_name = label_map.get(idx, f"Class {idx}")
                            results_text += f"{rank}. {class_name}: {prob:.2f}%\n"
                        
                        st.download_button(
                            label="📥 Download Results (TXT)",
                            data=results_text,
                            file_name="classification_results.txt",
                            mime="text/plain"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")
                        st.exception(e)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 2rem;">
            <p><strong>CIL-MTC: Class Incremental Learning for Malware Traffic Classification</strong></p>
            <p>Paper: "Malware Traffic Classification via Expandable Class Incremental Learning With Architecture Search"</p>
            <p>IEEE Transactions on Information Forensics and Security, 2025</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - About
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.info("""
        This tool uses Class Incremental Learning (CIL) 
        to classify network traffic patterns without 
        forgetting previously learned classes.
        
        **Features:**
        - PCAP file processing
        - Image classification
        - Real-time predictions
        - Confidence scores
        - Top-K results
    """)
    
    # Sidebar - Class List
    if label_map:
        st.sidebar.subheader("📋 Trained Classes")
        for idx in sorted(label_map.keys()):
            st.sidebar.text(f"[{idx:2d}] {label_map[idx]}")


if __name__ == "__main__":
    main()
