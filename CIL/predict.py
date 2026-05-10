"""
predict.py - Prediction and Evaluation Script
Works with models trained using base.py and incmodel.py

Compatible with USTC-TK2016 toolkit preprocessing
"""

import torch
import os
from model import ExpandableFeatureExtractor, ExpandableClassifier
from dataset import load_label_mapping, IDXDataset


def predict_single_sample(
    image_source,
    backbone_path,
    classifier_path,
    label_mapping_path,
    trained_classes,
    num_incremental_tasks,
    C=16,
    layers=3
):
    """
    Predict class for a single image
    
    Args:
        image_source: Either (idx_path, image_index) tuple or path to image file
        backbone_path: Path to saved backbone weights
        classifier_path: Path to saved classifier weights
        label_mapping_path: Path to label mapping file
        trained_classes: List of class indices the model was trained on
        num_incremental_tasks: Number of incremental tasks (not including base)
        C: Initial channels (default 16)
        layers: Number of layers (default 3)
    
    Returns:
        predicted_label: Class name
        confidence_score: Confidence percentage
        class_idx: Predicted class index
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load label mapping
    label_map = load_label_mapping(label_mapping_path)
    
    total_classes = len(trained_classes)
    
    # Reconstruct the architecture
    print(f"Reconstructing model architecture...")
    print(f"  Trained classes: {trained_classes}")
    print(f"  Total classes: {total_classes}")
    print(f"  Incremental tasks: {num_incremental_tasks}")
    
    backbone = ExpandableFeatureExtractor(C=C, layers=layers)
    
    # Add branches for incremental tasks
    for i in range(num_incremental_tasks):
        print(f"  Adding branch {i+2}/{num_incremental_tasks+1}...")
        backbone.add_new_task_backbone()
    
    classifier = ExpandableClassifier(backbone.out_dim, total_classes)
    
    print(f"  Backbone output dim: {backbone.out_dim}")
    print(f"  Classifier output: {total_classes} classes")
    
    # Move to device
    backbone.to(device)
    classifier.to(device)
    
    # Load saved weights
    print(f"\nLoading weights...")
    print(f"  Backbone: {backbone_path}")
    print(f"  Classifier: {classifier_path}")
    
    try:
        backbone.load_state_dict(torch.load(backbone_path, map_location=device, weights_only=True))
        classifier.load_state_dict(torch.load(classifier_path, map_location=device, weights_only=True))
    except:
        # Fallback for older PyTorch versions
        backbone.load_state_dict(torch.load(backbone_path, map_location=device))
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    
    backbone.eval()
    classifier.eval()
    
    # Load and preprocess image
    if isinstance(image_source, tuple):
        # Load from IDX file
        idx_images_path, idx_labels_path, image_index = image_source
        print(f"\nLoading image from IDX:")
        print(f"  Images: {idx_images_path}")
        print(f"  Labels: {idx_labels_path}")
        print(f"  Index: {image_index}")
        
        # Load using IDXDataset
        test_ds = IDXDataset(
            idx_images_path,
            idx_labels_path,
            class_filter=trained_classes
        )
        
        if image_index >= len(test_ds):
            print(f"  Warning: Index {image_index} out of range (max: {len(test_ds)-1}), using index 0")
            image_index = 0
        
        img_tensor, true_label = test_ds[image_index]
        print(f"  True label: [{true_label}] {label_map.get(true_label, 'Unknown')}")
        
    else:
        # Load from image file (PNG, JPG, etc.)
        import cv2
        print(f"\nLoading image: {image_source}")
        img = cv2.imread(image_source, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return f"Error: Image not found at {image_source}", 0.0, -1
        
        img = cv2.resize(img, (28, 28))
        img_tensor = torch.tensor(img / 255., dtype=torch.float32).unsqueeze(0)
        true_label = None
    
    img_tensor = img_tensor.unsqueeze(0).to(device)  # Add batch dimension
    
    # Inference
    print(f"\nPerforming inference...")
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
        predicted_label = f"Unknown Class {pred_idx}"
    
    return predicted_label, confidence_score, pred_idx


def batch_evaluate(
    test_images_path,
    test_labels_path,
    backbone_path,
    classifier_path,
    label_mapping_path,
    trained_classes,
    num_incremental_tasks,
    num_samples=None,
    C=16,
    layers=3
):
    """
    Evaluate model on multiple test samples
    
    Args:
        num_samples: Number of samples to evaluate (None = all samples)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load label mapping
    label_map = load_label_mapping(label_mapping_path)
    
    # Load test data with class filter
    print(f"\nLoading test data for classes: {trained_classes}")
    test_ds = IDXDataset(
        test_images_path,
        test_labels_path,
        class_filter=trained_classes
    )
    print(f"  Total test samples available: {len(test_ds)}")
    
    total_classes = len(trained_classes)
    
    # Reconstruct model
    print(f"\nReconstructing model...")
    backbone = ExpandableFeatureExtractor(C=C, layers=layers)
    for i in range(num_incremental_tasks):
        backbone.add_new_task_backbone()
    
    classifier = ExpandableClassifier(backbone.out_dim, total_classes)
    
    print(f"  Backbone dim: {backbone.out_dim}")
    print(f"  Classifier classes: {total_classes}")
    
    backbone.to(device)
    classifier.to(device)
    
    # Load weights
    print(f"\nLoading weights...")
    try:
        backbone.load_state_dict(torch.load(backbone_path, map_location=device, weights_only=True))
        classifier.load_state_dict(torch.load(classifier_path, map_location=device, weights_only=True))
    except:
        # Fallback for older PyTorch versions
        backbone.load_state_dict(torch.load(backbone_path, map_location=device))
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    
    backbone.eval()
    classifier.eval()
    
    # Evaluate
    correct = 0
    total = 0
    class_correct = {}
    class_total = {}
    
    if num_samples is None:
        num_samples = len(test_ds)
    else:
        num_samples = min(num_samples, len(test_ds))
    
    print(f"\nEvaluating on {num_samples} samples...")
    
    with torch.no_grad():
        for i in range(num_samples):
            img, true_label = test_ds[i]
            img = img.unsqueeze(0).to(device)
            
            features = backbone(img)
            logits = classifier(features)
            pred = logits.argmax(dim=1).item()
            
            if pred == true_label:
                correct += 1
                class_correct[true_label] = class_correct.get(true_label, 0) + 1
            
            total += 1
            class_total[true_label] = class_total.get(true_label, 0) + 1
            
            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{num_samples} samples evaluated...")
    
    accuracy = correct / total if total > 0 else 0.0
    per_class_acc = {
        c: class_correct.get(c, 0) / class_total[c]
        for c in class_total
    }
    
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS:")
    print(f"{'='*80}")
    print(f"Overall Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    print(f"\nPer-Class Accuracy:")
    for c in sorted(per_class_acc.keys()):
        class_name = label_map.get(c, f"Class {c}")
        acc = per_class_acc[c]
        count = class_total[c]
        correct_count = class_correct.get(c, 0)
        print(f"  [{c:2d}] {class_name:40s}: {acc*100:6.2f}% ({correct_count}/{count})")
    print(f"{'='*80}\n")
    
    return accuracy, per_class_acc


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    # ========================================================================
    # CONFIGURATION - MUST MATCH YOUR base.py AND incmodel.py!
    # ========================================================================
    
    print("\n" + "="*80)
    print("🔍 PREDICTION & EVALUATION SCRIPT")
    print("="*80)
    
    # ========================================================================
    # SCENARIO SELECTION (MUST MATCH YOUR TRAINING!)
    # ========================================================================
    
    SCENARIO = "T=2"  # Change to match what you used in base.py/incmodel.py
    
    print(f"\nScenario: {SCENARIO}")
    
    # ========================================================================
    # TASK CONFIGURATION (MUST MATCH YOUR TRAINING!)
    # ========================================================================
    
    if SCENARIO == "T=5":
        # T=5 scenario
        # Task 0: [0-5] (6 classes)
        # Task 1: [6-7] (2 classes)
        # Task 2: [8-9] (2 classes)
        # Task 3: [10-11] (2 classes)
        BASE_CLASSES = list(range(6))
        INCREMENTAL_TASKS = [[6, 7], [8, 9], [10, 11]]
    else:  # T=2
        # T=2 scenario
        # Task 0: [0-4] (5 classes)
        # Task 1: [5-7] (3 classes)
        # Task 2: [8-10] (3 classes)
        BASE_CLASSES = list(range(6))
        INCREMENTAL_TASKS = [[6, 7, 8], [9, 10, 11]]
    
    # Combine all trained classes
    TRAINED_CLASSES = BASE_CLASSES.copy()
    for task_classes in INCREMENTAL_TASKS:
        TRAINED_CLASSES.extend(task_classes)
    
    NUM_INCREMENTAL_TASKS = len(INCREMENTAL_TASKS)
    
    print(f"\nTask Configuration:")
    print(f"  Base Task (Task 0): {BASE_CLASSES}")
    for i, task in enumerate(INCREMENTAL_TASKS, 1):
        print(f"  Incremental Task {i}: {task}")
    print(f"\nTotal trained classes: {TRAINED_CLASSES}")
    print(f"Number of incremental tasks: {NUM_INCREMENTAL_TASKS}")
    
    # ========================================================================
    # ARCHITECTURE PARAMETERS (MUST MATCH YOUR TRAINING!)
    # ========================================================================
    
    C = 16      # Initial channels (from base.py/incmodel.py)
    LAYERS = 3  # Number of layers (from base.py/incmodel.py)
    
    print(f"\nArchitecture:")
    print(f"  Initial channels (C): {C}")
    print(f"  Layers: {LAYERS}")
    
    # ========================================================================
    # PATHS - UPDATE THESE TO YOUR ACTUAL PATHS!
    # ========================================================================
    
    # CRITICAL: Update these paths to match your setup
    
    # IDX files from USTC-TK2016 toolkit preprocessing
    IDX_BASE = "E:/Project R1/Processed_IDX/vpn"  # Update this path!
    
    # USTC-TK2016 toolkit uses these naming conventions:
    TEST_IMAGES = f"{IDX_BASE}/test-images-idx3-ubyte.gz"
    TEST_LABELS = f"{IDX_BASE}/test-labels-idx1-ubyte.gz"
    LABEL_MAPPING = f"{IDX_BASE}/label_mapping.txt"
    
    # Model files (saved by incmodel.py)
    BACKBONE_PATH = f"checkpoints/final_backbone_{SCENARIO.replace('=', '')}.pth"
    CLASSIFIER_PATH = f"checkpoints/final_classifier_{SCENARIO.replace('=', '')}.pth"
    
    print(f"\nPaths:")
    print(f"  IDX Base: {IDX_BASE}")
    print(f"  Test Images: {TEST_IMAGES}")
    print(f"  Test Labels: {TEST_LABELS}")
    print(f"  Label Mapping: {LABEL_MAPPING}")
    print(f"  Backbone: {BACKBONE_PATH}")
    print(f"  Classifier: {CLASSIFIER_PATH}")
    
    # ========================================================================
    # CHECK IF FILES EXIST
    # ========================================================================
    
    print("\n" + "="*80)
    print("📁 CHECKING FILES")
    print("="*80)
    
    # Check model files
    if not os.path.exists(BACKBONE_PATH):
        print(f"\n❌ Error: Backbone model not found!")
        print(f"  Expected: {BACKBONE_PATH}")
        print(f"\n💡 Please train the model first:")
        print(f"   1. python base.py")
        print(f"   2. python incmodel.py")
        exit(1)
    
    if not os.path.exists(CLASSIFIER_PATH):
        print(f"\n❌ Error: Classifier model not found!")
        print(f"  Expected: {CLASSIFIER_PATH}")
        print(f"\n💡 Please train the model first:")
        print(f"   1. python base.py")
        print(f"   2. python incmodel.py")
        exit(1)
    
    print(f"✅ Model files found!")
    
    # Check IDX files
    if not os.path.exists(TEST_IMAGES):
        print(f"\n❌ Error: Test images file not found!")
        print(f"  Expected: {TEST_IMAGES}")
        print(f"\n💡 Please preprocess your data using USTC-TK2016 toolkit")
        print(f"   The toolkit should generate files in: {IDX_BASE}")
        exit(1)
    
    if not os.path.exists(TEST_LABELS):
        print(f"\n❌ Error: Test labels file not found!")
        print(f"  Expected: {TEST_LABELS}")
        print(f"\n💡 Please preprocess your data using USTC-TK2016 toolkit")
        exit(1)
    
    print(f"✅ Test data files found!")
    
    # Check label mapping
    if not os.path.exists(LABEL_MAPPING):
        print(f"\n⚠️  Warning: Label mapping file not found!")
        print(f"  Expected: {LABEL_MAPPING}")
        print(f"\n💡 The USTC-TK2016 toolkit should create this file automatically.")
        print(f"   If it doesn't exist, the label indices will be used instead.")
        LABEL_MAPPING = None
    else:
        print(f"✅ Label mapping found!")
        # Print label mapping for verification
        label_map = load_label_mapping(LABEL_MAPPING)
        print(f"\nLabel Mapping ({len(label_map)} classes):")
        for idx in sorted(label_map.keys()):
            print(f"  [{idx:2d}] {label_map[idx]}")
    
    print("="*80)
    
    # ========================================================================
    # OPTION 1: SINGLE SAMPLE PREDICTION FROM IDX
    # ========================================================================
    
    print("\n" + "="*80)
    print("📊 OPTION 1: PREDICT SINGLE SAMPLE FROM IDX TEST SET")
    print("="*80)
    
    IMAGE_INDEX = 0  # Change this to test different samples
    
    label, confidence, class_idx = predict_single_sample(
        image_source=(TEST_IMAGES, TEST_LABELS, IMAGE_INDEX),
        backbone_path=BACKBONE_PATH,
        classifier_path=CLASSIFIER_PATH,
        label_mapping_path=LABEL_MAPPING,
        trained_classes=TRAINED_CLASSES,
        num_incremental_tasks=NUM_INCREMENTAL_TASKS,
        C=C,
        layers=LAYERS
    )
    
    print(f"\n{'='*80}")
    print(f"PREDICTION RESULT:")
    print(f"  Image Index:      {IMAGE_INDEX}")
    print(f"  Predicted Class:  [{class_idx}] {label}")
    print(f"  Confidence:       {confidence:.2f}%")
    print(f"{'='*80}")
    
    # ========================================================================
    # OPTION 2: PREDICT FROM CUSTOM IMAGE FILE
    # ========================================================================
    
    print("\n" + "="*80)
    print("🖼️  OPTION 2: PREDICT FROM CUSTOM IMAGE FILE")
    print("="*80)
    
    # Example: Predict from a custom 28x28 grayscale image
    CUSTOM_IMAGE_PATH = r"E:\Project R1\Processed_IDX\vpn\Test\nonvpn_chat\aim_chat_3b.pcap.UDP_131-202-240-242_51434_131-202-244-32_161.png"
    # CUSTOM_IMAGE_PATH = r"C:\path\to\your\traffic_image.png"
    
    if CUSTOM_IMAGE_PATH and os.path.exists(CUSTOM_IMAGE_PATH):
        label, confidence, class_idx = predict_single_sample(
            image_source=CUSTOM_IMAGE_PATH,
            backbone_path=BACKBONE_PATH,
            classifier_path=CLASSIFIER_PATH,
            label_mapping_path=LABEL_MAPPING,
            trained_classes=TRAINED_CLASSES,
            num_incremental_tasks=NUM_INCREMENTAL_TASKS,
            C=C,
            layers=LAYERS
        )
        
        print(f"\n{'='*80}")
        print(f"CUSTOM IMAGE PREDICTION:")
        print(f"  Image Path:       {os.path.basename(CUSTOM_IMAGE_PATH)}")
        print(f"  Predicted Class:  [{class_idx}] {label}")
        print(f"  Confidence:       {confidence:.2f}%")
        print(f"{'='*80}")
    elif CUSTOM_IMAGE_PATH:
        print(f"\n⚠️  Warning: Custom image not found at {CUSTOM_IMAGE_PATH}")
    else:
        print(f"\n💡 Tip: Set CUSTOM_IMAGE_PATH to test with your own 28x28 grayscale image!")
        print(f"   Example: CUSTOM_IMAGE_PATH = r'C:\\path\\to\\your\\image.png'")
    
    # ========================================================================
    # OPTION 3: BATCH EVALUATION
    # ========================================================================
    
    print("\n" + "="*80)
    print("📈 OPTION 3: BATCH EVALUATION ON TEST SET")
    print("="*80)
    
    NUM_EVAL_SAMPLES = 1000  # Set to None to evaluate all samples
    
    accuracy, per_class_acc = batch_evaluate(
        test_images_path=TEST_IMAGES,
        test_labels_path=TEST_LABELS,
        backbone_path=BACKBONE_PATH,
        classifier_path=CLASSIFIER_PATH,
        label_mapping_path=LABEL_MAPPING,
        trained_classes=TRAINED_CLASSES,
        num_incremental_tasks=NUM_INCREMENTAL_TASKS,
        num_samples=NUM_EVAL_SAMPLES,
        C=C,
        layers=LAYERS
    )
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print("✅ EVALUATION SUMMARY")
    print("="*80)
    print(f"Scenario:              {SCENARIO}")
    print(f"Trained Classes:       {len(TRAINED_CLASSES)} classes {TRAINED_CLASSES}")
    print(f"Incremental Tasks:     {NUM_INCREMENTAL_TASKS}")
    print(f"Architecture:          C={C}, Layers={LAYERS}")
    print(f"Test Accuracy:         {accuracy*100:.2f}%")
    print(f"Samples Evaluated:     {NUM_EVAL_SAMPLES if NUM_EVAL_SAMPLES else 'All'}")
    print("="*80 + "\n")