"""
base.py - Base Task Training (Task 0 only)
Extracted from your main1.py

Run this file first: python base.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import os

# Custom imports
from model import ExpandableFeatureExtractor, ExpandableClassifier, AuxiliaryClassifier
from dataset import IDXDataset, load_label_mapping
from trainer import train_representation, train_classifier_tuning
from loss_tracker import LossTracker


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    """Base task training function"""
    
    # ========================================================================
    # DATASET CONFIGURATION (USTC-TK2016 / ISCVPN)
    # ========================================================================

    # Paths to your preprocessed IDX files
    IDX_BASE = "E:/Project R1/Processed_IDX/vpn"
    
    # Note: If your files are compressed, add '.gz' to the end of these names
    TRAIN_IMAGES = f"{IDX_BASE}/train-images-idx3-ubyte.gz"
    TRAIN_LABELS = f"{IDX_BASE}/train-labels-idx1-ubyte.gz"
    TEST_IMAGES = f"{IDX_BASE}/test-images-idx3-ubyte.gz"
    TEST_LABELS = f"{IDX_BASE}/test-labels-idx1-ubyte.gz"
    LABEL_MAPPING = f"{IDX_BASE}/label_mapping.txt"

    # Load label mapping
    label_map = load_label_mapping(LABEL_MAPPING)
    num_total_available_classes = len(label_map) # Automatically detect 11 or 12 classes
    
    print("\n" + "="*80)
    print(f"LABEL MAPPING (Detected {num_total_available_classes} classes):")
    for idx, name in sorted(label_map.items()):
        print(f"  {idx:2d} → {name}")
    print("="*80)

    # ========================================================================
    # TASK DIVISION - BASE TASK ONLY
    # ========================================================================
    
    print("\n" + "="*80)
    print("🚀 BASE TASK TRAINING (Task 0)")
    print("="*80)
    
    # Simple: Use first classes in order
    all_class_indices = list(range(num_total_available_classes))
    
    # Choose scenario
    SCENARIO = "T=2"  # Change to "T=5" for different scenario
    
    if SCENARIO == "T=5":
        base_task_classes = all_class_indices[0:6]
    else:  # T=2
        base_task_classes = all_class_indices[0:6]

    # Print base task info
    print(f"\nTask 0 [BASE]: {len(base_task_classes)} classes → {base_task_classes}")
    for c in base_task_classes:
        print(f"  - [{c:2d}] {label_map.get(c, 'Unknown')}")
    print(f"{'='*80}\n")

    # ========================================================================
    # HYPERPARAMETERS (Paper Table II)
    # ========================================================================

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # ========================================================================
    # TRAINING PARAMETERS
    # ========================================================================
    print("⚙️  Training Parameters:")
    EPOCHS_STAGE1 = 300              # Representation learning epochs
    EPOCHS_STAGE2 = 300              # Classifier tuning epochs
    MEMORY_SIZE = 500               # Total exemplar memory
    MAX_SAMPLES_PER_CLASS = 5000    # Limit samples per class
    
    print(f"   Stage 1 Epochs: {EPOCHS_STAGE1}")
    print(f"   Stage 2 Epochs: {EPOCHS_STAGE2}")
    print(f"   Memory Size: {MEMORY_SIZE}")
    print(f"   Max Samples/Class: {MAX_SAMPLES_PER_CLASS}")
    print()
    
    # Fixed hyperparameters (Paper Table II)
    BATCH_SIZE = 64                  # Paper Table II
    LR_STAGE1 = 0.025                # Paper Table II: learning rate α
    LR_STAGE2 = 0.025                # Classifier tuning learning rate
    MOMENTUM = 0.9                   # Paper Table II
    WEIGHT_DECAY = 3e-4              # Paper Table II
    TEMPERATURE = 2.0                # Paper Section IV-C2: temperature δ

    # Architecture parameters (Paper Table II)
    C_INIT = 16                      # Initial channels (Paper compliant)
    N_LAYERS = 3                     # Number of layers

    # ========================================================================
    # MODEL INITIALIZATION
    # ========================================================================

    backbone = ExpandableFeatureExtractor(C=C_INIT, layers=N_LAYERS).to(device)
    classifier = ExpandableClassifier(backbone.out_dim, len(base_task_classes)).to(device)

    print(f"Initial model:")
    print(f"  Backbone output dim: {backbone.out_dim}")
    print(f"  Classifier classes: {len(base_task_classes)}")
    print(f"  Total parameters: {count_parameters(backbone) + count_parameters(classifier):,}\n")

    # ========================================================================
    # BASE TASK TRAINING
    # ========================================================================
    
    task_start_time = time.time()
    num_new_classes = len(base_task_classes)
    
    # Paper Algorithm 2, Line 2: λ_a = 0 for first task
    lambda_a = 0.0
    
    # Create auxiliary classifier (Paper Eq. 11)
    aux_classifier = AuxiliaryClassifier(
        backbone.single_branch_dim,
        num_new_classes
    ).to(device)
    
    params = count_parameters(backbone) + count_parameters(classifier)
    print(f"   Total trainable parameters: {params:,}")

    loss_tracker = LossTracker(save_dir='graphs')
    loss_tracker.start_task(0, base_task_classes)

    
    
    # --------------------------------------------------------------------
    # DATA PREPARATION: D̃_t = E_t ∪ D_t (only D_t for base task)
    # --------------------------------------------------------------------
    print(f"\n>> Loading data...")
    
    # Load current task data D_t with sample limit
    current_ds = IDXDataset(
        TRAIN_IMAGES,
        TRAIN_LABELS,
        class_filter=base_task_classes,
        max_samples_per_class=MAX_SAMPLES_PER_CLASS
    )
    print(f"   Current task (D_t): {len(current_ds)} samples")
    
    train_loader = DataLoader(
        current_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    # --------------------------------------------------------------------
    # STAGE 1: REPRESENTATION LEARNING (Algorithm 2, Lines 5-7)
    # --------------------------------------------------------------------
    print(f"\n>> STAGE 1: Representation Learning")
    print(f"   Epochs: {EPOCHS_STAGE1}")
    print(f"   λ_a: {lambda_a}")
    
    optimizer_stage1 = torch.optim.SGD(
        list(backbone.extractors[-1].parameters()) +
        list(classifier.parameters()) +
        list(aux_classifier.parameters()),
        lr=LR_STAGE1,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )
    
    scheduler_stage1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_stage1,
        T_max=EPOCHS_STAGE1
    )
    
    for epoch in range(EPOCHS_STAGE1):
        loss = train_representation(
            backbone=backbone,
            classifier=classifier,
            aux_classifier=aux_classifier,
            loader=train_loader,
            optimizer=optimizer_stage1,
            device=device,
            task_start_idx=0,
            lambda_a=lambda_a
        )
        scheduler_stage1.step()
        
        # Track loss
        loss_tracker.add_stage1_loss(0, epoch+1, loss)
        
        if(epoch + 1) % 20 == 0:
            print(f"   Epoch [{epoch+1}/{EPOCHS_STAGE1}] | Loss: {loss:.4f}")
    
    # --------------------------------------------------------------------
    # STAGE 2: BALANCED CLASSIFIER TUNING (Algorithm 2, Lines 8-10)
    # --------------------------------------------------------------------
    print(f"\n>> STAGE 2: Balanced Classifier Tuning")
    
    new_total_classes = num_new_classes
    
    # For base task, use all training data (no exemplar balancing needed)
    balanced_loader = train_loader
    
    # Reinitialize classifier
    print(f"   Reinitializing classifier for {new_total_classes} classes...")
    classifier = ExpandableClassifier(backbone.out_dim, new_total_classes).to(device)
    
    optimizer_stage2 = torch.optim.Adam(
        classifier.parameters(),
        lr=LR_STAGE2
    )
    
    # Train with temperature δ
    for epoch in range(EPOCHS_STAGE2):
        loss = train_classifier_tuning(
            backbone=backbone,
            classifier=classifier,
            loader=balanced_loader,
            optimizer=optimizer_stage2,
            device=device,
            T=TEMPERATURE,
            task_classes=new_total_classes
        )
        
        # Track loss
        loss_tracker.add_stage2_loss(0, epoch+1, loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch [{epoch+1}/{EPOCHS_STAGE2}] | Loss: {loss:.4f}")
    
    # --------------------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------------------
    print(f"\n>> Evaluating on all {new_total_classes} classes...")
    
    # Load test data for all seen classes
    test_ds = IDXDataset(
        TEST_IMAGES,
        TEST_LABELS,
        class_filter=base_task_classes,
        max_samples_per_class=None
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    backbone.eval()
    classifier.eval()
    
    # Per-class accuracy tracking
    class_correct = {}
    class_total = {}
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            features = backbone(x)
            logits = classifier(features)
            preds = logits.argmax(dim=1)
            
            for gt, pred in zip(y, preds):
                gt_idx = gt.item()
                class_total[gt_idx] = class_total.get(gt_idx, 0) + 1
                if gt_idx == pred.item():
                    class_correct[gt_idx] = class_correct.get(gt_idx, 0) + 1
    
    # Compute per-class accuracy
    acc_dict = {
        c: class_correct.get(c, 0) / class_total[c]
        for c in class_total
    }
    
    # Print per-class results
    print(f"\n   Per-class accuracy:")
    for c in sorted(acc_dict.keys()):
        print(f"     [{c:2d}] {label_map.get(c, 'Unknown'):40s}: {acc_dict[c]*100:6.2f}%")
    
    # Overall accuracy
    total_correct = sum(class_correct.values())
    total_samples = sum(class_total.values())
    overall_acc = (total_correct / total_samples) * 100

    print(f"\\n   Overall Base Task Accuracy: {overall_acc:.2f}%")
        

    
    task_duration = time.time() - task_start_time

    # ========================================================================
    # SAVE BASE MODEL
    # ========================================================================

    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    backbone_path = f"{save_dir}/base_backbone_{SCENARIO.replace('=', '')}.pth"
    classifier_path = f"{save_dir}/base_classifier_{SCENARIO.replace('=', '')}.pth"
    
    # Save configuration
    config_path = f"{save_dir}/base_config_{SCENARIO.replace('=', '')}.pth"
    config = {
        'scenario': SCENARIO,
        'base_task_classes': base_task_classes,
        'num_total_available_classes': num_total_available_classes,
        'current_total_classes': new_total_classes,
        'c_init': C_INIT,
        'n_layers': N_LAYERS,
        'memory_size': MEMORY_SIZE,
        'max_samples_per_class': MAX_SAMPLES_PER_CLASS,
        'base_task_accuracy': overall_acc  # <-- ADD THIS LINE
    }
    # Plot and save loss graphs
    loss_tracker.plot_task_losses(0, SCENARIO)
    loss_tracker.print_summary()
    torch.save(backbone.state_dict(), backbone_path)
    torch.save(classifier.state_dict(), classifier_path)
    torch.save(config, config_path)

    print(f"\n{'='*80}")
    print(f"✅ BASE TASK TRAINING COMPLETED!")
    print(f"{'='*80}")
    print(f"Models saved:")
    print(f"  {backbone_path}")
    print(f"  {classifier_path}")
    print(f"  {config_path}")
    print(f"")
    print(f"Base Task Results:")
    print(f"  Test Accuracy: {overall_acc:.2f}%")
    print(f"  Training Time: {task_duration/60:.2f} minutes")
    print(f"  Total Parameters: {params:,}")
    print(f"{'='*80}\n")
    
    print(f"📌 Next step: Run 'python incmodel.py' for incremental learning")


if __name__ == '__main__':
    main()
