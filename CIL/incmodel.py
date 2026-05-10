"""
incmodel.py - Incremental Task Training (Task 1, 2, 3, ...)
Extracted from your main1.py

Run this file after base.py: python incmodel.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import time
import os

# Custom imports
from model import ExpandableFeatureExtractor, ExpandableClassifier, AuxiliaryClassifier
from dataset import IDXDataset, MemoryDataset, load_label_mapping
from trainer import train_representation, train_classifier_tuning
from exemplar import select_exemplars
from metrics import IncrementalMetrics
from loss_tracker import LossTracker


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    """Incremental tasks training function"""
    
    # ========================================================================
    # DATASET CONFIGURATION (USTC-TK2016 / ISCVPN)
    # ========================================================================

    # Paths to your preprocessed IDX files
    IDX_BASE = "E:/Project R1/Processed_IDX/vpn"
    
    TRAIN_IMAGES = f"{IDX_BASE}/train-images-idx3-ubyte.gz"
    TRAIN_LABELS = f"{IDX_BASE}/train-labels-idx1-ubyte.gz"
    TEST_IMAGES = f"{IDX_BASE}/test-images-idx3-ubyte.gz"
    TEST_LABELS = f"{IDX_BASE}/test-labels-idx1-ubyte.gz"
    LABEL_MAPPING = f"{IDX_BASE}/label_mapping.txt"

    # Load label mapping
    label_map = load_label_mapping(LABEL_MAPPING)
    num_total_available_classes = len(label_map)
    
    print("\n" + "="*80)
    print(f"LABEL MAPPING (Detected {num_total_available_classes} classes):")
    for idx, name in sorted(label_map.items()):
        print(f"  {idx:2d} → {name}")
    print("="*80)

    # ========================================================================
    # LOAD BASE MODEL AND CONFIGURATION
    # ========================================================================
    
    print("\n" + "="*80)
    print("🚀 INCREMENTAL TASK TRAINING")
    print("="*80)
    
    # Detect scenario from saved config
    SCENARIO = "T=2"  # Should match what you used in base.py
    
    config_path = f"checkpoints/base_config_{SCENARIO.replace('=', '')}.pth"
    
    if not os.path.exists(config_path):
        print(f"\n❌ ERROR: Base model config not found at {config_path}")
        print("   Please run 'python base.py' first!")
        return
    
    # Load configuration
    config = torch.load(config_path)
    base_task_classes = config['base_task_classes']
    C_INIT = config['c_init']
    N_LAYERS = config['n_layers']
    MEMORY_SIZE = config['memory_size']
    MAX_SAMPLES_PER_CLASS = config['max_samples_per_class']
    
    print(f"\n✅ Loaded base configuration:")
    print(f"   Scenario: {SCENARIO}")
    print(f"   Base task classes: {base_task_classes}")
    print(f"   Architecture: C={C_INIT}, Layers={N_LAYERS}")

    # ========================================================================
    # TASK DIVISION - INCREMENTAL TASKS ONLY
    # ========================================================================
    
    all_class_indices = list(range(num_total_available_classes))
    
    # Define incremental tasks based on scenario
    if SCENARIO == "T=5":
        remaining_classes = all_class_indices[6:12]
        incremental_tasks = [
            remaining_classes[i:i+2] for i in range(0, len(remaining_classes), 2)
        ]
    else:  # T=2
        incremental_tasks = [
            all_class_indices[6:9],
            all_class_indices[9:12]
        ]
    
    # Print task division
    print(f"\n{'='*80}")
    print(f"TASK DIVISION ({SCENARIO}):")
    print(f"{'='*80}")
    print(f"Task 0 [BASE] (already trained): {base_task_classes}")
    for t, task_classes in enumerate(incremental_tasks, 1):
        print(f"Task {t} [INCREMENTAL {t}]: {task_classes}")
        for c in task_classes:
            print(f"  - [{c:2d}] {label_map.get(c, 'Unknown')}")
    print(f"{'='*80}\n")

    # ========================================================================
    # HYPERPARAMETERS (Paper Table II)
    # ========================================================================

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Training parameters
    print("⚙️  Training Parameters:")
    EPOCHS_STAGE1 = 300
    EPOCHS_STAGE2 = 300
    
    print(f"   Stage 1 Epochs: {EPOCHS_STAGE1}")
    print(f"   Stage 2 Epochs: {EPOCHS_STAGE2}")
    print(f"   Memory Size: {MEMORY_SIZE}")
    print(f"   Max Samples/Class: {MAX_SAMPLES_PER_CLASS}")
    print()
    
    # Fixed hyperparameters (Paper Table II)
    BATCH_SIZE = 64
    LR_STAGE1 = 0.025
    LR_STAGE2 = 0.025
    MOMENTUM = 0.9
    WEIGHT_DECAY = 3e-4
    TEMPERATURE = 2.0

    # ========================================================================
    # LOAD BASE MODEL
    # ========================================================================
    
    print(">> Loading base model...")
    
    backbone_path = f"checkpoints/base_backbone_{SCENARIO.replace('=', '')}.pth"
    classifier_path = f"checkpoints/base_classifier_{SCENARIO.replace('=', '')}.pth"
    
    if not os.path.exists(backbone_path):
        print(f"❌ ERROR: Base backbone not found at {backbone_path}")
        return
    # Initialize loss tracker
    loss_tracker = LossTracker(save_dir='graphs')
    
    # Dictionary to store ALL task accuracies (including base task)
    task_accuracies = {}
    
    # Load base task accuracy from config
    if 'base_task_accuracy' in config:
        task_accuracies[0] = config['base_task_accuracy']
        print(f"   Base task accuracy: {task_accuracies[0]:.2f}%")
    else:
        print("   ⚠️  Warning: Base task accuracy not found in config!")

    
    # Initialize models
    backbone = ExpandableFeatureExtractor(C=C_INIT, layers=N_LAYERS).to(device)
    classifier = ExpandableClassifier(backbone.out_dim, len(base_task_classes)).to(device)
    
    # Load weights
    backbone.load_state_dict(torch.load(backbone_path, map_location=device))
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    
    print(f"   ✅ Loaded from {backbone_path}")
    print(f"   ✅ Loaded from {classifier_path}\n")

    # ========================================================================
    # METRICS TRACKING
    # ========================================================================

    total_classes = len(base_task_classes) + sum(len(task) for task in incremental_tasks)
    
    metrics = IncrementalMetrics(
        total_tasks=1 + len(incremental_tasks),  # Base + incremental
        total_classes=total_classes
    )

    # ========================================================================
    # BUILD INITIAL EXEMPLAR MEMORY FROM BASE TASK
    # ========================================================================
    
    print(">> Building initial exemplar memory from base task...")
    
    exemplar_list = []
    current_total_classes = len(base_task_classes)
    m = MEMORY_SIZE // current_total_classes
    
    for class_idx in base_task_classes:
        class_ds = IDXDataset(
            TRAIN_IMAGES,
            TRAIN_LABELS,
            class_filter=[class_idx],
            max_samples_per_class=MAX_SAMPLES_PER_CLASS
        )
        
        selected = select_exemplars(
            backbone=backbone,
            dataset=class_ds,
            m=min(m, len(class_ds)),
            device=device
        )
        
        exemplar_list.extend(selected)
    
    print(f"   Initial memory: {len(exemplar_list)} exemplars ({m} per class)\n")
    
    training_start = time.time()


    # ========================================================================
    # INCREMENTAL LEARNING LOOP (Paper Algorithm 2)
    # ========================================================================

    for t, task_classes in enumerate(incremental_tasks, 1):
        task_type = f"INCREMENTAL TASK {t}"
        
        print("\n" + "="*80)
        print(f"TASK {t}/{len(incremental_tasks)} [{task_type}]")
        loss_tracker.start_task(t, task_classes)
        print(f"Classes: {task_classes}")
        print(f"Class names: {[label_map.get(c, 'Unknown') for c in task_classes]}")
        print("="*80)
        
        task_start_time = time.time()
        num_new_classes = len(task_classes)
        
        # Paper Algorithm 2, Line 2: λ_a = 0.5 for incremental tasks
        lambda_a = 0.5
        
        # --------------------------------------------------------------------
        # EXPAND MODEL (Algorithm 2, Lines 3-4)
        # --------------------------------------------------------------------
        print(f">> Expanding model for {num_new_classes} new classes...")
        backbone.add_new_task_backbone()
        backbone.to(device)
        classifier.expand(backbone.out_dim, num_new_classes)
        classifier.to(device)
        print(f"   New backbone dim: {backbone.out_dim}")
        print(f"   New classifier size: {classifier.fc.out_features} classes")
        
        # Create auxiliary classifier (Paper Eq. 11)
        aux_classifier = AuxiliaryClassifier(
            backbone.single_branch_dim,
            num_new_classes
        ).to(device)
        
        params = count_parameters(backbone) + count_parameters(classifier)
        print(f"   Total trainable parameters: {params:,}")
        
        # --------------------------------------------------------------------
        # DATA PREPARATION: D̃_t = E_t ∪ D_t
        # --------------------------------------------------------------------
        print(f"\n>> Loading data...")
        
        # Load current task data D_t with sample limit
        current_ds = IDXDataset(
            TRAIN_IMAGES,
            TRAIN_LABELS,
            class_filter=task_classes,
            max_samples_per_class=MAX_SAMPLES_PER_CLASS
        )
        print(f"   Current task (D_t): {len(current_ds)} samples")
        
        # Combine with exemplar memory E_t
        memory_ds = MemoryDataset(exemplar_list)
        train_ds = ConcatDataset([current_ds, memory_ds])
        print(f"   Exemplar memory (E_t): {len(memory_ds)} samples")
        print(f"   Combined dataset (D̃_t): {len(train_ds)} samples")
        
        train_loader = DataLoader(
            train_ds,
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
                task_start_idx=current_total_classes,
                lambda_a=lambda_a
            )
            scheduler_stage1.step()
            
            # Track loss
            loss_tracker.add_stage1_loss(t, epoch+1, loss)
            
            if(epoch + 1) % 20 == 0:
                print(f"   Epoch [{epoch+1}/{EPOCHS_STAGE1}] | Loss: {loss:.4f}")
        
        # --------------------------------------------------------------------
        # STAGE 2: BALANCED CLASSIFIER TUNING (Algorithm 2, Lines 8-10)
        # --------------------------------------------------------------------
        print(f"\n>> STAGE 2: Balanced Classifier Tuning")
        
        new_total_classes = current_total_classes + num_new_classes
        m = MEMORY_SIZE // new_total_classes
        print(f"   Exemplars per class (m): {m}")
        
        # Create class-balanced subset
        balanced_samples = []
        
        # Sample m exemplars from each OLD class
        for cid in range(current_total_classes):
            old_class_exemplars = [e for e in exemplar_list if e[1] == cid][:m]
            balanced_samples.extend(old_class_exemplars)
        print(f"   Old class samples: {len(balanced_samples)}")
        
        # Sample m samples from each NEW class
        new_class_count = 0
        for class_idx in task_classes:
            class_ds = IDXDataset(
                TRAIN_IMAGES, 
                TRAIN_LABELS, 
                class_filter=[class_idx],
                max_samples_per_class=MAX_SAMPLES_PER_CLASS
            )
            for i in range(min(m, len(class_ds))):
                img, lbl = class_ds[i]
                balanced_samples.append((img, lbl))
                new_class_count += 1
        print(f"   New class samples: {new_class_count}")
        print(f"   Total balanced samples: {len(balanced_samples)}")
        
        balanced_loader = DataLoader(
            MemoryDataset(balanced_samples),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
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
            loss_tracker.add_stage2_loss(t, epoch+1, loss)
            
            if (epoch + 1) % 5 == 0:
                print(f"   Epoch [{epoch+1}/{EPOCHS_STAGE2}] | Loss: {loss:.4f}")
        
        # --------------------------------------------------------------------
        # MEMORY UPDATE: HERDING
        # --------------------------------------------------------------------
        print(f"\n>> Updating Exemplar Memory (Herding)")
        
        new_exemplar_list = []
        
        # Retain m exemplars from each OLD class
        for cid in range(current_total_classes):
            old_class_exemplars = [e for e in exemplar_list if e[1] == cid][:m]
            new_exemplar_list.extend(old_class_exemplars)
        
        # Select m exemplars from each NEW class using herding
        for class_idx in task_classes:
            class_ds = IDXDataset(
                TRAIN_IMAGES, 
                TRAIN_LABELS, 
                class_filter=[class_idx],
                max_samples_per_class=MAX_SAMPLES_PER_CLASS
            )
            
            selected = select_exemplars(
                backbone=backbone,
                dataset=class_ds,
                m=min(m, len(class_ds)),
                device=device
            )
            
            new_exemplar_list.extend(selected)
        
        exemplar_list = new_exemplar_list
        print(f"   Updated memory: {len(exemplar_list)} exemplars ({len(exemplar_list)/new_total_classes:.1f} per class)")
        
        # --------------------------------------------------------------------
        # EVALUATION
        # --------------------------------------------------------------------
        print(f"\n>> Evaluating on all {new_total_classes} classes...")
        
        # Load test data for all seen classes
        seen_classes = base_task_classes + sum(incremental_tasks[:t], [])
        test_ds = IDXDataset(
            TEST_IMAGES,
            TEST_LABELS,
            class_filter=seen_classes,
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
                # Calculate overall accuracy for this task
        total_correct = sum(class_correct.values())
        total_samples = sum(class_total.values())
        overall_accuracy = (total_correct / total_samples) * 100
        
        # Store accuracy for plotting
        task_accuracies[t] = overall_accuracy  # <-- NEW: Add this
        print(f"\\n   Overall Task {t} Accuracy: {overall_accuracy:.2f}%")

        
        # Update metrics
        metrics.update(
            task_id=t,
            class_acc_dict=acc_dict,
            num_params=params,
            duration_seconds=time.time() - task_start_time
        )
        
        # Print cumulative results
        metrics.print_paper_table()
        loss_tracker.plot_task_losses(t, SCENARIO)
        loss_tracker.save_all_graphs(SCENARIO, accuracies_dict=task_accuracies)
        loss_tracker.print_summary()

        # Print accuracy progression
        print(f"\\n{'='*80}")
        print("📊 ACCURACY PROGRESSION:")
        print(f"{'='*80}")
        for task_id in sorted(task_accuracies.keys()):
            task_type = "Base" if task_id == 0 else f"Incremental {task_id}"
            print(f"  Task {task_id} [{task_type}]: {task_accuracies[task_id]:.2f}%")
        print(f"{'='*80}\\n")

        
        current_total_classes = new_total_classes


    # ========================================================================
    # FINAL RESULTS (Paper Table V)
    # ========================================================================

    total_time = time.time() - training_start

    print(f"\n{'='*80}")
    print(f"FINAL RESULTS ({SCENARIO})")
    print(f"{'='*80}")
    print(f"Total Training Time: {total_time/3600:.4f} hours")
    print(f"")
    print(f"Paper Metrics:")
    print(f"  Average Incremental Accuracy (A): {metrics.get_avg_acc()*100:.2f}%")
    print(f"  Average Trainable Parameters:      {int(metrics.get_avg_params()):,}")
    print(f"  Average Training Time:             {metrics.get_avg_time_hours():.6f} h")
    print(f"  Forgetting Measure (F):            {metrics.get_forgetting():.4f}")
    print(f"{'='*80}\n")

    # Save final models
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    backbone_path = f"{save_dir}/final_backbone_{SCENARIO.replace('=', '')}.pth"
    classifier_path = f"{save_dir}/final_classifier_{SCENARIO.replace('=', '')}.pth"

    torch.save(backbone.state_dict(), backbone_path)
    torch.save(classifier.state_dict(), classifier_path)

    print(f"✅ INCREMENTAL TRAINING COMPLETED!")
    print(f"\nFinal models saved:")
    print(f"  {backbone_path}")
    print(f"  {classifier_path}")
    
    print(f"\n{'='*80}")
    print(f"TRAINING CONFIGURATION SUMMARY:")
    print(f"{'='*80}")
    print(f"  Scenario: {SCENARIO}")
    print(f"  Max Samples/Class: {MAX_SAMPLES_PER_CLASS}")
    print(f"  Total Classes: {total_classes}")
    print(f"  Total Tasks: {1 + len(incremental_tasks)}")
    print(f"  Memory Size: {MEMORY_SIZE}")
    print(f"  Architecture: C={C_INIT}, Layers={N_LAYERS}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
