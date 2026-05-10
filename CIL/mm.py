import torch
import torch.nn.functional as F

from model import PDARTSBackbone, ExpandableClassifier, genotype

def detailed_macro_and_micro_trace():
    init_channel = 32
    layer_count = 3
    num_classes = 20

    print("\n" + "="*90)
    print(" DETAILED MACRO & MICRO ARCHITECTURE TRACE (Stem -> Cell DAGs -> MLP) ")
    print("="*90)
    
    # Instantiate the backbone
    backbone = PDARTSBackbone(C=init_channel, layers=layer_count)
    classifier = ExpandableClassifier(in_dim=backbone.feature_dim, n_classes=num_classes)
    
    x = torch.randn(1, 1, 28, 28)
    print(f"[*] Input Data (Grayscale Image)")
    print(f"    Shape: {list(x.shape)}\n")
    
    s0 = s1 = backbone.stem(x)
    print(f"[*] Stem Network (Conv2d -> BatchNorm2d)")
    print(f"    Outputs (s0 & s1) Shape : {list(s1.shape)}\n")

    print(f"[*] Stacking {layer_count} Cells:\n")
    
    for i, cell in enumerate(backbone.cells):
        
        # We manually process the cell inner logic to trace the exact inner DAG processing
        cell_s0 = cell.pre0(s0)
        cell_s1 = cell.pre1(s1)
        
        if cell_s0.size(2) != cell_s1.size(2):
            cell_s0 = F.interpolate(cell_s0, size=cell_s1.shape[2:], mode="nearest")
            
        states = [cell_s0, cell_s1]
        
        # In PDARTSBackbone, reduction logic applies to layers exactly matching the reductions list
        reductions = [layer_count // 3, 2 * layer_count // 3]
        is_reduction = (i in reductions)
        cell_type = "Reduction Cell" if is_reduction else "Normal Cell"
        
        print(f"    " + "="*70)
        print(f"    --- Cell {i + 1} ({cell_type}) ---")
        print(f"    " + "="*70)
        print(f"    [Macro] Input s0 (from Cell {i-1 if i > 0 else 'Stem'}) : {list(s0.shape)}")
        print(f"    [Macro] Input s1 (from Cell {i if i > 0 else 'Stem'})   : {list(s1.shape)}")
        print(f"    [Micro] Pre0 Output (State 0)   : {list(cell_s0.shape)}")
        print(f"    [Micro] Pre1 Output (State 1)   : {list(cell_s1.shape)}\n")
        
        # Determine genotype names based on reduction or normal
        op_names_raw = genotype.reduce if is_reduction else genotype.normal
        concat_indices = genotype.reduce_concat if is_reduction else genotype.normal_concat

        print(f"      [*] Inner DAG Node Execution:")
        node_idx = 2
        for j in range(0, len(cell._ops), 2):
            op1_name = op_names_raw[j][0]
            op2_name = op_names_raw[j+1][0]
            idx1 = cell._indices[j]
            idx2 = cell._indices[j+1]
            
            input1_shape = states[idx1].shape
            input2_shape = states[idx2].shape
            
            h1 = cell._ops[j](states[idx1])
            h2 = cell._ops[j+1](states[idx2])
            
            if h1.size(2) != h2.size(2):
                h1 = F.interpolate(h1, size=h2.shape[2:], mode="nearest")
                
            out_state = h1 + h2
            states.append(out_state)
            
            print(f"         --- Node {node_idx} ---")
            print(f"         h1 = {op1_name:<15} ( input: State {idx1} {list(input1_shape)} ) -> shape: {list(h1.shape)}")
            print(f"         h2 = {op2_name:<15} ( input: State {idx2} {list(input2_shape)} ) -> shape: {list(h2.shape)}")
            print(f"         Node {node_idx} Output (h1 + h2)     : {list(out_state.shape)}\n")
            
            node_idx += 1
            
        concat_states = [states[c_idx] for c_idx in concat_indices]
        s1_new = torch.cat(concat_states, dim=1)
        
        print(f"      [*] Cell End Concatenation:")
        print(f"         Concatenating States   : {list(concat_indices)}")
        print(f"         [Macro] Output Shape (new s1) : {list(s1_new.shape)}\n")
        
        # Update states for the next cell
        s0, s1 = s1, s1_new

    feat = backbone.gap(s1)
    print(f"[*] Global Average Pooling (GAP)")
    print(f"    Input shape             : {list(s1.shape)}")
    print(f"    Output shape            : {list(feat.shape)}\n")
    
    feat_flat = feat.view(feat.size(0), -1)
    print(f"[*] Flattening")
    print(f"    Output shape            : {list(feat_flat.shape)}\n")

    logits = classifier(feat_flat)
    print(f"[*] Final MLP Classifier ({num_classes} Classes)")
    print(f"    Input Feature Vector    : {list(feat_flat.shape)}")
    print(f"    Linear Layer Weights    : {list(classifier.fc.weight.shape)}")
    print(f"    Output shape            : {list(logits.shape)}")
    print(f"    (Yields classification logits for {num_classes} classes)\n")
    print("="*90 + "\n")

if __name__ == "__main__":
    detailed_macro_and_micro_trace()
