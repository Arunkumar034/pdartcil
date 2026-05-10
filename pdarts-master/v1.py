"""
visualize.py  —  reads a P-DARTS log.txt and draws cell graphs for
                 Stage 1 (5 ops/edge), Stage 2 (3 ops/edge),
                 Stage 3 (1 op/edge) and the Final Genotype.

Usage:
    python visualize.py log.txt

Output PDFs (saved next to log.txt or in current directory):
    stage1_normal.pdf   stage1_reduce.pdf
    stage2_normal.pdf   stage2_reduce.pdf
    stage3_normal.pdf   stage3_reduce.pdf
    final_normal.pdf    final_reduce.pdf
"""

import sys
import os
import re
import ast
from graphviz import Digraph


# ---------------------------------------------------------------
# Colour palette per operation type
# ---------------------------------------------------------------
OP_COLORS = {
    'none':          '#d3d3d3',   # light grey
    'skip_connect':  '#aec6cf',   # pastel blue
    'max_pool_3x3':  '#ffb347',   # pastel orange
    'avg_pool_3x3':  '#ffb347',
    'sep_conv_3x3':  '#77dd77',   # pastel green
    'sep_conv_5x5':  '#4caf50',   # darker green
    'dil_conv_3x3':  '#c39bd3',   # pastel purple
    'dil_conv_5x5':  '#9b59b6',   # darker purple
}

def op_color(op):
    return OP_COLORS.get(op, '#ffffff')


# ---------------------------------------------------------------
# Parse one block of 14 edge-op lists from the log
# Each call returns:  list of 14 lists-of-strings
# ---------------------------------------------------------------
def parse_ops_block(lines, start_idx):
    """
    From start_idx, read 14 consecutive lines that look like:
        04/05 ... ['op1', 'op2', ...]
    Returns (ops_per_edge_list, next_idx)
    """
    ops = []
    idx = start_idx
    while len(ops) < 14 and idx < len(lines):
        m = re.search(r"\[('[\w_]+'(?:,\s*'[\w_]+')*)\]", lines[idx])
        if m:
            op_list = ast.literal_eval('[' + m.group(1) + ']')
            ops.append(op_list)
        idx += 1
    return ops, idx


# ---------------------------------------------------------------
# Build a multi-op graph for Stage 1 / Stage 2
# Each edge can carry MULTIPLE operations (shown as parallel edges
# or stacked labels).
# ---------------------------------------------------------------
def plot_multi_op(ops_14, cell_name, stage_label, filename):
    """
    ops_14  : list of 14 elements, each a list of op-strings
    cell_name : 'Normal Cell' or 'Reduction Cell'
    stage_label : e.g. 'Stage 1  (5 ops / edge)'
    filename  : output pdf path (without .pdf extension)
    """
    g = Digraph(
        format='pdf',
        graph_attr=dict(
            label=f'{stage_label}\n{cell_name}',
            labelloc='t', fontsize='38', fontname='times',
            rankdir='LR'
        ),
        edge_attr=dict(fontsize='38', fontname='times'),
        node_attr=dict(
            style='filled', shape='rect', align='center',
            fontsize='18', height='0.6', width='0.8',
            penwidth='2', fontname='times'
        )
    )

    # Input nodes
    g.node('c_{k-2}', fillcolor='darkseagreen2')
    g.node('c_{k-1}', fillcolor='darkseagreen2')

    # 4 intermediate nodes
    steps = 4
    for i in range(steps):
        g.node(str(i), fillcolor='lightblue')

    # Output node
    g.node('c_{k}', fillcolor='palegoldenrod')

    # Draw edges — edge index layout:
    # node0 → edges [0,1]          (2 inputs from c_{k-2}, c_{k-1})
    # node1 → edges [2,3,4]        (3 inputs)
    # node2 → edges [5,6,7,8]      (4 inputs)
    # node3 → edges [9,10,11,12,13](5 inputs)
    n = 2
    start = 0
    for node_i in range(steps):
        end = start + n
        for edge_idx in range(start, end):
            src_j = edge_idx - start   # 0 → c_{k-2}, 1 → c_{k-1}, ≥2 → prev node
            if src_j == 0:
                u = 'c_{k-2}'
            elif src_j == 1:
                u = 'c_{k-1}'
            else:
                u = str(src_j - 2)
            v = str(node_i)
            ops = ops_14[edge_idx]
            # Draw one edge per op, coloured by op type
            for op in ops:
                color = op_color(op)
                g.edge(u, v, label=op,
                       color=color, fontcolor='#333333',
                       style='solid', penwidth='1.5')
        start = end
        n += 1

    # Concat edges
    for i in range(steps):
        g.edge(str(i), 'c_{k}', color='gray', style='dashed')

    g.render(filename, view=False, cleanup=True)
    print(f'  Saved: {filename}.pdf')


# ---------------------------------------------------------------
# Build a single-op graph for Stage 3 / Final Genotype
# Each edge carries exactly ONE operation.
# ---------------------------------------------------------------
def plot_single_op(ops_14, cell_name, stage_label, filename):
    """
    ops_14  : list of 14 elements, each a list with ONE op-string
              OR a list of (op, src_node) tuples (for final genotype)
    """
    g = Digraph(
        format='pdf',
        graph_attr=dict(
            label=f'{stage_label}\n{cell_name}',
            labelloc='t', fontsize='22', fontname='times',
            rankdir='LR'
        ),
        edge_attr=dict(fontsize='15', fontname='times'),
        node_attr=dict(
            style='filled', shape='rect', align='center',
            fontsize='18', height='0.6', width='0.8',
            penwidth='2', fontname='times'
        )
    )

    g.node('c_{k-2}', fillcolor='darkseagreen2')
    g.node('c_{k-1}', fillcolor='darkseagreen2')

    steps = 4
    for i in range(steps):
        g.node(str(i), fillcolor='lightblue')
    g.node('c_{k}', fillcolor='palegoldenrod')

    n = 2
    start = 0
    for node_i in range(steps):
        end = start + n
        for edge_idx in range(start, end):
            src_j = edge_idx - start
            if src_j == 0:
                u = 'c_{k-2}'
            elif src_j == 1:
                u = 'c_{k-1}'
            else:
                u = str(src_j - 2)
            v = str(node_i)
            op = ops_14[edge_idx][0] if ops_14[edge_idx] else 'none'
            color = op_color(op)
            g.edge(u, v, label=op,
                   color=color, fontcolor='#333333',
                   penwidth='2.5')
        start = end
        n += 1

    for i in range(steps):
        g.edge(str(i), 'c_{k}', color='gray', style='dashed')

    g.render(filename, view=False, cleanup=True)
    print(f'  Saved: {filename}.pdf')


# ---------------------------------------------------------------
# Plot Final Genotype (list of (op, src) tuples)
# ---------------------------------------------------------------
def plot_final_genotype(gene, cell_name, filename):
    """
    gene : list of (op_str, src_node_int) pairs — standard DARTS genotype
    """
    g = Digraph(
        format='pdf',
        graph_attr=dict(
            label=f'Final Genotype\n{cell_name}',
            labelloc='t', fontsize='22', fontname='times',
            rankdir='LR'
        ),
        edge_attr=dict(fontsize='45', fontname='times'),
        node_attr=dict(
            style='filled', shape='rect', align='center',
            fontsize='18', height='0.6', width='0.8',
            penwidth='2', fontname='times'
        )
    )

    g.node('c_{k-2}', fillcolor='darkseagreen2')
    g.node('c_{k-1}', fillcolor='darkseagreen2')

    assert len(gene) % 2 == 0
    steps = len(gene) // 2
    for i in range(steps):
        g.node(str(i), fillcolor='lightblue')
    g.node('c_{k}', fillcolor='palegoldenrod')

    for i in range(steps):
        for k in [2 * i, 2 * i + 1]:
            op, j = gene[k]
            if j == 0:
                u = 'c_{k-2}'
            elif j == 1:
                u = 'c_{k-1}'
            else:
                u = str(j - 2)
            v = str(i)
            color = op_color(op)
            g.edge(u, v, label=op,
                   color=color, fontcolor='#333333',
                   penwidth='2.5')

    for i in range(steps):
        g.edge(str(i), 'c_{k}', color='gray', style='dashed')

    g.render(filename, view=False, cleanup=True)
    print(f'  Saved: {filename}.pdf')


# ---------------------------------------------------------------
# Parse the entire log file
# Returns dict with keys: stage1, stage2, stage3, final
# Each stage → {'normal': ops_14, 'reduce': ops_14}
# final       → {'normal': gene_list, 'reduce': gene_list}
# ---------------------------------------------------------------
def parse_log(log_path):
    with open(log_path, 'r') as f:
        lines = f.readlines()

    # Find all occurrences of 'switches_normal' and 'switches_reduce'
    normal_indices = [i for i, l in enumerate(lines) if 'switches_normal =' in l]
    reduce_indices = [i for i, l in enumerate(lines) if 'switches_reduce =' in l]

    print(f'Found {len(normal_indices)} switches_normal blocks (expect 3 for 3 stages)')
    print(f'Found {len(reduce_indices)} switches_reduce blocks (expect 3 for 3 stages)')

    stages = {}
    stage_names = ['stage1', 'stage2', 'stage3']
    ops_per_stage = [5, 3, 1]

    for idx, (ni, ri, sname, nops) in enumerate(
            zip(normal_indices, reduce_indices, stage_names, ops_per_stage)):
        # ops block starts on the line AFTER the switches_normal line
        normal_ops, _ = parse_ops_block(lines, ni + 1)
        reduce_ops, _ = parse_ops_block(lines, ri + 1)
        stages[sname] = {
            'normal': normal_ops,
            'reduce': reduce_ops,
            'ops_per_edge': nops
        }
        print(f'  {sname}: normal edges={len(normal_ops)}, '
              f'reduce edges={len(reduce_ops)}, ops/edge≈{nops}')

    # Parse final genotype line
    final_line = None
    for line in lines:
        if 'Genotype(normal=' in line:
            final_line = line
            break

    final = None
    if final_line:
        m = re.search(r'Genotype\((.+)\)', final_line)
        if m:
            content = m.group(0)
            # Replace range(...) so eval works
            content = re.sub(r'range\((\d+),\s*(\d+)\)',
                             lambda x: str(list(range(int(x.group(1)),
                                                      int(x.group(2))))),
                             content)
            # Build a simple namespace to eval into
            from collections import namedtuple
            Genotype = namedtuple('Genotype',
                                  ['normal', 'normal_concat',
                                   'reduce', 'reduce_concat'])
            try:
                geno = eval(content, {'Genotype': Genotype})
                final = {'normal': geno.normal, 'reduce': geno.reduce}
                print(f'  Final genotype parsed OK')
            except Exception as e:
                print(f'  Warning: could not parse final genotype: {e}')

    return stages, final


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print('Usage: python visualize.py <path_to_log.txt>')
        sys.exit(1)

    log_path = sys.argv[1]
    if not os.path.exists(log_path):
        print(f'Error: file not found: {log_path}')
        sys.exit(1)

    out_dir = os.path.dirname(os.path.abspath(log_path))
    print(f'\nParsing: {log_path}')
    print(f'Output:  {out_dir}\n')

    stages, final = parse_log(log_path)

    # ---- Stage 1 : 5 ops per edge ----
    if 'stage1' in stages:
        print('\n[Stage 1 — 5 ops per edge]')
        d = stages['stage1']
        plot_multi_op(d['normal'], 'Normal Cell',
                      'Stage 1  (5 ops / edge)',
                      os.path.join(out_dir, 'stage1_normal'))
        plot_multi_op(d['reduce'], 'Reduction Cell',
                      'Stage 1  (5 ops / edge)',
                      os.path.join(out_dir, 'stage1_reduce'))

    # ---- Stage 2 : 3 ops per edge ----
    if 'stage2' in stages:
        print('\n[Stage 2 — 3 ops per edge]')
        d = stages['stage2']
        plot_multi_op(d['normal'], 'Normal Cell',
                      'Stage 2  (3 ops / edge)',
                      os.path.join(out_dir, 'stage2_normal'))
        plot_multi_op(d['reduce'], 'Reduction Cell',
                      'Stage 2  (3 ops / edge)',
                      os.path.join(out_dir, 'stage2_reduce'))

    # ---- Stage 3 : 1 op per edge ----
    if 'stage3' in stages:
        print('\n[Stage 3 — 1 op per edge]')
        d = stages['stage3']
        # Wrap each single op in a list for plot_single_op
        normal_wrapped = [[op] for op in
                          [e[0] for e in d['normal']]]
        reduce_wrapped = [[op] for op in
                          [e[0] for e in d['reduce']]]
        plot_single_op(normal_wrapped, 'Normal Cell',
                       'Stage 3  (1 op / edge)',
                       os.path.join(out_dir, 'stage3_normal'))
        plot_single_op(reduce_wrapped, 'Reduction Cell',
                       'Stage 3  (1 op / edge)',
                       os.path.join(out_dir, 'stage3_reduce'))

    # ---- Final Genotype ----
    if final:
        print('\n[Final Genotype]')
        plot_final_genotype(final['normal'], 'Normal Cell',
                            os.path.join(out_dir, 'final_normal'))
        plot_final_genotype(final['reduce'], 'Reduction Cell',
                            os.path.join(out_dir, 'final_reduce'))

    print('\nDone! All PDFs saved.')
    print('\nColour legend:')
    for op, col in OP_COLORS.items():
        print(f'  {col}  →  {op}')


if __name__ == '__main__':
    main()