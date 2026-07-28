'''Convert the heterophilous benchmarks (Platonov et al. 2023) to ViRGo edgelist/label/.nodes files: roman_empire, tolokers, questions - structural-only.'''
# Added 2026-07-27 to test the AUGMENT side of the rule. The six-dataset panel held no low-homophily graph, so the
# only "augment" verdict was a single cell (ogbl-ddi LP) and the decision boundary was fitted from one point.
# These two sit on the two axes the rule predicts from - adjusted homophily (node classification) and average degree
# (link prediction) - and the prediction is recorded in docs/paper_log.md BEFORE the run, so the outcome is held out.
# Purely structural: node_features (roman-empire 300-dim fastText, tolokers 10-dim worker profile) are IGNORED by design.
# Platonov's own 10 train/val/test masks are also ignored: these run the ViRGo CORE protocol (stratified 70% node
# classification, 70:30 link prediction) so their cells stay comparable with the four core datasets.
# Adjusted homophily is the metric this panel already reports, and it comes from this same paper.

import argparse
from pathlib import Path

from make_ogb import _write_edges, _write_nodes   # same writers as the OGB conversion; the import also installs the torch.load allowlist

RAW = "output/hetero_raw"                         # PyG download cache: derived, never hand-edited, kept out of input/

# ViRGo name -> PyG name. Only the three the panel needs; amazon_ratings/minesweeper ship in the same class.
HETERO = {"roman_empire": "Roman-empire", "tolokers": "Tolokers", "questions": "Questions"}


# Builds input/<name>.edgelist + .nodes + labels/<name>.labels from the PyG copy; drops data.x (structural-only).
def make_hetero(name):
    '''One heterophilous benchmark -> edgelist + .nodes + labels. node_features and the official masks ignored by design.'''
    from torch_geometric.datasets import HeterophilousGraphDataset
    g = HeterophilousGraphDataset(root=RAW, name=HETERO[name])[0]
    n = g.num_nodes
    ne = _write_edges(f"input/{name}.edgelist", g.edge_index.numpy())   # PyG already symmetrized it (to_undirected in process)
    _write_nodes(f"input/{name}.edgelist", n)
    y = g.y.numpy().reshape(-1)
    Path("labels").mkdir(exist_ok=True)
    with open(f"labels/{name}.labels", "w") as f:
        for i in range(n):
            f.write(f"{i} {int(y[i])}\n")
    print(f"{name}: nodes={n} edges={ne} classes={len(set(y))} avg_degree={2 * ne / n:.2f} | "
          f"data.x ({g.x.shape[1]}-dim) IGNORED (structural-only) | official masks IGNORED (ViRGo core protocol)")
    print(f"WROTE input/{name}.edgelist + .nodes + labels/{name}.labels")


# Defines command-line options: which heterophilous dataset to build (default both).
def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Convert the heterophilous benchmarks to ViRGo edgelist/label/.nodes files (structural-only).")
    p.add_argument('--dataset', default='both', choices=['both'] + list(HETERO), help='Which dataset to build. Default both.')
    return p.parse_args()


# Builds the requested dataset(s) from the command line.
def main(args):
    for name in HETERO:
        if args.dataset in ('both', name):
            make_hetero(name)


if __name__ == "__main__":
    main(parse_args())
