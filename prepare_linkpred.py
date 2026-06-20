'''Split graph edges 70/30 for link prediction, keep the train graph connected, sample equal non-edges. No leakage.'''

import argparse
import os
import random
import networkx as nx


# Defines command-line options: which graph, dataset name, output dir, test fraction, seed.
def parse_args():
    parser = argparse.ArgumentParser(description="Prepare leakage-free link-prediction splits.")
    parser.add_argument('--input', default='input/cora.edgelist', help='Input graph edgelist')
    parser.add_argument('--name', default='cora', help='Dataset name (prefix for output files)')
    parser.add_argument('--outdir', default='splits', help='Directory to write splits. Default splits.')
    parser.add_argument('--test-frac', type=float, default=0.3, help='Fraction of edges held out for test. Default 0.3.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed. Default 42.')
    return parser.parse_args()


# Reads an edgelist into an undirected graph (largest connected component only, so connectivity holds).
def build_graph(path):
    '''Read input network, drop self-loops, keep its largest connected component.'''
    G = nx.read_edgelist(path, nodetype=int, create_using=nx.Graph())
    G.remove_edges_from(nx.selfloop_edges(G))   # a self-loop is not a valid link-pred pair and collapses to a 1-node "edge"
    largest = max(nx.connected_components(G), key=len)
    return G.subgraph(largest).copy()


# Holds out test_frac of edges as positives while forcing a spanning tree to stay in train (keeps train connected).
def split_edges(G, test_frac, seed):
    '''Spanning-tree edges stay in train; test positives are sampled from the rest.'''
    rng = random.Random(seed)
    tree = {frozenset(e) for e in nx.minimum_spanning_tree(G).edges()}
    removable = [frozenset(e) for e in G.edges() if frozenset(e) not in tree]
    rng.shuffle(removable)
    n_test = min(int(round(test_frac * G.number_of_edges())), len(removable))
    test_pos = removable[:n_test]
    train_pos = list(tree) + removable[n_test:]
    return [tuple(sorted(e)) for e in train_pos], [tuple(sorted(e)) for e in test_pos]   # clean sorted 2-tuples


# Samples k node pairs that are not real edges and not already used (the fake/negative pairs).
def sample_non_edges(G, k, seed, exclude):
    '''Draw k unique non-edges disjoint from the graph and from exclude.'''
    rng = random.Random(seed)
    nodes = list(G.nodes())
    neg = set()
    while len(neg) < k:
        u, v = rng.choice(nodes), rng.choice(nodes)
        e = frozenset((u, v))
        if u != v and not G.has_edge(u, v) and e not in exclude and e not in neg:
            neg.add(e)
    return [tuple(sorted(e)) for e in neg]   # clean sorted 2-tuples


# Writes node-pair lines "u v" to a file.
def write_pairs(path, pairs):
    '''Save one "u v" pair per line.'''
    with open(path, 'w') as f:
        for u, v in pairs:
            f.write(f"{u} {v}\n")


# Builds and writes all split files for one dataset; returns the counts. Importable by the runner.
def prepare(input_path, name, outdir, test_frac=0.3, seed=42):
    '''Split edges -> sample negatives -> write train graph + pos/neg pair files. Returns counts.'''
    os.makedirs(outdir, exist_ok=True)
    G = build_graph(input_path)
    train_pos, test_pos = split_edges(G, test_frac, seed)
    pos = {frozenset(e) for e in train_pos} | {frozenset(e) for e in test_pos}
    neg = sample_non_edges(G, len(train_pos) + len(test_pos), seed, pos)
    train_neg, test_neg = neg[:len(train_pos)], neg[len(train_pos):]

    p = os.path.join(str(outdir), name)
    write_pairs(f"{p}_train.edgelist", train_pos)   # retrain embeddings on THIS graph only
    write_pairs(f"{p}_train_neg.txt", train_neg)
    write_pairs(f"{p}_test_pos.txt", test_pos)
    write_pairs(f"{p}_test_neg.txt", test_neg)
    return {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
            "train_pos": len(train_pos), "train_neg": len(train_neg),
            "test_pos": len(test_pos), "test_neg": len(test_neg)}


# Runs the split from the command line and prints a summary.
def main(args):
    c = prepare(args.input, args.name, args.outdir, args.test_frac, args.seed)
    print(f"{args.name}: nodes={c['nodes']} edges={c['edges']} seed={args.seed}")
    print(f"  train_pos={c['train_pos']} train_neg={c['train_neg']} test_pos={c['test_pos']} test_neg={c['test_neg']}")
    print(f"  wrote splits to {args.outdir}/  -> next: retrain I2V on {args.outdir}/{args.name}_train.edgelist")


if __name__ == "__main__":
    main(parse_args())
