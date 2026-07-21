'''Unsupervised GraphSAGE over a virtual graph: same Skipgram-analog objective as I2V, lookup table replaced by message passing.'''
# Phase 3 (ViRGo-SAGE). Features come from the ORIGINAL graph's cached structural signals; messages pass over the VIRTUAL graph.
# Positives = direct virtual edges by default (ablation-A winner); "walk" kept as the Phase-2-bridge-comparable option.
# Ablation D: the `feats` knob selects which structural features enter the GNN ("random" = the no-structure control).
# Ablation D6: layers=0 -> no convolutions, embeddings ARE the z-normed features (features without message passing; nothing to train).

import argparse
from pathlib import Path

import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GraphConv

import graph_io
from virtual_graph import VirtualGraph

# Walk-corpus settings mirror I2V_PARAMS (scripts/benchmark_config.py) — the Phase-2 bridge corpus.
WALKS = {"num_walks": 10, "walk_length": 40, "window": 10}


class SageEncoder():
    '''2-layer GraphSAGE trained with Skipgram-style positives/negatives; neighbor aggregation mean/weighted/sum/max (ablation B).'''

    def __init__(self, G, V, seed, hidden=64, dimensions=64, layers=2, agg="mean", feats="all"):
        # layers=0 (D6) -> no convs: forward() returns the features unchanged, V is unused, train() is invalid.
        self.G, self.V, self.seed, self.feats = G, V, seed, feats
        torch.manual_seed(seed); np.random.seed(seed)         # CPU-only + seeded -> bit-reproducible
        self.nodes = list(G.nodes)
        self.index = {n: i for i, n in enumerate(self.nodes)}
        e = [(self.index[u], self.index[v]) for u, v in V.edges]
        self.edge_index = torch.tensor([[a for a, b in e] + [b for a, b in e],
                                        [b for a, b in e] + [a for a, b in e]], dtype=torch.long)   # undirected -> both directions
        if agg == "weighted":                                  # ablation B "weighted": Ψ edge weights, normalized per target -> weighted mean
            w = torch.tensor([float(d.get("weight", 1.0)) for _, _, d in V.edges(data=True)] * 2, dtype=torch.float)
            denom = torch.zeros(len(self.nodes)).scatter_add_(0, self.edge_index[1], w)
            self.edge_weight = w / denom[self.edge_index[1]].clamp(min=1e-12)
        else:
            self.edge_weight = None
        self.X = self.features()
        dims = [self.X.shape[1]] + [hidden] * (layers - 1) + [dimensions]
        Conv = GraphConv if agg == "weighted" else SAGEConv    # GraphConv = same root+neighbor form as SAGE but edge-weight-aware
        self.convs = torch.nn.ModuleList(Conv(dims[i], dims[i + 1], aggr='add' if agg == "weighted" else agg) for i in range(layers))

    def features(self):
        '''Structural node features from the ORIGINAL graph, z-normalized; ablation D `feats` selects the subset.'''
        if self.feats == "random":                             # D4 control: node identity only, zero structural signal
            X = np.random.default_rng(self.seed).normal(size=(len(self.nodes), 4))
        elif self.feats == "const":                            # D5 floor: identical rows -> z-norm gives all-zeros
            X = np.ones((len(self.nodes), 1))
        else:
            vg = VirtualGraph(self.G)
            psi_nodes, psi = vg.signatures('psi')              # reference-free I2V KL->Poisson score per node
            psi = dict(zip(psi_nodes, psi[:, 0]))              # keyed by node, like deg/ev/clus: signatures() owns its own node order
            deg, ev = vg.core.degree_node(), vg.core.eigenvector_centrality()
            clus = nx.clustering(self.G)
            cols = {"degree": [0], "deg_cent": [0, 1], "psi": [2], "all": [0, 1, 2, 3]}[self.feats]
            X = np.array([[deg[n], ev[n], float(psi[n]), clus[n]] for n in self.nodes])[:, cols]
        return torch.tensor((X - X.mean(0)) / (X.std(0) + 1e-9), dtype=torch.float)

    def corpus(self, max_pairs=2_000_000, positives="walk"):
        '''Skipgram positives — ablation A: "walk" = A1 window co-occurrence on virtual-graph walks (= Phase-2 bridge corpus); "edge" = A2 direct virtual edges, no walks.'''
        if positives == "edge":
            pairs = [(self.index[u], self.index[v]) for u, v in self.V.edges]
            pairs += [(b, a) for a, b in pairs]            # both directions: each endpoint also learns as the center node
        elif positives == "walk":
            from node2vec import Node2Vec
            U = nx.Graph(); U.add_nodes_from(self.V.nodes); U.add_edges_from(self.V.edges)   # unweighted copy = the bridge's walks
            walks = Node2Vec(U, dimensions=1, walk_length=WALKS["walk_length"], num_walks=WALKS["num_walks"],
                             p=1, q=1, workers=1, seed=self.seed, quiet=True).walks
            w = WALKS["window"]
            pairs = [(self.index[int(a)], self.index[int(b)]) for walk in walks
                     for i, a in enumerate(walk) for b in walk[i + 1:i + 1 + w] if a != b]
        else:
            raise ValueError(f"Unknown positives '{positives}'. Use: walk (A1), edge (A2).")
        assert len(pairs) > 0, "no positive pairs — is the virtual graph edgeless?"
        pairs = torch.tensor(pairs, dtype=torch.long)
        if len(pairs) > max_pairs:                             # deterministic cap: runtime/memory guard on large graphs
            keep = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(self.seed))[:max_pairs]
            pairs = pairs[keep]
        return pairs

    def forward(self):
        '''Full-graph message passing over the virtual edges -> one embedding per node.'''
        z = self.X
        for i, conv in enumerate(self.convs):
            z = conv(z, self.edge_index) if self.edge_weight is None else conv(z, self.edge_index, self.edge_weight)
            if i < len(self.convs) - 1:
                z = F.relu(z)
        return z

    def train(self, epochs, lr=0.01, negatives=5, pairs_per_epoch=100_000, max_pairs=2_000_000, positives="walk"):
        '''Skipgram-analog objective: pull positive pairs together, push deg^0.75 negatives apart.'''
        assert len(self.convs) > 0, "layers=0 (D6) has no parameters to train — save() the features directly instead"
        pairs = self.corpus(max_pairs, positives)
        deg = torch.tensor([d for _, d in self.V.degree(self.nodes)], dtype=torch.float) ** 0.75
        neg_dist = deg / deg.sum() if deg.sum() > 0 else torch.full((len(self.nodes),), 1.0 / len(self.nodes))
        opt = torch.optim.Adam((p for c in self.convs for p in c.parameters()), lr=lr)
        g = torch.Generator().manual_seed(self.seed)
        losses = []
        for epoch in range(epochs):
            z = self.forward()
            idx = torch.randint(len(pairs), (min(pairs_per_epoch, len(pairs)),), generator=g)
            u, v = pairs[idx, 0], pairs[idx, 1]
            n = torch.multinomial(neg_dist, len(u) * negatives, replacement=True, generator=g).view(len(u), negatives)
            pos = F.logsigmoid((z[u] * z[v]).sum(-1))                        # pull positives together
            neg = F.logsigmoid(-(z[u].unsqueeze(1) * z[n]).sum(-1)).sum(-1)  # push Q negatives apart
            loss = -(pos + neg).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"  epoch {epoch:3d} | loss {loss:.4f}")
        return losses

    def save(self, out):
        '''Write embeddings in word2vec text format so the existing eval scripts read them unchanged.'''
        with torch.no_grad():
            z = self.forward().numpy()
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            f.write(f"{len(self.nodes)} {z.shape[1]}\n")
            for node, row in zip(self.nodes, z):
                f.write(f"{node} " + " ".join(f"{x:.6f}" for x in row) + "\n")
        return out


# Reads an edgelist file into a networkx graph (shared loader: one graph definition for every stage).
def build_graph(path):
    '''Read input network.'''
    return graph_io.load_graph(path)


# Runs the whole pipeline: load original + virtual graph -> train ViRGo-SAGE -> save .emb.
def main(args):
    ds = Path(args.input).stem
    tag = ("features_only" if args.layers == 0 else                          # D6: no message passing -> not a graphsage row
           ("graphsage_walk" if args.positives == "walk" else "graphsage_edge") + ("" if args.agg == "mean" else f"_{args.agg}")
           + ("" if args.features == "all" else f"_feat_{args.features}"))   # each ablation writes its own files
    virtual = args.virtual or f"output/notebook2_create_vir_graph/virtual_graphs/{ds}/k{args.k}/{args.sim}/virtual_graph.edgelist"
    output = args.output or f"output/notebook3_gnn_encoder/node_classification/{ds}/k{args.k}/{args.sim}/{tag}_s{args.seed}.emb"
    G = build_graph(args.input)
    V = nx.read_weighted_edgelist(virtual, nodetype=int)
    V.add_nodes_from(G.nodes)                                  # edgelists omit isolated nodes -> restore the full node set
    enc = SageEncoder(G, V, args.seed, hidden=args.hidden, dimensions=args.dimensions, layers=args.layers,
                      agg=args.agg, feats=args.features)
    if args.layers:                                            # D6 (layers=0): the features are the embedding, nothing to train
        enc.train(args.epochs, lr=args.lr, negatives=args.negatives, positives=args.positives)
    print(f"virgo-sage | sim={args.sim} k={args.k} positives={args.positives} agg={args.agg} feats={args.features} "
          f"seed={args.seed} -> {enc.save(output)}")


# Defines command-line options (mirrors virtual_graph.py); defaults mirror scripts/benchmark_config.GNN_PARAMS.
def parse_args():
    '''Parses arguments.'''
    parser = argparse.ArgumentParser(description="Train unsupervised GraphSAGE over a saved virtual graph.")
    parser.add_argument('--input', nargs='?', default='input/cora.edgelist', help='Original graph (features + node set)')
    parser.add_argument('--virtual', nargs='?', default=None,
                        help='Virtual edgelist (default: output/notebook2_create_vir_graph/virtual_graphs/<ds>/k<K>/<sim>/virtual_graph.edgelist)')
    parser.add_argument('--output', nargs='?', default=None,
                        help='Output .emb (default: output/notebook3_gnn_encoder/node_classification/<ds>/k<K>/<sim>/<encoder>_s<seed>.emb; '
                             'trains on the full graph -> for link-prediction embeddings pass a train graph via --input and set --output)')
    parser.add_argument('--sim', default='psi', choices=['psi', 'degree', 'centrality', 'original', 'hybrid'],
                        help='Virtual-graph variant (original=copy of input graph, hybrid=original + psi top-K union). Default psi.')
    parser.add_argument('--k', type=int, default=10, help='Top-K of the virtual graph. Default 10.')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs. Default 50.')
    parser.add_argument('--lr', type=float, default=0.01, help='Adam learning rate. Default 0.01.')
    parser.add_argument('--hidden', type=int, default=64, help='Hidden width. Default 64.')
    parser.add_argument('--dimensions', type=int, default=64, help='Embedding size (matches I2V). Default 64.')
    parser.add_argument('--layers', type=int, default=2,
                        help='GraphSAGE layers (ablation C depth 1-3; 0=D6 features-only, no message passing). Default 2.')
    parser.add_argument('--negatives', type=int, default=5, help='Negatives per positive (matches Word2Vec negative=5). Default 5.')
    parser.add_argument('--positives', default='edge', choices=['walk', 'edge'],
                        help='Ablation A: edge=A2 direct virtual edges (winner, default), walk=A1 walk co-occurrence (bridge-comparable).')
    parser.add_argument('--agg', default='mean', choices=['mean', 'weighted', 'sum', 'max'],
                        help='Ablation B neighbor aggregation: mean | weighted (Ψ-weighted mean) | sum | max. Default mean.')
    parser.add_argument('--features', default='all', choices=['all', 'degree', 'deg_cent', 'psi', 'random', 'const'],
                        help='Ablation D input features: all=D0 [deg,Ω,ψ,clustering] | degree=D1 | deg_cent=D2 | psi=D3 | '
                             'random=D4 control (no structural signal) | const=D5 floor. Default all.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility. Default 42.')
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
