'''Build a top-K structural-similarity virtual graph: same nodes, edges rewired to structural role-neighbors.'''
# Phase 2. Reuses I2V's cached structural signals; the virtual edgelist is a drop-in for the walk/GNN encoders.
# A virtual graph links nodes by structural role (hub<->hub, bridge<->bridge), not by original edges.

import argparse
from pathlib import Path

import numpy as np
import networkx as nx
from scipy.special import gammaln

import graph_io
import identity2vec_cached


class VirtualGraph():
    '''Connects each node to its top-K structurally most-similar nodes under a chosen similarity.'''

    def __init__(self, nx_Graph, e=2.7182, seed=42, policy=None):
        self.seed = seed                                     # ties are sampled, not ordered -> the build needs a seed
        self.policy = graph_io.policy_of(policy)             # centrality mode + tie tolerance come from THE graph policy
        self.G = nx_Graph
        self.core = identity2vec_cached.Graph(nx_Graph, e, per_component=self.policy["centrality"] == "per_component")

    # Reference-free I2V identity score: KL(Δ_neigh || Ω_neigh) -> Poisson, dropping the walk shortest-path term.
    def psi_signature(self, node, neigh_map, deg_dist, ev, deg):
        '''Per-node I2V KL->Poisson structural fingerprint (context-free; path factor dropped, no walk reference).'''
        neigh = neigh_map[node]
        if len(neigh) == 0:
            return 0.0
        rt = 0.0
        for w in neigh:
            p, q = deg_dist[w], ev[w]                         # I2V p=Δ (degree-dist), q=Ω (eigenvector centrality)
            if p > 0 and q > 0:
                rt += p * np.log(p / q)                       # KL divergence rate λ
        k = len(neigh)
        drt = max(rt / (deg[node] + ev[node]), 1e-12)         # I2V Fix 4A normalizer; node has neighbors so deg>=1
        return k * np.log(drt) - drt - gammaln(k + 1)         # I2V Fix 8 Poisson log-score Ψ

    # Per-node 1-D structural signature for a similarity variant (add a variant = add one branch here).
    def signatures(self, sim):
        '''node -> structural signature vector; distance in this space defines "structurally similar".'''
        nodes = list(self.G.nodes)
        if sim == 'psi':
            neigh_map, deg_dist = self.core.node_neighbors(), self.core.degree_distribution()
            ev, deg = self.core.eigenvector_centrality(), self.core.degree_node()
            vals = {n: self.psi_signature(n, neigh_map, deg_dist, ev, deg) for n in nodes}
        elif sim == 'degree':
            vals = self.core.degree_node()
        elif sim == 'centrality':
            vals = self.core.eigenvector_centrality()
        else:
            raise ValueError(f"Unknown sim '{sim}'. Use: psi, degree, centrality, original, hybrid.")
        X = np.array([[float(vals[n])] for n in nodes])
        return nodes, np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)   # guard: no NaN/inf signatures

    # Build the virtual graph: same nodes, no self-loops, exactly top-K per node (undirected union), finite weights.
    def build(self, sim, k):
        '''Top-K nearest structural neighbors per node -> weighted undirected virtual graph.'''
        if sim == 'original':                                 # control: unchanged original edges, weight 1.0 -> GNN on the real graph (K unused)
            V = nx.Graph()
            V.add_nodes_from(self.G.nodes)
            V.add_edges_from(self.G.edges, weight=1.0)
            return V
        if sim == 'hybrid':                                   # original ∪ psi top-K: physical neighbors (1.0) + role neighbors (Ψ weight)
            V = self.build('psi', k)
            V.add_edges_from(self.G.edges, weight=1.0)        # overlap edge takes 1.0: original-edge semantics win
            return V
        nodes, X = self.signatures(sim)
        k = min(k, len(nodes) - 1)
        n = len(nodes)
        vals = X[:, 0]
        span = float(vals.max() - vals.min())
        tol = self.policy["sig_tol"]
        keys = np.round(vals / (span * tol)) if span > 0 else np.zeros(n)       # equal key = tied signature = interchangeable role
        order = np.argsort(keys, kind='stable')               # tie classes contiguous, ordered by signature value
        pos = np.empty(n, dtype=int)
        pos[order] = np.arange(n)                             # where each node sits in that order
        starts = np.flatnonzero(np.r_[True, keys[order][1:] != keys[order][:-1]])
        ends = np.r_[starts[1:], n]
        cls = np.searchsorted(starts, pos, side='right') - 1  # node -> its tie class
        rng = np.random.default_rng(self.seed)

        V = nx.Graph()
        V.add_nodes_from(nodes)                               # keep ALL original nodes, even isolated ones -> none missing
        for i, node in enumerate(nodes):
            lo, hi = int(starts[cls[i]]), int(ends[cls[i]])
            picked = []
            if hi - lo - 1 >= k:                              # class alone can fill K: members are tied, so "nearest" is undefined -> sample
                seen = set()
                while len(picked) < k:
                    j = int(order[rng.integers(lo, hi)])
                    if j != i and j not in seen:
                        seen.add(j)
                        picked.append(j)
            else:                                             # class too small: take it all, then widen to the nearest signature values
                picked = [int(j) for j in order[lo:hi] if j != i]
                left, right = lo - 1, hi
                while len(picked) < k:
                    dl = abs(vals[order[left]] - vals[i]) if left >= 0 else np.inf
                    dr = abs(vals[order[right]] - vals[i]) if right < n else np.inf
                    if dl <= dr:
                        picked.append(int(order[left])); left -= 1
                    else:
                        picked.append(int(order[right])); right += 1
            for j in picked:                                  # exactly K per node, no self-loops
                V.add_edge(node, nodes[j], weight=1.0 / (1.0 + abs(float(vals[i] - vals[j]))))   # similarity in (0,1], always finite
        assert V.number_of_nodes() == self.G.number_of_nodes(), "node set changed"
        assert np.all(np.isfinite([w for _, _, w in V.edges(data='weight')])), "non-finite edge weight"
        return V


# Reads the input edgelist file into a networkx graph (shared loader: one graph definition for every stage).
def build_graph():
    '''Read input network.'''
    return graph_io.load_graph(args.input)


# Runs the whole pipeline: build graph -> build virtual graph -> save weighted edgelist.
def main(args):
    np.random.seed(args.seed)                                # build is deterministic given the seed (tied signatures are sampled)
    if args.output is None:
        ds = Path(args.input).stem
        args.output = f"output/notebook2_create_vir_graph/virtual_graphs/{ds}/k{args.k}/{args.sim}/virtual_graph.edgelist"
    G = build_graph()
    V = VirtualGraph(G, args.e, args.seed).build(args.sim, args.k)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)   # auto-create per-dataset subfolder
    nx.write_weighted_edgelist(V, args.output)               # "u v weight" -> readable by NetworkX & PyG
    isolated = sum(1 for _, d in V.degree if d == 0)
    print(f"virtual graph | sim={args.sim} k={args.k} | nodes={V.number_of_nodes()} edges={V.number_of_edges()} "
          f"avg_deg={2 * V.number_of_edges() / V.number_of_nodes():.2f} isolated={isolated} -> {args.output}")


# Defines command-line options (mirrors train.py): input graph, output edgelist, similarity variant, K, seed.
def parse_args():
    '''Parses arguments.'''
    parser = argparse.ArgumentParser(description="Build a top-K structural-similarity virtual graph.")
    parser.add_argument('--input', nargs='?', default='input/cora.edgelist', help='Input graph path')
    parser.add_argument('--output', nargs='?', default=None,
                        help='Output virtual edgelist (default: output/notebook2_create_vir_graph/virtual_graphs/<ds>/k<K>/<sim>/virtual_graph.edgelist)')
    parser.add_argument('--sim', default='psi', choices=['psi', 'degree', 'centrality', 'original', 'hybrid'],
                        help='Structural similarity: psi=I2V KL/Poisson, degree-only, centrality-only, '
                             'original=exact copy of input graph (control, K unused), hybrid=original + psi top-K union. Default psi.')
    parser.add_argument('--k', type=int, default=10, help='Top-K structural neighbors per node. Default 10.')
    parser.add_argument('--e', type=float, default=2.7182, help='Euler constant (I2V).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility. Default 42.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
