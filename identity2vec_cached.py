"""Identity2Vec walker with cached structural signals — identical output to the baseline, much faster.

Degree, eigenvector centrality, neighbor lists and shortest-path lengths are invariant for a static
graph, but the baseline recomputes them inside the walk loop. Caching them changes no computed value
(so embeddings are byte-identical) and removes the dominant cost. This is the I2V 'cache fix'."""

import networkx as nx
import identity2vec


class Graph(identity2vec.Graph):
    """Drop-in Graph that caches the structural signals the baseline recomputes per step."""

    def __init__(self, nx_Graph, e):
        super().__init__(nx_Graph, e)
        self._deg = None
        self._ev = None
        self._neigh = None
        self._spl = {}

    # Degree per node, computed once.
    def degree_node(self):
        if self._deg is None:
            self._deg = {n: d for n, d in self.G.degree}
        return self._deg

    # Eigenvector centrality, computed once (deterministic power iteration).
    def eigenvector_centrality(self):
        if self._ev is None:
            self._ev = nx.eigenvector_centrality(self.G, max_iter=1000)
        return self._ev

    # Neighbor lists, looked up once; returns fresh copies because callers mutate them (skip_visited).
    def node_neighbors(self):
        if self._neigh is None:
            self._neigh = {n: list(self.G.neighbors(n)) for n in self.G.nodes}
        return {n: v[:] for n, v in self._neigh.items()}

    # Shortest-path length via one cached BFS per source (callers use only the length).
    def s_path(self, source, destination):
        if source not in self._spl:
            self._spl[source] = nx.single_source_shortest_path_length(self.G, source)
        dist = self._spl[source]
        if destination in dist:
            return None, dist[destination]
        return [], 0
