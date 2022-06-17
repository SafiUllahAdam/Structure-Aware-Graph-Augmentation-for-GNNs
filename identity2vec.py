import numpy as np 
import networkx as nx
from tqdm import tqdm
import time
from decimal import Decimal


class Graph():
    def __init__(self, nx_Graph, e):
        self.G = nx_Graph
        self.e = e
   
    
    def degree_node(self):
        G = self.G
        deg_dict = {}
        for node, deg in G.degree:
            deg_dict[node] = deg
        return deg_dict 
    
    
    def eigenvector_centrality(self):
        G = self.G
        ev = nx.eigenvector_centrality(G, max_iter=1000)
        return ev
    
        
    def node_neighbors(self):
        G = self.G
        node_neigh = {}
        for node in G.nodes:
            node_neigh[node] = list(G.neighbors(node))
        return node_neigh


    def test_source(self, s):
        mnn = self.node_neighbors()[s] 
        arr_new_nn = np.array(mnn)
        np.random.shuffle(arr_new_nn)
        next_node = np.random.choice(arr_new_nn)
        return next_node
    
    
    def skip_visited(self, snn, visited):
        if len(snn) != 1:
            if len(visited) > 1:
                last_visit = visited[-2]
                if last_visit in snn:
                    snn.remove(last_visit)
                    self.skip_visited(snn, visited)
        return snn
    
    
    def identity_walker(self, node, walk_length):
        walk = [node]
        visited = [node]
        while len(walk) < walk_length:
            current_node = walk[-1]
            nn = self.node_neighbors()[current_node] 
            if len(nn) == 0:
                break 
            if visited[-1] == node:
                next_node = self.test_source(visited[-1])
                walk.append(next_node)
                visited.append(next_node)
            else:
                nn = self.skip_visited(nn, visited)
                bounded_curr = walk[-2]
                pdn = self.poisson_dist(nn, bounded_curr)
                next_node = min(pdn, key=pdn.get) 
                walk.append(next_node)
                visited.append(next_node) 
                
        return walk
    
    
    def s_path(self, source, destination): 
        G = self.G
        if nx.has_path(G, source, destination):
            path = nx.shortest_path(G, source, destination)
            path_length = nx.shortest_path_length(G, source, destination) 
        else:
            path = []
            path_length = len(path)
        return path, path_length
    
       
    def get_prob(self, n, curr):
        G = self.G
        neigh = list(G.neighbors(n))
        p_val = self.degree_node()
        ev = self.eigenvector_centrality()
        p = []
        q = []
        for node in neigh:
            _, path_length = self.s_path(curr, node)
            path_length += 0.01     #Prevent PathLength 0
            p.append(p_val[node] * ev[node])
            q.append(path_length)
        return p, q
        
    
    def poisson_dist(self, mnn, bounded_curr):
        #k = Number of adjacent neighbors
        #λ = Divergence rate
        #pdn = (λ**k * e**-λ) / k! 
        #KLDivergence = #sum(p(x) * log(p(x)/q(x)))
        e = self.e
        pdn = {}
        k = len(mnn)       
        for node in mnn:
            rt = 0
            p, q = self.get_prob(node, bounded_curr)
            for i in range(len(p)):
                t = p[i] * np.log(p[i] / q[i])
                rt += t 
            drt = (1 / (self.degree_node()[bounded_curr]+self.eigenvector_centrality()[bounded_curr])) * rt
            #poiss = Decimal((np.power(drt, k) * np.power(e, -drt))) / Decimal(np.math.factorial(k))
            poiss = (np.power(drt, k) * np.power(e, -drt)) / (np.math.factorial(k))
            pdn[node] = poiss
        return pdn
    
    
    def identity2vec_walk(self, num_walk, walk_length):
        
        G = self.G
        print("STARTING RANDOM WALK... ")
        print("Number of Nodes:", len(G.nodes))
        time.sleep(3)
        
        nodes = list(G.nodes)
        walk_corpus = []
        
        for cw in range(1, num_walk+1):
            print("\n")
            print("Current Walk: " + str(cw) + " of " + str(num_walk))
            for node in tqdm(nodes):
                node_walk = self.identity_walker(node, walk_length)
                walk_corpus.append(node_walk)            
               
        return walk_corpus
    




    
    