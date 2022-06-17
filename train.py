'''
Codes created by: Ikenna Oluigbo 
Email: ikenna.oluigbo@gmail.com
'''

import argparse
from gensim.models import Word2Vec 
import identity2vec
import networkx as nx


def parse_args():
    '''
    Parses arguments.
    '''
    parser = argparse.ArgumentParser(description="Run identity2vec.")

    parser.add_argument('--input', nargs='?', default='input/cora.edgelist',
                        help='Input graph path')        
    
    parser.add_argument('--output', nargs='?', default='output/cora.emb',
                        help='Output embedding path')

    parser.add_argument('--dimensions', type=int, default=64,
                        help='Number of dimensions. Default is 64.')

    parser.add_argument('--walk-length', type=int, default=40,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--epochs', default=1, type=int,
                      help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--min-count', type=int, default=0,
                        help='Minimum count of Training words. Default is 0.')
    
    parser.add_argument('--sg', type=int, default=1,
                        help='Training Algorithm. CBOW=0,SkipGram=1. Default is 1.')
    
    parser.add_argument('--e', type=int, default=2.7182,
                        help='Euler Constant')
    
    return parser.parse_args()

def build_graph():
    '''Read input network''' 
    
    G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.Graph())
    for e in G.edges:
        G.edges[e]['weight'] = 1 
        
    return G
    

def learn_embeddings(walks):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    identitywalks = [list(map(str, walk)) for walk in walks]
    print("Training Node Corpus...")
    model = Word2Vec(identitywalks, vector_size=args.dimensions, window=args.window_size, 
                     min_count=args.min_count, sg=args.sg, workers=args.workers, epochs=args.epochs,  
                    sample=1e-5, alpha=0.25, min_alpha=0.01, negative=5)
    print("Saving Embeddings...")
    model.wv.save_word2vec_format(args.output)
    
    return model

        
def main(args):
    nx_Graph = build_graph()
    G = identity2vec.Graph(nx_Graph, args.e)
    walks = G.identity2vec_walk(args.num_walks, args.walk_length)
    learn_embeddings(walks) 

if __name__ == "__main__":
    args = parse_args()
    main(args)
    


