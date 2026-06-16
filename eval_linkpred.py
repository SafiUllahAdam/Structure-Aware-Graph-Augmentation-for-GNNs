'''Predict whether node pairs are real edges or fake edges using embeddings; report link-prediction AUC.'''

import argparse
import os
import numpy as np
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# Binary operators that turn two node vectors into one edge feature (Hadamard is the node2vec default).
OPERATORS = {
    'hadamard': lambda a, b: a * b,
    'average': lambda a, b: (a + b) / 2.0,
    'l1': lambda a, b: np.abs(a - b),
    'l2': lambda a, b: (a - b) ** 2,
}


# Defines command-line options: embedding file, splits dir, dataset name, edge operator, seed.
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate link prediction (AUC) from embeddings.")
    parser.add_argument('--emb', required=True, help='Embedding trained on the TRAIN graph (no leakage)')
    parser.add_argument('--splits', default='splits', help='Directory holding the split files')
    parser.add_argument('--name', default='cora', help='Dataset name (prefix of split files)')
    parser.add_argument('--op', default='hadamard', choices=list(OPERATORS), help='Edge feature operator. Default hadamard.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed. Default 42.')
    return parser.parse_args()


# Reads "u v" pair lines into a list of (str, str) node-id pairs.
def load_pairs(path):
    '''Load node-id pairs as strings (gensim keys are strings).'''
    with open(path) as f:
        return [tuple(line.split()) for line in f if line.strip()]


# Builds an edge-feature matrix from pairs, skipping pairs whose nodes are missing from the embedding.
def edge_features(kv, pairs, op):
    '''Apply the operator to each present pair; return features and how many were skipped.'''
    fn = OPERATORS[op]
    feats = [fn(kv[u], kv[v]) for u, v in pairs if u in kv and v in kv]
    return np.array(feats), len(pairs) - len(feats)


# Trains logistic regression on edge features and returns the test AUC. Importable by the runner.
def evaluate(emb, splits, name, op='hadamard', seed=42):
    '''Load embedding + splits -> edge features -> logistic regression -> test AUC.'''
    kv = KeyedVectors.load_word2vec_format(str(emb))
    p = os.path.join(str(splits), name)
    sets = {k: load_pairs(f"{p}_{k}.txt") if k != 'train_pos' else load_pairs(f"{p}_train.edgelist")
            for k in ['train_pos', 'train_neg', 'test_pos', 'test_neg']}

    feats, skipped = {}, 0
    for k, pairs in sets.items():
        feats[k], s = edge_features(kv, pairs, op)
        skipped += s

    X_train = np.vstack([feats['train_pos'], feats['train_neg']])
    y_train = np.r_[np.ones(len(feats['train_pos'])), np.zeros(len(feats['train_neg']))]
    X_test = np.vstack([feats['test_pos'], feats['test_neg']])
    y_test = np.r_[np.ones(len(feats['test_pos'])), np.zeros(len(feats['test_neg']))]

    clf = LogisticRegression(max_iter=1000, random_state=seed).fit(X_train, y_train)
    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    if skipped:
        print(f"  warning: skipped {skipped} pairs (node absent from embedding)")
    return auc


# Runs the evaluation from the command line and prints the AUC.
def main(args):
    auc = evaluate(args.emb, args.splits, args.name, args.op, args.seed)
    print(f"{args.name} link prediction | op={args.op} seed={args.seed} | AUC={auc:.4f}")


if __name__ == "__main__":
    main(parse_args())
