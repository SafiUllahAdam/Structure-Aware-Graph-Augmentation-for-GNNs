'''Evaluate OGB-sourced datasets under the OFFICIAL OGB protocol: ogbn-arxiv Accuracy (official split), ogbl-ddi Hits@20.'''
# Separate from eval_nodeclass/eval_linkpred: OGB datasets keep OGB's FIXED split + OGB Evaluator, not the ViRGo random 70/30.
# Discipline: model choices (which variant/encoder) are picked on VALIDATION; TEST is read once -> no test-peeking.
# Same cosine scorer + same linear probe for every variant -> fair internal comparison; only the split/metric are OGB's.

import argparse
import numpy as np
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier


# Reads a labels file into {node_id(int) -> class(int)}.
def _load_labels(path):
    '''Load "node_id label" lines into an int->int dict.'''
    y = {}
    for line in open(path):
        if line.strip():
            n, lab = line.split()[:2]
            y[int(n)] = int(lab)
    return y


# ogbn-arxiv node classification on the official split: linear probe on train, Accuracy (primary) + weighted/macro F1 (secondary).
def evaluate_nodeclass(emb, labels_path, split_npz, seed=42):
    '''Fit the logreg probe on official TRAIN nodes; return valid/test Accuracy (OGB Evaluator) + test weighted/macro F1.'''
    from ogb.nodeproppred import Evaluator
    kv = KeyedVectors.load_word2vec_format(str(emb))
    y = _load_labels(labels_path)
    s = np.load(split_npz)
    X = lambda ids: np.array([kv[str(int(i))] for i in ids])                       # same probe protocol as the core datasets
    clf = OneVsRestClassifier(LogisticRegression(max_iter=300, solver="lbfgs", random_state=seed))
    clf.fit(X(s["train"]), np.array([y[int(i)] for i in s["train"]]))
    ev, out = Evaluator(name="ogbn-arxiv"), {}
    for name in ("valid", "test"):
        ids = s[name]
        pred = clf.predict(X(ids))
        true = np.array([y[int(i)] for i in ids])
        out[f"{name}_acc"] = ev.eval({"y_true": true.reshape(-1, 1), "y_pred": pred.reshape(-1, 1)})["acc"]
        if name == "test":
            out["test_weighted_f1"] = f1_score(true, pred, average="weighted")
            out["test_macro_f1"] = f1_score(true, pred, average="macro")
    return out


# ogbl-ddi link prediction: cosine-score OGB pos/neg pairs, Hits@20 via the OGB Evaluator (K=20 fixed). Same scorer for every variant.
def evaluate_linkpred(emb, pairs_npz):
    '''Cosine-score OGB valid/test pos+neg pairs (train-only embedding); return valid/test Hits@20 (OGB Evaluator).'''
    from ogb.linkproppred import Evaluator
    kv = KeyedVectors.load_word2vec_format(str(emb))
    P = np.load(pairs_npz)
    ev = Evaluator(name="ogbl-ddi")                                                # K=20 is OGB's fixed cutoff for ddi
    def cos(pairs):
        a = np.array([kv[str(int(u))] for u, v in pairs]); b = np.array([kv[str(int(v))] for u, v in pairs])
        return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    out = {}
    for name in ("valid", "test"):
        res = ev.eval({"y_pred_pos": cos(P[f"{name}_pos"]), "y_pred_neg": cos(P[f"{name}_neg"])})
        out[f"{name}_hits@20"] = res["hits@20"]
    return out


# Defines command-line options: OGB task + embedding + label/split/pairs paths.
def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Evaluate OGB-sourced datasets under the official OGB metric (Accuracy / Hits@20).")
    p.add_argument('--task', required=True, choices=['nodeclass', 'linkpred'], help='ogbn-arxiv=nodeclass, ogbl-ddi=linkpred')
    p.add_argument('--emb', required=True, help='Embedding file (word2vec text format)')
    p.add_argument('--labels', help='ogbn-arxiv labels file (nodeclass)')
    p.add_argument('--split', help='ogbn-arxiv split npz (nodeclass)')
    p.add_argument('--pairs', help='ogbl-ddi pos/neg pairs npz (linkpred)')
    p.add_argument('--seed', type=int, default=42, help='Random seed. Default 42.')
    return p.parse_args()


# Runs the OGB evaluation from the command line and prints the official metric.
def main(args):
    if args.task == 'nodeclass':
        r = evaluate_nodeclass(args.emb, args.labels, args.split, args.seed)
        print(f"ogbn-arxiv node classification | valid_acc={r['valid_acc']:.4f} test_acc={r['test_acc']:.4f} "
              f"(secondary: test_weighted_f1={r['test_weighted_f1']:.4f} test_macro_f1={r['test_macro_f1']:.4f})")
    else:
        r = evaluate_linkpred(args.emb, args.pairs)
        print(f"ogbl-ddi link prediction | valid_hits@20={r['valid_hits@20']:.4f} test_hits@20={r['test_hits@20']:.4f}")


if __name__ == "__main__":
    main(parse_args())
