'''Evaluate OGB-sourced datasets under the OFFICIAL OGB protocol: ogbn-arxiv Accuracy (official split), ogbl-ddi Hits@20.'''
# Separate from eval_nodeclass/eval_linkpred: OGB datasets keep OGB's FIXED split + OGB Evaluator, not the ViRGo random 70/30.
# No-test-peeking is ENFORCED by the split argument: one call scores ONE split. Use "valid" while choosing
# variants/encoders; call with "test" exactly once, after all choices are locked. Never both in one call.
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


# ogbn-arxiv node classification: probe fits on official TRAIN ids, scores ONE chosen split (valid=selection, test=final).
def evaluate_nodeclass(emb, labels_path, split_npz, seed=42, split="valid"):
    '''Fit the logreg probe on official train nodes -> Accuracy (OGB Evaluator) + weighted/macro F1 on ONE split.'''
    from ogb.nodeproppred import Evaluator
    assert split in ("valid", "test"), f"split must be 'valid' (selection) or 'test' (final, read once), got '{split}'"
    kv = KeyedVectors.load_word2vec_format(str(emb))
    y = _load_labels(labels_path)
    s = np.load(split_npz)
    X = lambda ids: np.array([kv[str(int(i))] for i in ids])                       # same probe protocol as the core datasets
    clf = OneVsRestClassifier(LogisticRegression(max_iter=300, solver="lbfgs", random_state=seed))
    clf.fit(X(s["train"]), np.array([y[int(i)] for i in s["train"]]))
    pred = clf.predict(X(s[split]))
    true = np.array([y[int(i)] for i in s[split]])
    acc = Evaluator(name="ogbn-arxiv").eval({"y_true": true.reshape(-1, 1), "y_pred": pred.reshape(-1, 1)})["acc"]
    return {"split": split, "acc": acc,
            "weighted_f1": f1_score(true, pred, average="weighted"), "macro_f1": f1_score(true, pred, average="macro")}


# ogbl-ddi link prediction: cosine-score ONE split's official pos/neg pairs -> Hits@20 (OGB Evaluator, K=20 fixed).
def evaluate_linkpred(emb, pairs_npz, split="valid"):
    '''Cosine-score the chosen split's OGB pos+neg pairs (train-only embedding) -> Hits@20.'''
    from ogb.linkproppred import Evaluator
    assert split in ("valid", "test"), f"split must be 'valid' (selection) or 'test' (final, read once), got '{split}'"
    kv = KeyedVectors.load_word2vec_format(str(emb))
    P = np.load(pairs_npz)
    def cos(pairs):
        a = np.array([kv[str(int(u))] for u, v in pairs]); b = np.array([kv[str(int(v))] for u, v in pairs])
        return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    res = Evaluator(name="ogbl-ddi").eval({"y_pred_pos": cos(P[f"{split}_pos"]), "y_pred_neg": cos(P[f"{split}_neg"])})
    return {"split": split, "hits@20": res["hits@20"]}                             # K=20 is OGB's fixed cutoff for ddi


# Defines command-line options: OGB task + embedding + label/split/pairs paths + which split to score.
def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Evaluate OGB-sourced datasets under the official OGB metric (Accuracy / Hits@20).")
    p.add_argument('--task', required=True, choices=['nodeclass', 'linkpred'], help='ogbn-arxiv=nodeclass, ogbl-ddi=linkpred')
    p.add_argument('--emb', required=True, help='Embedding file (word2vec text format)')
    p.add_argument('--labels', help='ogbn-arxiv labels file (nodeclass)')
    p.add_argument('--split-npz', help='ogbn-arxiv split npz (nodeclass)')
    p.add_argument('--pairs', help='ogbl-ddi pos/neg pairs npz (linkpred)')
    p.add_argument('--split', default='valid', choices=['valid', 'test'],
                   help='valid = model selection (default); test = the FINAL number, run once after all choices are locked.')
    p.add_argument('--seed', type=int, default=42, help='Random seed. Default 42.')
    return p.parse_args()


# Runs the OGB evaluation from the command line and prints the official metric for the chosen split.
def main(args):
    if args.task == 'nodeclass':
        r = evaluate_nodeclass(args.emb, args.labels, args.split_npz, args.seed, args.split)
        print(f"ogbn-arxiv node classification | split={r['split']} | acc={r['acc']:.4f} "
              f"(secondary: weighted_f1={r['weighted_f1']:.4f} macro_f1={r['macro_f1']:.4f})")
    else:
        r = evaluate_linkpred(args.emb, args.pairs, args.split)
        print(f"ogbl-ddi link prediction | split={r['split']} | hits@20={r['hits@20']:.4f}")


if __name__ == "__main__":
    main(parse_args())
