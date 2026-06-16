# labels/

Node labels for classification (Cora, Citeseer, Politics, Enzymes).

**Format** — one node per line, `node_id` then class label, whitespace-separated:

```
1 Neural_Networks
2 Rule_Learning
3 Reinforcement_Learning
```

- `node_id` must match the IDs used in the matching `input/*.edgelist` and the trained `.emb`
  (Cora uses IDs `1..2708`). Mismatched IDs silently produce wrong F1 — verify alignment.
- File name convention: `{dataset}.labels` (e.g. `cora.labels`, `citeseer.labels`).

Consumed by `eval_nodeclass.py --labels labels/cora.labels`.
