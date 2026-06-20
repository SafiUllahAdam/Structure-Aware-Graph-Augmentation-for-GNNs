# labels/

Node labels for classification (Cora, Citeseer, Politics, Enzymes, WebKB-Wisconsin).

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

**WebKB note.** `webkb_wisc.labels` pairs with `input/webkb_wisc.edgelist` (the I2V author's Wisconsin
graph, from `github.com/ikenna-oluigbo/webkb-dataset`; node id = `.content` line order, label = last field).
It is **isomorphic** to the existing `input/webkb.edgelist` (479/479 edges) but uses a **different node
numbering**, so labels do **not** transfer to `input/webkb.edgelist` — that graph's labels are unrecoverable
(13 of 265 nodes are structurally ambiguous). Use `webkb_wisc` for any labelled WebKB task. Rebuild with
`python -c "import make_labels; make_labels.make_webkb()"`.
