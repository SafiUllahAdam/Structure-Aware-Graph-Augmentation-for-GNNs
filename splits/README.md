# splits/

Exact, repeatable link-prediction splits produced by `prepare_linkpred.py` (seed=42).
Generated files — do not edit by hand.

Per dataset `{name}`:

| file | contents |
|------|----------|
| `{name}_train.edgelist` | 70% positive edges — **retrain I2V/ViRGo on this graph only** (no leakage) |
| `{name}_train_neg.txt`  | non-edges, same count as train positives (classifier negatives) |
| `{name}_test_pos.txt`   | 30% held-out positive edges |
| `{name}_test_neg.txt`   | non-edges, same count as test positives |

All files are `u v` node-id pairs, one per line.

Pipeline:
```
python prepare_linkpred.py --input input/cora.edgelist --name cora     # writes splits/
python train.py --input splits/cora_train.edgelist --output output/cora_lp.emb
python eval_linkpred.py --emb output/cora_lp.emb --name cora           # reports AUC
```
