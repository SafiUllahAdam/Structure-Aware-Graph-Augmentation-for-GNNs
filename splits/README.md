# splits/

Exact, repeatable link-prediction splits produced by `prepare_linkpred.py`.
Generated files - do not edit by hand. (Node classification needs no split files - its 70/30 split happens in memory from the labels.)

Layout - one folder per dataset per seed:

```
splits/link_prediction/
├── original_graph/<dataset>/seed_<seed>/       # Phase-1 splits (I2V vs baselines)
└── virtual_graph_study/<dataset>/seed_<seed>/  # Phase-2/3 shared splits (bridge vs GNN, fair comparison)
```

Each seed folder holds exactly 4 files:

| file | contents |
|------|----------|
| `train.edgelist` | 70% positive edges - **retrain the encoder on this graph only** (no leakage) |
| `train_neg.txt`  | non-edges, same count as train positives (classifier negatives) |
| `test_pos.txt`   | 30% held-out positive edges |
| `test_neg.txt`   | non-edges, same count as test positives |

All files are `u v` node-id pairs, one per line. Same seed → byte-identical split (deterministic rebuild).

Pipeline:
```
python -m virgo.data.prepare_linkpred --input input/cora.edgelist --seed 42   # writes splits/link_prediction/original_graph/cora/seed_42/
python experiments/train.py --input splits/link_prediction/original_graph/cora/seed_42/train.edgelist --output output/cora_lp.emb --cached
python eval_linkpred.py --emb output/cora_lp.emb --splits splits/link_prediction/original_graph/cora/seed_42   # reports AUC
```
