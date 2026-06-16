# scripts/ — ViRGo reproduction framework

CoBench-style scaffolding that **wraps the existing root scripts** (`identity2vec.py`, `train.py`,
`prepare_linkpred.py`, `eval_linkpred.py`, `eval_nodeclass.py`). Nothing is moved; these files only
import or subprocess-call the root files.

| file | role |
|------|------|
| `benchmark_config.py` | single source of truth: paths, dataset registry, I2V params, seed=42 |
| `utils.py` | `set_seed`, `load_embeddings`, `next_run_id` |
| `results_io.py` | write `results/NNN.{dataset}.{task}.csv` with a JSON `#META` header |
| `runner.py` | orchestrate split → (optional) embed → eval; returns metrics |
| `main.py` | CLI entry point |

## Run

```bash
python scripts/main.py --list                              # show datasets

# Link prediction (leakage-free): split 70/30, retrain on the 70% graph, AUC
python scripts/main.py --task linkpred --dataset cora --retrain

# Link prediction plumbing check on an existing full-graph embedding (NOT a paper number — leaks)
python scripts/main.py --task linkpred --dataset cora --emb output/cora.emb

# Node classification (needs labels/cora.labels)
python scripts/main.py --task nodeclass --dataset cora --emb output/cora.emb
```

Requires `scikit-learn` (`pip install scikit-learn`). Results land in `results/`.
