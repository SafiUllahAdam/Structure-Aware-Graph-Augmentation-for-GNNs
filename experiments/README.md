# experiments/ — the runnable entry points

Every file here is an `argparse` CLI. They hold **no method code**: the algorithms live in `virgo/`
(imported), and these scripts only wire a dataset to a pipeline and record the result. Notebooks 4
and 5 import `run_ogb.py` and `characterize.py` for their functions instead of duplicating them.

| file | role |
|------|------|
| `run_core.py` | the frozen sweep for the non-OGB datasets: virtual graph → encoder → NC + LP → scoreboard row |
| `run_ogb.py` | the same sweep under the **official OGB protocol** (train-only embeddings, valid/test discipline) |
| `characterize.py` | measures graph properties + the encoder's raw inputs, relates them to the original-vs-augmented gap, and screens each property as an executable keep-vs-augment rule — each rule's split is also refitted with one dataset hidden, on its own and with the property re-chosen (`--datasets` defaults to the seven-dataset `PANEL`) |
| `train.py` | Phase-1 run file: graph → I2V walks → Word2Vec → `.emb` (`--cached` uses the fast path) |
| `train_encoder.py` | trains ONE encoder over ONE virtual graph; `--arch` picks any encoder registered in `virgo.encoders` |
| `benchmark_baselines.py` | I2V vs DeepWalk / node2vec / struc2vec across the benchmark datasets (no flags — runs the whole loop) |
| `run_task.py` | small reproduction CLI: one task, one dataset, one result CSV |
| `plot_emb.py` | draws an embedding as a 2-D picture |

Run them from the repo root — data paths (`input/`, `output/`, …) are resolved against it.

## Run

```bash
python experiments/run_task.py --list                              # show datasets

# Link prediction (leakage-free): split 70/30, retrain on the 70% graph, AUC
python experiments/run_task.py --task linkpred --dataset cora --retrain

# Link prediction plumbing check on an existing full-graph embedding (NOT a paper number — leaks)
python experiments/run_task.py --task linkpred --dataset cora --emb output/notebook1_reproduce_i2v/cora/node_classification/i2v_s42.emb

# Node classification (needs labels/cora.labels)
python experiments/run_task.py --task nodeclass --dataset cora --emb output/notebook1_reproduce_i2v/cora/node_classification/i2v_s42.emb

# One encoder over one virtual graph (--arch = any registered encoder)
python experiments/train_encoder.py --input input/cora.edgelist --arch graphsage --sim psi --k 10 --seed 42
```

Requires `scikit-learn` (`pip install scikit-learn`). Results land in `results/notebook1_reproduce_i2v/`.
