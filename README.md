# ViRGo - Virtual Role-Graph Embedding for Structural Identity

ViRGo extends **Identity2Vec** (I2V; Oluigbo et al.) to study **when structural graph augmentation helps a GNN, and when the original graph is already sufficient**. We build *virtual graphs* that connect nodes by structural role rather than by their original edges, train a GNN over them, and evaluate on node classification and link prediction across citation and molecular datasets. The aim is a characterization: which graph properties - homophily first - predict whether role-based rewiring improves on the original graph.

The method is **purely structural**: its features are graph-derived - degree, eigenvector centrality, the I2V score Ψ, and clustering - and it never uses external node attributes such as OGB text embeddings or biological descriptions. Excluding attributes is deliberate. They would confound the study, because any gain could then come from the attributes rather than from the structural rewiring under test. For the same reason we do not compare against top OGB leaderboard entries, which may rely on those attributes; the goal is structural analysis, not leaderboard ranking. Target venue: the **Learning on Graphs (LoG)** conference; the thesis draws on the same work.

---

## Approach

A node's *structural identity* is the role it plays - hub, bridge, periphery - independent of where it sits in the graph. A standard GNN passes messages only along real edges, so two nodes that share a role but have no path between them never interact. A **virtual graph** reconnects each node to its top-K most role-similar nodes, giving the encoder a channel between role-twins.

The variable under study is the graph, not the encoder. We compare five constructions under one fixed GNN:

| variant | node similarity |
|---|---|
| `psi` | I2V Poisson/KL structural score Ψ |
| `degree` | degree only |
| `centrality` | eigenvector centrality only |
| `original` | input graph, unchanged - control |
| `hybrid` | original edges combined with `psi` role edges |

A secondary question: on a given graph, does GraphSAGE (message passing) outperform the older walk + Skipgram approach (DeepWalk)?

---

## Pipeline

```
input graph → structural signals → virtual graph → encoder → embeddings → evaluation
              (degree, centrality,  (top-K most     (GraphSAGE  (64-dim      (node classification F1,
               cached once)          similar nodes)   or DeepWalk) per node)   link prediction AUC)
```

1. **Structural signals** - degree and eigenvector centrality are computed once and cached. I2V recomputes them inside its walk loop; caching returns identical output roughly 200× faster.
2. **Virtual graph** - `virtual_graph.py` links each node to its K nearest nodes under the chosen variant.
3. **Encoder** - `encoder.py` trains a 2-layer GraphSAGE on the virtual graph with an unsupervised Skipgram-style loss (linked nodes attract, random nodes repel). DeepWalk on the same graph is the baseline.
4. **Evaluation** - node classification uses logistic regression on the embeddings (weighted F1). Link prediction scores held-out edges by AUC, with the virtual graph rebuilt from the 70% training edges so no test edge leaks.

Runs are seeded (42/43/44) and cached by filename, so reruns reuse existing files.

---

## Status

- **Phase 1 - reproduce I2V.** Done. The cached implementation returns byte-identical embeddings ~200× faster, and Cora lands within ±0.05 of the published paper. DeepWalk, node2vec and struc2vec are included as published baselines (not tuned).
- **Phase 2 - virtual graphs.** Done. All five variants, deterministic, each logged to `results/graph_health.csv` (size, components, isolates, max degree).
- **Phase 3 - GraphSAGE encoder.** Done. The design is fixed by ablations on enzymes: training pairs come from the virtual edges (A), aggregation is mean (B), depth is two layers (C - three over-smooth), and the structural features are required (D - replacing them with random features drops performance to the DeepWalk baseline or below). Those four features do double duty: they define the virtual graph and serve as the encoder's input. K = 10.
- **Phase 4 - characterization and scale.** Current. (1) Add small-to-medium OGB datasets. On each, the original and virtual graphs receive identical structural features, so only the edges differ and graph structure stays the single variable; OGB's extra attributes are ignored. (2) Relate graph properties (homophily first) to the original-vs-augmented gap. (3) Swap GraphSAGE for GIN.
- **Future work.** A learnable weight that blends the original and virtual graphs per dataset (needs synthetic data), and embeddings as compact graph summaries for LLMs.

---

## Results (K = 10, 3 seeds)

The primary question is which virtual graph performs best per dataset and task, with the encoder held fixed at GraphSAGE. `original` is the control and is excluded from Table 1.

**Table 1 - best virtual graph per dataset × task** (GraphSAGE, K = 10):

| dataset | node classification (F1) | link prediction (AUC) |
|---|---|---|
| cora | centrality 0.29 | centrality 0.57 |
| citeseer | centrality 0.28 | centrality 0.54 |
| enzymes | hybrid 0.55 | centrality 0.66 |
| proteins | hybrid 0.56 | centrality 0.58 |

Centrality is the strongest single signal - best for link prediction on all four datasets and for node classification on the two citation graphs. Hybrid, which keeps the real edges alongside the role edges, leads node classification on the molecular graphs.

**Table 2 - virtual graph vs. original** (same encoder on both sides):

| dataset | NC · original | NC · best virtual | LP · original | LP · best virtual |
|---|---|---|---|---|
| cora | 0.43 | 0.29 | 0.61 | 0.57 |
| citeseer | 0.33 | 0.28 | 0.62 | 0.54 |
| enzymes | 0.56 | 0.55 | 0.70 | 0.66 |
| proteins | 0.58 | 0.56 | 0.67 | 0.58 |

The original graph leads every cell, and the margin varies with the dataset. On the molecular graphs the best virtual graph reaches within 0.01–0.02 of the original for node classification, so structural role alone nearly suffices for molecular labels. On the citation graphs the gap is wider: the labels track citation communities, which role-based rewiring discards. For link prediction, no virtual graph matches the original on any dataset - role similarity does not imply adjacency.

**Encoder comparison (secondary).** On role graphs, GraphSAGE outperforms DeepWalk in nearly every cell. On the original graph the order reverses for link prediction, where DeepWalk reaches 0.90–0.94, the highest link-prediction scores in the study. Message passing helps when the graph encodes role; walks are stronger when the graph encodes adjacency.

Full results live in `results/scoreboard.csv` (one row per dataset × encoder × graph × K × task). Notebook 3 §10 builds both tables. The research log is `docs/paper_log.md`.

---

## Repository map

```
identity2vec/
├── input/                    # original graphs (.edgelist) - never edit
├── labels/                   # node labels for classification
├── splits/                   # saved 70/30 link-prediction splits (per dataset, per seed)
├── output/                   # everything generated: notebook1_*/  notebook2_*/  notebook3_*/
├── results/                  # scoreboard.csv (master) · graph_health.csv · snapshots/
├── docs/                     # paper_log.md (research log) · notes.md (lab notebook) · designs
│
├── identity2vec.py           # original I2V walk algorithm
├── identity2vec_cached.py    # same algorithm with caching (identical output, ~200× faster)
├── train.py                  # graph → I2V walks → Word2Vec → .emb
├── virtual_graph.py          # Phase 2: build a virtual graph (psi/degree/centrality/original/hybrid)
├── encoder.py                # Phase 3: unsupervised GraphSAGE over a virtual graph → .emb
├── embedding_models.py       # wrappers: I2V / DeepWalk / node2vec / struc2vec → same .emb format
├── prepare_linkpred.py       # makes the leakage-free 70/30 edge split
├── eval_nodeclass.py         # scores node classification (weighted F1)
├── eval_linkpred.py          # scores link prediction (AUC)
├── make_labels.py            # downloads + builds label files
├── make_ogb.py               # Phase 4: OGB -> ViRGo files (edgelist, .nodes, labels, official splits)
├── eval_ogb.py               # Phase 4: official OGB metrics (arxiv Accuracy, ddi Hits@20 via a trained link decoder), one split per call
├── run_ogb.py                # Phase 4: OGB pipeline functions (notebook 4 imports these; also a CLI)
│
├── notebooks/
│   ├── 1-reproduce_i2v.ipynb          # Phase 1 - reproduce the I2V paper
│   ├── 2-phase_2_virtual_graph.ipynb  # Phase 2 - build + inspect the virtual graphs
│   ├── 3-phase3_gnn_encoder.ipynb     # Phase 3 - train GraphSAGE, all result tables
│   └── 4-phase4_ogb.ipynb             # Phase 4 - OGB datasets under the official protocol
│
└── scripts/                  # main.py (CLI) · benchmark_config.py (all settings) · runner.py · results_io.py
```

---

## Setup

The project uses the conda environment **`i2v`** (Python 3.12):

```bash
conda activate i2v
```

From scratch:

```bash
pip install numpy==1.26.4 networkx gensim==4.3.3 scipy==1.12.0 scikit-learn matplotlib jupyter ipykernel
pip install torch torch-geometric node2vec        # Phase 3 encoder + DeepWalk baseline
```

---

## Usage

The three notebooks are the main workflow. Run them in order, top to bottom, on the "Python (i2v)" kernel; each reuses files already on disk.

1. `notebooks/1-reproduce_i2v.ipynb` - reproduces the I2V paper numbers.
2. `notebooks/2-phase_2_virtual_graph.ipynb` - builds and saves the virtual graphs and the DeepWalk baseline embeddings. Pick the dataset with the `DATASET` knob at the top.
3. `notebooks/3-phase3_gnn_encoder.ipynb` - trains GraphSAGE and renders the result tables: §7 encoder comparison, §8 variant sweep, §8b feature ablation, §10 research-question tables.
4. `notebooks/4-phase4_ogb.ipynb` - runs an OGB dataset end to end under the official protocol (pick it with the `DATASET` knob): download, virtual graphs, embeddings, validation selection, then a single guarded test read.

Command-line equivalents:

```bash
# build one virtual graph
python virtual_graph.py --input input/cora.edgelist --sim psi --k 10

# train GraphSAGE on it
python encoder.py --input input/cora.edgelist --sim psi --k 10 --seed 42

# I2V embedding (fast cached path)
python train.py --input input/cora.edgelist --output output/cora_mine.emb --cached --seed 42

# Phase-1 tasks
python scripts/main.py --list
python scripts/main.py --task nodeclass --dataset cora
python scripts/main.py --task linkpred --dataset cora --retrain
```

---

## Reproducibility and conventions

- Seeds are fixed at 42/43/44 across splits, initialization and sampling; every result is a 3-seed mean ± std.
- `input/` is read-only. Derived files go to `output/`, scores to `results/`.
- Link prediction retrains on the 70% training graph alone, so no test edge leaks.
- A paper number is treated as reproduced only within ±0.05 of the original.
- Walk length is pinned to 40 (the repo default; the paper's 80 is a recorded deviation - see `docs/notes.md`).
- Research decisions and results go to `docs/paper_log.md`; day-to-day changes go to `docs/notes.md`.

---

## Credits

- **Identity2Vec** - *Learning mesoscopic structural identity representations via a Poisson probability metric*, Oluigbo et al. (`identity2vec.py`).
- **ViRGo** builds on that work with a cached walker, the virtual-graph study, and a GNN encoder.
