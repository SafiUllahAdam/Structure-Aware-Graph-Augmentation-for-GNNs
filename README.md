# ViRGo — Virtual Role-Graph Embedding for Structural Identity

ViRGo is a research project that extends **Identity2Vec** (I2V, Oluigbo et al.). It investigates one practical question:

> **When does a GNN need a rewired graph, and when is the original graph already enough?** We rewire a graph so that nodes playing the *same structural role* become neighbors — a "virtual graph" — compare it against the untouched graph, and try to predict which one wins from the graph's own properties.

The immediate target is the **LoG (Learning on Graphs) conference**; the thesis reuses the same content. Throughout we stay **purely structural** — we never use a dataset's built-in text or attribute features, because the whole point is to show what structure alone can do.

---

## The idea in plain words

- A **graph** is dots and lines: the dots are nodes (papers, people) and the lines are edges (citations, friendships).
- Two nodes that sit far apart can still play the **same role** — both hubs, both bridges. That role is what we call **structural identity**.
- An **embedding** is a short list of numbers (64 of them here) that describes a node. Nodes with similar roles should receive similar numbers.
- A normal GNN passes messages only along real edges, so role-twins that sit far apart never communicate. A **virtual graph** connects each node to its top-K most role-similar nodes, which lets them exchange information.

**What we study.** The variable under study is the virtual graph, not the encoder. We build five graph variants and compare them under one fixed GNN:

| variant | how nodes are matched |
|---|---|
| `psi` | I2V's Poisson/KL structural score Ψ |
| `degree` | degree only (simplest) |
| `centrality` | eigenvector centrality only |
| `original` | the untouched input graph — **control** |
| `hybrid` | original edges + psi role edges together |

**Secondary question.** Given the same graph, does a modern GNN (GraphSAGE) outperform the older walk + Skipgram approach (DeepWalk)?

---

## How the pipeline works

```
input graph → structural signals → virtual graph → encoder → embeddings → evaluation
              (degree, centrality,  (top-K most     (GraphSAGE  (64 numbers  (node classification F1,
               cached once)          similar nodes)   or DeepWalk) per node)   link prediction AUC)
```

1. **Structural signals** — we compute degree and eigenvector centrality for every node once, then cache them. The original I2V recomputed these constantly; caching produces identical output around 200× faster.
2. **Virtual graph** — `virtual_graph.py` connects each node to its K most similar nodes under the chosen variant.
3. **Encoder** — `encoder.py` trains an unsupervised 2-layer GraphSAGE on the virtual graph, using a Skipgram-style loss that pulls linked nodes together and pushes random nodes apart. DeepWalk on the same graph serves as the baseline.
4. **Evaluation** — node classification uses logistic regression on the embeddings (weighted F1); link prediction scores held-out edges (AUC). Link prediction is leakage-free, because the virtual graph is rebuilt from the 70% training edges alone.

Every run is seeded (42/43/44) and cached by filename, so rerunning reuses whatever already exists on disk.

---

## Status

- **Phase 1 — Reproduce I2V. Done.** The cached implementation returns byte-identical embeddings around 200× faster, and Cora scores land within ±0.05 of the published paper. We also compare against DeepWalk, node2vec and struc2vec (baselines are used as published, not tuned).
- **Phase 2 — Build the virtual graphs. Done.** The builder covers all five variants, runs deterministically, and saves each graph to disk. Every graph also records a health row (size, components, isolates, max degree) in `results/graph_health.csv`.
- **Phase 3 — GraphSAGE encoder. Done.** We tuned each design choice one at a time (all on enzymes): training pairs come from the **virtual edges** (A), a node combines its neighbors with a **plain mean** (B), **two layers** work best — three over-smooth link prediction to chance (C), and the **structural input features are necessary** — with random features the GNN drops below DeepWalk (D). K is fixed at **10**.
- **Phase 4 — The study + more data. ← current.** Three steps, in order: (1) run the same pipeline on small-to-medium **OGB** datasets (structural features only); (2) build the **characterization table** that links a graph's properties (homophily first) to whether augmentation helps; (3) swap GraphSAGE for **GIN** and re-check.
- **Future work.** A learnable weight to auto-blend the original and virtual graphs (needs synthetic data), and embeddings as graph summaries for LLMs. Not started.

---

## Results so far (K=10, 3 seeds)

**Main finding: the original graph is a very strong baseline — it wins or ties everywhere.** After fixing several bugs in how the graphs and edge-splits were built, and re-running every dataset, no rewired graph beats the untouched graph on any dataset × task. (Best score per cell shown; every cell's winner is the original graph.)

| dataset | node classification (F1) | link prediction (AUC) |
|---|---|---|
| cora | **original 0.81** | **original 0.90** |
| citeseer | **original 0.55** | **original 0.92** |
| enzymes | **original 0.56** | **original 0.90** |
| proteins | **original 0.58** | **original 0.94** |

**But the size of the gap depends on the data — and that gap is the real result:**

- On the **molecular graphs** (enzymes, proteins), the role-based graphs land within **0.01–0.02** of the original on node classification — structure alone is nearly enough to label a node.
- On the **citation graphs** (cora, citeseer), they trail by a lot — the labels are topics carried by real citations, which role-rewiring throws away.
- On **link prediction**, role graphs fail everywhere (0.50–0.66 vs 0.90–0.94). Sharing a role does not make two nodes likely to be connected.

So the paper is an honest **study**: a guide for *when* structural augmentation helps, not a claim that our virtual graph always wins.

**Encoder comparison (secondary).** On the same role graph, GraphSAGE (message passing) beats DeepWalk (walks) in almost every cell; on the original graph the reverse holds for link prediction. Message passing helps when the graph encodes role; walks win when the graph encodes real connections.

All numbers live in `results/scoreboard.csv` (the master table, one row per dataset × encoder × graph × K × task). The full research log is `docs/paper_log.md`.

---

## Repository map

```
identity2vec/
├── input/                    # original graphs (.edgelist) — never edit
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
│
├── notebooks/
│   ├── 1-reproduce_i2v.ipynb          # Phase 1 — reproduce the I2V paper
│   ├── 2-phase_2_virtual_graph.ipynb  # Phase 2 — build + inspect the virtual graphs
│   └── 3-phase3_gnn_encoder.ipynb     # Phase 3 — train GraphSAGE, all result tables
│
└── scripts/                  # main.py (CLI) · benchmark_config.py (all settings) · runner.py · results_io.py
```

---

## Setup

Uses the conda environment **`i2v`** (Python 3.12):

```bash
conda activate i2v
```

From scratch:

```bash
pip install numpy==1.26.4 networkx gensim==4.3.3 scipy==1.12.0 scikit-learn matplotlib jupyter ipykernel
pip install torch torch-geometric node2vec        # Phase 3 encoder + DeepWalk baseline
```

---

## How to run

**The three notebooks are the main workflow.** Run them in order, top to bottom, using the "Python (i2v)" kernel. Each notebook reuses files already on disk, so reruns are fast.

1. `notebooks/1-reproduce_i2v.ipynb` — reproduces the I2V paper numbers.
2. `notebooks/2-phase_2_virtual_graph.ipynb` — builds and saves the virtual graphs together with the DeepWalk baseline embeddings. Choose the dataset with the `DATASET` knob at the top.
3. `notebooks/3-phase3_gnn_encoder.ipynb` — trains GraphSAGE on the saved graphs and displays every result table: §7 the encoder comparison, §8 the variant sweep, §8b the feature ablation, and §10 the research-question tables.

If you prefer the command line:

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

## Project rules

- Seeds are fixed at **42/43/44** everywhere (splits, initialization, sampling), and every result is a 3-seed mean ± std.
- **Never edit `input/`.** Derived files belong in `output/`, and scores in `results/`.
- Link prediction always retrains on the 70% training graph alone, so no test edge ever leaks.
- A paper number counts as "reproduced" only when it falls within **±0.05** of the original.
- Walk length is pinned to 40 (the repo default; the paper's 80 is a recorded deviation — see `docs/notes.md`).
- Research-significant decisions and results belong in `docs/paper_log.md`; day-to-day changes go to `docs/notes.md`.

---

## Credits

- **Identity2Vec** — *Learning mesoscopic structural identity representations via a Poisson probability metric*, Oluigbo et al. (`identity2vec.py`).
- **ViRGo** extends that work with a cached walker, the virtual-graph study, and a GNN encoder.
