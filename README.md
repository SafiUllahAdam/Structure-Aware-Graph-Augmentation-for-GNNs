# ViRGo — Virtual Role-Graph Embedding for Structural Identity

Research project. Builds on **Identity2Vec** (I2V, Oluigbo et al.) and asks one question:

> **Which graph should a GNN learn on?** We rewire a graph so that nodes with the *same structural role* become neighbors (a "virtual graph"), then test which rewiring works best for each dataset and task.

---

## The idea in plain words

- A **graph** is dots and lines: dots = nodes (papers, people), lines = edges (citations, friendships).
- Two nodes far apart can still play the **same role** — both hubs, both bridges. This role is called **structural identity**.
- An **embedding** is a short list of numbers (64 here) describing each node. Nodes with similar roles should get similar numbers.
- A normal GNN only passes messages along real edges, so role-twins that sit far apart never talk. A **virtual graph** connects each node to its top-K most role-similar nodes — now they talk.

**What we study:** the virtual graph is the variable, not the encoder. We build five graph variants and compare them under one fixed GNN:

| variant | how nodes are matched |
|---|---|
| `psi` | I2V's Poisson/KL structural score Ψ |
| `degree` | degree only (simplest) |
| `centrality` | eigenvector centrality only |
| `original` | the untouched input graph — **control** |
| `hybrid` | original edges + psi role edges together |

**Secondary question:** on the same graph, does a modern GNN (GraphSAGE) beat the old walk + Skipgram (DeepWalk)?

---

## How the pipeline works

```
input graph → structural signals → virtual graph → encoder → embeddings → evaluation
              (degree, centrality,  (top-K most     (GraphSAGE  (64 numbers  (node classification F1,
               cached once)          similar nodes)   or DeepWalk) per node)   link prediction AUC)
```

1. **Structural signals** — degree + eigenvector centrality per node, computed once and cached (the original I2V recomputed them constantly; caching gives identical output ~200× faster).
2. **Virtual graph** — `virtual_graph.py` connects each node to its K most similar nodes under the chosen variant.
3. **Encoder** — `encoder.py` trains an unsupervised 2-layer GraphSAGE on the virtual graph (Skipgram-style loss: pull linked nodes together, push random nodes apart). DeepWalk on the same graph is the baseline.
4. **Evaluation** — logistic regression on the embeddings for node classification (weighted F1); held-out edge scoring for link prediction (AUC). Link prediction is leakage-free: the virtual graph is rebuilt from the 70% training edges only.

Everything is seeded (42/43/44) and cached by filename — rerunning reuses what exists.

---

## Status

- **Phase 1 — reproduce I2V. Done.** Cached I2V byte-identical and ~200× faster; Cora scores within ±0.05 of the paper; compared against DeepWalk / node2vec / struc2vec (baselines used as published, not tuned).
- **Phase 2 — build virtual graphs. Done.** Builder with all five variants, deterministic, saved to disk; every graph logs a health row (size, components, isolates) to `results/graph_health.csv`.
- **Phase 3 — GNN encoder. Done (K=10, cora + enzymes).** Encoder settings locked by ablations: positives = direct virtual edges (A), aggregation = mean (B); feature ablation (D) run on enzymes. Full 5-variant × 2-encoder grid scored on cora and enzymes.
- **Phase 4 — remaining datasets, K = 5/20 sweep, anomaly detection.** Next.
- **Phase 5 — embeddings as graph summaries for LLMs.** Stretch, not started.

---

## Results so far (K=10, 3 seeds)

**Main finding: the best graph depends on the data.**

- **Cora** (citation network, community-like labels): the **original graph wins everything** — DeepWalk on the real edges gets NC 0.81 / LP 0.90; every rewired graph scores far lower. Role edges destroy Cora's community signal.
- **Enzymes** (molecular graphs): rewiring **helps link prediction** — the centrality virtual graph (0.72–0.74 AUC) beats the original graph (0.65); node classification stays close to the control.

**Headline answers, GraphSAGE fixed** (notebook 3, section 11):

| dataset | task | best virtual graph | score | original (control) |
|---|---|---|---|---|
| cora | NC (F1) | hybrid | 0.31 | **0.45** |
| cora | LP (AUC) | centrality | 0.57 | **0.62** |
| enzymes | NC (F1) | degree | 0.55 | **0.57** |
| enzymes | LP (AUC) | **centrality** | **0.72** | 0.65 |

So far rewiring clearly wins only enzymes link prediction — the honest story is *per-data, per-task*, not "virtual graphs always win".

**Encoder comparison** (secondary, section 7): GraphSAGE beats DeepWalk on the pure role graphs (psi, degree); DeepWalk wins on the original graph. The feature ablation (D) shows most of the GNN's gain comes from its structural input features, not message passing alone.

All numbers live in `results/scoreboard.csv` (master table, one row per dataset × encoder × graph × K × task). The full research log is `docs/paper_log.md`.

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

**The three notebooks are the main workflow** — run them in order, top to bottom, kernel "Python (i2v)". Each one reuses files already on disk, so reruns are fast.

1. `notebooks/1-reproduce_i2v.ipynb` — reproduces the I2V paper numbers.
2. `notebooks/2-phase_2_virtual_graph.ipynb` — builds and saves the virtual graphs + the DeepWalk baseline embeddings. Pick the dataset with the `DATASET` knob at the top.
3. `notebooks/3-phase3_gnn_encoder.ipynb` — trains GraphSAGE on the saved graphs and shows every result table: §7 encoder comparison, §8 variant sweep, §11 the research-question tables.

Command line, if you prefer:

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

- Fixed seeds **42/43/44** everywhere (splits, init, sampling). Every result is a 3-seed mean ± std.
- **Never edit `input/`.** Derived files go to `output/`, scores to `results/`.
- Link prediction always retrains on the 70% train graph only — no leakage.
- A paper number counts as "reproduced" only within **±0.05** of the original.
- Walk length pinned to 40 (repo default; the paper's 80 is a recorded deviation — see `docs/notes.md`).
- Research-significant decisions and results go to `docs/paper_log.md`; day-to-day changes to `docs/notes.md`.

---

## Credits

- **Identity2Vec** — *Learning mesoscopic structural identity representations via a Poisson probability metric*, Oluigbo et al. (`identity2vec.py`).
- **ViRGo** extends it with a cached walker, the virtual-graph study, and a GNN encoder.
