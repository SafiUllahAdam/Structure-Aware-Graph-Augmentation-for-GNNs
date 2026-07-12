# ViRGo — Virtual Role-Graph Embedding for Structural Identity

> A research project on **structural-identity graph representation learning**, aimed to make a research contribution.
> It studies how connecting role-similar nodes into a **virtual graph** and encoding it with a modern **GNN** improves node embeddings across downstream tasks.

---

## 📌 About Project?

A **graph** is just **dots joined by lines** — dots are *nodes* (e.g. papers, people), lines are *edges* (e.g. a citation, a friendship).

Two nodes can play the **same role** even if they sit far apart — both might be *hubs* (many connections) or *bridges* (connecting two groups). This "role" is called **structural identity**.

**Identity2Vec** turns each node into a short list of 64 numbers (an **embedding**, or "fingerprint") that captures its role. Nodes with similar roles get similar fingerprints.

**ViRGo** (this project) asks a research question on top of that:

> Instead of passing messages only along the real edges, we build a **virtual graph** that connects *role-similar* nodes. The central question: **which virtual graph makes a GNN perform best** for each task (node classification, link prediction, anomaly detection)? The virtual graph — not the encoder — is the variable under study; I2V's Poisson/KL graph is just one generic option.


---

## 🗺️ Roadmap

**Phase 1 — Reproducibility (match the I2V paper). ✅ done.**
- [x] Cached I2V variant (identical output, 207× faster).
- [x] Link-prediction AUC vs paper Table 4 — Cora LP **0.8305** vs **0.8413**, within ±0.05.
- [x] Node-classification F1 vs paper — Cora weighted F1 **0.7486** ≈ paper Section 4.4 / Figure 5.
- [x] Cross-model baseline comparison (vs DeepWalk / node2vec / struc2vec) — baselines used as-is, **not fine-tuned**.

**Phase 2 — Virtual-graph creation. ✅ core built** _(the core study)_
- [x] `virtual_graph.py`: top-K virtual-graph builder, 3 variants (Poisson/KL **Ψ**, degree-only, centrality-only), deterministic, CLI + notebook.
- [x] Phase-2 notebook (`notebooks/2-phase_2_virtual_graph.ipynb`): build → verify constraints → K sweep → NC/LP comparison under a fixed DeepWalk bridge encoder (leakage-free LP).
- [x] Graph-health table auto-logged for every graph built → `results/graph_health.csv` (dataset × sim × K: edges, components, isolates…).
- [x] First comparison (Cora, K=10, 3 seeds → `results/snapshots/` + master table `results/scoreboard.csv`): **best virtual graph is task-dependent** — centrality wins node classification, degree wins link prediction; Ψ not best on Cora under the bridge (indicative only).
- [ ] Full scoring of K ∈ {5, 20} + the 3 remaining datasets — **deliberately deferred** until the Phase-3 GNN replaces the DeepWalk bridge (4 datasets total for the published comparison).

**Phase 3 — Modern GNN encoder (ViRGo-SAGE).** ← current _(replace walk + Skipgram with **unsupervised GraphSAGE**, Skipgram-analog loss; full design: `docs/phase3_gnn_design.md`)_

*Spine first (must exist before any variant test):*
- [x] 1. `GNN_PARAMS` in `scripts/benchmark_config.py` (lr, epochs, hidden, layers, Q negatives, agg) — single source of truth like `I2V_PARAMS`.
- [x] 2. Feature builder — structural X = [degree, eigenvector centrality Ω, ψ, clustering], z-normalized (`SageEncoder.features()`; SAGE needs input features, I2V never did).
- [x] 3. Walk-corpus positives — same walk generation as the Phase-2 bridge (I2V params, seeded) on the virtual graph ⇒ Phase-2 vs Phase-3 differ in **one component only** (Skipgram lookup vs message passing) (`SageEncoder.corpus()`).
- [x] 4. `encoder.py` — `SageEncoder` default spine: virtual edgelist → PyG → 2-layer mean GraphSAGE → unsupervised loss → `train(epochs)` → 64-d `.emb` (+ CLI mirroring `virtual_graph.py`).
- [x] 5. Verify the spine once — **done 2026-07-06** (enzymes, Ψ, K=10, 3 seeds: loss falls, evals read the `.emb`, outputs in `output/notebook3_gnn_encoder/`).

*Variant tests (flag flips over the working spine):*
- [ ] Stage 1 — lock the encoder: **A** positives ✅ implemented + first result (`--positives {walk,edge}`; enzymes Ψ K=10, 3 seeds: A2 edge ≥ A1 walk → **A2 default**, cora repeat pending) · **B** aggregation (mean vs Ψ-weighted) next.
- [ ] Stage 2 — the study: winning config × {Ψ, degree, centrality} × K {5,10,20} — "which virtual graph?" under the GNN, vs the Phase-2 bridge table and I2V.
- [ ] Stage 3 — ablations: **C** depth (1–3 layers), **D** features (structural-4 / degree-only / random), **E** original-graph **control row only** (small baseline for comparison, not a full design).

**Phase 4 — Downstream tasks.**
- [ ] Node classification (F1), link prediction (AUC), anomaly detection (new, AUC/AP); virtual-graph ablation (which graph best per data/task).

**Phase 5 — LLM context-window issue.** _(stretch — not yet)_
- [ ] Structural embeddings as a compact large-graph summary for an LLM's context window.

- [ ] Reproducible package + paper draft.

> **Current focus.** Phases 1–2 core are complete (virtual-graph system built, first Cora comparison in). Now **Phase 3 — ViRGo-SAGE**: unsupervised GraphSAGE over the virtual graphs (design locked, then the spine steps 1–5 above). Full K sweep + 4-dataset scoring deliberately deferred until the GNN replaces the DeepWalk bridge. Baselines stay at published/default settings — not fine-tuned — because the contribution is the method (virtual graph + GNN), not baseline tuning.

---

## ✅ What works today

| Task | Dataset | Metric | Result | Status |
|------|---------|--------|--------|--------|
| Node classification | Cora | weighted F1 | **0.6992** | ✅ reproduced (author's embedding) |
| Cached I2V (speedup) | webkb | walk time | **207× faster, byte-identical** | ✅ done (Deliverable #1) |
| Paper-fidelity fixes | I2V core | — | scoring aligned to paper (Δ, `p=Δ`/`q=Ω·d`, candidate-norm, log-space Poisson) | ✅ applied (re-run `.emb` for numbers) |
| Cross-model benchmark | cora · citeseer · webkb · enzymes | F1 / AUC | I2V vs DeepWalk / node2vec / struc2vec | ✅ runs (notebook Steps 5–6) |

*"Reproduced" = our number is within **±0.05** of the paper, with a fixed seed.*

**Latest (2026-06-24):** the core I2V scoring was corrected to follow the paper's equations — degree-distribution Δ, `p = Δ` / `q = Ω·d`, candidate-side normalisation, and a numerically-safe log-space Poisson — plus a gentler Word2Vec setup. Node classification now matches the paper; link prediction stays within paper range. Next-node selection is the paper-exact greedy `min |Ψ−Ψ_curr|` (a temperature sampler was tried and removed 2026-07-07). Details in `docs/notes.md`; re-generate `.emb` files to pick up the changes.

---

## 📁 Repository structure

We normally only touch the **notebook** (`notebooks/1-reproduce_i2v.ipynb`) or **one command** (`scripts/main.py`). Everything else is here for completeness.

```
identity2vec/
├── README.md                 # this file
├── CLAUDE.md                 # instructions for agentic coding
│
├── input/                    # original graphs (.edgelist) — ⚠️ NEVER edit these
├── output/                   # everything generated, notebook-first; each path reads notebook → task → dataset → K → variant:
│   ├── notebook1_reproduce_i2v/<dataset>/{node_classification | link_prediction}/<model>_s<seed>.emb
│   ├── notebook2_create_vir_graph/{virtual_graphs | node_classification | link_prediction}/<dataset>/k<K>/<variant>/…
│   └── notebook3_gnn_encoder/{node_classification | link_prediction}/<dataset>/k<K>/<variant>/<encoder>_s<seed>.emb
├── labels/                   # node categories for classification (cora.labels)
├── splits/                   # link_prediction/{original_graph | virtual_graph_study}/<dataset>/seed_<s>/ — 4 files per seed folder (no leakage; NC needs no split files)
├── results/                  # scoreboard.csv (★ master table) · graph_health.csv · snapshots/ (per-run comparison CSVs) · notebook1_reproduce_i2v/ (Phase-1 tables + benchmark/)
├── logs/                     # training run logs
├── docs/                     # papers (PDFs), notes.md (lab notebook), paper_log.md (curated paper-worthy log), phase3_gnn_design.md (Phase-3 design)
│
├── identity2vec.py           # CORE: the I2V walk algorithm (aligned to the paper's equations — see docs/notes.md)
├── identity2vec_cached.py    # same algorithm, cached → identical output, ~200× faster
├── train.py                  # ▶ makes embeddings:  graph → walks → Word2Vec → .emb
├── plot_emb.py               # draws embeddings as a 2D picture (hubs vs leaves)
│
├── virtual_graph.py          # Phase 2: top-K structural-similarity virtual graph (Ψ / degree / centrality)
├── encoder.py                # Phase 3: ViRGo-SAGE — unsupervised GraphSAGE over the virtual graph → .emb
├── embedding_models.py       # model wrappers (I2V / DeepWalk / node2vec / struc2vec) → same .emb format
│
├── make_labels.py            # downloads + builds label files (cora)
├── prepare_linkpred.py       # builds the 70/30 edge split
├── eval_nodeclass.py         # scores node classification (weighted F1)
├── eval_linkpred.py          # scores link prediction (AUC)
│
├── notebooks/
│   ├── 1-reproduce_i2v.ipynb   # ⭐ Phase 1 — click-through reproduction
│   ├── 2-phase_2_virtual_graph.ipynb # Phase 2 — build, verify + test virtual graphs (graphs only)
│   └── 3-phase3_gnn_encoder.ipynb    # Phase 3 — train + evaluate ViRGo-SAGE on the saved virtual graphs
│
├── scripts/                  # one tidy CLI for every task
│   ├── main.py               #   the single entry point
│   ├── benchmark_config.py   #   all settings in one place (datasets, seed, params)
│   ├── runner.py             #   runs a task end-to-end
│   ├── results_io.py         #   saves scores to results/
│   └── utils.py              #   small shared helpers
│
└── configs/                  # saved run settings (.json)
```

---

## ⚙️ Setup

This project uses the **conda environment `i2v`** (Python 3.12).

```bash
conda activate i2v
```
OCREATE OUR OWN CONDA OR VIRTUAL ENV 
The core libraries are already installed there: `numpy 1.26.4`, `networkx`, `gensim 4.3.3`, `scipy 1.12.0`, `scikit-learn 1.9.0`, `matplotlib`, `jupyter`.

Starting from scratch instead?

```bash
pip install numpy==1.26.4 networkx gensim==4.3.3 scipy==1.12.0 scikit-learn matplotlib jupyter ipykernel
```

---

## 🚀 Quick start — the notebook (easiest)

1. `conda activate i2v`
2. Open `notebooks/1-reproduce_i2v.ipynb` (in VS Code, or run `jupyter lab`).
3. Pick the kernel **"Python (i2v)"**.
4. Run the cells top to bottom (**Shift + Enter**).

We don't type any code — each cell just calls a project function. It reproduces **Cora node classification, weighted F1 = 0.6992**.

---

## 💻 Quick start — command line

```bash
conda activate i2v

# 1. See the available datasets
python scripts/main.py --list

# 2. Build the label file (needs internet, one-time)
python make_labels.py

# 3. Node classification → weighted F1
python scripts/main.py --task nodeclass --dataset cora

# 4. Link prediction → AUC (retrains on the 70% graph, leakage-free, uses the cache)
python scripts/main.py --task linkpred --dataset cora --retrain
```

Make our own embedding from a graph (the **fast cached** path):

```bash
python train.py --input input/cora.edgelist --output output/cora_mine.emb --cached --seed 42
```

Results are saved to `results/notebook1_reproduce_i2v/NNN.<date>.<dataset>.<task>.csv` with a settings header.

---

## 🔬 How the pipeline works

```
graph            structural signal           guided          embedding          evaluation
(dots + lines) → (degree + centrality,   →   walks      →   (64 numbers   →    (F1 / AUC)
                  computed once & cached)     + Word2Vec     per node)
```

1. **Structural signal** — each node's *degree* and *eigenvector centrality*. The original I2V recomputed these inside the walk loop (very slow); `identity2vec_cached.py` computes them **once** → identical results, ~200× faster.
2. **Guided walks** — random walks steered by a Poisson/KL similarity score (the heart of I2V).
3. **Word2Vec (Skipgram)** — turns the walks into one embedding per node.
4. **Evaluation** — a simple model uses the embeddings to classify nodes (F1) or predict missing edges (AUC).

---

## 🔁 Reproducibility

Non-negotiables for this project:

- **Fixed seed `42`** everywhere (splits, initialisation, sampling).
- **Walk-length pinned to `40`** (the repo default; the paper's 80 is ~1.87× slower with no confirmed gain — kept as a recorded deviation, see `docs/notes.md`).
- **Never edit anything in `input/`** — write derived files alongside, outputs to `output/`.
- Every run and decision is logged in **`docs/notes.md`** (the lab notebook).
- A result counts as "reproduced" only when it lands **within ±0.05** of the paper's number.


---

## 📚 Credits

- **Identity2Vec** — *Learning mesoscopic structural identity representations via a Poisson probability metric*, Oluigbo et al. The original algorithm lives in `identity2vec.py`.
- **ViRGo** extends it with a cached walker, a virtual-graph study, and a GNN encoder.
