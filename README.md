# ViRGo — Virtual Role-Graph Embedding for Structural Identity

> A research project that extends **Identity2Vec (I2V)** by Oluigbo et al. toward a publishable paper.
> Built on top of the original I2V code (`identity2vec.py`), corrected to follow the paper's equations and kept reproducible.

---

## 📌 About Project?

A **graph** is just **dots joined by lines** — dots are *nodes* (e.g. papers, people), lines are *edges* (e.g. a citation, a friendship).

Two nodes can play the **same role** even if they sit far apart — both might be *hubs* (many connections) or *bridges* (connecting two groups). This "role" is called **structural identity**.

**Identity2Vec** turns each node into a short list of 64 numbers (an **embedding**, or "fingerprint") that captures its role. Nodes with similar roles get similar fingerprints.

**ViRGo** (this project) asks a research question on top of that:

> Instead of passing messages only along the real edges, can we build a **virtual graph** that connects *role-similar* nodes, and does a modern **GNN** over that virtual graph beat the original walk-based method?


---

## 🗺️ Roadmap

**Phase 1 — Reproducibility (match the I2V paper). ✅ done.**
- [x] Cached I2V variant (identical output, 207× faster).
- [x] Link-prediction AUC vs paper Table 4 — Cora LP **0.8305** vs **0.8413**, within ±0.05.
- [x] Node-classification F1 vs paper — Cora weighted F1 **0.7486** ≈ paper Section 4.4 / Figure 5.
- [x] Cross-model baseline comparison (vs DeepWalk / node2vec / struc2vec) — baselines used as-is, **not fine-tuned**.
- [ ] (optional) Train-ratio sweep (30–70%) to match Figure 5 *exactly*.

**Phase 2 — Virtual-graph creation.** ← next
- [ ] `virtual_graph.py`: top-K Poisson/KL Ψ builder + degree-only / centrality-only comparison graphs.

**Phase 3 — Modern GNN encoder.** _(technical contribution)_
- [ ] GNN over the virtual graph (GraphSAGE / GIN / GAT) replacing walk + Skipgram; design + compare architecture variants.

**Phase 4 — Downstream tasks.**
- [ ] Node classification (F1), link prediction (AUC), anomaly detection (new, AUC/AP); virtual-graph ablation (which graph best per data/task).

**Phase 5 — LLM context-window issue.** _(stretch — not yet)_
- [ ] Structural embeddings as a compact large-graph summary for an LLM's context window.

- [ ] Reproducible package + paper draft.

> **Current focus.** Phase 1 is complete. Next is **Phase 2 — building the virtual graph**, then **Phase 3 — the GNN encoder** replacing walk + Skipgram. Baselines stay at published/default settings — not fine-tuned — because the contribution is the encoder, not baseline tuning.

---

## ✅ What works today

| Task | Dataset | Metric | Result | Status |
|------|---------|--------|--------|--------|
| Node classification | Cora | weighted F1 | **0.6992** | ✅ reproduced (author's embedding) |
| Cached I2V (speedup) | webkb | walk time | **207× faster, byte-identical** | ✅ done (Deliverable #1) |
| Paper-fidelity fixes | I2V core | — | scoring aligned to paper (Δ, `p=Δ`/`q=Ω·d`, candidate-norm, log-space Poisson) | ✅ applied (re-run `.emb` for numbers) |
| Cross-model benchmark | cora · citeseer · webkb · enzymes | F1 / AUC | I2V vs DeepWalk / node2vec / struc2vec | ✅ runs (notebook Steps 5–6) |
| I2V + temperature (τ=0.3) | Cora | NC F1 / LP AUC | NC **0.7486** · LP **0.8305** (seed 42) | ✅ improved, near paper |

*"Reproduced" = our number is within **±0.05** of the paper, with a fixed seed.*

**Latest (2026-06-24):** the core I2V scoring was corrected to follow the paper's equations — degree-distribution Δ, `p = Δ` / `q = Ω·d`, candidate-side normalisation, and a numerically-safe log-space Poisson — plus a gentler Word2Vec setup. Node classification now matches the paper; link prediction stays within paper range. A non-greedy **temperature** sampler (τ=0.3) was then added to next-node selection — on Cora it lifts both tasks (NC weighted F1 **0.7486**, LP AUC **0.8305**, seed 42). Details in `docs/notes.md`; re-generate `.emb` files to pick up the changes.

---

## 📁 Repository structure

We normally only touch the **notebook** (`notebooks/reproduce_i2v.ipynb`) or **one command** (`scripts/main.py`). Everything else is here for completeness.

```
identity2vec/
├── README.md                 # this file
├── CLAUDE.md                 # instructions for agentic coding
│
├── input/                    # original graphs (.edgelist) — ⚠️ NEVER edit these
├── output/                   # trained embeddings (.emb): cora.emb (author), webkb.emb (trained)
├── labels/                   # node categories for classification (cora.labels)
├── splits/                   # 70/30 edge splits for link prediction (no leakage)
├── results/                  # scores (numbered .csv) + plots (.png)
├── logs/                     # training run logs
├── docs/                     # papers (PDFs) + notes.md (the lab notebook)
│
├── identity2vec.py           # CORE: the I2V walk algorithm (aligned to the paper's equations — see docs/notes.md)
├── identity2vec_cached.py    # same algorithm, cached → identical output, ~200× faster
├── train.py                  # ▶ makes embeddings:  graph → walks → Word2Vec → .emb
├── plot_emb.py               # draws embeddings as a 2D picture (hubs vs leaves)
│
├── make_labels.py            # downloads + builds label files (cora)
├── prepare_linkpred.py       # builds the 70/30 edge split
├── eval_nodeclass.py         # scores node classification (weighted F1)
├── eval_linkpred.py          # scores link prediction (AUC)
│
├── notebooks/
│   └── reproduce_i2v.ipynb   # ⭐ START HERE — click-through reproduction
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
2. Open `notebooks/reproduce_i2v.ipynb` (in VS Code, or run `jupyter lab`).
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

Results are saved to `results/NNN.<dataset>.<task>.csv` with a settings header.

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
