# ViRGo — Guide
*Onboarding guide. Read this once, top to bottom, and you can run and extend the project on your own.*

---

## 1. What is this project?

**ViRGo** (*Virtual Role-Graph Embedding for Structural Identity*) is a research project aiming at a published paper. It **extends** an existing method called **Identity2Vec (I2V)** by Oluigbo et al.

**Plain-English background.**
- A **graph** = dots joined by lines. Dots are **nodes** (e.g. web pages, papers, people); lines are **edges** (a link, a citation, a friendship).
- Two nodes can share a **role** even if far apart — both *hubs* (many links) or both *bridges* (connect two groups). This role is a node's **structural identity**.
- An **embedding** = a list of 64 numbers per node (its "fingerprint"). Nodes with similar roles should get similar fingerprints.
- **I2V** builds these fingerprints with a *guided random walk* + a word model (Word2Vec). ViRGo studies a better way to feed structure into them.

## 2. Goal, contributions, methodology

**Research question:** when is a graph's own structure enough for a GNN, and when does adding structural "role" edges/features help — and can we predict which case from the graph's properties? We build a **virtual graph** (links nodes by *structural role*, not real edges), compare it against the untouched graph, and characterize datasets to explain the difference. **Immediate target: the LoG conference; the thesis reuses it.**

**Honest headline (7 datasets, 12 cells, 11 usable):** 7 keep the original, 4 augment, 1 tie. **All four augment cells are link prediction; no node classification cell augments at all.** Two candidate rules survive the screen, both LP: augment when *adjusted homophily is below ~0.23*, or when *largest-component fraction is above ~0.96* (i.e. the graph is one connected piece). Both cuts are **intervals, not numbers** — the panel pins them only to (0.093, 0.361) and (0.918, 1.0) — and refitting with one dataset hidden gives 4/5 and 5/6 against 0.60 / 0.67 majority baselines. So this is a *study* — a "when to augment" guide — not "our graph always wins".

> Superseded, kept so it is not re-introduced: the old line "role graphs never help link prediction" held only on the core-4 + OGB panel, before `roman_empire` / `tolokers` / `questions` were added on 2026-07-27.

| # | Contribution | Status |
|---|---|---|
| 1 | **"When to augment" study** — compare graph variants under a fixed GNN across many datasets; characterize each dataset (homophily first) → predict when augmentation helps. | main work |
| Tech | **GNN encoder** — GraphSAGE over the virtual graph, then GIN — replacing I2V's walk + Word2Vec. | done (SAGE) / GIN next |
| Future | Learnable alpha (auto-blend original vs virtual; needs synthetic data) · embeddings as LLM graph summaries. | not yet |

**Stay purely structural:** ViRGo's features are graph-derived — degree, eigenvector centrality, Ψ, clustering — and drive both the virtual graph and the GNN input. Ablation D showed they are necessary (random features collapse performance). We never use external node attributes (OGB text embeddings, biological descriptions): they would confound the study, hiding whether a gain came from structural rewiring or from the attributes. The point is to isolate structural identity.

**Project phases:**
1. **Phase 1 — reproduce I2V. ✅ done** — cached I2V + cross-model baselines (used as-is, **not fine-tuned**); within ±0.05 of the paper.
2. **Phase 2 — build the virtual graphs. ✅ done** — five variants: psi (Poisson/KL), degree, centrality, original (control), hybrid.
3. **Phase 3 — GraphSAGE encoder. ✅ done** — design locked by ablations A–D (edge pairs, mean, 2 layers, all four features, K=10; tuned on enzymes).
4. **Phase 4 — the study + more data ← current** — (1) run the pipeline on small/medium **OGB** datasets (ogbn-arxiv, ogbl-collab, ogbl-ddi), structural only ✅; (2) build the characterization table and screen candidate rules ✅ — **two LP rules carried forward** (adjusted homophily, largest-component fraction); (3) **Module 3** — freeze those two, predict unseen datasets *before* training, then run ← next; (4) swap GraphSAGE → **GIN**.
5. **Future work** — learnable alpha + synthetic datasets; LLM graph summaries. Anomaly detection is set aside for now.

**Pipeline (the method):** `graph → structural signal (degree + eigenvector centrality, cached) → virtual graph (top-K role-similar nodes) → GraphSAGE encoder → 64-number embedding → evaluation`.
Out of scope: hyperbolic/non-Euclidean space (a second paper).







## 3. Repository map (what each file does)

> **Two code folders, one rule: `virgo/` is imported, `experiments/` is run. Everything else is data, docs, or results.**

| File / folder | Purpose |
|---|---|
| `virgo/identity2vec.py` | **CORE baseline** (frozen, never edit). The I2V `Graph` class + guided walk: uses node degree & eigenvector centrality → KL → Poisson Ψ to pick the next node. |
| `virgo/identity2vec_cached.py` | Same algorithm but caches the structural signals → **identical output, ~200× faster**. (Deliverable #1.) |
| `experiments/train.py` | The **run file**: read graph → make walks → Word2Vec (Skipgram) → save `.emb`. Flag `--cached` uses the fast path. |
| `virgo/data/make_labels.py` | Downloads & builds node **labels** (cora/citeseer from LINQS; webkb from the author's repo) and verifies they match our graph. |
| `virgo/data/prepare_linkpred.py` | Builds the link-prediction **edge split** (70/30) + fake "negative" pairs, leakage-free. |
| `virgo/eval/nodeclass.py` | Scores **node classification** → micro / macro / weighted F1. |
| `virgo/eval/linkpred.py` | Scores **link prediction** → AUC (cosine = headline; Hadamard + logreg = second column). |
| `experiments/plot_emb.py` | Draws an embedding as a 2-D picture (optional). |
| `virgo/config.py` | **Single source of truth**: dataset registry, hyperparameters, seed, split fractions. |
| `virgo/eval/runner.py` | Glue: `embed()`, `run_linkpred()`, `run_nodeclass()`, plus the 3-seed `run_*_repeated()`. |
| `virgo/eval/results_io.py` | Saves a run to `results/NNN.<date>.<dataset>.<task>.csv`. |
| `virgo/utils.py` | Helpers: set seed, load embedding, next run id. |
| `experiments/run_task.py` | Terminal CLI (alternative to the notebook). |
| `virgo/graph_io.py` | **THE graph policy** + the single `load_graph()` every stage reads through. |
| `virgo/virtual_graph.py` | Phase 2: builds the five virtual-graph variants (`psi`/`degree`/`centrality`/`original`/`hybrid`). |
| `virgo/encoders/` | Phase 3+: `base.py` (architecture-free training), `sage.py`, `gin.py`, `walk.py`, and the `ENCODERS` registry. |
| `experiments/train_encoder.py` | Trains ONE encoder over ONE virtual graph; `--arch` picks any registered encoder. |
| `experiments/run_core.py` / `run_ogb.py` | The two frozen sweep drivers that produce every scoreboard row. |
| `experiments/characterize.py` | Measures graph properties + encoder inputs, then relates them to the original-vs-augmented gap. |
| `third_party/struc2vec/` | Vendored baseline, used as published (never edited). |
| `notebooks/1-reproduce_i2v.ipynb` | ⭐ **Primary entry point** — click-through reproduction (see §5). |
| `input/` | Original graphs (`.edgelist`). **Never edit.** |
| `output/` | Trained embeddings (`.emb`). |
| `labels/` | Node categories for classification. |
| `splits/` | Edge splits for link prediction. |
| `results/` | Score sheets (`.csv`) + plots (`.png`). Phase-2 subfolders: `vir_graph_stats/` (per-graph health table) + `vir_graph_variants/` (variant task-score comparison CSVs). |
| `docs/` | The paper PDF, proposal, `notes.md` (lab notebook), **this guide**. |
| `CLAUDE.md` | The project's governing rules (read it). |

## 4. Setup (one time)

1. The project uses the **conda environment `i2v`** (Python 3.12). Activate it:
   `conda activate i2v` — it already has numpy 1.26.4, networkx, gensim 4.3.3, scipy 1.12.0, scikit-learn, pandas, jupyter.
2. Open the notebook in VS Code (or `jupyter lab`) and pick the kernel **"Python (i2v)"**.
3. **Internet** is needed the first time you build cora/citeseer labels (a download).
4. A harmless `libtinfo.so` warning may appear in the terminal — ignore it.

## 5. How to run (the notebook = single entry point)

Open `notebooks/1-reproduce_i2v.ipynb`, run cells **top to bottom (Shift+Enter)**. You edit **one cell** (Step 1).

- **Step 0** — set up (imports, paths).
- **Step 1** — set `DATASET` and the seed list (e.g. `[42, 43, 44]`). ⬅️ the only cell you edit; every cell below follows it.
- **Step 2** — **node classification**: train (if missing) one I2V embedding per seed → logistic regression → micro/macro/weighted F1.
- **Step 3** — **link prediction**: per seed make the 70/30 split → retrain on the 70% graph → Hadamard + logreg → AUC.
- **Step 4** — **results + summary**: mean ± std over the seeds, saved to `results/`.
- **Step 5** — **cross-model benchmark**: I2V vs DeepWalk / node2vec / struc2vec on the benchmark datasets → tables in `results/`.

**CLI alternative (terminal):**
```bash
python experiments/run_task.py --list                              # show datasets
python experiments/run_task.py --task nodeclass --dataset cora
python experiments/run_task.py --task linkpred  --dataset cora --retrain
```

## 6. Datasets

Registered in `virgo/config.py` (`DATASETS`). The **four study datasets** (node classification + link prediction) are **cora** (7 classes, citation), **citeseer_linqs** (6 classes, citation), **enzymes** (3 classes, molecular), **proteins** (3 classes, molecular). Next we add small-to-medium **OGB** datasets — **ogbn-arxiv** (node property), **ogbl-collab** and **ogbl-ddi** (link property) — using their graphs only, never their built-in text features. Skip the huge 100M-node OGB graphs (too slow before the deadline). On each OGB dataset the original and virtual graphs receive the same structural features, so only the edges differ - a fair, single-variable comparison; we ignore the extra attributes (text embeddings, product descriptions, biological annotations) and do not compare against the OGB leaderboard, whose top models may use them. The goal is structural analysis, not leaderboard ranking.

Notes: the author **citeseer** graph has no aligned labels, so the study uses **`citeseer_linqs`** (the Phase-1 benchmark used author `citeseer` for link prediction only). **politics** ships no labels (link-pred only). Proteins' 3 classes are imbalanced, so report macro F1 alongside weighted.

## 7. Evaluation — splits, settings, metrics

> **The most important idea: the two tasks split differently.**

**Node classification — *transductive*.** Build the embedding on the **whole** graph once, then split the **nodes** 70/30 (`train_frac=0.7`, stratified, keeps class ratios), train a one-vs-rest logistic-regression classifier (`OneVsRestClassifier(LogisticRegression(solver="lbfgs", max_iter=300))`) on 70%, score the held-out 30%. The test *labels* are hidden → no leakage. **Metrics: micro / macro / weighted F1.**

**Link prediction — *inductive, leakage-free*.** Split the **edges** 70/30. A *spanning tree* stays in train so the graph stays connected; equal numbers of fake "negative" (non-edge) pairs are added. **Retrain a fresh embedding on the 70% train graph only** (so test edges are never seen). Score AUC two ways: **unsupervised cosine similarity** of the two node vectors — the **headline** metric, paper-faithful, no classifier — and a supervised **Hadamard-feature logistic regression** (node2vec protocol), kept as a second column. **Metric: AUC (cosine headline).**

**Fixed settings:** `seed = 42` (benchmark sweeps seeds `42, 43, 44`); embedding `dimensions = 64, num_walks = 10, window = 10, epochs = 1`; `walk_length` = **40** everywhere (train.py, benchmark_config, notebook) — the repo default (the paper's 80 is ≈1.87× slower, kept as a recorded deviation, see `docs/notes.md`). Classifier = one-vs-rest logistic regression (L-BFGS, 300 iters, L2 — the paper's protocol). No separate validation set (we don't tune).

**Caveat (research rigor):** the pipeline reports **mean ± std over 3 seeds** (42/43/44) with macro-F1 included. Small datasets stay noisy — widen to **≥5 seeds** for the final paper.

## 8. Results & outputs

| Where | What |
|---|---|
| `output/{ds}/{ds}_nc_orig_s{seed}.emb` (I2V) · `{model}_{ds}_nc_orig_s{seed}.emb` (baselines) | full-graph embedding per model/seed (node classification) |
| `output/{ds}/..._lp_orig_s{seed}.emb` | train-only embedding per model/seed (link prediction) |
| `splits/{ds}_train.edgelist`, `_train_neg.txt`, `_test_pos.txt`, `_test_neg.txt` | link-pred split |
| `results/NNN.DD.MM.{ds}.{task}.csv` | one run: a `#META` JSON header (all settings) + metric rows; auto-numbered |
| `results/table*.csv` | cross-model benchmark tables (Step 5); LP headline = cosine AUC, logreg AUC kept in `benchmark_per_seed.csv` |

**Reproduced so far:** Cora I2V lands in the paper's range on both tasks — the live numbers are whatever the latest 3-seed run wrote to `results/` (report **mean ± std**, not a single value). webkb_wisc node-class F1 ≈ random (structure ≠ content topics — see §10); webkb is **not** a paper benchmark. Cache fix: **207× faster, byte-identical** (verified, Deliverable #1). NOTE: all `.emb` were just cleared for the walk-length-40 revert — re-run the notebook to regenerate before quoting any metric.

## 9. Reproducibility rules (non-negotiable)

- **`seed = 42` everywhere** (split, init, sampling). Keep `--workers 1` (gensim is otherwise non-deterministic).
- **Never modify anything in `input/`** — write derived files alongside or into `output/`.
- A result counts as **"reproduced" only within ±0.05** of the paper.
- **Log every run and decision in `docs/notes.md`** (the lab notebook).

## 10. Coding rules & where to change things

**Coding style (match I2V):** `argparse` CLI; `build_graph()` / `learn_embeddings()` / `main(args)` shape; **fewest functions**, each short with a one-line docstring; self-explanatory file names; models expose **`train(epochs)`** not `fit()`; add no dependency without need.

**✅ Safe modification zones**
- Pick dataset/task → notebook **Step 1** (`DATASET`), then run the node-class or link-pred cells.
- Tune hyperparameters → notebook **Step 1** vars, or `virgo/config.py` (`I2V_PARAMS`) for the CLI.
- Add a dataset → drop the `.edgelist` in `input/`, add an entry to `DATASETS`, add labels in `labels/`.
- Add a task → new `virgo/eval/<task>.py` + register it in `virgo/eval/runner.py` (`TASKS`).
- Add an encoder → new `virgo/encoders/<name>.py` subclassing `GNNEncoder` with one `build_convs()`, then one line in `ENCODERS` (`virgo/encoders/__init__.py`). No driver changes.

**⛔ Do NOT touch**
- `virgo/identity2vec.py` (frozen baseline) and don't move it or `experiments/train.py`.
- Files in `input/`. The `seed`. The location of `CLAUDE.md` (root).

## 11. Known issues / gotchas

- **Notebook clobber:** if the `.ipynb` is open in an editor while a script edits it, the editor can overwrite changes. Close it before bulk edits; reload from disk after.
- **citeseer labels** may be rejected by the overlap safety-check (derived edgelist) → node classification can stop. Link prediction still works.
- **webkb structure ≠ content:** I2V learns *structural roles*; webkb classes are *content topics* and the graph is heterophilous, so node-class F1 ≈ random. Expected, not a bug. Use webkb_wisc for *link prediction*.
- **Single-split variance** on small graphs (§7 caveat).
- The original I2V recomputes centrality inside the walk loop (very slow) — always use `--cached`.

## 12. Deliverables & how to continue

| Deliverable | State | Next action |
|---|---|---|
| 1. Cached I2V (identical + faster) | ✅ done | — |
| —. I2V reproduction + cross-model baselines | ✅ done | used as-is, **not fine-tuned** |
| 2. `virgo/virtual_graph.py` (five-variant builder) | ✅ done | — |
| 3. GraphSAGE encoder over the virtual graph | ✅ done | GIN next (swap in after SAGE) |
| 4. Node-class + link-pred eval on 7 datasets | ✅ done | numbers in `results/scoreboard.csv` |
| 5. Characterization table + "when to augment" rule | ✅ done | 7 properties screened per task; **2 LP candidate rules final**, each with its separating interval and a hidden-dataset refit; `results/candidate_rules.csv`, `results/nested_loo.csv` |
| 5b. **Module 3 — pre-registered validation** | 🔵 **current** | freeze the 2 rules, predict unseen datasets before training, then run |
| 6. GIN results | 🔵 next | after the GraphSAGE + OGB runs |
| 7. LoG paper draft | 🔵 | thesis reuses it |

**Where to start as the new intern:** (1) open `notebooks/3-phase3_gnn_encoder.ipynb` and read its result tables (§7 encoder comparison, §8 variant sweep, §10 research-question view) against `results/scoreboard.csv`; (2) read `virgo/virtual_graph.py` and `virgo/encoders/base.py` + `sage.py` to see how a virtual graph and its GraphSAGE embedding are built; (3) skim `docs/paper_log.md` for the findings and `docs/notes.md` for history; (4) then start **Phase 4** — add a small OGB dataset through the `DATASETS` registry (graph only, no text features), run the pipeline, and help build the characterization table.

## 13. Mini-glossary

**Node/edge** — dot / line. **Embedding** — 64-number fingerprint per node. **Structural identity** — a node's role (hub, bridge), independent of position. **Transductive** — embed the whole graph, hide only labels. **Inductive** — retrain on a sub-graph so test items are unseen. **Weighted F1** — accuracy-like score that accounts for class sizes. **AUC** — probability the model ranks a real edge above a fake one. **Leakage** — letting test information into training (forbidden).


Command:
python docs/build_pdf.py docs/virgo_guide.md docs/virgo_guide.pdf