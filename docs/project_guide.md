# Structure Aware Graph Augmentation for Graph Neural Networks — Guide

*Onboarding guide. Read this once, top to bottom, and you can run and extend the project on your own.*

> Renamed 2026-08-15 (was *ViRGo — Virtual Role-Graph Embedding for Structural Identity*). The Python package directory is still `virgo/`; that is an internal name only.

---

## 1. What is this project?

A research project aiming at a published paper. It **extends** an existing method, **Identity2Vec (I2V)** by Oluigbo et al.

**Plain-English background.**
- A **graph** = dots joined by lines. Dots are **nodes** (web pages, papers, people); lines are **edges** (a link, a citation, a friendship).
- Two nodes can share a **role** even if far apart — both *hubs* (many links), both *bridges* (join two groups). That role is a node's **structural identity**.
- A **role graph** (built here, sometimes called a *virtual graph* in older code and comments) adds edges between role-twins so a GNN can pass messages between them.
- An **embedding** = 64 numbers per node — its fingerprint. Nodes with similar roles should get similar fingerprints.

**The research question.** Adding role edges is *structural augmentation*. It helps on some graphs and hurts on others. So: **for a given graph, should you augment at all — and if so, with which structural signal?** We answer both **before training**, from properties of the graph itself.

**What we measured** (GraphSAGE, K=10, 10 seeds, five locked variants):

| task | augment | tie | keep original |
|---|---|---|---|
| link prediction | **9** | 4 | 6 (19 datasets) |
| node classification | **0** | 2 | 13 (15 cells) |

Node classification never benefits — that is a **boundary**, not a rule. Link prediction genuinely splits, so the decision is worth predicting.

---

## 2. The framework (what we are contributing)

Everything is computable **before an encoder is trained**. It lives frozen in `virgo/frozen_rules.py`.

**Stage 1 — augment, or keep the original?** (`predict_gated`)

1. **Adjusted homophily** — do connected nodes share a label more than chance? **High ⇒ keep the original.** This is a reliable *veto*. Low homophily is **necessary but not sufficient** for augmentation to help.
2. **The gate** — only inside the low-homophily zone. `original_retention` = how many of the real edges survive into the built role graph. **Low ⇒ augment.** Counter-intuitive and worth saying out loud: *the less of the original graph the rewiring keeps, the more augmentation helps.*
3. **Largest-component fraction** — the fallback when the graph has no labels at all, so homophily cannot be computed.

Why a second step exists: `minesweeper` (homophily 0.0094) keeps the original and `squirrel_filtered` (0.0086) augments. They are 0.0008 apart with opposite answers, so **no single-variable rule can ever separate them.**

**Stage 2 — which structural signal?** (`predict_strategy`) — consulted only when stage 1 says augment.

- **Adjusted neighbour-label predictability** — even when neighbours carry *different* labels, do they carry a *consistent mix* that reveals a node's class? **High ⇒ use eigenvector centrality.**
- Below the cut it answers `"psi or degree (undetermined)"`. Separating Ψ from degree is an open problem; **the current scope is centrality only.**

**The cuts** — always quote the interval, never the 4-decimal number:

| stage | property | cut | any cut in this range fits equally well |
|---|---|---|---|
| 1 | `homophily_adjusted` | < 0.227 | (0.0926, 0.3613) |
| 1 | `original_retention` | < 0.0119 | (0.0062, 0.0176) |
| 1 | `largest_component_frac` | > 0.9588 | (0.9177, 1.0) |
| 2 | `nbr_predictability_adjusted` | > 0.0092 | (0.006, 0.0123) |

---

## 3. What is proven and what is not

This is the part to get right in any write-up.

| component | fitted on | tested on unseen graphs | result |
|---|---|---|---|
| adjusted homophily | 7 datasets | 9, pre-registered | **4/6** decided cells |
| largest-component fraction | 7 datasets | same 9 | **4/6** |
| the gate | 13 datasets (*includes those 9*) | 4 LINKX graphs | **2/3** |
| stage-2 centrality rule | 14 datasets, nothing held out | 3 LINKX graphs | **3/3** top-scoring |

**Stage 1's held-out record.** Adjusted homophily said *augment* on all four LINKX graphs; the gate overruled it on `reed98` and called *keep original*. `reed98` then measured **augment at 2.55σ** — the biggest gain of the four. So the combined call scored 2/3 and **homophily alone would have scored 3/3**. The gate's one differentiating call was its one error. It stays in the framework because it is the only component that addresses the decisive pair and the only one that needs no labels, but it is reported as **screened and fitted, not yet transferring**.

**Stage 2's held-out record.** All three graphs stage 1 routed to *augment* were predicted **centrality** (0.2128 / 0.2278 / 0.2760 against the 0.0092 cut), and centrality was the top-scoring variant on all three, above the original in each case:

| dataset | original | centrality | runner-up | note |
|---|---|---|---|---|
| amherst41 | 0.6651 | 0.6749 | hybrid-centrality 0.6718 | top two both add centrality, so the order does not change the signal call |
| johnshopkins55 | 0.6854 | 0.7160 | hybrid-centrality 0.7021 | clear margin |
| cornell5 | 0.6910 | 0.7078 | hybrid 0.7050 | centrality slightly ahead |

The limitation to state with it: the cut is low enough that every ordinary labelled graph clears it, so all three calls were *centrality* — a constant "always centrality" predictor would score the same here. Confirming the negative side needs an augmenting graph **below** the cut.

**Two habits that keep the work honest.**
- **A threshold is an interval.** The search returns the midpoint between the two datasets straddling the boundary, so "0 in-sample exceptions" is *guaranteed* whenever the data separates — it measures separability, not the rule.
- **Say the seed count.** `results/scoreboard.csv` is the 10-seed board; `results/scoreboard_3seed.csv` is the archived 3-seed one. Module 3's published 5/7 and 4/7 are 3-seed figures; at 10 seeds they are 4/6 and 4/6 because `lastfm_asia` moved from *keep* to *tie*.

---

## 4. Repository map

> **Two code folders, one rule: `virgo/` is imported, `experiments/` is run. Everything else is data, docs, or results.**

| File / folder | Purpose |
|---|---|
| `virgo/frozen_rules.py` | **THE frozen artifacts**: the panels, both stage-1 rules, the gate, the stage-2 rule, and `predict_gated` / `predict_strategy`. |
| `virgo/config.py` | **Single source of truth**: paths, dataset registry, the seven variants, hyperparameters, seed. |
| `virgo/graph_io.py` | **THE graph policy** + the single `load_graph()` every stage reads through. |
| `virgo/identity2vec.py` | **CORE baseline** (frozen, never edit). I2V `Graph` + guided walk: degree & eigenvector centrality → KL → Poisson Ψ. |
| `virgo/identity2vec_cached.py` | Same algorithm, signals cached → **identical output, ~200× faster**. |
| `virgo/virtual_graph.py` | Builds the role graph — all seven variants. |
| `virgo/encoders/` | `base.py` (architecture-free training), `sage.py`, `gin.py`, `walk.py`, and the `ENCODERS` registry. |
| `virgo/data/` | `make_labels` · `make_ogb` · `make_hetero` · `make_pyg` · `prepare_linkpred`. |
| `virgo/eval/` | `nodeclass` (F1) · `linkpred` (AUC, leakage-free) · `ogb` (official) · `runner` · `results_io`. |
| `experiments/run_core.py` / `run_ogb.py` | The two frozen sweep drivers that produce every scoreboard row. |
| `experiments/characterize.py` | **FITS**: graph properties → the gap → candidate stage-1 rules. Panel-guarded. |
| `experiments/gate_rules.py` | **FITS**: role-graph properties → the gate. Panel-guarded. |
| `experiments/strategy_select.py` | **FITS**: which signal wins, and whether properties predict it. Panel-guarded. |
| `experiments/predict_module3.py` / `score_module3.py` | **READ-ONLY**: pre-register stage-1 calls before training, score them after. |
| `experiments/predict_gate.py` | **READ-ONLY**: build role graph → pre-register → train → score (the gate). |
| `experiments/predict_strategy.py` | **READ-ONLY**: pre-register and score the stage-2 call. |
| `experiments/train.py` · `train_encoder.py` · `benchmark_baselines.py` · `run_task.py` · `plot_emb.py` | Individual pieces. |
| `third_party/struc2vec/` | Vendored baseline, used as published (never edited). |
| `input/` · `labels/` · `splits/` · `output/` · `results/` | Data in, labels, edge splits, embeddings out, scores. **Never edit `input/`.** |
| `docs/` | `paper_log.md` (research log) · `notes.md` (lab notebook) · this guide · the I2V paper. |
| `CLAUDE.md` | The project's governing rules (read it). |

**The fit/predict split is load-bearing.** A prediction script must never import a fitting function, or the validation becomes circular. That is enforced by panel assertions, not by convention.

---

## 5. Setup (one time)

1. Conda environment **`i2v`** (Python 3.12): `conda activate i2v` — has numpy 1.26.4, networkx, gensim 4.3.3, scipy 1.12.0, scikit-learn, pandas, torch, torch-geometric, jupyter.
2. Open a notebook in VS Code (or `jupyter lab`) and pick the kernel **"Python (i2v)"**.
3. **Internet** is needed the first time a dataset's labels are built (a download).
4. A harmless `libtinfo.so` warning may appear in the terminal — ignore it.

---

## 6. How to run

### Apply the framework to a brand-new graph

Order matters, and it is what makes the test honest: **build → predict → train → score.** Building the role graph is deterministic and involves no learning, so the prediction is still made with the outcome unknown.

```bash
# 1 · build the role graph (no training happens here)
python -m virgo.virtual_graph --input input/<ds>.edgelist --sim psi --k 10

# 2 · freeze stage 1 BEFORE any encoder runs
python experiments/predict_gate.py --datasets <ds> --step predict

# 3 · freeze stage 2, only if stage 1 said "augment"
python experiments/predict_strategy.py --datasets <ds> --step predict

# 4 · now train
python experiments/run_core.py --datasets <ds> --task link_prediction

# 5 · score the frozen calls
python experiments/predict_gate.py --datasets <ds> --step score
python experiments/predict_strategy.py --datasets <ds> --step score
```

Both predict scripts refuse a dataset the rule was fitted on — predicting one measures fit, not transfer.

### Refit the analyses

```bash
python experiments/characterize.py --step all   # the stage-1 rule screen
python experiments/gate_rules.py                # the gate screen
python experiments/strategy_select.py           # the stage-2 screen
```

### Individual pieces

```bash
python experiments/run_task.py --list
python experiments/run_task.py --task nodeclass --dataset cora
python experiments/run_task.py --task linkpred  --dataset cora --retrain
python experiments/train_encoder.py --input input/cora.edgelist --sim psi --k 10 --seed 42
python experiments/train.py --input input/cora.edgelist --output output/cora_mine.emb --cached --seed 42
```

### The notebooks

Run top to bottom; each reuses files already on disk.

| notebook | what it shows |
|---|---|
| `1-reproduce_i2v.ipynb` | reproduce the Identity2Vec paper |
| `2-phase_2_virtual_graph.ipynb` | build and inspect the role graphs |
| `3-phase3_gnn_encoder.ipynb` | train GraphSAGE; encoder comparison, variant sweep, feature ablation |
| `4-phase4_ogb.ipynb` | OGB datasets under the official protocol |
| `5-phase5_characterization.ipynb` | properties vs the gap; how the stage-1 rules were found |
| `6-phase6_module3_validation.ipynb` | the pre-registered stage-1 test |
| `7-phase7_strategy_selection.ipynb` | the stage-2 rule and the two-stage held-out test |

---

## 7. Datasets

20 registered in `virgo/config.py` (`DATASETS`), spanning citation, molecular, co-purchase, crowdsourcing, linguistic, social and drug-interaction domains; adjusted homophily from −0.047 to 0.856; average degree from 2.8 to 88.3.

| group | datasets |
|---|---|
| discovery panel (stage-1 rules fitted here) | cora, enzymes, ogbn_arxiv, ogbl_ddi, roman_empire, tolokers, questions |
| Module-3 held-out | citeseer_linqs, proteins, pubmed, actor, minesweeper, amazon_photo, lastfm_asia, amazon_ratings, squirrel_filtered |
| **stage-2 validation datasets** | reed98, amherst41, johnshopkins55, cornell5 (LINKX Facebook100) · chameleon_filtered, texas, twitch_pt (earlier batch, withdrawn 2026-08-14 — scores kept in `results/module7_withdrawn.csv`) |

Only `amherst41`, `johnshopkins55` and `cornell5` actually reached stage 2 with a pre-registered call. `reed98` was stopped at stage 1 (wrongly — see §3). The earlier batch ran before stage 1 existed, so its stage-2 verdict is retrospective; of those three, `chameleon_filtered` and `texas` kept the original graph and `twitch_pt` augmented but `hybrid_degree` won, making it a stage-2 miss. It is listed rather than dropped on purpose.

A dataset lives in **exactly one** panel — asserted in `frozen_rules.py`.

Gotchas worth knowing:
- The author's **citeseer** graph has no aligned labels → link prediction only; the study uses **`citeseer_linqs`**.
- **politics** ships no labels (link prediction only).
- **ogbl_ddi** is unlabelled — homophily cannot be computed for it, which is exactly when the stage-1 fallback rule is needed.
- **Facebook100** labels are **gender, missing for ~10% of users** (LINKX codes it −1). Those nodes are left out of the `.labels` file rather than written as a third class; the graph keeps every node.
- **OGB text features are never loaded**, on purpose (see §9).

---

## 8. Evaluation — splits, settings, metrics

> **The most important idea: the two tasks split differently.**

**Node classification — *transductive*.** Build the embedding on the whole graph once, then split the **nodes** 70/30 (stratified), train one-vs-rest logistic regression on 70%, score the held-out 30%. Test *labels* are hidden → no leakage. **Metric: weighted F1** (macro reported alongside for imbalanced graphs).

**Link prediction — *inductive, leakage-free*.** Split the **edges** 70/30, keeping a spanning tree in train so the graph stays connected, with equal numbers of negative (non-edge) pairs sampled inside a component. **Retrain a fresh embedding on the 70% train graph only**, and rebuild the role graph from those edges too, so no test edge is ever seen. **Metric: AUC** (unsupervised cosine, paper-faithful).

**OGB datasets** use their official split and official Evaluator instead (Accuracy for `ogbn-arxiv`, Hits@20 for `ogbl-ddi`), with a trained MLP decoder for `ogbl-ddi`. The winner is always decided *inside* one dataset, so a differing scorer cancels out.

**Fixed settings.** `seed = 42`; embedding `dimensions = 64`, `num_walks = 10`, `window = 10`; `walk_length = 40` (repo default; the paper's 80 is a recorded deviation in `docs/notes.md`); GraphSAGE 2 layers, mean aggregation, edge positives, 50 epochs, K = 10. Frozen-rule fitting used seeds 42/43/44; the strategy work and every held-out test use 42–51.

**How a verdict is decided.** A gap counts as *augment* only if it clears the pooled seed noise (1σ); inside that band it is a **tie**, which is a no-decision, not a failure. For picking a *winning variant* the band is the standard error of the difference of the two means (`sem`) — the only band more seeds can narrow.

---

## 9. Purely structural — and why

Every feature is graph-derived: degree, eigenvector centrality, Ψ, clustering. They do double duty — they define the role graph *and* feed the encoder.

**External node attributes are never used.** Not OGB text embeddings, not fastText vectors, not bag-of-words, not biological annotations. This is not laziness: attributes would confound the study, because any gain could then be credited to them rather than to the rewiring under test. For the same reason there is no comparison against top OGB leaderboard entries. The goal is structural analysis, not leaderboard ranking.

An ablation shows the features are necessary rather than decorative — replacing them with random features drops performance to the DeepWalk baseline or below.

Out of scope: hyperbolic / non-Euclidean latent space (a second paper); anomaly detection (set aside).

---

## 10. Reproducibility rules (non-negotiable)

- **`seed = 42` everywhere** (split, init, sampling). Keep `--workers 1` (gensim is otherwise non-deterministic).
- **Name the seed count on every number**, or re-score first.
- **Never modify anything in `input/`** — write derived files alongside, or into `output/`.
- A result counts as **reproduced only within ±0.05** of the paper.
- **Predictions are written to disk before the encoder runs.** Never refit after seeing an outcome.
- **Log every run and decision** — research findings in `docs/paper_log.md`, day-to-day changes in `docs/notes.md`.

---

## 11. Coding rules & where to change things

**Style (match I2V):** `argparse` CLI; `build_graph()` / `learn_embeddings()` / `main(args)` shape; **fewest functions**, each short with a one-line docstring; self-explanatory file names; models expose **`train(epochs)`** not `fit()`; add no dependency without need.

**✅ Safe modification zones**
- Tune hyperparameters → `virgo/config.py` (`I2V_PARAMS`, `GNN_PARAMS`).
- Add a dataset → drop the `.edgelist` in `input/`, add an entry to `cfg.DATASETS`, labels in `labels/`, then register it in `characterize.STUDY`, `run_core.RUNNABLE`, and **exactly one** panel in `frozen_rules.py`.
- Add a task → new `virgo/eval/<task>.py` + register it in `virgo/eval/runner.py` (`TASKS`).
- Add an encoder → new `virgo/encoders/<name>.py` subclassing `GNNEncoder` with one `build_convs()`, then one line in `ENCODERS`. No driver changes.

**⛔ Do NOT touch**
- `virgo/identity2vec.py` (frozen baseline), or move it or `experiments/train.py`.
- Files in `input/`. The `seed`. The location of `CLAUDE.md` (root).
- Anything in `virgo/frozen_rules.py` — the numbers there are the record of what was frozen *before* a test, and rewriting one retroactively invalidates the test it was used for.

---

## 12. Known issues / gotchas

- **Notebook clobber:** if an `.ipynb` is open in an editor while a script edits it, the editor can overwrite the change. Close it before bulk edits; reload from disk after.
- **Ψ is numerically unstable where Ω underflows** — ARPACK's random restart perturbs Ω by ~7e-12, which `log(p/Ω)` can amplify. Two `psi` builds on `proteins` share only ~34% of their edges, though the downstream metrics move by ~0.0005. The feature cache contains it.
- **Metric-reproducible, not bit-reproducible.** Repeating a run reproduces the metric exactly; embeddings agree to ~2e-6 (float32 aggregation order). Verify stored embeddings by *reproduction*, never by file dates.
- **Single-split variance** on small graphs — hence 10 seeds for anything decisive.
- The original I2V recomputes centrality inside the walk loop (very slow) — always use `--cached`.

---

## 13. Where to start as a newcomer

1. Read **§2 and §3 above** — the framework and, more importantly, exactly how far it has been validated.
2. Open `virgo/frozen_rules.py`. It is short, and it is the whole contribution in one file.
3. Open `notebooks/7-phase7_strategy_selection.ipynb` — the two-stage held-out test end to end.
4. Read `virgo/virtual_graph.py` and `virgo/encoders/base.py` + `sage.py` to see how a role graph and its embedding are built.
5. Skim `docs/paper_log.md` for the findings, including the negative ones and the method fixes that invalidated earlier numbers.
6. Then pick up the open work: a stage-2 test **below** the cut, Ψ-vs-degree (currently blocked for lack of datasets in that zone), or GIN alongside GraphSAGE.

---

## 14. Mini-glossary

**Node / edge** — dot / line. **Embedding** — 64-number fingerprint per node. **Structural identity** — a node's role (hub, bridge), independent of position. **Role graph** — the augmented graph linking role-twins. **Adjusted homophily** — do neighbours share a label, corrected for class count and balance so graphs are comparable. **Neighbour-label predictability** — can a node's class be read off the *mix* of labels around it, even when they differ. **Transductive** — embed the whole graph, hide only labels. **Inductive** — retrain on a sub-graph so test items are unseen. **Weighted F1** — accuracy-like score accounting for class sizes. **AUC** — probability the model ranks a real edge above a fake one. **Leakage** — letting test information into training (forbidden). **Pre-registered** — the prediction was written to disk before the experiment ran.

---

Build the PDF:

```bash
python docs/build_pdf.py docs/project_guide.md docs/project_guide.pdf
```
