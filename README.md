# Structure Aware Graph Augmentation for Graph Neural Networks

*A characterization-guided, two-stage decision framework for structural role-graph augmentation.*

Structural-role augmentation adds edges between nodes that occupy **similar topological positions** — two hubs, two bridges, two peripheral nodes — even when no path connects them. A standard GNN can only pass messages along real edges, so role-twins never interact; a **role graph** gives them a channel.

The problem is that this does not always help. Measured across 19 graphs, role augmentation improves link prediction on some and actively damages others, and it never once improves node classification. So the useful question is not *"does rewiring work?"* but **"for this graph, should I rewire at all — and if so, with which structural signal?"**

This project answers both, **before any encoder is trained**, with a two-stage decision framework read off the graph's own properties.

Target venue: the **Learning on Graphs (LoG)** conference. The thesis draws on the same work.

---

## 1 · The headline result

GraphSAGE, K = 10, 10 seeds (42–51), five locked graph variants, one frozen pipeline.

| task | augment | tie | keep original | cells |
|---|---|---|---|---|
| **link prediction** | **9** | 4 | 6 | 19 datasets |
| **node classification** | **0** | 2 | 13 | 15 cells |

Two things follow, and they are the reason the framework exists:

1. **Node classification has a boundary, not a rule.** No dataset, in any panel, at any seed count, has ever shown a significant node-classification gain from role augmentation. The reportable finding is *"never augment for NC"*.
2. **Link prediction genuinely splits.** Roughly half the graphs benefit and half are hurt, so the decision is worth making — and worth predicting.

---

## 2 · The two-stage framework

Everything below is computable **before training an encoder**. Stage 1 needs the role graph *built* (a deterministic top-K construction, no learning); stage 2 needs only the original graph and its labels.

```
                     ┌─────────────────────────────────────────┐
   graph  ─────────► │ STAGE 1 — augment, or keep the original? │
                     └─────────────────────────────────────────┘
                                │                    │
                      keep original                augment
                                │                    ▼
                                │      ┌──────────────────────────────┐
                                │      │ STAGE 2 — which signal?      │
                                │      └──────────────────────────────┘
                                │                    │
                                ▼                    ▼
                     train on the original    build the centrality
                     graph, unchanged         role graph, then train
```

### Stage 1 — whether to augment

Three components, applied in order (`virgo.frozen_rules.predict_gated`):

| step | rule | cut | any cut in | needs labels |
|---|---|---|---|---|
| 1 | **adjusted homophily** — high ⇒ keep the original | `< 0.227` ⇒ augment | (0.0926, 0.3613) | yes |
| 2 | **the gate**, inside the low-homophily zone only — `original_retention`, the fraction of real edges the role graph preserves | `< 0.0119` ⇒ augment | (0.0062, 0.0176) | no |
| 3 | **largest-component fraction** — fallback for unlabelled graphs | `> 0.9588` ⇒ augment | (0.9177, 1.0) | no |

Adjusted homophily is **a veto, not a predictor**. High adjusted homophily reliably means *keep the original*; low adjusted homophily is **necessary but not sufficient** for augmentation to help. The decisive evidence is `minesweeper` (0.0094, keeps) against `squirrel_filtered` (0.0086, augments) — 0.0008 apart with opposite outcomes, so **no single-variable split can separate them**. The gate was introduced to break exactly that ambiguity; see §3 for how well it does.

### Stage 2 — which structural signal

Consulted **only** when stage 1 says augment, because "not centrality" would otherwise silently include "do not augment at all", which is stage 1's question.

| rule | cut | any cut in | fitted on |
|---|---|---|---|
| **adjusted neighbour-label predictability** high ⇒ use **eigenvector centrality** | `> 0.0092` | (0.006, 0.0123) | 14 datasets, 7 of them augmenting |

Read plainly: **when a node's class can be read off the mix of labels around it, centrality-based rewiring is the right augmentation.** Below the cut the rule returns `"psi or degree (undetermined)"` — it separates centrality from the rest and deliberately claims nothing about Ψ versus degree.

---

## 3 · What is validated, and what is not

The distinction is load-bearing and is kept in the code: `characterize.py` and `gate_rules.py` **fit** and are panel-guarded; `predict_module3.py`, `predict_gate.py` and `predict_strategy.py` only ever **read** `virgo/frozen_rules.py`. A prediction script that imported a fitting function would make the validation circular.

| component | fitted on | held-out test | result |
|---|---|---|---|
| **adjusted homophily** (stage 1) | 7 datasets, 12 cells | 9 unseen datasets, pre-registered | **4/6** decided cells |
| **largest-component fraction** (stage 1 fallback) | same 7 | same 9 | **4/6** decided cells |
| **the gate** (stage 1) | 13 datasets — *includes the 9 above, so it is fitted, not validated* | 4 unseen LINKX graphs | **2/3** decided cells |
| **centrality rule** (stage 2) | 14 datasets, 10 seeds — nothing held out | 3 unseen LINKX graphs | **3/3** — centrality scored highest on every one |

### Stage 2's held-out record, stated exactly

All three graphs stage 1 routed to *augment* were predicted **centrality** (0.2128, 0.2278, 0.2760 against a 0.0092 cut). All three came back with centrality as the highest-scoring variant, above the original graph in every case.

| dataset | original | centrality | runner-up | note |
|---|---|---|---|---|
| amherst41 | 0.6651 | **0.6749** | hybrid-centrality 0.6718 | top two both add centrality, so the signal call does not depend on their order |
| johnshopkins55 | 0.6854 | **0.7160** | hybrid-centrality 0.7021 | clear margin |
| cornell5 | 0.6910 | **0.7078** | hybrid 0.7050 | centrality slightly ahead |

---

## 4 · Method

The variable under study is **the graph**, not the encoder. One fixed GNN sees every variant.

```
input graph → structural signals → role graph → encoder → embeddings → evaluation
              (degree, eigenvector  (top-K most   (GraphSAGE  (64-dim    (node classification F1,
               centrality, Ψ,        role-similar  or DeepWalk) per node)  link prediction AUC)
               clustering; cached)   nodes)
```

1. **Structural signals** — degree and eigenvector centrality computed once per graph and cached. Identity2Vec recomputes them inside its walk loop; caching returns identical output roughly 200× faster.
2. **Similarity** — KL divergence λ → Poisson Ψ, exactly as Identity2Vec defines it.
3. **Role graph** — `virgo/virtual_graph.py` links each node to its top-K most structurally similar nodes. K = 10 throughout (a locked hyperparameter: sparsity against over-smoothing).
4. **Encoder** — a 2-layer GraphSAGE with an unsupervised Skipgram-style loss (linked nodes attract, random nodes repel). DeepWalk on the same graph is the walk-based comparison.
5. **Evaluation** — node classification by logistic regression on the embeddings (weighted F1); link prediction by AUC over a 70/30 edge split, with the role graph **rebuilt from the 70% training edges** so no test edge leaks.

### The graph variants

Seven official constructions. The five marked *locked* are the ones every frozen rule was fitted against; the analysis scripts read that fixed list, so promoting a new variant can never silently refit a locked rule.

| variant | role signal | construction | locked |
|---|---|---|---|
| `original` | — | the input graph, unchanged (the control) | ✓ |
| `psi` | Identity2Vec Poisson/KL score Ψ | replaces the edges | ✓ |
| `degree` | degree | replaces the edges | ✓ |
| `centrality` | eigenvector centrality | replaces the edges | ✓ |
| `hybrid` | Ψ | original ∪ role edges | ✓ |
| `hybrid_degree` | degree | original ∪ role edges | |
| `hybrid_centrality` | eigenvector centrality | original ∪ role edges | |

The hybrids vote for the signal they **add**, so a stage-2 answer names a structural signal rather than one of seven files.

---

## 5 · Datasets

20 registered graphs spanning citation, molecular, co-purchase, crowdsourcing, linguistic, social, Wikipedia, web and drug-interaction domains, with adjusted homophily from −0.047 to 0.856 and average degree from 2.8 to 88.3, plus 3 that were measured and then withdrawn.

| group | datasets | role |
|---|---|---|
| discovery panel | cora, enzymes, ogbn_arxiv, ogbl_ddi, roman_empire, tolokers, questions | stage-1 rules fitted here |
| Module 3 held-out | citeseer_linqs, proteins, pubmed, actor, minesweeper, amazon_photo, lastfm_asia, amazon_ratings, squirrel_filtered | pre-registered test of the stage-1 rules |
| **stage-2 validation datasets** | reed98, amherst41, johnshopkins55, cornell5 (LINKX Facebook100) · chameleon_filtered, texas, twitch_pt (earlier batch) | the unseen set the two-stage framework was run end to end on |

**Not every one of those seven tested stage 2**, and the table should not be read as if they did:

| dataset | stage 1 | reached stage 2? |
|---|---|---|
| amherst41, johnshopkins55, cornell5 | augment | **yes** — the three pre-registered stage-2 tests |
| reed98 | keep (wrongly, see §3) | no — tested stage 1 only |
| chameleon_filtered, texas, twitch_pt | ran **before** stage 1 existed | no pre-registered call; their stage-2 verdict is retrospective |

The earlier batch was **withdrawn on 2026-08-14**: the graphs and their scoreboard rows were deleted, and the measured scores are preserved in `results/module7_withdrawn.csv`. `chameleon_filtered` (890 nodes) and `texas` (183 nodes) both kept the original graph; `twitch_pt` (1,912 nodes) augmented but `hybrid_degree` won, not centrality — so it is a stage-2 miss. It is listed here on purpose: reporting the two that were consistent with the rule while omitting the one that was not would be selective.

A registered dataset lives in exactly one panel, enforced by assertions in `virgo/frozen_rules.py`. The Facebook100 label is **gender, missing for ~10% of users** (LINKX codes it −1); those nodes are left out of the `.labels` file rather than written as a third class, and the graph keeps every node.

---

## 6 · Evidence trail

| module | question | code | tables |
|---|---|---|---|
| 1 | reproduce Identity2Vec | `virgo/identity2vec_cached.py` | `results/notebook1_reproduce_i2v/` |
| 2 | which properties predict the augment/keep gap | `experiments/characterize.py` | `characterization_*.csv`, `candidate_rules.csv` |
| 3 | do the stage-1 rules transfer to unseen graphs | `predict_module3.py` → `score_module3.py` | `module3_*.csv` |
| 4 | screen a second variable for the ambiguous zone | `experiments/gate_rules.py` | `gate_candidates.csv`, `vg_characterization.csv` |
| 5 | does the gate transfer | `experiments/predict_gate.py` | `module5_*.csv` |
| 6 | which structural signal to use | `experiments/strategy_select.py` | `strategy_winners.csv`, `strategy_patterns.csv` |
| 7 | does the signal rule transfer | `experiments/predict_strategy.py` | `module7_*.csv` |

`docs/paper_log.md` is the full research log, including the negative results and the method fixes that invalidated earlier numbers.

---

## 7 · Repository map

Two code folders, one rule: **`virgo/` is imported, `experiments/` is run.** Everything else is data, docs or results. (`virgo/` remains the Python package name — an internal identifier the project title no longer matches.)

```
├── input/                    # original graphs (.edgelist) — never edit
├── labels/                   # node labels for classification
├── splits/                   # saved 70/30 link-prediction splits (per dataset, per seed)
├── output/                   # everything generated: notebook1_* / notebook2_* / notebook3_*
├── results/                  # scoreboard.csv (master) · graph_health.csv · module*.csv · snapshots/
├── docs/                     # paper_log.md (research log) · notes.md (lab notebook) · project_guide.md
├── third_party/struc2vec/    # vendored baseline, used as published (never edited)
│
├── virgo/                    # ── the library: import only, never run directly ──
│   ├── config.py             # THE settings: paths, dataset registry, variants, GNN params, seed=42
│   ├── frozen_rules.py       # THE frozen artifacts: panels, both stage-1 rules, the gate, the stage-2 rule
│   ├── graph_io.py           # THE graph policy + the single loader every stage reads through
│   ├── identity2vec.py       # original I2V walk algorithm (frozen baseline)
│   ├── identity2vec_cached.py# same algorithm with caching (identical output, ~200× faster)
│   ├── virtual_graph.py      # the top-K role-graph builder (all seven variants)
│   ├── encoders/             # base.GNNEncoder + sage · gin · walk, behind an ENCODERS registry
│   ├── data/                 # make_labels · make_ogb · make_hetero · make_pyg · prepare_linkpred
│   └── eval/                 # nodeclass (F1) · linkpred (AUC) · ogb (official) · runner · results_io
│
├── experiments/              # ── the entry points: each one an argparse CLI ──
│   ├── run_core.py           # the frozen sweep for the non-OGB datasets
│   ├── run_ogb.py            # the same sweep under the official OGB protocol
│   ├── characterize.py       # FITS: graph properties → the gap → candidate stage-1 rules
│   ├── gate_rules.py         # FITS: role-graph properties → the gate
│   ├── strategy_select.py    # FITS: which signal wins, and can properties predict it
│   ├── predict_module3.py    # READS frozen_rules: pre-registers stage-1 calls before training
│   ├── score_module3.py      # scores them afterwards
│   ├── predict_gate.py       # build role graph → pre-register the gate call → score
│   ├── predict_strategy.py   # pre-registers and scores the stage-2 call
│   ├── train.py · train_encoder.py · benchmark_baselines.py · run_task.py · plot_emb.py
│
└── notebooks/
    ├── 1-reproduce_i2v.ipynb            # reproduce the Identity2Vec paper
    ├── 2-phase_2_virtual_graph.ipynb    # build and inspect the role graphs
    ├── 3-phase3_gnn_encoder.ipynb       # train GraphSAGE, all result tables
    ├── 4-phase4_ogb.ipynb               # OGB datasets under the official protocol
    ├── 5-phase5_characterization.ipynb  # properties vs the gap; the stage-1 rules
    ├── 6-phase6_module3_validation.ipynb# the pre-registered stage-1 test
    └── 7-phase7_strategy_selection.ipynb# the stage-2 rule and the two-stage held-out test
```

**Adding an encoder** (GIN, GAT, …): write `virgo/encoders/<name>.py` with a `GNNEncoder` subclass defining `build_convs()`, then add one line to `ENCODERS` in `virgo/encoders/__init__.py`. Every driver, CLI and scoreboard row picks it up with no further edit.

---

## 8 · Setup

Conda environment **`i2v`** (Python 3.12):

```bash
conda activate i2v
```

From scratch:

```bash
pip install numpy==1.26.4 networkx gensim==4.3.3 scipy==1.12.0 scikit-learn matplotlib jupyter ipykernel
pip install torch torch-geometric node2vec
```

---

## 9 · Usage

```bash
# --- apply the framework to a new graph -------------------------------------
# stage 1 needs the role graph BUILT (deterministic; not training)
python -m virgo.virtual_graph --input input/<ds>.edgelist --sim psi --k 10
python experiments/predict_gate.py --datasets <ds> --step predict      # writes the call BEFORE training
python experiments/predict_strategy.py --datasets <ds> --step predict  # only if stage 1 said augment

# --- then train and score ----------------------------------------------------
python experiments/run_core.py --datasets <ds> --task link_prediction
python experiments/predict_gate.py --datasets <ds> --step score
python experiments/predict_strategy.py --datasets <ds> --step score

# --- refit the analyses (panel-guarded) --------------------------------------
python experiments/characterize.py --step all      # stage-1 rule screen
python experiments/gate_rules.py                   # the gate screen
python experiments/strategy_select.py              # the stage-2 screen

# --- individual pieces --------------------------------------------------------
python experiments/train_encoder.py --input input/cora.edgelist --sim psi --k 10 --seed 42
python experiments/run_task.py --task linkpred --dataset cora --retrain
python experiments/run_ogb.py --dataset ogbn_arxiv
```

The notebooks are the narrative version of the same commands; run them in order on the "Python (i2v)" kernel. Each reuses files already on disk.

---

## 10 · Reproducibility and conventions

- Seed fixed at 42 everywhere (split, initialization, sampling). Frozen-rule fitting used seeds 42/43/44; the strategy work and every held-out test use 42–51.
- **Which seed count a number came from matters.** `results/scoreboard.csv` now holds the 10-seed sweep; the 3-seed board it replaced is archived at `results/scoreboard_3seed.csv`. Re-scoring an older module against the current board can move a verdict — always name the seed count.
- Every scoreboard row comes from one frozen pipeline (`run_core.py` + `run_ogb.py`, settings in `virgo/config.py`). Metrics reproduce exactly; the embeddings themselves agree to ~2e-6 (float32 aggregation order), invisible at four decimals.
- `input/` is read-only. Derived files go to `output/`, scores to `results/`.
- Link prediction retrains on the 70% training graph alone, so no test edge leaks.
- A paper number counts as reproduced only within ±0.05 of the original.
- Predictions are written to disk **before** the encoder runs. A prediction script never imports a fitting function.
- Research decisions and results go to `docs/paper_log.md`; day-to-day changes go to `docs/notes.md`.

---

## 11 · Scope

**Out of scope, deliberately:** external node attributes — every feature is graph-derived, so a gain cannot be credited to attributes instead of to the rewiring under test; non-Euclidean / hyperbolic latent spaces (reserved for separate work); anomaly detection (set aside when the characterization study became the focus).

**Future work, not started:** **rules for degree and Ψ** — stage 2 covers centrality only; a learnable weight blending the original and role graphs per dataset (needs many, likely synthetic, datasets to train); embeddings as compact structural summaries so a large graph fits an LLM's context window.

**Open, and next:** a stage-2 test below the cut, to exercise the rule's negative side; a rule separating Ψ from degree, currently blocked — the sub-cut zone holds one Ψ dataset against two degree datasets, where every property separates the groups perfectly and therefore carries no information; and GIN alongside GraphSAGE (`virgo/encoders/gin.py` is wired and registered but has produced no results yet).

---

## Credits

- **Identity2Vec** — *Learning mesoscopic structural identity representations via a Poisson probability metric*, Oluigbo et al. Provides the Ψ score and the walk baseline (`virgo/identity2vec.py`, frozen).
- **Heterophilous benchmarks** — Platonov et al. 2023 (`roman_empire`, `tolokers`, `questions`, `minesweeper`, `amazon_ratings`, `squirrel_filtered`).
- **LINKX / Facebook100** — Lim et al. 2021 (`reed98`, `amherst41`, `johnshopkins55`, `cornell5`).
- **struc2vec** — vendored in `third_party/`, used as published.

This project contributes the cached Identity2Vec walker, the seven-variant role-graph builder, the GraphSAGE encoder over role graphs, and the two-stage decision framework above.
