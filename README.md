# Structure Aware Graph Augmentation for Graph Neural Networks

*A characterization-guided, two-stage decision framework for structural role-graph augmentation.*

Structural-role augmentation adds edges between nodes that occupy **similar topological positions** - two hubs, two bridges, two peripheral nodes - even when no path connects them. A standard GNN can only pass messages along real edges, so role-twins never interact; a **role graph** gives them a channel.

The problem is that this does not always help. Measured across 19 graphs, role augmentation improves link prediction on some and actively damages others, and it never once improves node classification. So the useful question is not *"does rewiring work?"* but **"for this graph, should I rewire at all - and if so, with which structural signal?"**

This project answers both, **before any encoder is trained**, with a two-stage decision framework read off the graph's own properties.

The Identity2Vec walker underneath is reproduced and cached: on a 265-node graph it drops from 916.6 s to 4.4 s (~207× faster), with byte-identical embeddings.

---

## 1 · The headline result

GraphSAGE, K = 10, 10 seeds (42–51), five locked graph variants, one frozen pipeline.

| task | augment | tie | keep original | datasets |
|---|---|---|---|---|
| **link prediction** | **9** | 4 | 6 | 19 |
| **node classification** | **0** | 2 | 13 | 15 |

Two things follow, and they are the reason the framework exists:

1. **Node classification has a boundary, not a rule.** No dataset, in any panel, at any seed count, has ever shown a significant node-classification gain from role augmentation. The reportable finding is *"never augment for NC"*.
2. **Link prediction genuinely splits.** Roughly half the graphs benefit and half are hurt, so the decision is worth making - and worth predicting.

---

## 2 · The two-stage framework

Everything below is computable **before training an encoder**.

```
graph ──► STAGE 1: augment, or keep the original?
             │                        │
       keep original               augment
             │                        ▼
             │            STAGE 2: which structural signal?
             ▼                        ▼
   train on the original      build that role graph, then train
```

**Stage 1 - whether to augment** (`virgo.frozen_rules.predict_gated`), applied in order:

| step | rule | cut | any cut in | needs labels |
|---|---|---|---|---|
| 1 | **adjusted homophily** - *the decision*: high ⇒ keep the original, low ⇒ normally augment | `< 0.227` | (0.0926, 0.3613) | yes |
| 2 | **average clustering** - *the exception*, inside the low-homophily zone only: high clustering ⇒ keep the original after all | `< 0.5573` | (0.5329, 0.5769) | no |
| 3 | **largest-component fraction** - fallback for unlabelled graphs | `> 0.9588` | (0.9177, 1.0) | no |

Read it as **one characteristic with one safety check**, not two equal conditions. Adjusted homophily makes the call and is a **veto on its high side**: high reliably means *keep the original*. Low homophily is **necessary but not sufficient**, and clustering is what catches the shortfall - a graph can be label-scrambled and still have dense, tightly knit neighbourhoods, in which case role edges add nothing it did not already have.

The evidence for the shortfall is `minesweeper` (0.0094, keeps) against `squirrel_filtered` (0.0086, augments) - 0.0008 apart with opposite outcomes, so **no single-variable split can separate them**. The clustering exception is measured on the **original graph**: no role graph has to be built, and no labels are needed, so it can be evaluated on any graph before anything is trained. It replaced a Module-4 gate on role-graph edge retention, which needed the rewiring built first and did not transfer (§3).

**Stage 2 - which structural signal**, consulted only when stage 1 says augment:

| rule | cut | any cut in | fitted on |
|---|---|---|---|
| **adjusted neighbour-label predictability** high ⇒ use **eigenvector centrality** | `> 0.0092` | (0.006, 0.0123) | 14 datasets, 7 of them augmenting |

Read plainly: **when a node's class can be read off the mix of labels around it, centrality-based rewiring is the right augmentation.** Below the cut the rule returns `"psi or degree (undetermined)"` - it separates centrality from the rest and deliberately claims nothing about Ψ versus degree.

**Status and known limits.** Stage 1 was locked in this form on 2026-09-01, after the clustering exception outscored the retention gate it replaced on six unseen graphs (§3). Both stage-1 properties are now read off the **original graph**, so nothing has to be built or trained to get a verdict. What is still open: the 0.5573 cut is a **candidate, not a proven universal threshold**, its negative side rests on two graphs, and one low-homophily *keep* (`minesweeper`, clustering 0.4355) remains unexplained by either condition - see §7.

---

## 3 · What is validated, and what is not

The distinction is kept in the code: `characterize.py`, `gate_rules.py` and `strategy_select.py` **fit** and are panel-guarded; `predict_module3.py`, `predict_gate.py` and `predict_strategy.py` only ever **read** `virgo/frozen_rules.py`. A prediction script that imported a fitting function would make the validation circular.

| component | fitted on | held-out test | result |
|---|---|---|---|
| adjusted homophily (stage 1) | 7 datasets | 9 unseen, pre-registered | **4/6** decided |
| largest-component fraction (fallback) | same 7 | same 9 | **4/6** decided |
| **clustering exception (stage 1)** | 13-dataset low-homophily zone - *fitted* | 6 unseen graphs, 4 decided | **4/4**, against rule 1 alone at 3/4 |
| centrality rule (stage 2) | 14 datasets, 10 seeds, nothing held out | 3 LINKX, `twitch_de`, then 9 corpus cells | **6/13** — equals the base rate out of sample |
| ~~retention gate~~ *(superseded 2026-09-01)* | 13 datasets - *includes those 9* | 4 unseen LINKX | **2/3**, beaten by rule 1 alone |

Adding clustering moves the panel from rule 1's 7/10 decided cells to **8/10** (it fixes `amazon_ratings`, still misses `minesweeper`, and `ogbl_ddi` is unlabelled so it counts against every row), and the six-graph test to **4/4**. Across panel and test together: **rule 1 alone 10/14, with the exception 12/14, with the old retention gate 11/14.**

Corrections and cautions that must travel with those numbers:

- **The retention gate's only differentiating call was its only error.** `reed98` augments at 2.55σ; the gate said keep. That call made stage 1 **2/3** where **rule 1 alone would have been 3/3** - which is why the gate was dropped rather than kept alongside. Any wording claiming "the gate stopped reed98" is false.
- **The clustering cut is a candidate, not a universal constant.** Its negative side rests on two graphs - `amazon_ratings` in the panel and `chameleon_filtered` out of it (adjusted homophily 0.0295, clustering 0.5769, keeps at −1.04σ) - and the six-graph test was **retrospective**: all six were trained and scored before the screen existed. That is transfer evidence, not a pre-registration.
- **The panel's leave-one-out ranking reversed out of sample.** `degree_skew` led on the panel (LOO 0.80 against clustering's 0.50) and then went 3/4 on the unseen graphs, where clustering went 4/4. Clustering's panel LOO is dragged down by `minesweeper`, the one low-homophily keep it never explained.
- **Stage 2's rule has not beaten the base rate out of sample.** On the 20-graph degree corpus, 9 cells resolve to a single signal (3 centrality, 3 degree, 3 Ψ) and the rule scores **3/9** - exactly what a constant "always centrality" predictor scores, because **no corpus graph falls below the 0.0092 cut**, so the rule never actually fires negative. Cumulative held-out record **6/13**. The cut is not shown to be wrong; it is shown to be untested, and above it the rule asserts centrality on graphs that split three ways.
- **Stage 2 also has a counterexample.** `twitch_de` (ingested 2026-09-01) has adjusted neighbour predictability 0.1235, far above the cut, so the rule says *centrality* — and centrality comes fourth (0.5886) behind Ψ (0.6258). The record is **3/4**, and the negative side is still untested: the first three held-out graphs all sat above the cut, so a constant "always centrality" predictor would have scored the same 3/3.
- **Stage 1's first post-freeze test split 1/2.** The clustering exception was locked before either graph was trained. `twitch_de` — low homophily, low clustering ⇒ augment — is right, at **+21.7σ**, the largest gap in the study. `deezer_europe` — also low on both ⇒ augment — is **wrong**: it keeps the original at −1.52σ, and no condition in stage 1 catches it. Stage-1 running total: **13/16** decided cells.

Seed counts matter: the published 5/7 and 4/7 for stage 1 are **3-seed** figures; re-scored on the current 10-seed board they are **4/6 and 4/6** (`lastfm_asia` flipped keep → tie). No frozen cut moved. Always name the seed count.

---

## 4 · Method

The variable under study is **the graph**, not the encoder. One fixed GNN sees every variant.

```
input graph → structural signals → role graph → encoder → embeddings → evaluation
              (degree, eigenvector  (top-K most   (GraphSAGE  (64-dim    (NC weighted F1,
               centrality, Ψ,        role-similar  or DeepWalk) per node)  LP AUC)
               clustering; cached)   nodes)
```

- **Signals** - degree and eigenvector centrality, computed once per graph and cached (Identity2Vec recomputes them inside its walk loop).
- **Similarity** - KL divergence λ → Poisson Ψ, exactly as Identity2Vec defines it.
- **Role graph** - `virgo/virtual_graph.py` links each node to its top-K most structurally similar nodes; K = 10 throughout, ties broken by seeded sampling.
- **Encoder** - 2-layer GraphSAGE, mean aggregation, unsupervised Skipgram-style loss. DeepWalk on the same graph is the walk-based comparison.
- **Evaluation** - node classification by logistic regression on the embeddings (weighted F1); link prediction by AUC over a 70/30 edge split, with the role graph **rebuilt from the 70% training edges** so no test edge leaks.

| variant | role signal | construction | locked |
|---|---|---|---|
| `original` | - | the input graph, unchanged (the control) | ✓ |
| `psi` | Identity2Vec Poisson/KL score Ψ | replaces the edges | ✓ |
| `degree` | degree | replaces the edges | ✓ |
| `centrality` | eigenvector centrality | replaces the edges | ✓ |
| `hybrid` | Ψ | original ∪ role edges | ✓ |
| `hybrid_degree` | degree | original ∪ role edges | |
| `hybrid_centrality` | eigenvector centrality | original ∪ role edges | |

The five *locked* variants are the ones every frozen rule was fitted against, and the analysis scripts read that fixed list, so promoting a variant can never silently refit a locked rule. Hybrids vote for the signal they **add**, so a stage-2 answer names a signal rather than one of seven files.

---

## 5 · Datasets

51 registered graphs - the 27-graph study corpus plus the separate 20-graph `DEGREE_RULE_CORPUS` used only for the stage-2 degree question - spanning citation, molecular, co-purchase, crowdsourcing, linguistic, social, Wikipedia, web, streaming, coauthorship, email, air-transport and drug-interaction domains, with adjusted homophily from −0.294 to 0.856 and average degree from 2.8 to 88.3. Which panel each one belongs to, where it came from, and the full citations are in **[DATASETS.md](DATASETS.md)**.

Every graph is used **structurally only**: published node features are ignored by design, so a gain cannot be credited to attributes instead of to the rewiring under test.

---

## 6 · Repository

Two code folders, one rule: **`virgo/` is imported, `experiments/` is run.** (`virgo/` is an internal package name the project title no longer matches.)

```
├── input/ labels/ splits/     # graphs, labels, saved 70/30 LP splits - input/ is read-only
├── output/ results/           # embeddings; scoreboard.csv, graph_health.csv, module*.csv
├── notebooks/ 1-7             # the narrative version: reproduce → role graphs → encoder → OGB
│                              #   → characterization → stage-1 validation → stage-2
├── virgo/                     # ── the library ──
│   ├── config.py              # THE settings: paths, dataset registry, variants, GNN params, seed=42
│   ├── frozen_rules.py        # THE frozen artifacts: panels, stage-1 rules + exception, stage-2 rule
│   ├── graph_io.py            # THE graph policy + the single loader every stage reads through
│   ├── identity2vec*.py       # original walk algorithm + the cached rewrite (identical output)
│   ├── virtual_graph.py       # the top-K role-graph builder (all seven variants)
│   ├── encoders/              # base.GNNEncoder + sage · gin · walk, behind an ENCODERS registry
│   ├── data/ eval/            # dataset builders; nodeclass (F1) · linkpred (AUC) · ogb · runner
│
└── experiments/               # ── the entry points, one argparse CLI each ──
    ├── run_core.py run_ogb.py # the frozen sweeps
    ├── characterize.py gate_rules.py strategy_select.py stage1_pairs.py degree_rule.py   # FIT (panel-guarded)
    ├── tie_break.py            # paired per-seed re-scoring of cached embeddings (never trains)
    └── predict_module3.py score_module3.py predict_gate.py predict_strategy.py   # READ frozen_rules only
```

**Adding an encoder** (GIN, GAT, …): write `virgo/encoders/<name>.py` with a `GNNEncoder` subclass defining `build_convs()`, then add one line to `ENCODERS` in `virgo/encoders/__init__.py`. Every driver, CLI and scoreboard row picks it up with no further edit.

---

## 7 · Roadmap

Ordered, and current:

1. **Finish stage 1.** The second condition now reads off the original graph (§2), so what remains is a **pre-registered** test of `avg_clustering < 0.5573` on graphs that have not yet been trained - the six-graph result is retrospective - plus more low-homophily *keep* cases, since the exception's negative side stands on two. Untried candidates if it does not hold: over-squashing diagnostics (balanced-Forman / Ollivier-Ricci curvature, spectral gap, effective resistance) and degree heterogeneity.
2. **More stage-2 rules.** Centrality is the only frozen signal rule, and it now has one counterexample (§3). A separate 20-graph pool, `DEGREE_RULE_CORPUS`, was ingested in three waves for the degree question (`experiments/degree_rule.py`, `tie_break.py`). Two results came out of it. **Degree versus everything: no rule** - 21 augmenting cells, 5 naming degree alone, best property ρ −0.33 with 4 exceptions; the degree winners span film co-occurrence, Q&A, streaming, air transport and co-purchase and share no structural property. That is now a measured negative, not a shortage of data. **Degree versus Ψ: a live candidate** - `nbr_label_entropy < 0.6724 ⇒ degree` separates all 9 relevant cells with LOO 8/9 (chance of a free split 0.016) and survived one pre-registered held-out call. It stays unpromoted: the property is exploratory tier, and one cell is not a validation. The obstacle underneath both is that **degree and Ψ role graphs usually score the same** - 9 of 20 corpus graphs cannot name a signal even under a paired comparison - which is expected, since Ψ is a Poisson/KL score over degree-based signatures. Measuring the edge overlap between the two constructions is the next experiment.
3. **Anomaly detection as a third downstream task.** Node classification and link prediction answer the augment question differently; anomaly detection is the case where role information should matter most, since structural outliers are the target. Structural-only, so structural anomaly injection rather than attribute-driven fraud benchmarks, then both stages re-screened on it.
4. **Other architectures: GIN and GAT.** `virgo/encoders/gin.py` is wired and registered but has produced no results; GAT is not written yet. Both are one file plus one registry line, and the point is whether the stage-1 and stage-2 calls survive a change of aggregator.

**Out of scope, deliberately:** external node attributes; non-Euclidean / hyperbolic latent spaces (separate work); learnable per-dataset blending of original and role edges (needs many, likely synthetic, datasets).

---

## 8 · Setup and usage

Conda environment **`i2v`** (Python 3.12):

```bash
conda activate i2v
# from scratch:
pip install numpy==1.26.4 networkx gensim==4.3.3 scipy==1.12.0 scikit-learn matplotlib jupyter ipykernel
pip install torch torch-geometric node2vec
```

```bash
# --- apply the framework to a new graph (predictions are written BEFORE any training) ---
python experiments/predict_gate.py     --datasets <ds> --step predict        # stage 1, original graph only
python experiments/predict_strategy.py --datasets <ds> --step predict        # only if stage 1 said augment

# --- then train and score ---
python experiments/run_core.py         --datasets <ds> --task link_prediction
python experiments/predict_gate.py     --datasets <ds> --step score
python experiments/predict_strategy.py --datasets <ds> --step score

# --- refit the analyses (panel-guarded) ---
python experiments/characterize.py --step all      # stage-1 rule screen
python experiments/stage1_pairs.py                  # the stage-1 second-condition screen
python experiments/strategy_select.py              # the stage-2 screen
```

Reproducibility conventions: seed 42 everywhere (split, init, sampling); frozen-rule fitting used 42/43/44, the strategy work and every held-out test 42–51; every scoreboard row comes from one frozen pipeline; link prediction retrains on the 70% graph alone; a paper number counts as reproduced only within ±0.05. Notebooks are the narrative version of the same commands - run them in order on the "Python (i2v)" kernel.

---

## Credits

- **Identity2Vec** - Oluigbo et al., *Learning mesoscopic structural identity representations via a Poisson probability metric*. Provides the Ψ score and the walk baseline (`virgo/identity2vec.py`, frozen).
- **Heterophilous benchmarks** - Platonov et al., 2023. **LINKX / Facebook100** - Lim et al., 2021. **struc2vec** - vendored in `third_party/`, used as published.

This project contributes the cached Identity2Vec walker, the seven-variant role-graph builder, the GraphSAGE encoder over role graphs, and the two-stage decision framework above.
