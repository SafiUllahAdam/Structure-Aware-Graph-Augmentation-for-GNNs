# Structure Aware Graph Augmentation for Graph Neural Networks

*A characterization-guided, two-stage pre-training decision framework for structural role-graph augmentation.*

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

**One characteristic with one safety check**, not two equal conditions. Adjusted homophily makes the call and is a **veto on its high side**; low homophily is necessary but not sufficient, and clustering catches the shortfall - a label-scrambled graph with dense neighbourhoods gains nothing from role edges. The decisive pair is `minesweeper` (0.0094, keeps) against `squirrel_filtered` (0.0086, augments): 0.0008 apart, opposite outcomes, so **no single-variable split can separate them**. Both properties are read off the **original graph**, so a verdict needs nothing built and nothing trained.

**Stage 2 - which structural signal**, consulted only when stage 1 says augment:

| rule | cut | any cut in | fitted on |
|---|---|---|---|
| **adjusted neighbour-label predictability** high ⇒ use **eigenvector centrality** | `> 0.0092` | (0.006, 0.0123) | 14 datasets, 7 of them augmenting |

Plainly: **when a node's class can be read off the labels around it, centrality-based rewiring is right.** Below the cut the rule returns `"psi or degree (undetermined)"` - it claims nothing about Ψ versus degree.

**Status.** Locked 2026-09-01, replacing a Module-4 gate that needed the role graph built first and did not transfer. Still open: the 0.5573 cut is a **candidate, not a proven threshold**, its negative side rests on two graphs, and `minesweeper` remains an unexplained low-homophily keep (§7).

---

## 3 · What is validated, and what is not

The distinction is kept in the code: `characterize.py`, `gate_rules.py` and `strategy_select.py` **fit** and are panel-guarded; `predict_module3.py`, `predict_gate.py` and `predict_strategy.py` only ever **read** `virgo/frozen_rules.py`. A prediction script that imported a fitting function would make the validation circular.

| component | fitted on | held-out test | result |
|---|---|---|---|
| adjusted homophily (stage 1) | 7 datasets | 9 unseen, pre-registered | **4/6** decided |
| largest-component fraction (fallback) | same 7 | same 9 | **4/6** decided |
| **clustering exception** (low-homophily zone) | 13-dataset zone | 6 unseen, 4 decided | **4/4**, against rule 1 alone at 3/4 |
| centrality rule (stage 2) | 14 datasets, 10 seeds, nothing held out | 3 LINKX, `twitch_de`, then 9 corpus cells | **6/13** — equals the base rate out of sample |

Adding the clustering exception takes stage 1 from **10/14** to **12/14** across panel and held-out test.

Four cautions travel with those numbers:

- **The clustering cut is a candidate, not a universal constant.** Its negative side rests on two graphs, and the six-graph test was retrospective - transfer evidence, not a pre-registration.
- **Stage 1 has unexplained misses.** `minesweeper` and `deezer_europe` are low-homophily graphs that keep the original, and no condition catches either. Running total **13/16** decided cells.
- **Stage 2 has not beaten the base rate out of sample.** Cumulative **6/13**, equal to a constant "always centrality" predictor - because no held-out graph has fallen below the 0.0092 cut, so the rule never fires negative. The cut is untested, not shown to be wrong.
- **Seed counts matter.** The published 5/7 and 4/7 for stage 1 are **3-seed** figures; on the current 10-seed board they are **4/6 and 4/6**. No frozen cut moved. Always name the seed count.

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

1. **Finish stage 1.** Stage 1 works and is locked. Both things it looks at are read straight off the input graph, so no training is needed to get an answer. Two jobs left: test it on graphs we have not trained yet (so far we only checked it on graphs we had already run), and find more examples of the case it is weakest on - low-homophily graphs that should *not* be augmented. We only have two of those.
2. **Stage 2 beyond centrality.** We can predict when to use **centrality** (right 8 times out of 17 on new graphs). We still cannot predict when to use **degree**, and not for lack of trying: we searched every graph property we have, twice, over 48 graphs, and froze two candidate rules that both failed when tested. Every rule we found was no better than guessing. What we did learn is that degree does not win at some threshold - it wins on a *type* of graph: bipartite, heterophilous, low-degree ones. There are 9 such wins out of 48 graphs. The only way forward is to collect more graphs of that type, which is slow.
3. **Anomaly detection as a third task.** So far we only test on node classification and link prediction. Anomaly detection looks for nodes that sit in an odd position in the graph - which is exactly what role information describes - so this is where augmentation should help most. We will create the odd nodes ourselves by changing the structure, not by using ready-made fraud datasets, because those depend on node attributes and this project uses structure only. Then both stages get checked again on the new results.
4. **GIN and GAT.** Every result so far uses one encoder, GraphSAGE. GIN is already coded but never run; GAT is not written. Each needs one new file and one line in a registry. The point is to check that the stage-1 and stage-2 answers do not change just because the encoder changed.

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
