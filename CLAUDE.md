# CLAUDE.md — Structure Aware Graph Augmentation for Graph Neural Networks

**A characterization-guided, two-stage decision framework for structural role-graph augmentation.**

Research project extending Identity2Vec (I2V, Oluigbo et al.). Target: a publishable paper. This file governs all agentic work in this repo.

> **Renamed 2026-08-15.** The project was formerly *ViRGo — Virtual Role-Graph Embedding for Structural Identity*. Do not reintroduce that title in docs or prose. The Python package directory is still `virgo/` (an internal identifier only); renaming it would break every import, notebook and cached path, so it stays until a deliberate refactor.

---

## 1. Goal & contribution

Structural-role augmentation connects nodes that occupy similar topological positions but are far apart in the graph. It does not always help. The contribution is a **two-stage decision framework, computable before any encoder is trained**, that decides (1) whether to augment at all and (2) which structural signal to build the role graph from.

**Headline (GraphSAGE, K=10, 10 seeds 42–51, five locked variants):**

| task | augment | tie | keep original | scope |
|---|---|---|---|---|
| link prediction | **9** | 4 | 6 | 19 datasets |
| node classification | **0** | 2 | 13 | 15 cells |

So NC has a reportable **boundary** ("never augment"), not a predictor; LP genuinely splits and is worth predicting. The work is a **study / characterization** — a guide for when structural augmentation helps — not a claim that role graphs always win.

> Superseded, kept so the old wording is not re-introduced: the pre-2026-07-27 claim "role graphs never help link prediction" held only on the core-4 + OGB panel. The heterophilous datasets added since produce genuine LP augment wins.

**Hard scope rule — stay purely structural.** Features are graph-derived — degree, eigenvector centrality, the I2V score Ψ, and clustering — used both to build the role graph and as the GNN's input. Ablation D confirmed they are necessary (random features drop to the DeepWalk baseline or below). Never use external node attributes (OGB text embeddings, fastText, bag-of-words, biological descriptions). Excluding them is deliberate: a gain could otherwise be credited to the attributes rather than to the rewiring under test.

Non-Euclidean / hyperbolic latent space is out of scope (reserved for a second paper). Do not implement it. Anomaly detection was set aside when the characterization study became the focus.

**Immediate target:** the LoG (Learning on Graphs) conference. The thesis reuses the same content.

---

## 2. THE FRAMEWORK (the thing being contributed)

Frozen in `virgo/frozen_rules.py`. Nothing here is ever recomputed from data that includes the graph being predicted.

### Stage 1 — whether to augment (`predict_gated`)

| step | rule | cut | interval | needs labels |
|---|---|---|---|---|
| 1 | `homophily_adjusted` low ⇒ augment | `< 0.227` | (0.0926, 0.3613) | yes |
| 2 | **the gate**, low-homophily zone only: `original_retention` low ⇒ augment | `< 0.0119` | (0.0062, 0.0176) | no |
| 3 | `largest_component_frac` high ⇒ augment (unlabelled fallback) | `> 0.9588` | (0.9177, 1.0) | no |

Order: **rule 1 vetoes on its reliable side → the gate decides inside the low-homophily zone → rule 2 covers unlabelled graphs.**

Adjusted homophily is a **veto, not a predictor**: high ⇒ keep original is reliable; low homophily is **necessary, not sufficient**. Decisive pair: `minesweeper` (0.0094, keeps) vs `squirrel_filtered` (0.0086, augments) — 0.0008 apart, opposite verdicts, so **no single-variable split can ever separate them**. That is why the gate exists.

`original_retention` = the share of ORIGINAL edges that survive into the built role graph, measured on the `psi` edgelist at K=10. Counter-intuitive direction, worth reporting: **the less of the original graph the rewiring keeps, the more augmentation helps.**

### Stage 2 — which signal (`predict_strategy`)

| rule | cut | interval | precondition |
|---|---|---|---|
| `nbr_predictability_adjusted` high ⇒ use **centrality** | `> 0.0092` | (0.006, 0.0123) | augmentation already indicated |

Below the cut it returns `"psi or degree (undetermined)"` — it separates centrality from the rest and claims nothing about Ψ vs degree. **Scope decision (user, 2026-08-15): centrality only, for now.** That is the objective, not a shortfall.

Fitted evidence: 0 exceptions over the 7 augmenting datasets, ρ 0.866, graded 0.0123 → 0.8498 (not a two-group split), LOO 5/7 vs a 4/7 majority baseline.

Four caveats that must travel with it:
1. **Fitted, not validated** on the panel — every labelled dataset is in it.
2. **Band-dependent** — exists under the `sem` tie band only; under the Module-2 `sigma` band the ties re-inflate and it takes 2 exceptions. The band choice is part of the rule, not context.
3. **Co-predictor** — `degree_assortativity` (exploratory tier) gives the identical split at ρ 0.866; the two correlate at 0.71, so which is THE predictor is undetermined by this data.
4. **Not homophily** — ρ 0.32 against `homophily_adjusted`; `roman_empire` has the panel's lowest adjusted homophily (−0.0468) and centrality still wins. Same property Module 2 screened and dropped for the *augment* question. Different question, different answer — report it that way, never as a resurrection.
5. The cut is pinned by `squirrel_filtered` (0.0123), which is itself a tie (`centrality|degree`). Dropping it keeps 0 exceptions, LOO 5/6, and widens the interval to (0.006, 0.3324).

---

## 3. Validation status — say this precisely

| component | fitted on | held-out test | result |
|---|---|---|---|
| rule 1 (homophily) | 7 datasets / 12 cells, 3 seeds | 9 unseen, pre-registered | **4/6** at 10 seeds |
| rule 2 (fragmentation) | same | same | **4/6** at 10 seeds |
| the gate | 13 datasets — *includes the 9 above* | 4 unseen LINKX | **2/3** |
| stage-2 centrality rule | 14 datasets, 10 seeds, nothing held out | 3 unseen LINKX | **3/3** top-scoring |

**Stage 1 on the LINKX set — the correction that must not be lost.** `reed98` was trained *after* `module5_scored.csv` was written and it **augments at 2.55σ** (+0.0401, the largest gap of the four; all five locked variants beat `original`). The gate had called it *keep original*. Therefore:
- the combined call is **2/3**, not 2/2;
- **rule 1 alone would have been 3/3** — it said augment on all four;
- the gate's only differentiating call was its only error, and ρ(`original_retention`, `gap_rel`) is **+0.4** on this set against a fitted −0.86.

Keep the gate (it addresses the decisive pair and needs no labels) but report it as **screened and fitted, not yet transferring**. Any prose claiming "the gate 2/2" or "stage 1 stopped reed98" is false — fix it wherever found.

**Stage 2 on the LINKX set.** All three graphs stage 1 routed to augment were predicted centrality (0.2128 / 0.2278 / 0.2760 vs the 0.0092 cut) and centrality was the top-scoring variant on all three, above the original in each case: amherst41 0.6749 vs 0.6651 (runner-up `hybrid_centrality` 0.6718 — the top two both add centrality, so the signal call does not depend on their order), johnshopkins55 0.7160 vs 0.6854 (clear margin), cornell5 0.7078 vs 0.6910 (centrality slightly ahead of `hybrid` 0.7050).

Standing limitation to state alongside it: the cut is low enough that every ordinary labelled graph clears it, so all three calls were *centrality* and a constant "always centrality" predictor scores the same on this set. Confirming the negative side needs an augmenting graph **below** the cut; only `tolokers` (−0.3274), `questions` (−0.0041) and `actor` (0.0060) qualify, and all three are panel members.

**Seed-count discipline.** `results/scoreboard.csv` is the 10-seed board; the 3-seed board is archived at `results/scoreboard_3seed.csv`. The published Module-3 figures **5/7 and 4/7 are 3-seed**; re-scored at 10 seeds they are **4/6 and 4/6**, because `lastfm_asia` flipped keep → tie. The frozen rules and the gate fit are unaffected (discovery-panel verdicts and all 7 low-homophily-zone verdicts are identical at 10 seeds). Always name the seed count, or re-score first.

**Quote a range, not a number.** The cut is only the midpoint between the two datasets either side of it, so "0 mistakes on the panel" happens automatically whenever the data separates — it shows the data separates, not that the rule is good. Same reason p-values are reported and never used as a gate at this panel size.

**Report the predictions that failed.** Homophily was proposed for node classification and density for link prediction; both failed (ρ −0.30 and +0.49). Quietly replacing them with the properties that happened to work would be HARKing.

---

## 4. Pipeline

1. **Structural signal** — per-node degree + eigenvector centrality, computed once per graph and **cached** (static graph → caching is exact). Removes I2V's per-step recomputation.
2. **Similarity scoring** — KL-divergence λ → Poisson Ψ, exactly as I2V.
3. **Role graph** — connect each node to its top-K most structurally similar nodes. K=10 locked.
4. **Encoder** — inductive GNN: **GraphSAGE** (primary), **GIN** (expressive alternative, wired but unrun), **GAT** (ablation).

**Variants** (`cfg.VG_SIMS`, 7 official): `original`, `psi`, `degree`, `centrality`, `hybrid`, `hybrid_degree`, `hybrid_centrality`. The five in `cfg.VG_SIMS_LOCKED` are what every frozen rule was fitted on — the fitting scripts read that list, so promoting a variant can never silently refit a locked rule. Hybrids vote for the signal they ADD.

**Tasks** — node classification (logistic regression on embeddings, weighted F1); link prediction (70:30 split, AUC, leakage-free: retrain on the 70% graph only); dataset characterization (graph properties → when augmentation helps).

---

## 5. Status & phases

- **Phase 1 — reproduce I2V. ✅** Cached I2V (byte-identical, ~200× faster) + cross-model baselines; within ±0.05 of the paper, 3 seeds. Baselines used as published — **not fine-tuned** (out of scope).
- **Phase 2 — role graphs. ✅** Deterministic; each build logs a health row.
- **Phase 3 — GraphSAGE encoder. ✅** Design locked by ablations A–D (edge positives, mean aggregation, 2 layers, all four structural features, K=10). Caveat to keep: A–D were tuned on `enzymes` only.
- **Module 2 — stage-1 rule screen. ✅** `characterize.py --step all`, notebook 5. Seven primary properties screened per task as executable rules; `components` and `largest_component_frac` merged (Spearman −1.000, one variable). `nbr_predictability_adjusted` was screened and **dropped** here (fails the fixed-variant control, LOO = the majority baseline). NC yields no rule at all.
- **Module 3 — stage-1 validation. ✅** 9 held-out datasets, verdicts pre-registered before training. Result and its 10-seed re-score in §3.
- **Module 4 — the gate. ✅ locked 2026-08-05.** Four role-graph properties screened off the built `psi` edgelists, no retraining. Winner `original_retention`; the other three failed (`role_diversity` +0.29, `role_edge_enrichment` −0.39, `vg_edge_ratio` +0.64 with LOO 0.00). Collinearity was measured, not assumed — no pair reaches |ρ| ≥ 0.99. Two caveats: it does **not** separate the decisive pair (emitted mechanically as `separates_decisive_pair`, never as prose), and `GATE_PANEL` includes the Module-3 datasets, so it is **fitted, not validated**.
- **Module 5 — test the gate. ✅** 4 LINKX graphs. Result in §3: **2/3, and beaten by rule 1 alone.**
- **Module 6 — the stage-2 rule. ✅ locked 2026-08-13.** 14 datasets × 10 seeds. Diversity, not precision, was the binding constraint: datasets naming a single winning signal went 3/9 → 6/14. The negative result survives underneath the rule — no property separates Ψ from degree.
- **Module 7 — test the stage-2 rule. ✅** 3 LINKX graphs. Result in §3.
- **Next.** (a) A stage-2 test **below** the cut — the only thing that exercises the rule's negative side. (b) Ψ-vs-degree: currently **blocked**, not merely undone — `contrast()` reports `n_zone < min_cells` (1 Ψ dataset against 2 degree datasets), where every property separates the groups perfectly and therefore carries zero information; needs ≥2 more augmenting datasets below the cut. (c) Swap GraphSAGE → GIN.
- **Future work, do not start.** Learnable alpha (auto-blend original vs role graph; needs synthetic datasets); embeddings as LLM graph summaries.

---

## 6. Coding rules (match I2V style)

- **Mirror the I2V codebase**: `argparse` CLI; `build_graph()` / `learn_embeddings()` / `main(args)` structure; a class holding the core method (cf. `identity2vec.Graph`).
- **Fewest functions possible.** Each short, single-purpose, self-explanatory name. No helper unless necessary.
- **One-line comments max.** Triple-quoted one-line docstrings as in I2V.
- Self-explanatory file names (e.g. `virgo/virtual_graph.py`, `virgo/encoders/sage.py`, `virgo/eval/linkpred.py`).
- Models expose **`train(epochs)`**, not `fit()`.
- Prefer the main script to call only functions defined in base/abstract classes.
- No new dependency without need. Reuse the existing env (numpy 1.26.4, networkx, gensim 4.3.3, scipy 1.12.0, torch/torch-geometric).

**Layout rule.** Two code folders and one rule: **`virgo/` is imported, `experiments/` is run.**
- `virgo/` — `config.py` (THE settings), `frozen_rules.py` (THE frozen artifacts), `graph_io.py` (THE graph policy), `identity2vec*.py`, `virtual_graph.py`, `utils.py`, plus `encoders/`, `data/`, `eval/`.
- `experiments/` — every `argparse` entry point. No method code here.
- **The fit/predict split is load-bearing**: `characterize.py`, `gate_rules.py` and `strategy_select.py` FIT (they call `threshold()`) and are panel-guarded; `predict_module3.py` / `score_module3.py` / `predict_gate.py` / `predict_strategy.py` only ever READ `virgo/frozen_rules.py`. Never let a predict script import a fitting function, or the validation becomes circular.
- `third_party/struc2vec/` — vendored, used as published.
- Library modules with a CLI run as `python -m virgo.<module>` from the repo root; entry points run as `python experiments/<script>.py`.
- **Adding an encoder** (the extension point): new `virgo/encoders/<name>.py` subclassing `GNNEncoder` with one `build_convs(dims, agg)`, then one line in `ENCODERS`. Every driver, CLI and scoreboard row picks it up with no further edit. `--encoder all` stays the locked `graphsage_edge + deepwalk` pair.

---

## 7. Reproducibility (non-negotiable)

- Fixed `seed=42` everywhere (split, init, sampling). Frozen-rule fitting used 42/43/44; the strategy work and every held-out test use 42–51.
- **Name the seed count on every number**, or re-score first (see §3).
- Never modify files in `input/`; write derived files alongside, outputs to `output/`.
- Log every run setting and deviation in `docs/notes.md` (e.g. walk-length now 40, paper's 80 kept as a recorded deviation).
- A result is "reproduced" only when our metric is within ~±0.05 of the paper's.
- Predictions are written to disk **before** the encoder runs. For stage 1 the order is **build the role graph → predict → train → score** (building is not training, so the verdict is still pre-registered).
- Ship splits, seeds, and eval scripts with the method.

---

## 8. Deliverables

1. ✅ Cached I2V — embeddings identical to the baseline, ~200× faster.
2. ✅ `virgo/virtual_graph.py` — seven-variant top-K role-graph builder.
3. ✅ GraphSAGE encoder over the role graph (`virgo/encoders/sage.py`, on `base.GNNEncoder`); `gin.py` is wired and registered but has produced **no results yet**.
4. ✅ Eval scripts: node classification (F1), link prediction (AUC, leakage-free) — `virgo/eval/`.
5. ✅ The characterization table + stage-1 rules. Tables in `results/characterization_*.csv`, `candidate_rules.csv`, `feature_usefulness.csv`; figures in `results/figures/`.
6. ✅ **The two-stage framework**, frozen in `virgo/frozen_rules.py`, with all three held-out tests run and reported honestly (§3). Tables: `module3_*.csv`, `gate_candidates.csv`, `vg_characterization.csv`, `module5_*.csv`, `strategy_*.csv`, `module7_*.csv`.
7. 🔵 A stage-2 test below the cut.
8. 🔵 GIN results next to GraphSAGE.
9. 🔵 LoG paper draft (the thesis reuses it).
