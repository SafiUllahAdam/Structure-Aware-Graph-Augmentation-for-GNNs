# CLAUDE.md — ViRGo

**ViRGo: Virtual Role-Graph Embedding for Structural Identity**

Research project extending Identity2Vec (I2V, Oluigbo et al.). Target: a publishable paper. This file governs all agentic work in this repo.

---

## 1. Goal & Contribution

Study a simple, practical question: **when does a GNN need a rewired "virtual" graph, and when is the original graph already enough?** — and predict the answer from the graph's own properties (starting with homophily). The virtual graph (role-based rewiring), not the encoder, is the thing we vary.

**Honest headline finding (9 datasets, 16 dataset × task cells).** The original graph is a very strong baseline: **11 keep original, 1 tie, 4 augment**. The four augment cells are **all link prediction** (`ogbl_ddi`, `roman_empire`, `tolokers`, `questions`), and **no node classification cell augments at all** — so NC has a reportable boundary ("never augment"), not a predictor. So the paper is a **study / characterization** — a guide for *when* structural augmentation helps — not a claim that our virtual graph always wins.

> Superseded, kept so the old wording is not re-introduced: the pre-2026-07-27 claim "role graphs never help link prediction" held only on the core-4 + OGB panel. The three heterophilous datasets added 2026-07-27 produce genuine LP augment wins.

**Immediate target:** the LoG (Learning on Graphs) conference. The thesis reuses the same content.

**Hard scope rule — stay purely structural.** ViRGo's features are graph-derived — degree, eigenvector centrality, the I2V score Ψ, and clustering — used both to build the virtual graph and as GraphSAGE's input. Ablation D confirmed they are necessary: replacing them with random features drops performance sharply (to the DeepWalk baseline or below). Never use external node attributes (OGB text embeddings, biological descriptions). Excluding them is deliberate — attributes would confound the study, since a gain could then be attributed to the attributes rather than to the structural rewiring under test. The goal is to isolate structural identity (the inherited I2V premise).

Non-Euclidean / hyperbolic latent space is out of scope (reserved for a second paper). Do not implement it.

## Research Contributions

**1st (main) contribution — a "when to augment" study.**
- *What:* Work out when the original graph is already enough for a GNN, and when adding structural "role" edges/features helps — then predict which case you are in from the graph's properties.
- *Why:* People reach for graph rewiring without knowing whether it helps. A clear rule — "for graphs like this, keep the original; for graphs like that, add these features" — is genuinely useful to the community.
- *How:* Compare five graph variants under one fixed GNN on many datasets; characterize each dataset by its properties (homophily first, then degree spread, clustering, ...); relate those properties to the original-vs-augmented gap. Offered as a conjecture, not a proof.

**Technical contribution — modern GNN over the graph.**
- *What:* Replace I2V's guided walk + Skipgram with GraphSAGE, then try GIN.
- *Why:* A GNN aggregates directly over structure instead of learning from sampled walks. GIN can tell more graphs apart (isomorphism power) — we test whether that helps.
- *How:* `graph → structural features (cached) → virtual graph → GraphSAGE (then GIN) → embeddings → evaluation`. Order: get all GraphSAGE results first, then swap in GIN.

**Future work — not in the next stretch.**
- *Learnable alpha:* one learned weight that automatically blends the original and virtual graphs per dataset. Needs many (likely synthetic) datasets to train properly — parked as future work.
- *Embeddings as graph summaries for LLMs:* compact structural embeddings as a large-graph summary so structure fits an LLM's context window. Stretch; not started.

---

## 2. Method (pipeline)

Keep I2V front end, replace back end:

1. **Structural signal** — per-node degree + eigenvector centrality, computed **once per graph and cached** (graph-level, static graph → caching is exact). Removes I2V's per-step recomputation.
2. **Similarity scoring** — KL-divergence λ → Poisson Ψ, exactly as I2V.
3. **Virtual graph** — connect each node to its top-K most structurally similar nodes under Ψ. K is a tuned hyperparameter (sparsity vs over-smoothing tradeoff).
4. **Encoder** — inductive GNN over the virtual graph: **GraphSAGE** (primary), **GIN** (expressive alternative), **GAT** (ablation).

---

## 3. Tasks (evaluation)

- **Node classification** — logistic regression on embeddings, weighted F1.
- **Link prediction** — 70:30 edge split, AUC, leakage-free (retrain on the 70% graph only).
- **Dataset characterization (the new core)** — compute each graph's properties (homophily first, then degree spread, clustering, component fraction, label-vs-topology agreement) and connect them to *when* augmentation helps.

Anomaly detection is set aside for now — the LoG study replaced it as the immediate focus.

---

## 4. Status & phases

- **Phase 1 — reproduce I2V. ✅ done.** Cached I2V (byte-identical, ~200× faster, Deliverable #1) + cross-model baselines; within ±0.05 of the paper, 3 seeds. Baselines (DeepWalk / node2vec / struc2vec) used as published — **not fine-tuned** (out of scope).
- **Phase 2 — build the virtual graphs. ✅ done.** Five variants: `psi` (I2V Poisson/KL), `degree`, `centrality`, `original` (control), `hybrid`. Deterministic; each build logs a health row.
- **Phase 3 — GraphSAGE encoder. ✅ done.** Design locked by ablations A–D (edge positives, mean aggregation, 2 layers, all four structural features, K=10). Caveat to keep: A–D were tuned on enzymes only.
- **Phase 4 — the study + more data. ← current.** Three steps, in order:
  1. Run the existing GraphSAGE pipeline on small-to-medium **OGB** datasets (node: ogbn-arxiv; link: ogbl-collab, ogbl-ddi), **structural features only** — skip the huge 100M-node graphs. On each dataset both graphs (original and virtual) get the **same** structural features, so only the edges differ and graph structure is the single variable; ignore OGB's extra attributes (text embeddings, product descriptions, biological annotations). **We do not chase the OGB leaderboard** — its top models may use those attributes, and the paper states plainly that the goal is structural analysis, not leaderboard superiority. Per-dataset: `ogbl-ddi` has no node features and is a drop-in structure-only benchmark; `ogbn-arxiv` (128-dim skip-gram text features) and `ogbl-collab` (128-dim text features) do carry features, which we simply do not load — so the structural methodology is unchanged across all three.
  2. Build the **characterization table**: graph properties per dataset → the original-vs-augmented gap → a "when to augment" rule. ✅ **done** (`characterize.py --step all`, notebook 5). **Active panel = 7 datasets** (`characterize.PANEL`; `citeseer_linqs` and `proteins` excluded from all forward work by user decision 2026-07-29). 12 cells, 11 usable → 7 keep original / 4 augment / 1 tie. Seven primary properties screened separately per task as executable rules; four passed, `components` and `largest_component_frac` merged (Spearman **−1.000**, one variable) → three findings.

  **THE TWO FINAL CANDIDATE RULES (both link prediction, locked 2026-07-29):**
  1. **augment when `homophily_adjusted` < ~0.23** — ρ −0.90 / **−1.00** on the bias-free gap, LODO min\|ρ\| 0.80, graded across 5 distinct values. Lead with this one.
  2. **augment when `largest_component_frac` > ~0.96** — ρ +0.78, LODO 0.71. Real but a **two-group split**: all four augment cells sit at exactly 1.0, so the cut is arbitrary in (0.9177, 1.0) and it coincides with the batch confound.

  **Quote intervals, never the 4-decimal cut (added 2026-07-31).** `threshold()` returns the midpoint between the two datasets straddling the boundary, so on separable data "0 exceptions" is *guaranteed* — it measures separability, not the rule. The panel pins only an interval: rule 1 anywhere in **(0.0926, 0.3613)**, rule 2 in **(0.9177, 1.0)**. Honest out-of-sample numbers, each rule making exactly one error: **rule 1 leave-one-out 4/5 vs a 0.60 majority baseline** (misses `enzymes`; fold cutoffs swing 0.191–0.4319), **rule 2 5/6 vs 0.67** (misses `cora`). **Nested LOO** (predictor re-chosen inside each fold) = **4/6**; homophily is re-selected in 5/6 folds so predictor choice is stable, and the second miss is `ogbl_ddi` — unlabelled, so rule 1 cannot be evaluated on it at all. That coverage gap is exactly when rule 2 is needed. LOO is **not** a substitute for Module 3: same panel, same confound; it measures how tightly the cut is pinned, not transfer.

  `nbr_predictability_adjusted` < 0.4084 was **screened and dropped**: it clears on `gap_rel` only, is rejected by the fixed-variant control (ρ −0.60 < the 0.7 gate), its LODO floor is 0.40, and it scores **3/5 = exactly the 0.60 majority baseline** out of sample — zero information beyond guessing, which is now a mechanical gate (`GATES["loo_above_majority"]`) rather than a judgement call. Its LODO *sign* was stable; the failure is magnitude, not direction.

  **NC yields no rule at all** — 0 of 5 usable NC cells augment, so nothing can be separated; only the boundary "never augment" is reportable, and any held-out NC test is a falsification attempt.

  The professor-proposed homophily (NC) and density (LP) rules were **tested and not supported** — but note the reversal worth reporting: adjusted homophily fails on NC (ρ −0.30, its intended target) and is the strongest **LP** rule. Density fails outright (LP × avg_degree ρ = +0.49; sparse `roman_empire` augments). Report both as failed pre-specified predictions; do not quietly swap in the post-hoc winners. Homophily and neighbour-label predictability are both class-balance-adjusted; raw values are not comparable across datasets with different class counts. Never quote a raw scipy Spearman p at small n — its t-approximation gives 1e-24 at |ρ| = 1, so `rho()` floors p at 2/n!. Screen gates and rationale in `characterize.GATES`; full write-up in `docs/paper_log.md`.
  2b. **Module 3 — validation. ✅ done (9 held-out datasets).** The two rules were frozen, verdicts pre-registered **before** training, then scored: **rule 1 5/7, rule 2 4/7**. The finding that matters is that rule 1 fails *asymmetrically* — high adjusted homophily called "keep original" and was right **3/3**; low homophily called "augment" and was right only **2/4**. So low homophily is **necessary, not sufficient**: rule 1 is a reliable **veto**, not a predictor. The decisive case is `minesweeper` (0.0094, keeps) vs `squirrel_filtered` (0.0086, augments) — 0.0008 apart with opposite verdicts, so **no single-variable split can ever separate them**. Binding constraint still open: no held-out graph has been **fragmented + heterophilous**, the quadrant where the two rules disagree.
  2c. **Module 4 — the SECOND GATE. ✅ done (locked 2026-08-05).** Because rule 1 only fails on one side, a second variable is needed on one side only. Four **virtual-graph** properties were screened — `role_diversity`, `role_edge_enrichment`, `original_retention`, `vg_edge_ratio` — measured off the **built** `psi` edgelists (K=10, seed 42), no retraining. They need no labels, so they also cover `ogbl_ddi` where rule 1 cannot be evaluated at all.

  **THE GATE (locked, `frozen_rules.FROZEN_GATE`): inside the low-homophily zone, augment when `original_retention` < ~0.012** (interval **(0.0062, 0.0176)**) — ρ −0.86, LODO 0.77, leave-one-out **6/7 vs a 0.714 majority baseline**; also credible on the bias-free fixed-variant target and standalone over all 11 decided cells, and it reproduces on the `centrality` role graph. Direction is counter-intuitive and worth reporting: **the less of the original graph the rewiring keeps, the more augmentation helps.** The other three failed (`role_diversity` +0.29, `role_edge_enrichment` −0.39, `vg_edge_ratio` +0.64 with LOO 0.00). Collinearity was **measured**, not assumed — no pair reaches \|ρ\| ≥ 0.99.

  Two caveats that must travel with the gate. (i) It **does not separate the decisive pair** — it fixes `minesweeper` and breaks `squirrel_filtered`, so it moves the zone from 2 errors to 1 without solving the case that motivated it (emitted mechanically as `separates_decisive_pair`, never as prose). (ii) It is **fitted data**: `GATE_PANEL` = the discovery panel plus the nine Module-3 datasets, which are no longer unseen. Nothing here is validated.

  The combined call is `frozen_rules.predict_gated()`: **rule 1 vetoes on its reliable side → the gate decides inside the low-homophily zone → rule 2 covers unlabelled graphs.**
  2d. **Module 5 — test the gate. 🔵 current.** A **third, genuinely unseen set** (`frozen_rules.GATE_HELDOUT`, empty until datasets are chosen). Order matters and differs from Module 3: **build the virtual graph → predict → train → score**, because the gate is a property of the rewiring. Building is not training, so the verdict is still pre-registered with the outcome unknown. `experiments/predict_gate.py` refuses any dataset in `GATE_PANEL` without `--allow-panel`. Priority target is still the untested quadrant: **heterophilous + fragmented**.
  3. Swap **GraphSAGE → GIN** and re-run, to see if its stronger isomorphism power helps.
- **Future work.** Learnable alpha (auto-blend original vs virtual; needs synthetic datasets); embeddings as LLM graph summaries. Do not start these yet.

---

## 5. Coding Rules (match I2V style)

- **Mirror the I2V codebase**: `argparse` CLI; `build_graph()` / `learn_embeddings()` / `main(args)` structure; a class holding the core method (cf. `identity2vec.Graph`).
- **Fewest functions possible.** Each short, single-purpose, self-explanatory name. No helper unless necessary.
- **One-line comments max.** Triple-quoted one-line docstrings as in I2V.
- Self-explanatory file names (e.g. `virgo/virtual_graph.py`, `virgo/encoders/sage.py`, `virgo/eval/linkpred.py`).
- Models expose **`train(epochs)`**, not `fit()`.
- Prefer the main script to call only functions defined in base/abstract classes.
- No new dependency without need. Reuse the existing env (numpy 1.26.4, networkx, gensim 4.3.3, scipy 1.12.0; add torch/torch-geometric for the GNN).

**Layout rule (restructured 2026-07-29).** Two code folders and one rule: **`virgo/` is imported, `experiments/` is run.**
- `virgo/` — `config.py` (THE settings), `graph_io.py` (THE graph policy), `identity2vec*.py`, `virtual_graph.py`, `utils.py`, plus `encoders/`, `data/`, `eval/`.
- `experiments/` — every `argparse` entry point (`run_core`, `run_ogb`, `characterize`, `gate_rules`, `predict_module3`, `score_module3`, `predict_gate`, `train`, `train_encoder`, `benchmark_baselines`, `run_task`, `plot_emb`). No method code here.
- **The fit/predict split is load-bearing**: `characterize.py` and `gate_rules.py` FIT (they call `threshold()`) and are panel-guarded; `predict_module3.py` / `score_module3.py` / `predict_gate.py` only ever READ `virgo/frozen_rules.py`. Never let a predict script import a fitting function, or the validation becomes circular.
- `third_party/struc2vec/` — vendored, used as published.
- Library modules with a CLI run as `python -m virgo.<module>` from the repo root; entry points run as `python experiments/<script>.py`.
- **Adding an encoder** (the extension point): new `virgo/encoders/<name>.py` subclassing `GNNEncoder` with one `build_convs(dims, agg)`, then one line in `ENCODERS` (`virgo/encoders/__init__.py`). Every driver, CLI and scoreboard row picks it up with no further edit. `--encoder all` stays the locked `graphsage_edge + deepwalk` pair, so a newly registered encoder never joins a sweep unless named.

---

## 6. Reproducibility (non-negotiable)

- Fixed `seed=42` everywhere (split, init, sampling).
- Never modify files in `input/`; write derived files alongside, outputs to `output/`.
- Log every run setting and deviation in `notes.md` (e.g. walk-length now 40, paper's 80 kept as a recorded deviation).
- A result is "reproduced" only when our metric is within ~±0.05 of the paper's.
- Ship splits, seeds, and eval scripts with the method.

---

## 7. Deliverables

1. ✅ Cached I2V — embeddings identical to the baseline, ~200× faster.
2. ✅ `virgo/virtual_graph.py` — five-variant top-K virtual-graph builder.
3. ✅ GraphSAGE encoder over the virtual graph (`virgo/encoders/sage.py`, on `base.GNNEncoder`); `gin.py` is wired and registered but has produced **no results yet**.
4. ✅ Eval scripts: node classification (F1), link prediction (AUC, leakage-free) — `virgo/eval/`.
5. ✅ The characterization table + rule: graph properties → when augmentation helps, across our datasets **plus small/medium OGB**. Tables in `results/characterization_*.csv` + `results/feature_usefulness.csv`; figures in `results/figures/`.
6. ✅ The **two-gate rule**: rule 1 (homophily veto) → the gate (`original_retention`, from the virtual graph) → rule 2 (unlabelled fallback). Frozen in `virgo/frozen_rules.py`; screen in `experiments/gate_rules.py`; tables `results/gate_candidates.csv`, `results/vg_characterization.csv`, `results/gate_collinearity.csv`. **Fitted, not validated** — Module 5 is the test.
7. 🔵 GIN results next to GraphSAGE.
8. 🔵 LoG paper draft (the thesis reuses it).
