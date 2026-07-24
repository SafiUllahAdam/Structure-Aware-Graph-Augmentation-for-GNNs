# ViRGo — Paper Log

**Purpose.** Curated source-of-truth for paper writing. Records only what a reviewer/author needs for the Methods, Results, and Discussion sections: design decisions and their rationale, method specifications, hyperparameters, quantitative results, findings, and deviations from prior work. Deliberately excludes repo housekeeping (folder renames, path fixes, tooling) — that lives in `docs/notes.md`, the operational lab notebook.

Maintained automatically: a new entry is added whenever something research-significant happens, dated, newest at the bottom. Every result records its seed(s), dataset, encoder, and any caveat that limits its interpretation.

---

## Contribution & research question

- **Research question.** Which virtual graph is best for given data and task, and does GNN message passing over a structural-similarity virtual graph beat walk+Skipgram on structural-identity embeddings? The **virtual-graph construction — not the encoder — is the variable under study.**
- **1st contribution (virtual-graph study).** A virtual graph connects nodes by structural role (hub↔hub, bridge↔bridge), not by original edges, so role-similar but distant nodes can exchange messages. We test which construction serves each downstream task (node classification, link prediction, later anomaly detection). Identity2Vec's (I2V, Oluigbo et al.) Poisson/KL similarity graph is treated as one *generic* construction to test against simpler ones (degree-only, centrality-only).
- **Technical contribution (GNN encoder).** Replace I2V's guided walk + Skipgram with a modern inductive GNN (GraphSAGE primary, GIN expressive alternative, GAT ablation) aggregating directly over the virtual graph.
- **Stretch (2nd contribution, not started).** Reuse compact ViRGo embeddings as a structural summary of a large graph so its structure fits an LLM context window.
- **Scope guard.** Baselines (DeepWalk / node2vec / struc2vec) are used as published/default and **not fine-tuned** — the contribution is the method, not baseline tuning. Non-Euclidean/hyperbolic latent space is out of scope (reserved for a second paper).

---

## Phase 1 — Reproducibility (DONE)

- **Cached I2V.** I2V recomputes the structural signal (degree + eigenvector centrality) inside the walk loop — per neighbor, per step, per walk — which dominates cost. For a static graph these are constant, so we compute them once and cache. Result: embeddings **byte-identical** to the original, **~200× faster** (Deliverable #1).
- **Reproduction bar.** A metric counts as reproduced only within **±0.05** of the paper. Cora I2V lands in the paper's range on both tasks under a 3-seed harness (seeds 42/43/44).
- **Hyperparameters (I2V, `I2V_PARAMS`).** dimensions=64, walk_length=40, num_walks=10, window_size=10, epochs=1, sg=1 (Skipgram), e=2.7182. **DEVIATION:** paper text states walk_length=80; we use the repo default 40 (matches the author's released `cora.emb`), recorded as a deliberate deviation.
- **Evaluation protocol.** Node classification = one-vs-rest logistic regression on embeddings, weighted F1, stratified split. Link prediction = 70:30 edge split, embedding retrained on the 70% train graph only (no leakage), test edges vs sampled non-edges ranked by **cosine similarity** of embeddings → test AUC (unsupervised, I2V-paper-faithful; this is the protocol every recorded LP number uses). A supervised alternative (Hadamard edge features → logistic regression) exists in `eval_linkpred.py` as an optional robustness check only — an earlier version of this entry wrongly named it as the main protocol.
- **FINDING (baseline comparison).** On homophilous Cora, proximity methods beat structural ones: node2vec (NC ≈0.82 / LP ≈0.91) > I2V (NC ≈0.69 / LP ≈0.80). This is expected — proximity embeddings suit community/homophily labels — and indicates the original paper's "I2V beats all" reflects under-tuned baselines. I2V's advantage is on **structural** tasks. This motivates studying the virtual graph rather than the encoder alone.

---

## Phase 2 — Virtual-graph construction (CORE STUDY, in progress)

### Method

- **Virtual graph.** For a chosen structural signature, connect each node to its **top-K** most structurally similar nodes (K-nearest on the per-node signature). Undirected union, no self-loops, all original nodes kept (isolated nodes preserved). Edge weight = `1/(1+distance) ∈ (0,1]`, always finite. Build is **deterministic** (byte-identical rebuild).
- **Signature = variable under study.** Three variants (`--sim`, pluggable):
  - `psi` — I2V's KL→Poisson structural score used as a **reference-free per-node signature**. I2V's Ψ is walk-contextual (needs a reference node + shortest-path); we lift it to an all-pairs virtual graph by dropping the walk shortest-path factor (use q = Ω eigenvector centrality instead of Ω·pathlen), keeping the exact KL rate λ, the Fix-4A normalizer, and the Fix-8 log-Poisson score. **DESIGN DECISION:** rejected the alternative of reusing I2V's pairwise `identity_score` literally (asymmetric, slower, off-walk).
  - `degree` — degree-only signature (simplest structural baseline).
  - `centrality` — eigenvector-centrality-only signature.
- **Sweep design.** K ∈ {5, 10, 20} (sparsity vs over-smoothing), seeds {42, 43, 44}. Same K and same seeds across all variants = fair comparison ("which graph is best?").
- **Fixed-encoder protocol.** Only the graph changes; the encoder is held fixed so any performance difference is attributable to the construction. Current bridge encoder = DeepWalk with I2V_PARAMS (Phase-3 GNN will replace it and will additionally use the edge weights, which DeepWalk ignores). Link prediction is leakage-free: the virtual graph is rebuilt from the 70% train edges only.
- **Graph-health table.** Every built graph logs one row (`dataset, sim, K, nodes, edges, avg_degree, components, isolates`) so a weak score can be traced to a degenerate graph (too sparse, disconnected, isolates, too dense); doubles as an ablation-quality table for the paper.

### First results — Cora, K=10, DeepWalk bridge, 3 seeds (INDICATIVE, not final)

| variant | node-class weighted-F1 | link-pred AUC |
|---|---|---|
| centrality | **0.381 ± 0.007** | 0.551 ± 0.008 |
| psi (I2V KL/Poisson) | 0.234 ± 0.005 | 0.511 ± 0.006 |
| degree | 0.152 ± 0.007 | **0.555 ± 0.002** |

Graph sizes (Cora, K=10): psi 16251 edges (avg deg 12.0), degree 26216 (19.4 — integer-degree ties inflate the K-NN union), centrality 16110 (11.9).

- **FINDING (indicative).** The best virtual graph is **task-dependent**: centrality wins node classification, degree wins link prediction, and I2V's `psi` is not best on Cora under the DeepWalk bridge. Directly supports the Phase-2 question.
- **CAVEATS (why not final).** (a) DeepWalk bridge, not the Phase-3 GNN — DeepWalk ignores the edge weights that `psi`/`centrality` carry, so the GNN may reorder these. (b) Cora is homophily/community-labelled, so structural embeddings score modestly on NC and cosine LP AUC sits near 0.55. (c) Single dataset, K=10 only.

### Planned (deferred by design, to run after Phase 3)

- Full K sweep (5/10/20) scored on all variants.
- **Four datasets total** for the published comparison (Cora + three more) — not Cora alone.
- Phase-3 GNN encoders (GraphSAGE / GIN / GAT) over the same virtual graphs, replacing the DeepWalk bridge, then re-run the full comparison.

---

## Phase 3 — GNN encoder (design locked 2026-07-04, implementation pending)

- **DECISION (user + professor).** Encoder = **GraphSAGE trained unsupervised** with a **Skipgram-analog objective** — same role Skipgram plays in I2V, so the GNN is a true drop-in for the walk+Skipgram back end. Study runs **on the virtual graphs** (starting with I2V's Poisson/KL Ψ graph); the original graph is only a control row. Output stays 64-dim `.emb` → identical evaluation protocol as I2V/Phase 2.
- **Core architecture (ViRGo-SAGE).** Structural input features `[degree, eigenvector centrality Ω, ψ, clustering]` (no node attributes → method stays structural + inductive) → 2-layer GraphSAGE (mean) over the virtual graph → 64-d z. Loss = GraphSAGE unsupervised objective (Skipgram with the lookup table replaced by the GNN): positives = walk co-occurrence on the *virtual* graph, negatives ∝ deg^{3/4}, Q=5 (matches I2V `negative=5`).
- **Comparability design.** Walk corpus for positives uses the exact I2V params of the Phase-2 DeepWalk bridge (num_walks=10, walk_length=40, window=10, seeds 42/43/44) ⇒ Phase-2 bridge vs Phase-3 SAGE differ in **one component only** (Skipgram lookup vs message passing) — a clean encoder ablation.
- **Variant axes for the study/ablations:** A positives (walk co-occurrence vs 1-hop virtual neighbors — "are walks still needed once similarity is explicit?"), B aggregation (mean / **Ψ-weighted mean** — first use of the virtual edge weights / max / sum), C depth 1–3 (over-smoothing: virtual graphs near-connected at K=10), D features (structural-4 / degree-only / random), E graph (Ψ, degree, centrality × K 5/10/20; dual virtual+original branch deferred to follow-up work).
- **Run plan:** Stage 1 lock encoder on Cora Ψ K=10 (A×B, 4 configs, 3 seeds) → Stage 2 full "which virtual graph?" matrix under the GNN → Stage 3 depth/feature ablations + original-graph control → Phase 4 remaining 3 datasets + anomaly detection. Full design: `docs/phase3_gnn_design.md`.
- **Implementation (2026-07-04).** Spine implemented as designed (`encoder.py` `SageEncoder` + `notebooks/3-phase3_gnn_encoder.ipynb`). Method details now fixed in code: input features are computed on the **original** graph (the structural-identity signal) while message passing runs on the **virtual** graph; loss positives come from the same seeded, unweighted walk generator the Phase-2 bridge used (num_walks=10, length=40, window=10); Q=5 negatives ∝ deg^0.75; training CPU-only with seeded RNGs for reproducibility; LP reuses the identical Phase-2 splits so the encoder comparison is split-for-split fair. First SAGE results pending the notebook run.

### First encoder head-to-head — Enzymes, Ψ virtual graph, K=10, seeds 42/43/44 (2026-07-06, INDICATIVE)

First full run of the spine (single-seed pass earlier the same day, superseded by this 3-seed result). Same virtual graph (Ψ, K=10), same walk corpus, same leakage-free LP splits (`enzymes_vglp_s{42,43,44}`), cosine LP scoring — **only the encoder differs**:

| task | DeepWalk bridge (Phase 2) | ViRGo-SAGE (Phase 3) | Δ |
|---|---|---|---|
| node classification (weighted F1) | 0.4971 | **0.5405 ± 0.0014** | +0.043 |
| link prediction (AUC) | 0.5110 ± 0.0297 | **0.6632 ± 0.0114** | **+0.152** |

Graph health (enzymes, Ψ, K=10): 19,474 nodes, 111,401 edges, avg degree 11.44, 10 components, 0 isolates. Training loss falls ~13–18 → ~4.1 over 50 epochs across seeds. Enzymes labels: 106 labelled nodes absent from the edgelist are skipped identically for both encoders (n=19,474, 3 classes).

- **FINDING (indicative).** Message passing over the virtual graph beats the Skipgram lookup on both tasks and across all 3 seeds; the LP gain is large (+0.152 AUC, from near-random 0.51 to 0.66) and exceeds the seed spread by an order of magnitude. Supports the technical contribution: GNN > walk+Skipgram on the *same* virtual graph.
- **CAVEATS.** (a) Single dataset (enzymes), single variant/K (Ψ, K=10) — spine verification, not the study. (b) SAGE additionally consumes 4 structural input features from the original graph (degree, Ω, ψ, clustering), which the lookup-table bridge cannot use by construction — deliberate design (that *is* the GNN's advantage), but the D-axis feature ablation (degree-only / random features) is what will isolate feature signal from message-passing signal. (c) Cosine-scored LP, as in Phase 2.
- **Ablation A implemented (2026-07-06).** `positives` knob on the ViRGo-SAGE loss: **A1 `walk`** = window-10 co-occurrence on virtual-graph walks (default; identical to the Phase-2 bridge corpus ⇒ fair Skipgram-vs-GNN comparison), **A2 `edge`** = the virtual edges themselves as positive pairs (both directions, no walks) — tests whether walks are still needed once structural similarity is explicit in the graph. Everything else (negatives ∝ deg^0.75 Q=5, loss, caps, seeds) unchanged, so A1-vs-A2 is a one-component comparison. A2 artifacts tagged `sage_edge_*` / scoreboard encoder `graphsage_edge`. Results below.

### Ablation A results — A1 walks vs A2 direct edges — Enzymes, Ψ, K=10, seeds 42/43/44 (2026-07-06, INDICATIVE)

Same graph, features, architecture, negatives, splits, seeds — **only the positive-pair source differs** (A2 corpus: 222,802 edge pairs, under the 2M cap; both schemes sample 100k pairs/epoch):

| task | DeepWalk bridge (Phase 2) | SAGE A1 (walk) | SAGE A2 (edge) |
|---|---|---|---|
| node classification (weighted F1) | 0.4971 ± 0.0057 | 0.5405 ± 0.0014 | **0.5413 ± 0.0011** |
| link prediction (AUC) | 0.5110 ± 0.0297 | 0.6632 ± 0.0114 | **0.6909 ± 0.0156** |

- **FINDING 1.** The GNN beats the bridge under *both* objectives (NC +0.044, LP +0.15–0.18) — the encoder win is robust to the positive-pair scheme, not an artifact of the walk corpus.
- **FINDING 2 (the A ablation).** NC: statistical tie (Δ 0.0008, within seed noise). LP: A2 +0.028 over A1 (≈2σ vs seed stds) — likely real, borderline at 3 seeds. **Walks are not needed once structural similarity is explicit in the graph**: I2V ran walks to *find* role-similar nodes; the virtual graph already lists them, so direct edges suffice and even help LP. A2 is also cheaper (no walk generation).
- **DECISION.** A2 (direct-edge positives) becomes the default training objective for the remaining ablations/study; A1 is kept and reported as the bridge-comparable configuration — the "Phase-2 vs Phase-3 differ in one component only" claim holds only under A1.
- **DEVIATION.** Design doc planned Stage 1 on Cora; this first A run used enzymes. Cora repeat pending (cheap — cora bridge embeddings ready at 3 seeds).
- **CAVEATS.** Single dataset (enzymes), single graph variant/K (Ψ, 10), 3 seeds, cosine LP; loss *values* are not comparable across A1/A2 (different positive distributions) — only the downstream metrics are; feature-vs-message-passing attribution still waits on the D-axis ablation.

### Ablation B — neighbor aggregation over the virtual graph — Enzymes, Ψ, K=10, seeds 42/43/44 (2026-07-07)

- **Question.** Virtual edges carry a similarity *strength* (weight `1/(1+dist)` from Ψ/degree/centrality distance); mean aggregation ignores it. Does the encoder need only edge *existence* (which A2 already exploits), or also edge *strength*?
- **Variants (`agg` knob).** `mean` = equal neighbors (baseline; = the existing A2 rows, no new runs). `weighted` = per-target-normalized Ψ-weighted mean — the first use of the virtual edge weights anywhere in the encoder. `sum` = keeps neighborhood-magnitude signal. `max` = most-similar neighbor dominates.
- **Method detail.** `mean/sum/max` via native `SAGEConv(aggr=…)`; `weighted` via `GraphConv(aggr='add')` — identical root-plus-neighbor form (W₁xᵢ + W₂·agg xⱼ) but edge-weight-aware, weights row-normalized per target so the aggregate is a weighted mean; parameter count unchanged ⇒ one-component ablation. All else frozen at the A winner (edge positives), enzymes Ψ K=10, seeds 42/43/44.
- **Hypothesis.** If `weighted` > `mean`, similarity *magnitude* is informative beyond top-K ranking — strengthening the claim that the virtual-graph construction (not just its topology) matters; it would also damp sensitivity to K (weak tail auto-downweighted), informing the Stage-2 K sweep.
- Artifacts tagged `sage_edge_<agg>_*` / scoreboard `graphsage_edge_<agg>` — mean keeps the unsuffixed names.

Same graph, features, architecture, negatives, splits, seeds — positives fixed at **edge (A2)**, **only aggregation differs**:

| aggregation | node classification (weighted F1) | link prediction (AUC) |
|---|---|---|
| **mean** | **0.5413 ± 0.0011** | **0.6909 ± 0.0156** |
| weighted (Ψ) | 0.5399 ± 0.0020 | 0.6528 ± 0.0240 |
| max | 0.5284 ± 0.0041 | 0.6378 ± 0.0353 |
| sum | 0.5114 ± 0.0017 | 0.5063 ± 0.0596 |

- **FINDING.** Ranking **mean > weighted > max > sum is identical on both tasks** — a robust, task-independent ordering. `mean` wins outright.
- **Hypothesis NOT supported.** Ψ-`weighted` (first use of the virtual edge weights anywhere in the encoder) was predicted to beat `mean` if similarity *magnitude* was informative; it **lost** (LP −0.038, NC tie). Conclusion: within a node's top-K, **edge existence carries the signal, not edge strength** — the `1/(1+dist)` weights add nothing beyond top-K membership. `sum` is degenerate (unnormalized → aggregate magnitude tracks degree → LP collapses to chance 0.51 with ×4 variance); `max` discards the neighbor distribution. Standard GraphSAGE `mean` confirmed best.
- **DECISION (A + B finalized).** ViRGo-SAGE default objective/encoder = **edge positives (A2) + mean aggregation (B)**. Both locked as defaults in `encoder.py` / `GNN_PARAMS`. All four aggregation variants retained in `scoreboard.csv` as the ablation record (not deleted — they are the evidence for the choice).
- **CAVEATS.** Enzymes / Ψ / K=10 / 3 seeds only; absolute NC ≈0.54 sits near the balanced-class floor and LP ≈0.69 — aggregation is an encoder-internal knob and cannot raise the ceiling set by the features + virtual graph (that is the E-axis / dataset job). Cora repeat pending.

### Ablation B results — aggregation — Enzymes, Ψ, K=10, edge positives, seeds 42/43/44 (2026-07-07, INDICATIVE)

| agg | NC weighted F1 | LP AUC |
|---|---|---|
| **mean** | **0.5413 ± 0.0011** | **0.6909 ± 0.0156** |
| weighted (Ψ-mean) | 0.5399 ± 0.0020 | 0.6528 ± 0.0240 |
| max | 0.5284 ± 0.0041 | 0.6378 ± 0.0353 |
| sum | 0.5114 ± 0.0017 | 0.5063 ± 0.0596 |

- **FINDING.** Plain mean wins both tasks. `weighted` ≈ mean on NC but −0.038 LP: the Ψ weight *magnitudes* add no information beyond the top-K *ranking* itself — the graph's topology already encodes the similarity signal. `sum` collapses LP to near-random (magnitude varies with virtual degree → scale noise); `max` loses too much (one neighbor can't summarize a role neighborhood).
- **DECISION.** `agg = "mean"` locked as default (already was). Encoder now fully locked for Stage 2: **edge positives + mean aggregation**.
- **CAVEATS.** Same as ablation A: single dataset (enzymes), single variant/K (Ψ, 10), 3 seeds, cosine LP.

### Ablation C results — encoder depth / over-smoothing — Enzymes, Ψ, K=10, edge positives, mean agg, seeds 42/43/44 (logged 2026-07-16)

Design axis from the Phase-3 variant list ("C depth 1-3 — over-smoothing: virtual graphs near-connected at K=10"). Same graph, features, positives, aggregation, negatives, splits, seeds — **only the number of message-passing layers differs**. Encoder names `graphsage_edge_l1` / `graphsage_edge_l3`; depth 2 is the locked default (`graphsage_edge`, `GNN_PARAMS["layers"]=2`), re-scored from its existing embeddings, not retrained.

| depth | NC (weighted F1) | LP (AUC) |
|---|---|---|
| 1 layer | 0.4971 ± 0.0018 | 0.6678 ± 0.0075 |
| **2 layers (default)** | **0.5413 ± 0.0011** | **0.6909 ± 0.0156** |
| 3 layers | 0.4506 ± 0.0187 | 0.4983 ± 0.0407 |

- **FINDING — the over-smoothing prediction is confirmed, and it is sharp.** Depth 2 wins both tasks. Depth 3 does not merely degrade: LP falls to **0.4983, i.e. chance**, and NC drops below the DeepWalk bridge (0.4506 vs 0.4971). The design's stated mechanism holds — the Ψ virtual graph is near-connected at K=10 (10 components, avg degree 11.44 on enzymes), so a 3-hop receptive field averages a large fraction of the graph and node embeddings converge. Depth 1 is under-powered on NC (0.4971, exactly at the bridge) while retaining most of the LP signal (0.6678), consistent with LP depending mostly on 1-hop role neighbourhoods.
- **DECISION.** Depth **2** confirmed as the ViRGo-SAGE default. No code change — the default was already 2; this ablation converts an assumption into evidence and supplies the over-smoothing curve for the paper.
- **CAVEATS.** enzymes only, Ψ graph only, K=10, 3 seeds, cosine LP; depth not swept at other K (a denser graph at K=20 should over-smooth *earlier* — untested). Like ablations A/B/D, the choice is made on a single dataset and inherited by cora, where the E-study story inverts; a cora depth repeat is not run. Depth 3's LP std (0.0407) is the largest in the C set, as expected once embeddings collapse toward a common vector.

### E-study first results — virtual-graph variants, locked encoder — Enzymes, K=10, seeds 42/43/44 (2026-07-07, INDICATIVE)

Locked ViRGo-SAGE (edge positives + mean agg, ablations A/B) run over all three virtual-graph constructions at K=10 (notebook 3 §8 sweep; user-run):

| virtual graph | NC weighted F1 | LP AUC |
|---|---|---|
| Ψ (I2V Poisson/KL) | 0.5413 ± 0.0011 | 0.6909 ± 0.0156 |
| degree-only | **0.5536 ± 0.0012** | 0.5998 ± 0.0782 |
| centrality-only | 0.5411 ± 0.0025 | **0.7197 ± 0.0625** |

- **FINDING (core research question).** No single virtual graph wins both tasks — and Ψ, the generic I2V construction, is best on **neither**: degree-only wins node classification (+0.012 over Ψ, tight std), centrality-only wins link prediction (+0.029 over Ψ). This directly supports the paper's thesis that the virtual-graph construction should be chosen **per data and task** rather than defaulting to the Poisson/KL similarity graph.
- **Ψ's role.** Ψ is the consistent all-rounder (2nd on both tasks, and by far the lowest LP variance: ±0.016 vs ±0.06–0.08 for the single-signal graphs) — a defensible generic default, but beatable by cheaper single-signal graphs when the task is known.
- **CAVEATS.** Enzymes only, K=10 only, 3 seeds; the single-signal winners carry 4–5× the LP std of Ψ, so the LP ordering (centrality > Ψ) is within ~½σ — needs the K=5/20 sweep and a 2nd dataset before it hardens. Deepwalk-bridge + walk-positive GraphSAGE rows for degree/centrality pending (fills the 3×3 encoder×graph grid).

- **Ablation D implemented (2026-07-09) — features vs message passing.** `feats` knob on `SageEncoder` selects the GNN's input features, holding the virtual graph, positives (edge), aggregation (mean), seeds (42/43/44) and LP splits fixed: **D0 `all`** = [degree, Ω, ψ, clustering] (the locked encoder — existing embeddings reused, re-scored identical to the recorded 0.5413 NC / 0.6909 LP), **D1 `degree`**, **D2 `deg_cent`**, **D3 `psi`**, **D4 `random`** (seeded Gaussian; node identity only, zero structural signal), **D5 `const`** (identical rows → z-norm zeros → all-nodes-identical embeddings → cosine LP AUC ≈ 0.50 by construction; a floor, not an experiment). Motivation: ViRGo-SAGE beats the Phase-2 DeepWalk bridge with **two** advantages at once — message passing *and* four structural input features the lookup-table bridge cannot consume — so "GraphSAGE > DeepWalk" currently reads "GNN + structural features > walks". **The deciding row is D4 vs deepwalk:** random-feature SAGE above the bridge ⇒ message passing over the virtual graph carries signal; at the bridge ⇒ the Phase-3 win was the features. LP features are computed on the 70% train graph (no leakage). D3 is **confounded by design** (ψ is both the feature and the signal the ψ graph was built from) and must be reported as such. Artifacts `graphsage_edge_feat_<set>_s<seed>.emb`; scoreboard encoder names likewise; snapshot `results/snapshots/<ds>_feature_ablation_K<K>.csv`. Expected on enzymes: D1 ≈ D2, because eigenvector centrality is degenerate there (96.9% of nodes < 1e-6 — 640 disconnected molecules) — a dataset property to report, not a failure. Results pending the notebook run.

- **Two new virtual-graph variants added to the E-study (2026-07-10) — `original` and `hybrid`.** Motivation: the E-table so far compares three *rewired* graphs (Ψ, degree, centrality) against each other, but not against the obvious control — the untouched input graph — so "virtual graph helps" is not yet evidenced within the locked-encoder grid. **`original`** = exact copy of the input graph, all edge weights 1.0, K unused (duplicated under each k<K>/ folder so sweeps stay uniform): GraphSAGE on the real edges, isolating the virtual-graph effect from the encoder effect. **`hybrid`** = union of original edges (weight 1.0) and Ψ top-K role edges (weight 1/(1+d)); on overlap the original edge's 1.0 wins. Rationale: message passing then reaches both physical neighbors and role neighbors in one graph — tests "structure + role beats either alone". Both flow through the existing pipeline unchanged (`virtual_graph.py --sim`, `encoder.py --sim`, `VG_SIMS`); for link prediction the input graph is the 70% train graph as before, so `original`/`hybrid` inherit the no-leakage protocol. Results pending.

### E-study extended — `original` + `hybrid` rows — Enzymes, K=10, locked encoder, seeds 42/43/44 (2026-07-10, INDICATIVE)

| virtual graph | NC weighted F1 | LP AUC |
|---|---|---|
| Ψ (I2V Poisson/KL) | 0.5413 ± 0.0011 | 0.6909 ± 0.0156 |
| degree-only | 0.5536 ± 0.0012 | 0.5998 ± 0.0782 |
| centrality-only | 0.5411 ± 0.0025 | **0.7197 ± 0.0625** |
| **original (control)** | **0.5671 ± 0.0022** | 0.6459 ± 0.0364 |
| hybrid (orig ∪ Ψ) | 0.5485 ± 0.0044 | 0.6967 ± 0.0302 |

- **FINDING.** The `original`-graph control **wins node classification outright** (0.5671, +0.014 over degree-only, +0.026 over Ψ, tightest std after Ψ): on enzymes, GraphSAGE over the real edges beats every rewired virtual graph for NC — the virtual-graph rewiring *costs* NC accuracy here. On LP the control drops to 4th (0.6459); role edges matter for predicting links.
- **hybrid.** 2nd on LP (0.6967, within noise of Ψ 0.6909, behind centrality 0.7197) and 2nd on NC (0.5485) — the only variant top-2 on **both** tasks; consistent all-rounder, mirrors Ψ's old role but stronger.
- **Implication for the thesis.** Sharpens, not breaks, the per-task story: NC on enzymes favors original edges (class signal is homophilous/local), LP favors role-augmented graphs (centrality, hybrid, Ψ). "Which graph is best" now provably includes "sometimes the original one" — the control row makes the virtual-graph claims falsifiable.
- **CAVEATS.** Enzymes only, K=10 only, 3 seeds, cosine LP; deepwalk-bridge rows for original/hybrid missing (notebook 2 not rerun — GraphSAGE rows unaffected). LP gaps between centrality/hybrid/Ψ are within ~½σ.

- **DeepWalk bridge rows for `original`/`hybrid` filled (2026-07-10)** — trained with the exact notebook-2 recipe (same DeepWalk params, splits, seeds 42/43/44), enzymes K=10: **original** deepwalk NC 0.5086 ± 0.0039 / LP 0.6574 ± 0.0170; **hybrid** deepwalk NC 0.4906 ± 0.0060 / LP 0.5087 ± 0.0246. Removes the "bridge rows missing" caveat from the E-study-extended entry. Notable: on the **original** graph DeepWalk edges out GraphSAGE on LP (0.6574 vs 0.6459, within ~½σ) — the GNN's LP advantage appears only on role-augmented graphs (Ψ +0.18, hybrid +0.19 over their bridges); on **hybrid**, DeepWalk collapses on LP (0.5087, near-random) while GraphSAGE holds 0.6967 — walks get lost mixing physical + role edges, message passing does not.

- **Cora extended to the 5-variant grid (2026-07-12)** — `original`/`hybrid` virtual graphs built for cora K=10 (`original`: 2708 nodes / 5278 edges = exact copy; `hybrid`: 21399 edges = original ∪ Ψ top-K), and the DeepWalk bridge rows filled with the exact notebook-2 recipe (same params, shared LP splits, seeds 42/43/44): **original** deepwalk NC **0.8100 ± 0.0158** / LP **0.9007 ± 0.0039**; **hybrid** deepwalk NC 0.4526 ± 0.0035 / LP 0.6699 ± 0.0142. Notable: cora is strongly homophilous — DeepWalk on the untouched graph is very strong (LP 0.90), so the original-graph control sets a high bar the virtual graphs must justify against; and the enzymes pattern repeats on NC — mixing role edges into the walks (**hybrid**) collapses DeepWalk's NC from 0.81 to 0.45 (walks get lost across physical + role edges). GraphSAGE rows for cora original/hybrid pending (notebook 3 §8 sweep).

### E-study on Cora — full 5-variant grid, K=10, locked encoder, seeds 42/43/44 (2026-07-12)

Second dataset for the E-study (first: enzymes). Same protocol: locked ViRGo-SAGE (`graphsage_edge` = edge positives + mean agg) vs the DeepWalk bridge, shared LP splits, cosine LP.

| virtual graph | deepwalk NC | graphsage NC | deepwalk LP | graphsage LP |
|---|---|---|---|---|
| Ψ (I2V Poisson/KL) | 0.2378 ± 0.0044 | 0.2757 ± 0.0010 | 0.5114 ± 0.0062 | 0.5246 ± 0.0173 |
| degree-only | 0.1592 ± 0.0118 | 0.2722 ± 0.0031 | 0.5547 ± 0.0018 | 0.5283 ± 0.0135 |
| centrality-only | 0.3830 ± 0.0151 | 0.3067 ± 0.0036 | 0.5507 ± 0.0081 | 0.5708 ± 0.0118 |
| **original (control)** | **0.8100 ± 0.0158** | 0.4504 ± 0.0104 | **0.9007 ± 0.0039** | 0.6212 ± 0.0080 |
| hybrid (orig ∪ Ψ) | 0.4526 ± 0.0035 | 0.3109 ± 0.0100 | 0.6699 ± 0.0142 | 0.5572 ± 0.0213 |

- **HEADLINE FINDING — the story inverts across datasets.** On cora, **DeepWalk on the untouched original graph wins both tasks by a landslide** (NC 0.81, LP 0.90); every rewired virtual graph collapses NC to ≤0.45 and LP to ≤0.67, and GraphSAGE loses to DeepWalk on the original graph on both tasks (0.45/0.62). Exactly opposite to enzymes, where the role-augmented graphs + message passing gave the best LP and GraphSAGE won everywhere. This is the strongest evidence yet for the paper's thesis: **which graph (and encoder) is best is a property of the data** — cora's citation labels are homophilous/community-like, so proximity walks on real edges are near-optimal and role-based rewiring destroys the signal; enzymes' molecular structure rewards role edges.
- **Why GraphSAGE trails on cora NC:** its input features are purely structural (degree, centrality, ψ, clustering) — role features, not community features. Cora classes are topics, not roles, so the GNN is handicapped by construction on this task; DeepWalk's positional embeddings capture community directly.
- **Consistent cross-dataset patterns (both datasets):** (1) hybrid is DeepWalk's best *rewired* graph on both tasks (it contains the original edges) yet mixing role edges still halves DeepWalk's cora NC (0.81 → 0.45); (2) GraphSAGE > DeepWalk on the pure role graphs' NC (Ψ, degree); (3) the GNN's LP advantage exists only on role-augmented graphs, never on the original graph.
- **CAVEATS.** K=10 only, 3 seeds, cosine LP; no encoder tuning per dataset (locked from enzymes ablations); structural-feature handicap above means "GraphSAGE loses on cora" = "role-feature GNN loses on a homophily task", not a general GNN result.

- **Result-presentation framing locked (2026-07-13, professor guidance).** PRIMARY research question: **which graph variant is best for a given dataset × task** (the virtual graph is the variable under study). SECONDARY: **within each graph**, does GraphSAGE (message passing) beat DeepWalk (walks + Skipgram) — the encoder comparison lives inside each graph, never headlines. Current K=10 answers: cora NC → original+DeepWalk (0.8100); cora LP → original+DeepWalk (0.9007); enzymes NC → original+GraphSAGE (0.5671); enzymes LP → centrality+DeepWalk (0.7382). Notable: the *best* configuration is DeepWalk-based in 3 of 4 cells — the GNN's contribution shows in the secondary Δ table (GraphSAGE > DeepWalk on Ψ everywhere, and on enzymes hybrid LP +0.188), not in the headline. Notebook 3 §11 renders both views from the scoreboard.

- **Headline tables locked to the fixed encoder (2026-07-13, user decision).** Notebook 3 §11 Table 1 (best virtual graph per dataset × task) and Table 2 (control vs original) now use **`graphsage_edge` only** — previously "best over both encoders", which confounded graph choice with encoder choice. Rationale: the virtual graph is the variable under study, so the encoder must be held fixed; the GraphSAGE-vs-DeepWalk comparison stays in §7 as the secondary question (same graph, encoders differ). K=10 answers under the locked encoder: cora NC → hybrid 0.3109; cora LP → centrality 0.5708; enzymes NC → degree 0.5536; enzymes LP → centrality 0.7197. Control check (graphsage both sides): original wins cora NC 0.4504 / cora LP 0.6212 and enzymes NC 0.5671; best virtual wins enzymes LP 0.7197 vs 0.6459 — per-data/per-task story unchanged, now unconfounded.

- **`original` repositioned as CONTROL baseline (2026-07-13, user decision).** The original graph stays in every sweep but is not a ViRGo contribution — headline tables now report the **best virtual/derived graph** (Ψ, degree, centrality, hybrid) per dataset × task, with a separate control-check table original-vs-best-virtual. K=10 answers: cora LP → hybrid+DeepWalk 0.6699; cora NC → hybrid+DeepWalk 0.4526; enzymes LP → centrality+DeepWalk 0.7382; enzymes NC → degree+GraphSAGE 0.5536. Control check: original much better on both cora tasks (0.9007/0.8100), virtual better on enzymes LP (0.7382 vs 0.6574), original slightly better on enzymes NC (0.5671 vs 0.5536). Honest headline: rewiring wins only enzymes LP at K=10 so far — the per-data/per-task story carries the paper, not a blanket "virtual graphs win".

- **ABLATION D RESULTS (2026-07-16) — the Phase-3 win is the FEATURES, not message passing.** enzymes, Ψ graph, K=10, seeds 42/43/44, edge positives + mean agg, cosine LP; only the GNN's input features change.

| id | features | dims | NC F1 | LP AUC |
|---|---|---|---|---|
| **D0** | all (degree, Ω, ψ, clustering) | 4 | **0.5413 ± 0.0011** | **0.6909 ± 0.0156** |
| D2 | degree + centrality | 2 | 0.5151 ± 0.0052 | 0.6805 ± 0.0302 |
| D1 | degree | 1 | 0.5203 ± 0.0014 | 0.5975 ± 0.0432 |
| *deepwalk (bridge reference)* | *none* | — | *0.4971 ± 0.0057* | *0.5110 ± 0.0297* |
| D5 | constant (floor) | 1 | 0.3277 ± 0.0000 | 0.5046 ± 0.0905 |
| D3 | ψ *(confounded)* | 1 | 0.4933 ± 0.0017 | 0.4983 ± 0.0251 |
| **D4** | **random (control)** | 4 | **0.4814 ± 0.0060** | **0.4798 ± 0.0349** |

- **DECIDING ROW ANSWERED — D4 lands BELOW the bridge on both tasks** (LP 0.4798 vs 0.5110; NC 0.4814 vs 0.4971), and its LP is indistinguishable from chance (0.48 ≈ 0.50, within its own 0.035 σ). Per the pre-registered reading in the ablation-D design entry, this is the "at the bridge ⇒ the Phase-3 win was the features" branch — in fact worse. **Message passing over the Ψ virtual graph, stripped of structural input features, carries no usable signal on enzymes: the structural features are NECESSARY.** ⚠️ **This does NOT establish that message passing contributes nothing.** The design's two-branch reading was under-specified: D4 tests only whether message passing is *sufficient alone*. Attributing the +0.18 LP win to the features requires the arm **structural features WITHOUT message passing** (D6, below), which is **not yet run**.
- **Revises the earlier indicative claim.** The Phase-3 entry "message passing over the virtual graph beats the Skipgram lookup … supports the technical contribution: GNN > walk+Skipgram on the *same* virtual graph" survives only in the weaker form **"GNN + structural features > walks"**. The isolated claim "message passing helps *by itself*" is **not supported** by this control. Report D4 alongside any GraphSAGE-vs-DeepWalk headline.
- **MISSING ARM — D6 `features, no message passing` (identified 2026-07-16, user challenge).** The ablation-D grid tests only two of four cells: *random features + MP* (D4 = 0.4798 LP) and *structural features + MP* (D0 = 0.6909 LP). The cell **structural features, no MP** is untested, so the split between "features" and "aggregation" is **not yet decidable**. D6 = feed the raw 4-dim [degree, Ω, ψ, clustering] vectors (computed on the original graph, z-normed, 70% train graph for LP) straight to the shared eval protocol — cosine for LP, one-vs-rest logistic regression for NC — no GNN, no training. Reading: **D6 ≈ D0** ⇒ message passing adds nothing and the Phase-3 win is purely the features; **D6 « D0** ⇒ message passing contributes the gap *on top of* the features, and the technical contribution stands in the form "MP amplifies structural features". Until D6 runs, the only defensible claims are: (1) structural features are **necessary** (D4 collapses without them); (2) message passing is **not sufficient alone**.
- **D3 (ψ alone) collapses to chance** (LP 0.4983 / NC 0.4933) — below the bridge — even though it is *confounded in ViRGo's favour* (ψ is both the feature and the signal the Ψ graph was built from). The I2V scalar is not a sufficient input feature.
- **Prediction partly wrong: D1 ≈ D2 held for NC, failed for LP.** The design entry expected D1 ≈ D2 because eigenvector centrality is degenerate on enzymes (96.9% of nodes < 1e-6, 640 disconnected molecules). NC matches (0.5203 vs 0.5151, D1 marginally higher), but LP shows a real gap (**D2 0.6805 vs D1 0.5975, +0.083**): the near-zero centrality column still carries link-predictive signal despite the degeneracy. Degree alone (D1) already beats the bridge on both tasks; degree+centrality (D2) recovers ~98% of full-feature LP (0.6805 / 0.6909), so ψ + clustering add little.
- **CAVEATS.** enzymes only, Ψ graph only, K=10, 3 seeds, cosine LP. The cora repeat is not run — given the cora inversion (DeepWalk+original wins by a landslide there), D4 on cora may read differently and must not be assumed. D5's NC 0.3277 is the majority-class floor by construction (identical rows → z-norm zeros), not an experiment. Notebook 3 §8b holds this ablation but its code cell is currently **commented out**; its markdown still points to the deleted §7b view.

---

## Ablation study — consolidated (paper-ready, 2026-07-16)

Single reference for the paper's ablation section. The dated entries above remain the provenance trail; this section is the writing surface. Every number below is read from `results/scoreboard.csv`.

### Common protocol

All of A–D: **enzymes**, **Ψ** virtual graph, **K=10**, seeds **42/43/44**, cosine link prediction, shared 70/30 splits. One axis changes per ablation; graph, features, architecture, negatives (∝ deg^0.75, Q=5), splits and seeds are otherwise identical, so each comparison is one-component. LP features/graphs are built from the 70% train edges only ⇒ no leakage. Reference bar throughout = the Phase-2 DeepWalk bridge: **NC 0.4971 ± 0.0057 · LP 0.5110 ± 0.0297**. Every variant is retained in the scoreboard as evidence; none are deleted.

**A–D purpose:** lock the encoder so it stops being a confound in E. **E purpose:** the research question itself.

### A — positives: are walks still needed? · DECIDED: `edge`

| id | positives | NC (F1) | LP (AUC) |
|---|---|---|---|
| A1 | walk co-occurrence (window 10, = bridge corpus) | 0.5405 ± 0.0014 | 0.6632 ± 0.0114 |
| **A2** | **direct virtual edges** | **0.5413 ± 0.0011** | **0.6909 ± 0.0156** |

NC is a tie (Δ 0.0008, within seed noise). LP favours direct edges (**Δ +0.0277**, ≈2σ vs the seed stds — likely real, borderline at 3 seeds). **Interpretation:** I2V ran walks to *discover* role-similar nodes; the virtual graph already enumerates them, so walks are redundant once similarity is explicit in the graph — and dropping them removes walk generation entirely (cheaper). **Cost of the choice:** the "Phase 2 and Phase 3 differ in exactly one component" claim holds only under A1, which is retained and reported as the bridge-comparable configuration.

### B — aggregation: does similarity *strength* matter? · DECIDED: `mean`

| id | aggregation | NC (F1) | LP (AUC) |
|---|---|---|---|
| **B0** | **mean (equal neighbors)** | **0.5413 ± 0.0011** | **0.6909 ± 0.0156** |
| B1 | Ψ-weighted mean *(the only variant reading edge weights)* | 0.5399 ± 0.0020 | 0.6528 ± 0.0240 |
| B2 | max | 0.5284 ± 0.0041 | 0.6378 ± 0.0353 |
| B3 | sum | 0.5114 ± 0.0017 | 0.5063 ± 0.0596 |

Plain `mean` wins both tasks. **The reportable finding:** `weighted` — the sole configuration that consumes the similarity strengths ViRGo computes (`1/(1+dist)`) — *loses* to discarding them (LP −0.038). **Edge existence carries the signal; edge strength does not.** Once top-K has selected the neighbourhood, how similar each member is adds nothing. `sum` collapses to chance on LP (0.5063): unnormalized neighbourhood magnitude swamps the embedding.

### C — depth: over-smoothing · DECIDED: `2 layers`

| id | depth | NC (F1) | LP (AUC) |
|---|---|---|---|
| C1 | 1 layer | 0.4971 ± 0.0018 | 0.6678 ± 0.0075 |
| **C2** | **2 layers** | **0.5413 ± 0.0011** | **0.6909 ± 0.0156** |
| C3 | 3 layers | 0.4506 ± 0.0187 | 0.4983 ± 0.0407 |

Depth 2 wins both. **The over-smoothing prediction is confirmed and sharp:** at depth 3 LP falls to **0.4983 = chance**, and NC drops below the bridge. Mechanism as designed — the Ψ graph is near-connected at K=10 (enzymes: 10 components, avg degree 11.44), so a 3-hop receptive field averages a large fraction of the graph and embeddings converge (C3 also carries the largest LP variance, 0.0407, consistent with collapse). Depth 1 is under-powered on NC (0.4971, exactly at the bridge) yet keeps most LP signal (0.6678) — LP depends mostly on 1-hop role neighbourhoods. No code change: 2 was already the default; this converts an assumption into evidence and supplies the depth curve.

### D — features vs message passing · PARTLY OPEN

| id | features | msg passing | dims | NC (F1) | LP (AUC) |
|---|---|---|---|---|---|
| **D0** | all four | ✔ | 4 | **0.5413 ± 0.0011** | **0.6909 ± 0.0156** |
| D2 | degree + centrality | ✔ | 2 | 0.5151 ± 0.0052 | 0.6805 ± 0.0302 |
| D1 | degree | ✔ | 1 | 0.5203 ± 0.0014 | 0.5975 ± 0.0432 |
| — | *deepwalk bridge (reference)* | ✘ | — | *0.4971 ± 0.0057* | *0.5110 ± 0.0297* |
| D3 | ψ *(confounded)* | ✔ | 1 | 0.4933 ± 0.0017 | 0.4983 ± 0.0251 |
| D4 | random **(control)** | ✔ | 4 | 0.4814 ± 0.0060 | 0.4798 ± 0.0349 |
| D5 | constant (floor) | ✔ | 1 | 0.3277 ± 0.0000 | 0.5046 ± 0.0905 |
| **D6** | **all four (control)** | **✘ `layers=0`** | 4 | **pending** | **pending** |

**Motivation.** ViRGo-SAGE beats the bridge with two advantages simultaneously — message passing *and* four structural features the lookup-table bridge cannot consume — so "GraphSAGE > DeepWalk" currently reads "GNN + structural features > walks". D isolates them.

**What D4 establishes.** Random-feature SAGE lands **below** the bridge on both tasks, its LP indistinguishable from chance (0.4798, σ 0.035). Therefore: **(1) the structural features are necessary; (2) message passing is not sufficient alone.** Both claims are supported.

**What D4 does NOT establish (correction, 2026-07-16).** It does not license "the win is *mostly* the features". The design's original two-branch reading of D4-vs-deepwalk was under-specified: the attribution needs the fourth cell of the 2×2 — *features without message passing* (**D6**). Decision rule, pre-registered: **D6 ≈ D0 ⇒ message passing adds nothing, the Phase-3 win is the features; D6 « D0 ⇒ message passing contributes on top of the features and the technical contribution stands.** Until D6 runs, only the two claims above may be stated.

**Secondary findings.** D3: the I2V ψ scalar alone collapses to chance (0.4983) despite being confounded *in ViRGo's favour* (ψ is both the feature and the signal the Ψ graph was built from) — ψ is not a sufficient input feature. D1: degree alone already clears the bridge on both tasks. D2: degree + centrality recovers **98% of D0's LP** (0.6805 vs 0.6909) — **ψ and clustering together add ≈0.01**. The prediction "D1 ≈ D2 because eigenvector centrality is degenerate on enzymes (96.9% of nodes < 1e-6, 640 disconnected molecules)" **holds for NC** (0.5203 vs 0.5151) but **fails for LP** (Δ +0.083): the near-degenerate centrality column still carries link-predictive signal. D5 NC 0.3277 is the majority-class floor by construction, not an experiment.

**D6 implementation (2026-07-16).** `layers=0` in `SageEncoder` ⇒ no convolutions; `forward()` returns the z-normed features unchanged (verified bit-identical to raw X), so D6 reuses **the identical feature builder as D0** and the only difference between the rows is message passing. Nothing is trained (`train()` asserts at `layers=0`); the virtual graph is unused. Artifacts `features_only_s<seed>.emb`, scoreboard encoder `features_only`; notebook 3 §8b, `RUN_D6` knob. **Reporting caveat:** D6 is 4-dimensional against D0's 64 — the contrast is "these features scored directly" vs "these features expanded and smoothed by a GNN", which is the intended question but is *not* dimension-matched; state this in the paper. D6 is deterministic across seeds, so its std reflects split variation only.

### E — which virtual graph? (the research question)

Locked encoder `graphsage_edge` (= A2 + B0 + C2 + D0) vs the DeepWalk bridge; K=10, seeds 42/43/44. Variants: `psi` (I2V Poisson/KL), `degree`, `centrality`, `original` (**control** — untouched input graph, not a ViRGo contribution), `hybrid` (original ∪ Ψ top-K).

**Enzymes, K=10**

| graph | DW NC | GS NC | DW LP | GS LP |
|---|---|---|---|---|
| psi | 0.4971 | 0.5413 | 0.5110 | **0.6909** |
| degree | 0.5136 | 0.5536 | 0.6240 | 0.5998 |
| centrality | 0.4903 | 0.5411 | **0.7382** | 0.7197 |
| original *(control)* | 0.5086 | **0.5671** | 0.6574 | 0.6459 |
| hybrid | 0.4906 | 0.5485 | 0.5087 | 0.6967 |

**Cora, K=10**

| graph | DW NC | GS NC | DW LP | GS LP |
|---|---|---|---|---|
| psi | 0.2378 | 0.2757 | 0.5114 | 0.5246 |
| degree | 0.1592 | 0.2722 | 0.5547 | 0.5283 |
| centrality | 0.3830 | 0.3067 | 0.5507 | 0.5708 |
| original *(control)* | **0.8100** | 0.4504 | **0.9007** | 0.6212 |
| hybrid | 0.4526 | 0.3109 | 0.6699 | 0.5572 |

**HEADLINE — the story inverts across datasets.** Enzymes rewards role-rewiring: centrality gives the best LP (0.7382), beating the real graph, and GraphSAGE wins nearly everywhere. Cora does the opposite: the untouched graph with DeepWalk wins both tasks by a landslide (0.8100 / 0.9007) and every rewired graph destroys the signal. **This is the thesis, evidenced: which graph is best is a property of the data, not a universal answer.**

**Why cora inverts.** Cora's labels are research topics — a *community* property carried by real citation edges, so proximity walks over them are near-optimal and role-rewiring discards exactly the needed signal. GraphSAGE is additionally handicapped by construction: its inputs are *role* features (degree, centrality, ψ, clustering), not topic features. Hence "GraphSAGE loses on cora" = "a role-feature GNN loses on a homophily task", **not** a general GNN result. Enzymes' molecular structure is precisely what roles describe.

**Consistent cross-dataset patterns.** (1) `hybrid` is DeepWalk's best *rewired* graph on both tasks (it contains the original edges), yet mixing role edges still halves cora NC (0.81 → 0.45). (2) GraphSAGE > DeepWalk on the pure role graphs' NC (Ψ, degree) on both datasets. (3) The GNN's LP advantage exists **only** on role-augmented graphs, never on the original graph (enzymes original: DW 0.6574 vs GS 0.6459). (4) On hybrid LP, DeepWalk collapses (0.5087) while GraphSAGE holds 0.6967 — walks get lost mixing physical and role edges; message passing does not.

**The honest headline.** Best configuration per dataset × task cell: **DeepWalk wins 3 of 4** (cora NC 0.8100, cora LP 0.9007, enzymes LP 0.7382); GraphSAGE takes only enzymes NC (0.5671) — and on the *control* graph. Under the locked encoder with `original` excluded as control, the best virtual graphs are: cora NC → hybrid 0.3109 · cora LP → centrality 0.5708 · enzymes NC → degree 0.5536 · enzymes LP → centrality 0.7197. Control check (GraphSAGE both sides): original wins cora NC (0.4504), cora LP (0.6212) and enzymes NC (0.5671); the best virtual graph wins **only enzymes LP** (0.7197 vs 0.6459). **Rewiring wins exactly one of four cells.** The per-data story carries the paper; a blanket "virtual graphs win" does not.

### Locked configuration — ViRGo-SAGE

| axis | locked to | evidence |
|---|---|---|
| A positives | `edge` | LP +0.028 over walks; NC tie; removes walk generation |
| B aggregation | `mean` | beat weighted / max / sum on both tasks; edge weights unhelpful |
| C depth | `2 layers` | 1 under-powered; 3 over-smooths to chance |
| D features | `all four` | best of every set tested; features shown necessary (D4) |
| K | `10` | only K with a complete grid on every dataset |

### Threats to validity — state these in the paper

1. **A, B, C and D were each decided on enzymes alone** (one dataset, Ψ only, K=10, 3 seeds); that locked encoder was then applied to cora, **where the entire E story inverts**. Nothing guarantees cora would select the same settings, and the cora repeat of A/B/C/D is **not run**. Any claim of the form "we selected X because it performed best" must carry "on enzymes".
2. **D6 not yet run** ⇒ the features-vs-message-passing attribution is undecided; the technical contribution currently stands only as "GNN + structural features > walks".
3. **D4 on cora not run.** Given the inversion, the random-feature control may read differently there; do not extrapolate.
4. **Scale:** 2 datasets, 3 seeds, K=10, cosine LP throughout; no per-dataset encoder tuning (locked from enzymes); baselines used as published, not fine-tuned (scope guard).
5. **A2 side-effect:** the one-component Phase-2-vs-Phase-3 comparison is only valid under A1.

## 2026-07-17 — Proteins added as a third study dataset (provenance + properties)

**Source and verification.** Proteins (networkrepository `PROTEINS`, nrvis.com/download/data/labeled/PROTEINS.zip). Graph and node labels are taken from the **same** archive member set (`proteins.edges` + `proteins.node_labels`), so node ids align by construction. The repo's pre-existing `input/proteins.edgelist` is **md5-identical** to the archive's `proteins.edges`; the rebuilt graph was checked against it at **edge overlap = 1.0000** before any label file was written. No labels were inferred or fabricated (same policy that keeps politics link-prediction-only).

**Graph properties** (`input/proteins_nr.edgelist`): **43,466 nodes / 81,044 edges**, mean degree 3.73, max degree 25, no self-loops, **1,195 connected components, largest component 620 nodes** — a disjoint union of many small protein graphs, i.e. structurally the *same family as enzymes* (19,474 nodes / 640 components), at ~2.2× the scale.

**Labels:** 3 classes, heavily imbalanced — 21,151 / 20,931 / 1,389 (the minority class is 3.2% of nodes). 43,471 label lines vs 43,466 graph nodes: 5 ids carry a label but no edge; `eval_nodeclass.evaluate` intersects on nodes present in the embedding, so they drop out. Enzymes has the identical situation (19,580 labels vs 19,474 nodes) — no new handling.
- *Implication for reporting:* with a 3.2% minority class, **weighted F1 is majority-dominated**; a trivial two-class predictor already scores well. Report macro F1 alongside weighted for proteins, or the NC numbers will overstate.

**Why this dataset earns its place in the study.** The E-story so far rests on two datasets that disagree (enzymes: role graphs + GNN win; cora: DeepWalk + original graph landslide), which makes "it depends on the data" a two-point claim. Proteins is a *same-family replicate of enzymes* — if role graphs win on proteins too, the enzymes result generalizes to the graph family rather than to one dataset, which is the weakest link in the current per-data thesis.

**Expected caveat, recorded before the run** (so it is not a post-hoc excuse): eigenvector centrality is degenerate on proteins — **76.9% of nodes < 1e-6**, driven by the 1,195 components. Enzymes is worse (96.9%). The `centrality` variant is therefore built on a near-constant signature on both, and any `centrality` result on proteins should be read as such, not as evidence that centrality-based rewiring works.

**Status:** wiring only — config, label builder, registries, and a build smoke test (psi/degree/centrality all build at K=10, 0 isolated nodes). **No proteins scores exist yet**; the virtual graphs, LP splits, DeepWalk bridge and GraphSAGE runs are unrun.

## 2026-07-17 — The `degree` virtual-graph variant degenerates into a star on tie-heavy graphs (measured)

**Finding.** The `degree` variant does not build a role graph on proteins. It builds a **star centred on ~10 arbitrary low-id nodes**. Measured on `proteins` K=10:

| variant | edges | max virtual degree | Σdeg² (drives node2vec runtime) | DeepWalk time / seed |
|---|---|---|---|---|
| psi | 250,223 | 91 | 6.03e6 | **2m23s** |
| centrality | 249,794 | 79 | 5.96e6 | ~2m |
| hybrid | 327,456 | 93 | 1.02e7 | ~3m |
| original | 81,044 | 25 | 6.62e5 | ~1m |
| **degree** | 434,111 | **14,644** | **4.89e9** | **~2h27m** |

**Mechanism** (`virtual_graph.py:67-79`). The `degree` signature is a **1-D integer**. Proteins has **43,466 nodes but only 16 distinct degree values** (range 1–25); **14,645 nodes share degree 3**. Every one of those nodes is at **distance exactly 0** from every other, so `NearestNeighbors.kneighbors` breaks the tie by **index order** and returns the *same* ~11 low-id nodes for all 14,645 queries. `V.add_edge` symmetrises, so those few nodes absorb an edge from every member of the tie group → verified: node 1 has virtual degree **14,644**, base degree 3, and **100% of its 14,644 virtual neighbours have base degree exactly 3**. Top-5 virtual degrees are all 14,644, all base-degree 3.

**Why the checks missed it.** The `added >= k` cap bounds only a node's *outgoing* picks; *incoming* picks are unbounded. Notebook 2 §4 asserts `min(degree) >= K` and never inspects the maximum — a star passes every constraint in the report.

**This is not proteins-only — it scales with tie density, and it already affects a published row:**

| dataset | nodes | distinct base degrees | max virtual degree (`degree`, K=10) | Σdeg² |
|---|---|---|---|---|
| cora | 2,708 | 37 | 582 | 1.16e7 |
| enzymes | 19,474 | **9** | **6,724** | 9.92e8 |
| proteins | 43,466 | 16 | **14,644** | 4.89e9 |

⚠️ **Threat to an existing claim.** `degree` currently **wins enzymes node classification (F1 0.5136, the best of the five variants)** — and the enzymes `degree` graph has a 6,724-degree hub from the same defect. That number is therefore **not evidence that degree-based role rewiring works**; it is a score obtained on a star graph whose hubs were selected by node id. The E-ablation `degree` arm on **both** enzymes and proteins must be re-read in this light, and the enzymes NC winner is in question.

**Interpretation.** Tie-breaking by index makes the graph a function of **node id**, which carries no structural meaning — and in proteins the ids are ordered by protein membership, so every degree-3 node in the dataset is wired to a handful of nodes from the *first* protein. Random tie-breaking would still be arbitrary per-edge but would spread degree ≈ 2K with no hub, keeping the construction's intent ("connect to K role-similar nodes") intact. `psi` and `centrality` are continuous-valued and do not tie at scale, which is exactly why their max degree stays at ~90.

**Status:** measured and recorded; **no fix applied and no result recomputed** — deliberate, pending a decision on tie handling. Runtime is the visible symptom (the ~810× Σdeg² gap = 2h27m vs 2m23s per DeepWalk seed), but the validity problem is the reason this matters.

## 2026-07-18 — METHOD FIX: tie collapse in the top-K virtual-graph construction (invalidates prior `degree` and `centrality` results)

**The defect.** `virtual_graph.py` gives every node a **1-D** structural signature (`signatures()` returns an `(N, 1)` matrix) and connects it to its top-K nearest nodes via `sklearn.NearestNeighbors`. For the `degree` variant that signature is a single integer, and real graphs have very few distinct degrees: **enzymes has 9 distinct degrees across 19,474 nodes; proteins has 16 across 43,466.** Thousands of nodes are therefore *exactly coincident points* at distance 0, and "top-K nearest" is **ill-posed** — there is no nearest neighbour to find.

`NearestNeighbors` resolved these ties **deterministically and query-independently**: every member of a tie class received the *same* K winners (the first K in the BallTree's internal layout — verified *not* node-id order). The resulting graph is not a role graph but a set of **disjoint stars**:

| dataset | tie class | class size | virtual hub degree (K=10 requested) |
|---|---|---|---|
| enzymes | degree = 4 | 6,725 | **6,724** |
| proteins | degree = 3 | 14,645 | **14,644** |
| cora | — | — | **582** |

Measured on enzymes: 19,392 of 19,474 nodes had virtual degree exactly 10, while **70 nodes (7 classes × 10 winners)** absorbed everything. All of a class's information was routed through 10 arbitrary representatives — a severe message-passing bottleneck and an artefact of the tie-break, not of structural role.

**A second, quieter instance.** `centrality` showed no hub (max degree 23) but **89.1% of its enzymes virtual edges had distance < 1e-12**, median distance exactly 0.0: eigenvector centrality is degenerate there (96.9% of nodes < 1e-6), so neighbour selection was driven by **float noise** below any meaningful precision.

**The fix (implemented).** Tied signatures mean the nodes are *genuinely interchangeable under that role definition*, so the principled operation is to **sample, not to order**:
1. Quantize signatures to `SIG_TOL = 1e-9 × spread` so noise-level differences collapse into exact ties (this is what makes `centrality` well-posed).
2. Group nodes into exact-tie classes; each node draws K neighbours **uniformly at random from its own class**, via a `seed`-derived RNG (`VirtualGraph(..., seed=42)`).
3. If a class holds fewer than K+1 members, take it whole and widen outward to the nearest signature values — so `psi`-like near-unique signatures keep the original nearest-neighbour semantics and the exact-K contract holds for every variant.

**Effect.** Enzymes degree max virtual degree **6,724 → 36**; proteins **14,644 → 35**; cora **582 → 30**. Mean degree ≈ 2K with a tight spread, zero isolates, all construction rules pass. Build is deterministic across reruns and genuinely varies with the seed.

**Side effect — this was also a compute bug.** DeepWalk/node2vec cost tracks `sum(deg²)`. Enzymes degree: 9.92e8 → 7.95e6 (**125× cheaper**); proteins: 4.89e9 → 1.78e7 (**275× cheaper**). The proteins `degree` run had been abandoned after 26 minutes at 26% of a single seed.

**Results invalidated — must be regenerated.** Jaccard overlap between the old saved graphs and the corrected builds:

| dataset | psi | degree | centrality | hybrid |
|---|---|---|---|---|
| cora | 0.824 | **0.047** | 0.765 | 0.864 |
| enzymes | 0.984 | **0.004** | **0.029** | 0.988 |

Every variant except `original` changed. **All non-`original` scoreboard rows are stale**, most critically the two enzymes E-study headlines: **NC `degree`+`graphsage_edge` = 0.5536** (Table 1 winner) and **LP `centrality`+`deepwalk` = 0.7382**. Neither is currently valid evidence. The cora `hybrid`/`psi` winners moved less but still changed and must be rerun before quoting.

**Why `graph_health.csv` did not catch it.** It logged `avg_degree`, `components`, `isolates` — enzymes degree K=10 read `avg_degree 19.95, isolates 0, components 8`, which looks healthy, because **a star has a perfectly normal mean degree**. A `max_degree` column has been added; the detector is max/skew, never the mean.

**Paper framing.** State this as a methodological point, not an erratum: *for a low-cardinality structural signature, top-K nearest-neighbour selection is ill-posed, and any deterministic tie-break silently manufactures hub artefacts.* Sampling within the tie class is the correct definition of the construction, and the `degree` variant is precisely the case where the signature is coarsest. Whether `degree` still wins enzymes NC after the fix is **open** — the star may have been hurting it.

## 2026-07-18 — METHOD FIX 2: link-prediction splits were restricted to the largest connected component

**The defect.** `prepare_linkpred.build_graph()` ended with `largest = max(nx.connected_components(G), key=len); return G.subgraph(largest).copy()`. Every link-prediction split — train edges, test positives, negatives — was therefore drawn from **one component only**. On a connected graph that is harmless; on the disjoint-union datasets central to this study it discards almost the entire dataset:

| dataset | full graph | largest component | LP actually covered | test positives (seed 42) |
|---|---|---|---|---|
| cora | 2,708 nodes | 2,485 | 91.8% | 1,521 |
| enzymes | 19,474 nodes | **125** | **0.6%** | **17** |
| proteins | 43,466 nodes | **620** | **1.4%** | 315 |

**Enzymes link prediction was measured on 125 of 19,474 nodes with 17 held-out positive edges.** An AUC over ~17 positives moves in steps of roughly 0.06 and carries enormous variance, so the recorded enzymes LP numbers — including the E-study headline **`centrality`+`deepwalk` = 0.7382** — describe a 125-node fragment, not the dataset. Node classification always used the full graph, so **NC and LP were never measured on the same node set**, and any per-dataset comparison between the two tasks was invalid.

**The fix.** `build_graph` now returns the whole graph (self-loops still dropped). No other change was needed: `nx.minimum_spanning_tree` returns a spanning **forest** on disconnected input, so the existing "spanning-tree edges stay in train" rule already preserves connectivity *per component* — no component can be split off by the holdout.

**Verified after the fix** (seeds 42, all three datasets): split covers 100% of nodes; test fraction exactly 30.0%; train component count equals graph component count (nothing disconnected); no train/test edge overlap; negatives are genuine non-edges with no self-pairs. Enzymes test positives **17 → 11,185**; proteins **315 → 24,313**; cora **1,521 → 1,583**.

**Consequence.** Every LP number ever recorded is superseded, cora included (its split changes too, 91.8% → 100%). This is independent of the tie-break fix: it lives in the split code, so it affects all LP results regardless of virtual-graph variant or encoder. LP splits must be regenerated before any LP result is quoted.

**Known remaining caveat (not fixed, state in the paper).** `sample_non_edges` draws negatives uniformly at random from all node pairs. On graphs with many components (enzymes 640, proteins 1,195) a random pair is almost always cross-component while every positive is within-component, so "same component?" alone is a strong predictor and AUC is inflated. This is the standard protocol and was left unchanged for comparability with I2V, but LP AUC on disjoint-union datasets should be read as an optimistic bound. A within-component negative sampler would be the stricter alternative.

## 2026-07-18 — METHOD FIX 3: link-prediction negatives are now sampled within a component

**The defect.** `sample_non_edges` drew negative pairs uniformly from all node pairs. Test **positives are within-component by construction** (an edge cannot cross components), so on a many-component graph the two classes were separable by a rule that has nothing to do with link structure:

| dataset | components | negatives that were within-component (old) | positives |
|---|---|---|---|
| cora | 78 | 84.2% | 100% |
| enzymes | 640 | **0.1%** | 100% |
| proteins | 1,195 | **0.2%** | 100% |

On enzymes and proteins, **99.8-99.9% of negatives were cross-component**, so a classifier answering only "are these two nodes in the same component?" separates the classes almost perfectly. Every LP AUC on those datasets was dominated by component membership rather than by the structural signal under study — which is exactly the quantity the virtual-graph comparison is supposed to isolate.

**The fix.** `sample_non_edges(..., negatives='component')` is now the default: both endpoints are drawn from the *same* connected component, so a negative pair is as hard as a real edge and the "same component?" shortcut carries zero information. Component choice is weighted by each component's **exact count of available non-edges** (`n(n-1)/2 - m`, accumulated into cumulative weights), which makes the draw uniform over all within-component non-edges rather than biased toward large components; components that are complete or single-node are excluded since they offer no non-edge. An assert fires if within-component capacity is below the requested count, pointing at `negatives='uniform'`.

`negatives='uniform'` retains the old behaviour and is exposed as `--negatives uniform`, so the **Phase-1 / I2V protocol stays reproducible** — Phase-1 numbers were produced under `uniform` and must be regenerated with `uniform` if ever re-run.

**Verified** (cora / enzymes / proteins, seed 42, both modes): negatives 100% within-component under the new default, zero invalid pairs (no real edges, no self-pairs), zero duplicates, deterministic across repeated calls.

**Secondary fix in the same function.** `sample_non_edges` was seeded with the *same* value as `split_edges`; it now uses `seed + 1` so the two draws are independent streams rather than correlated by construction.

**Consequence.** Combined with the largest-component fix, **all LP splits and therefore all LP results are superseded**, cora included. Expect measured AUC to *drop* on enzymes and proteins — that is the artefact being removed, not a regression. Because the old protocol made the task partly trivial, any previous claim that a virtual-graph variant "wins link prediction" on those datasets carries no weight until rerun.

## 2026-07-18 — METHOD FIX 4 + unified graph policy

**Omega per component, rescaled to max=1.** I2V's Ω is a *global* eigenvector, so on a disjoint union it is supported only on the component with the largest spectral radius and every other component underflows: enzymes reached **1.2e-115**, with **96.9%** of nodes below 1e-6 and 81.6% below 1e-12. That is power-iteration underflow, not centrality. It contaminated more than the `centrality` variant — psi's `q` term *is* Ω, so on enzymes psi correlated **+0.51 with log₁₀ Ω** versus only +0.25 with degree, i.e. psi tracked numerical underflow depth more strongly than structure. (On cora, psi correlated −0.80 with degree — behaving as intended — which is why this never surfaced.)

Ω is now computed inside each connected component. Degenerate share: enzymes 96.9% → **0.2%**, proteins 76.9% → **3.0%**, cora 17.9% → 8.2%. A numerical guard was added: the eigensolver fallback returned values like −8.6e-18, and psi's `q > 0` test would have silently dropped those nodes.

Each component is then rescaled to **max = 1**, so Ω reads "centrality relative to my component's most central node". Under NetworkX's native L2 = 1 the *scale* depended on component size (a 2-node fragment's nodes got 0.707, the maximum possible, while a 2,485-node component's hub got ~0.35), making cross-component comparison meaningless — and cross-component comparison is exactly what a role graph does.

**Honest limitation.** Max-normalization does **not** remove the correlation between Ω and component size (cora −0.92 → −0.94). Measured means by component size on cora: size 2485 → mean Ω 0.008; size 9 → 0.714; size 2 → 1.000. Every component's top node is 1.0 by construction, so in a 2-node component *both* nodes are maximally central — which is locally true, not an artefact. The consequence to state in the paper: **Ω partly encodes "am I in a fragment or in the giant component"**, strongly so on cora. Max-normalization fixes the arbitrary scale distortion; the distributional effect is inherent to any per-component normalization.

**Unified graph policy (`graph_io.GRAPH_POLICY`, re-exported as `benchmark_config.GRAPH_POLICY`).** All six cross-dataset decisions now live in one dict instead of four files: `self_loops`, `directed`, `centrality`, `sig_tol`, `lp_negatives`. `I2V_BASELINE_POLICY` names the Phase-1 contract (`centrality='global'`, `lp_negatives='uniform'`) under which Deliverable #1's byte-identical guarantee is valid — verified still `True` after every change. Unknown policy keys raise rather than being silently ignored.

`graph_io.load_graph()` is the single reader for every stage (splits, virtual graphs, encoder features), so they cannot disagree about the graph — verified `encoder.build_graph` and `prepare_linkpred.build_graph` return identical graphs. `graph_io.check()` reports the properties that silently broke earlier runs — disconnection, coarse degree signature, isolates — with `strict=True` converting them to a hard failure.

**A false positive caught before it shipped.** The first version of `check()` inferred directedness by counting edges with no reverse edge, and flagged **proteins as 81,044/81,044 directed** — proteins is a molecular graph rebuilt from sorted unique pairs, i.e. undirected. The metric was measuring *storage format*: a normal undirected edgelist lists each edge once and is **indistinguishable from a directed one**. Directedness is now declared as dataset metadata (`DATASETS["politics"]["directed_source"] = True`, a retweet relation) and never guessed. Recorded because a warning that fires on every dataset trains its reader to ignore it.

## 2026-07-20 — METHOD FIX 5: tie widening samples, and the split protocol is made interpreter-independent

Follow-ups from a full code review. **Implemented, not measured** — no run was performed, so every statement below is about mechanism, not about a number.

**The tie fix was only half a fix.** The 2026-07-18 degree-tie fix made top-K selection sample *inside* a node's own tie class, which removed the star collapse (enzymes hub degree 6,724 → bounded). But when a node's class holds fewer than K members, `build()` widens outward to the nearest signature values — and that widening walked outward **one index at a time in `argsort` order**. Members of the adjacent class are by definition *equidistant* from the node, so index order is arbitrary; the consequence is that every node needing to widen into the same neighbouring class borrowed the **same first-indexed members** of it. The stated guarantee "ties are sampled, never ordered" therefore held only within a node's own class, not at the boundary where widening happens.

`build()` now widens **one whole tie class at a time** and samples inside it (`rng.choice(..., replace=False)`) whenever the class offers more candidates than the node still needs — the same rule that already governed the node's own class, applied consistently. Exactly-K per node, determinism given the seed, and the node-set/finite-weight asserts are unchanged.

**Who this changes.** Only nodes whose own tie class is smaller than K+1 *and* whose nearest neighbouring class is larger than the remaining need. On a continuous signature (`psi`, where nearly every class is a singleton) the widened class is almost always a singleton too, so selection is unchanged. The effect concentrates on the **coarse signatures — `degree` and `centrality`** — precisely the variants that motivated the original tie fix, and on their rare-value tail nodes (a uniquely-high-degree node widening into a large common-degree class). **Consequence: virtual graphs must be rebuilt before the `degree`/`centrality` (and `hybrid`) rows are quoted; `psi` and `original` are expected to be unaffected, but that expectation is untested.**

**Split protocol: determinism no longer depends on the interpreter.** `split_edges` built its train list as `list(tree) + ...` from a **set**, and `sample_non_edges` accumulated negatives into a **set** that `prepare()` then sliced into train/test. Set iteration order is stable for a given CPython build but is not part of the language contract, so two things silently rode on it: the row order of `train.edgelist` (which fixes node insertion order, hence the walk RNG, hence the embedding) and **which negatives landed in train versus test** (which affects the supervised logreg scorer). Positives are now sorted; negatives are kept in RNG draw order — deliberately not sorted, because sorting would place every low-id pair in train and every high-id pair in test. The seed already determined *which* edges are held out; this only removes the interpreter from the reproducibility claim. **All LP splits change once regenerated, so LP embeddings must be retrained** — `runner` now enforces this automatically by refusing to reuse an embedding older than its split.

**A claim in the 2026-07-18 entry above was too strong.** That entry states that weighting components by their exact non-edge capacity makes the draw "uniform over all within-component non-edges". It does not, quite: a component is chosen by capacity, a pair is drawn inside it, and the pair is *rejected and redrawn* if it hits a real edge — so a denser component yields an accepted negative slightly less often than its capacity share, and the weights are never decremented as pairs are consumed. The resulting tilt toward sparser components is on the order of a few percent and does not change which components are reachable, but the paper should claim "capacity-weighted within-component sampling", **not** "exact uniform". The code comment now says so.

**Phase-1 protocol was unenforced in code.** `I2V_BASELINE_POLICY` (`centrality='global'`, `lp_negatives='uniform'`) names the contract under which Deliverable #1's byte-identical guarantee holds, but no caller passed it: both `runner` link-prediction paths called `prepare()` with no `negatives` argument and so inherited the new `component` default. Rerunning Phase-1 would have overwritten its splits under the *post-fix* protocol while still being reported as the I2V-comparable baseline. Both call sites now pass the baseline policy explicitly, so the comparability claim is enforced by the code rather than by remembering a flag.

**Reported dispersion was mixing two conventions.** `results_io.record_score` (the master scoreboard) used population std while `runner.summarize_seed_results` and the pandas benchmark tables used sample std — over 3 seeds the scoreboard understated dispersion by ~18% relative to the benchmark tables for identical inputs. Unified on sample std (ddof=1). Means are unaffected; **`std` values already in `scoreboard.csv` are population std until those rows are rerun**, so scoreboard and benchmark error bars should not be compared across that boundary.

**Node-classification F1 could be quietly understated.** Both notebooks suppress warnings globally, so an LBFGS fit that hit `max_iter=300` without converging was reported as a normal F1. `max_iter` is unchanged (paper protocol); non-convergence is now surfaced per evaluation as an explicit "this F1 is a lower bound" line. Whether any past run actually hit the cap is unknown — no run was made to check.

## 2026-07-20 — FULL RERUN after the 07-18 + 07-20 fixes: cora, enzymes, proteins (K=10, seeds 42/43/44)

First complete post-fix sweep. Notebooks 2 and 3, all five virtual-graph variants, both tasks, three datasets. All previous result files were cleared first, so nothing below is contaminated by a pre-fix artifact. Headline scorer is cosine (`REPRO["linkpred_score"]`); node classification is weighted F1.

### 1. Did the 2026-07-20 fixes move any number? No.

Proteins is the clean control: it was last run on 2026-07-18 *after* the three method fixes but *before* the 07-20 review fixes, so the proteins delta isolates the 07-20 changes alone.

| proteins, deepwalk | LP AUC before → after | proteins, graphsage_edge | LP AUC before → after |
|---|---|---|---|
| centrality | 0.5216 → 0.5222 | centrality | 0.5829 → 0.5831 |
| degree | 0.5045 → 0.5067 | degree | 0.5500 → 0.5508 |
| hybrid | 0.8029 → 0.8023 | hybrid | 0.5633 → 0.5630 |
| original | 0.9414 → 0.9413 | original | 0.6726 → 0.6722 |
| psi | 0.5222 → 0.5232 | psi | 0.5588 → 0.5582 |

Every meaningful row moves by ≤0.005; node classification likewise (largest shift 0.5256 → 0.5215 on deepwalk/original). The only larger delta is the D5 constant-feature *floor* control (0.4407 → 0.4528), whose own std is ±0.062 — pure noise, as designed. **Conclusion: the 07-20 fixes were correctness and reproducibility hygiene, not result-changing.** That is the intended outcome and it is now evidenced rather than asserted.

The tie-widening change (whole tie class, sampled) did alter the graphs, but marginally: proteins psi 317,534 → 317,480 edges (−0.017%), centrality 256,820 → 256,867, hybrid 396,403 → 396,310, degree 434,169 → 434,175, original byte-identical. **Correction to the prediction logged earlier on 07-20:** psi was expected to be completely unaffected because its signature is continuous; it in fact shifted by 54 edges, so psi does contain a small number of exact ties. The direction of the claim held; the absolute "unaffected" did not.

### 2. The 07-18 method fixes changed enzymes drastically — and removed the study's only role-graph win

Enzymes was last run 2026-07-16, so its delta bundles all four method fixes (degree ties, LP largest-component, within-component negatives, per-component Ω) plus the 07-20 hygiene.

| enzymes, deepwalk LP AUC | before (07-16) | after (07-20) |
|---|---|---|
| centrality | 0.7382 ± 0.0840 | 0.5283 ± 0.0014 |
| degree | 0.6240 ± 0.0725 | 0.5052 ± 0.0008 |
| psi | 0.5110 ± 0.0297 | 0.5039 ± 0.0031 |
| hybrid | 0.5087 ± 0.0246 | 0.7765 ± 0.0025 |
| original | 0.6574 ± 0.0170 | 0.9007 ± 0.0007 |

**The headline consequence.** Before the fix, `centrality` (0.738) beat `original` (0.657) on enzymes link prediction — the single strongest piece of evidence that a role graph can outperform the real graph. After the fix, `original` (0.901) beats `centrality` (0.528) by 0.37. **That win was an artifact of the 125-node largest-component split (17 test edges); it does not survive.** Any earlier claim resting on it is withdrawn.

Note also the standard deviations collapsing by roughly two orders of magnitude (0.084 → 0.0014). With 17 test edges the old metric was noise; with 11,185 it is stable. Node classification barely moved (≤0.01 on every variant) because it never used the split — and `original` reproduced to 16 significant digits (0.5086297830053176 both times), a clean determinism check that the unrelated machinery did not drift.

The enzymes psi graph changed most in construction: 111,401 → 150,078 edges (+35%), from the per-component Ω fix (psi's `q` term *is* Ω, which was underflowing to ~1e-115 on this disjoint union).

### 3. Post-fix findings (these are the numbers to build the paper on)

**a. The original graph wins all six dataset × task cells.**

| best configuration | node classification (F1) | link prediction (AUC) |
|---|---|---|
| cora | deepwalk + original **0.8100** | deepwalk + original **0.8971** |
| enzymes | graphsage + original **0.5589** | deepwalk + original **0.9007** |
| proteins | graphsage + original **0.5799** | deepwalk + original **0.9413** |

No virtual graph beats the original graph anywhere. The best non-original variant is always `hybrid` — which contains the original edges by construction — and even it trails by 0.12–0.22 on link prediction and by 0.36 on cora node classification.

**b. The gap is entirely dataset-dependent, and that is the per-data thesis.** On the two molecular graphs the pure role graphs come within 0.01–0.02 of the original for node classification (enzymes: degree 0.5474 vs original 0.5589; proteins: degree 0.5591 vs original 0.5799). On cora they trail by 0.36–0.44. Structural role is nearly sufficient for molecular node labels and nearly worthless for citation-community labels — the intended "which virtual graph for which data" result, now with the artifact removed.

**c. Link prediction is where role graphs fail categorically.** Pure `psi`/`degree`/`centrality` sit at 0.50–0.66 AUC on every dataset against 0.90–0.94 for the original graph. Role similarity is not adjacency: two nodes sharing a role are no likelier to be connected. State this as a finding, do not bury it.

**d. Technical contribution holds: the GNN beats walk+Skipgram on the virtual graph, 6/6.** Same psi graph, encoder swapped:

| psi graph | NC: deepwalk → graphsage | LP: deepwalk → graphsage |
|---|---|---|
| cora | 0.2133 → 0.2354 (+0.022) | 0.4990 → 0.5175 (+0.019) |
| enzymes | 0.5005 → 0.5470 (+0.047) | 0.5039 → 0.6237 (+0.120) |
| proteins | 0.4882 → 0.5588 (+0.071) | 0.5232 → 0.5582 (+0.035) |

**e. The GNN advantage is specific to role graphs.** On the *original* graph the ordering reverses for link prediction on all three datasets (cora 0.897 → 0.609, enzymes 0.901 → 0.700, proteins 0.941 → 0.672) and for cora node classification (0.810 → 0.426). GraphSAGE wins node classification on the original graph only for the molecular datasets. Honest framing: message passing helps when the graph encodes role; random walks with a lookup table remain far stronger when the graph encodes adjacency.

**f. Ablation D6 — message passing helps node classification, contributes nothing to link prediction on molecular graphs.** Comparing "features only, no message passing" (layers=0) against the full 2-layer encoder on the psi graph: NC gains +0.052 (cora), +0.047 (enzymes), +0.037 (proteins); LP gains +0.034 (cora) but only +0.004 (enzymes) and **−0.008 (proteins, i.e. message passing hurts)**. On the molecular graphs the LP signal is carried by the input features alone.

**g. Ablation D on cora — best features are degree + centrality, not all four.** D2 (deg+cent) reaches 0.2460 NC against D0 (all four) at 0.2354, while D3 (psi only) is worst at 0.1754 — below the D4 random-feature control on NC. psi as an input feature is confounded (the psi graph was built from it) and adds nothing once the graph already encodes it.

**Caveats on all of the above.** K=10 only; three seeds; the virtual-graph build seed is fixed at 42, so the reported ±std covers split and encoder-init variance but not tie-sampling variance. Encoder settings are the previously locked A2 (`positives="edge"`) and B-mean, whose ablations were decided on pre-fix enzymes runs and were deliberately not re-validated.

## 2026-07-21 — Research direction locked for LoG: a "when to augment" characterization study

Supervisor meeting. The post-fix results (original graph wins all 8 dataset × task cells; role graphs fail link prediction; role graphs approach the original only on molecular node classification) were presented and accepted as the honest result — no attempt to rescue a "virtual graph wins" headline.

**Reframed contribution.** ViRGo is a **study**: given a graph and a task, when is the original topology enough, and when do structural augmented features help? The deliverable is a **characterization** — connect each dataset's properties (homophily first, then degree spread, clustering, component fraction, label-vs-topology agreement) to the original-vs-best-augmented gap, so the paper can say "for graphs like this, keep the original; for graphs like that, add these features." This is the supervisors' framing, offered as a conjecture for the community, not a proof.

**Scope decisions (locked).**
- **Purely structural.** ViRGo's features are graph-derived (degree, eigenvector centrality, Ψ, clustering) and do double duty — they build the virtual graph and feed GraphSAGE — and ablation D already showed they are necessary (random features drop to the DeepWalk baseline or below). No external node attributes anywhere, including OGB text features and biological descriptions: they would confound the study, since a gain could then be attributed to the attributes rather than to the structural rewiring under test. Isolating structural identity is the point (the inherited I2V premise).
- **Datasets.** Current four (cora, citeseer_linqs, enzymes, proteins) + small-to-medium OGB: ogbn-arxiv (node property), ogbl-collab and ogbl-ddi (link property). Large-scale OGB (100M-node) is out — too slow before the deadline.
- **OGB evaluation protocol (fair, structural-only).** On every OGB dataset both graphs receive the *same* structural features (degree, centrality, Ψ, clustering); only the edges differ — the original graph carries the real edges, the virtual graph the role-based edges. OGB's extra attributes (text embeddings, product descriptions, biological annotations) are ignored, so graph structure is the single variable and any difference measures the effect of virtual rewiring. We do **not** compare against top OGB leaderboard entries, which may use those attributes; the paper states plainly that the goal is structural analysis, not leaderboard superiority.
- **Per-dataset node features (verified on OGB docs, 2026-07-21).** `ogbl-ddi` has no node features (a homogeneous drug-drug interaction graph) and is used directly as a structure-only link-prediction benchmark. `ogbn-arxiv` (128-dim skip-gram embeddings of title + abstract) and `ogbl-collab` (128-dim word embeddings of authors' papers) do carry node features, which we do not load; the structural methodology is applied unchanged. No implementation change is needed for any of the three. (This corrects an in-conversation slip that "ogbn-arxiv has no node features" — the featureless dataset is ogbl-ddi; ogbn-arxiv has 128-dim features that we deliberately ignore.)
- **Encoder order.** GraphSAGE results first; then swap in GIN to test whether its stronger isomorphism (WL) power helps. GIN is in scope for this paper, after GraphSAGE.
- **Future work (not now):** the learnable-alpha graph (a single learned weight that blends original vs virtual per dataset) needs many synthetic datasets to train, so it is a paper/thesis extension; plus the LLM graph-summary stretch. Anomaly detection is dropped from the immediate plan.

**Venue + timeline.** Immediate target LoG (Learning on Graphs), abstract ~end of July, full paper ~start of August; published via JMLR/PMLR. Thesis (~one month) reuses the same content.

**Threats to validity to carry into the paper** (unchanged): the A–D encoder ablations were decided on pre-fix enzymes and not re-validated; K=10 only; 3 seeds; the virtual-graph build seed is fixed at 42; baselines used as published.

## 2026-07-22 — OGB evaluation protocol decided: official splits + official metrics (revises the earlier "keep our protocol" plan)

Adding the first two OGB datasets (ogbn-arxiv for node classification, ogbl-ddi for link prediction). A protocol question was worked through: should OGB datasets be scored under the ViRGo protocol (random 70/30 split, weighted-F1, cosine-AUC) for cross-dataset uniformity, or under OGB's official split + metric? **Decision: use OGB's official split and official metric for the OGB datasets.** This *revises* the earlier direction-lock note that leaned toward reusing our protocol everywhere.

**Rationale.** The OGB split is part of the scientific task, not file organization: ogbn-arxiv uses a *time* split (train on older papers, test on newer) and ogbl-ddi a *protein-target* split (interactions among drugs with different mechanisms). A random 70/30 deletes exactly the intended generalization challenge and inflates scores. OGB's Hits@20 (rank one true interaction against ~100k negatives) is also far stricter than balanced ROC-AUC. Keeping the official protocol is the more honest and more publishable choice.

**Consequence — mixed-protocol study, framed as internal-comparison-only.** The four core datasets stay on the ViRGo protocol; the OGB datasets move to official split + metric. This is acceptable because the study's real axis is *internal* (original vs each virtual graph, within one dataset, same split + metric). Across datasets we compare only *direction and ranking* — which graph wins, the ordering of variants, whether rewiring helps/hurts/ties, and how that relates to graph properties — never raw magnitudes (a +0.05 Accuracy gain is not a +0.05 Hits@20 gain). Results are reported as "OGB-sourced datasets evaluated under the official OGB protocol", not leaderboard entries (data.x is still ignored — purely structural).

**Leakage rules differ by dataset and are coded accordingly.** ogbn-arxiv is *transductive* — the whole graph is observed and only labels are split — so structural features and the virtual graph are built on the full graph; only the scored node set (official test ids) is restricted. ogbl-ddi *splits the graph itself* — features, virtual graph and embeddings are built from the **training edges only** (no valid/test edge touches structure), then the supplied positive/negative pairs are scored. Model selection follows the no-test-peeking rule: which variant/encoder wins is decided on **validation**; **test** is read once for the final number.

**Metrics.** ogbn-arxiv: Accuracy (primary, via OGB `Evaluator`) plus weighted-F1 and macro-F1 (secondary) on the official test nodes. ogbl-ddi: Hits@20 (OGB `Evaluator`, K=20 fixed), with the same cosine scorer used for every graph variant so the internal comparison stays fair. The four structural features (degree, eigenvector centrality, Ψ, clustering) are computed once from the built graph and shared across all five variants — only the edges differ (confirmed already implemented in `encoder.features()`, which reads the original graph).

**Caveats.** GraphSAGE first, GIN later (unchanged order). ogbn-arxiv (~169k nodes) is the only heavy run; the 1-D-signature virtual-graph build is O(N log N) so there is no algorithmic blocker, but encoder runtime should be timed on one seed before the full sweep.

## 2026-07-23 — ogbl-ddi scorer: cosine was measuring a ceiling, not a model

**Finding (methodological, worth a paragraph in the paper).** Our first ogbl-ddi sweep returned Hits@20 = 0.0000 for GraphSAGE on four of five graph variants. That number was **not** a model failure. ogbl-ddi's training graph contains a class of 156 structurally identical nodes (degree 4, clustering exactly 1/6, eigenvector centrality equal to 9 decimal places). A purely structural encoder is a deterministic function of (features, neighbourhood), so all 156 receive the *same* embedding vector, and every one of the C(156,2) = 12,090 pairs among them has cosine similarity exactly 1.0. Between 18 and 46 of those pairs appear in OGB's official negative set. OGB computes `Hits@K` as `mean(y_pred_pos > kth_largest(y_pred_neg))` with a **strict** inequality, so the threshold lands on 1.0 - the ceiling of cosine - and no positive pair can ever exceed it. Hits@20 is then identically zero regardless of embedding quality.

Three independent checks confirm the mechanism: (i) the same embeddings reach AUC 0.66-0.76 and Hits@100 = 0.033; (ii) `hybrid` - the only variant whose role edges break the twin symmetry, leaving zero ties - is the only nonzero GraphSAGE cell (0.0013); (iii) DeepWalk, whose random walks inject node identity and therefore never produce identical vectors, is nonzero everywhere.

**Generalisable caution:** a bounded similarity (cosine) combined with a strict-inequality top-K metric is degenerate for any deterministic structural embedding on a graph with automorphic classes. This is a measurement artifact, not a property of the method, and any structural-identity paper reporting Hits@K should check for it.

**Protocol change.** Pair scoring for OGB link prediction moves from fixed cosine to a trained decoder, matching OGB's own ddi reference model (`examples/linkproppred/ddi/gnn.py`): the elementwise product of the two endpoint embeddings through a 2-layer MLP (256 hidden), fitted with BCE on **training edges only** against uniformly sampled non-edges. Embeddings stay frozen - we are replacing the scorer, not the encoder - and the identical decoder, hyperparameters and seed are used for every graph variant, so the graph remains the single variable. Positives are always the real training links, never the virtual role edges. This does not weaken the "purely structural" rule: the decoder learns from graph-derived supervision only, and no external node attribute is loaded anywhere.

**Note for the write-up:** ViRGo still differs from the OGB leaderboard entries in a second, deliberate way - those models feed a free `torch.nn.Embedding` of 256 dimensions per node, trained end to end, i.e. pure node identity. ViRGo refuses that by design (four graph-derived features only), so a gap to the leaderboard is expected and is not the quantity under study. The comparison that matters stays internal: original vs each virtual graph, same scorer, same split, same metric.

**Effect (seed 42, validation, pre-sweep check):** DeepWalk/original 0.0124 -> 0.0857, GraphSAGE/original 0.0000 -> 0.0411, GraphSAGE/psi 0.0000 -> 0.0228. The full re-sweep (validation, selection, single test read) follows; the cosine-era rows are retained in `results/scoreboard.csv` under `*_hits@20_cos`, and the cosine-era selection lock was voided so the winner is re-chosen under the new scorer before test is read.

## 2026-07-23 — ogbl-ddi results under the trained decoder, and what they change

**Result (K = 10, 3 seeds, official split + Evaluator, decoder trained on training edges only).** Winner selected on validation, test read once.

| graph | valid GraphSAGE | valid DeepWalk | test GraphSAGE | test DeepWalk |
|---|---|---|---|---|
| Ψ | 0.0223 | 0.0071 | 0.0412 | 0.0081 |
| degree | 0.0213 | 0.0071 | 0.0265 | 0.0080 |
| centrality | 0.0309 | 0.0077 | 0.0514 | 0.0100 |
| original | 0.0385 | 0.0772 | 0.0102 | 0.0378 |
| hybrid | 0.0385 | **0.1038** | 0.0117 | **0.0533** |

Locked winner: hybrid + DeepWalk, validation 0.1038 ± 0.0040, test **0.0533 ± 0.0062**.

**Finding 1 — the scorer, not the graph, produced the earlier null result.** Under fixed cosine every ddi cell was ≤ 0.013 and four GraphSAGE cells were identically zero. Replacing cosine with the OGB-style trained decoder raised every cell by one to two orders of magnitude and removed the zeros. Any claim of the form "role graphs carry no link signal on ddi" was an artifact of the scorer and is withdrawn.

**Finding 2 — with a trained decoder, role graphs beat the original graph for GraphSAGE on test.** Test: centrality 0.0514 and Ψ 0.0412 against original 0.0102 — a 4-5× gap in favour of role-based rewiring. This is the first dataset × task cell in the study where an augmented graph clearly beats the original under the same encoder, and it runs against the locked headline that the original graph wins everywhere. The headline is not yet revised: this is one dataset, under a different (official-OGB) protocol, and the effect appears on test but not on validation.

**Finding 3 — validation does not predict test on ogbl-ddi.** GraphSAGE's ranking inverts between the two splits (original/hybrid lead on validation, role graphs lead on test); DeepWalk keeps its order but halves. The official split is by protein target, so validation and test are deliberately different distributions, and the OGB leaderboard shows the same decorrelation. Consequence for the write-up: on this dataset a single-split selection is fragile, and the honest presentation is both splits side by side, not the test number alone. Our locked choice happened to also be the best test cell, so no selection cost was incurred, but that was luck and should be stated as such.

**Open question raised for the core four.** Their link-prediction numbers come from `eval_linkpred.py` with the default `score='cosine'`. Given Finding 1, the LP conclusions across the whole study may be scorer-dependent. A supervised alternative (`--score logreg`, hadamard + logistic regression) already exists in that script, so the check needs no new code. This should be resolved before the paper's link-prediction claims are finalised.

## 2026-07-23 — density-matched controls: the role edges carry signal, and the encoder comparison was partly a density artifact

**Motivation.** Across the study the original graph's density is not comparable to the role graphs', and the mismatch *reverses sign*: on ogbl-ddi the role graphs are 39× sparser than the original (avg degree ~12 vs 500.5), while on cora, citeseer, enzymes and proteins they are 3-5× denser (12-20 vs 2.8-3.9). Any cross-dataset pattern of the form "role graphs help here but not there" could therefore be tracking edge *count* rather than edge *meaning*. Two controls hold the count fixed, built with the same union-of-K-per-node construction as the role graphs so counts match by construction: `original_k` (K uniformly sampled real neighbours per node) and `random_k` (K arbitrary nodes).

**Results (ogbl-ddi, validation, decoder scorer, 3 seeds, K=10).** Both controls landed ~1.5-1.7× denser than the role graphs, which handicaps the role graphs and makes the comparisons below conservative.

| graph | edges | GraphSAGE | DeepWalk |
|---|---|---|---|
| psi | 27,042 | 0.0251 | 0.0071 |
| degree | 27,109 | 0.0227 | 0.0071 |
| centrality | 24,886 | 0.0326 | 0.0077 |
| original_k | 39,588 | 0.0371 | 0.0303 |
| random_k | 42,632 | 0.0078 | 0.0006 |
| original | 1,067,911 | 0.0413 | 0.0772 |
| hybrid | 1,088,727 | 0.0554 | 0.1038 |

**Finding 1 — role edges beat a random scaffold decisively.** GraphSAGE on the role graphs scores 2.9-4.2× the random control (0.0227-0.0326 vs 0.0078); DeepWalk roughly 12× (0.0071-0.0077 vs 0.0006). The random control has *more* edges than any role graph, so the gap cannot be a density effect. This is the study's first clean demonstration that Ψ, degree and centrality edges carry structural information rather than merely giving the encoder something to aggregate over - it is the edge-level counterpart of ablation D, which established the same for the input features.

**Finding 2 — GraphSAGE extracts nothing from 96% of the original graph's edges.** A uniform K=10 sample of the real edges (39,588) scores 0.0371 against the full graph's 0.0413 with 1,067,911 edges - a 0.3 sigma difference. Whatever advantage the original graph gives GraphSAGE, it is not volume.

**Finding 3 — the encoder ranking inverts once density is matched.** On the full graphs DeepWalk leads GraphSAGE (0.0772 vs 0.0413); at matched density GraphSAGE leads DeepWalk (0.0371 vs 0.0303), because DeepWalk loses a factor of 2.5 when the original graph is sparsified while GraphSAGE loses almost nothing. The earlier reading that "walks beat message passing when the graph encodes adjacency" is, on this dataset, substantially a statement about edge volume. Any encoder claim in the paper should be stated at matched density or explicitly qualified.

**Design note for the write-up.** `original_k` does not replace `original`: possessing a million real links is a genuine property of that graph, not an unfair advantage, so the full graph remains the baseline a practitioner would use. The controls answer a different question - how much of the observed gap is meaning and how much is volume. Also note `original_k` is only constructible where the original graph is denser than K, i.e. on ogbl-ddi; on the core four the matching must run the other way (role graphs rebuilt at the K that matches the original's average degree, K≈2). Matching should therefore be specified on average degree, not on a fixed K.

## 2026-07-23 — where the seed noise comes from, and the frozen configuration

**Question.** The ogbl-ddi validation spread (std/mean 34-51% on GraphSAGE) is comparable to the differences between graph variants, so before extending the study to a second OGB dataset we established which component produces it.

**Decoder is not the source on the role graphs.** Holding one saved embedding fixed and refitting the link decoder under five seeds gives std 0.0014 on GraphSAGE/psi against a total across-seed std of 0.0127 — about 1% of the variance. On the dense original graph the decoder contributes more (std 0.0065 of 0.0142, ~21%), consistent with it fitting 1.07M training edges. DeepWalk's entire (small) spread, 0.0008, is decoder noise: the walk encoder is effectively deterministic across seeds on this graph. Conclusion: **the seed spread is a property of the GraphSAGE encoder**, and error bars on GraphSAGE rows should be read as encoder variance, not measurement variance.

**Encoder training budget.** Loss over 10-epoch windows continues to drift after epoch 50 (psi 4.0191 → 3.8901 by epoch 300; original 3.9005 → 3.7270 by epoch 200), but against a drop of ~38 from initialisation this is under half a percent of the range. The budget is held at 50 epochs for every variant. This is equal in both relevant senses — identical epoch count and identical total gradient samples (50 × 100,000 = 5M pairs per variant, independent of graph size). **Limitation to state in the paper:** the encoder is trained to a fixed budget rather than to a per-graph convergence criterion, and because `pairs_per_epoch` is fixed while corpus size varies with density, a fixed budget corresponds to many passes over a sparse role graph and few over a dense one.

**Configuration frozen (2026-07-23).** K=10; four structural features computed on the original graph and shared by every variant; GraphSAGE mean / 2 layers / 64-dim / lr 0.01 / 50 epochs / edge positives / Q=5 negatives with real-neighbour rejection; DeepWalk bridge as the second encoder; link scoring by an OGB-style trained decoder (hadamard → 256-unit MLP) fitted on training edges only; official OGB split and Evaluator; seeds 42/43/44; winner selected on validation and test read once. Every subsequent dataset runs this configuration unchanged, so that differences between datasets are attributable to the graph and not to per-dataset tuning.

## 2026-07-24 — the six-dataset result set is complete and comes from ONE pipeline

Preparation for the characterization study: before relating graph properties to the original-vs-augmented gap, every number that will enter that regression had to come from the same code. It did not.

**The defect.** The core-four GraphSAGE embeddings were trained on 2026-07-20/21; the negative-sampling rule was changed on 2026-07-22 while validating ogbl-ddi (reject a sampled negative that is a real neighbour of the centre node, redraw). ogbn-arxiv and ogbl-ddi ran after that change, the core four before it, so the cross-dataset table mixed two encoders — exactly the confound the frozen-pipeline rule exists to prevent. (Verified by reproduction, not by timestamps: the pre-rejection encoder taken from git reproduces the stored core-four file to 1.91e-6, the noise floor, with the identical F1, while the current encoder differs by 3.4e-1. The change had been sitting uncommitted in the working tree, so commit dates were useless as evidence. See the follow-up entry for ogbn-arxiv, which reproduces under neither.)

**Audit.** Re-scoring all 240 stored core-four embeddings with the current evaluation code returned their scoreboard rows exactly (80 cells, max |delta| 0.00000, no missing files). The evaluation side was never in question; only the encoder version was.

**Scope of the fix.** The rejection rule was designed for dense graphs (32% of draws collide on ogbl-ddi), but the collision rate is non-zero on sparse graphs too — cora/psi 0.552%, proteins/degree 0.055% of draws at epoch 0 — so the redraw fires there as well and the whole training trajectory shifts. All 120 core-four GraphSAGE embeddings were retrained under the frozen configuration (pre-freeze files kept in `output/superseded_pre_freeze_2026-07-24/`).

**Effect on the results: none that matters.** 37 of 40 GraphSAGE rows moved; the largest move was 0.0054 (cora, hybrid, LP AUC), mean 0.0011, i.e. below the 3-seed std of nearly every cell and far below the +-0.05 reproduction bar. **No cell changed its winning graph variant**, so every published reading survives unchanged: the original graph wins all eight core-four cells, role graphs approach it only for molecular node classification, and role graphs fail link prediction everywhere. DeepWalk rows re-scored byte-identical (its code path was untouched), which is the control on the re-run itself.

**Reproducibility floor, measured.** Repeating one configuration five times under the current code (cora/psi/seed 42) gives the identical metric every time (weighted F1 0.2381) with embeddings agreeing to 2.15e-6 — float32 aggregation order in the SAGE scatter, present even single-threaded. So results are exactly reproducible at the reported precision, and the paper should claim metric-level reproducibility rather than bit-identical tensors. The same evaluation on the pre-freeze embedding gave 0.2376, confirming the 0.0005 gap is the code change and not run-to-run noise.

**The result set now standing (K=10, seeds 42/43/44, GraphSAGE + DeepWalk, five graph variants each).**

| dataset | node classification | link prediction | protocol |
|---|---|---|---|
| cora | weighted F1 | AUC | ViRGo 70/30 split, cosine scoring |
| citeseer_linqs | weighted F1 | AUC | ViRGo 70/30 split, cosine scoring |
| enzymes | weighted F1 | AUC | ViRGo 70/30 split, cosine scoring |
| proteins | weighted F1 | AUC | ViRGo 70/30 split, cosine scoring |
| ogbn-arxiv | Accuracy (+ weighted/macro F1) | not applicable | official OGB time split + Evaluator |
| ogbl-ddi | not applicable (no labels) | Hits@20 | official OGB split + Evaluator, trained decoder |

Ten dataset x task cells, each 5 variants x 2 encoders x 3 seeds. The two "not applicable" entries are properties of the datasets, not gaps: ogbl-ddi ships no node labels, and ogbn-arxiv has no official link-prediction split. Metrics are comparable *within* a dataset, which is what the characterization needs; they are not comparable across the OGB/non-OGB boundary, and no table should place them in one column.

**Best score per dataset x task under the locked encoder (GraphSAGE):**

| dataset | NC: original | NC: best role graph | LP: original | LP: best role graph |
|---|---|---|---|---|
| cora | 0.4266 | centrality 0.2942 | 0.6130 | centrality 0.5659 |
| citeseer_linqs | 0.3298 | centrality 0.2818 | 0.6218 | centrality 0.5437 |
| enzymes | 0.5591 | hybrid 0.5546 | 0.7000 | centrality 0.6563 |
| proteins | 0.5797 | hybrid 0.5646 | 0.6720 | centrality 0.5834 |
| ogbn-arxiv | 0.3918 | hybrid 0.3618 | - | - |
| ogbl-ddi | - | - | 0.0173 | **centrality 0.0519** |

**New: the first cell where augmentation wins.** On ogbl-ddi with GraphSAGE, every role graph beats the original (centrality 0.0519, psi 0.0423, hybrid 0.0344 against original 0.0173 test Hits@20). ogbl-ddi is also the extreme of the dataset panel on density (average degree 500 against 2.8-13.7 elsewhere). That is one point, and the absolute scores are near the floor, but it is the shape the characterization is looking for and it should be tested first when the graph-property table is built: augmentation helps where the original graph is too dense for message passing to be selective, and the locked *DeepWalk* row on the same dataset does not show it (original 0.0378 vs psi 0.0081), so it is an encoder-conditional effect.

**Caveat carried forward.** The ablation-D rows (`graphsage_edge_feat_*`) still come from pre-freeze runs; they justify the locked configuration rather than reporting results under it, and their numbers would move by the same ~0.001. Any ablation table in the paper must say so or be re-run.

**Tooling.** `run_core.py` now runs the core-four sweep headless (mirrors `run_ogb.py`: reuse-or-create, same zones, same scoreboard rows, locked defaults, `--train-only` for parallel training). The whole study reproduces from two commands.

## 2026-07-24 (final) — CONFIGURATION LOCKED. Best and final; the characterization study starts from here

Everything below is settled. No further method, encoder, graph or protocol change is made for the LoG paper; the next work item is the characterization table, which only *reads* these results.

### The locked configuration (quote this in the methods section)

| stage | setting |
|---|---|
| graph loading | `graph_io.GRAPH_POLICY` — self-loops dropped, directed sources read undirected (recorded deviation), `.nodes` sidecar restores isolated nodes, per-component eigenvector centrality (max 1) |
| structural features | four, computed on the ORIGINAL graph and shared by every variant: degree, eigenvector centrality, Ψ (I2V KL→Poisson), clustering; z-normalised; disk-cached by content hash (`encoder.feature_cache`), verified a numerical no-op |
| virtual graphs | K = 10; variants `psi`, `degree`, `centrality`, `original` (control), `hybrid`; deterministic build at seed 42; ties broken by sampling within tie classes |
| encoder (primary) | GraphSAGE, mean aggregation, 2 layers, hidden 64, output 64, lr 0.01, Adam, 50 epochs, edge positives (A2), Q = 5 negatives with real-neighbour rejection, `pairs_per_epoch` 100,000, `max_pairs` 2,000,000 |
| encoder (second) | DeepWalk bridge over the same virtual graph (p = q = 1, `I2V_PARAMS`: dim 64, walk length 40, 10 walks, window 10) |
| node classification | one-vs-rest logistic regression, stratified 70/30, weighted F1 (core four); official OGB time split + Evaluator Accuracy, F1 secondaries (ogbn-arxiv) |
| link prediction | 70/30 edge split, retrain on the 70% graph only, unsupervised cosine ranking, AUC (core four); official OGB split + Evaluator, trained hadamard→256-MLP decoder fitted on train edges only, Hits@20 (ogbl-ddi) |
| seeds | 42 / 43 / 44 everywhere; virtual-graph build seed fixed at 42 |
| selection discipline | OGB winners chosen on validation and locked to `results/ogb_selection.json`; test read once, never selected on |
| drivers | `run_core.py` (core four) and `run_ogb.py` (OGB pair); all settings from `scripts/benchmark_config.py` |

### What "locked" is backed by

Every stored embedding was traced to the code that produced it by re-running that code, not by trusting file dates. Core four: pre-rejection encoder reproduces them to 1.9e-6 (noise floor) with identical metrics — they were stale, and have been retrained (37/40 rows moved, max 0.0054, zero winner flips). ogbl-ddi: the current encoder reproduces all four tested variants to 1e-6-2e-5 — already frozen-pipeline. ogbn-arxiv: reproduces under neither encoder at tensor level, with features, node order, node count, virtual graph and hyper-parameters all verified identical; the gap (mean 3.07e-3) is 33x below an independent-seed trajectory, and the metrics agree to 0.0012, inside the 3-seed std of 0.0020.

**Reproducibility claim for the paper, stated at the level the evidence supports:** results are reproducible at the reported metric precision. They are *not* bit-reproducible. Measured floors: a repeated run of one configuration changes the embedding by 1.5e-7 (arxiv) to 2e-6 (cora) and the metric not at all on the small graphs; changing CPU thread count moves arxiv by 7.6e-6 and cora by 2e-6; an independent seed moves arxiv by 1.0e-1. Large graphs amplify environment perturbation far more than small ones, which is why the arxiv tensors could not be re-derived while its scores could.

### Scope decisions taken with the lock

1. **Ablation-D rows are frozen as-is and will NOT be re-run.** They were tested, verified and frozen when the encoder was chosen; they are the evidence *for* the locked configuration rather than results *under* it. Their numbers predate the negative-rejection change and would move by ~0.001, which does not affect any ablation conclusion. The paper states this rather than hiding it.
2. **No density-matched controls on the core four.** `original_k`/`random_k` remain an ogbl-ddi-only analysis, valid there because the original graph is denser than K. On the core four the matching would run the other way (role graphs at K≈2) — a different experiment, deliberately out of scope.
3. **The two link-prediction scorers stay different, by protocol.** Core four use ViRGo's 70/30 split with cosine-ranking AUC; OGB uses its official split with a trained decoder and Hits@20. The characterization compares graph variants **within** a dataset, so the scorer is a constant that cancels; no table or figure places AUC and Hits@20 in one column, and no cross-dataset claim is made about metric magnitude.

### The result set the characterization consumes

Six datasets, ten dataset x task cells, each 5 graph variants x 2 encoders x 3 seeds at K = 10, all in `results/scoreboard.csv` (212 rows). cora, citeseer_linqs, enzymes and proteins carry both tasks; ogbn-arxiv is node classification only (no official link split) and ogbl-ddi is link prediction only (no labels) — properties of the datasets, not gaps. Graph properties per variant are in `results/graph_health.csv`.

Headline standing at the lock: the original graph wins all eight core-four cells; role graphs fail link prediction on every core dataset (role similarity is not adjacency); role graphs come within 0.01-0.02 of the original for molecular node classification and trail by 0.2-0.44 on citation graphs; and on ogbl-ddi with GraphSAGE every role graph beats the original (centrality 0.0519 vs 0.0173 test Hits@20) — the one augmentation win in the panel, on the densest graph by two orders of magnitude, and absent under DeepWalk.

**Next step (characterization only, no re-running):** compute per-dataset graph properties — homophily first, then degree spread, clustering, component fraction, label-vs-topology agreement — and relate them to the original-vs-best-augmented gap already in the scoreboard. Density enters as a candidate predictor because of the ogbl-ddi cell.
