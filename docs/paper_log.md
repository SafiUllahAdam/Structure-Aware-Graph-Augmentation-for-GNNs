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

Direction review. The post-fix results (original graph wins all 8 dataset × task cells; role graphs fail link prediction; role graphs approach the original only on molecular node classification) were presented and accepted as the honest result — no attempt to rescue a "virtual graph wins" headline.

**Reframed contribution.** ViRGo is a **study**: given a graph and a task, when is the original topology enough, and when do structural augmented features help? The deliverable is a **characterization** — connect each dataset's properties (homophily first, then degree spread, clustering, component fraction, label-vs-topology agreement) to the original-vs-best-augmented gap, so the paper can say "for graphs like this, keep the original; for graphs like that, add these features." This is the agreed framing, offered as a conjecture for the community, not a proof.

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

## 2026-07-24 (later) — Ablation D completed: all nine feature arms, one code version, plus two new single-feature arms

The ablation table was extended to isolate **every** structural input, and the whole table was re-run under the corrected encoder so no arm is compared across code versions. This reverses the earlier "ablations stay frozen" decision, deliberately: once the table gains new arms, freezing the rest makes the comparison internally inconsistent.

**Change.** `encoder.py` gained two `--features` options — `centrality` (D7, eigenvector centrality only) and `clustering` (D8, local clustering only) — selecting columns 1 and 3 of the same cached four-column feature matrix; registered in `cfg.D_FEATURES` and in notebook 3 §8b. `run_core.py` gained a `--features` passthrough so the ablation runs headless and in parallel like the main sweep. Verified before running: each single-feature arm reproduces exactly the z-normalised raw column it claims (degree→col 0, centrality→col 1, Ψ→col 2, clustering→col 3).

**Runs.** 7 trained arms x 4 datasets x 2 tasks x 3 seeds = 168 embeddings on the psi graph at K=10. D0 ("all") is the locked `graphsage_edge` row and was already post-fix; D6 (`features_only`, layers=0) never trains and was verified byte-identical under the current code, so neither was re-run. Re-running the five older arms moved them by at most 0.0209 (proteins D5 const LP, an arm that is chance by construction), mean 0.0016 — no ablation conclusion changed.

**Node classification (weighted F1), psi graph, K=10, 3 seeds**

| arm | citeseer | cora | enzymes | proteins |
|---|---|---|---|---|
| D0 all four | 0.2327 | 0.2351 | **0.5480** | **0.5588** |
| D1 degree | 0.1886 | 0.2153 | 0.5019 | 0.5059 |
| D2 degree+centrality | **0.2412** | **0.2462** | 0.5204 | 0.5349 |
| D3 Ψ | 0.1940 | 0.1743 | 0.5002 | 0.4901 |
| **D7 centrality only** | 0.2365 | 0.2328 | 0.5117 | 0.4983 |
| **D8 clustering only** | 0.1707 | 0.1791 | 0.5401 | 0.5449 |
| D4 random (control) | 0.1726 | 0.1497 | 0.4789 | 0.4753 |
| D5 constant (floor) | 0.0717 | 0.1406 | 0.3277 | 0.3184 |
| D6 features only, no MP | 0.2139 | 0.1834 | 0.4997 | 0.5216 |
| DeepWalk bridge | 0.2255 | 0.2133 | 0.5005 | 0.4882 |

**Link prediction (AUC), same setting**

| arm | citeseer | cora | enzymes | proteins |
|---|---|---|---|---|
| D0 all four | 0.5057 | 0.5150 | 0.6236 | 0.5578 |
| D1 degree | **0.5489** | 0.4960 | 0.5468 | 0.5389 |
| D2 degree+centrality | 0.5014 | 0.4864 | 0.6445 | **0.5914** |
| D3 Ψ | 0.4695 | 0.4594 | 0.6385 | 0.5483 |
| **D7 centrality only** | 0.4199 | 0.4349 | **0.6498** | 0.5491 |
| **D8 clustering only** | 0.4970 | **0.5298** | 0.5320 | 0.5054 |
| D4 random (control) | 0.5144 | 0.5134 | 0.5103 | 0.5105 |
| D5 constant (floor) | 0.4942 | 0.4970 | 0.4877 | 0.4737 |
| D6 features only, no MP | 0.4505 | 0.4831 | 0.6201 | 0.5661 |
| DeepWalk bridge | 0.5331 | 0.4990 | 0.5039 | 0.5232 |

**What the two new arms add — a per-feature, per-domain split.**

1. **Clustering is the molecular workhorse and dead weight on citation graphs.** Alone it reaches 0.5401 / 0.5449 node-classification F1 on enzymes / proteins — within 0.008-0.014 of all four features together, and the best single feature there — while on citeseer / cora it scores 0.1707 / 0.1791, at or barely above the random-feature control (0.1726 / 0.1497). Triangle structure carries molecular node identity and carries nothing about citation topics.
2. **Centrality is the opposite, and on citation link prediction it is worse than useless.** Alone it is the best arm for enzymes link prediction (0.6498, above all four features at 0.6236) but scores 0.4199 / 0.4349 on citeseer / cora link prediction — *below* the 0.50 chance line and below the constant-feature floor, i.e. ranking candidate edges by centrality similarity is actively anti-correlated with adjacency on citation graphs.
3. **More features is not better.** Degree+centrality (D2) beats all four (D0) on node classification for **both** citation graphs (0.2412 vs 0.2327; 0.2462 vs 0.2351) and on proteins link prediction (0.5914 vs 0.5578). The earlier cora-only observation now holds across the citation pair; adding Ψ and clustering dilutes the citation signal.
4. **Ψ remains the weakest structural feature and stays confounded** (the psi graph was built from it): worst real feature on node classification everywhere except enzymes link prediction, where it is mid-table.
5. The controls behave: constant is the floor everywhere, random sits between the floor and the real features, and every real feature beats random on the molecular graphs.

This is the per-feature evidence the study set out to produce, and it feeds Family 2 of the characterization directly: the arm that wins on a dataset should be predictable from that feature's raw distribution on that dataset.

**Caveat unchanged:** all of D is measured on the psi graph at K=10, on the four core datasets. The OGB pair has no ablation rows, and no arm was re-tuned per dataset.

**Plain-language summary of this entry:** `claude.ai/code/artifact/c6dd674b-8db1-4cf4-b340-929588bdd815` — the same tables in plain language, updated in place from the earlier A-E review page. Numbers there are read from this scoreboard; if either moves, both must be updated together.

## 2026-07-24 (later) — Ψ is numerically unstable where Ω underflows; the psi graph on proteins is not unique

Found while verifying that the untrained D6 arm reproduced. It did on cora and did not on proteins, and the difference was confined to the Ψ column.

**Mechanism, traced end to end.** 20 of proteins' 1,195 connected components fail networkx's power iteration and fall back to `eigenvector_centrality_numpy`, whose ARPACK solver starts from a **random** vector. That leaves Ω differing by ~7e-12 between runs. Ψ then evaluates `p·log(p/q)` with `q = Ω`: where Ω is itself ~1e-12, a 7e-12 absolute perturbation is a change of hundreds of percent, and the logarithm turns it into a Ψ shift of up to **52** on a range of −321 to −1. Cora is exactly stable (no failing components); enzymes moves one node by 1e-8.

**Consequence for the role graph.** Ψ is the 1-D signature that orders top-K neighbours, so on proteins two consecutive builds of the psi graph share only **33.6%** of their edges (Jaccard 0.3356); cora and enzymes rebuild identically (Jaccard 1.0). The saved psi graphs used for node classification are fixed on disk, so those results are unaffected; link prediction rebuilds the role graph from the 70% training edges on every run, so proteins link prediction is exposed.

**Consequence for the numbers: negligible.** Two full rebuild-and-retrain passes of proteins link prediction on the psi graph gave AUC 0.5614 and 0.5619 against the recorded 0.5617 — a spread of 0.0005, well inside that cell's 3-seed std of 0.0042. So the proteins role graph is **highly non-unique but the metric is insensitive to which draw is realised**, which is worth reporting as a robustness result rather than hidden as a defect.

**Containment already in place.** Since the feature cache was introduced, `run_core.py` and `run_ogb.py` compute Ψ once per graph and reuse it, so every current result shares one Ψ vector. Notebook 3 does **not** pass the cache, so a notebook re-run would recompute Ψ and could differ from the scripted pipeline on proteins. Recorded as a known gap; the clean fixes, none of them applied here, are to seed the ARPACK fallback with a fixed `v0`, to floor Ω before the logarithm, or to make the notebook use the cache like the drivers do.

## 2026-07-24 (later) — Characterization portion 1: both families measured

Portion 1 is **measurement only** — no joins, no correlations, no claims about which property predicts anything. That is portion 2. What exists now is a frozen, reproducible description of each dataset and of the exact numbers the encoder is fed.

**Method.** One new script, `characterize.py`, reads only artifacts that already exist: the dataset edgelists (through `graph_io.load_graph`, so the graph definition cannot drift), the encoder's own **feature caches**, and `results/scoreboard.csv`. Family 2 deliberately reads the cache rather than recomputing the features — the cache holds the raw, pre-z-normalization matrix that every embedding in the study was actually trained on, so the description is of the real inputs, and it is immune to the Ψ recomputation instability recorded earlier. Runtime 17 s for all six datasets; three re-runs produce byte-identical CSVs.

### Family 1 — graph characterization (measured on the original graph)

| dataset | domain | nodes | edges | avg degree | max degree | degree Gini | density |
|---|---|---|---|---|---|---|---|
| cora | citation | 2,708 | 5,278 | 3.90 | 168 | 0.4051 | 0.00144 |
| citeseer_linqs | citation | 3,264 | 4,536 | 2.78 | 99 | 0.4353 | 0.000852 |
| enzymes | biological | 19,474 | 37,282 | 3.83 | 9 | 0.1569 | 0.000197 |
| proteins | biological | 43,466 | 81,044 | 3.73 | 25 | 0.1634 | 0.000086 |
| ogbn-arxiv | citation | 169,343 | 1,157,799 | 13.67 | 13,161 | 0.6295 | 0.000081 |
| ogbl-ddi | drug interaction | 4,267 | 1,067,911 | 500.54 | 2,234 | 0.4689 | 0.117333 |

| dataset | components | largest comp. | avg clustering | assortativity | classes | edge homophily | node homophily |
|---|---|---|---|---|---|---|---|
| cora | 78 | 91.8% | 0.2407 | −0.0659 | 7 | **0.8100** | 0.8252 |
| citeseer_linqs | 390 | 64.6% | 0.1447 | +0.0481 | 6 | **0.7377** | 0.7203 |
| enzymes | 640 | **0.6%** | 0.4024 | +0.1755 | 3 | 0.6653 | 0.6682 |
| proteins | 1,195 | **1.4%** | 0.3814 | +0.1516 | 3 | 0.6568 | 0.6513 |
| ogbn-arxiv | 1 | 100% | 0.2261 | −0.0431 | 40 | 0.6542 | 0.6353 |
| ogbl-ddi | 1 | 100% | 0.5143 | +0.0378 | — | n/a (unlabelled) | n/a |

**External validation of the homophily column.** Measured edge homophily reproduces the published values for the three datasets where one exists: cora 0.8100 vs ~0.81, citeseer 0.7377 vs ~0.736, ogbn-arxiv 0.6542 vs ~0.654. The property that the study nominated as its primary predictor is therefore being computed correctly, which was not previously verified anywhere in this project.

**What the table establishes as fact (not yet as explanation).** The six datasets separate cleanly on three axes at once, and the axes are not independent of each other: the citation graphs are homophilous (0.65–0.81), sparse, slightly disassortative and low-clustering; the two molecular graphs are the opposite on every count — assortative (+0.15 to +0.18), high-clustering (0.38–0.40), degree-regular (Gini 0.16, max degree 9 and 25) and **extremely fragmented**, with the largest component holding only 0.6% (enzymes) and 1.4% (proteins) of nodes; ogbl-ddi is the density outlier by two orders of magnitude (average degree 500, density 0.117). Fragmentation is the axis that had not been quantified before: enzymes and proteins are effectively collections of a few hundred small graphs, which is exactly the regime where a role graph can connect nodes that no path connects.

### Family 2 — the structural inputs, raw and pre-normalization

| dataset | feature | mean | std | skew | % zero | % unique |
|---|---|---|---|---|---|---|
| cora | clustering | 0.2407 | 0.3220 | 1.31 | **45.7** | 4.84 |
| citeseer_linqs | clustering | 0.1447 | 0.2877 | 2.11 | **69.1** | 3.46 |
| enzymes | clustering | 0.4024 | 0.2979 | 0.38 | 19.7 | 0.22 |
| proteins | clustering | 0.3814 | 0.3151 | 0.47 | 25.0 | 0.13 |
| ogbn-arxiv | clustering | 0.2261 | 0.2438 | 1.55 | 27.0 | 5.99 |
| ogbl-ddi | clustering | 0.5143 | 0.2191 | −0.34 | 4.4 | 88.49 |
| cora | eigenvector centrality | 0.0772 | 0.2459 | 3.29 | 2.6 | 87.08 |
| citeseer_linqs | eigenvector centrality | 0.2989 | 0.4189 | 0.87 | 12.5 | 62.87 |
| enzymes | eigenvector centrality | 0.3346 | 0.3220 | 0.62 | 0.1 | 85.30 |
| proteins | eigenvector centrality | 0.2700 | 0.3206 | 0.94 | 0.9 | 86.16 |
| ogbn-arxiv | eigenvector centrality | 0.0020 | 0.0064 | 47.75 | 24.8 | 96.03 |
| ogbl-ddi | eigenvector centrality | 0.2428 | 0.2206 | 0.61 | 0.0 | 91.99 |

Degree and Ψ rows are in `results/node_feature_characterization.csv`; the headline facts there are that degree is **coarse** wherever the graph is degree-regular (9 distinct degrees across 19,474 enzymes nodes; 0.03–0.32% unique on proteins, citeseer and arxiv) and that Ψ's raw scale spans three orders of magnitude across datasets (mean −14 on cora, −16,622 on ddi) — a spread that z-normalization removes before the encoder ever sees it, which is precisely why it has to be measured here.

**The observation Family 2 exists to make.** Local clustering is **45.7% and 69.1% exactly zero** on cora and citeseer — most citation nodes sit in no triangle at all — against 19.7% and 25.0% on the molecular graphs and 4.4% on ddi. This is the raw-distribution counterpart of the ablation-D result that clustering-only is the best single feature on molecular node classification and lands at the random-feature control on the citation graphs. Portion 2 will state that correspondence properly, across all four features and all ten cells; it is recorded here only as the measurement that makes the test possible.

**A correction to an earlier belief.** An older note held that eigenvector centrality was degenerate on enzymes (96.9% of nodes below 1e-6). That was true of the **global** Ω used before the per-component policy; under the current `GRAPH_POLICY` (per-component, rescaled to max 1) centrality is one of the best-spread features in the study — 85.3% distinct values on enzymes, 0.09% zeros. The belief is withdrawn, and it is consistent with centrality-only being the strongest single arm for enzymes link prediction.

**Caveat on the `degenerate` flag.** The flag (`% unique < 1` or `% zero > 95`) fires on molecular clustering, which the ablation shows to be the most useful single feature there — few distinct values because small-degree nodes admit only a handful of rational clustering coefficients, not because the feature is uninformative. **Portion 2 must not use the boolean**; the raw columns beside it carry the real signal.

### Step 1 — the frozen experimental side

`results/characterization_inputs.csv` snapshots the locked GraphSAGE scores this will be joined against: 50 rows = 10 dataset x task cells x 5 graph variants, each the 3-seed mean and std of that task's single primary metric (weighted F1 / AUC for the core four, official Accuracy / Hits@20 for the OGB pair). Metrics are never mixed into one column, per the within-dataset rule.

## 2026-07-24 (later still) — Adjusted homophily added to the characterization

The homophily column in the Family-1 table above is **raw edge homophily**, and its chance baseline depends on how many classes a graph has and how balanced they are — near 0.5 for a 3-class graph, near 0.03 for a 40-class one. Raw values are therefore **not comparable across datasets with different class structure**, which matters because homophily is the predictor the study leans on first. Two rows made the flaw concrete: enzymes (0.6653, 3 classes) and ogbn-arxiv (0.6542, 40 classes) sit almost on top of each other on the raw scale while being opposite graphs on every other axis.

**Fix — one measured column, no pipeline change.** `characterize.py`'s `homophily()` now also returns the chance null and adjusted homophily (Platonov et al.),

  `h_adj = (h_edge − Σ_c (D_c / 2m)²) / (1 − Σ_c (D_c / 2m)²)`,  D_c = summed labelled-degree of class c,  2m = Σ_c D_c,

written to `dataset_characterization.csv` as `homophily_null` and `homophily_adjusted`; raw `edge_homophily` and `node_homophily` stay beside them for provenance. The null is **degree-weighted**, not node-count-weighted (Σ p²), because edge homophily counts edges and edges are weighted by degree; the two nulls agree only when the graph is degree-regular.

| dataset | classes | edge homophily (raw) | null Σ(D_c/2m)² | **adjusted** |
|---|---|---|---|---|
| cora | 7 | 0.8100 | 0.1698 | **0.7711** |
| citeseer_linqs | 6 | 0.7377 | 0.1975 | **0.6731** |
| ogbn-arxiv | 40 | 0.6542 | 0.1614 | **0.5877** |
| enzymes | 3 | 0.6653 | 0.4759 | **0.3613** |
| proteins | 3 | 0.6568 | 0.4678 | **0.3552** |
| ogbl-ddi | — | n/a | n/a | n/a |

**What it buys the study.** Adjusting separates the two domains where the raw column hid them: citation graphs cluster at **0.59–0.77**, molecular graphs at **~0.36**, with no overlap, and the enzymes/arxiv collision resolves (arxiv 0.588 against enzymes 0.361). Degree-weighting earns its keep specifically on arxiv — a node-count null would give ~0.625 there, the degree-weighted null gives 0.588, because arxiv is the one graph with heavy degree spread (Gini 0.63); the degree-regular molecular graphs barely move either way. **Portion 2 correlates against `homophily_adjusted`; raw is kept only for provenance.** The re-run touched only the three characterization CSVs and is byte-stable; scoreboard, embeddings and splits are unchanged.

## 2026-07-25 — Characterization portion 2: the when-to-augment rule

Portion 1 measured; portion 2 **relates**. The gap being explained is computed **inside** one dataset × task cell — `best augmented − original`, where "augmented" is the best of `psi` / `degree` / `centrality` / `hybrid` — so the metric is a constant that cancels and no AUC is ever compared against a Hits@20. Only the *ranking* of those gaps is compared across datasets, which is why every correlation below is Spearman. Code is `characterize.py --step all` (portion 1 unchanged, four functions added); the figures are `notebooks/5-phase5_characterization.ipynb` → `results/figures/`.

### The outcome per cell

A gap is called only when it clears the pooled 3-seed noise of the two sides it compares (1σ band); inside that band the cell is a **tie**, which is what makes "the original wins or ties everywhere" a measured statement rather than a reading of point estimates.

| dataset | task | metric | original | best augmented | variant | gap | σ | verdict |
|---|---|---|---|---|---|---|---|---|
| cora | NC | weighted F1 | 0.4266 | 0.2942 | centrality | −0.1324 | −8.6 | keep original |
| citeseer_linqs | NC | weighted F1 | 0.3298 | 0.2818 | centrality | −0.0480 | −3.8 | keep original |
| ogbn-arxiv | NC | Accuracy | 0.3918 | 0.3618 | hybrid | −0.0300 | −6.2 | keep original |
| proteins | NC | weighted F1 | 0.5797 | 0.5646 | hybrid | −0.0151 | −8.9 | keep original |
| enzymes | NC | weighted F1 | 0.5591 | 0.5546 | hybrid | −0.0045 | −0.7 | **tie** |
| cora | LP | AUC | 0.6130 | 0.5659 | centrality | −0.0471 | −11.5 | keep original |
| citeseer_linqs | LP | AUC | 0.6218 | 0.5437 | centrality | −0.0781 | −3.6 | keep original |
| enzymes | LP | AUC | 0.7000 | 0.6563 | centrality | −0.0437 | −6.1 | keep original |
| proteins | LP | AUC | 0.6720 | 0.5834 | centrality | −0.0886 | −23.3 | keep original |
| **ogbl-ddi** | LP | Hits@20 | 0.0173 | **0.0519** | centrality | **+0.0346** | **+2.7** | **augment** |

**8 keep original · 1 tie · 1 augment.** The original graph holds rank 1 of 5 in nine of ten cells; on ogbl-ddi it is rank 5 of 5 — the only cell where every augmented graph beats it.

### The rule: the predictor is task-dependent

This is the substantive finding, and it was not the expected one. Homophily was nominated as *the* predictor; it turns out to predict **node classification only**, while link prediction answers to density instead.

| predictor | NC (n=5) | LP (n=5) | pooled (n=10) |
|---|---|---|---|
| **homophily_adjusted** | **−0.90** | +0.40 (n=4) | −0.55 |
| degree_assortativity | +0.90 | −0.20 | +0.39 |
| **avg_degree** | −0.10 | **+0.80** | +0.39 |
| density | −0.60 | +0.70 | −0.09 |
| avg_clustering | +0.70 | +0.70 | **+0.72** |
| degree_skew | −0.70 | −0.50 | −0.60 |

- **Node classification — the more homophilous the graph, the worse augmentation does** (ρ = −0.90, near-monotone: cora 0.771 → −31% gap, citeseer 0.673 → −15%, arxiv 0.588 → −8%, proteins 0.355 → −2.6%, enzymes 0.361 → −0.8%). Homophilous labels are a *community* property carried by the real edges; role-based rewiring discards exactly that signal. Where homophily is weak — the molecular graphs — role structure is nearly sufficient and the gap closes to a tie.
- **Link prediction — augmentation only becomes competitive as the graph gets dense** (avg degree ρ = +0.80). ogbl-ddi (avg degree 500) is the single augmentation win; the five sparse graphs (2.8–13.7) all keep the original. Note **ogbl-ddi carries no labels, so it is absent from the homophily column** — the LP homophily correlation rests on n = 4 and should not be quoted.
- **avg_clustering is the only property that holds the same sign in both tasks** (+0.70 / +0.70), making it the best single-column summary if one is wanted.
- `degree_assortativity` (+0.90 on NC) is a **mirror of homophily, not independent evidence** — the molecular graphs are simultaneously assortative and weakly homophilous, so with n = 5 the two cannot be separated.

**Strength of claim.** n is 4–5 datasets per task. |ρ| = 0.90 at n = 5 is p ≈ 0.037, but with twelve properties tested this does not survive any multiple-comparison correction. These are stated as a **conjecture offered to the community**, consistent with the project's framing — the direction and the near-monotonicity are the evidence, not the p-value.

### Feature usefulness: the `degenerate` flag is empirically dead

Ablation D, `psi` graph, K=10, core four only, measured as **lift over the random-feature control (D4)** — the honest zero point, since it is message passing with the structural signal removed.

| dataset | task | best single feature | lift | that feature's % zero | % unique |
|---|---|---|---|---|---|
| cora | NC | centrality | +0.083 | 2.6 | 87.1 |
| citeseer_linqs | NC | centrality | +0.064 | 12.5 | 62.9 |
| enzymes | NC | clustering | +0.061 | 19.7 | **0.22** |
| proteins | NC | clustering | +0.070 | 25.0 | **0.13** |
| enzymes | LP | centrality | +0.140 | 0.1 | 85.3 |
| proteins | LP | centrality | +0.039 | 0.9 | 86.2 |
| cora | LP | clustering | +0.016 | 45.7 | 4.8 |
| citeseer_linqs | LP | degree | +0.035 | 0.0 | **0.95** |

Across the 32 (dataset × task × single feature) cells, **raw spread does not predict usefulness**: %zero ρ = −0.06, %unique ρ = −0.02, skew ρ = +0.01. The winners on the molecular graphs are the two features the flag calls degenerate — clustering at 0.13–0.22% unique, and degree at 0.95% unique on citeseer LP. This is a **measured refutation** of the flag rather than an argument against it: a feature with few distinct values still separates the nodes that matter. The flag stays in `node_feature_characterization.csv` as description only; **usefulness is decided by ablation score throughout**.

### Files

`results/characterization_gaps.csv` (10 cells) · `results/characterization_correlations.csv` (48 rows, both scopes) · `results/feature_usefulness.csv` (74 rows) · `results/figures/fig1–fig5.png`. Deliverable #5 is complete; GIN (#6) is the remaining experimental item.

---

## 2026-07-27 — Panel extension: two heterophilous datasets + a pre-registered prediction

**Why.** The ten-cell characterization produced one `augment` verdict (ogbl-ddi LP). A decision rule needs points on
both sides of its boundary; with a single positive cell the boundary is fitted from one point and the `augment` side
of the rule was never tested. The cause is visible in the panel itself: **all six datasets are homophilous**
(adjusted homophily 0.355–0.771), and average degree has an empty span between 13.7 (ogbn-arxiv) and 500 (ogbl-ddi).
The rule's two predictors were therefore each measured over a range that excludes the region where the rule predicts
augmentation. Adding datasets there is not a search for wins — it is the missing half of the experiment.

**Datasets added** — Platonov et al. (2023), *A critical look at the evaluation of GNNs under heterophily*; the same
paper this study's adjusted-homophily definition is taken from.

| dataset | domain | nodes | edges | avg degree | classes | edge homophily | **adjusted homophily** | components | assortativity |
|---|---|---|---|---|---|---|---|---|---|
| roman_empire | linguistic (word adjacency + dependency arcs) | 22,662 | 32,927 | 2.91 | 18 | 0.0469 | **−0.0468** | 1 | −0.028 |
| tolokers | crowdsourcing (workers who shared a task) | 11,758 | 519,000 | 88.28 | 2 | 0.5945 | **0.0926** | 1 | −0.080 |

Both reproduce the published counts exactly, and our independent `homophily()` implementation reproduces the paper's
adjusted homophily to two decimals (−0.05 / 0.09) — an **external validation of the metric**, not only of the data.

**What they add to the panel.** roman_empire is the first dataset with *negative* adjusted homophily, extending the
range from [0.355, 0.771] to [−0.047, 0.771]. tolokers lands at degree 88.28, inside the previously empty span
between ogbn-arxiv and ogbl-ddi, and is only the second graph in the panel with weak homophily *and* high density.

**Protocol — unchanged, deliberately.** Structural features only: `data.x` (roman_empire 300-dim fastText,
tolokers 10-dim worker profile) is ignored, exactly as OGB's text features are. Platonov's own ten train/val/test
masks are also ignored; both datasets run the ViRGo **core** protocol (stratified 70% node classification → weighted
F1; 70:30 link prediction → AUC) so their cells stay directly comparable with the four core datasets. K=10,
GraphSAGE mean/2-layer/edge-positives, seeds 42/43/44 — the frozen pipeline, with **no retuning**. Retuning on the
new datasets would void the held-out status of the prediction below.

**Note on link prediction.** Neither dataset ships an LP task; the 70:30 split is ours, as for the core four. This is
a defined extension of the benchmark, not a published protocol, and is reported as such.

### Pre-registered prediction (recorded before any embedding was trained)

The rule from 2026-07-25 states: node classification augments as adjusted homophily *falls* (ρ = −0.90, n=5); link
prediction augments as average degree *rises* (ρ = +0.80, n=5). Applied to the two new datasets it predicts:

| dataset | task | driving property | value | rank in panel | **predicted verdict** |
|---|---|---|---|---|---|
| roman_empire | NC | adjusted homophily | −0.047 | lowest of 7 | **augment** |
| tolokers | NC | adjusted homophily | 0.093 | 2nd lowest of 7 | **augment** |
| tolokers | LP | avg degree | 88.28 | 2nd highest of 7 | **augment** |
| roman_empire | LP | avg degree | 2.91 | lowest of 7 | **keep original** |

Three augment, one keep. The fourth is the control: roman_empire is simultaneously the panel's least homophilous
graph *and* its sparsest, so the two predictors disagree on it by construction — NC should augment while LP should
not. A result that splits that way is evidence the predictor really is task-dependent rather than a single latent
"hard dataset" axis; a result that augments both would mean the two correlations are measuring the same thing.

**Falsification condition, stated in advance.** If roman_empire and tolokers NC return `keep original`, the
homophily rule is falsified on the only range that could test it, and the study reports that. The value of the
extension does not depend on which way it comes out — only on the prediction having been fixed beforehand.

**Panel after this change:** 8 datasets, 14 dataset × task cells (NC 7, LP 7). Correlation n rises from 5 to 7 per
task. Registered in `scripts/benchmark_config.DATASETS`, `characterize.STUDY`, and `run_core.CORE`; converter is
`make_hetero.py`. Nothing scored yet.

---

## 2026-07-27 — Result of the pre-registered test: the homophily rule is falsified

The 2026-07-27 run of `run_core.py --datasets roman_empire tolokers` completed (40 cells: 5 variants × 2 encoders ×
2 tasks × 3 seeds, K=10, frozen pipeline, no retuning). Scores are rows 228–267 of `results/scoreboard.csv`.

### Outcome vs prediction

| dataset | task | original | best augmented | gap | σ | **verdict** | predicted |
|---|---|---|---|---|---|---|---|
| roman_empire | NC | **0.2561 ± 0.0092** | hybrid 0.2144 ± 0.0076 | −0.0417 | −4.94 | keep original | ~~augment~~ |
| tolokers | NC | **0.7345 ± 0.0037** | hybrid 0.7265 ± 0.0060 | −0.0080 | −1.60 | keep original | ~~augment~~ |
| tolokers | LP | 0.6444 ± 0.0188 | **psi 0.7007 ± 0.0032** | +0.0563 | +4.18 | **augment** | augment ✓ |
| roman_empire | LP | 0.6019 ± 0.0124 | **centrality 0.6980 ± 0.0139** | +0.0961 | +7.30 | **augment** | ~~keep original~~ |

One of four predictions held. **The falsification condition stated in advance was met**: both new NC cells returned
`keep original`, on the only homophily range that could have tested the rule.

### What the correlations did

Recomputed by `characterize.py --step all` with n = 7 per task (was 5):

| relation | before (n=5) | after (n=7) |
|---|---|---|
| NC gap ~ adjusted homophily | ρ = −0.90 | **ρ = −0.36** |
| LP gap ~ average degree | ρ = +0.80 | **ρ = +0.54** |

Both drivers lost most of their strength out of sample. The strongest predictors on the extended panel are now
LP ~ `components` (ρ = −0.85), `largest_component_frac` (+0.74), `avg_clustering` (+0.71); NC ~ `n_classes`
(ρ = −0.68), `avg_clustering` (+0.57). These are **post-hoc on the same data that broke the first rule** and are
recorded as candidates to be tested, not as a replacement rule. The honest statement is that a two-property rule
fitted on five points did not survive two new points.

### Why the homophily premise was wrong

roman_empire has adjusted homophily −0.047 — neighbouring words rarely share a syntactic role — yet the original
graph beats every role graph by 4.9σ. Low adjusted homophily means neighbour labels **differ**, not that the edges
carry no information: a word's role is determined by its sequence context, and differing *predictably* is still
exploitable signal that message passing can use. The rule conflated "neighbours share my label" with "neighbours are
informative about my label". Only a heterophilous dataset could expose that conflation, which is precisely why the
panel needed one.

### Panel-level effect

`keep original` 10 | `augment` 3 | `tie` 1, over 8 datasets / 14 cells. The augment count rose from 1 to 3, so the
characterization no longer rests on a single cell — but the two new augment verdicts are both link prediction, and
neither was obtained where the rule said it would be.

### Caveat on the roman_empire LP cell — do not read it as support for role edges

Node ids in roman_empire are word positions in the source text, so the graph is a 22,662-node path plus dependency
arcs: 68.8% of edges join consecutive ids, 14.7% join ids two apart. Link prediction there is largely "are these two
ids adjacent", which any position-sensitive embedding solves — DeepWalk on the original graph reaches
AUC 0.9994 ± 0.0000 while GraphSAGE on the same graph reaches 0.6019. The `augment` verdict in that cell therefore
records **GraphSAGE failing to represent a path**, not role edges helping link prediction. Reported with this caveat
attached; whether the cell is retained in the correlations is an open decision.

**Artifacts:** `results/scoreboard.csv` (70 GraphSAGE score rows), `results/characterization_*.csv`,
`results/figures/fig1–fig5` regenerated, notebook 5 executed end to end on the 8-dataset panel.

---

## 2026-07-27 — Why the original graph wins node classification on the heterophilous pair (audit, no code changed)

Read-only audit of the roman_empire / tolokers node-classification cells. Implementation verified correct; the
`keep original` verdicts are real, and the mechanism turns out **not** to be the one the homophily rule assumed.

### Implementation checks (all pass)

| check | roman_empire | tolokers |
|---|---|---|
| labels ↔ graph node alignment | 22,662 / 22,662, 0 unlabelled, 0 orphan | 11,758 / 11,758, 0 / 0 |
| nodes present in every embedding | 22,662 (all 5 variants) | 11,758 (all 5 variants) |
| nodes silently skipped by the evaluator | 0 | 0 |
| virtual graphs: isolates / star collapse | 0 isolates, max degree 14–35 | 0 isolates, max degree 18–2138 |
| stored embeddings reproduce the scoreboard | yes (seed 42 within seed spread) | yes |
| degenerate (duplicate) embedding vectors | ≤ 2 of 22,662 | ≤ 157 of 11,758 |
| majority-class weighted-F1 floor | 0.0342 (all variants ≈ 6–7× above it) | 0.6860 (78/22 binary) |

### The mechanism: predictability, not agreement

Same-label rate across each graph's edges, against the chance rate for that class distribution:

| graph | roman_empire same-label (chance 0.0880) | neighbour-label entropy | tolokers same-label (chance 0.6588) | entropy |
|---|---|---|---|---|
| original | **0.0469 (−0.0412)** | **0.913** | **0.5945 (−0.0643)** | 0.568 |
| psi | 0.0988 (+0.0108) | 1.862 | 0.6876 (+0.0288) | 0.503 |
| degree | 0.1169 (+0.0288) | 2.001 | 0.6750 (+0.0162) | 0.510 |
| centrality | 0.0894 (+0.0014) | 2.082 | 0.6760 (+0.0172) | 0.517 |
| hybrid | 0.0882 (+0.0002) | 1.998 | 0.6087 (−0.0501) | 0.573 |

The original graph is the **least** label-agreeing graph of the five on both datasets — below chance — and still wins.
The role graphs raise same-label agreement above chance and still lose. Agreement is therefore not what the encoder
uses. What separates them is the **entropy of a node's neighbour-label distribution**: 0.913 for the original graph
against 1.86–2.08 for the role graphs (maximum ln 18 = 2.89). Original edges give each node a sharply peaked
neighbour-label mix; role edges connect structurally similar words drawn from all over the corpus, so the mix
collapses onto the global prior and message passing returns approximately the prior.

### The control that settles it

Raw structural features fed straight to the same logistic regression, **no message passing** (the D6 arm, never run
on these datasets before):

| | raw features only | GraphSAGE on role graphs | GraphSAGE on original |
|---|---|---|---|
| roman_empire | 0.1915 | 0.2039 – 0.2095 | **0.2462** (s42) / 0.2561 (3 seeds) |
| tolokers (weighted) | 0.7101 | 0.7116 – 0.7205 | **0.7367** |
| tolokers (macro) | 0.4982 | 0.4959 – 0.5184 | **0.5492** |

**GraphSAGE over a role graph is worth about as much as not doing message passing at all.** Role rewiring does not
damage the signal — it fails to add one, and every role variant lands within ~0.01 of the feature-only control.
Only the original edges contribute information beyond the four input features (+0.055 on roman_empire, +0.051 macro
on tolokers). This is a sharper statement of the same conclusion as ablation D on the core four.

### Caveats

- tolokers node classification is a 78/22 binary: the majority floor is 0.6860, so the entire five-variant spread
  (0.7116–0.7367) lives inside 0.05 of usable headroom. Its −1.60σ verdict is the weakest in the panel and sits just
  outside the tie band — report it as marginal.
- Absolute numbers are far below Platonov's published GraphSAGE results because those use the 300-dim fastText node
  features, which our scope rule excludes. The comparison here is between graphs at fixed features, not against the
  literature.

---

## 2026-07-27 — Panel extension: `questions`, and a head-to-head between two competing rules

Third heterophilous benchmark added (Platonov et al. 2023), same converter and same frozen pipeline as roman_empire
and tolokers. Chosen because it is the one dataset that **separates the two explanations now on the table**.

### The dataset

Users of a Q&A website; an edge means one user answered the other's question. Built by `make_hetero.py --dataset
questions`. Our measurements reproduce the published values exactly, a second external validation of the loader and
of `homophily()`:

| property | published | measured here |
|---|---|---|
| nodes / edges | 48,921 / 153,540 | 48,921 / 153,540 |
| adjusted homophily | 0.02 | **0.0207** |
| average degree | 6.28 | 6.28 |
| average local clustering | 0.03 | 0.0307 |

Additional structure: one connected component, degree median 1 against maximum 1,539 (heavier tail than any other
non-OGB dataset in the panel), edge homophily 0.8396.

Structural-only as always: the 301-dim fastText node features and Platonov's ten official masks are ignored; the
ViRGo core protocol applies (stratified 70% node classification → weighted F1; 70:30 link prediction → AUC), K = 10,
GraphSAGE mean/2-layer/edge-positives, seeds 42/43/44, **no retuning**.

### Why this dataset and not another

After roman_empire and tolokers, two incompatible explanations fit the panel:

- **H1, the homophily rule** (2026-07-25): node classification augments as adjusted homophily falls. Already
  weakened out of sample (ρ −0.90 → −0.36) but not dead.
- **H2, the mechanism** (2026-07-27 audit): role edges are a *function of the four input features*, so they cannot
  carry information about a target that is not itself structural. Labels are not structural; adjacency is. H2
  predicts node classification can **never** augment, and that the three observed augment cells are all link
  prediction because that is the only structural target.

`questions` has adjusted homophily 0.0207 — second lowest in the panel, deep inside the region where H1 predicts
augmentation — while H2 predicts `keep original`. The two hypotheses disagree on this cell, which is the reason it
was selected.

### Pre-registered prediction (recorded before any embedding was trained)

| dataset | task | H1 (homophily rule) | H2 (mechanism) | **registered prediction** |
|---|---|---|---|---|
| questions | NC | augment (h_adj = 0.021) | keep original | **keep original** (H2) |
| questions | LP | keep original (avg deg 6.28) | keep original | **keep original** |

The registered call follows H2. Link prediction is not discriminating — both hypotheses say keep original, because
6.28 is far from the density at which the over-smoothing failure appears (tolokers 61.8 in-train, ogbl-ddi ~500) and
the graph is not a near-path, so roman_empire's smooth-centrality route does not apply either.

**Falsification, stated in advance.** A `keep original` on node classification retires H1 for good: three
heterophilous datasets in a row would have sat in its predicted-augment region and refused to augment. An `augment`
falsifies H2, which claims the outcome is impossible, and revives the homophily reading.

### Caveat to read the metric with

`questions` is a **97.0 / 3.0** binary (47,461 vs 1,460). Weighted F1 is therefore dominated by the majority class
and will sit near the trivial floor for every variant; the informative column in this cell is **macro F1**, exactly
as on tolokers (78/22) but more extreme. The verdict is still computed on the panel's primary metric for
comparability, and the compression is reported rather than corrected.

**Panel after this change:** 9 datasets, 16 dataset × task cells (NC 8, LP 8). Registered in
`scripts/benchmark_config.DATASETS`, `characterize.STUDY`, `run_core.CORE`; built by `make_hetero.py`.
Nothing scored yet.

---

## 2026-07-27 — `questions` result: prediction half-failed, and a robustness test that splits the augment verdicts in two

The `questions` sweep completed (20 score rows = 5 variants × 2 encoders × 2 tasks, 3 seeds each, frozen pipeline).

### Node classification: a dead cell

Every one of the ten node-classification rows returned **0.9555 ± 0.0000**, which is exactly the all-majority-class
weighted F1 for a 97.0/3.0 split. All five variants, both encoders, three seeds, zero variance: the classifiers are
constant predictors. The verdict prints as `keep original` with `gap = 0.0000` and an undefined sigma.

The pre-registration selected `questions` precisely because H1 (homophily rule) and H2 (mechanism) disagreed on this
cell. **They were not tested.** The metric collapsed before either hypothesis could be discriminated, so this cell
contributes nothing to that question and is reported as uninformative rather than as support for either side. The
class imbalance was flagged in advance; its severity was underestimated.

### Link prediction: the registered prediction failed, then survived a robustness check

Registered: `keep original`. GraphSAGE returned **augment** — original 0.4665 ± 0.0130, best augmented (hybrid)
0.5526 ± 0.0232, gap +0.0861 = +4.58σ. Taken alone that is a fourth augment verdict and a failed prediction.

But GraphSAGE on the original graph scored **0.4665 — below chance**. That prompted running every link-prediction
cell through the second encoder already in the pipeline (the DeepWalk bridge, same splits, same seeds):

| dataset | avg degree | GraphSAGE: winner | DeepWalk: winner | verdict survives encoder change? |
|---|---|---|---|---|
| ogbl_ddi | ~500 | **augment** (0.0173 → 0.0554) | **augment** (0.0378 → 0.1038) | **yes** |
| tolokers | 88.3 | **augment** (0.6444 → 0.7007) | **augment** (0.7859 → 0.8954) | **yes** |
| roman_empire | 2.9 | augment (0.6019 → 0.6980) | keep original (**0.9994** → 0.8065) | **no** |
| questions | 6.3 | augment (0.4665 → 0.5526) | keep original (0.6589 → 0.6537) | **no** |
| cora | 3.9 | keep original | keep original | yes (control) |
| enzymes | 3.9 | keep original | keep original | yes (control) |

**Two of the four augment verdicts are GraphSAGE artifacts.** On roman_empire and questions the augment verdict
appears only because GraphSAGE on the original graph is broken — 0.4665 (below chance) and 0.6019 against DeepWalk's
0.9994 on the identical split. Swap the encoder and the original graph wins both. On ogbl_ddi and tolokers the
augment verdict is reproduced by an unrelated encoder, so it is a property of the graph, not of GraphSAGE.

For `questions` specifically, the registered `keep original` is what the encoder-robust reading gives; the GraphSAGE
row that contradicted it is the artifact.

### Refined statement of the link-prediction rule

The two encoder-robust augment cells are the panel's two densest graphs (ogbl_ddi ~500, tolokers 88.3); the two
artifacts are sparse (2.9, 6.3). This restores average degree as the link-prediction predictor **once the
encoder-artifact cells are removed**, and supplies the mechanism already measured on tolokers: at high density a
2-layer receptive field covers ~46% of the graph and 24.6% of non-edges already share a neighbour (core four:
0.6–9.7%), so the original graph cannot separate edges from non-edges while a sparser role graph can.

The claim the study can defend is therefore narrower and better supported than either earlier version:
**role-based augmentation helps link prediction on dense graphs, does not help on sparse ones, and never helps node
classification.** Every apparent exception in the panel is a case where the baseline encoder failed.

### Panel state

9 datasets, 16 dataset × task cells: 11 keep original, 4 augment (all link prediction), 1 tie — and of the 4
augment cells only 2 survive an encoder swap. No node-classification cell has ever augmented, across 8 datasets
spanning adjusted homophily −0.047 to 0.771.

---

## 2026-07-27 — Ablation D completed panel-wide, plus a feature-status taxonomy and a degenerate control on the OGB pair

### Coverage closed

Ablation D previously existed only for the core four. It now covers **all nine datasets** (psi graph, K=10,
GraphSAGE only, 7 arms × 3 seeds): roman_empire, tolokers and questions via `run_core.py --features`, and the OGB
pair via a new `--features` flag added to `run_ogb.py` (mirrors `run_core.py`; ablation arms return before
`select()`/`report()` so they cannot touch the locked-winner bookkeeping). 126 ablation rows in the scoreboard.

**Recorded deviation:** the OGB ablation arms were run with `--final`, so the official test split received 7
additional diagnostic reads per dataset beyond the single locked read the protocol allows. This was unavoidable —
`characterize.PRIMARY` maps the OGB pair to `test_acc` / `test_hits@20`, so valid-split rows would never be read.
These reads are ablation diagnostics and were not used for model selection, but they are reported rather than left
for a reviewer to discover.

### Feature status: a stated rule instead of a bare number

`useful_feature` (one string, e.g. `centrality (+0.083)`) is replaced by three columns — `best_single_feature`,
`feature_lift_vs_random`, `feature_status` — with the classification in `characterize.status()`:

| status | rule |
|---|---|
| `useful` | lift clears **+2σ** of the pooled 3-seed noise of the arm and its control |
| `no clear evidence` | inside the band; three seeds cannot resolve it |
| `not useful` | below **−2σ** |
| `not evaluated` | no random-feature control on disk for that cell |
| `invalid metric` | no arm separates from any other *and* no seed moves — a constant predictor |

The band is deliberately **2σ**, stricter than the 1σ band `gaps()` uses for keep/augment: this table is descriptive
and must never decide a verdict, so "clearly worse than random" should not be claimed on a margin three seeds cannot
resolve. Result over the 16 cells: 11 `useful`, 3 `no clear evidence`, 1 `not useful` (questions LP, −0.034),
1 `invalid metric` (questions NC).

The ablation is reported as **"best single structural feature on the Ψ graph"**. It was never run on whichever graph
won the cell, and that limitation is now in the column name rather than in a footnote.

### A degenerate random control on the OGB pair — flagged, not yet corrected

The lift is measured against the D4 random-feature arm. On the OGB pair that control has collapsed:

| dataset | random control | const control (D5) | verdict |
|---|---|---|---|
| ogbn_arxiv | 0.0586 ± 0.0000 | **0.0586 ± 0.0000** | identical — random features carry no more signal than constant ones |
| ogbl_ddi | 0.0005 ± 0.0001 | 0.0000 ± 0.0000 | both at the floor |

On ogbn_arxiv the random-feature and constant-feature arms return **exactly** the same accuracy with zero seed
variance, i.e. the random control has degenerated into the majority-class floor. Every arxiv lift is therefore
"distance from the trivial floor", not "distance from a competitive random baseline", and because the control's
standard deviation is 0 the sigma values inflate absurdly — centrality reports **+305σ**. The `useful` labels remain
directionally correct, but **sigma magnitudes are not comparable between the OGB pair and the core-protocol
datasets**, and no arxiv sigma should be quoted in the paper. The core-protocol datasets are unaffected: their
random controls carry real variance (e.g. cora LP random std 0.0093).

This is a property of the OGB setting — one fixed official split, so the only seed variance is encoder
initialisation — not of the ablation code.

---

## Recommendation table: the tie case now names the augmented graph (2026-07-28)

`recommended_graph` in notebook 5 §7a changed from

```python
np.where(verdict == "augment", best_variant, "original")     # tie -> original
np.where(verdict == "keep original", "original", best_variant)  # tie -> best_variant
```

so a *tie* verdict now reports the best augmented variant rather than falling back to the original graph. The
verdict banding in `characterize.gaps()` is untouched — only the presentation column changes.

**Affected cells: exactly one.** `enzymes` node classification is the panel's only tie: hybrid 0.5546 vs original
0.5591, gap **−0.0045 weighted F1 = −0.7σ**, inside the 1σ band. It is also the panel's closest NC cell to an
augment verdict, and the cell where the role graph recovers the largest share of the original's gain over raw
features (93%, vs 46% on cora).

**Caveat that must travel with the table.** On this cell the hybrid graph's mean is *below* the original's; the tie
verdict says the two are indistinguishable at three seeds, not that hybrid is ahead. The column should be read as
"a role graph is a defensible choice here", not as a measured win. The headline finding is unchanged: **no node
classification cell in the panel produces an augment verdict**, and all four augment verdicts remain link
prediction.

---

## Module 2 complete: neighbour-label predictability + an automatic credibility screen (2026-07-29)

`experiments/characterize.py` gains portion 2 (D). Pipeline, seeds, K and encoder untouched — this is measurement and
screening over the frozen scoreboard only. New output: `results/candidate_rules.csv`, plus notebook 5 §8.

**New property — neighbour-label predictability.** Homophily asks "do neighbours share the label?"; this asks "does a
node connect to a *consistent* class mix, even when the labels differ?" Naive Bayes over each node's neighbour class
histogram, leave-one-out (the node's own contribution is removed from its class row of the compatibility matrix),
reported adjusted against the majority-class floor so it is comparable across datasets. Entropy is reported alongside
as descriptive only.

It separates from homophily exactly where it was predicted to. **`roman_empire`: adjusted homophily −0.0468 (neighbours
essentially never share a label) yet predictability 0.4256 raw vs a 0.1396 majority floor = +0.3324 adjusted.** The
edges carry class information without carrying the class. That is the mechanism for why the original graph must be kept
for `roman_empire` node classification (−0.0417 F1, −4.9σ) even though the graph is strongly heterophilous. Converse
case: `tolokers` scores −0.3274 adjusted (below its 0.7818 majority floor) — its edges carry no usable class signal at all.

**Predictor tiering.** 7 primary (`homophily_adjusted`, `nbr_predictability_adjusted`, `components`,
`largest_component_frac`, `avg_degree`, `avg_clustering`, `n_classes`) + 8 exploratory. Only primary may gate a rule and
only primary carries the Bonferroni correction. **Caveat that must be reported: the tier split was fixed after the
correlations were already visible.** `components` clears corrected significance (p_bonf = 0.0406) partly because
tiering cut the family from 12 tests to 7; at 12 it did not.

**Selection-bias control.** `gap = best_augmented − original` is a max over four variants and therefore biased upward.
Added `gap_fixed_*` against one variant fixed per task (LP = `centrality`, NC = `hybrid`; each already wins 6/8 cells).
**Result: the two gap definitions agree on all 16 cells — `verdict_agrees` is True everywhere.** No augment verdict in
the panel is an artefact of the max. `questions` LP is the only cell where the magnitude moves materially (+0.0861 →
+0.0413, +4.58σ → +2.14σ); the verdict stays *augment*.

**Degenerate-cell exclusion.** A cell is dropped when no variant separates from any other and no seed moves
(spread < 1e-3 and max std < 1e-6). Mechanical, not hand-named: it selects exactly one cell in the panel, `questions`
node classification (spread 1e-4, std 0.0000, 97.0% majority class pinning weighted F1). 16 cells → 15 usable.

**The screen.** Gates are |ρ| ≥ 0.7, leave-one-dataset-out sign stability, ≤ 1 misclassified cell. **Significance is
reported but deliberately not gated**: at n = 8 the two-tailed 0.05 critical Spearman value is 0.738, so the |ρ| gate
already sits at about p ≤ 0.07, and adding a p gate would reject usable patterns on this few datasets. A pass means
*credible candidate*, never *proven rule*.

**Three candidates clear the gates, all link prediction, all with 0 exceptions:**

| rule | ρ | LODO min abs ρ | p_bonf | n |
|------|---|----------------|--------|---|
| augment when `components` < 39.5 | −0.8625 | 0.7881 | 0.0406 | 8 |
| augment when `largest_component_frac` > 0.9588 | +0.7864 | 0.7412 | 0.1442 | 8 |
| augment when `homophily_adjusted` < 0.2239 | −0.7143 | 0.5429 | 0.4991 | 7 |

The first two are the same variable read two ways, so this is **one** fragmentation candidate plus one homophily
candidate — not three independent findings. All three hold identically under `gap_fixed_rel`.

**Node classification produces no rule, and the screen says so mechanically.** 0 of 7 usable NC cells are *augment*, so
there is nothing for a threshold to separate; every NC row fails on `n_exceptions = -1` (not evaluable). The reportable
NC result is a boundary — *never augment* — not a predictor. Any held-out NC test is therefore a falsification attempt,
not a validation.

**Open confound, unchanged by this work.** The four LP *keep original* cells are exactly the core-4 (fragmented,
citation/molecular); the four *augment* cells are exactly the four datasets added later (single-component,
web/interaction). `components` and dataset provenance are currently the same variable. Module 3's held-out set must
include a fragmented + heterophilous graph and a single-component + homophilous graph or the confound survives the
validation intact.

**Flag for the write-up.** The strongest NC correlation in the whole table is an *exploratory* predictor:
`nbr_label_entropy` vs the NC gap, ρ = +0.79 (gap_rel) and +0.89 (gap_fixed_rel, p = 0.0068) — higher than any primary
predictor for NC. It is not screened because it is not in the primary tier, and it cannot become a rule while NC has
zero augment cells. Promoting it now would be post-hoc; it is recorded here so the decision is visible either way.

Scope, unchanged: top-K role graphs, K = 10, GraphSAGE, seeds 42/43/44, five variants.

---

## Panel cut to seven datasets; Module 2 re-run (2026-07-29)

User decision: `citeseer_linqs` and `proteins` are excluded from all forward work. `characterize.PANEL` is now the
default for `--datasets`; `STUDY` keeps all nine so past rows retain their meaning. **All numbers in the previous entry
are the nine-dataset figures and are superseded by what follows.**

**Panel: 7 datasets / 12 cells, 11 usable — 7 keep original, 4 augment, 1 tie.** Every verdict is unchanged from the
nine-dataset run; only the two dropped datasets are gone. `gap_fixed` still agrees with max-over-variants on all 12 cells.

**Four credible LP candidates, up from three:**

| rule | ρ (gap_rel) | ρ (gap_fixed_rel) | LODO min\|ρ\| | p_bonf | n |
|---|---|---|---|---|---|
| augment when `homophily_adjusted` < 0.227 | −0.90 | **−1.00** | 0.80 | 0.117 | 5 |
| augment when `components` < 39.5 | −0.78 | −0.78 | 0.71 | 0.481 | 6 |
| augment when `largest_component_frac` > 0.9588 | +0.78 | +0.78 | 0.71 | 0.481 | 6 |
| augment when `nbr_predictability_adjusted` < 0.4084 | −0.70 | −0.60 | 0.40 | 1.000 | 5 |

**The headline reverses.** On nine datasets `components` led (ρ −0.86) and was the only predictor to clear Bonferroni.
On seven it falls to −0.78 and **no longer survives correction**; `homophily_adjusted` takes over and separates the LP
cells *perfectly* under the fixed-variant gap. This is a direct demonstration that the panel, not the data-generating
process, is currently selecting the winner — exactly the fragility the pre-registered Module 3 exists to resolve.

**Reportable reversal on the supervisor-proposed rules.** Adjusted homophily was proposed as the *node classification*
predictor. It fails there (ρ = −0.30, and NC has no augment cell to predict at all) but is the **strongest link
prediction** rule in the study. Right variable, wrong task — worth stating as a finding rather than burying.

**`nbr_predictability_adjusted` is the weakest of the four and should be reported as such.** It clears the gates on
`gap_rel` only (ρ = −0.70, exactly at the 0.7 threshold) and is *rejected* by the bias-free `gap_fixed_rel` control
(ρ = −0.60). Its LODO floor is 0.40 — a single dataset moves it. It is a candidate to watch, not to carry.

**Node classification still yields no rule:** 0 of 5 usable NC cells augment, every NC row fails on
`n_exceptions = -1`. Only the boundary "never augment" is reportable.

**Two defects fixed in the same run.**
1. `feature_scores()` read `results/scoreboard.csv` directly with no dataset filter, so the two excluded datasets
   reappeared in `feature_usefulness.csv` with NaN spreads. Now filtered to the panel at source (notebook 5's manual
   `isin(DATASETS)` workaround becomes a no-op).
2. `rho()` reported p = 1.4e-24 for a perfect rank match at n = 5, because scipy's Spearman p is a t-approximation that
   diverges as |ρ| → 1. The smallest p any permutation of n points can produce is 2/n!, so p is now floored there:
   the homophily rule's corrected p goes from a meaningless 0.0000 to **0.117**. The floor is negligible for larger n.
   Any p quoted from the previous entry at n ≤ 6 with |ρ| near 1 should be re-read from the regenerated CSVs.

**Confound tightened, as predicted.** `proteins` (1195 components) and `citeseer_linqs` (390) were two of the four
fragmented *keep original* LP cells; the fragmentation candidate now rests on 6 LP cells with 2 fragmented ones.
Module 3's held-out set must carry a fragmented + heterophilous graph.

---

## Collinear predictors merged into findings (2026-07-29)

`components` and `largest_component_frac` rank the seven-dataset panel at **Spearman −1.000** — identical information,
no exception. They were being reported as two credible rules; they are one. `characterize.FAMILY` / `CANONICAL` now
collapse them, `candidate_rules.csv` gains `predictor_family` / `canonical`, and the credible list prints one row per
finding. No other primary pair exceeds |ρ| 0.85, so this is the only merge.

**Canonical member is `largest_component_frac`, not `components`, and the reason is transfer.** A rule reading
"< 39.5 components" cannot be applied to an unseen graph ten times larger; "> 0.9588 of nodes in one component" can.
Both give ρ = ±0.7775 identically, so the choice costs nothing statistically.

**Four credible predictors → three credible findings, ranked:**

| finding | ρ (gap_rel / gap_fixed) | LODO min\|ρ\| | p_bonf | distinct values (all / augment side) |
|---|---|---|---|---|
| augment when `homophily_adjusted` < 0.227 | −0.90 / **−1.00** | 0.80 | 0.117 | 5 / 3 |
| augment when `largest_component_frac` > 0.9588 | +0.78 / +0.78 | 0.71 | 0.481 | **3 / 1** |
| augment when `nbr_predictability_adjusted` < 0.4084 | −0.70 / −0.60 | 0.40 | 1.000 | 5 / 3 |

**New diagnostic, and it demotes the fragmentation finding.** `distinct_augment` counts how many distinct predictor
values the *augmenting* cells span. For `largest_component_frac` it is **1**: all four augment cells sit at exactly
1.0000, and the two keep-original cells are the only non-1.0 values (cora 0.9177, enzymes 0.0064). So the fragmentation
"threshold" is a **group label, not a graded trend** — it says "fully connected vs not", the 0.9588 cut is arbitrary
anywhere in (0.9177, 1.0), and it is precisely the batch confound already on record (the two fragmented graphs are
early-panel, the four connected ones are late additions). Adjusted homophily spans 5 distinct values with 3 on the
augment side and is graded throughout — genuinely stronger evidence, and it is also the one that separates perfectly
under the bias-free gap.

**Standing order of the LP candidates for Module 3:** adjusted homophily first (graded, perfect separation,
LODO 0.80), fragmentation second (real but two-group, confounded), neighbour predictability third (fails the
fixed-gap control, LODO 0.40).

---

## FINAL: two candidate rules locked for Module 3 (2026-07-29)

Seven primary properties were screened **separately for node classification and link prediction** as executable
decisions — one threshold, one side that says augment. Panel: seven datasets, 12 cells, 11 usable. Four predictors
passed the gates; `components` and `largest_component_frac` rank the panel at Spearman **−1.000**, so they are one
variable and merge, leaving **three distinct LP findings**. After the robustness checks, **two are carried forward**:

### Rule 1 — augment when adjusted homophily < 0.227

ρ = −0.90 on `gap_rel`, **−1.00** on the bias-free `gap_fixed_rel`; LODO min |ρ| = 0.80; 0 exceptions; n = 5.
Graded across 5 distinct predictor values with 3 on the augment side. The strongest finding in the study, and the one
to lead with. Mechanism: when neighbours carry no label agreement, the original edges are not the signal the encoder
needs, so replacing them with role edges costs nothing and can help.

### Rule 2 — augment when largest-component fraction > 0.9588

ρ = +0.78 on both gap definitions; LODO min |ρ| = 0.71; 0 exceptions; n = 6. Canonical over `components` because it is
scale-free — a "< 39.5 components" threshold cannot be applied to a graph ten times larger. **Weaker evidence than
rule 1 and must be reported as such:** all four augment cells sit at exactly 1.0000, so `distinct_augment` = 1 — this
is a *fully connected vs not* group split, not a graded trend, the 0.9588 cut is arbitrary anywhere in (0.9177, 1.0),
and the split coincides exactly with the standing batch confound. Mechanism: on a fragmented graph, component identity
alone makes link prediction easy, and role edges bridge components and destroy it.

### Dropped — neighbour predictability < 0.4084

ρ = −0.70 on `gap_rel` (exactly at the gate) but **−0.60 on `gap_fixed_rel`**, so the bias-free control rejects it;
LODO min |ρ| = 0.40, meaning a single dataset moves it. Precision point for the write-up: its LODO **sign** was stable
(`lodo_sign_stable = True`) — the failure is magnitude fragility plus the fixed-gap rejection, not a direction flip.
It stays in `results/candidate_rules.csv` as a screened-and-rejected row, which is the honest record.

### Node classification — no rule, by construction

0 of 5 usable NC cells augment, so there is nothing for a threshold to separate and every NC row fails at
`n_exceptions = -1`. The reportable NC result is a **boundary** — "never augment" — not a predictor. A held-out NC
experiment is therefore a falsification attempt, not a validation.

### Status

Module 2 is **closed**. These two rules are frozen; Module 3 (pre-register the prediction, then run the unchanged
pipeline on unseen datasets) is the test that decides whether they are rules or artefacts. The held-out set must
contain a fragmented + heterophilous graph — rules 1 and 2 *disagree* there, which is exactly why it is the
informative case.

Docs updated to this state: `README.md`, `docs/virgo_guide.md`, `CLAUDE.md` §4, `experiments/README.md`,
notebook 5 §8. Tables: `results/candidate_rules.csv`, `results/characterization_*.csv`.

---

## 2026-07-31 — Threshold honesty: the split is fitted on the cells it is scored on

### The issue

The screen reported each rule as "0 exceptions", and that number was **not evidence**. `threshold()` searches every
midpoint between adjacent predictor values and keeps the one with fewest errors, then the same cells are used to count
those errors. Whenever the two verdict classes are linearly separable on a predictor, the search *must* find a split
with zero errors. The count is a property of separability, not of the rule's reach.

Two separate defects, worth stating apart:

1. **The reported number is over-precise.** `0.227` is the midpoint between `tolokers` (0.0926, augments) and `enzymes`
   (0.3613, keeps) — the two datasets straddling the boundary. It is a max-margin choice, which is defensible, but the
   panel only pins the **interval** (0.0926, 0.3613); every cut inside it fits the seven-dataset panel exactly as well.
   Quoting three decimals implies a precision the data do not contain. Same for rule 2: the interval is (0.9177, 1.0).
2. **The error count is in-sample.** No held-out estimate existed at all, so nothing in the screen distinguished a rule
   that generalizes from one that memorizes five points.

Origin: raised by the user, who proposed hiding one dataset at a time, refitting the cutoff on the rest, and predicting
the hidden one. That proposal is correct and is what was implemented, with one addition described below.

### The design

Three additions to `experiments/characterize.py`, all measurement — pipeline, seeds, K, encoder and the frozen
scoreboard are untouched.

**1. `threshold()` now returns the separating interval.** Error as a function of the cut is piecewise constant between
adjacent data values, so the routine widens from the chosen cut across every neighbouring cut with the same error count
and returns the two data values bounding that run. New columns `interval_lo` / `interval_hi`. The reported threshold is
still the midpoint; the interval is what the write-up quotes.

**2. `loo_threshold()` — leave-one-out refit of the split.** Hide one cell, refit the cut on the remainder, predict the
hidden cell. New columns `loo_accuracy`, `loo_correct`, `loo_folds`, `loo_threshold_lo`, `loo_threshold_hi`, plus
`majority_baseline` (always guessing the more common verdict) and `loo_beats_majority`. The baseline matters: with 3
augment / 2 keep, guessing "augment" every time already scores 0.60, so raw LOO accuracy is meaningless on its own.

**3. `nested_loo()` — leave-one-out over the whole screen.** LOO on the threshold alone still leaks: the *predictor*
`homophily_adjusted` was itself chosen by looking at all cells. `nested_loo()` hides a dataset and re-runs predictor
selection **and** thresholding inside each fold, then predicts the hidden one. Predictor selection inside a fold breaks
ties on (fewest errors, then larger |ρ|) rather than list order, so the first-listed predictor is not favoured. A
dataset the fold's chosen predictor cannot be evaluated on counts as a **miss, not a skip** — `ogbl_ddi` has no labels,
so a rule built on homophily genuinely fails to predict it. New output: `results/nested_loo.csv`.

**Gate change.** `GATES["loo_above_majority"] = True`; `credible` now also requires `loo_accuracy > majority_baseline`.
Rationale: `n_exceptions` cannot fail on separable data, so it was carrying no weight. Both frozen rules pass unchanged,
so no conclusion moves — the gate only makes an existing hand-judgement mechanical (see below).

### Results

| rule | reported cut | any cut in | LOO | majority baseline | fold cutoffs ranged |
|------|--------------|-----------|-----|-------------------|---------------------|
| augment when `homophily_adjusted` < 0.227 | 0.227 | (0.0926, 0.3613) | **4/5 = 0.80** | 0.60 | 0.191 – 0.4319 |
| augment when `largest_component_frac` > 0.9588 | 0.9588 | (0.9177, 1.0) | **5/6 = 0.83** | 0.67 | 0.5032 – 0.9588 |
| ~~`nbr_predictability_adjusted` < 0.4084~~ | 0.4084 | (0.3324, 0.4844) | **3/5 = 0.60** | 0.60 | 0.2402 – 0.5734 |

**Both frozen rules survive, each with exactly one out-of-sample error.** The in-sample "0 exceptions" becomes "1 of 5"
and "1 of 6" once honestly scored — that is the number the paper should quote.

- Rule 1 misses **`enzymes`**: with enzymes hidden the boundary pair becomes `tolokers` (0.0926) and `cora` (0.7711),
  the fold cutoff jumps to 0.4319, and enzymes at 0.3613 falls on the augment side. The 2.3× spread in fold cutoffs
  (0.191 – 0.4319) is the direct measure of how weakly the panel pins the number.
- Rule 2 misses **`cora`**: with cora hidden the only keep-side dataset left is `enzymes` (0.0064), the cutoff drops to
  0.5032, and cora at 0.9177 is called augment. Consistent with the standing "two-group split" caveat.

**`nbr_predictability_adjusted` is now rejected mechanically.** It scores exactly the majority baseline — the split
carries zero information beyond guessing. It was already dropped on 2026-07-29 by reading the fixed-gap control and the
LODO floor; the gate now reaches the same verdict without judgement, which is the stronger version of the same result.

**Nested LOO: 4/6 on both gap definitions.** `homophily_adjusted` is re-selected in 5 of 6 folds (`tolokers`'s fold
picks `nbr_predictability_adjusted` on `gap_rel`), so *predictor choice is stable* — the study is not fishing between
properties. The two misses are `enzymes` (as above) and `ogbl_ddi`, which is unpredictable rather than mispredicted:
it has no labels, hence no homophily. That is a genuine coverage limit of the leading rule and belongs in the paper —
**rule 1 cannot be applied to an unlabelled graph at all**, which is precisely when rule 2 is needed.

### What this changes in the write-up

- Quote **intervals**, not points: "augment when adjusted homophily is below roughly 0.1–0.36" with 0.23 as the point
  estimate. Do not quote 4 significant figures for either rule.
- Report LOO 4/5 and 5/6 against their 0.60 / 0.67 baselines, never the in-sample 0 exceptions alone.
- State the nested result 4/6 and name both failure modes (`enzymes` boundary, `ogbl_ddi` coverage).
- This is **not** a substitute for Module 3. Same seven datasets, same provenance confound, n = 5–6 per fold. LOO tests
  how tightly the panel pins the cut; only unseen datasets test whether the rule transfers.

Code: `experiments/characterize.py` (`threshold`, `loo_threshold`, `nested_loo`, `GATES`, `rule`). Tables:
`results/candidate_rules.csv` (9 new columns), `results/nested_loo.csv` (new). Notebook 5 §8 and new §8c.

## 2026-08-04 — LastFM Asia: the first held-out case where the two rules disagree AND the experiment decides

**Why this dataset.** Module 3's two frozen rules had never been separated on unseen data. `pubmed` and `amazon_photo`
are both homophilous + single-component, so rule 1 says *keep original* and rule 2 says *augment* — but both cells came
back a **tie**, which scores neither rule. LastFM Asia (Rozemberczki & Sarkar 2020) sits in the same quadrant and is a
domain the study had none of: a music-platform friendship network. 7,624 users, 27,806 undirected edges (avg degree
7.29), 18 country classes, one connected component.

**Provenance and a recorded deviation.** `torch_geometric.datasets.LastFMAsia` downloads from `graphmining.ai`, which
no longer serves the file from this environment (TLS handshake failure; `404` when the handshake is forced). The raw
graph is therefore read from **SNAP's primary archive of the same dataset** — the source PyG repackages — so node ids,
edges and country labels are the publisher's own. SNAP ships features as a liked-artist JSON rather than PyG's 128-dim
matrix; irrelevant here, since node features are dropped by design (`virgo/data/make_pyg.py`, structural-only).
Counts reproduce the published statistics exactly (7,624 / 27,806 / 18).

**Pre-registration.** `experiments/predict_module3.py --datasets lastfm_asia` was run **before any training** and the
row is frozen write-once in `results/module3_predictions.csv`:

| property | value | rule | prediction |
|----------|-------|------|-----------|
| `homophily_adjusted` | **0.8562** — the highest in the whole study (above `cora` 0.7711 and `amazon_photo` 0.7850) | rule 1 (`< 0.227`) | **keep original** |
| `largest_component_frac` | **1.0000** | rule 2 (`> 0.9588`) | **augment** |
| | | combined | **rules disagree** |

**Result (pipeline unchanged: `graphsage_edge`, K=10, seeds 42/43/44, five variants, both tasks).**

| task | original | best augmented | gap | 3-seed noise | gap/noise | verdict |
|------|----------|----------------|-----|--------------|-----------|---------|
| link prediction (AUC) | 0.7218 ± 0.0251 | 0.7002 ± 0.0086 (`hybrid`) | −0.0216 | 0.0188 | **−1.15σ** | **keep original** |
| node classification (weighted F1) | 0.3770 ± 0.0067 | 0.2844 ± 0.0118 (`hybrid`) | −0.0926 | — | **−9.65σ** | keep original |

**Rule 1 correct, rule 2 wrong.** The disagreement resolves in favour of the lead rule — the one Module 2 designated
primary on the discovery panel. Held-out tally is now **rule 1 4/5, rule 2 3/5** (`pubmed` and `amazon_photo` still
score neither: tie). This is the first out-of-sample evidence that separates the two, and it points the same way the
discovery evidence did (ρ −1.00 on the bias-free gap for homophily, vs a two-group split for connectivity).

**Encoder-robust.** The verdict does not depend on the study encoder: under the `deepwalk` bridge the same cell reads
original 0.9432 ± 0.0013 vs `hybrid` 0.8304 ± 0.0029 — keep original by a wider margin.

**The NC boundary holds again.** No held-out node-classification cell has ever augmented: **0 of 6 usable cells**, this
one at −9.65σ. The `hybrid` variant is additive (it keeps every original edge) and still loses by that margin, so on a
homophilous social graph the role edges are actively harmful to the label signal, not merely uninformative.

**Caveat worth stating.** LastFM Asia does *not* break the standing confound: it is single-component like every other
augment-side dataset, so it tests the two rules against each other but adds nothing to the fragmented + heterophilous
cell Module 3 still needs. What it does remove is the "the disagreement quadrant is untested" limitation.

Code: `virgo/data/make_pyg.py` (`_lastfm_asia`), registry entries in `virgo/config.py`, `experiments/characterize.py`
(`STUDY`), `virgo/frozen_rules.py` (`HELDOUT`), `experiments/run_core.py` (`HELDOUT`). Tables:
`results/module3_predictions.csv`, `results/module3_scored.csv`, `results/scoreboard.csv`.

## 2026-08-05 — Amazon-ratings: rule 1's augment side fails a second time, and the Platonov family alone falsifies the threshold

**Why this dataset.** Rule 1 is a low-homophily → *augment* rule, and its augment side had been tested on exactly two
held-out graphs: `actor` (correct) and `minesweeper` (wrong). Amazon-ratings (Platonov et al. 2023) is a clean
heterophily benchmark — explicitly built to replace Chameleon/Squirrel, which have duplicate-node leakage — and it is
large enough to matter without being impractical: 24,492 nodes, 93,050 undirected edges (avg degree 7.60), 5 rating
classes, one connected component. It is **not** `amazon_photo`, the homophilous co-purchase graph already held out;
the two share a domain word and nothing else (adjusted homophily 0.1402 vs 0.7850).

Counts reproduce the published statistics exactly. Built by `make_hetero.py`, the same converter as the three discovery
heterophilous graphs, from the authors' own release; its 300-dim fastText product-description features are dropped by
design, and the official 10 train/val/test masks are ignored so the cell stays on the ViRGo core protocol.

**Pre-registration.** `experiments/predict_module3.py --datasets amazon_ratings` ran **before any training**; the row
is frozen write-once in `results/module3_predictions.csv`:

| property | value | rule | prediction |
|----------|-------|------|-----------|
| `homophily_adjusted` | **0.1402** — inside rule 1's frozen interval (0.0926, 0.3613) | rule 1 (`< 0.227`) | **augment** |
| `largest_component_frac` | **1.0000** | rule 2 (`> 0.9588`) | **augment** |
| | | combined | **augment** (rules agree) |

**Result (pipeline unchanged: `graphsage_edge`, K=10, seeds 42/43/44, five variants, both tasks).**

| task | original | best augmented | gap | 3-seed noise | gap/noise | verdict |
|------|----------|----------------|-----|--------------|-----------|---------|
| link prediction (AUC) | 0.7536 ± 0.0204 | 0.6550 ± 0.0116 (`hybrid`) | −0.0986 | 0.0166 | **−5.94σ** | **keep original** |
| node classification (weighted F1) | 0.2875 ± 0.0029 | 0.2623 ± 0.0015 (`hybrid`) | −0.0252 | 0.0023 | **−10.92σ** | keep original |

**Both rules wrong.** Held-out tally is now **rule 1 4/6, rule 2 3/6** (`pubmed` and `amazon_photo` remain ties and
score neither). Not a marginal miss: −5.94σ, and the runner-up variants are far worse still (`psi` 0.5654, `degree`
0.4551 — below chance). Encoder-robust in the same direction and more extreme: under the `deepwalk` bridge the
original graph scores **0.9982 ± 0.0001** against `hybrid` 0.9407 ± 0.0011.

**The asymmetry is now the headline.** Rule 1's six scored predictions split cleanly by which side they call:

| rule 1 says | datasets | correct |
|-------------|----------|---------|
| keep original | `citeseer_linqs`, `proteins`, `lastfm_asia` | **3 / 3** |
| augment | `actor` ✅, `minesweeper` ❌, `amazon_ratings` ❌ | **1 / 3** |

Low adjusted homophily is a *necessary-looking* but plainly insufficient condition. Every graph the rule expected to
keep its original edges did; two of the three it expected to benefit from role edges did not. The paper should report
rule 1 as a one-sided screen — "high homophily ⇒ do not bother augmenting" is the part that has survived contact with
unseen data — rather than as a two-sided predictor.

**The Platonov family alone falsifies the threshold.** All three discovery augment cells (`roman_empire`, `tolokers`,
`questions`) come from Platonov et al.; `minesweeper` and `amazon_ratings` come from the same paper, the same
converter and the same protocol. Sort those five by adjusted homophily and the verdicts interleave:

| dataset | `homophily_adjusted` | LP verdict |
|---------|---------------------|------------|
| `roman_empire` | −0.0468 | augment |
| `minesweeper` | 0.0094 | **keep original** |
| `questions` | 0.0207 | augment |
| `tolokers` | 0.0926 | augment |
| `amazon_ratings` | 0.1402 | **keep original** |

**A K A A K** — no cut anywhere on this axis separates them. Within a single dataset family, held at fixed provenance,
adjusted homophily does not order the outcome. This is the strongest negative evidence Module 3 has produced, and it
also cuts against the lazy reading of the batch confound: dataset provenance does not explain the augment cells
either, since these five share it and disagree.

**What the interval says, and what we are not allowed to do with it.** 0.1402 is the first held-out value to land
*inside* rule 1's frozen interval, where every cut fits the discovery panel equally well. Had the cut been set at the
interval floor (0.0926, i.e. just below `tolokers`) this prediction would have been *keep original* and correct. So
the evidence points to a lower cut — but `minesweeper` at 0.0094 sits below the entire interval and still keeps its
original graph, so **no** choice inside the interval rescues the rule. Refitting the threshold on these datasets is
forbidden regardless: it would make Module 3 circular. Reported as a falsification, not a correction.

**The NC boundary holds.** Still **0 of 7** usable held-out node-classification cells augment, this one at −10.92σ.
Notably this is a *heterophilous* graph — the case where role-based rewiring should have the most to offer NC — and
`hybrid`, which is additive and keeps every original edge, still loses by ten sigma.

**Confound status unchanged.** Amazon-ratings is single-component, so the fragmented + heterophilous cell Module 3
needs is still missing. What it removes is a different limitation: rule 1's augment side is no longer supported by a
single dataset family.

Code: `virgo/data/make_hetero.py` (`HETERO` gains `amazon_ratings`; `--dataset both` → `all` now that the converter
covers five graphs), registry entries in `virgo/config.py`, `experiments/characterize.py` (`STUDY`),
`virgo/frozen_rules.py` (`HELDOUT`), `experiments/run_core.py` (`HELDOUT`). Tables:
`results/module3_predictions.csv`, `results/module3_scored.csv`, `results/scoreboard.csv`.

### 2026-08-05 (addendum) — verification of the amazon_ratings cell, and what actually tracks the verdict

The `amazon_ratings` result was checked at five independent levels before being accepted as a falsification; all pass.

| check | result |
|-------|--------|
| edge set vs the PyG source | **identical** — 186,100 directed columns → 93,050 undirected pairs, 0 self-loops, source already symmetric |
| labels | 24,492 rows, ids exactly 0..n−1, values **match `data.y` element-wise**, 5 classes |
| `.nodes` sidecar | 24,492 rows; 0 isolated nodes (min degree 5) |
| LP split (seed 42) | 65,135 train / 27,915 test_pos / 27,915 test_neg; test fraction 0.3000; train ∩ test = **∅**; train ∪ test = the full edge set; **0** negatives are real edges |
| rule inputs recomputed from scratch | adjusted homophily **0.1402** (edge 0.3804, degree-weighted null 0.2793) and largest-component fraction **1.0000** — both reproduce the frozen row exactly |

**Why the augmented graph loses here, mechanically.** On the seed-42 training graph, `psi` retains **381 of 65,135**
original edges (0.6%) — it is a nearly disjoint graph, not an enrichment of the original one. `hybrid` is additive but
adds 140,717 role edges on top of the 65,135 real ones, so **68% of its edges are role edges** and every real edge's
message is diluted roughly 1 : 2.16. Of those 140,717 role edges only **169 are actual held-out test links** (0.12%;
~13× chance, so role similarity carries a faint link signal — nowhere near enough to pay for the dilution).

**A cleaner correlate of the LP verdict than either frozen rule — reported as an observation, NOT promoted to a rule.**
Sorting all 14 usable LP cells by the *original* graph's own `graphsage_edge` AUC separates the verdicts almost
perfectly, with a single exception:

| original AUC | 0.0173 | 0.4665 | 0.5953 | 0.6019 | 0.6130 | 0.6218 | 0.6392 | 0.6444 | 0.6720 | 0.7000 | 0.7093 | 0.7218 | 0.7536 | 0.7857 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dataset | ddi | quest. | actor | roman | cora | cites. | pubmed | tolok. | prot. | enz. | mines. | lastfm | am_rat | am_photo |
| verdict | A | A | A | A | K | K | tie | **A** | K | K | K | K | K | tie |

Every cell below ~0.60 augments; every cell above ~0.65 keeps its original graph or ties. `tolokers` is the lone
exception. This says augmentation helps exactly where the original graph is *weak for the encoder* — i.e. it buys
headroom, not structure.

**This is not usable as the study's rule and must not be swapped in for one.** The original graph's AUC is a *result*,
not a graph property: obtaining it means already having trained on the original graph, at which point running the
augmented variant too is cheaper than consulting any rule. Its value is explanatory — it accounts for why the
homophily threshold keeps missing on the augment side (`minesweeper` 0.7093 and `amazon_ratings` 0.7536 are both
*high-headroom-free* cells that happen to be heterophilous) — and it is a candidate for a future **pre-registered**
test on unseen data, framed as a property measurable in advance. Promoting it now on the strength of the same cells
that suggested it would be exactly the HARKing this log has refused elsewhere.

## 2026-08-05 — Squirrel-filtered: both rules correct, and the sharpest evidence yet that homophily is not sufficient

**Why this dataset, and which copy.** Rule 1's augment side stood at 1/3 after `minesweeper` and `amazon_ratings`.
Squirrel-filtered is Platonov et al.'s **de-duplicated** Squirrel: the original `WikipediaNetwork("squirrel")` contains
repeated nodes that leak between train and test, so that copy is deliberately not used. The filtered `.npz` is fetched
from the authors' own repository — **the same URL `HeterophilousGraphDataset` itself downloads from**, so the data path
is identical to the five names PyG exposes; only the file list differs (`virgo/data/make_hetero.py::_npz`).

**Edge-count correction worth recording.** The dataset is often quoted as 23,499 undirected edges, i.e. 46,998 halved
on the assumption that the archive stores both directions. It does not: all 46,998 rows are **distinct** sorted pairs,
zero reciprocals. Our builds reproduce the paper's table exactly for `roman_empire` (32,927), `tolokers` (519,000) and
`amazon_ratings` (93,050), which confirms Platonov reports *undirected* counts. So squirrel-filtered is **2,223 nodes /
46,998 undirected edges / 5 classes / avg degree 42.28** — the densest graph in the study bar `tolokers` and `ogbl-ddi`,
and ~14× denser than `roman_empire`.

**Pre-registration** (`experiments/predict_module3.py`, run before any training, frozen write-once):

| property | value | rule | prediction |
|----------|-------|------|-----------|
| `homophily_adjusted` | **0.0086** — below rule 1's whole interval (0.0926, 0.3613) | rule 1 (`< 0.227`) | **augment** |
| `largest_component_frac` | **1.0000** | rule 2 (`> 0.9588`) | **augment** |
| | | combined | **augment** (rules agree) |

**Result (pipeline unchanged: `graphsage_edge`, K=10, seeds 42/43/44, five variants, both tasks).**

| task | original | best augmented | gap | 3-seed noise | gap/noise | verdict |
|------|----------|----------------|-----|--------------|-----------|---------|
| link prediction (AUC) | 0.7399 ± 0.0114 | 0.7736 ± 0.0033 (`degree`) | +0.0337 | 0.0084 | **+4.02σ** | **augment** ✅ |
| node classification (weighted F1) | 0.3358 ± 0.0187 | 0.3500 ± 0.0103 (`psi`) | +0.0142 | 0.0151 | **+0.94σ** | **tie** |

**Both rules correct.** Held-out tally: **rule 1 5/7, rule 2 4/7**. Rule 1's two sides now read: keep original **3/3**,
augment **2/4** (`actor` ✅, `minesweeper` ❌, `amazon_ratings` ❌, `squirrel_filtered` ✅). Unusually, all three role
variants beat the original here (`degree` 0.7736, `centrality` 0.7728, `psi` 0.7609) — this is not one lucky variant.

**The decisive pair.** `squirrel_filtered` and `minesweeper` have adjusted homophily **0.0086** and **0.0094** — a
difference of 0.0008, the closest pair in the entire study — and **opposite LP verdicts** (+4.02σ augment vs −5.96σ
keep original, both far outside noise). No threshold on this axis can separate them, in either direction, at any cut.
Rule 1 predicting `squirrel_filtered` correctly is therefore a *coincidence of sign*, not evidence the variable is
sufficient. Extending the fixed-provenance Platonov ordering to six graphs:

| dataset | `homophily_adjusted` | LP verdict |
|---------|---------------------|------------|
| `roman_empire` | −0.0468 | augment |
| `squirrel_filtered` | 0.0086 | **augment** |
| `minesweeper` | 0.0094 | keep original |
| `questions` | 0.0207 | augment |
| `tolokers` | 0.0926 | augment |
| `amazon_ratings` | 0.1402 | keep original |

**A A K A A K.** Six graphs from one paper, one converter, one protocol; the verdicts do not order on homophily. The
correct claim for the paper is that **low adjusted homophily is necessary but not sufficient** for augmentation to
help: every graph above the boundary kept its original edges (3/3, plus the two ties), while below the boundary the
outcome is genuinely mixed (2/4). That is a screen, not a rule, and it is worth reporting as exactly that.

**Retraction: the "headroom" observation logged earlier today is falsified.** That addendum noted that sorting LP cells
by the original graph's own `graphsage_edge` AUC separated the verdicts with one exception, and flagged it as an
observation for a future pre-registered test — explicitly not promoted to a rule. `squirrel_filtered` breaks it: its
original AUC is **0.7399**, *above* `minesweeper` (0.7093) and `lastfm_asia` (0.7218) which both keep original, and it
augments anyway. The ordering now has two exceptions, one of them near the top, and the "augmentation only buys
headroom" reading does not survive. Recorded here rather than deleted: it is a clean example of why a pattern found on
the same cells that suggested it must not be adopted, and it was falsified by the very next dataset ingested.

**Density is still rejected too.** Sorted by average degree the verdicts remain interleaved at the sparse end —
`citeseer_linqs` 2.78 keeps original while `roman_empire` 2.91 augments — so the professor's density proposal gains
nothing from this cell, even though the three densest graphs (`squirrel_filtered` 42.3, `tolokers` 88.3, `ogbl_ddi`
500.5) do all augment.

**First non-negative NC cell.** Every held-out node-classification cell so far had been keep original, most by many
sigma. This one is a **tie** at +0.94σ — the augmented graph is nominally ahead, just inside noise. The boundary claim
survives (**0 of 8 usable held-out NC cells augment**) but should now be stated as "never *significantly* augments"
rather than "always loses". Consistent with the rest: a very dense, strongly heterophilous graph is where role edges
come closest to paying off for labels.

**Confound status.** Still single-component, so the untested quadrant remains **heterophilous + fragmented**, where
rule 1 says augment and rule 2 says keep original. That is the one cell no held-out dataset has yet occupied.

Code: `virgo/data/make_hetero.py` (`_npz`, `REPO`; `HETERO` maps to `None` for archives PyG does not list), registry
entries in `virgo/config.py`, `experiments/characterize.py` (`STUDY`), `virgo/frozen_rules.py` (`HELDOUT`),
`experiments/run_core.py` (`HELDOUT`). Tables: `results/module3_predictions.csv`, `results/module3_scored.csv`,
`results/scoreboard.csv`.
