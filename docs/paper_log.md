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
- **Hyperparameters (I2V, `I2V_PARAMS`).** dimensions=64, walk_length=40, num_walks=10, window_size=10, epochs=1, sg=1 (Skipgram), e=2.7182, temperature=0.3. **DEVIATION:** paper text states walk_length=80; we use the repo default 40 (matches the author's released `cora.emb`), recorded as a deliberate deviation.
- **Evaluation protocol.** Node classification = one-vs-rest logistic regression on embeddings, weighted F1, stratified split. Link prediction = 70:30 edge split, embedding retrained on the 70% train graph only (no leakage), Hadamard edge features → logistic regression → test AUC.
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
