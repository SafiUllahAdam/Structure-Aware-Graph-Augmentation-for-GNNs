# CLAUDE.md — ViRGo

**ViRGo: Virtual Role-Graph Embedding for Structural Identity**

Research project extending Identity2Vec (I2V, Oluigbo et al.). Target: a publishable paper. This file governs all agentic work in this repo.

---

## 1. Goal & Contribution

Study a simple, practical question: **when does a GNN need a rewired "virtual" graph, and when is the original graph already enough?** — and predict the answer from the graph's own properties (starting with homophily). The virtual graph (role-based rewiring), not the encoder, is the thing we vary.

**Honest headline finding (post-fix, 4 datasets).** The original graph is a very strong baseline — it wins or ties every dataset × task cell. Role-based virtual graphs only get close for node classification on the molecular graphs, and they never help link prediction (two nodes that share a role are not more likely to be linked). So the paper is a **study / characterization** — a guide for *when* structural augmentation helps — not a claim that our virtual graph always wins.

**Immediate target:** the LoG (Learning on Graphs) conference. The thesis reuses the same content.

**Hard scope rule — stay purely structural.** Never use a dataset's built-in node attributes (e.g. OGB's text features). The whole point, inherited from I2V, is to show what *structure alone* can do; adding attributes would change the method.

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
  1. Run the existing GraphSAGE pipeline on small-to-medium **OGB** datasets (node: ogbn-arxiv; link: ogbl-collab, ogbl-ddi), **structural features only** — skip the huge 100M-node graphs.
  2. Build the **characterization table**: graph properties per dataset (homophily first) → the original-vs-augmented gap → a "when to augment" rule.
  3. Swap **GraphSAGE → GIN** and re-run, to see if its stronger isomorphism power helps.
- **Future work.** Learnable alpha (auto-blend original vs virtual; needs synthetic datasets); embeddings as LLM graph summaries. Do not start these yet.

---

## 5. Coding Rules (match I2V style)

- **Mirror the I2V codebase**: `argparse` CLI; `build_graph()` / `learn_embeddings()` / `main(args)` structure; a class holding the core method (cf. `identity2vec.Graph`).
- **Fewest functions possible.** Each short, single-purpose, self-explanatory name. No helper unless necessary.
- **One-line comments max.** Triple-quoted one-line docstrings as in I2V.
- Self-explanatory file names (e.g. `virtual_graph.py`, `encoder.py`, `eval_linkpred.py`).
- Models expose **`train(epochs)`**, not `fit()`.
- Prefer the main script to call only functions defined in base/abstract classes.
- No new dependency without need. Reuse the existing env (numpy 1.26.4, networkx, gensim 4.3.3, scipy 1.12.0; add torch/torch-geometric for the GNN).

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
2. ✅ `virtual_graph.py` — five-variant top-K virtual-graph builder.
3. ✅ GraphSAGE encoder over the virtual graph (`encoder.py`); GIN to follow.
4. ✅ Eval scripts: node classification (F1), link prediction (AUC, leakage-free).
5. 🔵 The characterization table + rule: graph properties → when augmentation helps, across our datasets **plus small/medium OGB**.
6. 🔵 GIN results next to GraphSAGE.
7. 🔵 LoG paper draft (the thesis reuses it).
