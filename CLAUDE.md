# CLAUDE.md — ViRGo

**ViRGo: Virtual Role-Graph Embedding for Structural Identity**

Research project extending Identity2Vec (I2V, Oluigbo et al.). Target: a publishable paper. This file governs all agentic work in this repo.

---

## 1. Goal & Contribution

Study **the impact of the virtual-graph construction on GNN performance** across downstream tasks. The virtual graph — not the encoder choice — is the variable under study. I2V's Poisson/KL similarity graph is the *generic* baseline construction; we test whether it is well-suited per data and task.

**Research question:** which virtual graph is best for given data, and does GNN message passing over it beat walk+Skipgram on structural-identity embeddings?

Non-Euclidean / hyperbolic latent space for the virtual graph is **out of scope** (reserved for a second paper). Do not implement it.

## Research Contributions

**1st (research) contribution — Virtual-graph study.**
- *What:* Study which virtual graph is best for different data and tasks. A virtual graph connects nodes by structural similarity, not only by original edges.
- *Why:* Normal GNNs pass messages only through physical neighbors, but two far-apart nodes may share a role (both hubs, both bridges). The virtual graph lets role-similar nodes communicate.
- *How:* Build virtual graphs from I2V's Poisson/KL structural similarity; test their effect on node classification, link prediction, and anomaly detection. Compare against simpler virtual graphs (degree-only, centrality-only) to answer "which graph is best for given data?".

**2nd contribution — Structural embeddings as graph summaries for LLMs.**
- *What:* Explore whether ViRGo's compact role-aware embeddings can serve as a summary of a large graph's structure, so that structural information fits within an LLM's limited context window.
- *Why:* An LLM cannot ingest a massive graph directly; a compact structural summary could let it reason over large-graph structure without exceeding the context limit.
- *How:* Reuse the trained ViRGo embeddings as a graph-summarization signal feeding large-graph structure to an LLM. **a stretch goal; do not implement until the main virtual-graph study is complete.**


**Technical contribution — GNN encoder over the virtual graph.**
- *What:* Replace I2V's guided walk + Skipgram with a modern GNN encoder (GraphSAGE / GIN; GAT as ablation).
- *Why:* Skipgram learns from sampled walk sequences; a GNN aggregates directly over graph structure, learning embeddings straight from structurally similar nodes.
- *How:* `graph → structural features (cached) → Poisson/KL virtual graph → GraphSAGE/GIN encoder → node embeddings → evaluation`.

---

## 2. Method (pipeline)

Keep I2V front end, replace back end:

1. **Structural signal** — per-node degree + eigenvector centrality, computed **once per graph and cached** (graph-level, static graph → caching is exact). Removes I2V's per-step recomputation.
2. **Similarity scoring** — KL-divergence λ → Poisson Ψ, exactly as I2V.
3. **Virtual graph** — connect each node to its top-K most structurally similar nodes under Ψ. K is a tuned hyperparameter (sparsity vs over-smoothing tradeoff).
4. **Encoder** — inductive GNN over the virtual graph: **GraphSAGE** (primary), **GIN** (expressive alternative), **GAT** (ablation).

---

## 3. Tasks (evaluation)

- **Link prediction** — 70:30 edge split, AUC. Retrain on 70% graph only (no leakage). For comparability with I2V Table 4.
- **Node classification** — logistic regression on embeddings, weighted F1. Comparability with I2V (Cora, Citeseer, Politics, Enzymes have labels).
- **Graph anomaly detection** — *new downstream application* (I2V did not do this). Standalone on GADBench/PyGOD first (AUC, AP), then integrate into (collaborative semi-supervised, cross-model pseudo-labels).

---

## 4. Status

- Reproducing I2V. Comparison subset: **Cora + Citeseer** (smallest labelled datasets).
- Embeddings: `cora.emb` (author's), `citeseer.emb` (training).
- Baseline confirmed: I2V recomputes centrality inside the walk loop (slow) — motivates the cache fix.

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
- Log every run setting and deviation in `notes.md` (e.g. walk-length now 80 to match the paper, was 40).
- A result is "reproduced" only when our metric is within ~±0.05 of the paper's.
- Ship splits, seeds, and eval scripts with the method.

---

## 7. Deliverables

1. Cached I2V variant — embeddings validated identical to baseline, with timing gain.
2. `virtual_graph.py` — top-K Ψ virtual-graph builder.
3. GNN encoder (GraphSAGE/GIN/GAT) over the virtual graph.
4. Eval scripts: link prediction (AUC), node classification (F1), anomaly detection (AUC/AP).
5. Benchmark table: ViRGo vs I2V across tasks; ablation over virtual-graph construction and K.
6. Reproducible package + paper draft.
