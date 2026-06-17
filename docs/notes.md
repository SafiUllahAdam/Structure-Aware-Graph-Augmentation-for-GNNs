# ViRGo — Lab Notes

Lab notebook. Append a dated entry whenever something happens. Rules:
- Always paste the **exact command** (so it reruns).
- Always note the **seed** (project standard: `seed=42`).
- Tag surprises: `FINDING:` (result/observation), `DEVIATION:` (repo ≠ paper/spec), `TODO:` (open thread).
- One file, append-only. Newest at bottom.

> Entries dated **before 2026-06-15** are *reconstructed* from file timestamps, logs, and git — not logged live. Treat their commands as best-guess (exact flags weren't recorded). Live logging starts 2026-06-15.

---

## Environment / defaults

- Env: numpy 1.26.4, networkx, gensim 4.3.3, scipy 1.12.0. (torch / torch-geometric to be added for the GNN.)
- I2V `train.py` defaults: `dimensions=64`, `walk-length=40`, `num-walks=10`, `window-size=10`, `epochs=1`, `sg=1` (skipgram), `min-count=0`, `workers=1`, `e=2.7182`. Word2Vec: `alpha=0.25 → min_alpha=0.01`, `negative=5`, `sample=1e-5`.
- Datasets in `input/`: cora, citeseer, dhfr, enzymes, firstmmedges, nci, politics, proteins, webkb (`.edgelist`); citeseer also has original `citeseer.txt`.
- Pretrained / trained embeddings in `output/`: `cora.emb` (author's, 2022-01-28), `webkb.emb` (trained 2026-06-13).

---

## 2026-06-12 — reconstructed

- Repo state: I2V core present — `identity2vec.py` (class `Graph`: guided walk + Poisson/KL), `train.py` (argparse CLI, `build_graph` / `learn_embeddings` / `main`).
- Created `plot_emb.py`: PCA-project an `.emb` to 2D, color nodes by degree (hubs deg>10 green, leaves deg<2 red). Sanity-check that I2V embeddings separate structural roles.
  - `python plot_emb.py output/cora.emb input/cora.edgelist cora_plot.png`
  - Outputs: `cora_plot.png`, `cora_3color.png`.
- Created `prepare_linkpred.py` — **empty stub (0 bytes).** TODO: implement 70:30 edge split for link prediction (AUC), retrain on 70% only (no leakage), per I2V Table 4.

## 2026-06-13 — reconstructed

- Trained I2V on webkb → `output/webkb.emb` (265 nodes).
  - `python train.py --input input/webkb.edgelist --output output/webkb.emb` (defaults; exact flags not logged)
- FINDING: very slow. `webkb_run.log` shows ~52 min for walk 1/10 and ~57 min for walk 2/10 on only 265 nodes → ~9 h for the full run. Confirms the I2V efficiency problem.

## 2026-06-15

- Derived `input/citeseer.edgelist` from `input/citeseer.txt` (created 11:56).
  - TODO: document the derivation step (script/command unknown — not logged). Needed for reproducibility.
- Started I2V training on citeseer (3264 nodes) → `citeseer_run.log` ("Number of Nodes: 3264").
  - `python train.py --input input/citeseer.edgelist --output output/citeseer.emb` (likely via nohup; exact flags not logged)
  - STATUS: **incomplete** — no `output/citeseer.emb` yet. Expected to be very slow at 3264 nodes given the webkb rate. TODO: confirm finished / rerun under the cached variant.

- FINDING (baseline efficiency, the project's motivation): I2V recomputes structural signal *inside the walk loop*.
  - `identity2vec.py:24` `eigenvector_centrality()` runs `nx.eigenvector_centrality(G, max_iter=1000)` over the **whole graph** on every call.
  - Called in `get_prob()` (`:92`) **and again** in `poisson_dist()` (`:117`) — i.e. per-neighbor, per-step, per-walk.
  - `node_neighbors()` (`:28`) rebuilds the full neighbor dict every `identity_walker` step (`:59`).
  - → For a static graph, degree + eigenvector centrality are constant. Computing once and **caching** is exact and removes the dominant cost. This is Deliverable #1 (cached I2V variant, embeddings identical, timing gain).

- DEVIATION: walk length. `train.py:27` sets `--walk-length` default = **40**, but the help text says "Default is 80." Repo uses 40; I2V paper says 80. Decide and pin one value for all reproductions; record which.

- Housekeeping: created `docs/` — copied `CLAUDE.md` into it (kept the root copy; **CLAUDE.md must stay in repo root to govern agentic work**), moved `Research_Proposal.pdf` in, started this `notes.md`. Also created `results/` (PNGs) and `logs/` (.log files).

## 2026-06-16

- Wrote the evaluation layer (the paper reports F1 / AUC; embeddings alone don't prove reproduction):
  - `prepare_linkpred.py` — 70:30 edge split, seed=42. Forces a spanning tree into train so the train graph stays connected; samples equal-count non-edges for train + test. Uses the largest connected component only.
  - `eval_linkpred.py` — Hadamard edge features (node2vec default) -> logistic regression -> test AUC. Operator switchable via `--op`.
  - `eval_nodeclass.py` — stratified split (default 80% train) -> logistic regression -> weighted F1.
  - `labels/` and `splits/` dirs, each with a format README.
- Ran the cora split: largest CC = 2485 nodes / 5069 edges -> train_pos=3548, test_pos=1521 (30.0%), balanced negatives. Files in `splits/`.
  - `python prepare_linkpred.py --input input/cora.edgelist --name cora`
- DECISIONS (proposal PDF is 1 page and silent on exact protocol -> chose standard I2V/node2vec conventions; revisit if the original I2V source differs):
  - Link-pred = logistic regression on Hadamard features, AUC.
  - Node-class = logistic regression, weighted F1 (per CLAUDE.md), single stratified 80/20 split, seed=42.
- DEPENDENCY: added `scikit-learn` (logreg, SVM, roc_auc, f1) — not in the base env. Install: `pip install scikit-learn`. Only the eval scripts use it; core I2V untouched.
- BLOCKER: no label files in the repo. Node classification cannot run until `labels/{cora,citeseer}.labels` exist with IDs matching the edgelists (Cora IDs 1..2708). Link prediction is unaffected.
- NOTE: a valid link-pred AUC needs an embedding retrained on `splits/{name}_train.edgelist` — the full-graph `cora.emb` would leak. Pipeline in `splits/README.md`.
- Cora train-only I2V run stopped: very slow and produced overflow warning at identity2vec.py line 128 during Poisson score calculation.

- Built a CoBench-style reproduction framework under `scripts/` (mirrors CLADBench's `benchmark_config.py` / `utils.py` / `results_io.py` / `runner.py` / `main.py`). **No files moved** — it wraps the existing root scripts (`train.py` via subprocess; `prepare_linkpred` / `eval_*` via import). Made the 3 eval scripts importable (callable `prepare()` / `evaluate()` core + thin `main`).
  - New dirs/files: `scripts/`, `configs/` (+ sample `cora_linkpred.json`), `notebooks/virgo_dev.ipynb`.
  - Run: `python scripts/main.py --list` / `--task linkpred --dataset cora [--retrain]`. Results -> `results/NNN.{dataset}.{task}.csv` with a JSON `#META` header (run id, seed, settings, counts).
  - DEPENDENCY: installed `scikit-learn` 1.9.0 — landed in conda env **`i2v`** (python 3.12), which is the active interpreter here (not `cobench`). Worth confirming this is the intended env for the project.
  - Smoke test: `python scripts/main.py --task linkpred --dataset cora --emb output/cora.emb` -> AUC=0.9578, saved `results/001.cora.linkpred.csv`. This is a **plumbing check only** — uses the full-graph `cora.emb`, so it leaks; not a paper number. A real AUC needs `--retrain` (embed on `splits/cora_train.edgelist` first).

## 2026-06-17

- DECISION: **walk-length pinned = 40** for all reproductions. Rationale: repo default + `benchmark_config.I2V_PARAMS` + the author's `output/cora.emb` (which gave F1=0.6992) all use 40; the paper text's 80 is kept as a recorded deviation but NOT used. Resolves the 06-15 walk-length DEVIATION.

- AUDIT (read-only, full repo vs CLAUDE.md / proposal): seed=42 consistent across `train` / `prepare_linkpred` / `eval_*` / `utils` / `REPRO`; `input/` untouched; cached overrides verified safe (`s_path` returns length only, `node_neighbors` returns fresh copies); split filenames match between `prepare` (writer) and `eval_linkpred` (reader). Env: `python` -> conda **i2v** (`sklearn 1.9.0`, `gensim 4.3.3`, `numpy 1.26.4`) — all deps present; the bash `libtinfo` warning is the interactive shell sourcing `cobench`, cosmetic only. Dataset sizes: cora 2708/5278, citeseer 3264/4536, webkb 265/479, politics 18470, enzymes 19474, dhfr 32075, nci 101924. NOTE: `input/proteins.edgelist` and `input/firstmmedges.edgelist` read as 0 nodes (empty/malformed) — not in the dataset registry, harmless for now.

- FIX (accuracy repairs — NOT the cache fix):
  - `notebooks/reproduce_i2v.ipynb` was BROKEN: Cell 8 imported the deleted `make_planetoid_labels`; its saved outputs were a failed run (overlap 0.0028 -> STOP -> `FileNotFoundError`). Changed import to `from make_labels import make_labels` and re-ran headless. Now clean end-to-end: `edge_overlap=1.0000`, labels written, **weighted F1 = 0.6992**, saved `results/003.cora.nodeclass.csv`.
    - `python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=python3 notebooks/reproduce_i2v.ipynb`
  - `train.py` stale help strings corrected: `--walk-length` "Default is 80." -> "40."; `--workers` "Default is 8." -> "1." (defaults were already 40 / 1; only the help text was wrong).

- FIX1 DONE — **cache wired + speedup proven** (Deliverable #1). `scripts/runner.embed` now passes `--cached` (default on) + `--seed`, so every pipeline retrain uses the fast path. Benchmark on webkb (265n/479e, num-walks=2 walk-length=10, seed=42): baseline **916.6s → cached 4.4s = 207.7×** (≈646× minus `train.py`'s fixed 3s sleep); both `.emb` md5 `e458fa5e2a360ac388803ee990afb312` → **BYTE-IDENTICAL**. Confirms the cache changes no computed value and removes the dominant cost.
  - `python train.py --input input/webkb.edgelist --output <out> --num-walks 2 --walk-length 10 --seed 42 [--cached]`

---

## TODO backlog (open threads)

- [x] `prepare_linkpred.py` (70:30 split, seed=42) — done, verified on cora.
- [x] CoBench-style framework under `scripts/` — done, smoke-tested (`results/001`).
- [x] Install scikit-learn — done (in env `i2v`; confirm that's the right env).
- [x] Notebook-first workflow: `notebooks/reproduce_i2v.ipynb` (one click-through notebook reusing the tested functions). Installed + registered `ipykernel` ("Python (i2v)" kernel) for VS Code. Removed the old `virgo_dev.ipynb`.
- [x] Made `make_planetoid_labels.py` importable (guarded its run-calls with `if __name__ == "__main__"`); node-class scorer verified with throwaway labels (the F1 path works).
- [x] Training reproducibility fixed (`train.py` only — `identity2vec.py` untouched): added `--seed` (default 42), seeded global `np.random` before the walks (covers identity2vec's `np.random.shuffle`/`choice`), and passed `seed=args.seed` to `Word2Vec`. Verified byte-identical `.emb` across two same-seed runs, and different across seeds. **Requires `--workers 1`** (the default) — gensim is nondeterministic with multiple workers even with a seed.
- [x] **Cora labels solved + node classification reproduced.** FINDING: the author's `input/cora.edgelist` numbers nodes by **order of appearance in the LINQS `cora.content`** file — that ordering reproduces the edgelist at **edge_overlap = 1.0000**. Planetoid Cora is the *same graph but a different numbering* (overlap 0.003, identical degree sequences), so its labels are WRONG here — the safety check correctly refused them. The author's `input.zip` ships **no labels** at all. Wrote `make_labels.py` (LINQS `.tgz` via plain `urllib` — bypasses PyG's flaky `fsspec` that gave `FSTimeoutError`; verifies overlap before writing); deleted the misleading `make_planetoid_labels.py`.
  - RESULT (first real reproduction): **Cora node classification weighted F1 = 0.6992** (2708 nodes, 7 classes, `train_frac=0.8`, `seed=42`, author's `output/cora.emb`). Saved `results/002.cora.nodeclass.csv`.
  - `notebooks/reproduce_i2v.ipynb` runs fully end-to-end (verified headless, all 8 cells pass).
  - [ ] Compare 0.6992 to the paper's **Figure 5 / Section 4.4** — that figure is likely an F1-vs-train-ratio curve, so add a train-ratio sweep (and confirm the paper's F1 averaging: weighted vs micro/macro) to match it exactly.
  - [ ] Citeseer labels: our `citeseer.edgelist` was *derived by us*, so its numbering may not match LINQS `citeseer.content` order — run `make_labels('Citeseer')` and check the overlap before trusting it.
- [ ] Retrain I2V on `splits/cora_train.edgelist` -> run `eval_linkpred.py` -> record AUC vs I2V Table 4.
- [ ] `eval_rank.py` (pairwise SVM, AUC) — only if reproducing learning-to-rank (the proposal omits it).
- [ ] Document `citeseer.edgelist` derivation from `citeseer.txt`.
- [ ] Finish/confirm citeseer embedding.
- [x] Build cached I2V variant; verify embeddings identical + measure speedup (Deliverable #1) — done 2026-06-17: webkb 207.7× faster, byte-identical; pipeline uses `--cached`.
- [x] Pin walk-length: **40** (decided 2026-06-17; paper's 80 recorded as deviation, not used).
- [ ] `virtual_graph.py` — top-K Ψ builder (Deliverable #2).
