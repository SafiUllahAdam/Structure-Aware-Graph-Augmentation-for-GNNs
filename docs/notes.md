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

- DECISION: **walk-length = 40 (active).** Flipped 40 -> 80 -> 40 today: set to 80 for paper-fidelity, then reverted to 40 per request. `train.py` default and `benchmark_config.I2V_PARAMS` are back to 40. The paper text says 80 (kept as a recorded deviation). 40 matches the repo default and the author's `output/cora.emb` (F1=0.6992). NOTE: on-disk `output/cora_lp.emb` is currently an **80-walk** embedding (AUC 0.7972); retrain at 40 to refresh a 40 link-pred number.

- AUDIT (read-only, full repo vs CLAUDE.md / proposal): seed=42 consistent across `train` / `prepare_linkpred` / `eval_*` / `utils` / `REPRO`; `input/` untouched; cached overrides verified safe (`s_path` returns length only, `node_neighbors` returns fresh copies); split filenames match between `prepare` (writer) and `eval_linkpred` (reader). Env: `python` -> conda **i2v** (`sklearn 1.9.0`, `gensim 4.3.3`, `numpy 1.26.4`) — all deps present; the bash `libtinfo` warning is the interactive shell sourcing `cobench`, cosmetic only. Dataset sizes: cora 2708/5278, citeseer 3264/4536, webkb 265/479, politics 18470, enzymes 19474, dhfr 32075, nci 101924. NOTE: `input/proteins.edgelist` and `input/firstmmedges.edgelist` read as 0 nodes (empty/malformed) — not in the dataset registry, harmless for now.

- FIX (accuracy repairs — NOT the cache fix):
  - `notebooks/reproduce_i2v.ipynb` was BROKEN: Cell 8 imported the deleted `make_planetoid_labels`; its saved outputs were a failed run (overlap 0.0028 -> STOP -> `FileNotFoundError`). Changed import to `from make_labels import make_labels` and re-ran headless. Now clean end-to-end: `edge_overlap=1.0000`, labels written, **weighted F1 = 0.6992**, saved `results/003.cora.nodeclass.csv`.
    - `python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=python3 notebooks/reproduce_i2v.ipynb`
  - `train.py` stale help strings corrected: `--walk-length` "Default is 80." -> "40."; `--workers` "Default is 8." -> "1." (defaults were already 40 / 1; only the help text was wrong).

- FIX1 DONE — **cache wired + speedup proven** (Deliverable #1). `scripts/runner.embed` now passes `--cached` (default on) + `--seed`, so every pipeline retrain uses the fast path. Benchmark on webkb (265n/479e, num-walks=2 walk-length=10, seed=42): baseline **916.6s → cached 4.4s = 207.7×** (≈646× minus `train.py`'s fixed 3s sleep); both `.emb` md5 `e458fa5e2a360ac388803ee990afb312` → **BYTE-IDENTICAL**. Confirms the cache changes no computed value and removes the dominant cost.
  - `python train.py --input input/webkb.edgelist --output <out> --num-walks 2 --walk-length 10 --seed 42 [--cached]`

## 2026-06-18

- WebKB labels resolved. Paper ref **[16] = Network Repository**; webkb appears ONLY in the paper's Figure 2 t-SNE viz (NOT in Table 1 stats nor eval Tables 2/3/4) — so I2V never ran a labelled task on it. Source found: the author's own repo `github.com/ikenna-oluigbo/webkb-dataset` (4 universities). The **Wisconsin** subset = our `input/webkb.edgelist` (265n/479e, **proven isomorphic**, 479/479 edges).
  - PROBLEM: author renumbered nodes between his two repos, so labels do NOT transfer to `input/webkb.edgelist` by id. Isomorphism recovery is ambiguous: 252/265 nodes uniquely labelled, **13 structurally ambiguous** (automorphisms cross class). So `input/webkb.edgelist` labels are unrecoverable.
  - DECISION: shipped the author's **consistent pair** instead (100% correct, zero ambiguity): `input/webkb_wisc.edgelist` + `labels/webkb_wisc.labels` (node id = `wisconsin.content` line order, label = last field). 5 classes — student 122, course 76, faculty 35, project 22, staff 10. Reproducible via `make_labels.make_webkb()`. Registered `webkb_wisc` in `benchmark_config`; `webkb` stays labels=None (the I2V-numbered graph, used for the cache benchmark).

## 2026-06-19

- Dataset **resolver + aligned Citeseer + micro/macro/weighted F1** (implements the "one version per dataset, auto-align" design).
  - `make_labels.py`: added `make_citeseer_linqs()` — builds `input/citeseer_linqs.edgelist` + `labels/citeseer_linqs.labels` from ONE LINQS source so ids align by construction (3312 papers, 4536 cites edges, 6 classes; largest content graph = 3264 connected nodes, 0 self-loops). Added `resolve_dataset()` — author `citeseer` prints the mismatch reason then auto-switches to `citeseer_linqs`; cora/webkb_wisc pass through. `ensure_labels` routes `citeseer_linqs`.
  - `scripts/benchmark_config.py`: `citeseer` labels=None (author graph has no aligned labels); added `citeseer_linqs`; `nodeclass_train_frac` 0.80 -> **0.70**.
  - `eval_nodeclass.evaluate` now returns **{micro,macro,weighted}** F1 dict (default train_frac 0.7); updated `main`, `runner.run_nodeclass`, notebook Steps 1/4/5.
  - Notebook Step 1 calls `resolve_dataset(DATASET)` so `citeseer` auto-aligns to `citeseer_linqs` for the WHOLE run (node-class + link-pred), printing why; author files untouched.
  - VERIFIED: cora node-class @0.7 = micro **0.7036** / macro 0.6710 / weighted **0.7009** (still reproduced). resolve passthrough OK. citeseer_linqs builds; node-class + link-pred plumbing runs (checked with a throwaway walk-10 emb).
  - NOTE: citeseer_linqs embedding is slow even cached (3264 nodes); the 207x cache win was on tiny webkb/walk-10. First build is minutes — run once in the background.
- WALK-LENGTH UNIFIED at **40** everywhere: notebook config 80 -> 40 (`train.py` + `benchmark_config` already 40). Paper's 80 = recorded deviation, not used. Existing walk-80 `.emb` files (`cora_lp`, `citeseer`, `citeseer_lp`, `webkb_wisc`, `webkb_wisc_lp`) are now **stale** vs the 40 setting -> rebuild (FORCE_EMBED / delete) for a clean 40 set. `cora.emb` = author reference, keep.

## 2026-06-20

- Repro bar relaxed **±0.03 -> ±0.05** (CLAUDE.md, README, virgo_guide) per request. Walk-length stays **40** (fixed decision).
- AUDIT (senior pass, read-only) vs I2V paper Eqs 1-9 + Tables 1-4: pipeline faithful to the AUTHOR's released code (upstream remote = ikenna-oluigbo/identity2vec; `identity2vec.py` untouched since import). FINDING: released code deviates from paper Eqs 2-4 — code uses p=degree*eigcent, q=dist, norm=deg+cent; paper uses p=degree, q=cent*dist, norm=cent. **Inherited, not ours** -> document in paper, do NOT change (keeps baseline comparable). DEVIATION (acceptable): link-pred forces full MST into train, so bridge edges are never test positives (mild AUC optimism). Cache verified exact; splits leakage-free; seeds reproducible.
- **Baseline benchmark framework added** (no runs executed; code-only per request):
  - NEW `embedding_models.py`: base `EmbeddingModel` + `Identity2VecModel` (wraps train.py), `DeepWalkModel` (node2vec p=q=1), `Node2VecModel` (p=1,q=0.5 — paper doesn't fix p/q, logged default), `Struc2VecModel` (vendored CLI). `MODELS` registry + `get_model`.
  - NEW `baselines/struc2vec/src/*` — vendored official leoribeiro/struc2vec, mechanically py2->py3 ported (iteritems/xrange/cPickle, gensim4 `vector_size`/`epochs`). DEP added: `node2vec` 0.5.0, `fastdtw`. **struc2vec UNTESTED — a py2 port needs one debug run to finalize; no --seed -> not bit-reproducible.**
  - NEW `scripts/benchmark_baselines.py`: `run_benchmark` loops `BENCH_DATASETS x BENCH_MODELS` reusing `run_*_repeated` (eval untouched); `benchmark_table` -> Table 1 (node-class weighted F1) + Table 2 (link-pred AUC), datasets x methods, mean±std. Saved `results/benchmark/`.
  - EDIT `runner.py`: `run_nodeclass_repeated`/`run_linkpred_repeated` gain `model=` (default identity2vec). I2V emb filenames stay UNPREFIXED so existing cora embeddings are reused; non-I2V get a `{model}_` prefix; splits stay model-independent (shared per dataset/seed = fair). Rows carry a `model` column. `summarize_seed_results` unchanged -> existing Steps 1-4 behave identically.
  - EDIT `benchmark_config.py`: `BENCH_DATASETS`, `BENCH_MODELS`; added `enzymes_nr`/`politics_nr` aligned-fallback registry entries.
  - EDIT `make_labels.py`: `make_enzymes()` (TU `ENZYMES.node_labels`, verified-or-fallback), `make_politics()` (networkrepository, **URL/format best-guess -> verified or STOPs**), `_make_nr_labels` (verify edge-overlap >=0.90 else build self-aligned `<name>_nr`), `_ensure_nr`; registered in `_VERSIONS`/`ensure_labels`/`prepare_dataset` (citeseer pattern).
  - EDIT notebook: appended **Step 5** (markdown + code) calling `run_benchmark`+`save_benchmark`; Steps 1-4 untouched.
  - VERIFIED: `py_compile` + import of all changed files OK (NO training/eval/benchmark executed). TODO before trusting numbers: (1) one struc2vec debug run; (2) decide node2vec p/q.
  - LABEL BUILD CHECK (ran builders only, no training): **enzymes OK** -> `make_enzymes` edge_overlap=1.0000, wrote `labels/enzymes.labels` (19580 nodes, 3 classes), aligned. **politics = LINK-PRED ONLY (no NC):** `input/politics.edgelist` IS networkrepository `rt-pol` (identical 61157 edges), graph only. Original left/right labels (Conover et al. 2011, ICWSM) are at cnets.indiana.edu = **offline (404)**, no clean github/zenodo mirror, and rt-pol's renumbering can't be mapped -> label alignment UNVERIFIABLE. Per policy: **no fabrication**. `make_politics` STOPs with that reason; `benchmark_baselines.run_benchmark` falls back to LP-only so politics appears in Table 2 (AUC) but NOT Table 1 (NC). [Tried Adamic-Glance polblogs as a self-aligned substitute, then REVERTED + deleted — polblogs is a different dataset, not rt-pol.] To enable politics NC, drop in a manually verified `labels/politics.labels` (node id = our numbering).
- **struc2vec py3 port FIXED + verified working** (integrated webkb_wisc run: micro 0.475 / weighted 0.369). Fixes in `baselines/struc2vec/src/`: `collections.abc.Iterable`; `np.int`->`np.int64`; `utils.partition` coerces dict-views to list; `simulate_walks` wraps `self.G.keys()` in `list()` (multiprocessing pickling); pickles dir = `baselines/struc2vec/pickles` (one level up from src). WRAPPER (`Struc2VecModel.train`) now **clears pickles/ + random_walks.txt before each run** — struc2vec caches under FIXED filenames, so without clearing, dataset B would reuse dataset A's structural distances (silent wrong embeddings). Cosmetic only: struc2vec prints `rm: cannot remove ...pickle` (its own startup cleanup) + a benign `invalid value in scalar divide`. struc2vec has NO seed -> not bit-reproducible (variance across seeds is real). WARNING: struc2vec on enzymes (19.5k nodes) with OPT may be very slow/memory-heavy.
- **benchmark hardened:** `run_benchmark` wraps each model's NC + LP in try/except -> one model crashing no longer aborts the sweep or loses other models' rows.
- FINDING (cora, same splits/seeds/params): node2vec NC 0.8165 / LP 0.9107 >> I2V NC 0.6906 / LP 0.8011. Expected — cora is homophilous, proximity methods (node2vec/deepwalk) beat structural-identity (I2V) on NC+LP. Our node2vec matches the node2vec literature; the paper's node2vec (LP 0.7658) is under-tuned, which is why the paper shows "I2V beats all". DO NOT claim I2V beats all on homophilous NC/LP; I2V's edge is structural tasks (heterophily/webkb, roles, anomaly).
- **DECISION: dropped politics from the benchmark, swapped in `webkb_wisc`** (265 nodes, 5 classes, labels already verified/isomorphic). `BENCH_DATASETS = [cora, citeseer_linqs, enzymes, webkb_wisc]`; notebook Step 5 examples updated. Both Table 1 (NC) + Table 2 (LP) now cover **all 4** datasets. CAVEAT: webkb is heterophilous (content classes vs structural roles) -> I2V/struc2vec NC F1 may be low/near-random on it; that's expected, report as-is.

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
- [x] walk-length = **40** (active; flipped 40 -> 80 -> 40 on 2026-06-17). `train.py` + `benchmark_config` = 40. Paper's 80 = recorded deviation.
- [ ] On-disk `output/cora_lp.emb` is an 80-walk embedding (AUC 0.7972); retrain at 40 to refresh a 40 link-pred number if wanted.
- [ ] `virtual_graph.py` — top-K Ψ builder (Deliverable #2).
