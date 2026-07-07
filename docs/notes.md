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
- I2V `train.py` defaults: `dimensions=64`, `walk-length=40`, `num-walks=10`, `window-size=10`, `epochs=1`, `sg=1` (skipgram), `min-count=0`, `workers=1`, `e=2.7182`. Word2Vec: `alpha=0.025 → min_alpha=0.01`, `negative=5`, `sample=1e-3` (Fix 6, 2026-06-24; was `alpha=0.25` / `sample=1e-5`).
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

- Built a reproduction framework under `scripts/` (mirrors CLADBench's `benchmark_config.py` / `utils.py` / `results_io.py` / `runner.py` / `main.py`). **No files moved** — it wraps the existing root scripts (`train.py` via subprocess; `prepare_linkpred` / `eval_*` via import). Made the 3 eval scripts importable (callable `prepare()` / `evaluate()` core + thin `main`).
  - New dirs/files: `scripts/`, `configs/` (+ sample `cora_linkpred.json`), `notebooks/virgo_dev.ipynb`.
  - Run: `python scripts/main.py --list` / `--task linkpred --dataset cora [--retrain]`. Results -> `results/NNN.{dataset}.{task}.csv` with a JSON `#META` header (run id, seed, settings, counts).
  - DEPENDENCY: installed `scikit-learn` 1.9.0 — landed in conda env **`i2v`** (python 3.12), which is the active interpreter here. Worth confirming this is the intended env for the project.
  - Smoke test: `python scripts/main.py --task linkpred --dataset cora --emb output/cora.emb` -> AUC=0.9578, saved `results/001.cora.linkpred.csv`. This is a **plumbing check only** — uses the full-graph `cora.emb`, so it leaks; not a paper number. A real AUC needs `--retrain` (embed on `splits/cora_train.edgelist` first).

## 2026-06-17

- DECISION: **walk-length = 40 (active).** Flipped 40 -> 80 -> 40 today: set to 80 for paper-fidelity, then reverted to 40 per request. `train.py` default and `benchmark_config.I2V_PARAMS` are back to 40. The paper text says 80 (kept as a recorded deviation). 40 matches the repo default and the author's `output/cora.emb` (F1=0.6992). NOTE: on-disk `output/cora_lp.emb` is currently an **80-walk** embedding (AUC 0.7972); retrain at 40 to refresh a 40 link-pred number.

- AUDIT (read-only, full repo vs CLAUDE.md / proposal): seed=42 consistent across `train` / `prepare_linkpred` / `eval_*` / `utils` / `REPRO`; `input/` untouched; cached overrides verified safe (`s_path` returns length only, `node_neighbors` returns fresh copies); split filenames match between `prepare` (writer) and `eval_linkpred` (reader). Env: `python` -> conda **i2v** (`sklearn 1.9.0`, `gensim 4.3.3`, `numpy 1.26.4`) — all deps present; the bash `libtinfo` warning is the interactive shell, cosmetic only. Dataset sizes: cora 2708/5278, citeseer 3264/4536, webkb 265/479, politics 18470, enzymes 19474, dhfr 32075, nci 101924. NOTE: `input/proteins.edgelist` and `input/firstmmedges.edgelist` read as 0 nodes (empty/malformed) — not in the dataset registry, harmless for now.

- FIX (accuracy repairs — NOT the cache fix):
  - `notebooks/1-reproduce_i2v.ipynb` was BROKEN: Cell 8 imported the deleted `make_planetoid_labels`; its saved outputs were a failed run (overlap 0.0028 -> STOP -> `FileNotFoundError`). Changed import to `from make_labels import make_labels` and re-ran headless. Now clean end-to-end: `edge_overlap=1.0000`, labels written, **weighted F1 = 0.6992**, saved `results/003.cora.nodeclass.csv`.
    - `python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=python3 notebooks/1-reproduce_i2v.ipynb`
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

### 2026-06-23 — walk-length 80 tried, then reverted to 40 (timing-driven)

- Flipped 40 -> 80 for paper-fidelity, then **reverted to 40** after measuring walk-generation cost. Code (`train.py` default+help, `scripts/benchmark_config.I2V_PARAMS`) and docs (README, `docs/virgo_guide.md`, `CLAUDE.md`, this file: defaults + checklist) are back to **40**. Notebook inherits the config (no hardcoded value).
- TIMING — cora, **cached** path, 10 walks/node, seed=42, walk-generation only (no Word2Vec):

  | walk_length | wall time |
  |---|---|
  | 40 | 1010.8 s (~16.8 min) |
  | 80 | 1894.0 s (~31.6 min) |
  | **diff** | **+883.2 s (~14.7 min) = 1.87x slower** |

  - Measured via `scratchpad/time_walks.py`: `identity2vec_cached.Graph(cora).identity2vec_walk(10, wl)` for `wl in {40, 80}`. Each call includes one fixed 3 s sleep -> the diff is sleep-free. ~1.87x (near-linear; sub-2x because the per-graph eigenvector compute + cached BFS amortize).
  - This is the CACHED path. Uncached is far slower: the notebook's uncached cora walk-80 ran ~2:13/walk x 10 ~= 22 min for NC alone.
- PERFORMANCE (40 vs 80) — _placeholder, fill by hand:_
  - cora NC weighted F1:  40 = ____  |  80 = ____
  - cora LP AUC:          40 = ____  |  80 = ____
- DECISION: **walk-length = 40 active.** 80 = paper value, kept as a recorded deviation, not used — 1.87x slower with no confirmed metric gain (see placeholder).

## 2026-06-24 — paper-fidelity fixes (professor review), Fix 8 effect, proposed sampling

Professor diffed repo vs paper -> 8 suggested fixes. Status verified fresh from disk (line refs current). Fixes landed over the last few sessions; recorded here together.

- **Fix 1 — selection direction. DONE, version (b).** `identity2vec.py:92-94`. Compute Ψ of the current node, then pick the candidate minimizing `|Ψ_candidate − Ψ_current|` (least-dissimilar). Option (a) `max(pdn, key=pdn.get)` left commented as a record.
- **Fix 2 — degree distribution, not raw degree. DONE.** Added `degree_distribution()` (Δ_u = n_d/n) at `identity2vec.py:27-39`; used for the `p` signal in `get_prob`. (Raw `degree_node()` now appears only in the Fix-4 divisor.)
- **Fix 3 — p/q composition. DONE.** `get_prob:132-133`: `p = Δ`, `q = Ω·d`. Eigenvector moved out of the numerator back into the denominator; distance penalty restored. Matches paper Eqs 3-4.
- **Fix 4 — normalize by candidate, not previous node. DONE ("4A").** `identity_score:154-158`: `normalizer = degree_node[node] + eigenvector[node]` of the candidate being scored (was `bounded_curr` — constant across candidates, so it didn't discriminate). NOTE: uses RAW degree + eigenvector; professor offered raw OR degree-distribution for ω -> confirm the choice.
- **Fix 5 — walk-length 80. NOT applied (intentional).** Kept 40 (timing 1.87× slower, see 2026-06-23 entry). Paper's 80 = recorded deviation.
- **Fix 6 — Word2Vec hyperparameters. DONE.** `train.py:86`: `alpha 0.25 -> 0.025`, `sample 1e-5 -> 1e-3` (min_alpha / negative / seed unchanged). Removes the 10× learning rate that only I2V used.
- **Fix 7 — cached walker in sync. DONE.** `degree_distribution` cached at `identity2vec_cached.py:29-43` (`_deg_dist`). `identity_score` (Fixes 4A+8) is inherited (not overridden) and calls the cached signals via `self` -> cached and non-cached paths stay identical.
- **Fix 8 — Poisson in log-space. DONE.** `identity2vec.py:160-167`, `from scipy.special import gammaln` (:9). `drt = max(drt, 1e-12)`; `log_poiss = k·log(drt) − drt − gammaln(k+1)`. Fixes underflow-to-0 on hubs AND the latent `factorial(k)` overflow for k>170.

- **REVERSAL of the 2026-06-20 "DO NOT change" decision.** That audit kept the released-code deviations (p=deg·eigcent, q=dist, norm=deg+cent) for baseline comparability. Per the professor review + the paper-fidelity goal, Fixes 2/3/4 now move the code to the paper's Eqs 2-4. The original released-code baseline stays in git history (can be a separate comparison column if needed).

- **FINDING — Fix 8 effect (cora):** node classification improved and now matches the paper; link-pred AUC dropped from best-recorded **0.8494 -> ~0.81**.
  - Why: before Fix 8 the Poisson underflowed to 0 on high-degree nodes -> many tied scores -> selection effectively RANDOM -> the walk wandered the local neighborhood (proximity) -> accidentally boosted LP. Fix 8 made the scores meaningful -> the walk now follows structural identity as intended -> NC up, LP down.
  - Context: the paper's own cora LP = **0.8413** (Table 4); 0.81 is within the ±0.05 repro bar. So the code now matches the paper on BOTH tasks — the 0.8494 was a bug-driven over-shoot, not a real LP capability.
  - CAVEAT 1: confirm 0.8494 -> 0.81 is real, not one-seed luck — re-run 3 seeds, mean±std.
  - CAVEAT 2: Fix 8 also interacts with Fix 1b — log-space warps the `|Ψ−Ψ|` closeness geometry, so selection changes even where nothing underflowed. Regenerate `.emb`.

- **STALE RESULTS.** Fixes 1/3/4/6/8 all changed the math -> existing `.emb` and `results/` are from old code and do NOT reflect it. Regenerate embeddings (delete / FORCE_EMBED) before trusting any metric.

### 2026-06-24 — temperature sampling (Fix 8 extension) — REMOVED 2026-07-07 ❌

A non-greedy softmax next-node sampler (`--temperature`, benchmark τ=0.3) was added here — cora s42 NC 0.7486 / LP 0.8305, an improvement on both — then **removed entirely on 2026-07-07** to keep I2V paper-exact for the ViRGo comparison (the paper selects "least dissimilar" = greedy). Selection is back to `min(pdn, key=|Ψ−Ψ_curr|)` (`identity2vec.py`); `--temperature` gone from `train.py`, `embedding_models.py`, `I2V_PARAMS`. Cached path inherits the walker, so it is greedy again too. Any I2V numbers produced with τ=0.3 (incl. the cross-model benchmark tables) are stale — regenerate `.emb` greedy before quoting.

### 2026-06-25 — cross-model config standardized for fair benchmark ✅

Goal: identical Skipgram/Word2Vec **training** across all 4 models so the comparison is fair; **I2V = anchor, left untouched** (it's stable). Walk generation + method-defining knobs stay per-model.

- I2V: **UNCHANGED** (alpha 0.025, min_alpha 0.01, sample 1e-3, negative 5, hs 0). Its embeddings stay valid -> keep cora s42 NC 0.7486 / LP 0.8305; **no I2V retrain**.
- node2vec + DeepWalk (`embedding_models.py:48-50`): pinned the Skipgram params explicitly to match I2V — `alpha=0.025, min_alpha=0.01, sample=1e-3, negative=5, hs=0`. Only `min_alpha` actually differed before (gensim default 0.0001 -> 0.01); the rest already equalled I2V via gensim defaults but are now explicit/locked.
- struc2vec (`baselines/struc2vec/src/main.py:81`): switched **hierarchical-softmax -> negative sampling** (`hs=0, negative=5`) + `alpha=0.025, min_alpha=0.01, sample=1e-3` to match I2V — also matches the paper's stated protocol ("negative sampling for DeepWalk and struc2vec"). Walk-length **default 80 -> 40** (`:30`) for standalone consistency.
- walk_length = **40 for all** — benchmark already forced 40 via `I2V_PARAMS`; struc2vec standalone default now 40 too. (80 tried earlier, no gain — see 2026-06-23.)
- KEPT per-model (NOT standardized — these define each method): node2vec `p=1/q=0.5`, DeepWalk `p=q=1`, I2V Poisson/KL walk, struc2vec `OPT1/2/3` + multilayer structural walk.
- VERIFIED: `py_compile` of both edited files OK (no run executed).
- EFFECT / next: regenerate ONLY the 3 baselines' `.emb` (their config changed; delete + re-run), then re-score. struc2vec will shift (hs->neg, likely up toward its paper ~0.71 LP); node2vec/DeepWalk shift slightly (min_alpha); I2V flat. struc2vec still has NO seed -> report mean±std.

---

## TODO backlog (open threads)

- [x] `prepare_linkpred.py` (70:30 split, seed=42) — done, verified on cora.
- [x] framework under `scripts/` — done, smoke-tested (`results/001`).
- [x] Install scikit-learn — done (in env `i2v`; confirm that's the right env).
- [x] Notebook-first workflow: `notebooks/1-reproduce_i2v.ipynb` (one click-through notebook reusing the tested functions). Installed + registered `ipykernel` ("Python (i2v)" kernel) for VS Code. Removed the old `virgo_dev.ipynb`.
- [x] Made `make_planetoid_labels.py` importable (guarded its run-calls with `if __name__ == "__main__"`); node-class scorer verified with throwaway labels (the F1 path works).
- [x] Training reproducibility fixed (`train.py` only — `identity2vec.py` untouched): added `--seed` (default 42), seeded global `np.random` before the walks (covers identity2vec's `np.random.shuffle`/`choice`), and passed `seed=args.seed` to `Word2Vec`. Verified byte-identical `.emb` across two same-seed runs, and different across seeds. **Requires `--workers 1`** (the default) — gensim is nondeterministic with multiple workers even with a seed.
- [x] **Cora labels solved + node classification reproduced.** FINDING: the author's `input/cora.edgelist` numbers nodes by **order of appearance in the LINQS `cora.content`** file — that ordering reproduces the edgelist at **edge_overlap = 1.0000**. Planetoid Cora is the *same graph but a different numbering* (overlap 0.003, identical degree sequences), so its labels are WRONG here — the safety check correctly refused them. The author's `input.zip` ships **no labels** at all. Wrote `make_labels.py` (LINQS `.tgz` via plain `urllib` — bypasses PyG's flaky `fsspec` that gave `FSTimeoutError`; verifies overlap before writing); deleted the misleading `make_planetoid_labels.py`.
  - RESULT (first real reproduction): **Cora node classification weighted F1 = 0.6992** (2708 nodes, 7 classes, `train_frac=0.8`, `seed=42`, author's `output/cora.emb`). Saved `results/002.cora.nodeclass.csv`.
  - `notebooks/1-reproduce_i2v.ipynb` runs fully end-to-end (verified headless, all 8 cells pass).
  - [ ] Compare 0.6992 to the paper's **Figure 5 / Section 4.4** — that figure is likely an F1-vs-train-ratio curve, so add a train-ratio sweep (and confirm the paper's F1 averaging: weighted vs micro/macro) to match it exactly.
  - [ ] Citeseer labels: our `citeseer.edgelist` was *derived by us*, so its numbering may not match LINQS `citeseer.content` order — run `make_labels('Citeseer')` and check the overlap before trusting it.
- [ ] Retrain I2V on `splits/cora_train.edgelist` -> run `eval_linkpred.py` -> record AUC vs I2V Table 4.
- [ ] `eval_rank.py` (pairwise SVM, AUC) — only if reproducing learning-to-rank (the proposal omits it).
- [ ] Document `citeseer.edgelist` derivation from `citeseer.txt`.
- [ ] Finish/confirm citeseer embedding.
- [x] Build cached I2V variant; verify embeddings identical + measure speedup (Deliverable #1) — done 2026-06-17: webkb 207.7× faster, byte-identical; pipeline uses `--cached`.
- [x] walk-length = **40** (active; flipped 40 -> 80 -> 40 again on 2026-06-23 after timing 80 = slower, see entry below). `train.py` + `benchmark_config` = 40. Paper's 80 = recorded deviation.
- [ ] On-disk `output/cora_lp.emb` is an 80-walk embedding (AUC 0.7972); retrain at 40 to refresh a 40 link-pred number if wanted.
- [x] Paper-fidelity fixes (professor review), done 2026-06-24: Fix 1b selection, Fix 2 Δ, Fix 3 p/q, Fix 4A normalizer, Fix 6 Word2Vec, Fix 7 cache sync, Fix 8 log-space. Fix 5 (walk-80) intentionally NOT applied.
- [ ] Regenerate all `.emb` — fixes 1/3/4/6/8 changed the math, on-disk results are stale.
- [ ] Confirm Fix 8 LP drop 0.8494 -> 0.81 is real vs seed noise (3-seed mean±std).
- [ ] Confirm Fix 4 ω choice (raw degree vs degree-distribution) with professor.
- [x] Temperature sampling (2026-06-24 ablation, τ=0.3) — **removed 2026-07-07**, I2V restored to paper-exact greedy; τ TODOs (seed sweep, runner threading) dropped with it. Regenerate I2V `.emb`.
- [x] Cross-model Skipgram config standardized to I2V 2026-06-24 (node2vec/DeepWalk `min_alpha=0.01`; struc2vec hs→negative sampling; walk_length 40 all). I2V untouched; per-model walk knobs kept.
- [ ] Regenerate the 3 baselines' `.emb` (config changed) + re-run benchmark; I2V embeddings stay valid (no retrain).
- [ ] `virtual_graph.py` — top-K Ψ builder (Deliverable #2).

---

### 2026-06-26 — walk-length flipped 40 → 80 (paper value)

- DECISION: **walk-length = 80 (active)**, reverting the 2026-06-23 "40 active" decision per request. 80 = the I2V paper value; ~1.87× slower than 40 (see 2026-06-23 timing), accepted.
- Changed everywhere: `train.py` default+help, `scripts/benchmark_config.I2V_PARAMS["walk_length"]` + comment, `baselines/struc2vec/src/main.py` default+help, and docs (`README.md`, `docs/virgo_guide.md`, `CLAUDE.md`). Notebook inherits the config (no hardcoded value).
- CACHE: deleted all 42 `output/**/*.emb` (every model/dataset/seed, nc+lp) — built at 40, now stale. Next benchmark run rebuilds at 80.
- Supersedes the checklist line above (`walk-length = 40`) and the 2026-06-24 standardization note (`walk_length 40 all`).

### 2026-06-26 — walk-length reverted 80 → 40 (back to repo default)

- DECISION: **walk-length = 40 (active)**, reverting the 2026-06-25 "80 active" flip per request. 40 = repo default + the setting behind author `cora.emb`; the paper's 80 (~1.87× slower, no confirmed gain) is kept as a recorded deviation.
- Reverted everywhere: `train.py` default+help, `scripts/benchmark_config.I2V_PARAMS["walk_length"]` + comment, `baselines/struc2vec/src/main.py` default+help, docs (`README.md`, `docs/virgo_guide.md`, `CLAUDE.md`). Notebook inherits the config (no hardcoded value).
- CACHE: deleted the 24 stale 80-walk `output/cora/*.emb` (rebuilt after the 2026-06-25 flip); next run rebuilds at 40. No author `output/cora.emb` present.
- Supersedes the 2026-06-25 "80 active" entry.

### 2026-06-37 — benchmark scoring finalized + first 3-seed cross-model run

Findings-driven cleanups (implemented one-by-one) + the first full 3-seed run.

- **Link-pred scoring — BOTH computed every seed; headline now COSINE.** `runner.run_linkpred_repeated` scores each `.emb` two ways: `auc_logreg` (Hadamard edge feature → logistic regression, node2vec protocol) and `auc_cosine` (unsupervised cosine similarity, paper-faithful, no classifier). `REPRO["linkpred_score"]` picks the headline `auc` column — now **`"cosine"`** (was `"logreg"`). `benchmark_baselines` writes `table2_linkpred_auc.csv` (=headline=cosine) + `table2_linkpred_auc_cosine.csv` (identical now); `auc_logreg` survives only in `benchmark_per_seed.csv`. Cosine for edges + logreg for labels = paper alignment. STALE COMMENTS (values correct): `benchmark_config.py:43` and `benchmark_baselines.py:79-80` still say "main = logreg".
- **Node-class classifier (Fix 6).** `eval_nodeclass.py:45` = `OneVsRestClassifier(LogisticRegression(max_iter=300, solver="lbfgs", random_state=seed))` (L2 = sklearn default; `multi_class="ovr"` avoided — removed in sklearn ≥1.7). Reports micro/macro/weighted F1.
- **struc2vec OPT (Fix 4).** `Struc2VecModel.train`: `--OPT1/2/3 = False` for graphs ≤10k nodes (cora/citeseer/webkb = exact distance), `True` only for large graphs (enzymes ~19.5k → memory). struc2vec still unseeded → report mean±std.
- **node2vec vs DeepWalk kept distinct.** DeepWalk `p=q=1` (uniform), node2vec `p=1/q=0.5` (biased) → node2vec is not a DeepWalk duplicate. Paper does not fix p/q.
- **FINDING — first 3-seed cross-model run (cora, seeds 42/43/44; I2V ran with τ=0.3, temperature since removed 2026-07-07).** Snapshot captured at **walk-length 80** (06-25→06-29 window); those `.emb` were deleted in the 40-revert → **regenerate at walk-40 + greedy I2V before quoting as final.**

  | model | NC weighted F1 | LP AUC (cosine headline) |
  |---|---|---|
  | identity2vec | 0.7403 ± 0.0116 | 0.8281 ± 0.0085 |
  | deepwalk | 0.8109 ± 0.0216 | 0.9011 ± 0.0017 |
  | node2vec | 0.8166 ± 0.0090 | 0.9031 ± 0.0029 |
  | struc2vec | 0.3219 ± 0.0053 | 0.5491 ± 0.0077 |

  - Confirms the 2026-06-20 finding: on homophilous cora, proximity methods (node2vec/deepwalk) beat structural I2V on NC+LP; struc2vec weak. I2V's edge is structural/heterophilous tasks, not cora.
- **DOCS.** `docs/virgo_guide.md` full-synced to current code (notebook Steps 0–5, 70/30 split, OvR classifier, dataset registry, per-seed `.emb` naming, honest §8). Notebook walk/training progress bars restored (`embedding_models` `quiet=False`).
- TODO: regenerate the table at walk-40; run τ-sweep {0,0.1,0.3,1,3} + τ=0 vs 0.3 (delete I2V `.emb` between τ — filename ignores τ).

### 2026-06-28 — direction: Phase 1 done, focus shifts to the GNN encoder

- DIRECTION (project steer): I2V reproduction is accepted as correct. Baselines are **not** to be fine-tuned — the paper's baselines look under-tuned, but tuning them is out of scope; they stand at published/default settings. Focus moves **fully to the technical contribution**: replace the walk + Skipgram back-end with a modern GNN encoder over the virtual graph.
- NEXT: design several GNN architecture variants (GraphSAGE / GIN / GAT over the Poisson/KL virtual graph) as candidates, then compare them.
- The cross-model baseline table (cora, 3 seeds; see entry above) stands as the reference — no further baseline tuning.

### 2026-06-30 — canonical project phases fixed

Project flow locked into 5 phases (Phase 1 done):
1. **Phase 1 — reproducibility (match the I2V paper). DONE.** Cached I2V + cross-model baselines (used as-is, not fine-tuned); within ±0.05, 3-seed harness.
2. **Phase 2 — virtual-graph creation.** top-K Poisson/KL Ψ graph + degree-only / centrality-only comparison graphs (`virtual_graph.py`). ← next
3. **Phase 3 — modern GNN encoder.** GraphSAGE / GIN / GAT over the virtual graph, replacing walk + Skipgram; design + compare variants.
4. **Phase 4 — downstream tasks.** node classification, link prediction, anomaly detection (new); virtual-graph ablation (which graph best per data/task).
5. **Phase 5 — LLM context-window issue.** structural embeddings as a compact large-graph summary (stretch).
- Refines the earlier 2026-06-30 note: the immediate next is **Phase 2 (virtual graph)**, then Phase 3 (GNN) — not the GNN directly.
- Mirrored in README, CLAUDE.md §4, docs/virgo_guide.md.

### 2026-06-30 — Phase 2 framing: the virtual graph IS the study

- The central research question is **which virtual graph makes a GNN perform best** per task (node classification, link prediction, later anomaly detection). The **virtual graph — not the encoder — is the variable under study.**
- I2V's Poisson/KL similarity graph is **one generic** virtual graph; test whether it works well per task, against simpler variants (degree-only, centrality-only).
- ViRGo is **NOT** mainly "GraphSAGE vs GIN" — encoder choice is secondary.
- Phase 2 order: (1) build the virtual-graph system, (2) test virtual-graph variants; Phase 3: run different GNN encoders on them.
- Mirrored in README, CLAUDE.md §4, docs/virgo_guide.md.

### 2026-07-01 — Benchmark switched to author `citeseer` (drop `citeseer_linqs`)
- DECISION: use the **author's own `citeseer` graph** (`input/citeseer.edgelist`, derived from the paper's `citeseer.txt`) instead of `citeseer_linqs`. It is the paper's actual file → **link prediction only** (no aligned labels, node classification not run on it).
- Note: the two graphs are structurally **identical** (3264 nodes / 4536 edges / largest CC 2110); `citeseer_linqs` was the same graph renumbered + LINQS labels attached (edge overlap 16/4536 = pure relabel). So this switch does **not** change LP structure — it just uses the paper's numbering and skips the unaligned-label workaround.
- Changes: `make_labels.py` — `resolve_dataset` no longer swaps citeseer→citeseer_linqs; `_VERSIONS["citeseer"]=("citeseer","orig","citeseer")`; `prepare_dataset` skips label build for citeseer (LP-only); `__main__` no longer builds citeseer Planetoid labels (they'd be misaligned). `benchmark_config.py` — `BENCH_DATASETS = [cora, citeseer, enzymes, webkb_wisc]`.
- `citeseer_linqs` files + registry entry kept (unused by the benchmark now); can still be called explicitly if NC on Citeseer is ever needed.
- FINDING — first author-`citeseer` LP result (seed 42, walk-40, ran with τ=0.3 — temperature since removed 2026-07-07, regenerate greedy; largest CC = 2110 nodes):

  | score | AUC | paper Table 4 (I2V) | within ±0.05? |
  |---|---|---|---|
  | cosine (headline, paper-faithful) | **0.8606** | 0.8373 | yes (Δ 0.023) |
  | logreg (Hadamard, node2vec protocol) | **0.8771** | — | — |

  - emb `output/citeseer/citeseer_lp_orig_s42.emb`; splits `splits/citeseer/citeseer_lp_orig_s42_*`.
  - **Single seed (s42) — indicative, not final.** TODO: run seeds 43/44 for mean±std.
- **Why LP only, no node classification on citeseer:** the author graph (`input/citeseer.edgelist`) is the paper's own file but ships **no labels**, and its node numbering does **not** match LINQS `citeseer.content` order (edge overlap ~0.00), so LINQS labels would point at the WRONG nodes → any NC would be fake. Node classification on Citeseer needs the aligned `citeseer_linqs` build; on the paper's graph we report **link prediction only** (LP needs no labels).

### 2026-07-01 — Phase 2 started: `virtual_graph.py` (top-K structural virtual-graph builder, Deliverable #2)

- NEW `virtual_graph.py` — mirrors `train.py` shape (`argparse` / `build_graph()` / `main(args)`, core class `VirtualGraph`). Reuses `identity2vec_cached.Graph` for cached degree / Δ / eigenvector centrality (computed once).
- **DECISION (option A):** I2V's Ψ is walk-contextual (needs a reference node + shortest-path). Lifted it to an all-pairs virtual graph via a **reference-free per-node signature**, then top-K nearest signatures. For `psi` the signature = I2V's exact KL→Poisson score (Fix 4A normalizer + Fix 8 log-Poisson) with the walk shortest-path factor dropped (`q = Ω` instead of `Ω·pathlen`). Rejected option B (reuse `identity_score` literally per pair — asymmetric, slower, off-walk).
- **3 variants** (`--sim`, pluggable = add one branch): `psi` (I2V KL/Poisson), `degree` (degree-only), `centrality` (eigenvector-only). Same-K baselines answer "which virtual graph is best per data/task".
- **Constraints enforced + verified on cora (k=10):** same node set (2708, none missing) · 0 self-loops · exactly top-K per node (min_deg=K) · undirected union (symmetric) · finite weights `1/(1+dist)` ∈ (0,1], no NaN/inf · weighted `.edgelist`, NetworkX + PyG (`from_networkx` OK) readable. **Build is deterministic** (byte-identical rebuild).
- cora edge counts: psi 16251 (avg_deg 12.0), degree 26216 (19.4 — integer-degree ties inflate the union), centrality 16110 (11.9). Artifacts: `output/cora/virtual_<sim>_k10.edgelist`.
- Config: `benchmark_config.py` gains `VG_SIMS=[psi,degree,centrality]`, `VG_K=[5,10,20]`, `VG_SEEDS=[42,43,44]`.
- Virtual edgelist is a **drop-in** for the existing walk encoders (DeepWalk/node2vec read any edgelist) and for the Phase 3 GNN (same file) → variant comparison uses ONE fixed encoder, only the graph changes.
- **Next:** (2) variant-comparison harness (build sweep → one fixed encoder → existing `eval_nodeclass`/`eval_linkpred`), then Phase 3 GNN encoder over these graphs.

### 2026-07-01 — Phase 2 notebook + first variant comparison (cora, indicative)

- NEW `notebooks/2-virtual_graph_phase2.ipynb` — self-contained, runs top-to-bottom, **no terminal**: build → visualize → verify constraints → variant sweep → compare on NC + LP → save. Executes clean (nbconvert exit 0), saved with outputs + charts. I2V-repro notebook left frozen as the Phase-1 record.
- **Fixed-encoder protocol:** only the graph changes — same DeepWalk bridge (I2V_PARAMS), same K=10, same seeds [42,43,44]. LP is leak-free (virtual graph rebuilt from the 70% train edges).
- **FIRST comparison (cora, DeepWalk bridge, 3 seeds), `results/vir_graph_variants/cora_variant_comparison_K10.csv`:**

  | variant | node-class weighted-F1 | link-pred AUC |
  |---|---|---|
  | centrality | **0.381** ± 0.007 | 0.551 ± 0.008 |
  | psi (I2V KL/Poisson) | 0.234 ± 0.005 | 0.511 ± 0.006 |
  | degree | 0.152 ± 0.007 | **0.555** ± 0.002 |

- FINDING (indicative): **best virtual graph is task-dependent** — centrality wins node classification, degree wins link prediction; I2V's psi is not best on cora under the DeepWalk bridge. Directly supports the Phase-2 question "which graph per data/task".
- **CAVEATS — not final:** (a) DeepWalk bridge, NOT the Phase-3 GNN (which uses edge weights psi/centrality carry but DeepWalk ignores); (b) cora is homophily/community-labelled, so structural embeddings score modestly (F1) and cosine LP AUC sits near 0.55; (c) single dataset, K=10 only. Re-run the sweep (K=5/20, more datasets) + the GNN before drawing conclusions.
- **Notebook UX (added):** `DATASET` = single knob (Section 1); everything reads the `DATASETS` registry, NC auto-skips + LP-only when `labels is None`. `FORCE_REBUILD=False` = **reuse-if-exists at every stage** (virtual edgelists, `.emb`, LP splits); `True` = regenerate. Verified: re-run reused all (no re-embedding), byte-stable results.

### 2026-07-02 — Output layout redesign + graph-health stats

- **New canonical layout, all datasets, both phases:** everything for a dataset under ONE folder — `output/<dataset>/i2v_main/` = ALL Phase-1 reproduction embeddings (I2V no-prefix + `deepwalk_/node2vec_/struc2vec_`); `output/<dataset>/k<K>/` = Phase-2 files (`virtual_<sim>.edgelist`, `vg_{nc|lp}_<sim>_s<seed>.emb`). No flat files, no sibling `cora_10/`-style folders.
- Code updated to match (folders lead, code follows): `runner.py` `run_nodeclass_repeated`/`run_linkpred_repeated` now write `output/<safe>/i2v_main/` (Step-5 benchmark inherits automatically); Phase-2 notebook writes `output/<dataset>/k<K>/`. Existing files moved, none regenerated (filenames unchanged ⇒ reuse intact); `core_i2vorg_embeddings/` merged into `cora/i2v_main/` (24 embs) and removed.
- **Graph-health stats (notebook):** every `build_virtual()` call upserts one row — `dataset, sim, K, nodes, edges, avg_degree, components, isolates, path` — into `results/vir_graph_stats/virtual_graph_stats.csv` (one accumulating table across datasets; key = dataset×sim×K, no duplicates). For debugging bad scores (sparse/disconnected/isolates) + ablation-quality table for the paper.

### 2026-07-02 — Renamed `results/phase2/` → `results/vir_graph_variants/` (user-requested)

- **Rename:** task-score comparison CSVs (e.g. `cora_variant_comparison_K10.csv`) moved from `results/phase2/` to `results/vir_graph_variants/`. Clearer name; sits beside `results/vir_graph_stats/` so both Phase-2 result folders are self-describing.
- **Two sibling Phase-2 folders under `results/`:** `vir_graph_stats/` = per-graph health table (one graph = one row); `vir_graph_variants/` = downstream variant task-score comparison CSVs.
- Updated every reference (folders lead, code follows): notebook code cell (`Path("results") / "vir_graph_variants"`) + its markdown/stale outputs, this file, README tree, `docs/virgo_guide.md`. No Python source referenced the old path. Directory moved with `git mv` (CSV preserved), notebook JSON re-verified valid.

### 2026-07-04 — Phase 3 spine implemented (ViRGo-SAGE)

- NEW `encoder.py` — `SageEncoder` (repo style: argparse / `build_graph()` / `main(args)`, exposes `train(epochs)`). Features X = [deg, Ω, ψ, clustering] z-normalized, computed on the **original** graph (reuses `VirtualGraph.signatures` + cached core + `nx.clustering`); message passing over the **virtual** graph (2-layer mean `SAGEConv`, 4→64→64). Loss = Skipgram-analog: positives = window-10 co-occurrence on **unweighted node2vec walks over the virtual graph** (num_walks=10, len=40, workers=1, seeded — the exact Phase-2 bridge corpus), negatives ∝ deg^0.75 with Q=5 (matches Word2Vec `negative=5`), Adam lr=0.01, 50 epochs, 100k pairs/epoch (corpus capped 2M, deterministic). CPU-only + seeded torch/np/generators → reproducible. Writes word2vec-format `.emb` (existing evals unchanged).
  - `python encoder.py --input input/cora.edgelist --sim psi --k 10 --seed 42` → `output/<ds>/k<K>/sage_<sim>_s<seed>.emb`
- `benchmark_config.py` + `GNN_PARAMS` (hidden/dims 64, layers 2, mean agg, lr 0.01, epochs 50, Q=5, pairs_per_epoch 100k, max_pairs 2M).
- NEW `notebooks/3-phase3_gnn_encoder.ipynb` (15 cells) — **reads Phase-2 artifacts only, builds no graphs** (separation of concerns per request): load saved `virtual_<sim>.edgelist` → train spine per seed → `sage_nc_<sim>_s<seed>.emb` + loss curve → verify (node count, finite, KeyedVectors-readable) → NC weighted-F1 → leakage-free LP (reuses the SAME `_vglp_` splits, train-virtual rebuilt from the 70% edges, `sage_lp_*.emb`) → summary CSV `results/vir_graph_variants/<ds>_sage_spine_K<K>.csv` + bridge table displayed alongside. `2-virtual_graph_phase2.ipynb` untouched (stays the graph builder + bridge record).
- `import encoder` verified in env `i2v` (torch 2.12.0+cu130, PyG 2.8.0). Notebook NOT executed — first SAGE numbers come when it runs (README Phase-3 step 5).

### 2026-07-06 — First Phase-3 run (enzymes) + §6 head-to-head table + rename fallout

- **First spine run (user-executed, enzymes, Ψ, K=10, seed 42):** SAGE NC F1 0.5398 vs bridge 0.5051; LP AUC 0.6609 vs 0.5121 (details → `docs/paper_log.md`). Spine verified end-to-end: loss falls, evals read the `.emb`, outputs land in `output/enzymes/k10/`.
- **Phase-3 §6 rewritten into a real head-to-head:** one table with `deepwalk_bridge / virgo_sage / delta` per task (was: two differently-shaped tables side by side). Saves a second CSV `results/vir_graph_variants/<ds>_encoder_comparison_K<K>.csv`; single-seed std now 0.0 (was NaN). Fairness note added: bridge CSV must come from a Phase-2 run with the same DATASET/SIM/SEEDS (CSV does not record seeds).
- **Notebook renamed again by user:** `2-virtual_graph_phase2.ipynb` → `2-phase_2_virtual_graph.ipynb`. Patched references in README (roadmap + tree) and Phase-3 notebook (intro + load-cell assert). Historical notes entries left as-is.
- **Files missing on disk (flagged, not restored):** `results/vir_graph_variants/enzymes_variant_comparison_K10.csv` + `enzymes_sage_spine_K10.csv` were saved by the runs (per notebook outputs) but are no longer present; `results/vir_graph_stats/virtual_graph_stats.csv` lost its 9 cora rows (only the enzymes row remains). Restore = rerun Phase-2 §5/§9 and Phase-3 §6 (instant — all edgelists/`.emb` reuse from disk).
- **§6 revised again (user feedback: one table, non-technical readable):** both encoders now scored *inside* §6 directly from their saved embeddings (`vg_*` = Phase-2 bridge, `sage_*` = Phase-3) with the shared eval scripts — no dependency on the Phase-2 comparison CSV, numbers always match the Setup knobs. Single display/CSV: `task / deepwalk_phase2 / graphsage_phase3 / improvement / winner` → `results/vir_graph_variants/<ds>_encoder_comparison_K<K>.csv`. The separate `<ds>_sage_spine_K<K>.csv` save + sage-only summary table were removed from §6 (per-seed sage numbers still printed in §4/§5).
- **Accumulating scoreboard added (user request: keep every variant's results, visualize comparisons):** new `record_score()` in `scripts/results_io.py` upserts one row per dataset × encoder × sim × K × task (with seeds, mean, std) into `results/vir_graph_variants/scoreboard.csv` — same replace-never-duplicate pattern as the graph-health table. Phase-3 §6 now records both encoders there on every run; NEW §7 "Scoreboard" cell shows the full table for the current dataset + one grouped bar chart per task (bars = sim×K variants, colors = encoders, error bars = std). Old §7 "Status + next" renumbered to §8.
- **Scoreboard columns renamed for readability (user request):** `sim` → `graph_variant`, `K` → `top_K_neighbors` — in `results_io.record_score()` (params + row keys + upsert key + sort), the existing `scoreboard.csv` header (sed, data rows untouched), and Phase-3 §7 (reader cell + markdown now includes a column-meanings line). §6 calls are positional → unchanged.
- **3-seed enzymes head-to-head (user-run):** bridge NC 0.4971 / LP 0.5110±0.0297 vs SAGE NC 0.5405±0.0014 / LP 0.6632±0.0114 → paper_log entry updated (supersedes the single-seed numbers).

### 2026-07-06 — Ablation A implemented (A1 walk vs A2 edge positives)

- `encoder.py`: `corpus()` + `train()` gained `positives=` ("walk" = A1 window co-occurrence, unchanged default; "edge" = A2 = the virtual edges themselves as positive pairs, added in both directions, no walks). Same deterministic cap, negatives, loss. CLI: `--positives {walk,edge}`; CLI output auto-tags `sage_edge_` when edge.
- `scripts/benchmark_config.py`: `GNN_PARAMS["positives"] = "walk"`.
- `notebooks/3-phase3_gnn_encoder.ipynb`: Setup gained `POSITIVES` knob + derived `TAG` (`sage` / `sage_edge` file prefix) + `ENCODER` (`graphsage` / `graphsage_edge` scoreboard name); §2/§3/§4/§5 use TAG paths + pass positives; §6 compares the selected variant vs the bridge (dynamic column name, snapshot CSV suffixed `_edge` for A2), records under ENCODER; §2 markdown explains the knob; §8 status updated. A2 runs can never overwrite A1 files, snapshots, or scoreboard rows.
- **RESULTS (user-run A2 pass, enzymes ψ K10, seeds 42/43/44) → full entry in `docs/paper_log.md`:** A1 NC 0.5405±0.0014 / LP 0.6632±0.0114; **A2 NC 0.5413±0.0011 / LP 0.6909±0.0156**; bridge 0.4971 / 0.5110. FINDING: NC tie, LP A2 +0.028 (≈2σ) — walks unnecessary once similarity is explicit in the graph. DECISION: **A2 = default objective** going forward; A1 kept/reported as the bridge-comparable config. DEVIATION: Stage-1 A ran on enzymes, design doc said cora — repeat pending. Artifacts: `output/enzymes/k10/sage_edge_{nc,lp}_psi_s4{2,3,4}.emb`, `results/vir_graph_variants/enzymes_encoder_comparison_K10_edge.csv`, scoreboard rows `graphsage_edge`.
- GAP (minor, logged for later): notebook 3 reads its own `POSITIVES` knob, not `GNN_PARAMS["positives"]` — config entry currently informational only; unify to one source of truth. Notebook intro markdown still A1-worded ("same walk corpus") — stale for A2 runs.
- TODO next (Stage 1): **B** aggregation — wire `agg` (mean vs Ψ-weighted) into `SAGEConv` (currently hardcoded `aggr='mean'`, `GNN_PARAMS["agg"]` unused); then repeat A×B on cora.
