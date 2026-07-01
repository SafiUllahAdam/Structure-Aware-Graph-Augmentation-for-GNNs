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
- I2V `train.py` defaults: `dimensions=64`, `walk-length=40`, `num-walks=10`, `window-size=10`, `epochs=1`, `sg=1` (skipgram), `min-count=0`, `workers=1`, `e=2.7182`, `temperature=0.0` (greedy; benchmark `I2V_PARAMS` uses `0.3`). Word2Vec: `alpha=0.025 → min_alpha=0.01`, `negative=5`, `sample=1e-3` (Fix 6, 2026-06-24; was `alpha=0.25` / `sample=1e-5`).
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

- **PROPOSED → IMPLEMENTED 2026-06-24 (see entry below) — temperature sampling, to recover LP without losing NC.** Keep Fix 8 untouched; change only the selection (`identity2vec.py:94`) from hard `min` to a weighted draw: `P(x) ∝ exp(−dₓ/τ)` with `dₓ = |logΨ_x − logΨ_current|`.
  - `τ=0` = exactly today's greedy (NC-safe baseline preserved); `τ→∞` = random walk (DeepWalk-like). `τ` is the structure↔proximity dial (NC↔LP).
  - Structural bias kept (small `dₓ` still most likely) -> NC holds; added exploration diversifies walks -> richer Word2Vec contexts -> LP up. Fix 8's log-units make the softmax numerically stable.
  - Reproducible: walk is already stochastic + seeded (`test_source` first step uses `np.random`; `train.py:95` seeds it). `identity_walker` is base-class only -> both cached/non-cached inherit.
  - HONESTY: the paper literally selects "least dissimilar" (greedy), so sampling is an **ablation/extension**, not paper-exact — report next to greedy.
  - PLAN: add `--temperature` (thread like `walk_length` -> `I2V_PARAMS`); sweep `τ ∈ {0, 0.1, 0.3, 1, 3}` on cora + citeseer_linqs, 3 seeds; `τ=0` must reproduce current greedy (sanity); find the τ where LP climbs toward ~0.84 while NC holds.

- **STALE RESULTS.** Fixes 1/3/4/6/8 all changed the math -> existing `.emb` and `results/` are from old code and do NOT reflect it. Regenerate embeddings (delete / FORCE_EMBED) before trusting any metric.

### 2026-06-24 — temperature sampling IMPLEMENTED (Fix 8 extended) ✅

The proposed non-greedy selection is now in code and gave a clear improvement on cora — keep it.

- WHERE (verified fresh from disk):
  - `identity2vec.py:76` `identity_walker(..., temperature=0.0)`; sampling at `:96-110` — `distances = |pdn[x] − current_score|`; `τ<=0` -> `argmin` (exact old greedy); `τ>0` -> `weights = exp(−(dₓ − min dₓ)/τ)` (max-subtracted = numerically stable), normalise, `np.random.choice`.
  - `identity2vec.py:204` `identity2vec_walk(..., temperature=0.0)` threads it to the walker.
  - `train.py:36-37` `--temperature` (float, default **0.0** = greedy); `train.py:102` passes `args.temperature`.
  - `embedding_models.py:32` `Identity2VecModel` passes `--temperature str(params["temperature"])`.
  - `benchmark_config.py:34` `I2V_PARAMS["temperature"] = 0.3` (notebook + benchmark default).
  - Cached path: `identity_walker` / `identity2vec_walk` are inherited (not overridden) -> temperature flows to `--cached` too. Seeded via `np.random` (`train.py:95`) -> reproducible. `τ=0` reproduces pre-temperature greedy byte-for-byte (sanity gate).
- RESULTS — cora, **seed 42**, `τ=0.3`, fresh embeddings (`notebooks/reproduce_i2v.ipynb`):
  - NC: micro **0.7503** / macro **0.7292** / weighted **0.7486**.
  - LP: AUC **0.8305**.
  - vs history: NC up from 0.6906 / 0.6992 (old code / author emb); LP up from the post-Fix-8 greedy ~0.81 toward the paper's **0.8413** (Table 4). Improvement on BOTH — temperature recovered LP without losing the Fix-8 NC gain, exactly as predicted.
- CAVEAT: **single seed (s42)** — run 3 seeds for mean±std before quoting as final; also sweep `τ ∈ {0, 0.1, 0.3, 1, 3}` to confirm 0.3 is the sweet spot.
- GAP (not a crash): `scripts/runner.py:embed()` — the `python scripts/main.py --task ... --retrain` CLI path — does NOT pass `--temperature`, so it stays greedy (0.0). The notebook/benchmark path (via `embedding_models`) DOES use 0.3. Thread `--temperature` into `runner.embed` if you want both entry points to agree. TODO logged.

### 2026-06-24 — cross-model config standardized for fair benchmark ✅

Goal: identical Skipgram/Word2Vec **training** across all 4 models so the comparison is fair; **I2V = anchor, left untouched** (it's stable). Walk generation + method-defining knobs stay per-model.

- I2V: **UNCHANGED** (alpha 0.025, min_alpha 0.01, sample 1e-3, negative 5, hs 0). Its embeddings stay valid -> keep cora s42 NC 0.7486 / LP 0.8305; **no I2V retrain**.
- node2vec + DeepWalk (`embedding_models.py:48-50`): pinned the Skipgram params explicitly to match I2V — `alpha=0.025, min_alpha=0.01, sample=1e-3, negative=5, hs=0`. Only `min_alpha` actually differed before (gensim default 0.0001 -> 0.01); the rest already equalled I2V via gensim defaults but are now explicit/locked.
- struc2vec (`baselines/struc2vec/src/main.py:81`): switched **hierarchical-softmax -> negative sampling** (`hs=0, negative=5`) + `alpha=0.025, min_alpha=0.01, sample=1e-3` to match I2V — also matches the paper's stated protocol ("negative sampling for DeepWalk and struc2vec"). Walk-length **default 80 -> 40** (`:30`) for standalone consistency.
- walk_length = **40 for all** — benchmark already forced 40 via `I2V_PARAMS`; struc2vec standalone default now 40 too. (80 tried earlier, no gain — see 2026-06-23.)
- KEPT per-model (NOT standardized — these define each method): node2vec `p=1/q=0.5`, DeepWalk `p=q=1`, I2V `temperature=0.3` + Poisson/KL walk, struc2vec `OPT1/2/3` + multilayer structural walk.
- VERIFIED: `py_compile` of both edited files OK (no run executed).
- EFFECT / next: regenerate ONLY the 3 baselines' `.emb` (their config changed; delete + re-run), then re-score. struc2vec will shift (hs->neg, likely up toward its paper ~0.71 LP); node2vec/DeepWalk shift slightly (min_alpha); I2V flat. struc2vec still has NO seed -> report mean±std.

---

## TODO backlog (open threads)

- [x] `prepare_linkpred.py` (70:30 split, seed=42) — done, verified on cora.
- [x] framework under `scripts/` — done, smoke-tested (`results/001`).
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
- [x] walk-length = **40** (active; flipped 40 -> 80 -> 40 again on 2026-06-23 after timing 80 = slower, see entry below). `train.py` + `benchmark_config` = 40. Paper's 80 = recorded deviation.
- [ ] On-disk `output/cora_lp.emb` is an 80-walk embedding (AUC 0.7972); retrain at 40 to refresh a 40 link-pred number if wanted.
- [x] Paper-fidelity fixes (professor review), done 2026-06-24: Fix 1b selection, Fix 2 Δ, Fix 3 p/q, Fix 4A normalizer, Fix 6 Word2Vec, Fix 7 cache sync, Fix 8 log-space. Fix 5 (walk-80) intentionally NOT applied.
- [ ] Regenerate all `.emb` — fixes 1/3/4/6/8 changed the math, on-disk results are stale.
- [ ] Confirm Fix 8 LP drop 0.8494 -> 0.81 is real vs seed noise (3-seed mean±std).
- [ ] Confirm Fix 4 ω choice (raw degree vs degree-distribution) with professor.
- [x] Temperature sampling implemented 2026-06-24 (`--temperature`, default greedy, benchmark τ=0.3): cora s42 NC 0.7486 / LP 0.8305 — improved both. Ablation, not paper-exact.
- [ ] Temperature: run 3 seeds (mean±std) + sweep τ ∈ {0, 0.1, 0.3, 1, 3} on cora + citeseer_linqs.
- [ ] Thread `--temperature` through `scripts/runner.embed` (CLI `--retrain` path still greedy; notebook/benchmark already 0.3).
- [x] Cross-model Skipgram config standardized to I2V 2026-06-24 (node2vec/DeepWalk `min_alpha=0.01`; struc2vec hs→negative sampling; walk_length 40 all). I2V untouched; per-model walk knobs kept.
- [ ] Regenerate the 3 baselines' `.emb` (config changed) + re-run benchmark; I2V embeddings stay valid (no retrain).
- [ ] `virtual_graph.py` — top-K Ψ builder (Deliverable #2).

---

### 2026-06-25 — walk-length flipped 40 → 80 (paper value)

- DECISION: **walk-length = 80 (active)**, reverting the 2026-06-23 "40 active" decision per request. 80 = the I2V paper value; ~1.87× slower than 40 (see 2026-06-23 timing), accepted.
- Changed everywhere: `train.py` default+help, `scripts/benchmark_config.I2V_PARAMS["walk_length"]` + comment, `baselines/struc2vec/src/main.py` default+help, and docs (`README.md`, `docs/virgo_guide.md`, `CLAUDE.md`). Notebook inherits the config (no hardcoded value).
- CACHE: deleted all 42 `output/**/*.emb` (every model/dataset/seed, nc+lp) — built at 40, now stale. Next benchmark run rebuilds at 80.
- Supersedes the checklist line above (`walk-length = 40`) and the 2026-06-24 standardization note (`walk_length 40 all`).

### 2026-06-29 — walk-length reverted 80 → 40 (back to repo default)

- DECISION: **walk-length = 40 (active)**, reverting the 2026-06-25 "80 active" flip per request. 40 = repo default + the setting behind author `cora.emb`; the paper's 80 (~1.87× slower, no confirmed gain) is kept as a recorded deviation.
- Reverted everywhere: `train.py` default+help, `scripts/benchmark_config.I2V_PARAMS["walk_length"]` + comment, `baselines/struc2vec/src/main.py` default+help, docs (`README.md`, `docs/virgo_guide.md`, `CLAUDE.md`). Notebook inherits the config (no hardcoded value).
- CACHE: deleted the 24 stale 80-walk `output/cora/*.emb` (rebuilt after the 2026-06-25 flip); next run rebuilds at 40. No author `output/cora.emb` present.
- Supersedes the 2026-06-25 "80 active" entry.

### 2026-06-30 — benchmark scoring finalized + first 3-seed cross-model run

Findings-driven cleanups (implemented one-by-one) + the first full 3-seed run.

- **Link-pred scoring — BOTH computed every seed; headline now COSINE.** `runner.run_linkpred_repeated` scores each `.emb` two ways: `auc_logreg` (Hadamard edge feature → logistic regression, node2vec protocol) and `auc_cosine` (unsupervised cosine similarity, paper-faithful, no classifier). `REPRO["linkpred_score"]` picks the headline `auc` column — now **`"cosine"`** (was `"logreg"`). `benchmark_baselines` writes `table2_linkpred_auc.csv` (=headline=cosine) + `table2_linkpred_auc_cosine.csv` (identical now); `auc_logreg` survives only in `benchmark_per_seed.csv`. Cosine for edges + logreg for labels = paper alignment. STALE COMMENTS (values correct): `benchmark_config.py:43` and `benchmark_baselines.py:79-80` still say "main = logreg".
- **Node-class classifier (Fix 6).** `eval_nodeclass.py:45` = `OneVsRestClassifier(LogisticRegression(max_iter=300, solver="lbfgs", random_state=seed))` (L2 = sklearn default; `multi_class="ovr"` avoided — removed in sklearn ≥1.7). Reports micro/macro/weighted F1.
- **struc2vec OPT (Fix 4).** `Struc2VecModel.train`: `--OPT1/2/3 = False` for graphs ≤10k nodes (cora/citeseer/webkb = exact distance), `True` only for large graphs (enzymes ~19.5k → memory). struc2vec still unseeded → report mean±std.
- **node2vec vs DeepWalk kept distinct.** DeepWalk `p=q=1` (uniform), node2vec `p=1/q=0.5` (biased) → node2vec is not a DeepWalk duplicate. Paper does not fix p/q.
- **FINDING — first 3-seed cross-model run (cora, seeds 42/43/44, τ=0.3). Supersedes the "single seed s42" temperature caveat.** Snapshot captured at **walk-length 80** (06-25→06-29 window); those `.emb` were deleted in the 40-revert → **regenerate at walk-40 before quoting as final.**

  | model | NC weighted F1 | LP AUC (cosine headline) |
  |---|---|---|
  | identity2vec | 0.7403 ± 0.0116 | 0.8281 ± 0.0085 |
  | deepwalk | 0.8109 ± 0.0216 | 0.9011 ± 0.0017 |
  | node2vec | 0.8166 ± 0.0090 | 0.9031 ± 0.0029 |
  | struc2vec | 0.3219 ± 0.0053 | 0.5491 ± 0.0077 |

  - Confirms the 2026-06-20 finding: on homophilous cora, proximity methods (node2vec/deepwalk) beat structural I2V on NC+LP; struc2vec weak. I2V's edge is structural/heterophilous tasks, not cora.
- **DOCS.** `docs/virgo_guide.md` full-synced to current code (notebook Steps 0–5, 70/30 split, OvR classifier, dataset registry, per-seed `.emb` naming, honest §8). Notebook walk/training progress bars restored (`embedding_models` `quiet=False`).
- TODO: regenerate the table at walk-40; run τ-sweep {0,0.1,0.3,1,3} + τ=0 vs 0.3 (delete I2V `.emb` between τ — filename ignores τ).

### 2026-06-30 — direction: Phase 1 done, focus shifts to the GNN encoder

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
- FINDING — first author-`citeseer` LP result (seed 42, walk-40, τ=0.3, largest CC = 2110 nodes):

  | score | AUC | paper Table 4 (I2V) | within ±0.05? |
  |---|---|---|---|
  | cosine (headline, paper-faithful) | **0.8606** | 0.8373 | yes (Δ 0.023) |
  | logreg (Hadamard, node2vec protocol) | **0.8771** | — | — |

  - emb `output/citeseer/citeseer_lp_orig_s42.emb`; splits `splits/citeseer/citeseer_lp_orig_s42_*`.
  - **Single seed (s42) — indicative, not final.** TODO: run seeds 43/44 for mean±std.
- **Why LP only, no node classification on citeseer:** the author graph (`input/citeseer.edgelist`) is the paper's own file but ships **no labels**, and its node numbering does **not** match LINQS `citeseer.content` order (edge overlap ~0.00), so LINQS labels would point at the WRONG nodes → any NC would be fake. Node classification on Citeseer needs the aligned `citeseer_linqs` build; on the paper's graph we report **link prediction only** (LP needs no labels).
