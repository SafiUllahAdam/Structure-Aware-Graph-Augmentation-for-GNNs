"""Single source of truth: paths, dataset registry, I2V hyperparameters, reproduction defaults."""

from pathlib import Path

# Project root = one level above virgo/. Everything else is derived from it.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
SPLITS_DIR = PROJECT_ROOT / "splits"
LABELS_DIR = PROJECT_ROOT / "labels"
RESULTS_DIR = PROJECT_ROOT / "results"

# Notebook-first output zones (2026-07-08 layout): every path reads notebook -> content -> dataset -> K -> variant.
NB1_DIR = OUTPUT_DIR / "notebook1_reproduce_i2v"        # Phase-1 embeddings: <dataset>/<task>/<model>_s<seed>.emb
NB2_DIR = OUTPUT_DIR / "notebook2_create_vir_graph"     # virtual_graphs/ + bridge embeddings per task folder
NB3_DIR = OUTPUT_DIR / "notebook3_gnn_encoder"          # GraphSAGE embeddings per task folder
LP_SPLITS_ORIG = SPLITS_DIR / "link_prediction" / "original_graph"       # Phase-1 splits: <dataset>/seed_<s>/
LP_SPLITS_VG = SPLITS_DIR / "link_prediction" / "virtual_graph_study"    # shared Phase-2/3 splits: <dataset>/seed_<s>/
SCOREBOARD_CSV = RESULTS_DIR / "scoreboard.csv"         # THE master table (one row per dataset x encoder x graph x K x task)
GRAPH_HEALTH_CSV = RESULTS_DIR / "graph_health.csv"     # one row per virtual graph built
SNAPSHOTS_DIR = RESULTS_DIR / "snapshots"               # per-run comparison CSVs
FIGURES_DIR = RESULTS_DIR / "figures"                   # characterization figures (notebook 5)

# Dataset registry: name -> edgelist + label file (labels may not exist yet -> None or a path to be filled).
DATASETS = {
    "cora":     {"edgelist": INPUT_DIR / "cora.edgelist",     "labels": LABELS_DIR / "cora.labels"},
    "citeseer": {"edgelist": INPUT_DIR / "citeseer.edgelist", "labels": None},  # author's own graph (paper's file) -> link-pred only, no aligned labels
    "citeseer_linqs": {"edgelist": INPUT_DIR / "citeseer_linqs.edgelist", "labels": LABELS_DIR / "citeseer_linqs.labels"},  # aligned (graph+labels from LINQS)
    "politics": {"edgelist": INPUT_DIR / "politics.edgelist", "labels": None, "directed_source": True},  # rt-pol ships no labels -> link-pred only; retweet = DIRECTED relation, loaded undirected (recorded deviation)
    "enzymes":  {"edgelist": INPUT_DIR / "enzymes.edgelist",  "labels": LABELS_DIR / "enzymes.labels"},  # labels built+verified by make_labels.make_enzymes
    "enzymes_nr": {"edgelist": INPUT_DIR / "enzymes_nr.edgelist", "labels": LABELS_DIR / "enzymes_nr.labels"},  # aligned fallback if enzymes ids mismatch
    "proteins": {"edgelist": INPUT_DIR / "proteins_nr.edgelist", "labels": LABELS_DIR / "proteins_nr.labels"},  # author input/proteins.edgelist is comma-delimited -> make_labels.make_proteins rebuilds a whitespace copy + labels from the same source (edge overlap 1.0)
    "proteins_nr": {"edgelist": INPUT_DIR / "proteins_nr.edgelist", "labels": LABELS_DIR / "proteins_nr.labels"},  # explicit alias: same rebuilt pair
    # OGB-sourced, structural-only, scored under the OFFICIAL OGB protocol (fixed split + OGB Evaluator), NOT the ViRGo random 70/30. Build via make_ogb.py.
    "ogbn_arxiv": {"edgelist": INPUT_DIR / "ogbn_arxiv.edgelist", "labels": LABELS_DIR / "ogbn_arxiv.labels",
                   "directed_source": True, "eval": "ogb", "split": SPLITS_DIR / "ogb" / "ogbn_arxiv_idx.npz"},   # full transductive graph, official time split -> Accuracy; data.x (128-dim text) ignored
    "ogbl_ddi":   {"edgelist": INPUT_DIR / "ogbl_ddi_train.edgelist", "labels": None,
                   "eval": "ogb", "pairs": SPLITS_DIR / "ogb" / "ogbl_ddi_pairs.npz"},   # graph = TRAINING edges only (no leakage); official pos/neg -> Hits@20; featureless
    # Heterophilous benchmarks (Platonov et al. 2023), structural-only, scored under the ViRGo CORE protocol (stratified 70% NC, 70:30 LP)
    # -> no "eval" key, so they route exactly like the core four. Added 2026-07-27: the panel had no low-homophily graph, so the
    # augment side of the rule was never tested. Their official train/val/test masks are ignored. Build via make_hetero.py.
    "roman_empire": {"edgelist": INPUT_DIR / "roman_empire.edgelist", "labels": LABELS_DIR / "roman_empire.labels",
                     "directed_source": True},   # word order + dependency arcs = a DIRECTED relation, symmetrized upstream by PyG; 300-dim fastText features ignored
    "tolokers":     {"edgelist": INPUT_DIR / "tolokers.edgelist", "labels": LABELS_DIR / "tolokers.labels"},   # worked-the-same-task = symmetric by construction; 10-dim worker profile ignored
    "questions":    {"edgelist": INPUT_DIR / "questions.edgelist", "labels": LABELS_DIR / "questions.labels",
                     "directed_source": True},   # answered-your-question = a DIRECTED relation, symmetrized upstream by PyG; 301-dim fastText features ignored
    # Module-3 HELD-OUT (2026-08-02): genuinely unseen datasets to validate the two frozen rules. Structural-only, ViRGo core
    # protocol (no "eval" key -> route like the core four). Build via make_pyg.py (pubmed, actor) / make_hetero.py (minesweeper).
    "pubmed":      {"edgelist": INPUT_DIR / "pubmed.edgelist",      "labels": LABELS_DIR / "pubmed.labels"},        # Planetoid citation, 3 classes; 500-dim TF-IDF ignored
    "actor":       {"edgelist": INPUT_DIR / "actor.edgelist",       "labels": LABELS_DIR / "actor.labels"},         # Geom-GCN film co-occurrence, low homophily, 5 classes; 932-dim features ignored
    "minesweeper": {"edgelist": INPUT_DIR / "minesweeper.edgelist", "labels": LABELS_DIR / "minesweeper.labels"},   # Platonov grid, adjusted homophily ~0.01, 2 classes; 7-dim features ignored
    "amazon_photo": {"edgelist": INPUT_DIR / "amazon_photo.edgelist", "labels": LABELS_DIR / "amazon_photo.labels"},  # co-purchase ("bought together"), 8 classes; 745-dim bag-of-words ignored
}
# webkb / webkb_wisc removed 2026-07-02: input edgelists deleted deliberately (recoverable from git history if ever needed).

# Cross-model benchmark scope: which datasets and methods the comparison loop sweeps (experiments/benchmark_baselines.py).
# (politics dropped: no verifiable labels; webkb_wisc dropped 2026-07-02 with its deleted input files.)
BENCH_DATASETS = ["cora", "citeseer", "enzymes"]  # citeseer = author graph, link-pred only (no aligned labels)
BENCH_MODELS = ["identity2vec", "deepwalk", "node2vec", "struc2vec"]

# Identity2Vec embedding hyperparameters (mirror experiments/train.py defaults; walk_length=40 = repo default, paper's 80 is a recorded deviation, see notes.md).
I2V_PARAMS = {
    "dimensions": 64, "walk_length": 40, "num_walks": 10,
    "window_size": 10, "epochs": 1, "sg": 1, "e": 2.7182,
}

# Virtual-graph study (Phase 2): variants + K sweep. SAME K across variants + SAME seeds = fair comparison.
VG_SIMS = ["psi", "degree", "centrality", "original", "hybrid"]   # psi = I2V Ψ; degree/centrality = simpler baselines; original = unchanged-graph control (K unused); hybrid = original ∪ psi top-K
# Density-matched controls, kept OUT of VG_SIMS so the Phase-2/3 study is unchanged; opt in with VG_SIMS + VG_CONTROLS.
# The original graph's density floats per dataset (ddi avg degree 500 vs role graphs 13; cora 3.9 vs 12) and even flips
# direction, so "role vs original" confounds edge meaning with edge count. These two hold the count fixed.
VG_CONTROLS = ["original_k", "random_k"]    # original_k = K real neighbors per node; random_k = K arbitrary nodes (null scaffold)
VG_K = [5, 10, 20]                          # top-K sweep (sparsity vs over-smoothing tradeoff)
VG_SEEDS = [42, 43, 44]                     # deterministic build; extra seeds cover the downstream walk/GNN encoder

# ViRGo-SAGE encoder (Phase 3): unsupervised GraphSAGE over the virtual graph, Skipgram-analog loss.
# Walk corpus for the positives reuses I2V_PARAMS (num_walks/walk_length/window) -> only the encoder changes vs the Phase-2 bridge.
GNN_PARAMS = {
    "hidden": 64, "dimensions": 64, "layers": 2,
    "agg": "mean",                                        # ablation B: "mean" | "weighted" (Ψ-weighted mean) | "sum" | "max"
    "lr": 0.01, "epochs": 50, "negatives": 5,             # Q=5 matches the bridge's Word2Vec negative=5
    "pairs_per_epoch": 100_000, "max_pairs": 2_000_000,   # deterministic corpus caps (runtime/memory on large graphs)
    "positives": "edge",                                  # ablation A DECIDED 2026-07-07: "edge" (A2) won LP, tied NC; "walk" = A1 bridge-comparable
    "features": "all",                                    # ablation D baseline (D0): all four structural input features
}

# Ablation D (input features): does the GNN win come from the structural features or from message passing?
# "random" (D4) is THE control — message passing with zero structural signal; compare it against the deepwalk bridge.
D_FEATURES = {
    "all":      ("D0 all",       "degree + centrality + psi + clustering"),
    "degree":   ("D1 degree",    "degree only"),
    "deg_cent": ("D2 deg+cent",  "degree + eigenvector centrality"),
    "psi":      ("D3 psi",       "psi only (confounded: the psi graph was built from it)"),
    "random":   ("D4 random",    "seeded random features - control: message passing alone"),
    "const":    ("D5 constant",  "identical rows -> z-norm zeros - floor, expect AUC ~ 0.50"),
    "none_mp":  ("D6 features only", "raw features, no message passing (layers=0) - control: features alone"),
    "centrality": ("D7 centrality", "eigenvector centrality only (per-component normalized, see GRAPH_POLICY)"),
    "clustering": ("D8 clustering", "local clustering coefficient only"),
}

# THE graph policy (defined in graph_io.py, the module that owns graph semantics) re-exported so config lives at one import.
# How every stage treats ANY dataset: self-loops, directed sources, centrality mode, signature ties, LP negatives.
from virgo.graph_io import GRAPH_POLICY, I2V_BASELINE_POLICY

# Reproduction defaults — fixed for every run so results are repeatable.
REPRO = {
    "seed": 42,
    "linkpred_test_frac": 0.30,    # 70:30 edge split
    "nodeclass_train_frac": 0.70,  # stratified split (paper sweeps 30-70%)
    "linkpred_op": "hadamard",     # edge operator, only used when linkpred_score='logreg' (node2vec protocol)
    "linkpred_score": "cosine",    # main result: unsupervised cosine similarity ranking (I2V-paper-faithful, basis of the Phase-1 repro); 'logreg' = supervised Hadamard->logistic-regression alternative, optional robustness check only
}


# Returns the registry entry for a dataset, or raises listing the valid names.
def dataset(name):
    """Look up a dataset by name."""
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Available: {list(DATASETS)}")
    return DATASETS[name]
