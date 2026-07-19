"""Single source of truth: paths, dataset registry, I2V hyperparameters, reproduction defaults."""

import sys
from pathlib import Path

# Project root = one level above scripts/. Everything else is derived from it.
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
}
# webkb / webkb_wisc removed 2026-07-02: input edgelists deleted deliberately (recoverable from git history if ever needed).

# Cross-model benchmark scope: which datasets and methods the comparison loop sweeps (benchmark_baselines.py).
# (politics dropped: no verifiable labels; webkb_wisc dropped 2026-07-02 with its deleted input files.)
BENCH_DATASETS = ["cora", "citeseer", "enzymes"]  # citeseer = author graph, link-pred only (no aligned labels)
BENCH_MODELS = ["identity2vec", "deepwalk", "node2vec", "struc2vec"]

# Identity2Vec embedding hyperparameters (mirror train.py defaults; walk_length=40 = repo default, paper's 80 is a recorded deviation, see notes.md).
I2V_PARAMS = {
    "dimensions": 64, "walk_length": 40, "num_walks": 10,
    "window_size": 10, "epochs": 1, "sg": 1, "e": 2.7182,
}

# Virtual-graph study (Phase 2): variants + K sweep. SAME K across variants + SAME seeds = fair comparison.
VG_SIMS = ["psi", "degree", "centrality", "original", "hybrid"]   # psi = I2V Ψ; degree/centrality = simpler baselines; original = unchanged-graph control (K unused); hybrid = original ∪ psi top-K
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
}

# THE graph policy (defined in graph_io.py, the module that owns graph semantics) re-exported so config lives at one import.
# How every stage treats ANY dataset: self-loops, directed sources, centrality mode, signature ties, LP negatives.
sys.path.insert(0, str(PROJECT_ROOT))
from graph_io import GRAPH_POLICY, I2V_BASELINE_POLICY   # noqa: E402  (path must be set first)

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
