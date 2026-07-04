"""Single source of truth: paths, dataset registry, I2V hyperparameters, reproduction defaults."""

from pathlib import Path

# Project root = one level above scripts/. Everything else is derived from it.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
SPLITS_DIR = PROJECT_ROOT / "splits"
LABELS_DIR = PROJECT_ROOT / "labels"
RESULTS_DIR = PROJECT_ROOT / "results"

# Dataset registry: name -> edgelist + label file (labels may not exist yet -> None or a path to be filled).
DATASETS = {
    "cora":     {"edgelist": INPUT_DIR / "cora.edgelist",     "labels": LABELS_DIR / "cora.labels"},
    "citeseer": {"edgelist": INPUT_DIR / "citeseer.edgelist", "labels": None},  # author's own graph (paper's file) -> link-pred only, no aligned labels
    "citeseer_linqs": {"edgelist": INPUT_DIR / "citeseer_linqs.edgelist", "labels": LABELS_DIR / "citeseer_linqs.labels"},  # aligned (graph+labels from LINQS)
    "politics": {"edgelist": INPUT_DIR / "politics.edgelist", "labels": None},  # rt-pol ships no labels -> link-pred only (no verifiable NC source)
    "enzymes":  {"edgelist": INPUT_DIR / "enzymes.edgelist",  "labels": LABELS_DIR / "enzymes.labels"},  # labels built+verified by make_labels.make_enzymes
    "enzymes_nr": {"edgelist": INPUT_DIR / "enzymes_nr.edgelist", "labels": LABELS_DIR / "enzymes_nr.labels"},  # aligned fallback if enzymes ids mismatch
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
    "temperature": 0.3,
}

# Virtual-graph study (Phase 2): variants + K sweep. SAME K across variants + SAME seeds = fair comparison.
VG_SIMS = ["psi", "degree", "centrality"]   # psi = I2V KL->Poisson Ψ; degree/centrality = simpler baselines ("which graph best?")
VG_K = [5, 10, 20]                          # top-K sweep (sparsity vs over-smoothing tradeoff)
VG_SEEDS = [42, 43, 44]                     # deterministic build; extra seeds cover the downstream walk/GNN encoder

# Reproduction defaults — fixed for every run so results are repeatable.
REPRO = {
    "seed": 42,
    "linkpred_test_frac": 0.30,    # 70:30 edge split
    "nodeclass_train_frac": 0.70,  # stratified split (paper sweeps 30-70%)
    "linkpred_op": "hadamard",     # edge operator for the logreg edge features (node2vec protocol)
    "linkpred_score": "cosine",    # main result: Hadamard edge features -> logistic regression (paper's node2vec link-pred protocol); 'cosine' = unsupervised similarity, kept as a second column
}


# Returns the registry entry for a dataset, or raises listing the valid names.
def dataset(name):
    """Look up a dataset by name."""
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Available: {list(DATASETS)}")
    return DATASETS[name]
