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
    "citeseer": {"edgelist": INPUT_DIR / "citeseer.edgelist", "labels": None},  # author graph; LINQS labels don't align (use citeseer_linqs)
    "citeseer_linqs": {"edgelist": INPUT_DIR / "citeseer_linqs.edgelist", "labels": LABELS_DIR / "citeseer_linqs.labels"},  # aligned (graph+labels from LINQS)
    "politics": {"edgelist": INPUT_DIR / "politics.edgelist", "labels": LABELS_DIR / "politics.labels"},
    "enzymes":  {"edgelist": INPUT_DIR / "enzymes.edgelist",  "labels": LABELS_DIR / "enzymes.labels"},
    "webkb":    {"edgelist": INPUT_DIR / "webkb.edgelist",    "labels": None},  # author's I2V numbering; labels unrecoverable (use webkb_wisc)
    "webkb_wisc": {"edgelist": INPUT_DIR / "webkb_wisc.edgelist", "labels": LABELS_DIR / "webkb_wisc.labels"},  # same graph, labelled (Wisconsin)
}

# Identity2Vec embedding hyperparameters (mirror train.py defaults; walk_length=40 in repo, 80 in paper = recorded deviation).
I2V_PARAMS = {
    "dimensions": 64, "walk_length": 40, "num_walks": 10,
    "window_size": 10, "epochs": 1, "sg": 1, "e": 2.7182,
}

# Reproduction defaults — fixed for every run so results are repeatable.
REPRO = {
    "seed": 42,
    "linkpred_test_frac": 0.30,    # 70:30 edge split
    "nodeclass_train_frac": 0.70,  # stratified split (paper sweeps 30-70%)
    "linkpred_op": "hadamard",     # node2vec edge operator
}


# Returns the registry entry for a dataset, or raises listing the valid names.
def dataset(name):
    """Look up a dataset by name."""
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Available: {list(DATASETS)}")
    return DATASETS[name]
