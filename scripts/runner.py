"""Orchestrate reproduction tasks by calling the existing root scripts — nothing is moved."""

import subprocess
import sys
from pathlib import Path

# Put repo root and scripts/ on the path so we can import both the root scripts and the siblings.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmark_config import OUTPUT_DIR, PROJECT_ROOT, REPRO, SPLITS_DIR, I2V_PARAMS, dataset
from utils import set_seed


# Runs Identity2Vec (train.py) as a subprocess to learn an embedding — this is the slow step.
def embed(input_path, output_path, params=None):
    """Train an I2V embedding on a given edgelist."""
    params = params or I2V_PARAMS
    cmd = [sys.executable, str(PROJECT_ROOT / "train.py"),
           "--input", str(input_path), "--output", str(output_path),
           "--dimensions", str(params["dimensions"]),
           "--walk-length", str(params["walk_length"]),
           "--num-walks", str(params["num_walks"]),
           "--window-size", str(params["window_size"]),
           "--epochs", str(params["epochs"]), "--sg", str(params["sg"])]
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    return Path(output_path)


# Link prediction: split edges 70/30, optionally retrain on the train graph, report held-out AUC.
def run_linkpred(name, emb=None, retrain=False, params=None, seed=None):
    """Return link-prediction metrics + the settings used."""
    seed = REPRO["seed"] if seed is None else seed
    set_seed(seed)
    from prepare_linkpred import prepare              # lazy: keeps --list working without sklearn
    from eval_linkpred import evaluate as linkpred_eval

    counts = prepare(dataset(name)["edgelist"], name, SPLITS_DIR, REPRO["linkpred_test_frac"], seed)
    if retrain:
        emb = OUTPUT_DIR / f"{name}_lp.emb"
        embed(SPLITS_DIR / f"{name}_train.edgelist", emb, params)
    elif emb is None:
        emb = OUTPUT_DIR / f"{name}.emb"
        print(f"  ! no --retrain: using full-graph {Path(emb).name} (LEAKAGE — plumbing check, not a paper number)")

    auc = linkpred_eval(emb, SPLITS_DIR, name, REPRO["linkpred_op"], seed)
    metrics = {"auc": auc}
    settings = {"seed": seed, "op": REPRO["linkpred_op"], "test_frac": REPRO["linkpred_test_frac"],
                "retrain": retrain, "emb": Path(emb).name, **counts}
    return metrics, settings


# Node classification: logistic regression on embeddings vs labels, report weighted F1.
def run_nodeclass(name, emb=None, seed=None):
    """Return node-classification metrics + the settings used."""
    seed = REPRO["seed"] if seed is None else seed
    set_seed(seed)
    from eval_nodeclass import evaluate as nodeclass_eval

    labels = dataset(name)["labels"]
    if labels is None or not Path(labels).exists():
        raise FileNotFoundError(
            f"No labels for '{name}' (expected {labels}). See labels/README.md — node classification is blocked.")
    emb = emb or OUTPUT_DIR / f"{name}.emb"
    f1, n_nodes, n_classes = nodeclass_eval(emb, labels, REPRO["nodeclass_train_frac"], seed)
    metrics = {"weighted_f1": f1, "n_nodes": n_nodes, "n_classes": n_classes}
    settings = {"seed": seed, "train_frac": REPRO["nodeclass_train_frac"], "emb": Path(emb).name}
    return metrics, settings


TASKS = {"linkpred": run_linkpred, "nodeclass": run_nodeclass}
