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
def embed(input_path, output_path, params=None, cached=True, seed=None):
    """Train an I2V embedding on a given edgelist (cached fast path + fixed seed by default)."""
    params = params or I2V_PARAMS
    seed = REPRO["seed"] if seed is None else seed
    cmd = [sys.executable, str(PROJECT_ROOT / "train.py"),
           "--input", str(input_path), "--output", str(output_path),
           "--dimensions", str(params["dimensions"]),
           "--walk-length", str(params["walk_length"]),
           "--num-walks", str(params["num_walks"]),
           "--window-size", str(params["window_size"]),
           "--epochs", str(params["epochs"]), "--sg", str(params["sg"]),
           "--seed", str(seed)]
    if cached:
        cmd.append("--cached")
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
    f1s, n_nodes, n_classes = nodeclass_eval(emb, labels, REPRO["nodeclass_train_frac"], seed)
    metrics = {"micro_f1": f1s["micro"], "macro_f1": f1s["macro"], "weighted_f1": f1s["weighted"],
               "n_nodes": n_nodes, "n_classes": n_classes}
    settings = {"seed": seed, "train_frac": REPRO["nodeclass_train_frac"], "emb": Path(emb).name}
    return metrics, settings


# Repeat node classification over seeds: a fresh full-graph embedding per seed -> per-seed metric rows.
def run_nodeclass_repeated(info, seeds=(42, 43, 44), params=None):
    """Train one embedding per seed, score node classification each time. Returns a list of per-seed rows."""
    params = params or I2V_PARAMS
    from eval_nodeclass import evaluate as nodeclass_eval
    out_dir = OUTPUT_DIR / info["safe"]                          # per-dataset subfolder
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in seeds:
        emb = out_dir / f"{info['base']}_nc_{info['version']}_s{s}.emb"
        if emb.exists():
            print(f"  [nc s{s}] reuse existing {emb.name}")     # already trained -> skip; delete file to force rebuild
        else:
            embed(info["edge_path"], emb, params, seed=s)
        f1s, n, c = nodeclass_eval(emb, info["label_path"], REPRO["nodeclass_train_frac"], s)
        rows.append({"dataset": info["base"], "version": info["version"], "task": "nodeclass", "seed": s,
                     "micro_f1": f1s["micro"], "macro_f1": f1s["macro"], "weighted_f1": f1s["weighted"]})
        print(f"  [nc s{s}] micro={f1s['micro']:.4f} macro={f1s['macro']:.4f} weighted={f1s['weighted']:.4f}")
    return rows


# Repeat link prediction over seeds: per-seed edge split + train-only embedding (no leakage) -> per-seed AUC rows.
def run_linkpred_repeated(info, seeds=(42, 43, 44), params=None):
    """Per seed: split edges, train on the 70% graph only, score AUC. Returns a list of per-seed rows."""
    params = params or I2V_PARAMS
    from prepare_linkpred import prepare
    from eval_linkpred import evaluate as linkpred_eval
    sp_dir = SPLITS_DIR / info["safe"]                          # per-dataset subfolders
    out_dir = OUTPUT_DIR / info["safe"]
    sp_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in seeds:
        name = f"{info['base']}_lp_{info['version']}_s{s}"      # split + emb filenames carry the seed
        prepare(info["edge_path"], name, sp_dir, REPRO["linkpred_test_frac"], s)   # deterministic: recreates same split cheaply
        emb = out_dir / f"{name}.emb"
        if emb.exists():
            print(f"  [lp s{s}] reuse existing {emb.name}")      # already trained -> skip; delete file to force rebuild
        else:
            embed(sp_dir / f"{name}_train.edgelist", emb, params, seed=s)
        auc = linkpred_eval(emb, sp_dir, name, REPRO["linkpred_op"], s)
        rows.append({"dataset": info["base"], "version": info["version"], "task": "linkpred", "seed": s, "auc": auc})
        print(f"  [lp s{s}] AUC={auc:.4f}")
    return rows


# Aggregate per-seed rows into a tidy summary: mean, sample-std (ddof=1), and range (delta = max-min) per metric.
def summarize_seed_results(node_rows, lp_rows):
    """Return (per_seed_df, summary_df) from the repeated-run rows."""
    import pandas as pd
    per_seed = pd.DataFrame(node_rows + lp_rows)
    recs = []
    for (ds, ver, task), g in per_seed.groupby(["dataset", "version", "task"], sort=False):
        metrics = ["micro_f1", "macro_f1", "weighted_f1"] if task == "nodeclass" else ["auc"]
        for m in metrics:
            v = g[m].dropna().to_numpy()
            recs.append({"dataset": ds, "version": ver, "task": task, "metric": m,
                         "mean": v.mean(), "std": v.std(ddof=1), "delta": v.max() - v.min(), "n": len(v)})
    return per_seed, pd.DataFrame(recs)


TASKS = {"linkpred": run_linkpred, "nodeclass": run_nodeclass}
