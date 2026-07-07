"""Save a benchmark run to results/ as a numbered CSV with a JSON metadata header."""

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from utils import next_run_id


# Writes one run: results/NNN.{dataset}.{task}.csv with a #META header line then metric rows.
def save_result(results_dir, dataset, task, metrics, settings):
    """Persist metrics + run settings; returns the file path."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    rid = next_run_id(results_dir)
    stamp = datetime.now().strftime("%d.%m")
    path = results_dir / f"{rid:03d}.{stamp}.{dataset}.{task}.csv"

    meta = {"run_id": rid, "dataset": dataset, "task": task,
            "timestamp": datetime.now().isoformat(), **settings}
    with open(path, "w", newline="") as f:
        f.write(f"#META:{json.dumps(meta)}\n")
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, f"{v:.4f}" if isinstance(v, float) else v])

    print(f"✓ saved {path.relative_to(results_dir.parent)}")
    return path


# Upserts one scoreboard row (key: dataset x encoder x graph_variant x top_K_neighbors x task); accumulates every variant ever scored.
def record_score(dataset, encoder, graph_variant, top_k, task, seeds, scores,
                 csv_path="results/vir_graph_variants/scoreboard.csv"):
    """Add/refresh one (dataset, encoder, graph_variant, top_K_neighbors, task) row in the accumulating scoreboard CSV."""
    row = {"dataset": dataset, "encoder": encoder, "graph_variant": graph_variant, "top_K_neighbors": top_k,
           "task": task, "seeds": "|".join(str(s) for s in seeds),
           "mean": round(float(np.mean(scores)), 4), "std": round(float(np.std(scores)), 4)}
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    board = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=list(row))
    key = ((board["dataset"] == dataset) & (board["encoder"] == encoder) & (board["graph_variant"] == graph_variant)
           & (board["top_K_neighbors"] == top_k) & (board["task"] == task))
    board = pd.concat([board[~key], pd.DataFrame([row])], ignore_index=True)
    (board.sort_values(["dataset", "task", "encoder", "graph_variant", "top_K_neighbors"])
          .reset_index(drop=True).to_csv(path, index=False))
    return path
