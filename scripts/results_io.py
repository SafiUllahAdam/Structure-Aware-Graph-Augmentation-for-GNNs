"""Save a benchmark run to results/ as a numbered CSV with a JSON metadata header (CoBench style)."""

import csv
import json
from datetime import datetime
from pathlib import Path

from utils import next_run_id


# Writes one run: results/NNN.{dataset}.{task}.csv with a #META header line then metric rows.
def save_result(results_dir, dataset, task, metrics, settings):
    """Persist metrics + run settings; returns the file path."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    rid = next_run_id(results_dir)
    path = results_dir / f"{rid:03d}.{dataset}.{task}.csv"

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
