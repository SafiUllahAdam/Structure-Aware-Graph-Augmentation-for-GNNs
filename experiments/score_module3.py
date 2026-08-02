'''Module 3, step 2 - SCORE. After the pipeline has trained on the held-out datasets, join the predictions saved BEFORE
training to the keep/augment verdict the training actually produced, and report pred-vs-actual.

Never fits: reads the frozen scoreboard through gaps(frozen_inputs(...)) and never calls threshold() or candidate_rules().
Both frozen rules are link-prediction rules, so the pre-registered verdict is scored against the LP cell.'''

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root -> `import virgo` works from any cwd

from virgo import config as cfg
from experiments.characterize import frozen_inputs, gaps
from experiments import predict_module3 as pm

SCORED_CSV = cfg.RESULTS_DIR / "module3_scored.csv"


def actual_lp_verdicts(datasets):
    '''The keep/augment verdict the trained pipeline produced for each held-out LINK-PREDICTION cell (from the frozen scoreboard).'''
    g = gaps(frozen_inputs(datasets))
    lp = g[g["task_family"] == "link prediction"]
    return lp[["dataset", "metric", "original", "best_augmented", "best_variant", "gap_rel", "gap_sigma", "verdict", "usable"]]


def score(datasets=None):
    '''Join the pre-training predictions to the actual held-out LP verdicts; returns the pred-vs-actual table. Never fits.'''
    assert pm.PRED_CSV.exists(), f"{pm.PRED_CSV} missing - run experiments/predict_module3.py BEFORE training first."
    pred = pd.read_csv(pm.PRED_CSV)
    dsets = datasets or pred["dataset"].tolist()
    act = actual_lp_verdicts(dsets).rename(columns={"verdict": "actual_verdict"})
    m = pred.merge(act, on="dataset", how="left")
    # correct only where both a firing prediction and an actual verdict exist; blank when a rule never fired or the cell is missing/unusable.
    m["correct"] = [bool(pv == av) if (pv in ("augment", "keep original") and isinstance(av, str)) else None
                    for pv, av in zip(m["predicted_verdict"], m["actual_verdict"])]
    cols = (["dataset", "homophily_adjusted", "largest_component_frac", "rule1_pred", "rule2_pred",
             "predicted_verdict", "actual_verdict", "original", "best_augmented", "best_variant", "correct"])
    return m[[c for c in cols if c in m.columns]]


def main(args):
    m = score(args.datasets)
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    m.to_csv(SCORED_CSV, index=False)
    print(f"{len(m):3d} rows -> results/module3_scored.csv\n")
    print(m.to_string(index=False))
    scored = m["correct"].dropna()
    if len(scored):
        print(f"\nPre-registered link-prediction accuracy: {int(scored.sum())}/{len(scored)} correct.")


def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Module 3 SCORE: compare the pre-training predictions to what training actually produced.")
    p.add_argument('--datasets', nargs='+', default=None, help='Held-out datasets to score. Default: whatever predict_module3 saved.')
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
