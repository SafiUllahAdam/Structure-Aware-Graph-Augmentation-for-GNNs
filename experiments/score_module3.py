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
from virgo import frozen_rules as fr
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
    # A cell only scores a rule when the experiment actually DECIDED: the verdict must be augment or keep original AND the
    # metric must resolve the variants. Module 2 fitted on exactly those cells (ties dropped, unusable dropped), so counting a
    # tie as a miss would invent an error the data cannot support - a tie means the 3-seed noise covers the gap, not that the rule was wrong.
    decided = m["actual_verdict"].isin(["augment", "keep original"]) & m["usable"].fillna(False)
    # Score EACH rule on its own, never the combined "rules disagree" verdict: a disagreement is not dropped, it just shows
    # which rule was right. Four states, kept apart: "pending" = not trained yet; "no decision" = tie or unusable cell;
    # None = the rule did not fire (property missing, e.g. homophily on an unlabelled graph); True/False = actually scored.
    for r in fr.FROZEN_RULES:
        m[f"{r.name}_correct"] = ["pending" if not isinstance(av, str) else "no decision" if not ok
                                  else (bool(p == av) if p in ("augment", "keep original") else None)
                                  for p, av, ok in zip(m[f"{r.name}_pred"], m["actual_verdict"], decided)]
    m["actual_verdict"] = m["actual_verdict"].fillna("pending")     # untrained held-out datasets read "pending", not NaN
    cols = (["dataset", "homophily_adjusted", "largest_component_frac", "rule1_pred", "rule1_correct",
             "rule2_pred", "rule2_correct", "predicted_verdict", "actual_verdict", "original", "best_augmented", "best_variant"])
    return m[[c for c in cols if c in m.columns]]


def main(args):
    m = score(args.datasets)
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    m.to_csv(SCORED_CSV, index=False)
    print(f"{len(m):3d} rows -> results/module3_scored.csv\n")
    print(m.to_string(index=False))
    print()
    for r in fr.FROZEN_RULES:                              # each rule reported separately, so a disagreement is never lost
        col = list(m[f"{r.name}_correct"])
        c = [v for v in col if isinstance(v, bool)]        # bool only: skip pending, no decision (tie/unusable) and did-not-fire
        skipped = [v for v in col if not isinstance(v, bool)]
        print(f"{r.name} ({r.predictor} {r.op} {r.point}): " + (f"{sum(c)}/{len(c)} correct" if c else "nothing scored yet")
              + (f"  [{len(skipped)} not scored: " + ", ".join(sorted({str(v) for v in skipped})) + "]" if skipped else ""))


def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Module 3 SCORE: compare the pre-training predictions to what training actually produced.")
    p.add_argument('--datasets', nargs='+', default=None, help='Held-out datasets to score. Default: whatever predict_module3 saved.')
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
