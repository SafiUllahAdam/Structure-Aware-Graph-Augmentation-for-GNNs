'''Module 5 - PREDICT and SCORE the two-gate rule on a third, genuinely unseen set.

Same contract as Module 3, one level up: the pre-registered verdict is written BEFORE any encoder runs, and this script
never fits. It imports virgo.frozen_rules and never calls threshold(), candidate_rules() or gate_rules.screen(), so an
unseen dataset can only be SCORED against the locked gate, never move it.

One ordering difference from Module 3, and it matters: the gate is a property of the VIRTUAL graph, so the rewiring must
already be BUILT for the dataset before predicting. Building is not training - no encoder, no split, no labels are
touched - so the prediction is still made with the outcome unknown. Order: build the virtual graph -> predict -> train -> score.'''

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root -> `import virgo` works from any cwd

from virgo import config as cfg
from virgo import frozen_rules as fr
from experiments.gate_rules import CANONICAL_SIM, vg_measure, vg_path
from experiments.predict_module3 import rule_properties
from experiments.score_module3 import actual_lp_verdicts

PRED_CSV = cfg.RESULTS_DIR / "module5_predictions.csv"
SCORED_CSV = cfg.RESULTS_DIR / "module5_scored.csv"

# What gate_properties() can produce. Guards against a future gate needing a property this measurer does not compute.
PRODUCES = {"homophily_adjusted", "largest_component_frac", "original_retention"}
assert {r.predictor for r in fr.FROZEN_RULES} | {fr.FROZEN_GATE.predictor} <= PRODUCES, \
    "a frozen rule or gate needs a property predict_gate does not measure"


def gate_properties(ds, k):
    '''The three frozen inputs for one dataset: the two rule properties from the original graph, the gate's from the built virtual graph.'''
    p = rule_properties(ds)
    assert vg_path(ds, CANONICAL_SIM, k).exists(), \
        (f"{vg_path(ds, CANONICAL_SIM, k)} missing -> build the virtual graph for {ds} first "
         f"(python -m virgo.virtual_graph --input {cfg.dataset(ds)['edgelist']} --sim {CANONICAL_SIM} --k {k}). "
         "Building is not training, so the prediction is still made before the outcome is known.")
    p["original_retention"] = vg_measure(ds, [CANONICAL_SIM], k)[0]["original_retention"]
    return p


def predictions(datasets, k):
    '''One row per unseen dataset: the three properties, each rule's call, the gate's call, and which of them decided.'''
    rows = []
    for ds in datasets:
        p = gate_properties(ds, k)
        calls, gate, combined, decided_by = fr.predict_gated(p)
        rows.append({"dataset": ds, "k": k, "sim": CANONICAL_SIM,
                     "homophily_adjusted": p["homophily_adjusted"], "largest_component_frac": p["largest_component_frac"],
                     "original_retention": p["original_retention"],
                     "rule1_pred": calls["rule1"], "rule2_pred": calls["rule2"], "gate_pred": gate,
                     "gate_interval": f"({fr.FROZEN_GATE.interval[0]}, {fr.FROZEN_GATE.interval[1]})",
                     "decided_by": decided_by, "predicted_verdict": combined})
    return pd.DataFrame(rows)


def freeze(datasets, k):
    '''Write-once pre-registration: already-saved rows are kept EXACTLY as frozen, only unseen datasets are appended.'''
    df = predictions(datasets, k)
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if PRED_CSV.exists():
        old = pd.read_csv(PRED_CSV)
        new = df[~df["dataset"].isin(old["dataset"])]
        out = pd.concat([old, new], ignore_index=True)
    else:
        new = out = df
    out.to_csv(PRED_CSV, index=False)
    return out, new["dataset"].tolist()


def score(datasets=None):
    '''Join the pre-training predictions to the LP verdict training actually produced. Never fits.'''
    assert PRED_CSV.exists(), f"{PRED_CSV} missing - run --step predict BEFORE training first."
    pred = pd.read_csv(PRED_CSV)
    m = pred.merge(actual_lp_verdicts(datasets or pred["dataset"].tolist()).rename(columns={"verdict": "actual_verdict"}),
                   on="dataset", how="left")
    # A cell scores only when the experiment actually DECIDED: a tie means the 3-seed noise covers the gap, not that the
    # call was wrong. Four states kept apart: "pending" = untrained; "no decision" = tie or unusable; None = did not fire.
    decided = m["actual_verdict"].isin(["augment", "keep original"]) & m["usable"].fillna(False)
    for col, out in [("rule1_pred", "rule1_correct"), ("rule2_pred", "rule2_correct"),
                     ("gate_pred", "gate_correct"), ("predicted_verdict", "combined_correct")]:
        m[out] = ["pending" if not isinstance(av, str) else "no decision" if not ok
                  else (bool(p == av) if p in ("augment", "keep original") else None)
                  for p, av, ok in zip(m[col], m["actual_verdict"], decided)]
    m["actual_verdict"] = m["actual_verdict"].fillna("pending")
    return m[["dataset", "homophily_adjusted", "original_retention", "rule1_pred", "gate_pred", "decided_by",
              "predicted_verdict", "actual_verdict", "rule1_correct", "gate_correct", "combined_correct",
              "original", "best_augmented", "best_variant", "gap_sigma"]]


# Guards the test: predicting a dataset the gate was FITTED on is not a test, it is the training accuracy.
def _assert_unseen(datasets, allow_panel):
    '''Raise when a dataset the gate was fitted on is passed as a test case.'''
    seen = sorted(set(datasets) & set(fr.GATE_PANEL))
    assert allow_panel or not seen, (f"{seen} are in the Module-4 GATE_PANEL, so the gate was FITTED on them - predicting "
                                     "them measures fit, not transfer. Pass --allow-panel for a sanity re-check only.")


def main(args):
    datasets = args.datasets or fr.GATE_HELDOUT
    assert datasets, ("no datasets to predict - pass --datasets, or add the third-set datasets to frozen_rules.GATE_HELDOUT. "
                      "Every registered graph is either in GATE_PANEL (fitted) or genuinely unseen.")
    _assert_unseen(datasets, args.allow_panel)
    if args.step in ("predict", "all"):
        out, added = freeze(datasets, args.k)
        print(f"{len(out):3d} rows -> results/module5_predictions.csv  (froze {len(added)} new: {added or 'none'}; existing rows kept untouched)\n")
        print(out.to_string(index=False))
    if args.step in ("score", "all"):
        m = score(datasets)
        m.to_csv(SCORED_CSV, index=False)
        print(f"\n{len(m):3d} rows -> results/module5_scored.csv\n")
        print(m.to_string(index=False))
        for name, col in [("rule 1 alone", "rule1_correct"), (f"the gate ({fr.FROZEN_GATE.predictor} "
                          f"{fr.FROZEN_GATE.op} {fr.FROZEN_GATE.point})", "gate_correct"), ("the two-gate call", "combined_correct")]:
            c = [v for v in m[col] if isinstance(v, bool)]     # bool only: skip pending, no decision and did-not-fire
            skipped = [v for v in m[col] if not isinstance(v, bool)]
            print(f"{name}: " + (f"{sum(c)}/{len(c)} correct" if c else "nothing scored yet")
                  + (f"  [{len(skipped)} not scored: " + ", ".join(sorted({str(v) for v in skipped})) + "]" if skipped else ""))
        # The gate only speaks inside the low-homophily zone; outside it the combined call is rule 1's veto, unchanged.
        n_gate = int((m["decided_by"] == "gate").sum())
        print(f"\n  the gate actually decided {n_gate}/{len(m)} cells; the rest were rule 1's veto or the no-labels fallback.")


def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Module 5: pre-register the two-gate prediction on unseen datasets, then score it after training.")
    p.add_argument('--datasets', nargs='+', default=None, help='Unseen datasets (registered in cfg, NOT in GATE_PANEL). Default: frozen_rules.GATE_HELDOUT.')
    p.add_argument('--k', type=int, default=10, help='Top-K the virtual graph was built with. Default 10 (the locked setting).')
    p.add_argument('--step', default='predict', choices=['predict', 'score', 'all'],
                   help="'predict' = freeze the verdict BEFORE training (the default, and the only step to run first); 'score' = after training.")
    p.add_argument('--allow-panel', action='store_true', help='Permit datasets the gate was fitted on (a sanity re-check only, never a real test).')
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
