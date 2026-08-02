'''THE frozen Module-3 artifact: the seven-dataset FITTING panel and the two candidate rules, locked 2026-07-29.

Nothing here is recomputed. The point and interval of each rule are the exact numbers candidate_rules() produced on the
panel (results/candidate_rules.csv, link-prediction credible rows) and must NEVER be re-derived from data that includes an
unseen dataset - that would make Module 3 circular. Prediction uses the POINT midpoint (deterministic); the interval is
carried only for honest reporting, since every cut inside it fits the panel equally well.'''

from collections import namedtuple

# The datasets the rules were FITTED on. A FIXED LITERAL, never derived from STUDY: adding a dataset to the study must not
# silently join the discovery panel and overwrite the seven-dataset Module-2 result. Anything outside this set is unseen -
# it may be PREDICTED, never fitted.
DISCOVERY_PANEL = ["cora", "enzymes", "ogbn_arxiv", "ogbl_ddi", "roman_empire", "tolokers", "questions"]

# The Module-3 held-out datasets, stored SEPARATELY so they can never leak into fitting. Append more as they are ingested;
# predict_module3.py defaults to this list. A dataset must live in exactly one of the two panels, never both.
HELDOUT = ["citeseer_linqs", "proteins"]

assert not (set(DISCOVERY_PANEL) & set(HELDOUT)), "a dataset is in BOTH the discovery panel and the held-out set - it must be in exactly one"

# predictor = the graph property; op = the side that says "augment"; point = the split used for prediction;
# interval = every cut that fits the panel equally well (report this, not the 4-decimal point); needs_labels = property is n/a on an unlabelled graph.
Rule = namedtuple("Rule", "name predictor op point interval needs_labels")
FROZEN_RULES = [
    Rule("rule1", "homophily_adjusted",     "<", 0.227,  (0.0926, 0.3613), True),
    Rule("rule2", "largest_component_frac", ">", 0.9588, (0.9177, 1.0),    False),
]


def predict_one(rule, value):
    '''augment / keep original for one rule's point, or n/a when the property is missing (e.g. homophily on an unlabelled graph).'''
    if value is None or value != value:                    # NaN -> the property needs labels this graph does not have
        return "n/a"
    hit = value < rule.point if rule.op == "<" else value > rule.point
    return "augment" if hit else "keep original"


def predict(props):
    '''Per-rule verdicts for one property dict, plus the combined call: the firing rules' shared verdict, or that they disagree.'''
    calls = {r.name: predict_one(r, props.get(r.predictor)) for r in FROZEN_RULES}
    live = [v for v in calls.values() if v != "n/a"]
    combined = "no rule fires" if not live else ("rules disagree" if len(set(live)) > 1 else live[0])
    return calls, combined
