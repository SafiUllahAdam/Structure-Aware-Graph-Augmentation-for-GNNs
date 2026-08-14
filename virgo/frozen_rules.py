'''THE frozen rule artifacts: the Module-2 fitting panel and its two rules (locked 2026-07-29), and the Module-4 SECOND
GATE fitted on top of them (locked 2026-08-05).

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
HELDOUT = ["citeseer_linqs", "proteins", "pubmed", "actor", "minesweeper", "amazon_photo", "lastfm_asia", "amazon_ratings", "squirrel_filtered"]

assert not (set(DISCOVERY_PANEL) & set(HELDOUT)), "a dataset is in BOTH the discovery panel and the held-out set - it must be in exactly one"

# predictor = the graph property; op = the side that says "augment"; point = the split used for prediction;
# interval = every cut that fits the panel equally well (report this, not the 4-decimal point); needs_labels = property is n/a on an unlabelled graph.
Rule = namedtuple("Rule", "name predictor op point interval needs_labels")
FROZEN_RULES = [
    Rule("rule1", "homophily_adjusted",     "<", 0.227,  (0.0926, 0.3613), True),
    Rule("rule2", "largest_component_frac", ">", 0.9588, (0.9177, 1.0),    False),
]


# The lead rule, frozen with the rules themselves: Module 2 designated rule 1 the primary (rho -1.00 on the bias-free gap,
# graded across 5 distinct values, vs rule 2's two-group split). It is what breaks a disagreement - a single call, not two.
LEAD = FROZEN_RULES[0]


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


# --- Module 4: the SECOND GATE, locked 2026-08-05 --------------------------------------------------------------------
# Module 3 showed rule 1 fails ASYMMETRICALLY: above the boundary it called "keep original" and was right 3/3; below it
# called "augment" and was right only 2/4. So rule 1 is a VETO, and a second variable is needed on one side only - inside
# the low-homophily zone. This gate is that variable: a property of the VIRTUAL graph, so it needs no labels, and the
# rewiring must already be BUILT (never trained) to measure it.
# Numbers are the exact low-homophily-zone / gap_rel row of results/gate_candidates.csv - never re-derive them from data
# that includes the graph being predicted.
Gate = namedtuple("Gate", "name predictor op point interval needs_labels applies_when fitted_on")
FROZEN_GATE = Gate("gate", "original_retention", "<", 0.0119, (0.0062, 0.0176), False,
                   "rule1 == augment", "GATE_PANEL")

# The Module-4 FITTING panel: the Module-2 discovery panel (minus ogbn_arxiv, which is node-classification only) PLUS the
# nine Module-3 datasets, whose verdicts are already published and so are no longer unseen. Declared, because the gate is
# FITTED here: a dataset in this list can never test the gate, only a genuinely unseen third set can.
GATE_PANEL = ["cora", "enzymes", "ogbl_ddi", "roman_empire", "tolokers", "questions",
              "pubmed", "actor", "minesweeper", "amazon_photo", "lastfm_asia", "amazon_ratings", "squirrel_filtered"]

# Module-5 held-out: the third, genuinely unseen set the GATE is tested on. Append datasets as they are ingested.
# The LINKX Facebook100 trio (2026-08-14) is measured here FIRST, as stage 1: only the graphs this combined call sends to
# "augment" go on to test the strategy rule, since that rule is conditional on augmentation being indicated.
GATE_HELDOUT = ["reed98", "amherst41", "johnshopkins55", "cornell5"]

assert not (set(GATE_PANEL) & set(GATE_HELDOUT)), "a dataset is in BOTH the gate panel and the gate held-out set - it must be in exactly one"


# --- Module 6: the STRATEGY rule, locked 2026-08-13 -------------------------------------------------------------------
# Modules 2-4 answer WHETHER to augment. This answers WHICH structural signal, and only partly: it separates centrality
# from the rest, and says nothing about psi vs degree below the cut. It is CONDITIONAL - it may only be consulted once
# augmentation is already indicated, because over all 14 panel datasets the same split takes 4 exceptions.
# Status, which must travel with it: FITTED, not validated. Nothing was held out - every labelled dataset is in the panel.
#   in-sample  0 exceptions over the 7 augmenting datasets, rho 0.866, graded over 0.0123 -> 0.8498 (not a two-group split)
#   out-of-sample  leave-one-out 5/7 against a 4/7 majority baseline - ONE call better than guessing
#   band-dependent  it exists under the sem tie band only; under the Module-2 sigma band the ties re-inflate and it takes
#                   2 exceptions. The band choice is a methodological decision, so it is part of the rule, not context.
#   co-predictor  degree_assortativity (exploratory tier) produces the identical split at rho 0.866; the two correlate at
#                 0.71, so which one is THE predictor is not determined by this data.
#   NOT homophily  rho 0.32 against homophily_adjusted. roman_empire is the proof: lowest adjusted homophily in the panel
#                 (-0.0468) yet centrality wins. This is the same property Module 2 screened and DROPPED for the augment
#                 question - a different question, with a different answer.
Strategy = namedtuple("Strategy", "name signal predictor op point interval needs_labels applies_when fitted_on")
FROZEN_STRATEGY = Strategy("strategy1", "centrality", "nbr_predictability_adjusted", ">", 0.0092, (0.006, 0.0123), True,
                           "augmentation is already indicated", "STRATEGY_PANEL")

# The Module-6 FITTING panel: every labelled dataset with a link-prediction score, at 10 seeds. ogbn_arxiv is
# node-classification only and ogbl_ddi is unlabelled, so neither can carry a signal label.
STRATEGY_PANEL = ["cora", "enzymes", "roman_empire", "tolokers", "questions", "squirrel_filtered", "amazon_ratings",
                  "amazon_photo", "lastfm_asia", "pubmed", "actor", "minesweeper", "citeseer_linqs", "proteins"]

# Module-7 held-out: the genuinely new datasets the strategy rule is tested on. Append as they are ingested.
# 2026-08-14, the LINKX Facebook100 networks that PASSED stage 1 - the combined augment call above sent them to "augment",
# which is the rule's own precondition, so they can actually test it. reed98 was measured with them and is deliberately
# ABSENT: stage 1 called it "keep original", so a strategy test on it would be scored on a graph the earlier rules say
# should not be augmented at all. Selecting the test set on the STAGE-1 CALL (never on an outcome) is what makes this
# conditional; the cost is that stage 2 is only ever tested on graphs stage 1 sends here.
STRATEGY_HELDOUT = ["amherst41", "johnshopkins55", "cornell5"]

assert not (set(STRATEGY_PANEL) & set(STRATEGY_HELDOUT)), "a dataset is in BOTH the strategy panel and its held-out set - it must be in exactly one"


def predict_strategy(props):
    '''Which signal to add, once augmentation is indicated: "centrality", or "psi or degree" which the rule cannot split.'''
    v = props.get(FROZEN_STRATEGY.predictor)
    if v is None or v != v:                                # needs labels; an unlabelled graph gets no strategy call
        return "n/a"
    return FROZEN_STRATEGY.signal if v > FROZEN_STRATEGY.point else "psi or degree (undetermined)"


def predict_gated(props):
    '''The two-gate call: rule 1 vetoes on its reliable side, the gate decides inside the low-homophily zone, rule 2 covers unlabelled graphs.'''
    calls, _ = predict(props)
    gate = predict_one(FROZEN_GATE, props.get(FROZEN_GATE.predictor))
    if calls["rule1"] == "keep original":                  # the veto side: 3/3 in Module 3, no second variable needed
        return calls, gate, "keep original", "rule 1 veto"
    if calls["rule1"] == "augment":                        # the ambiguous side: this is the only place the gate speaks
        return calls, gate, (gate if gate != "n/a" else "augment"), ("gate" if gate != "n/a" else "rule 1 (gate not measurable)")
    # No labels -> rule 1 cannot fire at all. The gate's cut was fitted INSIDE the zone, so it does not transfer here; fall
    # back to rule 2, exactly as Module 3 did for ogbl_ddi, and carry the gate value as information only.
    return calls, gate, calls["rule2"], "rule 2 (no labels)"
