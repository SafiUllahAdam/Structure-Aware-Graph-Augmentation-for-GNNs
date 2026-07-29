'''ViRGo: virtual role-graph embedding for structural identity. Library half of the repo — import only, never run.'''
# Deliberately empty of imports: `from virgo import graph_io` must not drag in torch, and `virgo.config` must not
# drag in the encoders. Every consumer names the module it needs.
