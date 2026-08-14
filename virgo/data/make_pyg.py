'''Convert torch_geometric benchmark graphs to ViRGo edgelist/label/.nodes files: pubmed (Planetoid), actor (film) - structural-only.'''
# Added 2026-08-02 as Module-3 held-out datasets (see frozen_rules.HELDOUT). Same policy as make_hetero/make_ogb:
# node features (PubMed 500-dim TF-IDF, Actor 932-dim) are IGNORED by design, and any official train/val/test masks are
# ignored - these run the ViRGo core protocol (stratified 70% node classification, 70:30 link prediction).
# lastfm_asia (2026-08-04) is the one exception to "download via the PyG class": torch_geometric.datasets.LastFMAsia
# fetches graphmining.ai, which no longer serves the file (TLS handshake failure / 404), so the raw graph is read from
# SNAP's primary archive of the same dataset (Rozemberczki & Sarkar 2020) instead. Node ids, labels and edges are the
# source's own; SNAP ships features as a liked-artist JSON rather than PyG's 128-dim matrix, which we would drop anyway.
# reed98/amherst41/johnshopkins55 (2026-08-14) are the LINKX non-homophilous Facebook100 college networks. Their label is
# gender, which is MISSING for some users - PyG codes that as y = -1, and those nodes are left out of the .labels file
# rather than written as a third class, exactly as the LINKX benchmark evaluates them. The graph keeps every node.

import argparse
from pathlib import Path

from virgo.data.make_ogb import _write_edges, _write_nodes   # shared writers; the import also installs the torch.load allowlist

RAW = "output/pyg_raw"                         # PyG download cache: derived, never hand-edited, kept out of input/

PYG = ["pubmed", "actor", "amazon_photo", "lastfm_asia", "reed98", "amherst41", "johnshopkins55", "cornell5"]
LINKX = ["reed98", "amherst41", "johnshopkins55", "cornell5"]


# Returns the single PyG graph for a supported benchmark; node features are dropped downstream (structural-only).
def _load(name):
    '''Load one PyG benchmark graph. PubMed = Planetoid citation (3 classes); Actor = Geom-GCN film (5); Amazon Photo = co-purchase (8); LastFM Asia = music social (18); the LINKX trio = Facebook100 college networks (2, gender).'''
    from torch_geometric.datasets import Planetoid, Actor, Amazon, LINKXDataset
    if name == "pubmed":
        return Planetoid(root=RAW, name="PubMed")[0]
    if name == "actor":
        return Actor(root=f"{RAW}/actor")[0]
    if name == "amazon_photo":
        return Amazon(root=RAW, name="Photo")[0]
    if name in LINKX:
        return LINKXDataset(root=RAW, name=name)[0]
    if name == "lastfm_asia":
        return _lastfm_asia()                             # SNAP archive, not the PyG class: its host is dead (see header)
    raise ValueError(f"Unknown dataset '{name}'. Use: {PYG}.")


# Reads SNAP's lastfm_asia archive (downloaded once into RAW) as a PyG graph; no data.x, the liked-artist JSON is not loaded.
def _lastfm_asia():
    '''LastFM Asia from SNAP: mutual-follower graph of 7,624 Asian users, 18 country classes. Structural-only, so features are skipped.'''
    import numpy as np
    import torch
    from torch_geometric.data import Data, download_url, extract_zip
    raw = Path(RAW) / "lastfm_asia" / "raw"
    inner = raw / "lasftm_asia"                           # the archive's own folder name, misspelled at the source
    if not (inner / "lastfm_asia_edges.csv").exists():
        extract_zip(download_url("https://snap.stanford.edu/data/lastfm_asia.zip", str(raw)), str(raw))
    e = np.loadtxt(inner / "lastfm_asia_edges.csv", delimiter=",", skiprows=1, dtype=np.int64)     # one row per undirected pair
    y = np.loadtxt(inner / "lastfm_asia_target.csv", delimiter=",", skiprows=1, dtype=np.int64)
    assert (y[:, 0] == np.arange(len(y))).all(), "lastfm_asia target ids are not 0..n-1 in order - labels would misalign"
    return Data(edge_index=torch.from_numpy(e.T), y=torch.from_numpy(y[:, 1]), num_nodes=len(y))


# Builds input/<name>.edgelist + .nodes + labels/<name>.labels from the PyG copy; drops data.x (structural-only).
def make_pyg(name):
    '''One PyG benchmark -> edgelist + .nodes + labels. node_features and official masks ignored by design.'''
    g = _load(name)
    n = g.num_nodes
    ne = _write_edges(f"input/{name}.edgelist", g.edge_index.numpy())   # dedup+sort symmetrizes; self-loops dropped
    _write_nodes(f"input/{name}.edgelist", n)
    y = g.y.numpy().reshape(-1)
    Path("labels").mkdir(exist_ok=True)
    with open(f"labels/{name}.labels", "w") as f:
        for i in range(n):
            if y[i] >= 0:                                 # Facebook100 codes a missing gender as -1: unlabelled, not a class
                f.write(f"{i} {int(y[i])}\n")
    feats = f"data.x ({g.x.shape[1]}-dim) IGNORED" if g.x is not None else "node features NOT LOADED"
    print(f"{name}: nodes={n} edges={ne} classes={len(set(y[y >= 0]))} labelled={int((y >= 0).sum())}/{n} avg_degree={2 * ne / n:.2f} | "
          f"{feats} (structural-only) | official masks IGNORED (ViRGo core protocol)")
    print(f"WROTE input/{name}.edgelist + .nodes + labels/{name}.labels")


# Defines command-line options: which PyG dataset to build (default all).
def parse_args():
    '''Parses arguments.'''
    p = argparse.ArgumentParser(description="Convert torch_geometric benchmark graphs to ViRGo edgelist/label/.nodes files (structural-only).")
    p.add_argument('--dataset', default='all', choices=['all'] + PYG, help='Which dataset to build. Default all.')
    return p.parse_args()


# Builds the requested dataset(s) from the command line.
def main(args):
    for name in PYG:
        if args.dataset in ('all', name):
            make_pyg(name)


if __name__ == "__main__":
    main(parse_args())
