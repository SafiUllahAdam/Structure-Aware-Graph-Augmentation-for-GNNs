"""Build labels/<dataset>.labels from the original LINQS data, matching the edgelist's node numbering.

The author's edgelists number nodes by order of appearance in the LINQS '<name>.content' file
(verified: cora edges reproduced at 100% overlap). Planetoid uses a different numbering, so it is
NOT used here. Downloads are plain urllib (no fsspec)."""

import io
import tarfile
import urllib.request
from pathlib import Path

LINQS = {
    "cora":     "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
    "citeseer": "https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz",
}


# Reads the local edgelist into its set of undirected integer edges.
def read_edgelist(path):
    edges = set()
    for line in open(path):
        if line.strip():
            a, b = map(int, line.split()[:2])
            edges.add(tuple(sorted((a, b))))
    return edges


# Downloads the LINQS .tgz and returns the .content and .cites lines for a dataset.
def fetch_linqs(name, retries=3, timeout=60):
    """Get original <name>.content and <name>.cites text from LINQS via urllib."""
    for attempt in range(1, retries + 1):
        try:
            raw = urllib.request.urlopen(LINQS[name], timeout=timeout).read()
            break
        except Exception as e:
            if attempt == retries:
                raise RuntimeError(f"Failed to download {LINQS[name]}: {e}")
    tar = tarfile.open(fileobj=io.BytesIO(raw))
    pick = lambda end: tar.extractfile(next(m for m in tar.getmembers() if m.name.endswith(end))).read().decode()
    return pick(f"{name}.content").splitlines(), pick(f"{name}.cites").splitlines()


# Builds labels/<name>.labels in the edgelist's numbering, after verifying the edges match.
def make_labels(name):
    """Map paper-id -> 1-based node id by content order; verify against the edgelist; write labels."""
    name = name.lower()
    content, cites = fetch_linqs(name)
    paper_ids = [ln.split("\t")[0] for ln in content]
    paper_labels = [ln.split("\t")[-1] for ln in content]
    idmap = {pid: i + 1 for i, pid in enumerate(paper_ids)}     # content order = the author's node numbering

    cite_edges = {tuple(sorted((idmap[a], idmap[b])))
                  for a, b in (ln.split("\t")[:2] for ln in cites) if a in idmap and b in idmap}
    local_edges = read_edgelist(f"input/{name}.edgelist")
    overlap = len(cite_edges & local_edges) / len(local_edges)
    print(f"{name}: content_nodes={len(paper_ids)} labels={len(set(paper_labels))} edge_overlap={overlap:.4f}")

    if overlap < 0.90:
        print(f"STOP: {name} content-order does not match input/{name}.edgelist (overlap {overlap:.4f})")
        return
    Path("labels").mkdir(exist_ok=True)
    out = f"labels/{name}.labels"
    with open(out, "w") as f:
        for node_id, label in enumerate(paper_labels, start=1):
            f.write(f"{node_id} {label}\n")
    print("WROTE", out, f"({len(paper_labels)} nodes)")


if __name__ == "__main__":
    make_labels("Cora")
    make_labels("Citeseer")
