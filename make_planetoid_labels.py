from pathlib import Path
from torch_geometric.datasets import Planetoid

def read_edges(path):
    edges, nodes = set(), set()
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            a, b = map(int, line.split()[:2])
            edges.add(tuple(sorted((a, b))))
            nodes.update([a, b])
    return edges, nodes

def make_labels(name):
    lname = name.lower()
    local_edges, local_nodes = read_edges(f"input/{lname}.edgelist")

    dataset = Planetoid(root="data/planetoid", name=name)
    data = dataset[0]

    pyg_edges = set()
    for u, v in data.edge_index.t().tolist():
        u, v = int(u) + 1, int(v) + 1
        pyg_edges.add(tuple(sorted((u, v))))

    overlap = len(local_edges & pyg_edges) / len(local_edges)
    print(name, "local_nodes=", len(local_nodes), "pyg_nodes=", data.num_nodes, "edge_overlap=", round(overlap, 4))

    if len(local_nodes) != data.num_nodes or overlap < 0.90:
        print("STOP:", name, "labels may not match this edgelist")
        return

    Path("labels").mkdir(exist_ok=True)
    out = f"labels/{lname}.labels"
    with open(out, "w") as f:
        for node_id, label in enumerate(data.y.tolist(), start=1):
            f.write(f"{node_id} {int(label)}\n")

    print("WROTE", out)

if __name__ == "__main__":
    make_labels("Cora")
    make_labels("Citeseer")
