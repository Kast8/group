from pathlib import Path
import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def describe_graph(adj: torch.Tensor):
    degree = adj.sum(dim=1)
    print(f"nodes: {adj.size(0)}")
    print(f"edges: {int(adj.sum().item() / 2)}")
    print(f"avg degree: {degree.mean().item():.2f}")
    print(f"min/max degree: {degree.min().item():.0f} / {degree.max().item():.0f}")


def accuracy(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred[mask] == y[mask]).float().mean().item()


def spectral_layout(adj: torch.Tensor) -> np.ndarray:
    """Place nearby graph nodes close together using only graph structure."""
    adj_np = adj.detach().cpu().numpy()
    degree = adj_np.sum(axis=1)
    laplacian = np.diag(degree) - adj_np
    _, eigenvectors = np.linalg.eigh(laplacian)
    positions = eigenvectors[:, 1:3]
    positions /= np.abs(positions).max(axis=0, keepdims=True).clip(min=1e-8)
    return positions


def plot_graph(adj: torch.Tensor, y: torch.Tensor, train_mask: torch.Tensor, ax, title: str = "Graph structure (spectral layout)"):
    adj_np = adj.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    train_mask_np = train_mask.detach().cpu().numpy()
    positions = spectral_layout(adj)

    edges = np.column_stack(np.nonzero(np.triu(adj_np, k=1)))
    ax.add_collection(LineCollection(positions[edges], colors="#B8B8B8", linewidths=0.35, alpha=0.35))

    color_map = plt.get_cmap("tab10")
    for class_id in np.unique(y_np):
        mask = y_np == class_id
        ax.scatter(
            *positions[mask].T,
            s=22,
            color=color_map(int(class_id) % 10),
            label=f"class {class_id}",
            zorder=2,
        )
    ax.scatter(
        *positions[train_mask_np].T,
        s=55,
        facecolors="none",
        edgecolors="black",
        linewidths=0.8,
        label="training node",
        zorder=3,
    )
    if title:
        ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="best", fontsize=8)


def load_cora(data_dir: str = "data/cora", graph_data_cls=None, train_per_class: int = 20, val_per_class: int = 30):
    if graph_data_cls is None:
        raise ValueError("graph_data_cls must be provided, for example graph_data_cls=GraphData")

    data_dir = Path(data_dir)
    content_path = data_dir / "cora.content"
    cites_path = data_dir / "cora.cites"
    missing = [str(path) for path in (content_path, cites_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("missing Cora files: " + ", ".join(missing))

    with content_path.open(encoding="utf-8") as file:
        rows = [line.split() for line in file if line.strip()]
    paper_ids = [row[0] for row in rows]
    id_to_index = {paper_id: i for i, paper_id in enumerate(paper_ids)}
    x_np = np.asarray([row[1:-1] for row in rows], dtype=np.float32)
    label_names = sorted({row[-1] for row in rows})
    label_to_index = {label: i for i, label in enumerate(label_names)}
    y_np = np.asarray([label_to_index[row[-1]] for row in rows], dtype=np.int64)

    x_np /= x_np.sum(axis=1, keepdims=True).clip(min=1.0)
    adj_np = np.zeros((len(rows), len(rows)), dtype=np.float32)
    with cites_path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            source_id, target_id = line.split()[:2]
            if source_id in id_to_index and target_id in id_to_index:
                source, target = id_to_index[source_id], id_to_index[target_id]
                adj_np[source, target] = adj_np[target, source] = 1.0

    rng = np.random.default_rng(42)
    train_mask = np.zeros(len(rows), dtype=bool)
    val_mask = np.zeros(len(rows), dtype=bool)
    test_mask = np.ones(len(rows), dtype=bool)
    for class_id in range(len(label_names)):
        indices = np.flatnonzero(y_np == class_id)
        rng.shuffle(indices)
        train_indices = indices[:train_per_class]
        val_indices = indices[train_per_class:train_per_class + val_per_class]
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[train_indices] = test_mask[val_indices] = False

    print("classes:", {i: name for i, name in enumerate(label_names)})
    return graph_data_cls(
        x=torch.from_numpy(x_np),
        y=torch.from_numpy(y_np),
        adj=torch.from_numpy(adj_np),
        train_mask=torch.from_numpy(train_mask),
        val_mask=torch.from_numpy(val_mask),
        test_mask=torch.from_numpy(test_mask),
    )


def neighborhood_sample(adj: torch.Tensor, max_nodes: int = 180) -> np.ndarray:
    adj_np = adj.detach().cpu().numpy()
    start = int(adj_np.sum(axis=1).argmax())
    selected, seen, queue = [], {start}, [start]
    while queue and len(selected) < max_nodes:
        node = queue.pop(0)
        selected.append(node)
        for neighbor in np.flatnonzero(adj_np[node]):
            neighbor = int(neighbor)
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return np.asarray(selected, dtype=np.int64)


def plot_multiclass_graph(adj: torch.Tensor, y: torch.Tensor, train_mask: torch.Tensor, ax):
    plot_graph(adj, y, train_mask, ax, title="")
