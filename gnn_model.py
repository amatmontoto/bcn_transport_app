"""
gnn_model.py
============
The Graph Neural Network for transport-equity scoring (Option B architecture).

Pipeline
--------
1. Build a PyG graph from processed/ (nodes + edges).
2. Compute an ANALYTIC spatial-coverage score per stop — this is the
   training label.  It is a deterministic formula based on how many
   elderly residents fall inside the 250 m walking threshold relative to
   the demand around the stop.
3. Train a GraphSAGE GNN (node regression) to predict that score from
   graph structure + node features.

Why a GNN and not just the formula?  Because once trained, the GNN can
re-score a MODIFIED graph (a stop moved, a stop added) and the change
propagates to neighbouring nodes through message passing — capturing
network-wide ripple effects a per-stop formula cannot see.  That is what
the prescriptive layer (proposals.py) exploits.

Run directly:  python gnn_model.py      (trains + caches the model)
"""

import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

BASE     = os.path.dirname(os.path.abspath(__file__))
PROC_DIR = os.path.join(BASE, "processed")
MODEL_PT = os.path.join(PROC_DIR, "gnn_model.pt")

R_ELDERLY = 250.0
R_GENERAL = 400.0

# ----------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------
FEATURE_COLS = [
    "lat_n", "lon_n",          # normalised position
    "is_metro",                # mode flag
    "elderly_250_n",           # elderly within 250 m  (normalised)
    "elderly_400_n",           # elderly within 400 m
    "pop_400_n",               # general pop within 400 m
    "degree_n",                # connectivity in the graph
    "n_lines_n",               # how many lines serve this stop
]


def compute_coverage_score(nodes):
    """
    Analytic spatial-coverage score per stop, in [0, 1].

    Idea: a stop is doing its job well when the elderly people *near enough
    to walk to it* (within 250 m) are a large share of all the elderly people
    in its wider 400 m demand zone.  A stop with lots of elderly just outside
    the 250 m ring (reachable in the 400 m ring but not the 250 m ring) is
    leaving demand stranded -> low score.

    score = elderly_250 / (elderly_400 + eps)        (reach ratio)
            tempered by an absolute-demand weight so that tiny-demand stops
            in empty areas don't dominate.
    """
    e250 = nodes["elderly_250"].values.astype(float)
    e400 = nodes["elderly_400"].values.astype(float)

    reach = e250 / (e400 + 1.0)                 # 0..1  (fraction within 250 m)
    reach = np.clip(reach, 0, 1)

    # demand weight: log-scaled elderly headcount, normalised 0..1
    dem = np.log1p(e400)
    dem = (dem - dem.min()) / (dem.max() - dem.min() + 1e-9)

    # final score: a stop is "good" if it reaches its elderly demand.
    # we blend reach (how well it covers) with demand (how much it matters)
    score = 0.75 * reach + 0.25 * dem
    return np.clip(score, 0, 1)


def build_features(nodes, edges):
    """Return (feature_matrix, score_label, node_index_map)."""
    nodes = nodes.reset_index(drop=True).copy()
    idx = {sid: i for i, sid in enumerate(nodes["stop_id"])}

    # degree + edge_index
    deg = np.zeros(len(nodes))
    ei = [[], []]
    for _, e in edges.iterrows():
        if e["u"] in idx and e["v"] in idx:
            a, b = idx[e["u"]], idx[e["v"]]
            ei[0] += [a, b]
            ei[1] += [b, a]
            deg[a] += 1
            deg[b] += 1

    def norm(x):
        x = np.asarray(x, float)
        return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-9)

    n_lines = nodes["lines"].fillna("").apply(
        lambda s: len(s.split(",")) if s else 0).values

    feat = pd.DataFrame({
        "lat_n":         norm(nodes["lat"]),
        "lon_n":         norm(nodes["lon"]),
        "is_metro":      (nodes["mode"] == "metro").astype(float),
        "elderly_250_n": norm(nodes["elderly_250"]),
        "elderly_400_n": norm(nodes["elderly_400"]),
        "pop_400_n":     norm(nodes["pop_400"]),
        "degree_n":      norm(deg),
        "n_lines_n":     norm(n_lines),
    })[FEATURE_COLS].fillna(0.0)

    score = compute_coverage_score(nodes)
    edge_index = torch.tensor(ei, dtype=torch.long)
    x = torch.tensor(feat.values, dtype=torch.float)
    y = torch.tensor(score, dtype=torch.float)
    return x, edge_index, y, idx, deg


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
class CoverageGNN(nn.Module):
    """2-layer GraphSAGE node-regression network."""
    def __init__(self, in_dim, hidden=32):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.head  = nn.Sequential(
            nn.Linear(hidden, 16), nn.ReLU(),
            nn.Dropout(0.1),       nn.Linear(16, 1),
        )

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        return torch.sigmoid(self.head(h)).squeeze(-1)


# ----------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------
def train(epochs=300, verbose=True):
    nodes = pd.read_parquet(os.path.join(PROC_DIR, "nodes.parquet"))
    edges = pd.read_parquet(os.path.join(PROC_DIR, "edges.parquet"))

    x, edge_index, y, idx, deg = build_features(nodes, edges)
    data = Data(x=x, edge_index=edge_index, y=y)

    torch.manual_seed(42)
    n = data.num_nodes
    perm = torch.randperm(n)
    n_train = int(0.8 * n)
    train_mask = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)
    train_mask[perm[:n_train]] = True
    test_mask[perm[n_train:]]  = True

    model = CoverageGNN(in_dim=x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)

    best_test = float("inf")
    best_state = None
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        pred = model(data.x, data.edge_index)
        loss = F.mse_loss(pred[train_mask], data.y[train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(data.x, data.edge_index)
            test_loss = F.mse_loss(pred[test_mask], data.y[test_mask]).item()
            if test_loss < best_test:
                best_test = test_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and ep % 50 == 0:
            # R^2 on test
            yt = data.y[test_mask]
            pt = pred[test_mask]
            ss_res = ((yt - pt) ** 2).sum()
            ss_tot = ((yt - yt.mean()) ** 2).sum()
            r2 = (1 - ss_res / ss_tot).item()
            print(f"  epoch {ep:3d}  train {loss.item():.4f}  "
                  f"test {test_loss:.4f}  R2 {r2:.3f}", flush=True)

    model.load_state_dict(best_state)

    # final metrics
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index)
        yt, pt = data.y[test_mask], pred[test_mask]
        ss_res = ((yt - pt) ** 2).sum()
        ss_tot = ((yt - yt.mean()) ** 2).sum()
        r2  = (1 - ss_res / ss_tot).item()
        mae = (yt - pt).abs().mean().item()
        # pearson
        pm, ym = pt - pt.mean(), yt - yt.mean()
        pear = (pm @ ym / (pm.norm() * ym.norm() + 1e-9)).item()

    torch.save({
        "state_dict": model.state_dict(),
        "in_dim": x.shape[1],
        "metrics": {"test_mse": best_test, "r2": r2,
                    "mae": mae, "pearson": pear},
        "feature_cols": FEATURE_COLS,
    }, MODEL_PT)

    if verbose:
        print(f"✓ trained — test R2={r2:.3f}  MAE={mae:.3f}  "
              f"Pearson={pear:.3f}")
    return model, {"test_mse": best_test, "r2": r2,
                   "mae": mae, "pearson": pear}


def load_model():
    """Load the cached trained model; train first if not present."""
    if not os.path.exists(MODEL_PT):
        return train(verbose=False)[0]
    ck = torch.load(MODEL_PT, weights_only=False)
    model = CoverageGNN(in_dim=ck["in_dim"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_scores(model, x, edge_index):
    model.eval()
    return model(x, edge_index).cpu().numpy()


if __name__ == "__main__":
    train()
