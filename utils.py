"""Utilities for topology construction, mesh deformation, ECT computation,
and model training."""

import numpy as np
from scipy.spatial import Delaunay
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

def build_topology(n_vertices, triangles):
    """Build simplicial complex topology from a triangle list.

    Returns a dictionary with vertex/edge/triangle counts, index arrays for
    boundary and coboundary adjacencies, and an undirected edge_index for
    graph baselines.
    """
    edge_map = {}
    edges_list = []
    triangle_edges = []

    for tri in triangles:
        te = []
        for i in range(3):
            v0, v1 = int(tri[i]), int(tri[(i + 1) % 3])
            key = (min(v0, v1), max(v0, v1))
            if key not in edge_map:
                edge_map[key] = len(edges_list)
                edges_list.append(key)
            te.append(edge_map[key])
        triangle_edges.append(te)

    edges = np.array(edges_list, dtype=np.int64) if edges_list else np.zeros((0, 2), dtype=np.int64)
    tri_edges = np.array(triangle_edges, dtype=np.int64) if triangle_edges else np.zeros((0, 3), dtype=np.int64)
    tri_arr = np.array(triangles, dtype=np.int64) if len(triangles) > 0 else np.zeros((0, 3), dtype=np.int64)
    ne = len(edges)
    nt = len(triangles)

    # Boundary: vertex -> edge, edge -> triangle
    b1_dst = np.repeat(np.arange(ne), 2) if ne > 0 else np.array([], dtype=np.int64)
    b1_src = edges.flatten() if ne > 0 else np.array([], dtype=np.int64)
    b2_dst = np.repeat(np.arange(nt), 3) if nt > 0 else np.array([], dtype=np.int64)
    b2_src = tri_edges.flatten() if nt > 0 else np.array([], dtype=np.int64)

    # Undirected edge index for graph baselines
    if ne > 0:
        edge_index = np.stack([
            np.concatenate([edges[:, 0], edges[:, 1]]),
            np.concatenate([edges[:, 1], edges[:, 0]]),
        ])
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    return {
        "n_vertices": n_vertices,
        "n_edges": ne,
        "n_triangles": nt,
        "edges": torch.from_numpy(edges).long(),
        "triangles": torch.from_numpy(tri_arr).long(),
        "b1_src": torch.from_numpy(b1_src).long(),
        "b1_dst": torch.from_numpy(b1_dst).long(),
        "b2_src": torch.from_numpy(b2_src).long(),
        "b2_dst": torch.from_numpy(b2_dst).long(),
        "cb0_src": torch.from_numpy(b1_dst).long(),  # coboundary = reverse boundary
        "cb0_dst": torch.from_numpy(b1_src).long(),
        "cb1_src": torch.from_numpy(b2_dst).long(),
        "cb1_dst": torch.from_numpy(b2_src).long(),
        "edge_index": torch.from_numpy(edge_index).long(),
    }


def topo_to_device(topo, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in topo.items()}


# ---------------------------------------------------------------------------
# Mesh generation and deformation
# ---------------------------------------------------------------------------

def generate_base_triangulation(n_vertices=40, seed=42):
    """Generate a Delaunay triangulation of random points in the unit disk."""
    rng = np.random.RandomState(seed)
    n_interior = n_vertices - 10
    angles = rng.uniform(0, 2 * np.pi, n_interior)
    radii = np.sqrt(rng.uniform(0.01, 0.85, n_interior))
    interior = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
    bnd_angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    boundary = np.stack([np.cos(bnd_angles), np.sin(bnd_angles)], axis=1)
    points = np.vstack([interior, boundary])
    tri = Delaunay(points)
    return points.astype(np.float32), tri.simplices


def apply_deformation(coords, dtype, strength, rng):
    """Apply a smooth deformation to vertex coordinates.

    Args:
        coords: (V, 2) or (V, 3) array of vertex coordinates.
        dtype: one of 'bend', 'twist', 'stretch', 'random_smooth'.
        strength: scalar controlling deformation magnitude.
        rng: numpy RandomState for reproducibility.
    """
    c = coords.copy()
    if dtype == "bend":
        c[:, 1] += strength * np.sin(np.pi * c[:, 0])
    elif dtype == "twist":
        r = np.sqrt(c[:, 0] ** 2 + c[:, 1] ** 2)
        th = np.arctan2(c[:, 1], c[:, 0]) + strength * r
        c[:, 0] = r * np.cos(th)
        c[:, 1] = r * np.sin(th)
    elif dtype == "stretch":
        c[:, 0] *= 1.0 + strength
        c[:, 1] /= 1.0 + 0.5 * strength
    elif dtype == "random_smooth":
        for _ in range(3):
            freq = rng.uniform(0.5, 2.5, 2)
            phase = rng.uniform(0, 2 * np.pi, 2)
            amp = strength * rng.uniform(-0.3, 0.3, 2)
            c[:, 0] += amp[0] * np.sin(freq[0] * c[:, 1] + phase[0])
            c[:, 1] += amp[1] * np.sin(freq[1] * c[:, 0] + phase[1])
    return c.astype(np.float32)


def embed_triangulation_2d(n_vertices, triangles, rng):
    """Embed an abstract triangulation in R^2 via spectral layout."""
    adj = np.zeros((n_vertices, n_vertices), dtype=np.float32)
    for t in triangles:
        for i in range(3):
            adj[t[i], t[(i + 1) % 3]] = 1
            adj[t[(i + 1) % 3], t[i]] = 1
    L = np.diag(adj.sum(1)) - adj
    try:
        _, ev = np.linalg.eigh(L)
        c = ev[:, 1:3].astype(np.float32)
        c -= c.mean(0)
        s = c.std()
        if s > 1e-6:
            c /= s
    except Exception:
        c = rng.randn(n_vertices, 2).astype(np.float32)
    return c


def augment_coords_3d(coords, n_aug, rng):
    """Generate augmented copies via small rotations and Gaussian noise.

    Args:
        coords: (N, V, 3) tensor.
        n_aug: number of total copies (including original).
        rng: numpy RandomState.

    Returns:
        (N * n_aug, V, 3) tensor.
    """
    augmented = [coords]
    for _ in range(n_aug - 1):
        angles = rng.uniform(-0.25, 0.25, 3).astype(np.float32)
        cx, cy, cz = np.cos(angles)
        sx, sy, sz = np.sin(angles)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
        R = torch.from_numpy(Rx @ Ry @ Rz)
        rotated = torch.einsum("nvc,cd->nvd", coords, R)
        augmented.append(rotated + torch.randn_like(rotated) * 0.02)
    return torch.cat(augmented, dim=0)


# ---------------------------------------------------------------------------
# ECT computation
# ---------------------------------------------------------------------------

def sample_directions_2d(n_dirs):
    """Uniformly spaced directions on the unit circle."""
    angles = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)


def compute_ect(coords, triangles, edges, directions, thresholds):
    """Compute sampled ECT values for an embedded 2-complex.

    Args:
        coords: (V, d) vertex coordinates.
        triangles: (T, 3) triangle vertex indices.
        edges: (E, 2) edge vertex indices.
        directions: (D, d) unit direction vectors.
        thresholds: (M,) threshold values.

    Returns:
        (D, M) array of ECT values.
    """
    nd, nt_ = len(directions), len(thresholds)
    ect = np.zeros((nd, nt_), dtype=np.float32)
    dim = min(coords.shape[1], directions.shape[1])

    for di, d in enumerate(directions):
        vp = coords[:, :dim] @ d[:dim]
        ep = np.maximum(vp[edges[:, 0]], vp[edges[:, 1]])
        if len(triangles) > 0:
            tp = np.max(np.stack([vp[triangles[:, i]] for i in range(3)]), axis=0)
        else:
            tp = np.array([])
        for ti, t in enumerate(thresholds):
            n0 = np.sum(vp <= t)
            n1 = np.sum(ep <= t)
            n2 = np.sum(tp <= t) if len(tp) > 0 else 0
            ect[di, ti] = n0 - n1 + n2
    return ect


def compute_vertex_curvature(coords, triangles, n_verts):
    """Discrete Gaussian curvature: K(v) = 2*pi - sum of incident angles."""
    angle_sum = np.zeros(n_verts, dtype=np.float64)
    edge_tri_count = {}
    for tri in triangles:
        for i in range(3):
            e = tuple(sorted([int(tri[i]), int(tri[(i + 1) % 3])]))
            edge_tri_count[e] = edge_tri_count.get(e, 0) + 1

    vertex_is_boundary = np.zeros(n_verts, dtype=bool)
    for e, count in edge_tri_count.items():
        if count == 1:
            vertex_is_boundary[e[0]] = True
            vertex_is_boundary[e[1]] = True

    for tri in triangles:
        for i in range(3):
            v = int(tri[i])
            e1 = coords[int(tri[(i + 1) % 3])] - coords[v]
            e2 = coords[int(tri[(i + 2) % 3])] - coords[v]
            cos_a = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-12)
            angle_sum[v] += np.arccos(np.clip(cos_a, -1, 1))

    curvature = np.where(
        vertex_is_boundary, np.pi - angle_sum, 2 * np.pi - angle_sum
    )
    return curvature.astype(np.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model, tr_c, tr_y, te_c, te_y, topo, n_epochs, lr,
                task="cls", verbose=True, batch_size=64):
    """Train a model with shared topology across samples.

    Args:
        model: nn.Module.
        tr_c, te_c: (N, V, d) coordinate tensors.
        tr_y, te_y: label tensors (long for cls, float for reg).
        topo: topology dict (on the correct device).
        n_epochs: number of training epochs.
        lr: learning rate.
        task: 'cls' for classification, 'reg' for regression.
        verbose: print progress every n_epochs/5 epochs.
        batch_size: mini-batch size.

    Returns:
        (best_metric, final_metric) tuple.
    """
    device = next(model.parameters()).device
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    bs = min(batch_size, len(tr_c))
    n_tr = len(tr_c)
    loss_fn = F.cross_entropy if task == "cls" else F.mse_loss
    best = -1.0 if task == "cls" else 1e9

    for ep in range(n_epochs):
        model.train()
        perm = torch.randperm(n_tr)
        ep_loss, ep_correct = 0.0, 0

        for i in range(0, n_tr, bs):
            idx = perm[i : i + bs]
            xb = tr_c[idx].to(device)
            yb = tr_y[idx].to(device)
            pred = model(xb, topo)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * len(idx)
            if task == "cls":
                ep_correct += (pred.argmax(-1) == yb).sum().item()

        sched.step()
        ep_loss /= n_tr

        # Evaluate
        model.eval()
        with torch.no_grad():
            preds = []
            for j in range(0, len(te_c), bs):
                preds.append(model(te_c[j : j + bs].to(device), topo))
            pred_all = torch.cat(preds, 0)
            ty = te_y.to(device)
            if task == "cls":
                metric = (pred_all.argmax(-1) == ty).float().mean().item()
                if metric > best:
                    best = metric
            else:
                metric = F.mse_loss(pred_all, ty).item()
                if metric < best:
                    best = metric

        if verbose and (ep + 1) % max(1, n_epochs // 5) == 0:
            if task == "cls":
                print(f"  Ep {ep+1:3d}  loss={ep_loss:.4f}  "
                      f"train_acc={ep_correct/n_tr:.4f}  test_acc={metric:.4f}")
            else:
                print(f"  Ep {ep+1:3d}  train={ep_loss:.4f}  test={metric:.4f}")

    return best, metric


def kfold_train_eval(model_fn, coords, labels, topo, n_folds, n_epochs, lr,
                     task="cls", device="cpu", n_aug=1, rng=None):
    """K-fold cross-validation with optional data augmentation.

    Args:
        model_fn: callable returning a fresh model instance.
        coords: (N, V, d) tensor.
        labels: (N,) or (N, d) tensor.
        topo: topology dict.
        n_folds: number of folds.
        n_epochs: epochs per fold.
        lr: learning rate.
        task: 'cls' or 'reg'.
        device: torch device string.
        n_aug: augmentation multiplier (1 = no augmentation).
        rng: numpy RandomState for augmentation.

    Returns:
        (mean_metric, std_metric) across folds.
    """
    n = len(coords)
    fold_size = n // n_folds
    metrics = []

    for fold in range(n_folds):
        te_start = fold * fold_size
        te_end = te_start + fold_size
        te_idx = list(range(te_start, te_end))
        tr_idx = list(range(0, te_start)) + list(range(te_end, n))

        tr_c = coords[tr_idx]
        tr_y = labels[tr_idx]
        te_c = coords[te_idx]
        te_y = labels[te_idx]

        if n_aug > 1 and rng is not None:
            tr_c = augment_coords_3d(tr_c, n_aug, rng)
            if tr_y.dim() == 1:
                tr_y = tr_y.repeat(n_aug)
            else:
                tr_y = tr_y.repeat(n_aug, 1)

        topo_dev = topo_to_device(topo, device)
        model = model_fn().to(device)
        best, final = train_model(
            model, tr_c, tr_y, te_c, te_y, topo_dev,
            n_epochs, lr, task, verbose=False,
        )
        metrics.append(final)
        print(f"    Fold {fold+1}/{n_folds}: {final:.3f}")

    return np.mean(metrics), np.std(metrics)
