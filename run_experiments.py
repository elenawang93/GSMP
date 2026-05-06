"""Reproduce all experiments from the paper.

Usage:
    python run_experiments.py --exp <name> [options]

Experiments:
    family      Deformation family classification
    ect         ECT vector regression
    coboundary  Per-vertex curvature with depth sweep
    permutation Permutation equivariance test
    mantra      Deformation classification on MANTRA
    mantra_ect  ECT regression on MANTRA
    faust_pose  FAUST 10-way pose classification
    faust_reg   FAUST geometric regression
    confounded  Geometric summary regression
    
Example:
    python run_experiments.py --exp family --hd 32 --L 4 --epochs 80
    python run_experiments.py --exp faust_pose --hd 16 --L 3 --epochs 200
"""

import argparse
import numpy as np
import torch
import warnings

from models import (
    SimplicialNet, SimplicialNetPerVertex,
    GraphNet, DeepSetBaseline, MLPBaseline,
)
from utils import (
    build_topology, topo_to_device, generate_base_triangulation,
    apply_deformation, embed_triangulation_2d, augment_coords_3d,
    sample_directions_2d, compute_ect, compute_vertex_curvature,
    train_model, kfold_train_eval,
)

warnings.filterwarnings("ignore")

FAMILIES = ["bend", "twist", "stretch", "random_smooth"]


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------

def generate_family_data(n_samples, n_vertices, seed):
    rng = np.random.RandomState(seed)
    base, triangles = generate_base_triangulation(n_vertices, seed=42)
    topo = build_topology(n_vertices, triangles)

    per_fam = n_samples // 4
    all_coords, all_labels = [], []
    for fi, fam in enumerate(FAMILIES):
        for _ in range(per_fam):
            s = rng.uniform(0.3, 1.5)
            c = apply_deformation(base, fam, s, rng)
            all_coords.append(c)
            all_labels.append(fi)

    return (
        torch.from_numpy(np.stack(all_coords)),
        torch.tensor(all_labels, dtype=torch.long),
        topo,
        triangles,
        base,
    )


def generate_ect_data(n_samples, n_vertices, seed):
    rng = np.random.RandomState(seed)
    base, triangles = generate_base_triangulation(n_vertices, seed=42)
    topo = build_topology(n_vertices, triangles)
    edges = topo["edges"].numpy()
    dirs = sample_directions_2d(8)
    threshs = np.linspace(-2.5, 2.5, 10)

    all_coords, all_ect = [], []
    for _ in range(n_samples):
        fam = FAMILIES[rng.randint(4)]
        s = rng.uniform(0.3, 1.5)
        c = apply_deformation(base, fam, s, rng)
        e = compute_ect(c, triangles, edges, dirs, threshs)
        all_coords.append(c)
        all_ect.append(e.flatten())

    ct = torch.from_numpy(np.stack(all_coords))
    et = torch.from_numpy(np.stack(all_ect))
    mu, std = et.mean(0), et.std(0).clamp(min=1e-6)
    et = (et - mu) / std
    return ct, et, topo


def generate_curvature_data(n_samples, n_vertices, seed):
    rng = np.random.RandomState(seed)
    base, triangles = generate_base_triangulation(n_vertices, seed=42)
    topo = build_topology(n_vertices, triangles)

    all_coords, all_curv = [], []
    for _ in range(n_samples):
        fam = FAMILIES[rng.randint(4)]
        s = rng.uniform(0.3, 1.5)
        c = apply_deformation(base, fam, s, rng)
        k = compute_vertex_curvature(c, triangles, n_vertices)
        all_coords.append(c)
        all_curv.append(k)

    ct = torch.from_numpy(np.stack(all_coords))
    kt = torch.from_numpy(np.stack(all_curv)).unsqueeze(-1)
    mu, std = kt.mean(0, keepdim=True), kt.std(0, keepdim=True).clamp(min=1e-6)
    kt = (kt - mu) / std
    return ct, kt, topo


def generate_confounded_data(n_samples, n_vertices, seed):
    rng = np.random.RandomState(seed)
    base, triangles = generate_base_triangulation(n_vertices, seed=42)
    topo = build_topology(n_vertices, triangles)
    edges = topo["edges"].numpy()

    all_coords, all_stats = [], []
    for _ in range(n_samples):
        n_def = rng.randint(1, 4)
        c = base.copy()
        for _ in range(n_def):
            fam = FAMILIES[rng.randint(4)]
            s = rng.uniform(0.2, 1.0)
            c = apply_deformation(c, fam, s, rng)
        all_coords.append(c)

        # Compute geometric summary statistics
        disp = c - base
        norms = np.linalg.norm(disp, axis=1)
        lengths = np.linalg.norm(c[edges[:, 1]] - c[edges[:, 0]], axis=1)
        areas = []
        angle_sum = np.zeros(n_vertices, dtype=np.float64)
        for tri in triangles:
            v0, v1, v2 = c[tri[0]], c[tri[1]], c[tri[2]]
            areas.append(0.5 * abs(
                (v1[0] - v0[0]) * (v2[1] - v0[1])
                - (v2[0] - v0[0]) * (v1[1] - v0[1])
            ))
            for i in range(3):
                v = int(tri[i])
                e1 = c[int(tri[(i + 1) % 3])] - c[v]
                e2 = c[int(tri[(i + 2) % 3])] - c[v]
                cos_a = np.dot(e1, e2) / (
                    np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-12
                )
                angle_sum[v] += np.arccos(np.clip(cos_a, -1, 1))
        areas = np.array(areas)
        defects = 2 * np.pi - angle_sum

        stats = np.concatenate([
            [norms.mean(), norms.std(), norms.max(), np.median(norms),
             disp[:, 0].mean(), disp[:, 1].mean(),
             np.abs(disp[:, 0]).mean(), np.abs(disp[:, 1]).mean()],
            [lengths.mean(), lengths.std(), lengths.min(), lengths.max(),
             np.percentile(lengths, 25), np.percentile(lengths, 75),
             lengths.max() / (lengths.min() + 1e-8),
             np.sum(lengths > lengths.mean() + lengths.std()) / len(lengths)],
            [areas.mean(), areas.std(), areas.min(), areas.max(), areas.sum(),
             areas.std() / (areas.mean() + 1e-8)],
            [defects.mean(), defects.std(), defects.min(), defects.max(),
             np.abs(defects).mean(),
             np.sum(np.abs(defects) > 0.1) / n_vertices],
        ])
        all_stats.append(stats)

    ct = torch.from_numpy(np.stack(all_coords))
    st = torch.from_numpy(np.stack(all_stats).astype(np.float32))
    mu, std = st.mean(0), st.std(0).clamp(min=1e-6)
    st = (st - mu) / std
    return ct, st, topo


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def split_data(coords, labels, seed, frac=0.75):
    n = len(coords)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_tr = int(frac * n)
    return (coords[idx[:n_tr]], labels[idx[:n_tr]],
            coords[idx[n_tr:]], labels[idx[n_tr:]])


def run_family(args):
    """Table 1: Deformation family classification."""
    print("\n" + "=" * 60)
    print("Deformation Family Classification (chance = 0.25)")
    print("=" * 60)
    coords, labels, topo, tri, base = generate_family_data(
        args.n_samples, args.n_vertices, args.seed
    )
    tr_c, tr_y, te_c, te_y = split_data(coords, labels, args.seed)
    topo_dev = topo_to_device(topo, args.device)

    models = [
        ("Combinatorial SMP", lambda: SimplicialNet(2, args.hd, args.L, 4, "combinatorial")),
        ("Coord SMP (summed)", lambda: SimplicialNet(2, args.hd, args.L, 4, "coord_summed")),
        ("Coord SMP (rich)", lambda: SimplicialNet(2, args.hd, args.L, 4, "coord_rich")),
        ("GCN + coords", lambda: GraphNet(2, args.hd, args.L, 4, "GCN")),
        ("GIN + coords", lambda: GraphNet(2, args.hd, args.L, 4, "GIN")),
        ("DeepSet", lambda: DeepSetBaseline(2, args.hd, 4)),
        ("MLP", lambda: MLPBaseline(args.n_vertices, 2, args.hd, 4)),
    ]
    for label, mfn in models:
        m = mfn().to(args.device)
        n_params = sum(p.numel() for p in m.parameters())
        print(f"\n  {label} ({n_params:,} params)")
        best, final = train_model(
            m, tr_c, tr_y, te_c, te_y, topo_dev,
            args.epochs, args.lr, "cls",
        )
        print(f"  => Final: {final:.3f}  Best: {best:.3f}")


def run_ect(args):
    """Table 2: ECT vector regression."""
    print("\n" + "=" * 60)
    print("ECT Regression (80-dim target)")
    print("=" * 60)
    coords, ect, topo = generate_ect_data(
        args.n_samples, args.n_vertices, args.seed
    )
    out_dim = ect.shape[1]
    tr_c, tr_y, te_c, te_y = split_data(coords, ect, args.seed)
    topo_dev = topo_to_device(topo, args.device)
    ep = int(args.epochs * 1.5)

    models = [
        ("Combinatorial SMP", lambda: SimplicialNet(2, args.hd, args.L, out_dim, "combinatorial")),
        ("Coord SMP (summed)", lambda: SimplicialNet(2, args.hd, args.L, out_dim, "coord_summed")),
        ("MLP", lambda: MLPBaseline(args.n_vertices, 2, args.hd, out_dim)),
    ]
    for label, mfn in models:
        m = mfn().to(args.device)
        print(f"\n  {label}")
        best, final = train_model(m, tr_c, tr_y, te_c, te_y, topo_dev, ep, args.lr, "reg")
        print(f"  => Final: {final:.4f}  Best: {best:.4f}")

    # Depth ablation
    print("\n  Depth ablation (Coord SMP summed):")
    for L in [1, 2, 3, 4, 6]:
        m = SimplicialNet(2, args.hd, L, out_dim, "coord_summed").to(args.device)
        best, _ = train_model(m, tr_c, tr_y, te_c, te_y, topo_dev, ep, args.lr, "reg", verbose=False)
        print(f"    L={L}: {best:.4f}")


def run_coboundary(args):
    """Table 5: Per-vertex curvature, coboundary depth sweep."""
    print("\n" + "=" * 60)
    print("Coboundary Necessity: Curvature Depth Sweep")
    print("=" * 60)
    coords, curv, topo = generate_curvature_data(400, args.n_vertices, args.seed)
    tr_c, tr_y, te_c, te_y = split_data(coords, curv, args.seed)
    topo_dev = topo_to_device(topo, args.device)
    ep = int(args.epochs * 1.5)

    print(f"\n  {'Depth':>5s}  {'Full':>8s}  {'Bdy-only':>8s}  {'Gap':>8s}")
    print("  " + "-" * 35)
    for L in [1, 2, 4, 8]:
        m_full = SimplicialNetPerVertex(2, args.hd, L, 1, "coord_rich", boundary_only=False).to(args.device)
        best_full, _ = train_model(m_full, tr_c, tr_y, te_c, te_y, topo_dev, ep, args.lr, "reg", verbose=False)

        m_bdy = SimplicialNetPerVertex(2, args.hd, L, 1, "coord_rich", boundary_only=True).to(args.device)
        best_bdy, _ = train_model(m_bdy, tr_c, tr_y, te_c, te_y, topo_dev, ep, args.lr, "reg", verbose=False)

        gap = best_bdy - best_full
        print(f"  {L:5d}  {best_full:8.3f}  {best_bdy:8.3f}  {gap:+8.3f}")


def run_permutation(args):
    """Table 6: Permutation equivariance test."""
    print("\n" + "=" * 60)
    print("Permutation Equivariance Test")
    print("=" * 60)
    nv = 50
    coords, labels, topo, tri, base = generate_family_data(400, nv, args.seed)
    tr_c, tr_y, te_c, te_y = split_data(coords, labels, args.seed)
    topo_dev = topo_to_device(topo, args.device)

    # Permuted test set
    rng = np.random.RandomState(args.seed + 1)
    te_c_perm = te_c.clone()
    for i in range(len(te_c_perm)):
        perm = rng.permutation(nv)
        te_c_perm[i] = te_c[i, perm]

    for label, mfn in [
        ("Coord SMP (rich)", lambda: SimplicialNet(2, args.hd, args.L, 4, "coord_rich")),
        ("MLP", lambda: MLPBaseline(nv, 2, args.hd, 4)),
    ]:
        m = mfn().to(args.device)
        print(f"\n  {label}")
        train_model(m, tr_c, tr_y, te_c, te_y, topo_dev, args.epochs, args.lr, "cls")

        # Evaluate on permuted test set
        m.eval()
        with torch.no_grad():
            pred = m(te_c_perm.to(args.device), topo_dev)
            acc_perm = (pred.argmax(-1) == te_y.to(args.device)).float().mean().item()
        pred_orig = m(te_c.to(args.device), topo_dev)
        acc_orig = (pred_orig.argmax(-1) == te_y.to(args.device)).float().mean().item()
        print(f"  => Unpermuted: {acc_orig:.3f}  Permuted: {acc_perm:.3f}")


def run_confounded(args):
    """Table 4: Geometric summary regression."""
    print("\n" + "=" * 60)
    print("Geometric Summary Regression (30-dim)")
    print("=" * 60)
    coords, stats, topo = generate_confounded_data(500, args.n_vertices, args.seed)
    out_dim = stats.shape[1]
    tr_c, tr_y, te_c, te_y = split_data(coords, stats, args.seed)
    topo_dev = topo_to_device(topo, args.device)
    ep = int(args.epochs * 1.5)

    for label, mfn in [
        ("Combinatorial SMP", lambda: SimplicialNet(2, args.hd, args.L, out_dim, "combinatorial")),
        ("Coord SMP (summed)", lambda: SimplicialNet(2, args.hd, args.L, out_dim, "coord_summed")),
        ("GCN + coords", lambda: GraphNet(2, args.hd, args.L, out_dim, "GCN")),
        ("GIN + coords", lambda: GraphNet(2, args.hd, args.L, out_dim, "GIN")),
        ("DeepSet", lambda: DeepSetBaseline(2, args.hd, out_dim)),
    ]:
        m = mfn().to(args.device)
        print(f"\n  {label}")
        best, final = train_model(m, tr_c, tr_y, te_c, te_y, topo_dev, ep, args.lr, "reg")
        print(f"  => Final: {final:.4f}  Best: {best:.4f}")


def run_mantra(args):
    """Table 9: Deformation classification on MANTRA."""
    print("\n" + "=" * 60)
    print("MANTRA Deformation Classification (chance = 0.25)")
    print("=" * 60)
    from mantra.datasets import ManifoldTriangulations
    dataset = ManifoldTriangulations(root="./data", dimension=2, version="latest")
    rng = np.random.RandomState(args.seed)

    by_name = {}
    for i in range(len(dataset)):
        d = dataset[i]
        name = str(d.name)
        nv = int(d.n_vertices)
        tri_raw = d.triangulation
        tri = tri_raw.numpy().tolist() if isinstance(tri_raw, torch.Tensor) else list(tri_raw)
        flat = [v for t in tri for v in t]
        if min(flat) == 1:
            tri = [[v - 1 for v in t] for t in tri]
        by_name.setdefault(name, []).append((nv, tri))

    for topo_name in ["S^2", "T^2", "Klein bottle", "RP^2"]:
        if topo_name not in by_name or len(by_name[topo_name]) < 3:
            continue
        best_sample = max(by_name[topo_name], key=lambda x: x[0])
        nv, triangles = best_sample
        topo = build_topology(nv, triangles)
        base = embed_triangulation_2d(nv, triangles, rng)

        all_c, all_l = [], []
        for fi, fam in enumerate(FAMILIES):
            for _ in range(50):
                s = rng.uniform(0.3, 1.5)
                c = apply_deformation(base, fam, s, rng)
                all_c.append(c)
                all_l.append(fi)

        coords_t = torch.from_numpy(np.stack(all_c))
        labels_t = torch.tensor(all_l, dtype=torch.long)
        tr_c, tr_y, te_c, te_y = split_data(coords_t, labels_t, args.seed)
        topo_dev = topo_to_device(topo, args.device)

        print(f"\n  --- {topo_name} (V={nv}) ---")
        for label, mfn in [
            ("Combinatorial", lambda: SimplicialNet(2, args.hd, args.L, 4, "combinatorial")),
            ("Coord SMP", lambda: SimplicialNet(2, args.hd, args.L, 4, "coord_summed")),
            ("DeepSet", lambda: DeepSetBaseline(2, args.hd, 4)),
        ]:
            m = mfn().to(args.device)
            best, final = train_model(
                m, tr_c, tr_y, te_c, te_y, topo_dev,
                args.epochs, args.lr, "cls", verbose=False,
            )
            print(f"    {label:20s}: {final:.3f}")


def run_mantra_ect(args):
    """Table 10: ECT regression on MANTRA."""
    print("\n" + "=" * 60)
    print("MANTRA ECT Regression")
    print("=" * 60)
    from mantra.datasets import ManifoldTriangulations
    dataset = ManifoldTriangulations(root="./data", dimension=2, version="latest")
    rng = np.random.RandomState(args.seed)
    dirs = sample_directions_2d(8)
    threshs = np.linspace(-2.5, 2.5, 10)

    by_name = {}
    for i in range(len(dataset)):
        d = dataset[i]
        name = str(d.name)
        nv = int(d.n_vertices)
        tri_raw = d.triangulation
        tri = tri_raw.numpy().tolist() if isinstance(tri_raw, torch.Tensor) else list(tri_raw)
        flat = [v for t in tri for v in t]
        if min(flat) == 1:
            tri = [[v - 1 for v in t] for t in tri]
        by_name.setdefault(name, []).append((nv, tri))

    for topo_name in ["S^2", "T^2"]:
        if topo_name not in by_name:
            continue
        best_sample = max(by_name[topo_name], key=lambda x: x[0])
        nv, triangles = best_sample
        topo = build_topology(nv, triangles)
        edges = topo["edges"].numpy()
        base = embed_triangulation_2d(nv, triangles, rng)

        all_c, all_e = [], []
        for _ in range(200):
            fam = FAMILIES[rng.randint(4)]
            s = rng.uniform(0.3, 1.5)
            c = apply_deformation(base, fam, s, rng)
            e = compute_ect(c, triangles, edges, dirs, threshs)
            all_c.append(c)
            all_e.append(e.flatten())

        ct = torch.from_numpy(np.stack(all_c))
        et = torch.from_numpy(np.stack(all_e))
        mu, std = et.mean(0), et.std(0).clamp(min=1e-6)
        et = (et - mu) / std
        out_dim = et.shape[1]
        tr_c, tr_y, te_c, te_y = split_data(ct, et, args.seed)
        topo_dev = topo_to_device(topo, args.device)
        ep = int(args.epochs * 1.5)

        print(f"\n  --- {topo_name} (V={nv}) ---")
        for label, mfn in [
            ("Combinatorial", lambda: SimplicialNet(2, args.hd, args.L, out_dim, "combinatorial")),
            ("Coord SMP", lambda: SimplicialNet(2, args.hd, args.L, out_dim, "coord_summed")),
        ]:
            m = mfn().to(args.device)
            best, final = train_model(
                m, tr_c, tr_y, te_c, te_y, topo_dev,
                ep, args.lr, "reg", verbose=False,
            )
            print(f"    {label:20s}: {final:.4f}")


def run_faust_pose(args):
    """Table 3: FAUST 10-way pose classification with 5-fold CV."""
    print("\n" + "=" * 60)
    print("FAUST 10-Way Pose Classification (5-fold CV)")
    print("=" * 60)
    from torch_geometric.datasets import FAUST
    dataset = FAUST(root="./data/FAUST")

    all_coords, all_faces = [], None
    pose_labels = []
    for i in range(len(dataset)):
        d = dataset[i]
        all_coords.append(d.pos)
        if all_faces is None:
            all_faces = d.face.T.numpy()
        pose_labels.append(i % 10)

    nv = all_coords[0].shape[0]
    topo = build_topology(nv, all_faces.tolist())
    coords_t = torch.stack(all_coords)
    pose_t = torch.tensor(pose_labels, dtype=torch.long)

    # Per-sample centering and global scaling
    coords_t = coords_t - coords_t.mean(dim=1, keepdim=True)
    coords_t = coords_t / coords_t.std()

    hd = min(args.hd, 16)
    L = min(args.L, 3)
    ep = max(args.epochs, 200)
    n_folds, n_aug = 5, 5
    rng = np.random.RandomState(args.seed)

    print(f"  {nv} vertices, {topo['n_edges']} edges, {topo['n_triangles']} triangles")
    print(f"  {len(coords_t)} samples, {n_folds}-fold CV, {n_aug}x augmentation\n")

    topo_dev = topo_to_device(topo, args.device)

    for label, mfn in [
        ("Combinatorial SMP", lambda: SimplicialNet(3, hd, L, 10, "combinatorial")),
        ("Coord SMP (summed)", lambda: SimplicialNet(3, hd, L, 10, "coord_summed")),
        ("GCN + coords", lambda: GraphNet(3, hd, L, 10, "GCN")),
        ("GIN + coords", lambda: GraphNet(3, hd, L, 10, "GIN")),
        ("DeepSet", lambda: DeepSetBaseline(3, hd * 2, 10)),
    ]:
        print(f"  {label}")
        mu, std = kfold_train_eval(
            mfn, coords_t, pose_t, topo_dev, n_folds, ep, args.lr,
            task="cls", device=args.device, n_aug=n_aug, rng=rng,
        )
        print(f"  => {mu:.3f} +/- {std:.3f}  (chance = 0.10)\n")


def run_faust_reg(args):
    """Table 11: FAUST geometric regression with 5-fold CV."""
    print("\n" + "=" * 60)
    print("FAUST Geometric Regression (5-fold CV)")
    print("=" * 60)
    from torch_geometric.datasets import FAUST
    from scipy.spatial.distance import pdist
    dataset = FAUST(root="./data/FAUST")

    all_coords, all_faces = [], None
    for i in range(len(dataset)):
        d = dataset[i]
        all_coords.append(d.pos)
        if all_faces is None:
            all_faces = d.face.T.numpy()

    nv = all_coords[0].shape[0]
    topo = build_topology(nv, all_faces.tolist())
    coords_raw = torch.stack(all_coords)

    # Rotation-invariant targets
    geo_targets = []
    for i in range(len(all_coords)):
        c = all_coords[i].numpy()
        sub = c[np.random.choice(len(c), 500, replace=False)] if len(c) > 500 else c
        extent = pdist(sub).max()
        centroid = c.mean(axis=0)
        mean_dist = np.linalg.norm(c - centroid, axis=1).mean()
        var_dist = np.linalg.norm(c - centroid, axis=1).var()
        geo_targets.append([extent, mean_dist, var_dist])

    geo_t = torch.tensor(geo_targets, dtype=torch.float32)
    mu_g, std_g = geo_t.mean(0), geo_t.std(0).clamp(min=1e-6)
    geo_t = (geo_t - mu_g) / std_g

    # Normalize coordinates
    coords_t = coords_raw - coords_raw.mean(dim=1, keepdim=True)
    coords_t = coords_t / coords_t.std()

    hd = min(args.hd, 16)
    L = min(args.L, 3)
    ep = max(int(args.epochs * 1.5), 300)
    n_folds, n_aug = 5, 5
    rng = np.random.RandomState(args.seed)
    topo_dev = topo_to_device(topo, args.device)

    for label, mfn in [
        ("Combinatorial SMP", lambda: SimplicialNet(3, hd, L, 3, "combinatorial")),
        ("Coord SMP (summed)", lambda: SimplicialNet(3, hd, L, 3, "coord_summed")),
        ("DeepSet", lambda: DeepSetBaseline(3, hd * 2, 3)),
    ]:
        print(f"\n  {label}")
        mu, std = kfold_train_eval(
            mfn, coords_t, geo_t, topo_dev, n_folds, ep, args.lr,
            task="reg", device=args.device, n_aug=n_aug, rng=rng,
        )
        print(f"  => {mu:.4f} +/- {std:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    "family": run_family,
    "ect": run_ect,
    "coboundary": run_coboundary,
    "permutation": run_permutation,
    "confounded": run_confounded,
    "mantra": run_mantra,
    "mantra_ect": run_mantra_ect,
    "faust_pose": run_faust_pose,
    "faust_reg": run_faust_reg,
}


def main():
    p = argparse.ArgumentParser(description="Reproduce paper experiments.")
    p.add_argument("--exp", type=str, required=True,
                   choices=list(EXPERIMENTS.keys()) + ["all"])
    p.add_argument("--n_samples", type=int, default=300)
    p.add_argument("--n_vertices", type=int, default=40)
    p.add_argument("--hd", type=int, default=32)
    p.add_argument("--L", type=int, default=4)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.exp == "all":
        for name, fn in EXPERIMENTS.items():
            try:
                fn(args)
            except Exception as e:
                print(f"\nSkipping {name}: {e}\n")
    else:
        EXPERIMENTS[args.exp](args)


if __name__ == "__main__":
    main()
