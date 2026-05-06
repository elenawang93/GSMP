"""Neural network architectures for simplicial and graph message passing."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def batched_scatter_add(src, idx, n_out):
    """Scatter-add src along dim=1 using idx. Handles empty index tensors."""
    if idx.numel() == 0:
        return src.new_zeros(src.shape[0], n_out, src.shape[-1])
    B, _, D = src.shape
    out = src.new_zeros(B, n_out, D)
    out.scatter_add_(1, idx.unsqueeze(0).unsqueeze(-1).expand(B, -1, D), src)
    return out


# ---------------------------------------------------------------------------
# Simplicial message passing
# ---------------------------------------------------------------------------

class SMPLayer(nn.Module):
    """One layer of simplicial message passing with boundary and coboundary
    aggregation across vertices, edges, and triangles.

    Args:
        hd: hidden dimension.
        rich: if True, use pre-summation maps phi(h_src, h_dst) before
              aggregation (richer architecture). Otherwise, project each
              neighbor independently before summing (summed-neighbor).
    """

    def __init__(self, hd, rich=False):
        super().__init__()
        self.hd = hd
        self.rich = rich

        if rich:
            mk = lambda: nn.Sequential(nn.Linear(2 * hd, hd), nn.ReLU())
            self.phi_b1, self.phi_b2 = mk(), mk()
            self.phi_cb0, self.phi_cb1 = mk(), mk()
        else:
            self.proj_b1 = nn.Linear(hd, hd)
            self.proj_b2 = nn.Linear(hd, hd)
            self.proj_cb0 = nn.Linear(hd, hd)
            self.proj_cb1 = nn.Linear(hd, hd)

        mk_up = lambda: nn.Sequential(
            nn.Linear(3 * hd, hd), nn.ReLU(), nn.Linear(hd, hd)
        )
        self.up0, self.up1, self.up2 = mk_up(), mk_up(), mk_up()
        self.ln0 = nn.LayerNorm(hd)
        self.ln1 = nn.LayerNorm(hd)
        self.ln2 = nn.LayerNorm(hd)

    def _agg(self, src, si, di, dst_f, n_dst, phi=None, proj=None):
        if si.numel() == 0:
            return src.new_zeros(src.shape[0], n_dst, self.hd)
        gs = src[:, si]
        if self.rich and phi is not None:
            msgs = phi(torch.cat([gs, dst_f[:, di]], -1))
        else:
            msgs = proj(gs) if proj is not None else gs
        return batched_scatter_add(msgs, di, n_dst)

    def forward(self, h0, h1, h2, topo):
        nv = topo["n_vertices"]
        ne = topo["n_edges"]
        nt = topo["n_triangles"]
        B = h0.shape[0]

        if self.rich:
            ab1 = self._agg(h0, topo["b1_src"], topo["b1_dst"], h1, ne, phi=self.phi_b1)
            ab2 = self._agg(h1, topo["b2_src"], topo["b2_dst"], h2, nt, phi=self.phi_b2)
            acb0 = self._agg(h1, topo["cb0_src"], topo["cb0_dst"], h0, nv, phi=self.phi_cb0)
            acb1 = self._agg(h2, topo["cb1_src"], topo["cb1_dst"], h1, ne, phi=self.phi_cb1)
        else:
            ab1 = self._agg(h0, topo["b1_src"], topo["b1_dst"], None, ne, proj=self.proj_b1)
            ab2 = self._agg(h1, topo["b2_src"], topo["b2_dst"], None, nt, proj=self.proj_b2)
            acb0 = self._agg(h1, topo["cb0_src"], topo["cb0_dst"], None, nv, proj=self.proj_cb0)
            acb1 = self._agg(h2, topo["cb1_src"], topo["cb1_dst"], None, ne, proj=self.proj_cb1)

        z_v = h0.new_zeros(B, nv, self.hd)
        z_t = h0.new_zeros(B, nt, self.hd)

        h0 = self.ln0(h0 + self.up0(torch.cat([h0, z_v, acb0], -1)))
        h1 = self.ln1(h1 + self.up1(torch.cat([h1, ab1, acb1], -1)))
        h2 = self.ln2(h2 + self.up2(torch.cat([h2, ab2, z_t], -1)))
        return h0, h1, h2


class SMPLayerBoundaryOnly(nn.Module):
    """Ablation: boundary aggregation only (no coboundary channel)."""

    def __init__(self, hd):
        super().__init__()
        self.hd = hd
        self.proj_b1 = nn.Linear(hd, hd)
        self.proj_b2 = nn.Linear(hd, hd)
        mk_up = lambda: nn.Sequential(
            nn.Linear(2 * hd, hd), nn.ReLU(), nn.Linear(hd, hd)
        )
        self.up0, self.up1, self.up2 = mk_up(), mk_up(), mk_up()
        self.ln0 = nn.LayerNorm(hd)
        self.ln1 = nn.LayerNorm(hd)
        self.ln2 = nn.LayerNorm(hd)

    def forward(self, h0, h1, h2, topo):
        nv = topo["n_vertices"]
        ne = topo["n_edges"]
        nt = topo["n_triangles"]
        B = h0.shape[0]

        ab1 = batched_scatter_add(self.proj_b1(h0[:, topo["b1_src"]]),
                                   topo["b1_dst"], ne)
        ab2 = batched_scatter_add(self.proj_b2(h1[:, topo["b2_src"]]),
                                   topo["b2_dst"], nt)

        h0 = self.ln0(h0 + self.up0(torch.cat([h0, h0.new_zeros(B, nv, self.hd)], -1)))
        h1 = self.ln1(h1 + self.up1(torch.cat([h1, ab1], -1)))
        h2 = self.ln2(h2 + self.up2(torch.cat([h2, ab2], -1)))
        return h0, h1, h2


class SimplicialNet(nn.Module):
    """Simplicial message passing network.

    Args:
        coord_dim: dimension of vertex coordinates (2 or 3).
        hd: hidden dimension.
        n_layers: number of SMP layers.
        out_dim: output dimension.
        mode: one of 'combinatorial', 'coord_summed', 'coord_rich'.
        boundary_only: if True, disable coboundary channel (ablation).
    """

    def __init__(self, coord_dim, hd, n_layers, out_dim, mode="coord_summed",
                 boundary_only=False):
        super().__init__()
        self.mode = mode
        self.hd = hd

        use_coords = mode in ("coord_summed", "coord_rich")
        if use_coords:
            self.emb0 = nn.Linear(coord_dim, hd)
            self.emb1 = nn.Linear(coord_dim + 1, hd)    # midpoint + length
            self.emb2 = nn.Linear(coord_dim + 1, hd)    # centroid + area
        else:
            self.emb0 = nn.Linear(1, hd)
            self.emb1 = nn.Linear(1, hd)
            self.emb2 = nn.Linear(1, hd)

        if boundary_only:
            self.layers = nn.ModuleList(
                [SMPLayerBoundaryOnly(hd) for _ in range(n_layers)]
            )
        else:
            rich = mode == "coord_rich"
            self.layers = nn.ModuleList(
                [SMPLayer(hd, rich=rich) for _ in range(n_layers)]
            )

        self.head = nn.Sequential(
            nn.Linear(3 * hd, hd), nn.ReLU(), nn.Linear(hd, out_dim)
        )

    def _init_features(self, coords, topo):
        B = coords.shape[0]
        nv = topo["n_vertices"]
        ne = topo["n_edges"]
        nt = topo["n_triangles"]

        if self.mode in ("coord_summed", "coord_rich"):
            h0 = self.emb0(coords)
            v0 = coords[:, topo["edges"][:, 0]]
            v1 = coords[:, topo["edges"][:, 1]]
            edge_feat = torch.cat([
                (v0 + v1) / 2,
                (v1 - v0).norm(dim=-1, keepdim=True),
            ], dim=-1)
            h1 = self.emb1(edge_feat)

            tv = [coords[:, topo["triangles"][:, i]] for i in range(3)]
            centroid = (tv[0] + tv[1] + tv[2]) / 3
            e1, e2 = tv[1] - tv[0], tv[2] - tv[0]
            if coords.shape[-1] == 3:
                area = 0.5 * torch.cross(e1, e2, dim=-1).norm(dim=-1)
            else:
                area = 0.5 * (e1[..., 0] * e2[..., 1]
                              - e2[..., 0] * e1[..., 1]).abs()
            tri_feat = torch.cat([centroid, area.unsqueeze(-1)], dim=-1)
            h2 = self.emb2(tri_feat)
        else:
            h0 = self.emb0(coords.new_ones(B, nv, 1))
            h1 = self.emb1(coords.new_ones(B, ne, 1))
            h2 = self.emb2(coords.new_ones(B, nt, 1))

        return h0, h1, h2

    def forward(self, coords, topo):
        h0, h1, h2 = self._init_features(coords, topo)
        for layer in self.layers:
            h0, h1, h2 = layer(h0, h1, h2, topo)
        pooled = torch.cat([h0.mean(1), h1.mean(1), h2.mean(1)], dim=-1)
        return self.head(pooled)


class SimplicialNetPerVertex(nn.Module):
    """SimplicialNet variant with per-vertex output (for curvature tasks)."""

    def __init__(self, coord_dim, hd, n_layers, out_dim, mode="coord_summed",
                 boundary_only=False):
        super().__init__()
        self.backbone = SimplicialNet(
            coord_dim, hd, n_layers, out_dim, mode, boundary_only
        )
        self.backbone.head = nn.Identity()
        self.vertex_head = nn.Sequential(
            nn.Linear(hd, hd), nn.ReLU(), nn.Linear(hd, out_dim)
        )

    def forward(self, coords, topo):
        h0, h1, h2 = self.backbone._init_features(coords, topo)
        for layer in self.backbone.layers:
            h0, h1, h2 = layer(h0, h1, h2, topo)
        return self.vertex_head(h0)


# ---------------------------------------------------------------------------
# Graph neural networks (1-skeleton baselines)
# ---------------------------------------------------------------------------

class GCNLayer(nn.Module):
    """GCN layer (Kipf & Welling, 2017)."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, h, edge_index, n_nodes):
        src, dst = edge_index
        deg = torch.zeros(n_nodes, device=h.device)
        deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        deg = deg.clamp(min=1)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
        h_j = h[src] * norm.unsqueeze(-1)
        out = torch.zeros_like(h)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(h_j), h_j)
        return self.linear(out + h)


class GINLayer(nn.Module):
    """GIN layer (Xu et al., 2019)."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
        )
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, h, edge_index, n_nodes):
        src, dst = edge_index
        agg = torch.zeros_like(h)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(h[src]), h[src])
        return self.mlp((1 + self.eps) * h + agg)


class GraphNet(nn.Module):
    """GCN or GIN on the 1-skeleton with vertex coordinates as features.

    Args:
        coord_dim: dimension of vertex coordinates.
        hd: hidden dimension.
        n_layers: number of GNN layers.
        out_dim: output dimension.
        conv_type: 'GCN' or 'GIN'.
    """

    def __init__(self, coord_dim, hd, n_layers, out_dim, conv_type="GCN"):
        super().__init__()
        self.embed = nn.Linear(coord_dim, hd)
        Layer = GCNLayer if conv_type == "GCN" else GINLayer
        self.convs = nn.ModuleList([Layer(hd, hd) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hd) for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.Linear(hd, hd), nn.ReLU(), nn.Linear(hd, out_dim)
        )

    def forward(self, coords, topo):
        B, V, C = coords.shape
        edge_index = topo["edge_index"]
        outputs = []
        for b in range(B):
            h = self.embed(coords[b])
            for conv, norm in zip(self.convs, self.norms):
                h = norm(h + F.relu(conv(h, edge_index, V)))
            outputs.append(h.mean(0))
        return self.head(torch.stack(outputs))


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class DeepSetBaseline(nn.Module):
    """Permutation-invariant MLP on vertex coordinates (sum-pooled)."""

    def __init__(self, coord_dim, hd, out_dim):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(coord_dim, hd), nn.ReLU(),
            nn.Linear(hd, hd), nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(hd, hd), nn.ReLU(), nn.Linear(hd, out_dim)
        )

    def forward(self, coords, topo=None):
        return self.rho(self.phi(coords).mean(1))


class MLPBaseline(nn.Module):
    """MLP on flattened coordinates (not permutation-equivariant)."""

    def __init__(self, n_vertices, coord_dim, hd, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_vertices * coord_dim, hd), nn.ReLU(),
            nn.Linear(hd, hd), nn.ReLU(),
            nn.Linear(hd, out_dim),
        )

    def forward(self, coords, topo=None):
        return self.net(coords.flatten(1))
