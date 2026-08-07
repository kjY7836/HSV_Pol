from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn import radius_graph
from torch_scatter import scatter_max, scatter_mean, scatter_softmax, scatter_sum

from affinityV2.data import NUM_RESIDUE_TYPES, clamp_atom_types
from models.molgnet3d import EMolGNet, rbf_encode_dist
from utils.molecular_utils import NUM_ATOM_TYPES, NUM_BOND_TYPES


def mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    *,
    dropout: float = 0.2,
    final_activation: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    ]
    if final_activation:
        layers.append(nn.SiLU())
    return nn.Sequential(*layers)


def load_unified_encoder_state_dict(ckpt_path: str) -> Dict[str, Tensor]:
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        if isinstance(raw.get("model_state_dict"), dict):
            state = raw["model_state_dict"]
        elif isinstance(raw.get("state_dict"), dict):
            state = raw["state_dict"]
        elif isinstance(raw.get("model"), dict):
            state = raw["model"]
        else:
            state = raw
    else:
        state = raw

    out: Dict[str, Tensor] = {}
    for key, value in state.items():
        if not isinstance(value, Tensor):
            continue
        if key.startswith("encoder."):
            out[key[len("encoder.") :]] = value
        elif key.startswith("module.encoder."):
            out[key[len("module.encoder.") :]] = value
    if not out:
        raise ValueError(f"No encoder.* parameters found in checkpoint: {ckpt_path}")
    return out


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = requires_grad


def set_encoder_last_layers_trainable(encoder: EMolGNet, last_n: int) -> None:
    set_requires_grad(encoder, False)
    if last_n <= 0:
        return
    for layer in encoder.layers[-last_n:]:
        set_requires_grad(layer, True)
    set_requires_grad(encoder.output_proj, True)


def batch_size_from(batch: Optional[Tensor]) -> int:
    if batch is None or batch.numel() == 0:
        return 1
    return int(batch.max().item()) + 1


def safe_scatter_max(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    out = scatter_max(src, index, dim=0, dim_size=dim_size)[0]
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


class BondMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.bond_embedding = nn.Embedding(NUM_BOND_TYPES, hidden_dim)
        self.message = mlp(hidden_dim * 2, hidden_dim, hidden_dim, dropout=dropout)
        self.update = nn.GRUCell(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        if edge_index.numel() == 0:
            return h
        src, dst = edge_index.long()
        bond = self.bond_embedding(edge_attr.long().clamp(min=0, max=NUM_BOND_TYPES - 1))
        messages = self.message(torch.cat([h[src], bond], dim=-1))
        aggregated = scatter_mean(messages, dst, dim=0, dim_size=h.size(0))
        return self.norm(self.update(aggregated, h))


class ResidueMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, rbf_kernels: int, cutoff: float, dropout: float) -> None:
        super().__init__()
        self.rbf_kernels = int(rbf_kernels)
        self.cutoff = float(cutoff)
        self.message = mlp(hidden_dim + rbf_kernels, hidden_dim, hidden_dim, dropout=dropout)
        self.update = nn.GRUCell(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: Tensor, pos: Tensor, edge_index: Tensor) -> Tensor:
        if edge_index.numel() == 0:
            return h
        src, dst = edge_index.long()
        dist = torch.linalg.vector_norm(pos[src] - pos[dst], dim=-1)
        rbf = rbf_encode_dist(
            dist,
            num_kernels=self.rbf_kernels,
            start=0.0,
            stop=self.cutoff,
            std_width=1.5,
        )
        messages = self.message(torch.cat([h[src], rbf], dim=-1))
        aggregated = scatter_mean(messages, dst, dim=0, dim_size=h.size(0))
        return self.norm(self.update(aggregated, h))


class SemanticEncoder(nn.Module):
    """Training-only 2D-ligand/residue-pocket contrastive encoder."""

    def __init__(
        self,
        *,
        hidden_dim: int = 256,
        num_layers: int = 3,
        residue_cutoff: float = 10.0,
        residue_max_neighbors: int = 32,
        residue_rbf_kernels: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.residue_cutoff = float(residue_cutoff)
        self.residue_max_neighbors = int(residue_max_neighbors)

        self.ligand_atom_embedding = nn.Embedding(NUM_ATOM_TYPES, hidden_dim)
        self.ligand_layers = nn.ModuleList(
            [BondMessageLayer(hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.pocket_atom_embedding = nn.Embedding(NUM_ATOM_TYPES, hidden_dim)
        self.residue_type_embedding = nn.Embedding(NUM_RESIDUE_TYPES, hidden_dim)
        self.residue_init = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.residue_layers = nn.ModuleList(
            [
                ResidueMessageLayer(hidden_dim, residue_rbf_kernels, residue_cutoff, dropout)
                for _ in range(num_layers)
            ]
        )
        self.complex_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, 512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
        )
        self.projection_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
        )

    def _encode_ligand(self, ligand: Batch) -> Tensor:
        h = self.ligand_atom_embedding(clamp_atom_types(ligand.x.long()))
        for layer in self.ligand_layers:
            h = layer(h, ligand.edge_index, ligand.edge_attr)
        return h

    def _encode_residues(self, pocket: Batch) -> Tuple[Tensor, Tensor]:
        atom_h = self.pocket_atom_embedding(clamp_atom_types(pocket.x.long()))
        residue_id = pocket.residue_id.long()
        stride = int(residue_id.max().item()) + 1 if residue_id.numel() else 1
        global_residue = pocket.batch.long() * max(1, stride) + residue_id
        _, residue_inverse = torch.unique(global_residue, sorted=True, return_inverse=True)
        n_residues = int(residue_inverse.max().item()) + 1 if residue_inverse.numel() else 0

        atom_mean = scatter_mean(atom_h, residue_inverse, dim=0, dim_size=n_residues)
        atom_max = safe_scatter_max(atom_h, residue_inverse, n_residues)
        residue_type = scatter_mean(
            pocket.residue_type.float().unsqueeze(-1),
            residue_inverse,
            dim=0,
            dim_size=n_residues,
        ).squeeze(-1).round().long().clamp(min=0, max=NUM_RESIDUE_TYPES - 1)
        residue_pos = scatter_mean(pocket.pos, residue_inverse, dim=0, dim_size=n_residues)
        residue_batch = scatter_mean(
            pocket.batch.float().unsqueeze(-1),
            residue_inverse,
            dim=0,
            dim_size=n_residues,
        ).squeeze(-1).round().long()
        h = self.residue_init(
            torch.cat([atom_mean, atom_max, self.residue_type_embedding(residue_type)], dim=-1)
        )
        edge_index = radius_graph(
            residue_pos,
            r=self.residue_cutoff,
            batch=residue_batch,
            max_num_neighbors=self.residue_max_neighbors,
            loop=False,
        )
        for layer in self.residue_layers:
            h = layer(h, residue_pos, edge_index)
        return h, residue_batch

    def forward(self, ligand: Batch, pocket: Batch) -> Dict[str, Tensor]:
        h_ligand = self._encode_ligand(ligand)
        h_residue, residue_batch = self._encode_residues(pocket)
        bsz = batch_size_from(ligand.batch)
        ligand_mean = scatter_mean(h_ligand, ligand.batch, dim=0, dim_size=bsz)
        ligand_max = safe_scatter_max(h_ligand, ligand.batch, bsz)
        residue_mean = scatter_mean(h_residue, residue_batch, dim=0, dim_size=bsz)
        residue_max = safe_scatter_max(h_residue, residue_batch, bsz)
        z_complex = self.complex_mlp(
            torch.cat([ligand_mean, ligand_max, residue_mean, residue_max], dim=-1)
        )
        semantic_z = torch.nn.functional.normalize(self.projection_head(z_complex), dim=-1)
        return {
            "semantic_z": semantic_z,
            "z_complex": z_complex,
        }


class AttnGlobalPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: Tensor, batch: Optional[Tensor], dim_size: int) -> Tensor:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        score = self.score(x).squeeze(-1)
        alpha = scatter_softmax(score.float(), batch, dim=0).to(dtype=x.dtype)
        return scatter_sum(x * alpha.unsqueeze(-1), batch, dim=0, dim_size=dim_size)


class InterfaceAttnPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.node_score = nn.Linear(dim, 1)

    def forward(
        self,
        x: Tensor,
        batch: Tensor,
        interface_score: Tensor,
        dim_size: int,
    ) -> Tensor:
        score = self.node_score(x).squeeze(-1) + interface_score
        alpha = scatter_softmax(score.float(), batch, dim=0).to(dtype=x.dtype)
        return scatter_sum(x * alpha.unsqueeze(-1), batch, dim=0, dim_size=dim_size)


def masked_mean_by_graph(x: Tensor, batch: Tensor, mask: Tensor, dim_size: int) -> Tensor:
    out = torch.zeros(dim_size, x.size(-1), device=x.device, dtype=x.dtype)
    if mask.any():
        out = scatter_mean(x[mask], batch[mask], dim=0, dim_size=dim_size)
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def masked_max_by_graph(x: Tensor, batch: Tensor, mask: Tensor, dim_size: int) -> Tensor:
    out = torch.zeros(dim_size, x.size(-1), device=x.device, dtype=x.dtype)
    if mask.any():
        out = safe_scatter_max(x[mask], batch[mask], dim_size)
    return out


def topk_mean_by_graph(
    values: Tensor,
    scores: Tensor,
    batch: Tensor,
    dim_size: int,
    k: int,
) -> Tensor:
    if values.dim() == 1:
        out = torch.zeros(dim_size, 1, device=values.device, dtype=values.dtype)
        val = values.unsqueeze(-1)
    else:
        out = torch.zeros(dim_size, values.size(-1), device=values.device, dtype=values.dtype)
        val = values
    for b in range(dim_size):
        mask = batch == b
        if not mask.any():
            continue
        take = min(int(k), int(mask.sum().item()))
        _, idx = torch.topk(scores[mask], k=take, largest=True)
        out[b] = val[mask][idx].mean(dim=0)
    return out


def build_cross_pairs_batched(
    pos_lig: Tensor,
    batch_lig: Tensor,
    pos_pocket: Tensor,
    batch_pocket: Tensor,
    *,
    cutoff: float,
    max_neighbors_per_lig: int,
    max_pairs_per_graph: int,
    training: bool,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    device = pos_lig.device
    dtype = pos_lig.dtype
    lig_list: list[Tensor] = []
    poc_list: list[Tensor] = []
    dist_list: list[Tensor] = []
    batch_list: list[Tensor] = []
    valid_list: list[Tensor] = []

    for b in batch_lig.unique(sorted=True):
        lig_global = (batch_lig == b).nonzero(as_tuple=True)[0]
        poc_global = (batch_pocket == b).nonzero(as_tuple=True)[0]
        if lig_global.numel() == 0 or poc_global.numel() == 0:
            continue
        d = torch.cdist(pos_lig[lig_global], pos_pocket[poc_global], p=2)
        nl, npocket = int(d.size(0)), int(d.size(1))
        k = min(max_neighbors_per_lig, npocket)
        selected_l: list[Tensor] = []
        selected_p: list[Tensor] = []
        selected_d: list[Tensor] = []
        selected_v: list[Tensor] = []

        for local_l in range(nl):
            row = d[local_l]
            near = (row < cutoff).nonzero(as_tuple=True)[0]
            if near.numel() > 0:
                near_dist = row[near]
                order = torch.argsort(near_dist)[:k]
                chosen = near[order]
                valid = torch.ones(chosen.numel(), dtype=torch.bool, device=device)
            else:
                chosen = torch.argmin(row).view(1)
                valid = torch.zeros(1, dtype=torch.bool, device=device)
            selected_l.append(lig_global[local_l].repeat(chosen.numel()))
            selected_p.append(poc_global[chosen])
            selected_d.append(row[chosen])
            selected_v.append(valid)

        gl = torch.cat(selected_l, dim=0)
        gp = torch.cat(selected_p, dim=0)
        gd = torch.cat(selected_d, dim=0)
        gv = torch.cat(selected_v, dim=0)
        if gl.numel() > max_pairs_per_graph:
            order_key = gd + (~gv).to(gd.dtype) * (cutoff + 1.0)
            perm = torch.argsort(order_key)[:max_pairs_per_graph]
            gl, gp, gd, gv = gl[perm], gp[perm], gd[perm], gv[perm]
        lig_list.append(gl)
        poc_list.append(gp)
        dist_list.append(gd.to(dtype=dtype))
        valid_list.append(gv)
        batch_list.append(torch.full((gl.numel(),), int(b.item()), dtype=torch.long, device=device))

    if not lig_list:
        z = torch.empty(0, dtype=torch.long, device=device)
        e = torch.empty(0, dtype=dtype, device=device)
        return z, z, e, z, torch.empty(0, dtype=torch.bool, device=device)
    return (
        torch.cat(lig_list, dim=0),
        torch.cat(poc_list, dim=0),
        torch.cat(dist_list, dim=0),
        torch.cat(batch_list, dim=0),
        torch.cat(valid_list, dim=0),
    )


class CrossInterfacePairBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        pair_dim: int,
        pair_rbf_kernels: int,
        atom_pair_emb_dim: int,
        residue_emb_dim: int,
        heads: int,
        dropout: float,
        interface_cutoff: float,
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        self.hidden_dim = hidden_dim
        self.pair_dim = pair_dim
        self.heads = heads
        self.dk = hidden_dim // heads
        self.interface_cutoff = float(interface_cutoff)
        self.pair_rbf_kernels = int(pair_rbf_kernels)
        self.lig_pair_atom_emb = nn.Embedding(NUM_ATOM_TYPES, atom_pair_emb_dim)
        self.poc_pair_atom_emb = nn.Embedding(NUM_ATOM_TYPES, atom_pair_emb_dim)
        self.poc_pair_res_emb = nn.Embedding(NUM_RESIDUE_TYPES, residue_emb_dim)

        pair_init_dim = hidden_dim * 4 + pair_rbf_kernels + atom_pair_emb_dim * 2 + residue_emb_dim
        self.pair_init = mlp(pair_init_dim, hidden_dim, pair_dim, dropout=dropout, final_activation=True)
        self.pair_update = mlp(pair_dim + hidden_dim * 2 + pair_rbf_kernels, hidden_dim, pair_dim, dropout=dropout)
        self.pair_norm = nn.LayerNorm(pair_dim)

        self.score_l = nn.Linear(pair_dim, heads)
        self.score_p = nn.Linear(pair_dim, heads)
        self.msg_l = nn.Linear(pair_dim, hidden_dim)
        self.msg_p = nn.Linear(pair_dim, hidden_dim)
        self.out_l = nn.Linear(hidden_dim, hidden_dim)
        self.out_p = nn.Linear(hidden_dim, hidden_dim)
        self.gru_l = nn.GRUCell(hidden_dim, hidden_dim)
        self.gru_p = nn.GRUCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def rbf(self, dist: Tensor) -> Tensor:
        return rbf_encode_dist(
            dist,
            num_kernels=self.pair_rbf_kernels,
            start=0.0,
            stop=self.interface_cutoff,
            std_width=1.5,
        )

    def _init_pair(
        self,
        h_l: Tensor,
        h_p: Tensor,
        lig_x: Tensor,
        poc_x: Tensor,
        poc_res: Tensor,
        idx_l: Tensor,
        idx_p: Tensor,
        dist: Tensor,
    ) -> Tensor:
        hi = h_l[idx_l]
        hj = h_p[idx_p]
        u = torch.cat(
            [
                hi,
                hj,
                hi * hj,
                torch.abs(hi - hj),
                self.rbf(dist),
                self.lig_pair_atom_emb(lig_x[idx_l]),
                self.poc_pair_atom_emb(poc_x[idx_p]),
                self.poc_pair_res_emb(poc_res[idx_p]),
            ],
            dim=-1,
        )
        return self.pair_init(u)

    def forward(
        self,
        h_l: Tensor,
        h_p: Tensor,
        lig_x: Tensor,
        poc_x: Tensor,
        poc_res: Tensor,
        idx_l: Tensor,
        idx_p: Tensor,
        dist: Tensor,
        e_pair: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if idx_l.numel() == 0:
            return h_l, h_p, h_l.new_zeros(0, self.pair_dim)

        if e_pair is None:
            e_pair = self._init_pair(h_l, h_p, lig_x, poc_x, poc_res, idx_l, idx_p, dist)

        pair_delta = self.pair_update(torch.cat([e_pair, h_l[idx_l], h_p[idx_p], self.rbf(dist)], dim=-1))
        e_pair = self.pair_norm(e_pair + pair_delta)

        msg_l = self.msg_l(e_pair).view(-1, self.heads, self.dk)
        alpha_l = scatter_softmax(self.score_l(e_pair).float(), idx_l, dim=0).to(dtype=h_l.dtype)
        delta_l = scatter_sum(
            (alpha_l.unsqueeze(-1) * msg_l).reshape(-1, self.hidden_dim),
            idx_l,
            dim=0,
            dim_size=h_l.size(0),
        )
        delta_l = self.dropout(self.out_l(delta_l))

        msg_p = self.msg_p(e_pair).view(-1, self.heads, self.dk)
        alpha_p = scatter_softmax(self.score_p(e_pair).float(), idx_p, dim=0).to(dtype=h_p.dtype)
        delta_p = scatter_sum(
            (alpha_p.unsqueeze(-1) * msg_p).reshape(-1, self.hidden_dim),
            idx_p,
            dim=0,
            dim_size=h_p.size(0),
        )
        delta_p = self.dropout(self.out_p(delta_p))

        h_l = self.gru_l(delta_l, h_l)
        h_p = self.gru_p(delta_p, h_p)
        return h_l, h_p, e_pair


class PairScoreHead(nn.Module):
    def __init__(
        self,
        pair_dim: int,
        pair_rbf_kernels: int,
        atom_pair_emb_dim: int,
        residue_emb_dim: int,
        interface_cutoff: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.interface_cutoff = float(interface_cutoff)
        self.pair_rbf_kernels = int(pair_rbf_kernels)
        self.lig_atom_emb = nn.Embedding(NUM_ATOM_TYPES, atom_pair_emb_dim)
        self.poc_atom_emb = nn.Embedding(NUM_ATOM_TYPES, atom_pair_emb_dim)
        self.poc_res_emb = nn.Embedding(NUM_RESIDUE_TYPES, residue_emb_dim)
        in_dim = pair_dim + pair_rbf_kernels + atom_pair_emb_dim * 2 + residue_emb_dim
        self.gate = mlp(in_dim, pair_dim, 1, dropout=dropout)
        self.score = mlp(in_dim, pair_dim, 1, dropout=dropout)

    def forward(
        self,
        e_pair: Tensor,
        dist: Tensor,
        lig_x: Tensor,
        poc_x: Tensor,
        poc_res: Tensor,
        idx_l: Tensor,
        idx_p: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if e_pair.numel() == 0:
            z = e_pair.new_zeros(0)
            return z, z, z
        rbf = rbf_encode_dist(
            dist,
            num_kernels=self.pair_rbf_kernels,
            start=0.0,
            stop=self.interface_cutoff,
            std_width=1.5,
        )
        u = torch.cat(
            [
                e_pair,
                rbf,
                self.lig_atom_emb(lig_x[idx_l]),
                self.poc_atom_emb(poc_x[idx_p]),
                self.poc_res_emb(poc_res[idx_p]),
            ],
            dim=-1,
        )
        gate = torch.sigmoid(self.gate(u)).squeeze(-1)
        score = self.score(u).squeeze(-1)
        return gate, score, gate * score


class AffinityScorerModel(nn.Module):
    def __init__(
        self,
        *,
        num_atom_types: int = NUM_ATOM_TYPES,
        num_residue_types: int = NUM_RESIDUE_TYPES,
        hidden_dim: int = 512,
        num_layers: int = 6,
        heads: int = 8,
        ligand_r_cut: float = 6.0,
        pocket_r_cut: float = 4.5,
        intra_max_neighbors: int = 32,
        rbf_kernels: int = 96,
        pair_dim: int = 256,
        pair_rbf_kernels: int = 32,
        interface_cutoff: float = 8.0,
        max_neighbors_per_lig: int = 32,
        max_pairs_per_graph: int = 4096,
        interface_topk: int = 8,
        atom_pair_emb_dim: int = 64,
        residue_emb_dim: int = 64,
        dropout: float = 0.2,
        y_mean: float = 0.0,
        y_std: float = 1.0,
    ) -> None:
        super().__init__()
        del num_residue_types
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.ligand_r_cut = float(ligand_r_cut)
        self.pocket_r_cut = float(pocket_r_cut)
        self.intra_max_neighbors = int(intra_max_neighbors)
        self.interface_cutoff = float(interface_cutoff)
        self.max_neighbors_per_lig = int(max_neighbors_per_lig)
        self.max_pairs_per_graph = int(max_pairs_per_graph)
        self.interface_topk = int(interface_topk)
        self.pair_dim = int(pair_dim)
        self.register_buffer("y_mean", torch.tensor(float(y_mean), dtype=torch.float32))
        self.register_buffer("y_std", torch.tensor(float(y_std), dtype=torch.float32))

        self.encoder_lig = EMolGNet(
            num_node_types=num_atom_types,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            rbf_kernels=rbf_kernels,
            rbf_start=0.0,
            rbf_stop=ligand_r_cut,
            update_coords=False,
            output_dim=hidden_dim,
            dropout=dropout,
        )
        self.encoder_pocket = EMolGNet(
            num_node_types=num_atom_types,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            rbf_kernels=rbf_kernels,
            rbf_start=0.0,
            rbf_stop=pocket_r_cut,
            update_coords=False,
            output_dim=hidden_dim,
            dropout=dropout,
        )
        self.pocket_residue_embedding = nn.Embedding(NUM_RESIDUE_TYPES, hidden_dim)
        self.pocket_init_norm = nn.LayerNorm(hidden_dim)

        self.cross_blocks = nn.ModuleList(
            [
                CrossInterfacePairBlock(
                    hidden_dim=hidden_dim,
                    pair_dim=pair_dim,
                    pair_rbf_kernels=pair_rbf_kernels,
                    atom_pair_emb_dim=atom_pair_emb_dim,
                    residue_emb_dim=residue_emb_dim,
                    heads=heads,
                    dropout=dropout,
                    interface_cutoff=interface_cutoff,
                )
                for _ in range(num_layers)
            ]
        )
        self.pair_score_head = PairScoreHead(
            pair_dim=pair_dim,
            pair_rbf_kernels=pair_rbf_kernels,
            atom_pair_emb_dim=atom_pair_emb_dim,
            residue_emb_dim=residue_emb_dim,
            interface_cutoff=interface_cutoff,
            dropout=dropout,
        )

        self.pool_lig = AttnGlobalPool(hidden_dim)
        self.pool_pocket = InterfaceAttnPool(hidden_dim)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
        )
        pair_pool_dim = pair_dim * 5 + 4
        self.pair_attn = nn.Linear(pair_dim, 1)
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_pool_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
        )
        self.complex_mlp = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
        )
        self.affinity_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )
        self.contrastive_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
        )

    def set_label_stats(self, y_mean: float, y_std: float) -> None:
        self.y_mean.fill_(float(y_mean))
        self.y_std.fill_(float(y_std) if abs(float(y_std)) > 1e-6 else 1.0)

    def load_towers_from_unified_checkpoint(self, ckpt_path: str, strict: bool = False) -> Dict[str, Any]:
        state = load_unified_encoder_state_dict(ckpt_path)
        lig_info = self.encoder_lig.load_state_dict(state, strict=strict)
        poc_info = self.encoder_pocket.load_state_dict(state, strict=strict)
        return {
            "ligand_missing": list(lig_info.missing_keys),
            "ligand_unexpected": list(lig_info.unexpected_keys),
            "pocket_missing": list(poc_info.missing_keys),
            "pocket_unexpected": list(poc_info.unexpected_keys),
        }

    def set_training_stage(self, stage: str, *, unfreeze_all: bool = False) -> None:
        stage_norm = stage.strip().lower()
        set_requires_grad(self, True)
        if stage_norm in {"stage1", "readout", "warmup"}:
            set_requires_grad(self.encoder_lig, bool(unfreeze_all))
            set_requires_grad(self.encoder_pocket, bool(unfreeze_all))
            set_requires_grad(self.contrastive_head, False)
        elif stage_norm in {"stage2", "stage2a", "stage2b", "cross", "conditional"}:
            if unfreeze_all:
                set_requires_grad(self.encoder_lig, True)
                set_requires_grad(self.encoder_pocket, True)
            else:
                set_encoder_last_layers_trainable(self.encoder_lig, 2)
                set_encoder_last_layers_trainable(self.encoder_pocket, 2)
        elif stage_norm in {"stage3", "finetune"}:
            if unfreeze_all:
                set_requires_grad(self.encoder_lig, True)
                set_requires_grad(self.encoder_pocket, True)
            else:
                set_encoder_last_layers_trainable(self.encoder_lig, 2)
                set_encoder_last_layers_trainable(self.encoder_pocket, 2)
            set_requires_grad(self.contrastive_head, False)
        elif stage_norm in {"calibration", "calibrate"}:
            if unfreeze_all:
                set_requires_grad(self.encoder_lig, True)
                set_requires_grad(self.encoder_pocket, True)
            else:
                set_encoder_last_layers_trainable(self.encoder_lig, 2)
                set_encoder_last_layers_trainable(self.encoder_pocket, 2)
            set_requires_grad(self.contrastive_head, False)
        else:
            raise ValueError(f"Unknown training stage: {stage}")

    def project_complex(self, z_complex: Tensor) -> Tensor:
        return torch.nn.functional.normalize(self.contrastive_head(z_complex), dim=-1)

    def _center(self, lig: Batch, pocket: Batch) -> Tuple[Tensor, Tensor]:
        bsz = batch_size_from(lig.batch)
        center = scatter_mean(pocket.pos, pocket.batch, dim=0, dim_size=bsz)
        return lig.pos - center[lig.batch], pocket.pos - center[pocket.batch]

    def _intra_edges(self, pos: Tensor, batch: Tensor, cutoff: float) -> Tensor:
        return radius_graph(
            pos,
            r=cutoff,
            batch=batch,
            max_num_neighbors=self.intra_max_neighbors,
            loop=False,
        )

    def _encode_interleaved(
        self,
        lig: Batch,
        pocket: Batch,
    ) -> Dict[str, Tensor]:
        lig_x = clamp_atom_types(lig.x.long())
        pocket_x = clamp_atom_types(pocket.x.long())
        residue_type = pocket.residue_type.long().clamp(min=0, max=NUM_RESIDUE_TYPES - 1)
        x_lig, x_pocket = self._center(lig, pocket)

        h_l = self.encoder_lig.node_embedding(lig_x)
        h_p = self.pocket_init_norm(
            self.encoder_pocket.node_embedding(pocket_x)
            + self.pocket_residue_embedding(residue_type)
        )

        e_lig: Optional[Tensor] = None
        e_pocket: Optional[Tensor] = None
        e_cross: Optional[Tensor] = None
        idx_l, idx_p, dist, bpair, valid_pair = build_cross_pairs_batched(
            x_lig,
            lig.batch,
            x_pocket,
            pocket.batch,
            cutoff=self.interface_cutoff,
            max_neighbors_per_lig=self.max_neighbors_per_lig,
            max_pairs_per_graph=self.max_pairs_per_graph,
            training=self.training,
        )

        eidx_l = self._intra_edges(x_lig, lig.batch, self.ligand_r_cut)
        eidx_p = self._intra_edges(x_pocket, pocket.batch, self.pocket_r_cut)

        for layer_idx in range(self.num_layers):
            h_l, _, e_lig = self.encoder_lig.layers[layer_idx](
                h_l,
                x_lig,
                eidx_l,
                None,
                e_lig,
                node_types=lig_x,
            )
            h_p, _, e_pocket = self.encoder_pocket.layers[layer_idx](
                h_p,
                x_pocket,
                eidx_p,
                None,
                e_pocket,
                node_types=pocket_x,
            )
            h_l, h_p, e_cross = self.cross_blocks[layer_idx](
                h_l,
                h_p,
                lig_x,
                pocket_x,
                residue_type,
                idx_l,
                idx_p,
                dist,
                e_cross,
            )

        h_l = self.encoder_lig.output_proj(h_l)
        h_p = self.encoder_pocket.output_proj(h_p)
        return {
            "h_lig": h_l,
            "h_pocket": h_p,
            "idx_l": idx_l,
            "idx_p": idx_p,
            "dist": dist,
            "pair_batch": bpair,
            "valid_pair": valid_pair,
            "e_pair": e_cross if e_cross is not None else h_l.new_zeros(0, self.pair_dim),
            "lig_x": lig_x,
            "pocket_x": pocket_x,
            "pocket_residue_type": residue_type,
            "x_lig_centered": x_lig,
            "x_pocket_centered": x_pocket,
        }

    def _readout(
        self,
        h_l: Tensor,
        h_p: Tensor,
        lig_batch: Tensor,
        pocket_batch: Tensor,
        idx_l: Tensor,
        idx_p: Tensor,
        dist: Tensor,
        pair_batch: Tensor,
        valid_pair: Tensor,
        e_pair: Tensor,
        lig_x: Tensor,
        pocket_x: Tensor,
        pocket_residue_type: Tensor,
    ) -> Dict[str, Tensor]:
        bsz = batch_size_from(lig_batch)
        gate, score, scalar = self.pair_score_head(
            e_pair,
            dist,
            lig_x,
            pocket_x,
            pocket_residue_type,
            idx_l,
            idx_p,
        )
        valid_gate = gate.masked_fill(~valid_pair, 0.0) if gate.numel() else gate
        pocket_interface_score = torch.zeros(h_p.size(0), device=h_p.device, dtype=h_p.dtype)
        if idx_p.numel() > 0 and valid_pair.any():
            max_gate = safe_scatter_max(valid_gate[valid_pair].unsqueeze(-1), idx_p[valid_pair], h_p.size(0)).squeeze(-1)
            pocket_interface_score = max_gate.to(dtype=h_p.dtype)

        g_lig_attn = self.pool_lig(h_l, lig_batch, bsz)
        g_lig_mean = scatter_mean(h_l, lig_batch, dim=0, dim_size=bsz)
        g_lig_max = safe_scatter_max(h_l, lig_batch, bsz)
        g_poc_if_attn = self.pool_pocket(h_p, pocket_batch, pocket_interface_score, bsz)
        pocket_if_mask = pocket_interface_score > 0
        g_poc_mean = masked_mean_by_graph(h_p, pocket_batch, pocket_if_mask, bsz)
        g_poc_max = masked_max_by_graph(h_p, pocket_batch, pocket_if_mask, bsz)
        z_node = self.node_mlp(
            torch.cat(
                [g_lig_attn, g_lig_mean, g_lig_max, g_poc_if_attn, g_poc_mean, g_poc_max],
                dim=-1,
            )
        )

        valid_e = e_pair[valid_pair] if valid_pair.any() else e_pair.new_zeros(0, self.pair_dim)
        valid_b = pair_batch[valid_pair] if valid_pair.any() else pair_batch.new_zeros(0)
        valid_scalar = scalar[valid_pair] if valid_pair.any() else scalar.new_zeros(0)
        valid_gate_only = valid_gate[valid_pair] if valid_pair.any() else gate.new_zeros(0)

        pair_mean = torch.zeros(bsz, self.pair_dim, device=h_l.device, dtype=h_l.dtype)
        pair_sum_scaled = torch.zeros_like(pair_mean)
        pair_max = torch.zeros_like(pair_mean)
        pair_attn = torch.zeros_like(pair_mean)
        pair_topk_mean = torch.zeros_like(pair_mean)
        interaction_sum = torch.zeros(bsz, 1, device=h_l.device, dtype=h_l.dtype)
        interaction_sum_scaled = torch.zeros_like(interaction_sum)
        interaction_topk_mean = torch.zeros_like(interaction_sum)
        log_pair_count = torch.zeros_like(interaction_sum)

        if valid_e.numel() > 0:
            n_raw = scatter_sum(
                torch.ones(valid_e.size(0), device=h_l.device, dtype=h_l.dtype),
                valid_b,
                dim=0,
                dim_size=bsz,
            )
            n_safe = n_raw.clamp(min=1.0)
            pair_mean = scatter_mean(valid_e, valid_b, dim=0, dim_size=bsz)
            pair_sum_scaled = scatter_sum(valid_e, valid_b, dim=0, dim_size=bsz) / torch.sqrt(n_safe).unsqueeze(-1)
            pair_max = safe_scatter_max(valid_e, valid_b, bsz)
            attn_score = self.pair_attn(valid_e).squeeze(-1)
            attn_alpha = scatter_softmax(attn_score.float(), valid_b, dim=0).to(dtype=h_l.dtype)
            pair_attn = scatter_sum(valid_e * attn_alpha.unsqueeze(-1), valid_b, dim=0, dim_size=bsz)
            pair_topk_mean = topk_mean_by_graph(valid_e, valid_gate_only, valid_b, bsz, self.interface_topk)
            interaction_sum = scatter_sum(valid_scalar.unsqueeze(-1), valid_b, dim=0, dim_size=bsz)
            interaction_sum_scaled = interaction_sum / torch.sqrt(n_safe).unsqueeze(-1)
            interaction_topk_mean = topk_mean_by_graph(
                valid_scalar,
                torch.abs(valid_scalar),
                valid_b,
                bsz,
                self.interface_topk,
            )
            log_pair_count = torch.log1p(n_raw).unsqueeze(-1)

        z_pair = self.pair_mlp(
            torch.cat(
                [
                    pair_mean,
                    pair_sum_scaled,
                    pair_max,
                    pair_attn,
                    pair_topk_mean,
                    interaction_sum,
                    interaction_sum_scaled,
                    interaction_topk_mean,
                    log_pair_count,
                ],
                dim=-1,
            )
        )
        z_complex = self.complex_mlp(torch.cat([z_node, z_pair], dim=-1))
        affinity_pred_norm = self.affinity_head(z_complex).squeeze(-1)
        affinity_pred = affinity_pred_norm * self.y_std.to(affinity_pred_norm.dtype) + self.y_mean.to(affinity_pred_norm.dtype)
        contrastive_z = self.project_complex(z_complex)
        return {
            "affinity_pred_norm": affinity_pred_norm,
            "affinity_pred": affinity_pred,
            "contrastive_z": contrastive_z,
            "pair_gate": gate,
            "pair_score": score,
            "pair_scalar": scalar,
            "z_node": z_node,
            "z_pair": z_pair,
            "z_complex": z_complex,
            "pair_batch": pair_batch,
            "valid_pair": valid_pair,
            "pair_dist": dist,
        }

    def forward(self, lig: Batch, pocket: Batch) -> Dict[str, Any]:
        enc = self._encode_interleaved(lig, pocket)
        out = self._readout(
            enc["h_lig"],
            enc["h_pocket"],
            lig.batch,
            pocket.batch,
            enc["idx_l"],
            enc["idx_p"],
            enc["dist"],
            enc["pair_batch"],
            enc["valid_pair"],
            enc["e_pair"],
            enc["lig_x"],
            enc["pocket_x"],
            enc["pocket_residue_type"],
        )
        out.update(enc)
        return out


def normalized_affinity_loss(
    out: Dict[str, Tensor],
    y_pk: Tensor,
    y_mean: Tensor,
    y_std: Tensor,
    *,
    loss_mode: str = "smooth_l1",
    mse_weight: float = 0.0,
    tail_weight_alpha: float = 0.0,
    tail_weight_threshold: float = 0.75,
    sample_weight: Optional[Tensor] = None,
) -> Tensor:
    y_norm = (y_pk.float() - y_mean.to(y_pk.device, dtype=y_pk.dtype)) / y_std.to(y_pk.device, dtype=y_pk.dtype).clamp(min=1e-6)
    pred_norm = out["affinity_pred_norm"].float()
    y_norm = y_norm.float()
    loss_mode = loss_mode.lower()
    if loss_mode == "smooth_l1":
        per_sample = torch.nn.functional.smooth_l1_loss(pred_norm, y_norm, reduction="none")
        if mse_weight > 0.0:
            per_sample = per_sample + float(mse_weight) * torch.nn.functional.mse_loss(pred_norm, y_norm, reduction="none")
    elif loss_mode == "mse":
        per_sample = torch.nn.functional.mse_loss(pred_norm, y_norm, reduction="none")
    elif loss_mode == "smooth_l1_mse":
        per_sample = torch.nn.functional.smooth_l1_loss(pred_norm, y_norm, reduction="none")
        per_sample = per_sample + float(mse_weight) * torch.nn.functional.mse_loss(pred_norm, y_norm, reduction="none")
    else:
        raise ValueError(f"Unknown affinity loss mode: {loss_mode}")
    if tail_weight_alpha > 0.0:
        excess = torch.clamp(torch.abs(y_norm) - float(tail_weight_threshold), min=0.0)
        per_sample = per_sample * (1.0 + float(tail_weight_alpha) * excess)
    if sample_weight is not None:
        weights = sample_weight.to(per_sample.device, dtype=per_sample.dtype).view_as(per_sample).clamp(min=0.0)
        denom = weights.sum().clamp(min=1e-6)
        return (per_sample * weights).sum() / denom
    return per_sample.mean()


def cross_modal_alignment_loss(student_z: Tensor, semantic_z: Tensor) -> Tensor:
    student_z = torch.nn.functional.normalize(student_z.float(), dim=-1)
    semantic_z = torch.nn.functional.normalize(semantic_z.float(), dim=-1)
    if student_z.shape != semantic_z.shape:
        raise ValueError("student_z and semantic_z must have matching shapes")
    return (1.0 - torch.sum(student_z * semantic_z, dim=-1)).mean()


def cross_modal_infonce_loss(student_z: Tensor, semantic_z: Tensor, *, temperature: float = 0.1) -> Tensor:
    """Symmetric native-3D/semantic InfoNCE; other complexes are negatives."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    student_z = torch.nn.functional.normalize(student_z.float(), dim=-1)
    semantic_z = torch.nn.functional.normalize(semantic_z.float(), dim=-1)
    if student_z.shape != semantic_z.shape or student_z.ndim != 2:
        raise ValueError("student_z and semantic_z must be matching 2D tensors")
    logits = student_z @ semantic_z.t() / float(temperature)
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (
        torch.nn.functional.cross_entropy(logits, labels)
        + torch.nn.functional.cross_entropy(logits.t(), labels)
    )


def semantic_decoy_triplet_loss(
    pose_z: Tensor,
    semantic_z: Tensor,
    complex_index: Tensor,
    is_native: Tensor,
    *,
    margin: float = 0.2,
    valid_pose: Optional[Tensor] = None,
) -> Tensor:
    """Require native 3D poses to be closer to semantic references than decoys."""
    pose_z = torch.nn.functional.normalize(pose_z.float(), dim=-1)
    semantic_z = torch.nn.functional.normalize(semantic_z.float(), dim=-1)
    complex_index = complex_index.long().view(-1)
    is_native = is_native.bool().view(-1)
    if pose_z.ndim != 2 or semantic_z.ndim != 2 or pose_z.size(-1) != semantic_z.size(-1):
        raise ValueError("pose_z and semantic_z must be 2D tensors with matching feature dimensions")
    if pose_z.size(0) != complex_index.numel() or pose_z.size(0) != is_native.numel():
        raise ValueError("pose_z, complex_index and is_native must have matching first dimensions")
    if valid_pose is None:
        valid_pose = torch.ones_like(is_native)
    else:
        valid_pose = valid_pose.bool().view(-1)
        if valid_pose.numel() != pose_z.size(0):
            raise ValueError("valid_pose must have the same length as pose_z")

    group_losses: List[Tensor] = []
    for group_id in torch.unique(complex_index).tolist():
        group_id = int(group_id)
        if group_id < 0 or group_id >= semantic_z.size(0):
            continue
        group = (complex_index == group_id) & valid_pose
        native_mask = group & is_native
        decoy_mask = group & (~is_native)
        if int(native_mask.sum().item()) != 1 or not decoy_mask.any():
            continue
        reference = semantic_z[group_id]
        positive_distance = 1.0 - torch.sum(pose_z[native_mask][0] * reference)
        negative_distance = 1.0 - torch.sum(pose_z[decoy_mask] * reference.unsqueeze(0), dim=-1)
        group_losses.append(torch.relu(float(margin) + positive_distance - negative_distance).mean())
    if not group_losses:
        return pose_z.sum() * 0.0
    return torch.stack(group_losses).mean()


def conditional_pose_contrastive_loss(
    pose_z: Tensor,
    semantic_z: Tensor,
    complex_index: Tensor,
    is_native: Tensor,
    *,
    margin: float = 0.2,
    temperature: float = 0.1,
    valid_pose: Optional[Tensor] = None,
) -> Tensor:
    """Smoothly rank each native pose above same-complex decoys."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    pose_z = torch.nn.functional.normalize(pose_z.float(), dim=-1)
    semantic_z = torch.nn.functional.normalize(semantic_z.float(), dim=-1)
    complex_index = complex_index.long().view(-1)
    is_native = is_native.bool().view(-1)
    if pose_z.ndim != 2 or semantic_z.ndim != 2 or pose_z.size(-1) != semantic_z.size(-1):
        raise ValueError("pose_z and semantic_z must be 2D tensors with matching feature dimensions")
    if pose_z.size(0) != complex_index.numel() or pose_z.size(0) != is_native.numel():
        raise ValueError("pose_z, complex_index and is_native must have matching first dimensions")
    if valid_pose is None:
        valid_pose = torch.ones_like(is_native)
    else:
        valid_pose = valid_pose.bool().view(-1)
        if valid_pose.numel() != pose_z.size(0):
            raise ValueError("valid_pose must have the same length as pose_z")

    group_losses: List[Tensor] = []
    for group_id in torch.unique(complex_index).tolist():
        group_id = int(group_id)
        if group_id < 0 or group_id >= semantic_z.size(0):
            continue
        group = (complex_index == group_id) & valid_pose
        native_mask = group & is_native
        decoy_mask = group & (~is_native)
        if int(native_mask.sum().item()) != 1 or not decoy_mask.any():
            continue
        reference = semantic_z[group_id]
        positive_similarity = torch.sum(pose_z[native_mask][0] * reference)
        negative_similarity = torch.sum(pose_z[decoy_mask] * reference.unsqueeze(0), dim=-1)
        violations = (negative_similarity - positive_similarity + float(margin)) / float(temperature)
        group_losses.append(torch.nn.functional.softplus(violations).mean())
    if not group_losses:
        return pose_z.sum() * 0.0
    return torch.stack(group_losses).mean()
