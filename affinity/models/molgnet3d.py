from __future__ import annotations

import math
from functools import partial
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import reset
from torch_scatter import scatter
from utils.molecular_utils import NUM_ATOM_TYPES, NUM_BOND_TYPES



def rbf_encode_dist(
    dist: torch.Tensor,
    num_kernels: int = 96,
    start: float = 0.0,
    stop: float = 6.0,
    std_width: float = 1.5,
) -> torch.Tensor:
    """
    RBF distance encoding using Gaussian basis
    
    Args:
        dist: [E] or [E, 1]
    Returns:
        [E, num_kernels]
    """
    dist = dist.view(-1, 1)
    device, dtype = dist.device, dist.dtype

    means = torch.linspace(start, stop, num_kernels, device=device, dtype=dtype)

    if num_kernels > 1:
        delta = (stop - start) / (num_kernels - 1)
        std = std_width * delta
    else:
        std = std_width

    rbf = torch.exp(-0.5 * ((dist - means) / std) ** 2)
    return rbf


def _emolgnet_ckpt_hp(
    layer: "EMolGNetLayer",
    edge_index: Tensor,
    node_types: Tensor,
    h: Tensor,
    pos: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """e_ij=None 且 edge_attr_emb=None：仅 h、pos 作为 checkpoint 张量入参。"""
    return layer(h, pos, edge_index, None, None, node_types=node_types)


def _emolgnet_ckpt_hpea(
    layer: "EMolGNetLayer",
    edge_index: Tensor,
    node_types: Tensor,
    h: Tensor,
    pos: Tensor,
    edge_attr_emb: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """e_ij=None，edge 嵌入参与 autograd：edge_attr_emb 显式传入 checkpoint。"""
    return layer(h, pos, edge_index, edge_attr_emb, None, node_types=node_types)


def _emolgnet_ckpt_hpe(
    layer: "EMolGNetLayer",
    edge_index: Tensor,
    node_types: Tensor,
    h: Tensor,
    pos: Tensor,
    e_ij: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """edge_attr_emb=None，已有 e_ij。"""
    return layer(h, pos, edge_index, None, e_ij, node_types=node_types)


def _emolgnet_ckpt_hpeea(
    layer: "EMolGNetLayer",
    edge_index: Tensor,
    node_types: Tensor,
    h: Tensor,
    pos: Tensor,
    e_ij: Tensor,
    edge_attr_emb: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """edge_attr_emb 与 e_ij 均作为 checkpoint 张量入参（避免仅闭包捕获可训练边嵌入）。"""
    return layer(h, pos, edge_index, edge_attr_emb, e_ij, node_types=node_types)





class EMolGNetLayer(MessagePassing):
    """
    Pair-enhanced EGNN Layer: 跨层持续的 pair state + 几何门控 attention.
    
    Key components:
    1. Pair state: e_ij^(0)=MLP([RBF(d_ij), Emb(t_i), Emb(t_j)])（不含 pair-role；pair_init_type_emb_dim=0 时退化为仅 RBF）
       e_ij^(l+1)=EdgeUpdate(e_ij^(l), h_i, h_j, d_ij, θ)
    2. Feature flow: I_ij = k_j + e_ij (attention 用 e_ij)
    3. Geometric gating: α̃_ij = α_ij + β·tanh(g_ij) (signed additive，可增强可抑制)
    4. Coordinate update 也用 e_ij
    5. Dual update streams: Semantic (GRU) + Geometric (equivariant coord)
    """
    def __init__(
        self,
        hidden_dim: int,
        heads: int = 8,
        rbf_kernels: int = 96,
        rbf_start: float = 0.0,
        rbf_stop: float = 6.0,
        rbf_std_width: float = 1.5,
        edge_dim: int = 0,
        dropout: float = 0.05,
        coord_scale: float = 0.1,
        update_coords: bool = True,
        num_atom_types: int = NUM_ATOM_TYPES,
        pair_init_type_emb_dim: int = 32,
    ):
        super().__init__(node_dim=0, aggr='add')
        
        assert hidden_dim % heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})"
        
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dim_per_head = hidden_dim // heads
        self.rbf_kernels = rbf_kernels
        self.update_coords = update_coords
        self.coord_scale = coord_scale
        
        # RBF encoding parameters
        self.rbf_start = rbf_start
        self.rbf_stop = rbf_stop
        self.rbf_std_width = rbf_std_width
        self.pair_init_type_emb_dim = pair_init_type_emb_dim

        # 与 node_embedding 解耦：专供 e_ij^(0) 的原子类型嵌入（便于与 Uni-Mol 式 pair-type 思路对齐、做消融）
        if pair_init_type_emb_dim > 0:
            self.atom_type_emb_pair = nn.Embedding(num_atom_types, pair_init_type_emb_dim)
            geo_in_dim = rbf_kernels + 2 * pair_init_type_emb_dim
        else:
            self.atom_type_emb_pair = None
            geo_in_dim = rbf_kernels

        # Attention projections
        self.lin_q = nn.Linear(hidden_dim, hidden_dim)
        self.lin_k = nn.Linear(hidden_dim, hidden_dim)
        self.lin_v = nn.Linear(hidden_dim, hidden_dim)
        
        # 初始化: e_ij^(0) = MLP([RBF, Emb(t_i), Emb(t_j)]) -> [E, hidden_dim]
        self.lin_geo = nn.Sequential(
            nn.Linear(geo_in_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # EdgeUpdate: e_ij^(l+1) = e_ij^(l) + Δ(e_ij, h_i, h_j, r_ij, θ)
        # 命名为 edge_mlp，避免与 MessagePassing.edge_update 接口同名造成歧义
        edge_update_in = hidden_dim * 4 + rbf_kernels  # e_ij, h_i, h_j, theta_emb, r_ij
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_update_in, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Edge attribute projection (if provided, for θ_kij)
        if edge_dim > 0:
            self.lin_edge = nn.Linear(edge_dim, hidden_dim)
        else:
            self.lin_edge = None
        
        # 几何门控: g_ij ∈ R, α̃_ij = α_ij + β·tanh(g_ij) (signed additive bias，可增强可抑制)
        self.mlp_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, heads)
        )
        # β 使用 sigmoid 约束在 (0, 0.5) 内，避免训练后期无限放大几何偏置
        self.gate_scale_raw = nn.Parameter(torch.zeros(1))
        
        # Coordinate update MLP (geometric stream)
        if update_coords:
            self.mlp_coord = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1, bias=False)
            )
        
        # Semantic stream: FFN + GRU
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.lin_q)
        reset(self.lin_k)
        reset(self.lin_v)
        reset(self.lin_geo)
        if self.atom_type_emb_pair is not None:
            nn.init.normal_(self.atom_type_emb_pair.weight, std=0.02)
        if self.lin_edge is not None:
            reset(self.lin_edge)
        for layer in self.edge_mlp:
            if isinstance(layer, nn.Linear):
                reset(layer)
        for layer in self.mlp_gate:
            if isinstance(layer, nn.Linear):
                reset(layer)
        if self.update_coords:
            for layer in self.mlp_coord:
                if isinstance(layer, nn.Linear):
                    reset(layer)
        for layer in self.ffn:
            if isinstance(layer, nn.Linear):
                reset(layer)
        nn.init.xavier_uniform_(self.gru.weight_ih)
        nn.init.xavier_uniform_(self.gru.weight_hh)
        nn.init.zeros_(self.gru.bias_ih)
        nn.init.zeros_(self.gru.bias_hh)
    
    def forward(
        self,
        h: Tensor,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor = None,
        e_ij: Tensor = None,
        node_types: Tensor = None,
    ):
        """
        Forward pass of pair-enhanced EGNN layer.
        
        Args:
            h: Node features [N, hidden_dim]
            x: Node coordinates [N, 3]
            edge_index: Edge indices [2, E]
            edge_attr: Edge attributes [E, edge_dim] or None
            e_ij: Pair state [E, hidden_dim], None 时用 RBF(+类型) 初始化
            node_types: 原子类型索引 [N] long；j=edge_index[0], i=edge_index[1] 时用于 Emb(t_j), Emb(t_i)
        
        Returns:
            h_out: Updated node features [N, hidden_dim]
            x_out: Updated node coordinates [N, 3]
            e_ij_out: Updated pair state [E, hidden_dim]
        """
        # 1. Compute distances and RBF encoding
        # edge_index[0] = source (j), edge_index[1] = target (i)
        rel_coors = x[edge_index[1]] - x[edge_index[0]]  # [E, 3]
        dist_sq = rel_coors.pow(2).sum(dim=-1, keepdim=True)  # [E, 1]
        dist = (dist_sq + 1e-6).sqrt()  # [E, 1]
        dist = torch.clamp(dist, min=1e-5)  # 防止 d≈0 的非物理碰撞
        
        r_ij = rbf_encode_dist(dist, self.rbf_kernels, self.rbf_start, self.rbf_stop, self.rbf_std_width)  # [E, rbf_kernels]
        
        # 2. Pair state: e_ij^(0)=MLP([RBF, Emb(t_i), Emb(t_j)]) 或 继承上一层的 e_ij
        if e_ij is None:
            if self.atom_type_emb_pair is not None and node_types is not None:
                t_i = node_types[edge_index[1]]
                t_j = node_types[edge_index[0]]
                geo_in = torch.cat(
                    [r_ij, self.atom_type_emb_pair(t_i), self.atom_type_emb_pair(t_j)],
                    dim=-1,
                )
            else:
                geo_in = r_ij
            e_ij = self.lin_geo(geo_in)  # [E, hidden_dim]
        
        # 3. EdgeUpdate: e_ij^(l+1) = e_ij^(l) + Δ(e_ij, h_i, h_j, d_ij, θ)
        h_i = h[edge_index[1]]  # [E, hidden_dim]
        h_j = h[edge_index[0]]  # [E, hidden_dim]
        theta_emb = self.lin_edge(edge_attr) if (edge_attr is not None and self.lin_edge is not None) else torch.zeros_like(e_ij)
        edge_update_in = torch.cat([e_ij, h_i, h_j, theta_emb, r_ij], dim=-1)  # [E, 4*hidden + rbf]
        e_ij_delta = self.edge_mlp(edge_update_in)
        e_ij_out = F.layer_norm(e_ij + e_ij_delta, (self.hidden_dim,))  # residual with bounded scale
        
        # 4. 从 e_ij 得到 r_geo (用于 I_ij) 和 attn_gate
        r_geo = e_ij_out.view(-1, self.heads, self.dim_per_head)  # [E, heads, dim_per_head]
        attn_gate = self.mlp_gate(e_ij_out)  # [E, heads], g_ij ∈ R
        
        edge_attr_proj = None
        if edge_attr is not None and self.lin_edge is not None:
            edge_attr_proj = self.lin_edge(edge_attr).view(-1, self.heads, self.dim_per_head)
        
        # 5. Pre-project Q/K/V
        q = self.lin_q(h).view(-1, self.heads, self.dim_per_head)
        k = self.lin_k(h).view(-1, self.heads, self.dim_per_head)
        v = self.lin_v(h).view(-1, self.heads, self.dim_per_head)
        
        # 6. Propagate messages with attention (α̃ = α + β·tanh(g))
        m_i, coord_delta = self.propagate(
            edge_index,
            q=q,
            k=k,
            v=v,
            r_geo=r_geo,
            edge_attr=edge_attr_proj,
            attn_gate=attn_gate,
            rel_coors=rel_coors,
            e_ij=e_ij_out,
            size=None
        )
        
        # 7. Semantic stream: FFN + GRU update
        m_i_flat = m_i.view(-1, self.hidden_dim)  # [N, hidden_dim]
        m_i_residual = m_i_flat + h
        m_i_norm = self.layer_norm1(m_i_residual)
        ffn_out = self.ffn(m_i_norm)
        ffn_residual = ffn_out + m_i_norm
        ffn_norm = self.layer_norm2(ffn_residual)
        h_out = self.gru(ffn_norm, h)  # [N, hidden_dim]
        
        # 8. Geometric stream: Coordinate update
        if self.update_coords:
            x_out = x + self.coord_scale * coord_delta
        else:
            x_out = x
        
        return h_out, x_out, e_ij_out
    
    def message(
        self,
        q_i: Tensor,
        k_j: Tensor,
        v_j: Tensor,
        r_geo: Tensor,
        attn_gate: Tensor,
        rel_coors: Tensor,
        e_ij: Tensor,
        edge_attr: Tensor = None,
        index: Tensor = None,
        size_i: int = None
    ):
        """
        Compute attention messages and coordinate updates.
        
        Feature flow: I_ij = k_j + e_ij (r_geo 来自 e_ij)
        Attention: α̃_ij = α_ij + β·tanh(g_ij) signed additive bias
        """
        # I_ij = k_j + e_ij
        I_ij = k_j + r_geo
        if edge_attr is not None:
            I_ij = I_ij + edge_attr
        
        # Keep attention logits in FP32 under AMP; fp16 dot products can overflow
        # late in training and poison the shared encoder with NaN gradients.
        q_attn = q_i.float()
        i_attn = I_ij.float()
        alpha_logits = (q_attn * i_attn).sum(dim=-1) / math.sqrt(self.dim_per_head)  # [E, heads]
        beta = 0.5 * torch.sigmoid(self.gate_scale_raw.float())  # β ∈ (0, 0.5)
        alpha_logits = alpha_logits + beta * torch.tanh(attn_gate.float())
        alpha_logits = torch.nan_to_num(alpha_logits, nan=0.0, posinf=30.0, neginf=-30.0)
        alpha_logits = torch.clamp(alpha_logits, min=-30.0, max=30.0)
        alpha = softmax(alpha_logits, index, num_nodes=size_i).to(dtype=v_j.dtype)  # [E, heads]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        messages = v_j * alpha.view(-1, self.heads, 1)  # [E, heads, dim_per_head]
        
        # Coordinate update 也用 e_ij: msg + e_ij -> mlp_coord
        if self.update_coords:
            alpha_agg = alpha.mean(dim=-1, keepdim=True)  # [E, 1]
            msg_flat = messages.view(-1, self.hidden_dim)
            coord_in = msg_flat + e_ij  # 融合 pair state
            coord_weights = self.mlp_coord(coord_in)  # [E, 1]
            coord_weights = torch.clamp(coord_weights, -self.coord_scale * 10, self.coord_scale * 10)
            coord_delta = rel_coors * alpha_agg * coord_weights  # [E, 3]
        else:
            coord_delta = torch.zeros_like(rel_coors)
        
        return messages, coord_delta
    
    def aggregate(self, inputs, index, dim_size=None):
        """
        Aggregate messages and coordinate updates separately.
        
        Args:
            inputs: Tuple of (messages, coord_delta) from message()
            index: Target node indices [E]
            dim_size: Number of nodes
        
        Returns:
            m_i: Aggregated messages [N, heads, dim_per_head]
            coord_i: Aggregated coordinate updates [N, 3]
        """
        messages, coord_delta = inputs
        
        # Aggregate messages (sum aggregation)
        m_i = super().aggregate(messages, index, dim_size=dim_size)  # [N, heads, dim_per_head]
        
        # Aggregate coordinate updates (sum aggregation)
        coord_i = scatter(coord_delta, index, dim=0, dim_size=dim_size, reduce='sum')  # [N, 3]
        
        return m_i, coord_i
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(hidden_dim={self.hidden_dim}, '
                f'heads={self.heads}, rbf_kernels={self.rbf_kernels}, '
                f'update_coords={self.update_coords})')


class EMolGNet(nn.Module):
    """
    EMolGNet Model: Multi-layer Equivariant MolGNet-EGNN Network
    
    Combines topological semantics (MolGNet) with 3D equivariant geometry (EGNN).
    Similar structure to EGNN but uses EMolGNetLayer instead of EGNNLayer.
    
    Notation:
        输入 x: 原子类型索引 [N]；pos: 坐标 [N, 3]；内部 h 为嵌入后节点特征 [N, hidden_dim]
    """
    def __init__(
        self,
        num_node_types: int = NUM_ATOM_TYPES,
        num_edge_types: int = None,
        hidden_dim: int = 512,
        num_layers: int = 6,
        heads: int = 8,
        rbf_kernels: int = 96,
        rbf_start: float = 0.0,
        rbf_stop: float = 6.0,
        rbf_std_width: float = 1.5,
        update_coords: bool = True,
        output_dim: int = None,
        dropout: float = 0.0,
        coord_scale: float = 0.1,
        pair_init_type_emb_dim: int = 32,
        use_gradient_checkpointing: bool = False,
    ):
        """
        Args:
            node_dim: Node feature dimension (after embedding)
            edge_dim: Edge feature dimension (after embedding)
            hidden_dim: Hidden layer dimension
            num_layers: Number of EMolGNet layers
            heads: Number of attention heads
            rbf_kernels: Number of RBF kernels for distance encoding
            rbf_start: Minimum distance for RBF kernel centers
            rbf_stop: Maximum distance for RBF kernel centers
            rbf_std_width: Width multiplier for RBF standard deviation
            update_coords: Whether to update coordinates
            output_dim: Output dimension, if None then equals hidden_dim
            num_node_types: Number of node types (for embedding), if None uses default
            num_edge_types: Number of edge types (for embedding), if None uses default
            dropout: Dropout rate
            coord_scale: Scale factor for coordinate updates
            pair_init_type_emb_dim: e_ij^(0) 中 Emb(t_i),Emb(t_j) 维度；0 表示仅 RBF（消融）
            use_gradient_checkpointing: 训练时对每层 GNN 使用 activation checkpoint（需与 self.training 同时为 True 才生效）
        """
        super(EMolGNet, self).__init__()
        self.num_layers = num_layers
        self.update_coords = update_coords
        self.output_dim = output_dim or hidden_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        
        self.node_embedding = nn.Embedding(num_node_types, hidden_dim)

        if num_edge_types is not None:
            self.edge_embedding = nn.Embedding(num_edge_types, hidden_dim)
        
        
        # EMolGNet layers
        self.layers = nn.ModuleList([
            EMolGNetLayer(
                hidden_dim=hidden_dim,
                heads=heads,
                rbf_kernels=rbf_kernels,
                rbf_start=rbf_start,
                rbf_stop=rbf_stop,
                rbf_std_width=rbf_std_width,
                edge_dim=hidden_dim,
                dropout=dropout,
                coord_scale=coord_scale,
                update_coords=update_coords,
                num_atom_types=num_node_types,
                pair_init_type_emb_dim=pair_init_type_emb_dim,
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, self.output_dim)
    
    def forward(self, x, pos, edge_index, edge_attr=None, return_pair: bool = False):
        """
        Forward pass
        
        Args:
            x: Node features (N,) - category indices (long)
            pos: Node coordinates (N, 3)
            edge_index: Edge indices (2, E)
            edge_attr: Edge features (E,) - category indices (long)
            return_pair: If True, also return final edge pair state e_ij [E, hidden_dim]
                (same dimension as pre-output_proj node features).
        
        Returns:
            h: Node features (N, output_dim)
            pos_out: Coordinates (updated if update_coords=True, otherwise original)
            e_ij (optional): Pair state on edges after the last layer
        """
        # Embedding: convert category indices to vectors
        x_emb = self.node_embedding(x)  # (N, hidden_dim)
        if edge_attr is not None:
            edge_attr_emb = self.edge_embedding(edge_attr)  # (E, edge_dim)
        else:
            edge_attr_emb = None
        
        h = x_emb
        pos_out = pos
        e_ij = None  # 首层由 layer 用 MLP([RBF, Emb(t_i), Emb(t_j)]) 初始化
        graph_requires_grad = h.requires_grad or pos_out.requires_grad
        if edge_attr_emb is not None and edge_attr_emb.requires_grad:
            graph_requires_grad = True
        use_ckpt = (
            self.training
            and self.use_gradient_checkpointing
            and graph_requires_grad
        )
        # Pass through EMolGNet layers (跨层传递 pair state)
        for layer in self.layers:
            if use_ckpt:
                if e_ij is None:
                    if edge_attr_emb is None:
                        h, pos_out, e_ij = checkpoint(
                            partial(_emolgnet_ckpt_hp, layer, edge_index, x),
                            h,
                            pos_out,
                            use_reentrant=False,
                        )
                    else:
                        h, pos_out, e_ij = checkpoint(
                            partial(_emolgnet_ckpt_hpea, layer, edge_index, x),
                            h,
                            pos_out,
                            edge_attr_emb,
                            use_reentrant=False,
                        )
                elif edge_attr_emb is None:
                    h, pos_out, e_ij = checkpoint(
                        partial(_emolgnet_ckpt_hpe, layer, edge_index, x),
                        h,
                        pos_out,
                        e_ij,
                        use_reentrant=False,
                    )
                else:
                    h, pos_out, e_ij = checkpoint(
                        partial(_emolgnet_ckpt_hpeea, layer, edge_index, x),
                        h,
                        pos_out,
                        e_ij,
                        edge_attr_emb,
                        use_reentrant=False,
                    )
            else:
                h, pos_out, e_ij = layer(
                    h, pos_out, edge_index, edge_attr_emb, e_ij, node_types=x
                )
        
        # Output projection (node only; e_ij stays in hidden_dim for pair heads)
        h = self.output_proj(h)  # (N, output_dim)
        
        if return_pair:
            return h, pos_out, e_ij
        return h, pos_out




if __name__ == "__main__":
    def test_emolgnet():
        """
        Test function for EMolGNet model.
        Tests forward pass, output shapes, and basic functionality.
        """
        print("=" * 60)
        print("Testing EMolGNet Model")
        print("=" * 60)
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # 与 EMolGNet 类默认一致：512 / 6 / 8 / RBF 96·[0,6]·std_width=1.5
        num_nodes = 20
        hidden_dim = 512
        num_layers = 6
        heads = 8
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Device: {device}")
        print(f"Number of nodes: {num_nodes}")
        print(f"Hidden dim: {hidden_dim}, Layers: {num_layers}, Heads: {heads}")
        print()
        
        # Create model（其余 RBF 等使用类默认值）
        model = EMolGNet(
            num_node_types=NUM_ATOM_TYPES,
            num_edge_types=NUM_BOND_TYPES,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            update_coords=True,
        ).to(device)
        
        print(f"Model created: {model}")
        print()
        
        # Generate test data
        # Node features: category indices (0 to NUM_ATOM_TYPES-1)
        x = torch.randint(0, NUM_ATOM_TYPES, (num_nodes,), device=device, dtype=torch.long)
        # Node coordinates: random 3D positions
        pos = torch.randn(num_nodes, 3, device=device)
        
        # Create a simple graph (fully connected for testing)
        edge_index = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()
        
        # Edge features: category indices (0 to NUM_BOND_TYPES-1)
        edge_attr = torch.randint(0, NUM_BOND_TYPES, (edge_index.size(1),), device=device, dtype=torch.long)
        
        print("Test 1: Forward pass")
        print("-" * 60)
        print(f"Input shapes:")
        print(f"  x (node types): {x.shape}")
        print(f"  pos (coordinates): {pos.shape}")
        print(f"  edge_index: {edge_index.shape}")
        print(f"  edge_attr (edge types): {edge_attr.shape}")
        
        # Forward pass
        h_out, pos_out = model(x, pos, edge_index, edge_attr)
        
        print(f"\nOutput shapes:")
        print(f"  h_out: {h_out.shape}")
        print(f"  pos_out: {pos_out.shape}")
        
        # Check output shapes
        assert h_out.shape == (num_nodes, hidden_dim), f"Expected h_out shape ({num_nodes}, {hidden_dim}), got {h_out.shape}"
        assert pos_out.shape == (num_nodes, 3), f"Expected pos_out shape ({num_nodes}, 3), got {pos_out.shape}"
        print("✓ Output shapes are correct")
        print()
        
        # Test 2: Gradient flow
        print("Test 2: Gradient flow")
        print("-" * 60)
        pos.requires_grad_(True)
        
        h_out2, pos_out2 = model(x, pos, edge_index, edge_attr)
        loss = h_out2.sum() + pos_out2.sum()
        loss.backward()
        
        assert pos.grad is not None, "Gradients should flow to pos"
        print("✓ Gradients flow correctly")
        print()
        
        # Test 3: Coordinate update
        print("Test 3: Coordinate update")
        print("-" * 60)
        pos_no_grad = pos.detach().clone()
        h_out3, pos_out3 = model(x, pos_no_grad, edge_index, edge_attr)
        
        coord_change = torch.norm(pos_out3 - pos_no_grad, dim=-1).mean().item()
        print(f"Average coordinate change: {coord_change:.6f}")
        
        if coord_change > 1e-6:
            print("✓ Coordinates are being updated")
        else:
            print("⚠ Coordinates show minimal change (may be normal)")
        print()
        
        # Test 4: Batch processing (multiple graphs)
        print("Test 4: Batch processing")
        print("-" * 60)
        batch_size = 3
        nodes_per_graph = [8, 10, 12]
        total_nodes = sum(nodes_per_graph)
        
        x_batch = torch.randint(0, NUM_ATOM_TYPES, (total_nodes,), device=device, dtype=torch.long)
        pos_batch = torch.randn(total_nodes, 3, device=device)
        
        # Create edges for each graph separately
        edge_list = []
        node_offset = 0
        for num_nodes_graph in nodes_per_graph:
            for i in range(num_nodes_graph):
                for j in range(num_nodes_graph):
                    if i != j:
                        edge_list.append([node_offset + i, node_offset + j])
            node_offset += num_nodes_graph
        
        edge_index_batch = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
        edge_attr_batch = torch.randint(0, NUM_BOND_TYPES, (edge_index_batch.size(1),), device=device, dtype=torch.long)
        
        h_out_batch, pos_out_batch = model(x_batch, pos_batch, edge_index_batch, edge_attr_batch)
        
        assert h_out_batch.shape == (total_nodes, hidden_dim)
        assert pos_out_batch.shape == (total_nodes, 3)
        print(f"✓ Batch processing works with {batch_size} graphs")
        print(f"  Total nodes: {total_nodes}")
        print(f"  Total edges: {edge_index_batch.size(1)}")
        print()
        
        # Test 5: Model without coordinate update
        print("Test 5: Model without coordinate update")
        print("-" * 60)
        model_no_coord = EMolGNet(
            num_node_types=NUM_ATOM_TYPES,
            num_edge_types=NUM_BOND_TYPES,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            update_coords=False
        ).to(device)
        
        pos_test = pos_no_grad.clone()
        h_out4, pos_out4 = model_no_coord(x, pos_test, edge_index, edge_attr)
        
        assert torch.allclose(pos_out4, pos_test, atol=1e-5), "Coordinates should not change when update_coords=False"
        print("✓ Coordinates remain unchanged when update_coords=False")
        print()
        
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    
    test_emolgnet()
