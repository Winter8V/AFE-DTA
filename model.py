
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_max_pool


class GraphEncodingAdaptiveFeatureFusion(nn.Module):
    """Adaptively fuses shallow and deep GCN representations."""

    def __init__(self, h1_dim: int, h2_dim: int, h3_dim: int, dropout: float) -> None:
        super().__init__()
        self.proj_h1 = nn.Linear(h1_dim, h3_dim)
        self.proj_h2 = nn.Linear(h2_dim, h3_dim)
        self.gate_h1 = nn.Sequential(nn.Linear(h3_dim * 2, h3_dim), nn.Sigmoid())
        self.gate_h2 = nn.Sequential(nn.Linear(h3_dim * 2, h3_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(h3_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, h1: torch.Tensor, h2: torch.Tensor, h3: torch.Tensor) -> torch.Tensor:
        h1_proj = self.proj_h1(h1)
        h2_proj = self.proj_h2(h2)
        w1 = self.gate_h1(torch.cat([h3, h1_proj], dim=-1))
        w2 = self.gate_h2(torch.cat([h3, h2_proj], dim=-1))
        fused = h3 + w1 * h1_proj + w2 * h2_proj
        fused = self.norm(fused)
        fused = self.activation(fused)
        return self.dropout(fused)


class DrugGraphEncoder(nn.Module):
    """Three-layer GCN drug graph encoder enhanced by GEAFF."""

    def __init__(
        self,
        node_feature_dim: int = 94,
        output_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        h1_dim = node_feature_dim * 2
        h2_dim = node_feature_dim * 3
        h3_dim = node_feature_dim * 4

        self.gcn1 = GCNConv(node_feature_dim, h1_dim)
        self.gcn2 = GCNConv(h1_dim, h2_dim)
        self.gcn3 = GCNConv(h2_dim, h3_dim)
        self.geaff = GraphEncodingAdaptiveFeatureFusion(h1_dim, h2_dim, h3_dim, dropout)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.graph_projection = nn.Sequential(
            nn.Linear(h3_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, output_dim),
        )

    def forward(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        x = data.x.float()
        edge_index = data.edge_index
        batch = data.batch

        h1 = self.dropout(self.activation(self.gcn1(x, edge_index)))
        h2 = self.dropout(self.activation(self.gcn2(h1, edge_index)))
        h3 = self.dropout(self.activation(self.gcn3(h2, edge_index)))

        node_repr = self.geaff(h1, h2, h3)
        graph_repr = global_max_pool(node_repr, batch)
        drug_feature = self.graph_projection(graph_repr)
        return drug_feature, node_repr


class GatedCNNProteinEncoder(nn.Module):
    """Gated CNN encoder for fixed-length protein sequences."""

    def __init__(
        self,
        vocab_size: int = 25,
        protein_length: int = 1000,
        embed_dim: int = 128,
        num_filters: int = 32,
        kernel_size: int = 8,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        self.protein_length = protein_length
        self.embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)

        self.conv1 = nn.Conv1d(protein_length, num_filters, kernel_size)
        self.gate1 = nn.Conv1d(protein_length, num_filters, kernel_size)
        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size)
        self.gate2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size)
        self.conv3 = nn.Conv1d(num_filters * 2, num_filters * 3, kernel_size)
        self.gate3 = nn.Conv1d(num_filters * 2, num_filters * 3, kernel_size)

        conv_length = embed_dim - 3 * (kernel_size - 1)
        if conv_length <= 0:
            raise ValueError("embed_dim must be larger than 3 * (kernel_size - 1).")
        self.output_projection = nn.Linear(num_filters * 3 * conv_length, output_dim)
        self.activation = nn.ReLU()

    @staticmethod
    def _gated_conv(x: torch.Tensor, conv: nn.Conv1d, gate: nn.Conv1d) -> torch.Tensor:
        return conv(x) * torch.sigmoid(gate(x))

    def forward(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        target = data.target.long()
        if target.dim() == 3 and target.size(1) == 1:
            target = target.squeeze(1)
        if target.dim() != 2:
            raise ValueError(f"Expected protein target tensor with shape [batch, length], got {tuple(target.shape)}")
        if target.size(1) != self.protein_length:
            raise ValueError(
                f"Expected fixed protein length {self.protein_length}, got {target.size(1)}. "
                "Please pad or truncate protein sequences consistently with the manuscript."
            )

        x = self.embed(target)  # [batch, 1000, embed_dim]
        x = self.activation(self._gated_conv(x, self.conv1, self.gate1))
        x = self.activation(self._gated_conv(x, self.conv2, self.gate2))
        x = self.activation(self._gated_conv(x, self.conv3, self.gate3))
        protein_feature = self.output_projection(x.flatten(start_dim=1))
        return protein_feature, x


class BidirectionalCrossAttentionModule(nn.Module):
    """Bidirectional cross-attention between drug and protein representations."""

    def __init__(self, dim: int = 128, num_heads: int = 8, dropout: float = 0.3) -> None:
        super().__init__()
        self.drug_to_protein = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.protein_to_drug = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_drug = nn.LayerNorm(dim)
        self.norm_protein = nn.LayerNorm(dim)
        self.ffn_drug = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim)
        )
        self.ffn_protein = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, drug_feature: torch.Tensor, protein_feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        drug_seq = drug_feature.unsqueeze(1)
        protein_seq = protein_feature.unsqueeze(1)

        drug_attn, _ = self.drug_to_protein(drug_seq, protein_seq, protein_seq)
        drug_updated = self.norm_drug(drug_seq + self.dropout(drug_attn)).squeeze(1)
        drug_updated = self.norm_drug(drug_updated + self.dropout(self.ffn_drug(drug_updated)))

        protein_attn, _ = self.protein_to_drug(protein_seq, drug_seq, drug_seq)
        protein_updated = self.norm_protein(protein_seq + self.dropout(protein_attn)).squeeze(1)
        protein_updated = self.norm_protein(protein_updated + self.dropout(self.ffn_protein(protein_updated)))

        return drug_updated, protein_updated


class CubicBSplineLayer(nn.Module):
    """Feature-wise learnable cubic B-spline transformation."""

    def __init__(self, input_dim: int, num_knots: int = 30, value_range: tuple[float, float] = (-5.0, 5.0)) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_knots = num_knots
        self.register_buffer("knots", torch.linspace(value_range[0], value_range[1], num_knots))
        self.weights = nn.Parameter(torch.randn(input_dim, num_knots) * 0.01)
        self.scale = nn.Parameter(torch.ones(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_expanded = x.unsqueeze(-1)
        knot_spacing = self.knots[1] - self.knots[0]
        distance = (x_expanded - self.knots) / knot_spacing
        distance = distance * self.scale.view(1, -1, 1)
        abs_distance = distance.abs()

        inner = (2.0 / 3.0) - abs_distance.pow(2) + 0.5 * abs_distance.pow(3)
        outer = (2.0 - abs_distance).clamp(min=0).pow(3) / 6.0
        basis = torch.where(abs_distance < 1.0, inner, torch.where(abs_distance < 2.0, outer, torch.zeros_like(distance)))
        return torch.sum(basis * self.weights.unsqueeze(0), dim=-1)


class LSTNet(nn.Module):
    """Learnable spline transformation network for affinity regression."""

    def __init__(self, input_dim: int = 512, output_dim: int = 1, dropout: float = 0.3) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(input_dim)
        self.feature_gate = nn.Sequential(nn.Linear(input_dim, input_dim), nn.Sigmoid())
        self.spline = CubicBSplineLayer(input_dim=input_dim, num_knots=30, value_range=(-5.0, 5.0))
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        x_filtered = x_norm * self.feature_gate(x_norm)
        x_spline = self.spline(x_filtered)
        return self.regressor(x_spline)


class AFEDTA(nn.Module):
    """Adaptive Feature Enhancement and Cross-Modal Interaction Learning for DTA prediction."""

    def __init__(
        self,
        node_feature_dim: int = 94,
        protein_vocab_size: int = 25,
        protein_length: int = 1000,
        embed_dim: int = 128,
        output_dim: int = 128,
        num_heads: int = 8,
        gcn_dropout: float = 0.2,
        dropout: float = 0.3,
        num_filters: int = 32,
        kernel_size: int = 8,
    ) -> None:
        super().__init__()
        self.drug_encoder = DrugGraphEncoder(node_feature_dim, output_dim, gcn_dropout)
        self.protein_encoder = GatedCNNProteinEncoder(
            vocab_size=protein_vocab_size,
            protein_length=protein_length,
            embed_dim=embed_dim,
            num_filters=num_filters,
            kernel_size=kernel_size,
            output_dim=output_dim,
        )
        self.bcam = BidirectionalCrossAttentionModule(output_dim, num_heads, dropout)
        self.lst_net = LSTNet(input_dim=4 * output_dim, output_dim=1, dropout=dropout)

    def forward(self, data, return_features: bool = False):
        drug_feature, drug_nodes = self.drug_encoder(data)
        protein_feature, protein_map = self.protein_encoder(data)
        drug_attended, protein_attended = self.bcam(drug_feature, protein_feature)

        fused_feature = torch.cat(
            [drug_attended, drug_feature, protein_attended, protein_feature], dim=-1
        )
        prediction = self.lst_net(fused_feature)

        if return_features:
            return prediction, {
                "drug_feature": drug_feature,
                "protein_feature": protein_feature,
                "drug_attended": drug_attended,
                "protein_attended": protein_attended,
                "fused_feature": fused_feature,
                "drug_node_feature": drug_nodes,
                "protein_conv_feature": protein_map,
            }
        return prediction
