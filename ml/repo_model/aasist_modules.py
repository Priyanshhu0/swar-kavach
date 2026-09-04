"""
Official AASIST Graph Attention Modules.
Extracted from: https://github.com/clovaai/aasist/blob/main/models/AASIST.py

Copyright (c) 2021-present NAVER Corp.
MIT license

Modules:
  - GraphAttentionLayer: Homogeneous graph attention with pairwise node
    multiplication, temperature-scaled softmax, dual projection.
  - HtrgGraphAttentionLayer (HS-GAL): Heterogeneous graph attention with
    master/stack node, type-aware attention weights.
  - GraphPool: Top-k graph pooling with learned scoring.

These are extracted VERBATIM from the official repo with only import
adjustments (no SincConv or ResNet encoder — those are replaced by WavLM).
"""

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================================================================
# GraphAttentionLayer (Homogeneous)
# ====================================================================

class GraphAttentionLayer(nn.Module):
    """
    Graph attention layer with pairwise node multiplication.
    Used for the initial spectral (GAT-S) and temporal (GAT-T) graphs.

    Input:  x — (#bs, #node, #dim)
    Output: x — (#bs, #node, out_dim)
    """

    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        super().__init__()

        # Attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)

        # Projection (with and without attention)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        # Batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # Dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # Activation
        self.act = nn.SELU(inplace=True)

        # Temperature
        self.temp = kwargs.get("temperature", 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (#bs, #node, #dim)
        Returns:
            (#bs, #node, out_dim)
        """
        x = self.input_drop(x)
        att_map = self._derive_att_map(x)
        x = self._project(x, att_map)
        x = self._apply_BN(x)
        x = self.act(x)
        return x

    def _pairwise_mul_nodes(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pairwise multiplication of nodes for attention map.
        x:          (#bs, #node, #dim)
        out_shape:  (#bs, #node, #node, #dim)
        """
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)
        return x * x_mirror

    def _derive_att_map(self, x: torch.Tensor) -> torch.Tensor:
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))
        att_map = torch.matmul(att_map, self.att_weight)
        att_map = att_map / self.temp
        att_map = F.softmax(att_map, dim=-2)
        return att_map

    def _project(self, x: torch.Tensor, att_map: torch.Tensor) -> torch.Tensor:
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _apply_BN(self, x: torch.Tensor) -> torch.Tensor:
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)
        return x

    @staticmethod
    def _init_new_params(*size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


# ====================================================================
# HtrgGraphAttentionLayer (HS-GAL) — Heterogeneous with Master Node
# ====================================================================

class HtrgGraphAttentionLayer(nn.Module):
    """
    Heterogeneous Stacking Graph Attention Layer (HS-GAL).

    Handles two types of nodes (spectral + temporal) with separate
    attention weights for intra-type and inter-type edges, plus a
    learnable master/stack node that aggregates information unidirectionally.

    Input:  x1 (#bs, #node1, #dim), x2 (#bs, #node2, #dim), master (#bs, 1, #dim)
    Output: x1 (#bs, #node1, out_dim), x2 (#bs, #node2, out_dim), master (#bs, 1, out_dim)
    """

    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        super().__init__()

        # Type projections
        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)

        # Attention map projections
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)

        # Separate attention weights for different edge types
        self.att_weight11 = self._init_new_params(out_dim, 1)  # type1 ↔ type1
        self.att_weight22 = self._init_new_params(out_dim, 1)  # type2 ↔ type2
        self.att_weight12 = self._init_new_params(out_dim, 1)  # type1 ↔ type2
        self.att_weightM = self._init_new_params(out_dim, 1)   # master attention

        # Projection (with and without attention)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        # Master node projection
        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)

        # Batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # Dropout
        self.input_drop = nn.Dropout(p=0.2)

        # Activation
        self.act = nn.SELU(inplace=True)

        # Temperature
        self.temp = kwargs.get("temperature", 1.0)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        master: torch.Tensor = None,
    ):
        """
        Args:
            x1:     (#bs, #node1, #dim) — e.g. temporal nodes
            x2:     (#bs, #node2, #dim) — e.g. spectral nodes
            master: (#bs, 1, #dim) — learnable stack node
        Returns:
            x1, x2, master — all projected to out_dim
        """
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)

        # Project each type
        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)

        # Concatenate
        x = torch.cat([x1, x2], dim=1)

        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)

        # Apply input dropout
        x = self.input_drop(x)

        # Derive attention map (heterogeneous)
        att_map = self._derive_att_map(x, num_type1, num_type2)

        # Update master node (directional edge)
        master = self._update_master(x, master)

        # Projection
        x = self._project(x, att_map)

        # Batch norm + activation
        x = self._apply_BN(x)
        x = self.act(x)

        # Split back into type1 and type2
        x1 = x.narrow(1, 0, num_type1)
        x2 = x.narrow(1, num_type1, num_type2)

        return x1, x2, master

    def _update_master(
        self, x: torch.Tensor, master: torch.Tensor
    ) -> torch.Tensor:
        att_map = self._derive_att_map_master(x, master)
        master = self._project_master(x, master, att_map)
        return master

    def _pairwise_mul_nodes(self, x: torch.Tensor) -> torch.Tensor:
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)
        return x * x_mirror

    def _derive_att_map_master(
        self, x: torch.Tensor, master: torch.Tensor
    ) -> torch.Tensor:
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))
        att_map = torch.matmul(att_map, self.att_weightM)
        att_map = att_map / self.temp
        att_map = F.softmax(att_map, dim=-2)
        return att_map

    def _derive_att_map(
        self, x: torch.Tensor, num_type1: int, num_type2: int
    ) -> torch.Tensor:
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))

        # Heterogeneous attention: different weights for different edge types
        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)

        # type1 ↔ type1
        att_board[:, :num_type1, :num_type1, :] = torch.matmul(
            att_map[:, :num_type1, :num_type1, :], self.att_weight11
        )
        # type2 ↔ type2
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(
            att_map[:, num_type1:, num_type1:, :], self.att_weight22
        )
        # type1 → type2
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(
            att_map[:, :num_type1, num_type1:, :], self.att_weight12
        )
        # type2 → type1
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(
            att_map[:, num_type1:, :num_type1, :], self.att_weight12
        )

        att_map = att_board / self.temp
        att_map = F.softmax(att_map, dim=-2)
        return att_map

    def _project(self, x: torch.Tensor, att_map: torch.Tensor) -> torch.Tensor:
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _project_master(
        self, x: torch.Tensor, master: torch.Tensor, att_map: torch.Tensor
    ) -> torch.Tensor:
        x1 = self.proj_with_attM(
            torch.matmul(att_map.squeeze(-1).unsqueeze(1), x)
        )
        x2 = self.proj_without_attM(master)
        return x1 + x2

    def _apply_BN(self, x: torch.Tensor) -> torch.Tensor:
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)
        return x

    @staticmethod
    def _init_new_params(*size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


# ====================================================================
# GraphPool — Top-K Graph Pooling
# ====================================================================

class GraphPool(nn.Module):
    """
    Top-k graph pooling with learned scoring.

    Selects top-k fraction of nodes based on learned attention scores,
    weighted by sigmoid gating.

    Input:  h — (#bs, #node, #dim)
    Output: h — (#bs, #node', #dim) where #node' = ceil(k * #node)
    """

    def __init__(self, k: float, in_dim: int, p: float):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.drop(h)
        weights = self.proj(z)
        scores = self.sigmoid(weights)
        new_h = self._top_k_graph(scores, h, self.k)
        return new_h

    @staticmethod
    def _top_k_graph(
        scores: torch.Tensor, h: torch.Tensor, k: float
    ) -> torch.Tensor:
        """
        Args:
            scores: (#bs, #node, 1)
            h:      (#bs, #node, #dim)
            k:      ratio of remaining nodes (float in (0, 1])
        Returns:
            h:      (#bs, #node', #dim) — pooled graph
        """
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)

        h = h * scores
        h = torch.gather(h, 1, idx)
        return h
