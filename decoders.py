from typing import Optional
import torch.nn as nn
import torch


def aggregated_sum(
    data: torch.Tensor,
    index: torch.LongTensor,
    num_segments: int,
    mean: bool = False
):
    index = index.unsqueeze(1).repeat(1, data.size(1))
    agg = data.new_full((num_segments, data.size(1)), 0).scatter_add_(0, index, data)
    if mean:
        counts = data.new_full((num_segments, data.size(1)), 0).scatter_add_(0, index, torch.ones_like(data))
        agg = agg / counts.clamp(min=1)
    return agg


class EuclideanDecoder(torch.nn.Module):

    def __init__(
        self,
        d1: Optional[float] = 1.0,
        d2: Optional[float] = 1.0,
        threshold: Optional[float] = 0.5,
        learnable: Optional[bool] = False,
        sqrt: Optional[bool] = False
    ):
        super(EuclideanDecoder, self).__init__()
        self.d1 = nn.Parameter(torch.ones(1) * d1) if learnable else d1
        self.d2 = nn.Parameter(torch.ones(1) * d2) if learnable else d2
        self.threshold = threshold
        self.learnable = learnable
        self.sqrt = sqrt
        self.criterion = nn.BCELoss(reduction='none')

    def sigmoid(
        self,
        dist: torch.Tensor
    ):
        # https://www.desmos.com/calculator/mkp3ewfmiu
        exp = (self.d2 * (dist - self.d1))
        return torch.sigmoid(-exp)

    def decode_edge(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        sigmoid: Optional[bool] = True
    ):
        if z.dim() == 3:
            return self.decode_edge_dense(z, edge_index, sigmoid)
        else:
            return self.decode_edge_sparse(z, edge_index, sigmoid)

    def decode_adj(
        self,
        z: torch.Tensor,
        n_nodes: Optional[torch.LongTensor] = None
    ):
        if z.dim() == 3:
            return self.decode_adj_dense(z, n_nodes)
        else:
            return self.decode_adj_sparse(z, n_nodes)

    def decode_edge_sparse(
        self,
        z: torch.Tensor,
        edge_index: torch.LongTensor,
        sigmoid: Optional[bool] = True
    ):
        dist = ((z[edge_index[0]] - z[edge_index[1]]) ** 2).sum(dim=1)
        if self.sqrt:
            dist = dist.sqrt()
        return self.sigmoid(dist) if sigmoid else dist

    def decode_edge_dense(
        self,
        z: torch.Tensor,
        adj: torch.Tensor,
        sigmoid: Optional[bool] = True
    ):
        dist = ((z.unsqueeze(2) - z.unsqueeze(1)) ** 2).sum(dim=-1)
        if self.sqrt:
            dist = dist.sqrt()
        out = (self.sigmoid(dist) if sigmoid else dist) * adj
        return out

    @torch.no_grad()
    def decode_adj_sparse(
        self,
        z: torch.Tensor,
        n_nodes: Optional[torch.LongTensor] = None,
        sort: Optional[bool] = False
    ):
        if n_nodes is None:
            edge_index = torch.ones(z.size(0), z.size(0)).tril(-1).nonzero().T.to(z.device)
        else:
            offset = [0] + torch.Tensor.tolist(n_nodes[:-1].cumsum(0))
            edge_index = torch.cat(
                [torch.ones(n, n).tril(-1).nonzero().T + o for o, n in zip(offset, n_nodes)], dim=1).to(z.device)

        edge_weight = self.decode_edge_sparse(z, edge_index, sigmoid=True)
        # the higher the threshold the sparser the graph
        mask = edge_weight >= self.threshold
        edge_index = edge_index[:, mask]
        edge_index = torch.cat([edge_index, torch.cat([edge_index[1:], edge_index[:1]], dim=0)], dim=1)
        edge_weight = edge_weight[mask].repeat(2)

        if sort:
            perm = edge_index[0].argsort()
            edge_index = edge_index[:, perm]
            edge_weight = edge_weight[perm]

        return edge_index, edge_weight

    def decode_adj_dense(
        self,
        z: torch.Tensor,
        n_nodes: Optional[torch.LongTensor] = None
    ):
        assert z.dim() == 3
        if n_nodes is None:
            n_nodes = torch.full((z.size(0),), z.size(1)).to(z.device)
        max_n_nodes = n_nodes.max()
        assert z.size(1) == max_n_nodes

        dist = ((z.unsqueeze(2) - z.unsqueeze(1)) ** 2).sum(-1)
        if self.sqrt:
            dist = dist.sqrt()
        weight = self.sigmoid(dist)

        adj = torch.zeros(n_nodes.size(0), max_n_nodes, max_n_nodes, dtype=torch.uint8).to(z.device)
        for i in range(n_nodes.size(0)):
            adj[i, :n_nodes[i], :n_nodes[i]] = 1
            adj[i].fill_diagonal_(0)
        adj = torch.logical_and(adj, weight >= self.threshold).to(torch.uint8)
        adj_weight = weight * adj

        # adj has no gradients, adj_weight does
        return adj, adj_weight

    def bce(
        self,
        z: torch.Tensor,
        pos_edge_index: torch.LongTensor,
        neg_edge_index: torch.LongTensor,
        neg_weight: Optional[float] = 1.0
    ):
        pos_loss = self.criterion(
            input=self.decode_edge_sparse(z, pos_edge_index, sigmoid=True),
            target=torch.ones(1, dtype=z.dtype, device=z.device).expand(pos_edge_index.size(1))).mean()
        neg_loss = self.criterion(
            input=self.decode_edge_sparse(z, neg_edge_index, sigmoid=True),
            target=torch.zeros(1, dtype=z.dtype, device=z.device).expand(neg_edge_index.size(1))).mean()
        loss = pos_loss + neg_weight * neg_loss
        return loss

    def bce_per_graph(
        self,
        z: torch.Tensor,
        pos_edge_index: torch.LongTensor,
        neg_edge_index: torch.LongTensor,
        n_pos_edges: torch.LongTensor,
        n_neg_edges: torch.LongTensor
    ):
        assert n_pos_edges.ndim == n_neg_edges.ndim == 1 and len(n_pos_edges) == len(n_neg_edges)
        num_graphs = len(n_pos_edges)

        pos_bce_per_edge = self.criterion(
            input=self.decode_edge_sparse(z, pos_edge_index, sigmoid=True),
            target=torch.ones(1, dtype=z.dtype, device=z.device).expand(pos_edge_index.size(1))).unsqueeze(1)
        neg_bce_per_edge = self.criterion(
            input=self.decode_edge_sparse(z, neg_edge_index, sigmoid=True),
            target=torch.zeros(1, dtype=z.dtype, device=z.device).expand(neg_edge_index.size(1))).unsqueeze(1)

        pos_index = torch.arange(num_graphs, device=n_pos_edges.device).repeat_interleave(n_pos_edges)
        pos_bce_per_graph = aggregated_sum(pos_bce_per_edge, pos_index, num_graphs, mean=True)
        neg_index = torch.arange(num_graphs, device=n_neg_edges.device).repeat_interleave(n_neg_edges)
        neg_bce_per_graph = aggregated_sum(neg_bce_per_edge, neg_index, num_graphs, mean=True)

        bce_per_graph = pos_bce_per_graph + neg_bce_per_graph
        return bce_per_graph

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(learnable={self.learnable})'
