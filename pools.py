from typing import Optional
import numpy as np
import torch


def damage_coord(
    coord: Optional[torch.Tensor],
    std: Optional[float] = 0.05,
    radius: Optional[float] = None
):
    assert coord.ndim == 2 or coord.ndim == 3
    if coord.ndim == 2:
        coord = coord.unsqueeze(0)
    if radius is None:
        coord = coord + torch.empty_like(coord).normal_(std=std)
    else:
        id_center = torch.randint(coord.size(1), size=(coord.size(0),))
        dist = ((coord - coord[torch.arange(len(coord)), id_center].unsqueeze(1)) ** 2).sqrt().sum(-1, keepdim=True)
        coord = coord + (dist < radius) * torch.empty_like(coord).normal_(std=std)
    return coord.squeeze()


class GaussianMultiSeedPool:

    def __init__(
        self,
        pool_size: int,
        coord_dim: int,
        node_dim: int,
        std: Optional[float] = 0.50,
        init_rand_node_feat: Optional[bool] = True,
        max_rep: Optional[int] = 1,
        device: Optional[str] = 'cuda'
    ):
        self.pool_size = pool_size
        self.coord_dim = coord_dim
        self.node_dim = node_dim
        self.std = std
        self.cache = dict()
        self.init_rand_node_feat = init_rand_node_feat
        self.max_rep = max_rep
        self.device = device

    def init(
        self,
        id_graph: int,
        num_nodes: int
    ):
        graph_cache = dict()
        graph_cache['coord'] = torch.empty(self.pool_size, num_nodes, self.coord_dim).normal_(std=self.std)
        if self.init_rand_node_feat:
            graph_cache['node_feat'] = torch.empty(self.pool_size, num_nodes, self.node_dim).normal_(std=self.std)
        else:
            graph_cache['node_feat'] = torch.ones(self.pool_size, num_nodes, self.node_dim)
        graph_cache['reps'] = [0] * self.pool_size
        graph_cache['num_nodes'] = num_nodes
        self.cache[id_graph] = graph_cache

    @property
    def avg_reps(self):
        all_reps = []
        for id_graph in self.cache:
            all_reps.extend(self.cache[id_graph]['reps'])
        return np.mean(all_reps) if len(all_reps) else -1

    def id_reset(
        self,
        id_graph: int,
        id_seed: int
    ):
        self.cache[id_graph]['coord'][id_seed].normal_(std=self.std)
        if self.init_rand_node_feat:
            self.cache[id_graph]['node_feat'][id_seed].normal_(std=self.std)
        else:
            self.cache[id_graph]['node_feat'][id_seed].fill_(1)
        self.cache[id_graph]['reps'][id_seed] = 0

    def get_batch(
        self,
        id_graphs: torch.LongTensor,
        n_nodes: torch.LongTensor
    ):
        id_seeds = torch.LongTensor(np.random.choice(self.pool_size, len(id_graphs), replace=True)).to(self.device)
        id_graphs_list = id_graphs.tolist()
        id_graph_reset = np.random.choice(id_graphs_list, size=(2, ))
        coord, node_feat = [], []
        for id_graph, num_nodes, id_seed in zip(id_graphs_list, n_nodes, id_seeds):
            if id_graph not in self.cache.keys():
                self.init(id_graph, num_nodes)
            elif self.cache[id_graph]['reps'][id_seed] == self.max_rep or id_graph in id_graph_reset:
                self.id_reset(id_graph, id_seed)
            coord.append(self.cache[id_graph]['coord'][id_seed])
            node_feat.append(self.cache[id_graph]['node_feat'][id_seed])
        coord = torch.cat(coord).to(self.device)
        node_feat = torch.cat(node_feat).to(self.device)
        return coord, node_feat, id_seeds

    def update(
        self,
        coord: torch.Tensor,
        node_feat: torch.Tensor,
        id_graphs: torch.LongTensor,
        id_seeds: torch.LongTensor
    ):
        assert len(id_graphs) == len(id_seeds)
        offset = 0
        for id_graph, id_seed in zip(id_graphs.tolist(), id_seeds):
            num_nodes = self.cache[id_graph]['num_nodes']
            self.cache[id_graph]['coord'][id_seed] = coord[offset: offset + num_nodes].detach().cpu()
            self.cache[id_graph]['node_feat'][id_seed] = node_feat[offset: offset + num_nodes].detach().cpu()
            self.cache[id_graph]['reps'][id_seed] += 1
            offset += num_nodes


class GaussianSeedPool:

    def __init__(
        self,
        pool_size: int,
        num_nodes: int,
        coord_dim: int,
        node_dim: int,
        std: Optional[float] = 0.5,
        sparse: Optional[bool] = True,
        fixed_init_coord: Optional[bool] = True,
        std_damage: Optional[bool] = 0.0,
        radius_damage: Optional[float] = None,
        device: Optional[str] = 'cuda'
    ):
        assert std > 0.0
        assert std_damage >= 0

        self.num_nodes = num_nodes
        self.pool_size = pool_size
        self.coord_dim = coord_dim
        self.node_dim = node_dim
        self.std = std
        self.sparse = sparse
        self.std_damage = std_damage
        self.radius_damage = radius_damage
        self.device = device

        if fixed_init_coord:
            self.init_coord = torch.empty(num_nodes, coord_dim).normal_(std=std)
            self.pool_coord = self.init_coord.clone().unsqueeze(0).repeat(pool_size, 1, 1)
        else:
            self.init_coord = None
            self.pool_coord = torch.empty(pool_size, num_nodes, coord_dim).normal_(std=std)
        self.init_node_feat = torch.ones(num_nodes, node_dim)
        self.pool_node_feat = self.init_node_feat.clone().unsqueeze(0).repeat(pool_size, 1, 1)

        self.pool_loss = torch.full((pool_size, ), torch.inf)
        self.reps = torch.zeros(pool_size, dtype=torch.long)

    @property
    def avg_reps(self):
        return np.mean(self.reps.tolist())

    def get_batch(
        self,
        batch_size: int
    ):
        id_seeds = torch.LongTensor(np.random.choice(self.pool_size, size=(batch_size,), replace=False))
        coord = self.pool_coord[id_seeds].to(self.device)
        node_feat = self.pool_node_feat[id_seeds].to(self.device)

        if self.std_damage > 0:
            # 1/4 of the coord get damaged globally, another 1/4 get damaged locally
            coord[len(coord)//2::2] = damage_coord(coord[len(coord)//2::2], self.std_damage)
            coord[len(coord)//2+1::2] = damage_coord(coord[len(coord)//2+1::2], self.std_damage, self.radius_damage)

        id_reset = self.pool_loss[id_seeds].argmax().item()
        self.reps[id_seeds[id_reset]] = 0
        node_feat[id_reset] = self.init_node_feat.clone().to(self.device)
        if self.init_coord is None:
            coord[id_reset].normal_(std=self.std)
        else:
            coord[id_reset] = self.init_coord.clone().to(self.device)

        if self.sparse:
            coord = coord.view(-1, self.coord_dim)
            node_feat = node_feat.view(-1, self.node_dim)
        return coord, node_feat, id_seeds

    def update(
        self,
        id_seeds: torch.LongTensor,
        coord: torch.Tensor,
        node_feat: torch.Tensor,
        losses: Optional[torch.Tensor] = None
    ):
        assert coord.ndim == node_feat.ndim and coord.size(0) == node_feat.size(0)
        if coord.ndim == 2:
            coord = coord.view(len(id_seeds), self.num_nodes, self.coord_dim)
            node_feat = node_feat.view(len(id_seeds), self.num_nodes, self.node_dim)
        self.pool_coord[id_seeds] = coord.detach().cpu()
        self.pool_node_feat[id_seeds] = node_feat.detach().cpu()
        self.reps[id_seeds] += 1
        if losses is not None:
            self.pool_loss[id_seeds] = losses.detach().cpu()
