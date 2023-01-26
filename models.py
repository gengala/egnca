from pools import GaussianMultiSeedPool, GaussianSeedPool
from decoders import EuclideanDecoder
from egnn import EGC, EGNN
from utils.utils import *

from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import PairNorm
from typing import Optional, List
from argparse import Namespace
import pytorch_lightning as pl
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch


class NodeNorm(nn.Module):
    def __init__(
        self,
        unbiased: Optional[bool] = False,
        eps: Optional[float] = 1e-5,
        root_power: Optional[float] =3
    ):
        super(NodeNorm, self).__init__()
        self.unbiased = unbiased
        self.eps = eps
        self.power = 1 / root_power

    def forward(self, x: torch.Tensor):
        std = (torch.var(x, unbiased=self.unbiased, dim=-1, keepdim=True) + self.eps).sqrt()
        x = x / torch.pow(std, self.power)
        return x

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class EncoderEGNCA(nn.Module):

    def __init__(
        self,
        coord_dim: int,
        node_dim: int,
        message_dim: int,
        init_rand_node_feat: Optional[bool] = False,
        act_name: Optional[str] = 'tanh',
        n_layers: Optional[int] = 1,
        std: Optional[float] = None,
        is_residual: Optional[bool] = True,
        has_attention: Optional[bool] = False,
        has_coord_act: Optional[bool] = True,
        fire_rate: Optional[float] = 1.0,
        norm_type: Optional[str] = None,
        norm_cap: Optional[float] = None,
    ):
        super(EncoderEGNCA, self).__init__()
        assert norm_type is None or norm_type == 'nn' or norm_type == 'pn'
        assert message_dim >= node_dim
        assert 0 < fire_rate <= 1.0

        self.std = std
        self.fire_rate = fire_rate
        self.init_rand_node_feat = init_rand_node_feat

        if norm_type == 'nn':
            self.normalise = NodeNorm(root_power=2.0 if norm_cap is None else norm_cap)
        elif norm_type == 'pn':
            self.normalise = PairNorm(scale=1.0 if norm_cap is None else norm_cap)
        else:
            self.normalise = None

        layers = []
        for _ in range(n_layers):
            layers.append(EGC(
                coord_dim=coord_dim,
                node_dim=node_dim,
                message_dim=message_dim,
                act_name=act_name,
                is_residual=is_residual,
                has_attention=has_attention,
                has_coord_act=has_coord_act))
        self.egnn = EGNN(layers)

    @property
    def coord_dim(self):
        return self.egnn.layers[0].coord_dim

    @property
    def node_dim(self):
        return self.egnn.layers[0].node_dim

    def init_coord(
        self,
        num_nodes: int,
        device: Optional[str] = 'cpu',
        dtype: Optional[torch.dtype] = torch.float32
    ):
        coord = torch.empty(num_nodes, self.coord_dim, dtype=dtype, device=device).normal_(self.std)
        return coord

    def init_node_feat(
        self,
        num_nodes: int,
        device: Optional[str] = 'cpu',
        dtype: Optional[torch.dtype] = torch.float32
    ):
        if self.init_rand_node_feat:
            node_feat = torch.empty(num_nodes, self.node_dim, dtype=dtype, device=device).normal_(self.std)
        else:
            node_feat = torch.ones(num_nodes, self.node_dim, dtype=dtype, device=device)
        return node_feat

    def stochastic_update(
        self,
        edge_index: torch.LongTensor,
        in_coord: torch.Tensor,
        in_node_feat: torch.Tensor,
        n_nodes: Optional[torch.LongTensor] = None
    ):
        assert 0 < self.fire_rate <= 1
        out_coord, out_node_feat = self.egnn(edge_index=edge_index, coord=in_coord, node_feat=in_node_feat)
        if isinstance(self.normalise, NodeNorm):
            out_node_feat = self.normalise(out_node_feat)
        elif isinstance(self.normalise, PairNorm):
            out_node_feat = self.normalise(out_node_feat, n_nodes if n_nodes is None else n_nodes2batch(n_nodes))
        if 0 < self.fire_rate < 1:
            mask = (torch.rand(out_coord.size(0), 1) <= self.fire_rate).byte().to(in_coord.device)
            out_coord = (out_coord * mask) + (in_coord * (1 - mask))
            out_node_feat = (out_node_feat * mask) + (in_node_feat * (1 - mask))
        return out_coord, out_node_feat

    def forward(
        self,
        edge_index: torch.LongTensor,
        coord: Optional[torch.Tensor] = None,
        node_feat: Optional[torch.Tensor] = None,
        n_steps: Optional[int] = 1,
        n_nodes: Optional[torch.LongTensor] = None,
        return_inter_states: Optional[bool] = False,
        progress_bar: Optional[bool] = False,
        dtype: Optional[torch.dtype] = torch.float32
    ):
        if coord is None:
            num_nodes = edge_index[0].max() + 1 if n_nodes is None else n_nodes.sum().item()
            coord = self.init_coord(num_nodes, dtype=dtype, device=edge_index.device)
        if node_feat is None:
            node_feat = self.init_node_feat(coord.size(0), dtype=dtype, device=coord.device)

        loop = tqdm(range(n_steps)) if progress_bar else range(n_steps)
        inter_states = [(coord, node_feat)] if return_inter_states else None
        for _ in loop:
            coord, node_feat = self.stochastic_update(edge_index, coord, node_feat, n_nodes)
            if return_inter_states: inter_states.append((coord, node_feat))

        return list(map(list, zip(*inter_states))) if return_inter_states else (coord, node_feat)


class FixedTargetGAE(pl.LightningModule):

    def __init__(
        self,
        args: Namespace
    ):
        super().__init__()

        # load target geometric graph as model attribute
        from data.datasets import get_geometric_graph
        target_coord, edge_index = get_geometric_graph(args.dataset)
        self.register_buffer('target_coord', target_coord * args.scale)
        self.register_buffer('edge_index', edge_index)

        self.encoder = EncoderEGNCA(
            coord_dim=self.target_coord.size(1),
            node_dim=args.node_dim,
            message_dim=args.message_dim,
            n_layers=args.n_layers,
            std=args.std,
            act_name=args.act,
            is_residual=args.is_residual,
            has_attention=args.has_attention,
            has_coord_act=args.has_coord_act,
            fire_rate=args.fire_rate,
            norm_type=args.norm_type)

        self.pool = GaussianSeedPool(
            pool_size=args.pool_size,
            num_nodes=self.target_coord.size(0),
            coord_dim=self.target_coord.size(1),
            node_dim=args.node_dim,
            std=args.std,
            std_damage=args.std_damage,
            radius_damage=args.std_damage,
            device=args.device,
            fixed_init_coord=True)

        self.register_buffer('init_coord', self.pool.init_coord.clone())
        self.mse = nn.MSELoss(reduction='none')

        self.args = args
        self.save_hyperparameters(ignore=['pool'])

    def training_step(
        self,
        batch: Data,
        batch_idx: int
    ):
        # next line increase batch size by increasing dataset length
        self.trainer.train_dataloader.loaders.dataset.length = \
            list_scheduler_step(self.args.batch_sch, self.current_epoch)
        batch_size = len(batch.n_nodes)

        n_steps = np.random.randint(self.args.n_min_steps, self.args.n_max_steps + 1)
        init_coord, init_node_feat, id_seeds = self.pool.get_batch(batch_size=batch_size)
        final_coord, final_node_feat = self.encoder(
            batch.edge_index, init_coord, init_node_feat, n_steps=n_steps, n_nodes=batch.n_nodes)

        edge_weight = torch.norm(final_coord[batch.rand_edge_index[0]] - final_coord[batch.rand_edge_index[1]], dim=-1)
        loss_per_edge = self.mse(edge_weight, batch.rand_edge_weight)
        loss_per_graph = torch.stack([lpe.mean() for lpe in loss_per_edge.chunk(batch_size)])
        loss = loss_per_graph.mean()
        self.pool.update(id_seeds, final_coord, final_node_feat, losses=loss_per_graph)

        # display & log
        print('%d \t %.6f \t %d \t %.6f \t %.2f' %
              (self.current_epoch, loss, batch_size,
               self.trainer.optimizers[0].param_groups[0]['lr'], self.pool.avg_reps))
        self.log('loss', loss, on_step=True, on_epoch=False, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.encoder.parameters(), lr=self.args.lr, betas=(self.args.b1, self.args.b2), weight_decay=self.args.wd
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=self.args.factor_sch,
            patience=self.args.patience_sch,
            min_lr=1e-5,
            verbose=True,
        )
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': 'loss'}

    @torch.no_grad()
    def eval(
        self,
        n_steps: int,
        init_coord: Optional[torch.Tensor] = None,
        init_node_feat: Optional[torch.Tensor] = None,
        rotate: Optional[bool] = False,
        translate: Optional[bool] = False,
        return_inter_states: Optional[bool] = False,
        progress_bar: Optional[bool] = True,
        dtype: Optional[torch.dtype] = torch.float64
    ):
        self.to(dtype)
        if init_coord is None:
            init_coord = self.init_coord.clone()
        if rotate:
            rotation = nn.init.orthogonal_(
                torch.empty(self.encoder.coord_dim, self.encoder.coord_dim)
            ).to(device=self.device, dtype=dtype)
            init_coord = torch.matmul(rotation, init_coord.T).T
        if translate:
            translation = torch.randn(1, self.encoder.coord_dim).to(device=self.device, dtype=dtype)
            init_coord += translation
        out = self.encoder(
            self.edge_index, coord=init_coord, node_feat=init_node_feat, n_steps=n_steps,
            return_inter_states=return_inter_states, progress_bar=progress_bar)
        return out

    @torch.no_grad()
    def eval_persistency(
        self,
        n_step_list: Optional[List[int]] = None,
        init_coord: Optional[torch.Tensor] = None,
        init_node_feat: Optional[torch.Tensor] = None,
        return_final_state: Optional[bool] = False,
        dtype: Optional[torch.dtype] = torch.float64
    ):
        self.to(dtype)
        if n_step_list is None:
            s1, s2 = self.args.n_min_steps, self.args.n_max_steps
            n_step_list = [s1, (s1 + s2) // 2, s2] + list(range(100, 1100, 100)) + list(range(10_000, 110_000, 10_000))
        if init_coord is None:
            init_coord = self.init_coord.clone()
        if init_node_feat is None:
            init_node_feat = self.init_coord.new_ones(init_coord.shape[0], self.encoder.node_dim)
        coord, node_feat = init_coord, init_node_feat
        results, progress_bar = dict(), tqdm(range(max(n_step_list) + 1))
        for n_step in progress_bar:
            if n_step in n_step_list:
                results[n_step] = coord_invariant_rec_loss(coord, self.target_coord)
                progress_bar.set_postfix_str('[step %d] [loss: %.5f]' % (n_step, results[n_step]), refresh=False)
            coord, node_feat = self.encoder(self.edge_index, coord, node_feat)
        return (results, coord, node_feat) if return_final_state else results


class GAE(pl.LightningModule):

    def __init__(
        self,
        args: Namespace
    ):
        super().__init__()

        self.encoder = EncoderEGNCA(
            coord_dim=args.coord_dim,
            node_dim=args.node_dim,
            message_dim=args.message_dim,
            n_layers=args.n_layers,
            std=args.std,
            act_name=args.act,
            is_residual=args.is_residual,
            has_attention=args.has_attention,
            has_coord_act=args.has_coord_act,
            fire_rate=args.fire_rate,
            norm_type=args.norm_type,
            norm_cap=args.norm_cap)

        self.decoder = EuclideanDecoder(
            d1=args.d1,
            d2=args.d2,
            learnable=args.learn_dec)

        self.pool = None
        if args.pool_size and args.rep_sch:
            self.pool = GaussianMultiSeedPool(
                pool_size=args.pool_size,
                coord_dim=args.coord_dim,
                node_dim=args.node_dim,
                std=args.std,
                device=args.device,
                init_rand_node_feat=args.init_rand_node_feat)

        self.args = args
        self.save_hyperparameters(ignore=['pool'])

    def on_train_epoch_start(self):
        if self.pool:
            self.pool.max_rep = list_scheduler_step(self.args.rep_sch, self.current_epoch)

    def _step(
        self,
        batch: Data,
        train: bool
    ):
        n_steps = np.random.randint(self.args.n_min_steps, self.args.n_max_steps + 1)
        if self.pool:
            init_coord, init_node_feat, id_seeds = self.pool.get_batch(batch.id_graphs, batch.n_nodes)
            final_coord, final_node_feat = self.encoder(
                batch.edge_index, init_coord, init_node_feat, n_steps=n_steps, n_nodes=batch.n_nodes)
            self.pool.update(final_coord, final_node_feat, batch.id_graphs, id_seeds)
        else:
            final_coord, final_node_feat = self.encoder(batch.edge_index, n_steps=n_steps)

        neg_edge_index, n_neg_edges = batched_neg_index_sampling(
            batch.neg_edge_index, batch.n_neg_edges, torch.div(batch.n_edges, 2, rounding_mode='trunc'))
        loss = self.decoder.bce(final_coord, batch.edge_index, neg_edge_index)

        # display log
        avg_reps = -1 if self.pool is None else self.pool.avg_reps
        print('%s \t %d \t %.5f \t %.6f \t %.2f' %
              ('TR' if train else 'VA', self.current_epoch, loss,
               self.trainer.optimizers[0].param_groups[0]['lr'], avg_reps))
        return loss

    def training_step(
        self,
        batch: Data,
        batch_idx: int
    ):
        loss = self._step(batch, train=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, batch_size=len(batch.id_graphs))
        return loss

    def validation_step(
        self,
        batch: Data,
        batch_idx: int
    ):
        loss = self._step(batch, train=False)
        self.log('val_loss', loss, on_step=True, on_epoch=True, batch_size=len(batch.id_graphs))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters(), 'lr': self.args.lr,
              'betas': (self.args.b1, self.args.b2), 'weight_decay': self.args.wd},
            {'params': self.decoder.parameters(), 'lr': self.args.dlr,
             'betas': (self.args.b1, self.args.b2), 'weight_decay': 0}
        ])
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=self.args.factor_sch,
            patience=self.args.patience_sch,
            min_lr=1e-5,
            verbose=True,
        )
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': 'val_loss_epoch'}

    @torch.no_grad()
    def eval_dataset(
        self,
        dataset: Dataset,
        n_steps: Optional[int] = 1,
        threshold: Optional[float] = 0.5,
        progress_bar_encoder: Optional[bool] = False,
        dtype: Optional[torch.dtype] = torch.float64
    ):
        self.to(dtype)
        self.decoder.threshold = threshold
        pred_coord_list, pred_edge_index_list = [], []
        for graph in tqdm(dataset):
            pred_coord_list.append(self.encoder(
                graph.edge_index.to(self.device), n_steps=n_steps, progress_bar=progress_bar_encoder, dtype=dtype)[0])
            pred_edge_index_list.append(self.decoder.decode_adj(pred_coord_list[-1])[0])
        return pred_coord_list, pred_edge_index_list

    @torch.no_grad()
    def eval_persistency(
        self,
        dataset: Dataset,
        n_step_list: Optional[List[int]] = None,
        threshold: Optional[float] = 0.5,
        n_evaluations: Optional[int] = 1,
        batch_size: Optional[int] = None,
        average_results: Optional[bool] = True,
        dtype: Optional[torch.dtype] = torch.float64
    ):
        self.to(dtype)
        self.decoder.threshold = threshold
        if n_step_list is None:
            s1, s2 = self.args.n_min_steps, self.args.n_max_steps
            n_step_list = [s1, (s1 + s2) // 2, s2] + list(range(100, 1100, 100)) + list(range(10_000, 110_000, 10_000))
        results = {n_step: {'bce': [], 'f1': [], 'cm': []} for n_step in n_step_list}
        loader = DataLoader(dataset, batch_size=len(dataset) if batch_size is None else batch_size, shuffle=True)
        tot_n_steps = max(n_step_list) + 1
        with tqdm(total=n_evaluations * len(loader) * tot_n_steps) as progress_bar:
            for _ in range(n_evaluations):
                for batch in loader:
                    coord = self.encoder.init_coord(batch.n_nodes.sum(), dtype=dtype, device=self.device)
                    node_feat = self.encoder.init_node_feat(coord.size(0), dtype=dtype, device=self.device)
                    for n_step in range(tot_n_steps):
                        if n_step in n_step_list:
                            results[n_step]['bce'].append(
                                self.decoder.bce(coord, batch.edge_index, batch.neg_edge_index).item())
                            pred_edge_index = self.decoder.decode_adj(coord, n_nodes=batch.n_nodes)[0]
                            cm, f1 = edge_cm(batch.edge_index, pred_edge_index, batch.n_nodes, True, True)
                            results[n_step]['cm'].append(cm)
                            results[n_step]['f1'].append(f1)
                            progress_bar.set_postfix_str('[step %d] [f1: %.5f]' %
                                                         (n_step, results[n_step]['f1'][-1]), refresh=False)
                        coord, node_feat = self.encoder(
                            batch.edge_index, coord, node_feat, n_nodes=batch.n_nodes, dtype=dtype)
                        progress_bar.update(1)
        if average_results:
            for key_1 in results:
                for key_2 in results[key_1]:
                    results[key_1][key_2] = (np.mean(results[key_1][key_2], 0), np.std(results[key_1][key_2], 0))
        return results

    @torch.no_grad()
    def threshold_tuning(
        self,
        dataset: Dataset,
        n_steps: Optional[int] = None,
        thresholds: List[int] = None,
        n_evaluations: Optional[int] = 1,
        batch_size: Optional[int] = None,
        dtype: Optional[torch.dtype] = torch.float64
    ):
        self.to(dtype)
        if n_steps is None:
            n_steps = self.args.n_max_steps
        if thresholds is None:
            thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
        f1_dict = {threshold: [] for threshold in thresholds}
        loader = DataLoader(dataset, batch_size=len(dataset) if batch_size is None else batch_size, shuffle=True)
        with tqdm(total=n_evaluations * len(loader) * len(thresholds)) as progress_bar:
            for _ in range(n_evaluations):
                for batch in loader:
                    final_coord = self.encoder(
                        batch.edge_index.to(self.device), n_steps=n_steps, dtype=dtype)[0]
                    for threshold in thresholds:
                        self.decoder.threshold = threshold
                        pred_edge_index = self.decoder.decode_adj(final_coord)[0]
                        f1_dict[threshold].append(
                            edge_cm(batch.edge_index, pred_edge_index, batch.n_nodes, return_f1=True)[1])
                        progress_bar.update(1)
        for threshold in thresholds:
            f1_dict[threshold] = np.mean(f1_dict[threshold])
        best_threshold = max(f1_dict, key=f1_dict.get)
        return best_threshold


class SimulatorEGNCA(pl.LightningModule):

    def __init__(
        self,
        args: Namespace
    ):
        super().__init__()

        self.vel2node_feat = nn.Linear(1, args.node_dim)
        layers = []
        for _ in range(args.n_layers):
            layers.append(EGC(
                coord_dim=3,
                node_dim=args.node_dim,
                message_dim=args.message_dim,
                act_name=args.act,
                is_residual=args.is_residual,
                has_attention=args.has_attention,
                has_coord_act=args.has_coord_act,
                has_vel_norm=args.has_vel_norm,
                has_vel=True))
        self.egnn = EGNN(layers)

        # if decoder is None, a full adjacency will be used
        self.decoder = None if args.radius is None else EuclideanDecoder(d1=args.radius, sqrt=True)

        # if box_dim is given, the simulation will take place in a box
        self.box_dim = args.box_dim
        if args.box_dim is not None:
            self.box_strength = nn.Parameter(torch.tensor([0.1]))

        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.args = args
        self.save_hyperparameters()

    def avoid_borders(
        self,
        coord: torch.Tensor,
        vel: torch.Tensor
    ):
        if self.box_dim is not None:
            vel_steer = (coord < - self.box_dim) * self.box_strength - (coord > self.box_dim) * self.box_strength
            vel = vel + vel_steer
            coord = coord + vel_steer
        return coord, vel

    def forward(
        self,
        coord: torch.Tensor,
        vel: torch.Tensor,
        n_steps: Optional[int] = 1,
        node_feat: Optional[torch.Tensor] = None,
        n_nodes: Optional[torch.LongTensor] = None
    ):
        assert coord.size() == vel.size() and (coord.ndim == 2 or coord.ndim == 3)

        if n_nodes is None:
            n_nodes = torch.LongTensor([len(coord)] if coord.ndim == 2 else [coord.size(1)] * len(coord)).to(self.device)
        if node_feat is None:
            node_feat = self.vel2node_feat(torch.norm(vel, p=2, dim=-1, keepdim=True))
        if self.decoder is None:
            edge_index = fully_connected_adj(n_nodes, sparse=coord.ndim == 2)

        coords, vels = [coord.clone()], [vel.clone()]
        for _ in range(n_steps):
            if self.decoder is not None:
                edge_index = self.decoder.decode_adj(coord, n_nodes)[0]
            coord, node_feat, vel = self.egnn(coord, node_feat, edge_index, vel=vel, n_nodes=n_nodes)
            coord, vel = self.avoid_borders(coord, vel)
            coords.append(coord)
            vels.append(vel)

        # if len(n_nodes) > 1, as batch is being processed
        return (coords, vels) if len(n_nodes) > 1 else (torch.stack(coords).squeeze(), torch.stack(vels).squeeze())

    def training_val_step(
        self,
        batch: List[torch.Tensor],
        train: bool
    ):
        # coord_traj_true and vel_traj_true are 4D tensors of shape (batch size, traj length, num nodes, coord dim)
        coord_traj_true, vel_traj_true = batch
        n_nodes = torch.LongTensor([vel_traj_true.size(2)] * vel_traj_true.size(0)).to(self.device)

        in_coord = coord_traj_true[:, 0].reshape(-1, 3) if self.args.sparse_training else coord_traj_true[:, 0]
        in_vel = vel_traj_true[:, 0].reshape(-1, 3) if self.args.sparse_training else vel_traj_true[:, 0]
        vel_traj_pred = self.forward(in_coord, in_vel, n_steps=vel_traj_true.size(1) - 1, n_nodes=n_nodes)[1]

        if self.args.sparse_training:
            vel_traj_pred = [v.reshape(-1, vel_traj_true.size(2), vel_traj_true.size(3)) for v in vel_traj_pred]
        loss = self.criterion(torch.cat([v.unsqueeze(1) for v in vel_traj_pred], dim=1)[:, 1:], vel_traj_true[:, 1:])

        # display training info
        print('%s \t %d \t %.5f \t %.6f \t %d' % (
            'TR' if train else 'VA', self.current_epoch, loss,
            self.trainer.optimizers[0].param_groups[0]['lr'], vel_traj_true.size(1)))

        return loss

    def training_step(
        self,
        batch: List[torch.Tensor],
        batch_idx: int
    ):
        loss = self.training_val_step(batch, train=True)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch[0]))
        return loss

    def validation_step(
        self,
        batch: List[torch.Tensor],
        batch_idx: int
    ):
        loss = self.training_val_step(batch, train=False)
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch[0]))
        return loss

    def on_train_epoch_start(self):
        old_seq_len = self.trainer.train_dataloader.dataset.datasets.dataset.seq_len
        new_seq_len = list_scheduler_step(self.args.seq_len_sch, self.current_epoch)
        if old_seq_len != new_seq_len:
            self.trainer.train_dataloader.dataset.datasets.dataset.seq_len = new_seq_len
            print('Training with sequences of length %d..' % new_seq_len)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.parameters(), 'lr': self.args.lr,
              'betas': (self.args.b1, self.args.b2), 'weight_decay': self.args.wd},
        ])
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=self.args.factor_sch,
            patience=self.args.patience_sch,
            min_lr=1e-5,
            verbose=True
        )
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': 'val_loss_epoch'}
