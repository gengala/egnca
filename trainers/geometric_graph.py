from data.datasets import get_geometric_graph, GeometricGraphDataset
from models import FixedTargetGAE

from pytorch_lightning.loggers import TensorBoardLogger
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
import argparse
import json
import time
import sys
import os


parser = argparse.ArgumentParser()

parser.add_argument('-device',                  type=str,   default='cuda', help='cpu | cuda')
parser.add_argument('-ds',  '--dataset',        type=str,   default=None,   help='dataset/point cloud name')
parser.add_argument('-sc',  '--scale',          type=float, default=1.0,    help='point cloud scale')
parser.add_argument('-dre', '--dens_rand_edge', type=float, default=1.0,    help='density of rand edges to sample')
parser.add_argument('-ts',  '--training_steps', type=int,   default=100000, help='number of training steps')
parser.add_argument('-pat', '--patience',       type=int,   default=5000,   help='early stopping patience (tr. steps)')

parser.add_argument('-ps',  '--pool_size',      type=int,   default=256,    help='pool size')
parser.add_argument('-bsc', '--batch_sch',      type=int,   default=[0, 8], help='batch size schedule', nargs='+')
parser.add_argument('-sdg', '--std_damage',     type=float, default=0.0,    help='std of coord damage')
parser.add_argument('-rdg', '--radius_damage',  type=float, default=None,   help='radius of coord damage')

parser.add_argument('-nd',  '--node_dim',       type=int,   default=16,     help='node feature dimension')
parser.add_argument('-md',  '--message_dim',    type=int,   default=32,     help='hidden feature dimension')
parser.add_argument('-nl',  '--n_layers',       type=int,   default=1,      help='number of EGNN layers')
parser.add_argument('-nt',  '--norm_type',      type=str,   default='pn',   help='norm type: nn, pn or None')
parser.add_argument('-act',                     type=str,   default='tanh', help='tanh | silu | lrelu')
parser.add_argument('-std',                     type=float, default=0.5,    help='standard deviation of init coord')
parser.add_argument('-s1',  '--n_min_steps',    type=int,   default=15,     help='minimum number of steps')
parser.add_argument('-s2',  '--n_max_steps',    type=int,   default=25,     help='maximum number of steps')
parser.add_argument('-fr',  '--fire_rate',      type=float, default=1.0,    help='prob of stochastic update')
parser.set_defaults(is_residual=True)
parser.add_argument('-r',   dest='is_residual', action='store_true',        help='use residual connection')
parser.add_argument('-nr',  dest='is_residual', action='store_false',       help='no residual connection')
parser.set_defaults(has_coord_act=True)
parser.add_argument('-ca',  dest='has_coord_act', action='store_true',      help='use tanh act for coord mlp')
parser.add_argument('-nca', dest='has_coord_act', action='store_false',     help='no act for coord mlp')
parser.set_defaults(has_attention=False)
parser.add_argument('-ha',  dest='has_attention', action='store_true',      help='use attention weights')
parser.add_argument('-nha', dest='has_attention', action='store_false',     help='no attention weights')

parser.add_argument('-lr',                      type=float, default=1e-3,   help='adam: learning rate')
parser.add_argument('-b1',                      type=float, default=0.9,    help='adam: beta 1')
parser.add_argument('-b2',                      type=float, default=0.999,  help='adam: beta 2')
parser.add_argument('-wd',                      type=float, default=1e-5,   help='adam: weight decay')
parser.add_argument('-gcv', '--grad_clip_val',  type=float, default=1.0,    help='gradient clipping value')

parser.add_argument('-pats', '--patience_sch',  type=int,   default=500,    help='ReduceOP: scheduler patience')
parser.add_argument('-facs', '--factor_sch',    type=float, default=0.5,    help='ReduceOP: scheduler factor')

args = parser.parse_args()
if args.patience is None: args.patience = args.training_steps
if args.patience_sch is None: args.patience_sch = args.training_steps
print(args)

target_coord, edge_index = get_geometric_graph(args.dataset)
dataset = GeometricGraphDataset(
    coord=target_coord,
    edge_index=edge_index,
    scale=args.scale,
    density_rand_edge=args.dens_rand_edge,
)
loader = DataLoader(dataset, batch_size=args.batch_sch[-1])

cp_best_model_valid = pl.callbacks.ModelCheckpoint(
    save_top_k=1,
    monitor='loss',
    mode='min',
    every_n_epochs=1,
    filename='best_model-{step}'
)
early_stopping = pl.callbacks.early_stopping.EarlyStopping(
    monitor='loss',
    mode='min',
    patience=args.patience,
    verbose=True,
)
trainer = pl.Trainer(
    logger=TensorBoardLogger('./log/', name='geometric_graph'),
    accelerator=args.device,
    max_epochs=args.training_steps,
    gradient_clip_val=args.grad_clip_val,
    log_every_n_steps=1,
    enable_progress_bar=False,
    callbacks=[early_stopping, cp_best_model_valid]
)

os.makedirs(trainer.logger.log_dir, exist_ok=True)
with open(trainer.logger.log_dir + '/commandline.txt', 'w') as f:
    f.write(' '.join(sys.argv))
with open(trainer.logger.log_dir + '/args.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2, default=lambda o: '<not serializable>')

model = FixedTargetGAE(args)
tik = time.time()
trainer.fit(model, loader)
tok = time.time()
print('Training time: %d (s)' % (tok - tik))
trainer.save_checkpoint(trainer.logger.log_dir + '/checkpoints/last_model.ckpt')
