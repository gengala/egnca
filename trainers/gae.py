from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
import argparse
import json
import time
import sys
import os

from data.datasets import load_dataset
from models import GAE


parser = argparse.ArgumentParser()

parser.add_argument('-device',                  type=str,   default='cuda', help='cpu | cuda')
parser.add_argument('-ds',  '--dataset',        type=str,   default=None,   help='dataset name')
# following 3 args are required for datasets created on the fly, like planar graphs
parser.add_argument('-ng', '--num_graphs',      type=int,   default=None,   help='number of graphs')
parser.add_argument('-n1', '--min_n_nodes',     type=int,   default=None,   help='minimum number of nodes')
parser.add_argument('-n2', '--max_n_nodes',     type=int,   default=None,   help='maximum number of nodes')

parser.add_argument('-ne',  '--n_epochs',       type=int,   default=5000,   help='number of epochs')
parser.add_argument('-bs',  '--batch_size',     type=int,   default=32,     help='batch size')
parser.add_argument('-tsp',                     type=float, default=0.8,    help='training split percentage')
parser.add_argument('-pat', '--patience',       type=int,   default=None,   help='early stopping patience')

parser.add_argument('-cd',  '--coord_dim',      type=int,   default=None,   help='coord dimension')
parser.add_argument('-nd',  '--node_dim',       type=int,   default=None,   help='node feature dimension')
parser.add_argument('-md',  '--message_dim',    type=int,   default=None,   help='message feature dimension')
parser.add_argument('-nl',  '--n_layers',       type=int,   default=1,      help='number of EGNN layers')
parser.add_argument('-nt',  '--norm_type',      type=str,   default='nn',   help='norm type: nn, pn or None')
parser.add_argument('-nc',  '--norm_cap',       type=float, default=None,   help='norm capacity')
parser.add_argument('-act',                     type=str,   default='tanh', help='tanh | silu | lrelu')
parser.add_argument('-std',                     type=float, default=0.5,    help='standard deviation of gaussian prior')
parser.add_argument('-s1',  '--n_min_steps',    type=int,   default=None,   help='minimum number of steps')
parser.add_argument('-s2',  '--n_max_steps',    type=int,   default=None,   help='maximum number of steps')
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
parser.set_defaults(init_rand_node_feat=False)
parser.add_argument('-ir',   dest='init_rand_node_feat', action='store_true',   help='init node feat set to one')
parser.add_argument('-nir',  dest='init_rand_node_feat', action='store_false',  help='init node feat set with noise')

# if following 2 args are given, pooling system is activated
parser.add_argument('-ps',  '--pool_size',      type=int,   default=None,   help='pool size')
parser.add_argument('-rs',  '--rep_sch',        type=int,   default=None,   help='repetition schedule', nargs='+')

parser.set_defaults(learn_dec=True)
parser.add_argument('-ld',  dest='learn_dec',   action='store_true',        help='learnable decoder')
parser.add_argument('-nld', dest='learn_dec',   action='store_false',       help='non learnable decoder')
parser.add_argument('-d1',                      type=float, default=1.0,    help='decoder delta 1 param')
parser.add_argument('-d2',                      type=float, default=1.0,    help='decoder delta 2 param')

parser.add_argument('-lr',                      type=float, default=5e-4,   help='adam: (encoder) learning rate')
parser.add_argument('-dlr',                     type=float, default=None,   help='adam: decoder learning rate')
parser.add_argument('-b1',                      type=float, default=0.9,    help='adam: beta 1')
parser.add_argument('-b2',                      type=float, default=0.999,  help='adam: beta 2')
parser.add_argument('-wd',                      type=float, default=1e-5,   help='adam: weight decay')
parser.add_argument('-gcv', '--grad_clip_val',  type=float, default=0.1,    help='gradient clipping value')

parser.add_argument('-pats', '--patience_sch',  type=int,   default=100,    help='ReduceOP: scheduler patience')
parser.add_argument('-facs', '--factor_sch',    type=float, default=0.5,    help='ReduceOP: scheduler factor')

args = parser.parse_args()
if args.patience is None: args.patience = args.n_epochs
if args.patience_sch is None: args.patience_sch = args.n_epochs
if args.dlr is None: args.dlr = args.lr
print(args)

dataset = load_dataset(
    dataset_name=args.dataset,
    num_graphs=args.num_graphs,
    min_n_nodes=args.min_n_nodes,
    max_n_nodes=args.max_n_nodes,
    device=args.device
)
train_idx, valid_test_idx = train_test_split(range(len(dataset)), train_size=args.tsp)
valid_idx, test_idx = train_test_split(range(len(valid_test_idx)), train_size=0.5)
train_set, valid_set, test_set = dataset[train_idx], dataset[valid_idx], dataset[test_idx]

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_set, batch_size=min(len(valid_set), args.batch_size), shuffle=False, drop_last=False)

cp_best_model_valid = pl.callbacks.ModelCheckpoint(
    save_top_k=1,
    monitor='val_loss_epoch',
    mode='min',
    every_n_epochs=1,
    filename='best_model_valid-{epoch}'
)
early_stopping = pl.callbacks.early_stopping.EarlyStopping(
    monitor='val_loss_epoch',
    mode='min',
    patience=args.patience,
    verbose=True,
)
trainer = pl.Trainer(
    logger=TensorBoardLogger('./log/', name='gae'),
    accelerator=args.device,
    max_epochs=args.n_epochs,
    gradient_clip_val=args.grad_clip_val,
    log_every_n_steps=1,
    enable_progress_bar=False,
    num_sanity_val_steps=0,
    callbacks=[cp_best_model_valid, early_stopping]
)

os.makedirs(trainer.logger.log_dir, exist_ok=True)
with open(trainer.logger.log_dir + '/commandline.txt', 'w') as f:
    f.write(' '.join(sys.argv))
with open(trainer.logger.log_dir + '/args.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2, default=lambda o: '<not serializable>')
with open(trainer.logger.log_dir + '/valid_idx.txt', 'w') as output:
    output.write(str(valid_idx))
with open(trainer.logger.log_dir + '/test_idx.txt', 'w') as output:
    output.write(str(test_idx))

model = GAE(args)
print(model)
tik = time.time()
trainer.fit(model, train_loader, valid_loader)
tok = time.time()
print('Training time: %d (s)' % (tok - tik))
trainer.save_checkpoint(trainer.logger.log_dir + '/checkpoints/last_model.ckpt')
