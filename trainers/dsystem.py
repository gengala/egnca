from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import argparse
import json
import time
import sys
import os

from data.nbody import NBodyDataset
from data.boids import BoidsDataset
from models import SimulatorEGNCA


parser = argparse.ArgumentParser()

parser.add_argument('-device',                  type=str,   default='cuda', help='cpu | cuda')

parser.add_argument('-ds',  '--dataset',        type=str,   default=None,   help='dataset name')
parser.add_argument('-ne',  '--n_epochs',       type=int,   default=500,   help='number of epochs')
parser.add_argument('-pat', '--patience',       type=int,   default=50,    help='early stopping patience')
parser.add_argument('-bs',  '--batch_size',     type=int,   default=16,     help='batch size')
parser.add_argument('-sls', '--seq_len_sch',    type=int,   default=[0, 2], help='seq length schedule', nargs='+')
parser.add_argument('-tsp',                     type=float, default=0.9,    help='training split percentage')

parser.add_argument('-lr',                      type=float, default=5e-4,   help='adam: learning rate')
parser.add_argument('-b1',                      type=float, default=0.9,    help='adam: beta 1')
parser.add_argument('-b2',                      type=float, default=0.999,  help='adam: beta 2')
parser.add_argument('-wd',                      type=float, default=1e-5,   help='adam: weight decay')
parser.add_argument('-gcv',  '--grad_clip_val', type=float, default=1.0,    help='gradient clipping value')
parser.add_argument('-pats', '--patience_sch',  type=int,   default=30,     help='ReduceOP: scheduler patience')
parser.add_argument('-facs', '--factor_sch',    type=float, default=0.5,    help='ReduceOP: scheduler factor')

parser.add_argument('-nd',  '--node_dim',       type=int,   default=None,   help='node feature dimension')
parser.add_argument('-md',  '--message_dim',    type=int,   default=None,   help='message feature dimension')
parser.add_argument('-nl',  '--n_layers',       type=int,   default=1,      help='number of EGNN layers')
parser.add_argument('-act',                     type=str,   default='tanh', help='tanh | silu | lrelu')
parser.add_argument('-rad', '--radius',         type=float, default=None,   help='radius of neighbors')

parser.set_defaults(is_residual=False)
parser.add_argument('-r',   dest='is_residual',     action='store_true',    help='use residual connection')
parser.add_argument('-nr',  dest='is_residual',     action='store_false',   help='no residual connection')
parser.set_defaults(has_vel_norm=True)
parser.add_argument('-hvn', dest='has_vel_norm',    action='store_true',    help='include vel norm in vel MLP')
parser.add_argument('-nvn', dest='has_vel_norm',    action='store_false',   help='no vel norm in vel MLP')
parser.set_defaults(has_coord_act=False)
parser.add_argument('-ca',  dest='has_coord_act',   action='store_true',    help='use tanh act for coord mlp')
parser.add_argument('-nca', dest='has_coord_act',   action='store_false',   help='no act for coord mlp')
parser.set_defaults(has_attention=True)
parser.add_argument('-ha',  dest='has_attention',   action='store_true',    help='use attention weights')
parser.add_argument('-nha', dest='has_attention',   action='store_false',   help='no attention weights')
parser.set_defaults(sparse_training=True)
parser.add_argument('-st',  dest='sparse_training', action='store_true',    help='sparse training')
parser.add_argument('-dt',  dest='sparse_training', action='store_false',   help='dense training')

args = parser.parse_args()
if args.patience_sch is None:
    args.patience_sch = args.n_epochs
if args.dataset == 'boids':
    dataset = BoidsDataset()
    args.box_dim = dataset.simulator.box_dim
    if args.radius is None:
        args.radius = dataset.simulator.outer_radius
elif args.dataset == 'nbody':
    dataset = NBodyDataset()
    args.box_dim = None
else:
    raise ValueError('Invalid dataset name.')
print(args)

train_set_len = int(len(dataset) * args.tsp)
train_set, valid_set = random_split(dataset, [train_set_len, len(dataset) - train_set_len])

train_loader = DataLoader(train_set, batch_size=args.batch_size, drop_last=True, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=args.batch_size, drop_last=True, shuffle=False)

early_stopping = pl.callbacks.early_stopping.EarlyStopping(
    monitor='val_loss_epoch',
    check_finite=True,
    patience=args.patience,
    verbose=True)
cp_best_model_valid = pl.callbacks.ModelCheckpoint(
    save_top_k=1,
    monitor='val_loss_epoch',
    every_n_epochs=1,
    filename='best_model_valid-{epoch}',
    verbose=True)
trainer = pl.Trainer(
    logger=TensorBoardLogger('./log/', name=args.dataset),
    accelerator=args.device,
    max_epochs=args.n_epochs,
    gradient_clip_val=args.grad_clip_val,
    log_every_n_steps=2,
    enable_progress_bar=False,
    num_sanity_val_steps=0,
    callbacks=[early_stopping, cp_best_model_valid])

os.makedirs(trainer.logger.log_dir, exist_ok=True)
with open(trainer.logger.log_dir + '/commandline.txt', 'w') as f:
    f.write(' '.join(sys.argv))
with open(trainer.logger.log_dir + '/args.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2, default=lambda o: '<not serializable>')

model = SimulatorEGNCA(args)
tik = time.time()
trainer.fit(model, train_loader, valid_loader)
tok = time.time()
print('Training time: %d (s)' % (tok - tik))
trainer.save_checkpoint(trainer.logger.log_dir + '/checkpoints/last_model.ckpt')
