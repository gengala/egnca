from torch.utils.data import Dataset
from typing import Optional
import numpy as np
import torch


class NBody:
    """
    Adapted from https://github.com/pmocz/nbody-python.
    """
    def __init__(
        self,
        dt: Optional[float] = 0.1,
        softening: Optional[float] = 1.0,
        tot_mass: Optional[float] = 20.0,
        grav_const: Optional[float] = 1.0
    ):
        self.dt = dt
        self.softening = softening
        self.tot_mass = tot_mass
        self.grav_const = grav_const

    def get_mass(
        self,
        n_bodies: int
    ):
        return self.tot_mass * np.ones((n_bodies, 1)) / n_bodies

    def sample_trajectory(
        self,
        n_steps: int,
        n_bodies: Optional[int] = None,
        coord: Optional[np.ndarray] = None,
        vel: Optional[np.ndarray] = None,
        to_torch: Optional[bool] = False
    ):
        if coord is not None and vel is not None:
            coord_traj = np.empty((n_steps, coord.shape[0], 3))
            vel_traj = np.empty((n_steps, coord.shape[0], 3))
        else:
            coord_traj = np.empty((n_steps, n_bodies, 3))
            vel_traj = np.empty((n_steps, n_bodies, 3))
            coord, vel = self.init_coord_vel(n_bodies)

        coord_traj[0], vel_traj[0] = coord, vel
        for i in range(1, n_steps):
            coord, vel = self.update_coord_vel(coord_traj[i - 1], vel_traj[i - 1])
            coord_traj[i] = coord
            vel_traj[i] = vel
        vel_traj = vel_traj * self.dt

        if to_torch:
            coord_traj = torch.Tensor(coord_traj)
            vel_traj = torch.Tensor(vel_traj)
        return coord_traj, vel_traj

    def update_coord_vel(
        self,
        in_coord: np.ndarray,
        in_vel: np.ndarray,
    ):
        assert in_coord.shape == in_vel.shape
        in_acc = self.get_acc(in_coord)
        out_vel = in_vel + in_acc * self.dt  # / 2.0
        out_coord = in_coord + out_vel * self.dt
        # out_acc = self.get_acc(out_coord)
        # out_vel = out_vel + out_acc * self.dt / 2.0
        return out_coord, out_vel

    def get_acc(
        self,
        in_coord: np.ndarray
    ):
        coord_diff = in_coord - np.expand_dims(in_coord, 1)
        inv_r3 = (coord_diff ** 2).sum(2) + self.softening ** 2
        mask = inv_r3 > 0
        inv_r3[mask] = inv_r3[mask] ** (-1.5)
        mass = self.get_mass(in_coord.shape[0])
        acc = (self.grav_const * (coord_diff * np.expand_dims(inv_r3, -1)).transpose(2, 0, 1) @ mass).squeeze().T
        return acc

    def init_coord_vel(
        self,
        n_bodies: int
    ):
        out_coord = np.random.randn(n_bodies, 3)
        out_vel = np.random.randn(n_bodies, 3)
        mass = self.get_mass(n_bodies)
        out_vel -= np.mean(mass * out_vel, 0) / np.mean(mass)
        return out_coord, out_vel


def test_nbody_equivariance():
    import torch
    sim = NBody()
    coord1, vel1 = sim.init_coord_vel(n_bodies=10)
    rotation = torch.nn.init.orthogonal_(torch.empty(3, 3, dtype=torch.float64)).numpy()
    translation = np.random.randn(1, 3)
    coord2 = (rotation @ coord1.T).T + translation
    vel2 = (rotation @ vel1.T).T
    res1, vel11 = sim.sample_trajectory(coord=coord1, vel=vel1, n_steps=500)
    res2, vel22 = sim.sample_trajectory(coord=coord2, vel=vel2, n_steps=500)
    assert np.isclose((rotation @ res1[-1].T).T + translation, res2[-1], atol=1e-4).all(), 'Test failed.'
    print('Test succeeded.')


def example_simulation():
    from utils.visualize import plot_trajectory
    sim = NBody()
    coord_traj, vel_traj = sim.sample_trajectory(n_bodies=5, n_steps=500)
    plot_trajectory(coord_traj=coord_traj, box_dim=5, tail_len=10, print_step=True)


class NBodyDataset(Dataset):

    def __init__(
        self,
        root: Optional[str] = './data/nbody/',
        seq_len: Optional[int] = 2
    ):
        import dill
        self.coord_dataset, self.vel_dataset, self.simulator = torch.load(root + 'nbody.pt', pickle_module=dill)
        assert self.coord_dataset.size() == self.vel_dataset.size()
        assert seq_len > 1
        self.n_steps = self.coord_dataset.size(1)
        self.seq_len = seq_len

    def __getitem__(self, index: int):
        seq_index = np.random.randint(self.n_steps - self.seq_len - 1)
        coord_traj = self.coord_dataset[index, seq_index: seq_index + self.seq_len]
        vel_traj = self.vel_dataset[index, seq_index: seq_index + self.seq_len]
        return coord_traj, vel_traj

    def __len__(self):
        return self.coord_dataset.size(0)


if __name__ == '__main__':

    from tqdm import tqdm
    import argparse
    import dill
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_simulations',  type=int,   default=500,    help='number of simulations')
    parser.add_argument('--n_steps',        type=int,   default=500,    help='number of steps of each simulation')
    parser.add_argument('--n_bodies',       type=int,   default=100,    help='number of bodies')
    parser.add_argument('--root',   type=str,   default='./data/nbody/',    help='root folder')
    args = parser.parse_args()
    print(args)

    print('Creating the NBody dataset..')
    simulator = NBody()
    if not os.path.exists(args.root):
        os.makedirs(args.root)
    coord_dataset = np.empty((args.n_simulations, args.n_steps, args.n_bodies, 3))
    vel_dataset = np.empty((args.n_simulations, args.n_steps, args.n_bodies, 3))
    for idx_sim in tqdm(range(args.n_simulations)):
        coord_dataset[idx_sim], vel_dataset[idx_sim] = simulator.sample_trajectory(args.n_steps, args.n_bodies)
    coord_dataset = torch.Tensor(coord_dataset)
    vel_dataset = torch.Tensor(vel_dataset)
    torch.save([coord_dataset, vel_dataset, simulator], args.root + '/nbody.pt', pickle_module=dill)
    print('Dataset created.')
