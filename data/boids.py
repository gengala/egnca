from torch.utils.data import Dataset
from typing import Optional
import numpy as np
import torch


class Boids:
    """
    Adapted from https://agentpy.readthedocs.io/en/latest/agentpy_flocking.html.
    """
    def __init__(
        self,
        coord_dim: int,
        outer_radius: Optional[float] = 5,
        inner_radius: Optional[float] = 2.5,
        cohesion_strength: Optional[float] = 0.005,
        alignment_strength: Optional[float] = 0.3,
        separation_strength: Optional[float] = 0.1,
        box_dim: Optional[float] = 10,
        box_strength: Optional[float] = 0.1
    ):
        assert coord_dim == 2 or coord_dim == 3
        self.coord_dim = coord_dim
        self.outer_radius = outer_radius
        self.inner_radius = inner_radius
        self.cohesion_strength = cohesion_strength
        self.alignment_strength = alignment_strength
        self.separation_strength = separation_strength
        self.box_dim = box_dim
        self.box_strength = box_strength

    def init_coord_vel(
        self,
        n_boids: int
    ):
        if self.box_dim is None:
            coord = np.random.uniform(-self.outer_radius * 3, self.outer_radius * 3, (n_boids, self.coord_dim))
        else:
            coord = np.random.uniform(-self.box_dim, self.box_dim, (n_boids, self.coord_dim))
        vel = np.random.uniform(-0.5, 0.5, (n_boids, self.coord_dim,))
        vel /= np.linalg.norm(vel, axis=-1, keepdims=True)
        return coord, vel

    @staticmethod
    def get_neighbors(
        dist: np.ndarray,
        radius: float
    ):
        neighbors = dist < radius
        np.fill_diagonal(neighbors, 0)
        return neighbors

    def update_coord_vel(
        self,
        coord: np.ndarray,
        vel: np.ndarray
    ):
        assert coord.shape == vel.shape
        dist = np.linalg.norm(coord[:, None, :] - coord[None, :, :], axis=-1)

        # Cohesion & Alignment
        neighbors = self.get_neighbors(dist, radius=self.outer_radius)
        vel_cohesion = np.zeros_like(vel)
        vel_alignment = np.zeros_like(vel)
        for i in range(len(coord)):
            if neighbors[i].any():
                vel_cohesion[i] = coord[neighbors[i]].mean(0) - coord[i]
                vel_alignment[i] = vel[neighbors[i]].mean(0) - vel[i]
        vel_cohesion *= self.cohesion_strength
        vel_alignment *= self.alignment_strength

        # Separation
        vel_separation = np.zeros_like(vel)
        neighbors_separation = self.get_neighbors(dist, radius=self.inner_radius)
        for i in range(len(coord)):
            if neighbors_separation[i].any():
                vel_separation[i] -= coord[neighbors_separation[i]].sum(0) - sum(neighbors_separation[i]) * coord[i]
        vel_separation *= self.separation_strength

        # Box
        if self.box_dim is not None:
            vel_box = (coord < -self.box_dim) * self.box_strength - (coord > self.box_dim) * self.box_strength
        else:
            vel_box = np.zeros_like(vel)

        # Update
        old_vel = vel
        vel = vel + (vel_separation + vel_alignment + vel_cohesion + vel_box)
        vel /= np.linalg.norm(old_vel, axis=-1, keepdims=True)
        coord = coord + vel
        return coord, vel

    def sample_trajectory(
        self,
        n_steps: int,
        n_boids: Optional[int] = None,
        coord: Optional[np.ndarray] = None,
        vel: Optional[np.ndarray] = None,
        to_torch: Optional[bool] = False
    ):
        if coord is not None and vel is not None:
            coord_traj = np.empty((n_steps, coord.shape[0], self.coord_dim))
            vel_traj = np.empty((n_steps, coord.shape[0], self.coord_dim))
        else:
            coord_traj = np.empty((n_steps, n_boids, self.coord_dim))
            vel_traj = np.empty((n_steps, n_boids, self.coord_dim))
            coord, vel = self.init_coord_vel(n_boids)

        coord_traj[0], vel_traj[0] = coord, vel
        for i in range(1, n_steps):
            coord_traj[i], vel_traj[i] = self.update_coord_vel(coord_traj[i - 1], vel_traj[i - 1])

        if to_torch:
            coord_traj = torch.Tensor(coord_traj)
            vel_traj = torch.Tensor(vel_traj)
        return coord_traj, vel_traj


def example_simulation():
    from utils.visualize import plot_trajectory
    simulator = Boids(coord_dim=3)
    coord_traj, vel_traj = simulator.sample_trajectory(n_steps=300, n_boids=100)
    plot_trajectory(coord_traj, box_dim=simulator.box_dim + 5, tail_len=0, print_step=True)


def test_equivariance():
    from scipy.stats import ortho_group
    simulator = Boids(coord_dim=3, box_dim=None)  # no box!
    init_coord_1, init_vel_1 = simulator.init_coord_vel(n_boids=100)
    rotation = ortho_group.rvs(3)

    init_coord_2 = np.matmul(rotation, init_coord_1.T).T
    init_vel_2 = np.matmul(rotation, init_vel_1.T).T

    coord_traj_1, vel_traj_1 = simulator.sample_trajectory(coord=init_coord_1, vel=init_vel_1, n_steps=50)
    coord_traj_2, vel_traj_2 = simulator.sample_trajectory(coord=init_coord_2, vel=init_vel_2, n_steps=50)

    assert np.isclose(np.matmul(rotation, coord_traj_1[-1].T).T, coord_traj_2[-1], atol=1e-5).all()
    assert np.isclose(np.matmul(rotation, vel_traj_1[-1].T).T, vel_traj_2[-1], atol=1e-5).all()
    print('Test succeeded.')


class BoidsDataset(Dataset):

    def __init__(
        self,
        root: Optional[str] = './data/boids/',
        seq_len: Optional[int] = 2
    ):
        import dill
        self.coord_dataset, self.vel_dataset, self.simulator = torch.load(root + 'boids.pt', pickle_module=dill)
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
    parser.add_argument('--n_boids',        type=int,   default=100,    help='number of boids')
    parser.add_argument('--root',   type=str,   default='./data/boids/',    help='root folder')
    args = parser.parse_args()
    print(args)

    print('Creating the boids dataset..')
    simulator_dataset = Boids(coord_dim=3)
    if not os.path.exists(args.root):
        os.makedirs(args.root)
    coord_dataset = np.empty((args.n_simulations, args.n_steps, args.n_boids, 3))
    vel_dataset = np.empty((args.n_simulations, args.n_steps, args.n_boids, 3))
    for i_sim in tqdm(range(args.n_simulations)):
        coord_dataset[i_sim], vel_dataset[i_sim] = simulator_dataset.sample_trajectory(args.n_steps, args.n_boids)
    coord_dataset = torch.Tensor(coord_dataset)
    vel_dataset = torch.Tensor(vel_dataset)
    torch.save([coord_dataset, vel_dataset, simulator_dataset], args.root + '/boids.pt', pickle_module=dill)
    print('Dataset created.')
