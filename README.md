# E(*n*)-equivariant Graph Cellular Automata

This repository is the official implementation of [E(*n*)-equivariant Graph Cellular Automata](https://arxiv.org/abs/2301.10497).

## Pattern Formation - Convergence to Geometric Graphs

E(*n*) Convergence to 3D-Torus           |  E(*n*) Regeneration of 3D-Cube
:-------------------------:|:-------------------------:
<img src="https://drive.google.com/uc?export=view&id=1YhDAPwd6oGiovlQXrxsfdLmYPn7c0ohR" width="300" height="300"> | <img src="https://drive.google.com/uc?export=view&id=1qfo-_9CMQ9whKluV_Ir_k4SR0RfpMHge" width="300" height="300">


    python -m trainers.geometric_graph -ds Grid2d -sdg 0.05 -rdg 1.0 -bsc 0 8 2000 16 4000 32 -pats 800
    python -m trainers.geometric_graph -ds Torus  -sdg 0.05 -rdg 1.0 -bsc 0 6 1000 8 2000 16 4000 32 -pats 800
    python -m trainers.geometric_graph -ds Cube   -sdg 0.05 -rdg 1.0 -bsc 0 16
    python -m trainers.geometric_graph -ds Bunny  -sdg 0.05 -rdg 1.0 -bsc 0 4 1000 8 2000 16 4000 32

For testing, play with `notebooks/test_geometric_graph.ipynb`.

## Graph Autoencoding

<img src="https://drive.google.com/uc?export=view&id=13Aoi_FcmbTFhiBT8zeIgKuGxMETZayRg" width="300" height="300">

First, unzip the [datasets](https://drive.google.com/file/d/1574shdPnsNLsx-cKCCEGhbD1MbzfOzTK/view?usp=share_link) in `./data/`.

### E(*n*)-GNCA 3D demo:

    python -m trainers.gae -ds community -ne 3000 -cd 3 -nd 16 -md 32 -s1 15 -s2 25 -ps 5 -rs 0 5 1000 1000
    python -m trainers.gae -ds planar    -ne 3000 -cd 3 -nd 16 -md 32 -s1 15 -s2 25 -ps 5 -rs 0 5 1000 1000 -ng 200 -n1 12 -n2 20

### E(*n*)-GNCA autoencoder: 

    python -m trainers.gae -ds community -ne 3000 -cd 8  -nd 16 -md 32 -s1 25 -s2 35 -ps 5 -rs 0 5 1000 1000
    python -m trainers.gae -ds planar    -ne 3000 -cd 8  -nd 16 -md 32 -s1 25 -s2 35 -ps 5 -rs 0 5 1000 1000 -ng 200 -n1 12 -n2 20
    python -m trainers.gae -ds planar    -ne 3000 -cd 8  -nd 16 -md 32 -s1 25 -s2 35 -ps 5 -rs 0 5 1000 1000 -ng 200 -n1 32 -n2 64 -nc 3
    python -m trainers.gae -ds proteins  -ne 1000 -cd 16 -nd 16 -md 32 -s1 25 -s2 35 -ps 5 -rs 0 5 100  1000 -pats 20
    python -m trainers.gae -ds sbm       -ne 3000 -cd 24 -nd 16 -md 32 -s1 25 -s2 35 -ps 5 -rs 0 5 1000 1000

For testing, play with `notebooks/test_gae.ipynb`.

## Simulation of E(*n*)-equivariant dynamical systems

### Boids

<img src="https://drive.google.com/uc?export=view&id=1u2H8ncBXE8ug2fNrFQ0988vzWaPtCAQC" width="300" height="300">

First create the dataset:

    python data/boids.py --n_simulations 500 --n_steps 500 --n_boids 100

then run:

    python -m trainers.dsystem -ds boids -ne 500 -bs 16 -nd 16 -md 32 -sls 0 20 -lr 1e-3

For testing, play with `notebooks/test_dsystems.ipynb`.

### NBody

<img src="https://drive.google.com/uc?export=view&id=1CYYGYk3RE3mrb5N1nieKp375zwApC0Ml" width="300" height="300">

First create the dataset:

    python data/nbody.py --n_simulations 5000 --n_steps 1000 --n_bodies 5

then run:

    python -m trainers.dsystem -ds nbody -ne 500 -nd 16 -md 32 -sls 0 25

For testing, play with `notebooks/test_dsystems.ipynb`.

