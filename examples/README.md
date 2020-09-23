# Examples

This directory contains some example files to better illustrate how to use the simulator. The examples comprehend discrete time quantum walks (DTQW) consisting of one or multiple particles and considering meshes with or without percolations (broken links):

- **[dtqw_1d1p.py](./dtqw_1d1p.py):** a DTQW with one particle over a one-dimensional mesh (e.g., line);
- **[dtqw_1d2p.py](./dtqw_1d2p.py):** a DTQW with two particles over a one-dimensional mesh (e.g., line);
- **[dtqw_2d1p.py](./dtqw_2d1p.py):** a DTQW with one particle over a two-dimensional mesh (e.g., diagonal lattice);
- **[dtqw_2d2p.py](./dtqw_2d2p.py):** a DTQW with two particles over a two-dimensional mesh (e.g., diagonal lattice).

Each script above comes with percolations generator usage examples. Also, for the multiparticle cases, the user can enable the entanglement between particles by removing the correspondent commented line.

Anyway, all the above scripts can be adapted by the user in order to change the number of particles, the coins and the mesh, if desired.
