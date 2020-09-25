# Examples

This directory contains some example files to better illustrate how to use the simulator. The examples comprehend discrete time quantum walks (DTQW) consisting of one or multiple particles and considering meshes with or without percolations (broken links):

- **[dtqw_1d1p.py](./dtqw_1d1p.py):** a DTQW with one particle over a one-dimensional mesh (e.g., line);
- **[dtqw_1d2p.py](./dtqw_1d2p.py):** a DTQW with two particles over a one-dimensional mesh (e.g., line);
- **[dtqw_2d1p.py](./dtqw_2d1p.py):** a DTQW with one particle over a two-dimensional mesh (e.g., diagonal lattice);
- **[dtqw_2d2p.py](./dtqw_2d2p.py):** a DTQW with two particles over a two-dimensional mesh (e.g., diagonal lattice);
- **[dtqw_logging_profiling.py](./dtqw_logging_profiling.py):** a DTQW with logging and profiling enabled;
- **[dtqw_perc_random.py](./dtqw_perc_random.py):** a DTQW with random mesh percolations. The walk is performed _n_ times to obtain the average distribution;
- **[dtqw_perc_permanent.py](./dtqw_perc_permanent.py):** a DTQW with permanent mesh percolations.

For the multiparticle examples, the user can enable the entanglement between particles by changing the correspondent commented lines. Anyway, all the above scripts can be adapted by the user in order to change the number of particles, the coins and the mesh, if desired.
