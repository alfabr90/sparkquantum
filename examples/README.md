# Examples

This directory contains some example files to better illustrate how to use the simulator. There are examples with simple quantum walks, quantum walks with mesh percolations (broken links) and examples which shows how to use the simulator's logger and profiler (in case the user wants to dump some information about usage of resources during execution).

All previous examples have variations that simulate one and two-dimensional walks with one and two particles.

Each file name gives a clue about which characteristics are being considered for the simulation. For instance, the Python script `dtqw_1d2p_percolations.py` simulates a Discrete Time Quantum Walk (`dtqw`) with two particles (`2p`) over a one-dimensional grid (`1d`) with mesh percolations (`percolations`).

Below, there is a list with a summary of each example file:

- **[dtqw_1d1p.py](./dtqw_1d1p.py):** a simple quantum walk with one particle over a one-dimensional mesh;
- **[dtqw_1d1p_percolations.py](./dtqw_1d1p_percolations.py):** same as above, but a random mesh percolations generator has been applied to the mesh;
- **[dtqw_1d1p_profiling.py](./dtqw_1d1p_profiling.py):** same as the corresponding simpler case, but the logger has been activated and the profiling data is exported;
- **[dtqw_1d2p.py](./dtqw_1d2p.py):** a simple quantum walk with two particles over a one-dimensional mesh. A variable inside the script may be set to enable the entanglement between them;
- **[dtqw_1d2p_percolations.py](./dtqw_1d2p_percolations.py):** same as above, but a random mesh percolations generator has been applied to the mesh;
- **[dtqw_1d2p_profiling.py](./dtqw_1d2p_profiling.py):** same as the corresponding simpler case, but the logger has been activated and the profiling data is exported;
- **[dtqw_2d1p.py](./dtqw_2d1p.py):** a simple quantum walk with one particle over a two-dimensional mesh;
- **[dtqw_2d1p_percolations.py](./dtqw_2d1p_percolations.py):** same as above, but a random mesh percolations generator has been applied to the mesh;
- **[dtqw_2d1p_profiling.py](./dtqw_2d1p_profiling.py):** same as the corresponding simpler case, but the logger has been activated and the profiling data is exported;
- **[dtqw_2d2p.py](./dtqw_2d2p.py):** a simple quantum walk with two particles over a two-dimensional mesh. A variable inside the script may be set to enable the entanglement between them;
- **[dtqw_2d2p_percolations.py](./dtqw_2d2p_percolations.py):** same as above, but a random mesh percolations generator has been applied to the mesh;
- **[dtqw_2d2p_profiling.py](./dtqw_2d2p_profiling.py):** same as the corresponding simpler case, but the logger has been activated and the profiling data is exported;

All the above scripts can be adapted by the user in order to change the coins, meshes and mesh percolations generators, if needed.
