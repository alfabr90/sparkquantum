# Examples

This directory contains some example files to better illustrate how to use the simulator. There are examples with simple quantum walks, quantum walks with mesh percolations (broken links) and examples which shows how to use the simulator's logger and profiler (in case the user wants to dump some information about usage of resources during execution).

All previous examples have variations that simulate one and two-dimensional walks with one and two particles.

Each file name gives a clue about which characteristics are being considered for the simulation. For instance, the python script `dtqw_1d2p_broken_links.py` simulates a Discrete Time Quantum Walk (`dtqw`) with two particles (`2p`) over a one-dimensional mesh (`1d`) with broken links (`broken_links`).

Below, there is a list with a summary of each example file:

- **[dtqw_1d1p.py](./dtqw_1d1p.py):** a simple quantum walk with one particle over a one-dimensional mesh;
- **[dtqw_1d1p_broken_links.py](./dtqw_1d1p_broken_links.py):** same as above, but a random broken links generator has been applied to the mesh;
- **[dtqw_1d1p_profiling.py](./dtqw_1d1p_profiling.py):** same as the corresponding simpler case, but the logger has been activated and a profiler have been set for all objects;
- **[dtqw_1d2p.py](./dtqw_1d2p.py):** a simple quantum walk with two particles over a one-dimensional mesh. A variable inside the script may be set to enable the entanglement between them;
- **[dtqw_1d2p_broken_links.py](./dtqw_1d2p_broken_links.py):** same as above, but a random broken links generator has been applied to the mesh;
- **[dtqw_1d2p_profiling.py](./dtqw_1d2p_profiling.py):** same as the corresponding simpler case, but the logger has been activated and a profiler have been set for all objects;
- **[dtqw_2d1p.py](./dtqw_2d1p.py):** a simple quantum walk with one particle over a two-dimensional mesh;
- **[dtqw_2d1p_broken_links.py](./dtqw_2d1p_broken_links.py):** same as above, but a random broken links generator has been applied to the mesh;
- **[dtqw_2d1p_profiling.py](./dtqw_2d1p_profiling.py):** same as the corresponding simpler case, but the logger has been activated and a profiler have been set for all objects;
- **[dtqw_2d2p.py](./dtqw_2d2p.py):** a simple quantum walk with two particles over a two-dimensional mesh. A variable inside the script may be set to enable the entanglement between them;
- **[dtqw_2d2p_broken_links.py](./dtqw_2d2p_broken_links.py):** same as above, but a random broken links generator has been applied to the mesh;
- **[dtqw_2d2p_profiling.py](./dtqw_2d2p_profiling.py):** same as the corresponding simpler case, but the logger has been activated and a profiler have been set for all objects;

All the above scripts can be adapted by the user in order to change the coins, meshes and broken links generators, if needed.
