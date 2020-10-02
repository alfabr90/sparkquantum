import math

import numpy as np
from pyspark import SparkContext, SparkConf

from sparkquantum import constants, plot, util
from sparkquantum.dtqw.coin.hadamard import Hadamard
from sparkquantum.dtqw.dtqw import DiscreteTimeQuantumWalk
from sparkquantum.dtqw.mesh.grid.onedim.line import Line
from sparkquantum.dtqw.mesh.percolation.random import Random
from sparkquantum.dtqw.observable.position import Position
from sparkquantum.dtqw.particle import Particle

# Choosing a directory to store plots and logs, if enabled
path = './output/dtqw_perc_random/'
util.create_dir(path)

# Supposing the machine/cluster has 4 cores
cores = 4

# Initiallizing the SparkContext with some options
conf = SparkConf().set('sparkquantum.cluster.totalCores', cores)
sc = SparkContext(conf=conf)
sc.setLogLevel('ERROR')

# In this example, the walk will last 100 steps.
steps = 100

# As we chose a `Line` mesh, its size must be
# 2 * steps + 1 sites
size = 2 * steps + 1

# Choosing a mesh with random percolations and instantiating the walk with it
mesh = Line((size, ), percolation=Random(0.1))
dtqw = DiscreteTimeQuantumWalk(mesh)

# To add particles to the walk, a coin must be instantiated with
# the correspondent dimension of the chosen mesh
coin = Hadamard(mesh.ndim)

# Instantiating a particle and giving it an identifier/name
particle = Particle(coin, identifier='Electron')

# Options of initial coin states for the particle
# |i> --> (|0> - i|1>) / sqrt(2)
cstate = (1 / math.sqrt(2), 1j / math.sqrt(2))

# |i> --> |0>
# cstate = (1, 0)

# |i> --> |1>
# cstate = (0, 1)

# Adding the particle to the walk with its coin state and
# position corresponding to the center site of the mesh
dtqw.add_particle(particle, cstate, mesh.center())

# Performing the walk 10 times to get the average distribution
nsims = 10

# As the mesh is small enough to fit into memory, use a numpy array
# to store the average distribution of the simulations
pdf = np.zeros((mesh.shape[0], len(dtqw.particles) + 1), dtype=float)

observable = Position()

for i in range(nsims):
    state = dtqw.walk(steps)

    # Measuring the state of the system and accumulating it to the numpy array
    joint = observable.measure(state)
    pdf += joint.ndarray()

    # Destroying the RDD to remove them from memory and disk
    state.destroy()
    joint.destroy()

    # Resetting the walk
    dtqw.reset()

pdf /= nsims

labels = ["{}'s position x".format(particle.identifier), 'Probability']
plot.line(mesh.axis()[0], pdf, path + 'joint', labels=labels, dpi=300)

# Destroying the RDD to remove them from memory and disk
dtqw.destroy()

# Stopping the SparkContext
sc.stop()
