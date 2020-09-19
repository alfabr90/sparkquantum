import math

from pyspark import SparkContext, SparkConf

from sparkquantum import constants, plot, util
from sparkquantum.dtqw.coin.hadamard import Hadamard
from sparkquantum.dtqw.dtqw import DiscreteTimeQuantumWalk
from sparkquantum.dtqw.mesh.grid.onedim.line import Line
from sparkquantum.dtqw.mesh.percolation.random import Random
from sparkquantum.dtqw.observable.position import Position
from sparkquantum.dtqw.particle import Particle

base_path = './output'
cores = 4

particles = 1

# In this example, the walk will last 30 steps.
# As we chose a `Line` mesh, its size must be
# 2 * steps + 1 sites
steps = 30
size = 2 * steps + 1

# The mesh will have percolations with the following likelihood
probability = 0.3

# Choosing a directory to store plots and logs
path = "{}/{}_{}_{}_{}_{}_{}/".format(
    base_path, 'hadamard', 'line', size, probability, steps, particles
)
util.create_dir(path)

# Initiallizing the SparkContext with some options
conf = SparkConf().set('sparkquantum.cluster.totalCores', cores)
sc = SparkContext(conf=conf)
sc.setLogLevel('ERROR')

# Choosing a mesh with a percolations generator and
# instantiating the walk with it
mesh = Line((size, ), percolation=Random(probability))
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

# Performing the walk
state = dtqw.walk(steps)

# Measuring the state of the system and plotting its distribution
position = Position()

joint = position.measure(state)

labels = ["{}'s position x".format(particle.identifier), 'Probability']
joint.plot(path + 'joint_1d1p', labels=labels, dpi=300)

# Destroying the RDD and stopping the SparkContext
state.destroy()
dtqw.destroy()
joint.destroy()
sc.stop()