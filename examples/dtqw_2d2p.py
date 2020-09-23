import math
import cmath

from pyspark import SparkContext, SparkConf

from sparkquantum import constants, plot, util
from sparkquantum.dtqw.coin.hadamard import Hadamard
from sparkquantum.dtqw.dtqw import DiscreteTimeQuantumWalk
from sparkquantum.dtqw.interaction.collision.phase import PhaseChange
from sparkquantum.dtqw.mesh.grid.twodim.diagonal.lattice import Lattice
from sparkquantum.dtqw.mesh.percolation.random import Random
from sparkquantum.dtqw.observable.position import Position
from sparkquantum.dtqw.particle import Particle

# Choosing a directory to store plots and logs, if enabled
path = './output/dtqw_2d2p/'
util.create_dir(path)

# Initiallizing the SparkContext with some options
conf = SparkConf().set('sparkquantum.cluster.totalCores', 4)
sc = SparkContext(conf=conf)
sc.setLogLevel('ERROR')

# In this example, the walk will last 7 steps.
steps = 7

# As we chose a `Line` mesh, its size must be
# 2 * steps + 1 sites
size = 2 * steps + 1

# The particles will change their phase when colliding
phase = 1.0 * cmath.pi

percolation = None

# The mesh will have percolations with the following likelihood
#percolation = Random(0.3)

# or even have permanent percolations
# percolation = Permanent([math.floor((size - 1) / 4),
#                         math.ceil(3 * (size - 1) / 4 + 1)])

# Choosing a mesh and instantiating the walk with it
mesh = Lattice((size, size), percolation=percolation)
dtqw = DiscreteTimeQuantumWalk(
    mesh,
    interaction=PhaseChange(phase),
    repr_format=constants.StateRepresentationFormatCoinPosition)

# To add particles to the walk, a coin must be instantiated with
# the correspondent dimension of the chosen mesh
coin = Hadamard(mesh.ndim)

# Instantiating the particle and giving them an identifier/name
particle1 = Particle(coin, identifier='Fermion')
particle2 = Particle(coin, identifier='Boson')

# Options of initial coin states for the particle
# |i,j> --> (|0,0> + i|0,1> - i|1,0> + |1,1>) / 2
cstate = (1 / 2, 1j / 2, -1j / 2, 1 / 2)

# |i,j> --> (|0,0> + i|0,1> + i|1,0> - |1,1>) / 2
# cstate = (1 / 2, 1j / 2, 1j / 2, -1 / 2)

# |i,j> --> (|0,0> - |0,1> - |1,0> + |1,1>) / 2
# cstate = (1 / 2, -1 / 2, -1 / 2, 1 / 2)

# Each particle's position will correspond to the center site of the mesh
position = mesh.center()

# The coin space for two-dimensional grids has size of 4
cspace = 4

# The position space has size corresponding to the number of sites
pspace = mesh.sites

if dtqw.repr_format == constants.StateRepresentationFormatCoinPosition:
    # |i1,j1>|x1,y1>|i2,j2>|x2,y2> --> (|1,1>|x1,y1>|0,0>|x2,y2> - |0,0>|x1,y1>|1,1>|x2,y2>) / sqrt(2)
    state = [[(3 * pspace + position) * cspace * pspace + (0 * pspace + position), 0, 1.0 / math.sqrt(2)],
             [(0 * pspace + position) * cspace * pspace + (3 * pspace + position), 0, -1.0 / math.sqrt(2)]]
elif dtqw.repr_format == constants.StateRepresentationFormatPositionCoin:
    # |x1,y1>|i1,j1>|x2,y2>|i2,j2> --> (|x1,y1>|1,1>|x2,y2>|0,0> - |x1,y1>|0,0>|x2,y2>|1,1>) / sqrt(2)
    state = [[(position * cspace + 3) * pspace * cspace + (position * cspace + 0), 0, 1.0 / math.sqrt(2)],
             [(position * cspace + 0) * pspace * cspace + (position * cspace + 3), 0, -1.0 / math.sqrt(2)]]

# Adding the particles to the walk, with their coin state and position
dtqw.add_particle(particle1, cstate, position)
dtqw.add_particle(particle2, cstate, position)

# Adding the entangled particles to the walk, informing the system state
#dtqw.add_entanglement([particle1, particle2], state)

# Performing the walk
state = dtqw.walk(steps)

# Measuring the state of the system and plotting its distributions
joint, collision, marginal = Position().measure(state)

labels = ["Particles' position x", "Particles' position y", 'Probability']
collision.plot(path + 'collision_2d2p', labels=labels, dpi=300)

for p in range(len(dtqw.particles)):
    labels = ["{}'s position x".format(dtqw.particles[p].identifier),
              "{}'s position y".format(dtqw.particles[p].identifier),
              'Probability']
    marginal[p].plot(path + 'marginal_2d2p_particle{}'.format(p + 1),
                     labels=labels, dpi=300)

# Destroying the RDD and stopping the SparkContext
state.destroy()
dtqw.destroy()
joint.destroy()
collision.destroy()
for m in marginal:
    m.destroy()

# Stopping the SparkContext
sc.stop()
